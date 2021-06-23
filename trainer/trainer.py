import os

import torch
import torch.distributed as dist
from prettytable import PrettyTable
from tqdm import tqdm

from .base import BaseTrainer
from eval.counter import EvalCounter
from eval.coco_eval import COCOMetrics


class Trainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, lr_scheduler, config,
                 train_loader, val_loader, postprocess, device, args):
        super(Trainer, self).__init__(model, loss, optimizer, lr_scheduler, config,
                                      train_loader, val_loader, device, args)
        # used to coco eval for val dataset
        self.postprocess = postprocess
        self.coco_metrics = COCOMetrics(
            gt_file=config['val_gt_file'],
            cat2label=self.val_loader.dataset.CAT2LABEL,
            with_mask=self.val_loader.dataset.with_mask,
            save_dir=self.checkpoint_dir
        )

    def _train_epoch(self, epoch):
        self.logger.info("Train on epoch %d" % epoch)
        self.model.train()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        # Perform training
        counter = EvalCounter()
        n_iter = len(self.train_loader)
        bar = tqdm(enumerate(self.train_loader, 1), total=n_iter,
                   postfix={'lr': '-1.00e0', 'loss': '-1.0000'}) \
            if self.device_rank == 0 else enumerate(self.train_loader, 1)

        # with torch.autograd.detect_anomaly():
        for batch_idx, sample in bar:
            # Load data to device
            image = sample[0].to(self.device)
            label = [anno.to(self.device) for anno in sample[1]]

            # Forward and backward
            predict = self.model(image)
            loss, loss_log, metric_log = self.loss(predict, label, training=True)
            loss.backward()

            if batch_idx % self.accumulate == 0 or batch_idx == n_iter:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if not torch.isfinite(loss).item():
                self.logger.error("Error: nan or inf found. Training stops at "
                                  "epoch {0} batch {1}.".format(epoch, batch_idx))
                if self.device_rank == 0:
                    for k, v in loss_log.items():
                        self.logger.error(k + ': ' + str(v))
                exit()

            # Loss and metric update
            counter.update('loss', loss.item())
            for key, value in loss_log.items():
                counter.update(key, value)

            # Tensorboard scalar writter
            step = (epoch - 1) * n_iter + batch_idx
            if step % self.writer_freq == 0:
                # barrier() blocks until all processes have reached a barrier
                if self.device_rank == 0:
                    if self.world_size > 1:
                        dist.barrier()
                    # Gather counter of all devices
                    for i in range(1, self.world_size):
                        file_name = os.path.join(self.checkpoint_dir, '_temp_counter_%d.pth' % i)
                        counter.merge(torch.load(file_name))
                        os.remove(file_name)
                    # Tensorboard and tqdm update
                    actual_step = step // self.accumulate
                    for i in range(len(self.optimizer.param_groups)):
                        lr_group_i = self.optimizer.param_groups[i]['lr'] * self.accumulate
                        self.tensorboard.add_scalar('lr/group%d' % i, lr_group_i, actual_step)
                    self.tensorboard.add_scalar('train/loss', counter.average('loss'), actual_step)
                    for loss_key in self.loss.loss_id:
                        self.tensorboard.add_scalar('train/%s' % loss_key, counter.average(loss_key), actual_step)
                    bar.set_postfix({'lr': '%.2e' % (self.optimizer.param_groups[0]['lr'] * self.accumulate),
                                     'loss': '%.4f' % counter.average('loss')})
                else:
                    # save counter statistics of all devices except for rank 0
                    base_name = '_temp_counter_%d.pth' % self.device_rank
                    counter.save(os.path.join(self.checkpoint_dir, base_name))
                    dist.barrier()
                counter.reset()

            # reach max_iter limitation
            if hasattr(self.lr_scheduler, 'max_iter') and step == self.lr_scheduler.max_iter:
                if self.device_rank == 0:
                    filename = os.path.join(self.checkpoint_dir, 'batch_{}.pth'.format(step))
                    torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, filename)
                    self.logger.info("Saving checkpoint at {}".format(filename))
                dist.barrier()
                exit()

        train_log = {}
        if self.device_rank == 0:
            if self.world_size > 1:
                dist.barrier()
            # Gather counter of all devices
            for i in range(1, self.world_size):
                file_name = os.path.join(self.checkpoint_dir, '_temp_counter_%d.pth' % i)
                counter.merge(torch.load(file_name))  # note: not merge_epoch
                os.remove(file_name)
            # Record train log
            train_log['train_loss'] = counter.average_epoch('loss')
            for loss_key in self.loss.loss_id:
                train_log['train_%s' % loss_key] = counter.average_epoch(loss_key)
        else:
            # save counter statistics of all devices except for rank 0
            base_name = '_temp_counter_%d.pth' % self.device_rank
            counter.save(os.path.join(self.checkpoint_dir, base_name))  # note: not save_epoch
            dist.barrier()
        counter.reset_epoch()

        # Validate
        if self.val_loader is not None and epoch % self.val_freq == 0:
            val_log = self._val_epoch(epoch)
            train_log.update(val_log)

        return train_log

    def _val_epoch(self, epoch):
        self.logger.info("Validate after epoch %d" % epoch)
        self.model.eval()
        self.coco_metrics.reset()

        # Perform Validation
        counter = EvalCounter()
        n_iter = len(self.val_loader)
        bar = tqdm(enumerate(self.val_loader, 1), total=n_iter) \
            if self.device_rank == 0 else enumerate(self.val_loader, 1)

        with torch.no_grad():
            for batch_idx, sample in bar:
                # Load data to device
                image = sample[0].to(self.device)
                label = [anno.to(self.device) for anno in sample[1]]
                batch_info = sample[2]

                # Forward
                predict = self.model(image)

                # Loss and metric update
                loss, loss_log, metric_log = self.loss(predict, label, training=False)

                # Extra AP and AR metric
                detections = self.postprocess(predict)
                coco_format_dets = self.coco_metrics.to_coco_format(batch_info, detections)
                self.coco_metrics.update_results(coco_format_dets)

                counter.update('loss', loss.item())
                for key, value in loss_log.items():
                    counter.update(key, value)
                for key, value in metric_log.items():
                    counter.update(key, value)

        val_log = {}
        if self.device_rank == 0:
            if self.world_size > 1:
                dist.barrier()
            # Gather counter and coco eval statistics of all devices
            for i in range(1, self.world_size):
                counter_file_name = os.path.join(self.checkpoint_dir, '_temp_counter_%d.pth' % i)
                counter.merge(torch.load(counter_file_name))
                os.remove(counter_file_name)
                coco_eval_file_name = os.path.join(self.checkpoint_dir, '_temp_coco_eval_%d.json' % i)
                self.coco_metrics.update_from_json(coco_eval_file_name)
                os.remove(coco_eval_file_name)
            # accumulate COCO metrics
            coco_eval_log = self.coco_metrics.coco_eval()
            # Tensorboard update
            self.tensorboard.add_scalar('val/loss', counter.average('loss'), epoch)
            for loss_key in self.loss.loss_id:
                self.tensorboard.add_scalar('val/%s' % loss_key, counter.average(loss_key), epoch)
            for metric_key in self.loss.metric_id:
                self.tensorboard.add_scalar('val/%s' % metric_key, counter.average(metric_key), epoch)
            for key, value in coco_eval_log.items():
                self.tensorboard.add_scalar('val/%s' % key, value, epoch)
            # Record val log
            val_log['val_loss'] = counter.average_epoch('loss')
            for loss_key in self.loss.loss_id:
                val_log['val_%s' % loss_key] = counter.average_epoch(loss_key)
            for metric_key in self.loss.metric_id:
                val_log['val_%s' % metric_key] = counter.average_epoch(metric_key)
            for key, value in coco_eval_log.items():
                val_log['val_%s' % key] = value
        else:
            # save counter and coco eval statistics of all devices except for rank 0
            counter_file_name = '_temp_counter_%d.pth' % self.device_rank
            counter.save(os.path.join(self.checkpoint_dir, counter_file_name))
            coco_eval_file_name = '_temp_coco_eval_%d.json' % self.device_rank
            self.coco_metrics.save_as_json(os.path.join(self.checkpoint_dir, coco_eval_file_name))
            dist.barrier()
        counter.reset_epoch()

        return val_log

    def _log_result(self, result):
        train_log_table = PrettyTable()
        train_log_table.field_names = ['TRAIN', *self.loss.scales_prefix, 'ALL']
        for loss_id in self.loss.loss_suffix:
            loss_key = 'train_{}_' + loss_id
            train_log_table.add_row([loss_id, *[result[loss_key.format(scale_id)]
                                                for scale_id in self.loss.scales_prefix],
                                     result[loss_key.format('cross_scale')]])
        train_log_table.align = 'r'
        train_log_table.float_format = '.3'
        self.logger.info('\n' + str(train_log_table))

        if 'val_{0}_{1}'.format(self.loss.scales_prefix[0], self.loss.loss_suffix[0]) in result:
            val_log_table = PrettyTable()
            val_log_table.field_names = ['VAL', *self.loss.scales_prefix, 'ALL']
            for loss_id in self.loss.loss_suffix:
                loss_key = 'val_{}_' + loss_id
                val_log_table.add_row([loss_id, *[result[loss_key.format(scale_id)]
                                                  for scale_id in self.loss.scales_prefix],
                                       result[loss_key.format('cross_scale')]])
            for metric_id in self.loss.metric_suffix:
                metric_key = 'val_{}_' + metric_id
                val_log_table.add_row([metric_id, *[result[metric_key.format(scale_id)]
                                                    for scale_id in self.loss.scales_prefix],
                                       result[metric_key.format('cross_scale')]])

            # union metric and coco metric into a single table
            height = len(self.loss.loss_suffix) + len(self.loss.metric_suffix)
            width = len(self.loss.scales_prefix) + 2
            extra_rows = max(len(self.coco_metrics.metric_keys) - height, 0)
            extra_items = max(height - len(self.coco_metrics.metric_keys), 0)
            for _ in range(extra_rows):
                val_log_table.add_row([''] * width)

            val_log_table.add_column('', self.coco_metrics.metric_keys + [''] * extra_items)
            bbox_stats = [result['val_bbox_' + key] for key in self.coco_metrics.metric_keys]
            val_log_table.add_column('BBOX', bbox_stats + [''] * extra_items)
            if self.coco_metrics.with_mask:
                segm_stats = [result['val_segm_' + key] for key in self.coco_metrics.metric_keys]
                val_log_table.add_column('SEGM', segm_stats + [''] * extra_items)

            val_log_table.align = 'r'
            val_log_table.float_format = '.3'
            self.logger.info('\n' + str(val_log_table))
            self.logger.info('BBOX ' + ' '.join('%.3f' % k for k in self.coco_metrics.bbox_eval_stats))
            if self.coco_metrics.with_mask:
                self.logger.info('SEGM ' + ' '.join('%.3f' % k for k in self.coco_metrics.segm_eval_stats))
