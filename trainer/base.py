import datetime
import json
import logging
import math
import os

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from utils.envs import *


class BaseTrainer:
    """A base trainer for generic supervised training

    `_train_epoch` implements the customized logic of training process
    and returns information as a dict to `_log_result` for logging.
    """

    def __init__(self, model, loss, optimizer, lr_scheduler,
                 config, train_loader, val_loader, device, args):
        self.config = config

        # Setup GPU device and rank
        self.device = device
        self.is_distributed = config['n_gpu'] > 1
        self.device_rank = get_device_rank()
        self.world_size = get_world_size()

        # Backup config and source code
        if args.resume is not None:
            self.checkpoint_dir = os.path.dirname(args.resume)
        else:
            start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
            self.checkpoint_dir = os.path.join(config['log_dir'], config['name'] + '_' + start_time)
            if self.device_rank == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=False)
                print('Set checkpoint directory: %s' % self.checkpoint_dir)
                # Save configuration file
                config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
                with open(config_save_path, 'w') as handle:
                    json.dump(config, handle, indent=4, sort_keys=False)
                # code backup
                # os.popen(r'zip -r {0} . -x "build/*" -x "checkpoints/*" -x "dist/*" -x "coco/*" '
                #          r'-x "*/__pycache__/*" -x ".ipynb_checkpoints/*" -x "*.egg-info/*" '
                #          r'-x "results/*" -x "*.ipynb" -x "*.so" -x "*.pth" -x "*.zip"'
                #          .format(os.path.join(self.checkpoint_dir, 'code.zip'))).read()
            if self.world_size > 1:
                dist.barrier()

        # Logger
        logging.basicConfig(
            level=logging.INFO if self.device_rank == 0 else logging.ERROR,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.checkpoint_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Fundamental model train settings
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Auxiliary train settings
        self.accumulate = config['accumulate']
        self.epochs = config['epochs']
        self.val_freq = config['val_freq']
        self.save_freq = config['save_freq']

        # Configuration to monitor model performance and save best
        self.monitor = 'val_' + config['monitor']
        self.monitor_mode = config['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1

        # Setup tensorboard writer instance
        self.tensorboard = SummaryWriter(self.checkpoint_dir)
        self.writer_freq = config['log_freq'] * self.accumulate

        # Resume from checkpoint
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # Start with weights
        if args.weights is not None:
            self._set_weights(args.weights)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info('\n--------------------------------------------------------------------')
            self.logger.info('[EPOCH %d]' % epoch)
            start_time = datetime.datetime.now()
            result = self._train_epoch(epoch)
            finish_time = datetime.datetime.now()
            self.logger.info('Finish at {}, Runtime: {}'
                             .format(datetime.datetime.now(), str(finish_time - start_time)))

            # Log result information
            if self.device_rank == 0:
                self._log_result(result)

            # Check performance improvement
            if epoch % self.val_freq == 0 and self.device_rank == 0:
                best = False
                if self.monitor_mode != 'off':
                    assert self.monitor in result, "Can\'t recognize monitor item named {}".format(self.monitor)
                    if (self.monitor_mode == 'min' and result[self.monitor] < self.monitor_best) or \
                            (self.monitor_mode == 'max' and result[self.monitor] > self.monitor_best):
                        self.logger.info("Monitor is improved from %f to %f"
                                         % (self.monitor_best, result[self.monitor]))
                        self.monitor_best = result[self.monitor]
                        best = True
                    else:
                        self.logger.info("Monitor is not improved from %f" % self.monitor_best)
                # Save checkpoint
                self._save_checkpoint(epoch, save_best=best)
            elif self.device_rank == 0:
                self._save_checkpoint(epoch, temp=True)

    def _train_epoch(self, epoch):
        """Training logic for one epoch

        Args:
            epoch (int): current epoch number

        Returns:
            train_log (dict): information need be logged
        """
        raise NotImplementedError

    def _log_result(self, result):
        for k, v in result.items():
            self.logger.info('{}: {}'.format(k, v))

    def _save_checkpoint(self, epoch, save_best=False, temp=False):
        if epoch % self.save_freq == 0 or save_best or temp:
            model = self.model.module if self.is_distributed else self.model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'monitor_best': self.monitor_best,
                'config': self.config
            }
            if epoch % self.save_freq == 0:
                filename = os.path.join(self.checkpoint_dir, 'epoch{}.pth'.format(epoch))
                torch.save(state, filename)
                self.logger.info("Saving checkpoint at {}".format(filename))
            if save_best:
                beat_rel_path = 'best_epoch{}.pth'.format(epoch)
                best_path = os.path.join(self.checkpoint_dir, beat_rel_path)
                torch.save(state, best_path)
                best_symlink = os.path.join(self.checkpoint_dir, 'best_model.pth')
                if os.path.islink(best_symlink):
                    os.remove(os.path.join(self.checkpoint_dir, os.readlink(best_symlink)))
                    os.remove(best_symlink)
                os.symlink(beat_rel_path, best_symlink)
                self.logger.info("Saving current best at {}".format(best_path))
            if temp:
                filename = os.path.join(self.checkpoint_dir, 'temp.pth')
                torch.save(state, filename)
                self.logger.info("Saving temp checkpoint at {}".format(filename))

    def _resume_checkpoint(self, resume_file):
        self.logger.info("Loading checkpoint: {}".format(resume_file))
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint
        assert checkpoint['config']['model'] == self.config['model'], \
            "Architecture configuration given in config file is different from that of checkpoint."
        model = self.model.module if self.is_distributed else self.model
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        # load optimizer state from checkpoint
        assert checkpoint['config']['optimizer'] == self.config['optimizer'], \
            "Optimizer configuration given in config file is different from that of checkpoint."
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load lr_scheduler state from checkpoint
        assert checkpoint['config']['lr_scheduler'] == self.config['lr_scheduler'], \
            "LRScheduler configuration given in config file is different from that of checkpoint."
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_file, self.start_epoch - 1))

    def _set_weights(self, weights_file):
        self.logger.info("Loading weights: {}".format(weights_file))
        weights = torch.load(weights_file, map_location=self.device)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        model = self.model.module if self.is_distributed else self.model
        not_matched_keys = model.load_state_dict(weights, strict=False)
        self.logger.info(not_matched_keys)
