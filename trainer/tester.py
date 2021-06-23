import itertools

import torch
from tqdm import tqdm
from tabulate import tabulate

import utils.timer as timer
from eval.coco_eval import COCOMetrics


class Tester:
    def __init__(self, model, postprocess, test_loader, checkpoint_dir, device, gt_file):
        self.model = model
        self.postprocess = postprocess
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gt_file = gt_file
        self.coco_metrics = COCOMetrics(
            gt_file=self.gt_file,
            cat2label=self.test_loader.dataset.CAT2LABEL,
            with_mask=self.test_loader.dataset.with_mask,
            save_dir=self.checkpoint_dir
        )

    def test(self):
        timer.reset()
        timer.cpu() if str(self.device) == 'cpu' else timer.cuda()

        self.model.eval()
        n_iter = len(self.test_loader)
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(self.test_loader), total=n_iter):
                # Load data to device
                image = sample[0].to(self.device)
                batch_info = sample[2]

                # Forward
                with timer.timer('Network Forward'):
                    predict = self.model(image)

                # Postprocess
                with timer.timer('Postprocess'):
                    detections = self.postprocess(predict)

                # Convert to coco format
                with timer.timer('Convert Format'):
                    coco_format_dets = self.coco_metrics.to_coco_format(batch_info, detections)

                self.coco_metrics.update_results(coco_format_dets)

        self.coco_metrics.coco_eval(per_cats=True)
        self.display_coco_eval(eval_type='bbox')
        if self.coco_metrics.with_mask:
            self.display_coco_eval(eval_type='segm')

        timer_log = timer.get_all_elapsed_time()
        print('\n--------------------------------------------------------------------')
        print('Speed Statistics (batch size = {})'.format(self.test_loader.batch_size))
        for key, value in timer_log.items():
            print('%s: %.3fms (%.3ffps)'
                  % (key, value / self.test_loader.batch_size, 1000 * self.test_loader.batch_size / value))

    def display_coco_eval(self, eval_type='bbox'):
        if eval_type == 'bbox':
            eval_stats = self.coco_metrics.bbox_eval_stats
            eval_per_cats_stats = self.coco_metrics.bbox_eval_per_cats_stats
        elif eval_type == 'segm':
            eval_stats = self.coco_metrics.segm_eval_stats
            eval_per_cats_stats = self.coco_metrics.segm_eval_per_cats_stats
        else:
            raise KeyError

        # print('COCO eval {}:'.format(eval_type), ' '.join('%.7f' % k for k in eval_stats))
        table = tabulate(
            eval_stats.reshape(1, -1),
            tablefmt="pipe",
            floatfmt=".3f",
            headers=['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
                     'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL'],
            numalign="left",
        )
        print('\nCOCO eval {}: \n'.format(eval_type) + table)
        eval_per_cats_stats = [(cat, stat) for cat, stat in zip(
            self.test_loader.dataset.CLASSES, eval_per_cats_stats)]
        N_COLS = min(6, len(eval_per_cats_stats) * 2)
        results_flatten = list(itertools.chain(*eval_per_cats_stats))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        print("\nPer-category {} AP: \n".format(eval_type) + table)
