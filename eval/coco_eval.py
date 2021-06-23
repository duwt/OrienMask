import json
import os
import sys

import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class block_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class COCOMetrics:
    """Prepare the predictions after network forward and postprocess for coco evaluation

    `_recover_shape_bbox' and `_recover_shape_segm' convert format to the scale of
    input image. `_to_bbox_coco_format' and `_to_segm_coco_format' step further and
    save as the dict format that official coco defines. `coco_eval' calls the apis in
    `pycocotools' to accumulate all AP and AR metrics. `_get_per_cats_stats' returns
    the AP metric for each category.
    """
    def __init__(self, gt_file, cat2label, with_mask, save_dir):
        self.gt_file = gt_file
        self.cat2label = torch.tensor(cat2label)
        self.with_mask = with_mask
        self.bbox_results = []
        self.segm_results = []
        self.bbox_eval_stats = []
        self.segm_eval_stats = []
        self.bbox_eval_per_cats_stats = []
        self.segm_eval_per_cats_stats = []
        self.bbox_pred_file = os.path.join(save_dir, 'bbox_prediction.json')
        self.segm_pred_file = os.path.join(save_dir, 'segm_prediction.json')
        self.metric_keys = [
            'AP', 'AP50', 'AP75', 'APS', 'APM', 'APL',
            'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL'
        ]

    def reset(self):
        self.bbox_results = []
        self.segm_results = []
        self.bbox_eval_stats = []
        self.segm_eval_stats = []
        self.bbox_eval_per_cats_stats = []
        self.segm_eval_per_cats_stats = []

    def to_coco_format(self, image_info, detections):
        coco_format_result = {'bbox': self._to_bbox_coco_format(image_info, detections)}
        if self.with_mask:
            coco_format_result['segm'] = self._to_segm_coco_format(image_info, detections)
        return coco_format_result

    def update_results(self, coco_format):
        self.bbox_results += coco_format['bbox']
        if self.with_mask:
            self.segm_results += coco_format['segm']

    def save_as_json(self, filename):
        with open(filename, 'w') as handle:
            json.dump({'bbox': self.bbox_results, 'segm': self.segm_results}, handle)

    def update_from_json(self, filename):
        update = json.load(open(filename))
        self.bbox_results += update['bbox']
        self.segm_results += update['segm']

    def coco_eval(self, per_cats=False):
        coco_eval_log = {}
        with block_print():
            gt_coco = COCO(self.gt_file)
            with open(self.bbox_pred_file, 'w') as handle:
                json.dump(self.bbox_results, handle)
            pd_bbox_coco = gt_coco.loadRes(self.bbox_pred_file)
            coco_eval_bbox = COCOeval(gt_coco, pd_bbox_coco, iouType='bbox')
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            self.bbox_eval_stats = coco_eval_bbox.stats
            if per_cats:
                self.bbox_eval_per_cats_stats = self._get_per_cats_stats(coco_eval_bbox)
            for key, value in zip(self.metric_keys, coco_eval_bbox.stats.tolist()):
                coco_eval_log['bbox_{}'.format(key)] = value
            if self.with_mask:
                with open(self.segm_pred_file, 'w') as handle:
                    json.dump(self.segm_results, handle)
                pd_segm_coco = gt_coco.loadRes(self.segm_pred_file)
                coco_eval_segm = COCOeval(gt_coco, pd_segm_coco, iouType='segm')
                coco_eval_segm.evaluate()
                coco_eval_segm.accumulate()
                coco_eval_segm.summarize()
                self.segm_eval_stats = coco_eval_segm.stats
                if per_cats:
                    self.segm_eval_per_cats_stats = self._get_per_cats_stats(coco_eval_segm)
                for key, value in zip(self.metric_keys, coco_eval_segm.stats.tolist()):
                    coco_eval_log['segm_{}'.format(key)] = value
        return coco_eval_log

    def _to_segm_coco_format(self, batch_info, detections):
        segm_results = []
        for sample_info, sample_detection in zip(batch_info, detections):
            sample_bbox = sample_detection['bbox']
            if sample_bbox.numel() == 0:
                continue
            sample_mask, sample_cls = sample_detection['mask'], sample_detection['cls']
            sample_mask = self._recover_shape_segm(sample_mask, sample_info)
            sample_score = sample_bbox[:, -1].tolist()
            image_id = sample_info['id']
            sample_cls = self.cat2label[sample_cls.flatten().cpu()].tolist()
            for mask, score, cls in zip(sample_mask, sample_score, sample_cls):
                mask = mask.cpu().numpy().astype(np.uint8)
                rle_mask = maskUtils.encode(np.asfortranarray(mask))
                rle_mask['counts'] = rle_mask['counts'].decode("utf-8")
                segm_results.append({
                    "image_id": image_id, "category_id": cls,
                    "segmentation": rle_mask, "score": score
                })
        return segm_results

    def _to_bbox_coco_format(self, batch_info, detections):
        bbox_results = []
        for sample_info, sample_detection in zip(batch_info, detections):
            sample_bbox, sample_cls = sample_detection['bbox'], sample_detection['cls']
            if sample_bbox.numel() == 0:
                continue
            sample_xywh = self._recover_shape_bbox(sample_bbox[:, :4], sample_info).tolist()
            sample_score = sample_bbox[:, -1].tolist()
            image_id = sample_info['id']
            sample_cls = self.cat2label[sample_cls.flatten().cpu()].tolist()
            for xywh, score, cls in zip(sample_xywh, sample_score, sample_cls):
                bbox_results.append({
                    "image_id": image_id, "category_id": cls,
                    "bbox": xywh, "score": score
                })
        return bbox_results

    @staticmethod
    def _recover_shape_bbox(bbox, sample_info):
        """Recover bounding boxes predictions matching original images

        Args:
            bbox (n, 4): bounding boxes after postprocessing with `x_center',
                `y_center', `width' and `height' in the normalized form [0, 1].
            sample_info (dict): for exactly recover the boxes attributes under
                original image conditions. `height' and `width' is the size of
                original image, `pad' contains the paddings in resize transformation
                and the size of network input, `collate_pad' is the padding before
                sending to the network. `hflip' and `vflip' represent two directional
                flipping transformations. (other info is not supported yet)
        """
        bx, by, bw, bh = bbox.split(1, dim=-1)
        if sample_info.get('collate_pad') is not None:
            left, right, top, down, h, w = sample_info['collate_pad']
            nh = h - top - down
            nw = w - left - right
            bx = (bx * w - left) / nw
            by = (by * h - top) / nh
            bw = bw * w / nw
            bh = bh * h / nh
        if sample_info.get('pad') is not None:
            top, down, left, right, h, w = sample_info['pad']
            nh = h - top - down
            nw = w - left - right
            bx = (bx * w - left) / nw
            by = (by * h - top) / nh
            bw = bw * w / nw
            bh = bh * h / nh
        if sample_info.get('hflip', False):
            bx = 1 - bx
        if sample_info.get('vflip', False):
            by = 1 - by

        oh, ow = sample_info['height'], sample_info['width']
        bx = (bx - bw / 2) * ow
        by = (by - bh / 2) * oh
        bw = bw * ow
        bh = bh * oh
        xywh = torch.cat([bx, by, bw, bh], dim=-1)
        return xywh

    @staticmethod
    def _recover_shape_segm(mask, sample_info):
        if sample_info.get('collate_pad') is not None:
            left, right, top, down = sample_info['collate_pad'][:4]
            mask = mask[:, top:-down if down else None, left:-right if right else None]
        if sample_info.get('pad') is not None:
            top, down, left, right = sample_info['pad'][:4]
            mask = mask[:, top:-down if down else None, left:-right if right else None]
        if sample_info.get('hflip', False):
            mask = torch.flip(mask, dims=(2,))
        if sample_info.get('vflip', False):
            mask = torch.flip(mask, dims=(1,))

        oh, ow = sample_info['height'], sample_info['width']
        mask = F.interpolate(mask.unsqueeze(0).float(), size=(oh, ow), mode='bilinear', align_corners=False)
        return mask.squeeze(0).round().to(torch.uint8)

    def _get_per_cats_stats(self, coco_eval_obj):
        precisions = coco_eval_obj.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert self.cat2label.numel() == precisions.shape[2]
        results_per_category = []
        for idx in range(self.cat2label.numel()):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(float(ap * 100))
        return results_per_category
