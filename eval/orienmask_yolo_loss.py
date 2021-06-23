import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .base import BaseLoss, BaseMultiScaleLoss
from .function import bbox_ious, anchor_ious


class OrienMaskYOLOLoss(BaseLoss):
    def __init__(self, grid_size, image_size, anchors, anchor_mask, num_classes,
                 loss_id, loss_sum_id, metric_id, center_region=0.6, valid_region=0.6,
                 label_smooth=False, obj_ignore_threshold=0.5, weight=None):
        super(OrienMaskYOLOLoss, self).__init__(loss_id, loss_sum_id, metric_id, weight)
        self.device = None
        self.grid_h, self.grid_w = _pair(grid_size)
        self.image_h, self.image_w = _pair(image_size)
        self.num_anchors = len(anchor_mask)
        self.anchor_mask = anchor_mask if anchor_mask else list(range(self.num_anchors))
        self.num_classes = num_classes
        self.center_region = center_region
        self.valid_region = valid_region
        self.label_smooth = 1.0 / max(num_classes, 40) if label_smooth else 0
        self.obj_ignore_threshold = obj_ignore_threshold

        # coordinate ratio between grid-level and pixel-level
        self.grid_wh = torch.tensor([self.grid_w, self.grid_h]).float()
        self.image_wh = torch.tensor([self.image_w, self.image_h]).float()
        self.scale_wh = self.image_wh / self.grid_wh
        # store pixel-level / grid-level / normalized all anchors
        self.pixel_all_anchors = torch.tensor(anchors).float()
        self.norm_all_anchors = self.pixel_all_anchors / self.image_wh
        self.grid_all_anchors = self.pixel_all_anchors / self.scale_wh
        # store selected anchors with anchor_mask
        self.pixel_anchors = self.pixel_all_anchors[anchor_mask]
        self.norm_anchors = self.norm_all_anchors[anchor_mask]
        self.grid_anchors = self.grid_all_anchors[anchor_mask]
        # store mesh indices
        self.grid_mesh_y, self.grid_mesh_x = torch.meshgrid([
            torch.arange(self.grid_h, dtype=torch.float32),
            torch.arange(self.grid_w, dtype=torch.float32)
        ])
        self.grid_mesh_xy = torch.stack([self.grid_mesh_x, self.grid_mesh_y], dim=-1)
        self.pixel_mesh_y, self.pixel_mesh_x = torch.meshgrid([
            torch.arange(self.image_h, dtype=torch.float32),
            torch.arange(self.image_w, dtype=torch.float32)
        ])
        self.pixel_mesh_xy = torch.stack((self.pixel_mesh_x, self.pixel_mesh_y), dim=-1)

        # loss type
        self.l1 = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def _set_device(self, device):
        # update device of tensors for convenience
        self.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def _get_loss(self, predict, target, training=True):
        # pred_bbox (nB, nA * (coord4 + obj1 + cls), grid_h, grid_w)
        # pred_orien (nB, nA * (x1 + y1), image_h, image_w)
        pred_bbox, pred_orien = predict
        if pred_bbox.device != self.device:
            self._set_device(pred_bbox.device)

        nB = pred_bbox.size(0)
        nA = self.num_anchors
        nH, nW = self.grid_h, self.grid_w

        # pred_bbox (nB, nA, grid_h, grid_w, coord4 + obj1 + cls)
        # pred_orien (nB, nA, image_h, image_w, x1 + y1)
        pred_bbox = pred_bbox.view(nB, nA, -1, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        pred_orien = F.interpolate(pred_orien, scale_factor=4, mode='bilinear', align_corners=False)
        pred_orien = pred_orien.view(nB, nA, 2, self.image_h, self.image_w).permute(0, 1, 3, 4, 2).contiguous()

        # predict x, y, w, h, obj, cls, orien
        pred_xy = pred_bbox[..., 0:2].sigmoid()
        pred_wh = pred_bbox[..., 2:4]
        pred_obj = pred_bbox[..., 4].sigmoid()
        pred_cls = pred_bbox[..., 5:].sigmoid()

        # predict boxes
        # xy = sigmoid(xy) + grid_xy
        # wh = exp(wh) * anchor_wh
        pred_boxes = torch.zeros(nB, nA * nH * nW, 4, device=self.device)
        pred_boxes[..., 0:2] = (pred_xy.detach() + self.grid_mesh_xy).view(nB, -1, 2)
        pred_boxes[..., 2:4] = (pred_wh.detach().exp() * self.grid_anchors.view(1, -1, 1, 1, 2)).view(nB, -1, 2)

        # build target
        bbox_pos_mask, bbox_neg_mask, bbox_pos_scale, txy, twh, tiou, tcls, \
        orien_pos_mask, orien_neg_mask, torien \
            = self.build_targets(pred_boxes, target)

        if not torch.isfinite(pred_wh).all():
            print('pred_wh not finite')
            exit()

        # calculate bbox loss
        loss_xy = (self.bce(pred_xy, txy) * bbox_pos_scale.unsqueeze(-1)).sum() / nB
        loss_wh = (self.mse(pred_wh, twh) * bbox_pos_scale.unsqueeze(-1)).sum() / 2 / nB
        loss_obj_all = self.bce(pred_obj, bbox_pos_mask)
        loss_obj_pos = (loss_obj_all * bbox_pos_mask).sum() / nB
        loss_obj_neg = (loss_obj_all * bbox_neg_mask).sum() / nB
        loss_cls = (self.bce(pred_cls, tcls) * bbox_pos_mask.unsqueeze(-1)).sum() / nB

        # calculate orien loss
        num_orien_pos = orien_pos_mask.sum()
        num_orien_neg = orien_neg_mask.sum()
        loss_orien_all = self.smoothl1(pred_orien, torien)
        loss_orien_pos = (loss_orien_all * orien_pos_mask.unsqueeze(-1)).sum() \
                         / num_orien_pos * bbox_pos_mask.sum() / nB \
            if num_orien_pos > 0 else pred_orien.new_zeros([])
        loss_orien_neg = (loss_orien_all * orien_neg_mask.unsqueeze(-1)).sum() \
                         / num_orien_neg * bbox_pos_mask.sum() / nB \
            if num_orien_neg > 0 else pred_orien.new_zeros([])

        loss_items = (loss_xy, loss_wh, loss_obj_pos, loss_obj_neg,
                      loss_cls, loss_orien_pos, loss_orien_neg)

        # calculate metric for evaluation
        metric_items = ()
        if training is not True:
            bbox_pos_count = bbox_pos_mask.sum()
            bbox_neg_count = bbox_neg_mask.sum()
            cls_conf = (pred_cls * (tcls > 0.5).float()).sum()
            obj_pos = (pred_obj * bbox_pos_mask).sum()
            obj_neg = (pred_obj * bbox_neg_mask).sum()
            avg_iou = tiou.sum()
            recall50 = (tiou > 0.5).sum()
            recall75 = (tiou > 0.75).sum()
            orien_pos_count = orien_pos_mask.sum()
            orien_neg_count = orien_neg_mask.sum()
            orien_delta = (pred_orien - torien).abs()
            orien_pos_acc = ((orien_delta < 0.5).float() * orien_pos_mask.unsqueeze(-1)).sum()
            orien_neg_acc = ((orien_delta < 0.5).float() * orien_neg_mask.unsqueeze(-1)).sum()

            # pass statistics to counter
            metric_cls_conf = (cls_conf, bbox_pos_count)
            metric_obj_pos = (obj_pos, bbox_pos_count)
            metric_obj_neg = (obj_neg, bbox_neg_count)
            metric_avg_iou = (avg_iou, bbox_pos_count)
            metric_recall50 = (recall50, bbox_pos_count)
            metric_recall75 = (recall75, bbox_pos_count)
            metric_orien_pos_acc = (orien_pos_acc, orien_pos_count * 2)
            metric_orien_neg_acc = (orien_neg_acc, orien_neg_count * 2)
            metric_items = (metric_cls_conf, metric_obj_pos, metric_obj_neg,
                            metric_avg_iou, metric_recall50, metric_recall75,
                            metric_orien_pos_acc, metric_orien_neg_acc)

        return loss_items, metric_items

    def build_targets(self, pred_boxes, target):
        # clone ground truth to avoid in-place operation
        gt_bbox, gt_cls, gt_index, gt_mask = \
            target[0].clone(), target[1].clone(), target[2].clone(), target[3].clone()

        nB = len(gt_index) - 1
        nA = self.num_anchors
        nH, nW = self.grid_h, self.grid_w

        bbox_pos_mask = torch.zeros(nB, nA, nH, nW, device=self.device)
        bbox_neg_mask = torch.ones(nB, nA, nH, nW, device=self.device)
        bbox_pos_scale = torch.zeros(nB, nA, nH, nW, device=self.device)
        txy = torch.zeros(nB, nA, nH, nW, 2, device=self.device)
        twh = torch.zeros(nB, nA, nH, nW, 2, device=self.device)
        tiou = torch.zeros(nB, nA, nH, nW, device=self.device)
        tcls = torch.full((nB, nA, nH, nW, self.num_classes), self.label_smooth, device=self.device, dtype=torch.float)
        orien_mask = torch.zeros(nB, nA, self.image_h, self.image_w, device=self.device, dtype=torch.long)
        torien = torch.zeros(nB, nA, self.image_h, self.image_w, 2, device=self.device)

        # use grid as unit size
        gt_bbox = gt_bbox * torch.tensor([nW, nH, nW, nH], device=self.device, dtype=torch.float32)

        for b in range(nB):
            # skip sample with no instance
            gt_index_current, gt_index_next = gt_index[b], gt_index[b + 1]
            if gt_index_current == gt_index_next:
                continue

            # take ground truth of b-th sample
            # refer to collate_fn for details
            gt_bbox_b = gt_bbox[gt_index_current:gt_index_next]
            gt_cls_b = gt_cls[gt_index_current:gt_index_next]
            gt_mask_b = gt_mask[gt_index_current:gt_index_next]

            # ignore predictions if iou(pred, ground_true) is larger than threshold
            iou_pred_gt = bbox_ious(pred_boxes[b], gt_bbox_b)
            is_ignore = (iou_pred_gt > self.obj_ignore_threshold).any(dim=1)
            bbox_neg_mask[b].masked_fill_(is_ignore.view_as(bbox_neg_mask[b]), 0)

            # match ground truth with anchor according to iou
            # remove anchors with argmax(iou) belonging to other scale
            iou_gt_anchors = anchor_ious(gt_bbox_b[:, 2:], self.grid_all_anchors)
            match_index = iou_gt_anchors.argmax(dim=1)
            match_mask = torch.tensor([best_n in self.anchor_mask for best_n in match_index], device=self.device)
            match_index = torch.masked_select(match_index, match_mask)
            if match_index.numel() == 0:
                continue

            # positive indices
            match_anchor = torch.zeros_like(match_index)
            for idx, mask_id in enumerate(self.anchor_mask):
                match_anchor[match_index == mask_id] = idx
            gt_xy, gt_wh = gt_bbox_b[match_mask].split(2, dim=-1)
            grid_x = torch.clamp(torch.floor(gt_xy[:, 0]), 0, nW - 1).long()
            grid_y = torch.clamp(torch.floor(gt_xy[:, 1]), 0, nH - 1).long()
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)

            # bbox targets
            # take grid as unit size
            bbox_pos_mask[b, match_anchor, grid_y, grid_x] = 1
            bbox_neg_mask[b, match_anchor, grid_y, grid_x] = 0
            bbox_pos_scale[b, match_anchor, grid_y, grid_x] = 2 - torch.prod(gt_wh, dim=-1) / (nW * nH)
            txy[b, match_anchor, grid_y, grid_x] = gt_xy - grid_xy.float()
            twh[b, match_anchor, grid_y, grid_x] = torch.log(gt_wh / self.grid_anchors[match_anchor])
            tcls[b, match_anchor, grid_y, grid_x, gt_cls_b[match_mask]] = 1 - self.label_smooth
            match_gt = torch.arange(gt_bbox_b.size(0), device=self.device)[match_mask]
            tiou[b, match_anchor, grid_y, grid_x] = \
                iou_pred_gt.view(*tiou.shape[1:], -1)[match_anchor, grid_y, grid_x, match_gt]

            # orientation targets
            # take pixel as coordinate unit
            gt_mask_b = gt_mask_b[match_mask]
            pixel_x = (gt_xy[:, 0] * self.scale_wh[0]).flatten()
            pixel_y = (gt_xy[:, 1] * self.scale_wh[1]).flatten()
            # extend box region
            valid_x = (gt_xy[:, 0] * self.scale_wh[0]).flatten()
            valid_y = (gt_xy[:, 1] * self.scale_wh[1]).flatten()
            valid_w = ((gt_wh[:, 0] * self.valid_region + 0.5) * self.scale_wh[0]).flatten()
            valid_h = ((gt_wh[:, 1] * self.valid_region + 0.5) * self.scale_wh[1]).flatten()
            center_wh = torch.stack([valid_w, valid_h], dim=-1) / self.valid_region * self.center_region
            region_x1 = (valid_x - valid_w).clamp(min=0, max=self.image_w - 1).round().long()
            region_x2 = (valid_x + valid_w).clamp(min=0, max=self.image_w - 1).round().long() + 1
            region_y1 = (valid_y - valid_h).clamp(min=0, max=self.image_h - 1).round().long()
            region_y2 = (valid_y + valid_h).clamp(min=0, max=self.image_h - 1).round().long() + 1

            for gt_inst_mask, a, x1, x2, y1, y2, x, y, wh in zip(
                    gt_mask_b, match_anchor, region_x1, region_x2,
                    region_y1, region_y2, pixel_x, pixel_y, center_wh):
                # relative position to box center
                offset_xy = self.pixel_mesh_xy.clone()
                offset_xy[..., 0] -= x
                offset_xy[..., 1] -= y

                # clone data to avoid ambiguity
                orien_mask_inst = orien_mask[b, a].clone()
                torien_inst = torien[b, a].clone()

                # roi region
                is_roi = (self.pixel_mesh_x >= float(x1)) & (self.pixel_mesh_x < float(x2)) & \
                         (self.pixel_mesh_y >= float(y1)) & (self.pixel_mesh_y < float(y2))

                # current instance region in roi
                # set orien_mask = -1 and set torien pointing to the base position
                is_inst = (is_roi & (gt_inst_mask > 0))
                orien_mask_inst.masked_fill_(is_inst, -1)
                torien_inst = torch.where(is_inst.unsqueeze(-1).expand_as(offset_xy), offset_xy, torien_inst)

                # no instance region in roi
                # set orien_mask += 1 and set torien pointing to the boarder of extended box
                not_inst = (is_roi & (gt_inst_mask == 0) & (orien_mask_inst >= 0))
                orien_mask_inst += not_inst.long()
                offset_xy_length = offset_xy.abs().clamp(min=1e-8)
                neg_offset_scale = (wh / offset_xy_length).clamp(min=1).min(dim=-1)[0] - 1
                neg_offset = neg_offset_scale.unsqueeze(-1) * offset_xy.sign() * offset_xy_length
                torien_inst = torch.where(not_inst.unsqueeze(-1).expand_as(offset_xy),
                                          torien_inst + neg_offset, torien_inst)

                # update orien_mask and torien
                orien_mask[b, a] = orien_mask_inst.clone()
                torien[b, a] = torien_inst.clone()

        # set negative ones as the average of their torien sums
        orien_pos_mask = (orien_mask < 0).float()
        orien_neg_mask = (orien_mask > 0).float()
        is_invalid = (orien_mask == 0)
        torien = torien / (self.pixel_anchors.view(1, nA, 1, 1, 2) / 2)
        orien_mask.masked_fill_(is_invalid, 1000)
        torien = torien / orien_mask.unsqueeze(-1).float()

        return (bbox_pos_mask, bbox_neg_mask, bbox_pos_scale, txy, twh, tiou, tcls,
                orien_pos_mask, orien_neg_mask, torien)


class OrienMaskYOLOMultiScaleLoss(BaseMultiScaleLoss):
    def __init__(self, grid_size, image_size, anchors, anchor_mask, num_classes,
                 loss_id=("loss_xy", "loss_wh", "loss_obj", "loss_noobj",
                          "loss_cls", "loss_orien_pos", "loss_orien_neg"),
                 loss_sum_id="loss_sum", scales_id=("S32", "S16", "S08"),
                 metric_id=("cls_conf", "obj_pos", "obj_neg", "avg_iou",
                            "recall50", "recall75", "orien_pos_acc", "orien_neg_acc"),
                 center_region=0.6, valid_region=0.7, label_smooth=False,
                 obj_ignore_threshold=0.5, weight=None, scales_weight=None):
        assert len(grid_size) == len(anchor_mask) == len(scales_id)
        self.grid_size = grid_size
        self.image_size = image_size
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.center_region = center_region
        self.valid_region = valid_region
        self.label_smooth = label_smooth
        self.obj_ignore_threshold = obj_ignore_threshold
        self.weight = weight
        super(OrienMaskYOLOMultiScaleLoss, self).__init__(
            loss_id, loss_sum_id, metric_id, scales_id, scales_weight)

    def _construct_scales_loss(self):
        loss = []
        for i in range(self.num_scales):
            scale_weight = [self.scales_weight[i] * weight_item for weight_item in self.weight] \
                if self.weight is not None else None
            loss.append(
                OrienMaskYOLOLoss(
                    self.grid_size[i], self.image_size, self.anchors, self.anchor_mask[i],
                    self.num_classes, self.scales_loss_id[i], self.scales_loss_sum_id[i],
                    self.scales_metric_id[i], self.center_region, self.valid_region,
                    self.label_smooth, self.obj_ignore_threshold, scale_weight
                )
            )
        return loss
