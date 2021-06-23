import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .function import batched_nms


class OrienMaskYOLOPostProcess:
    def __init__(self, grid_size, image_size, anchors, anchor_mask, num_classes,
                 conf_thresh=0.05, nms_func=None, nms_pre=400,
                 nms_post=100, orien_thresh=0.3, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.nHs = [size[0] for size in grid_size]
        self.nWs = [size[1] for size in grid_size]
        self.scales = len(grid_size)
        self.image_h, self.image_w = _pair(image_size)
        self.pixel_anchors = torch.tensor(anchors, device=self.device, dtype=torch.float32)
        self.normalized_anchors = torch.empty_like(self.pixel_anchors)
        self.normalized_anchors[:, 0] = self.pixel_anchors[:, 0] / self.image_w
        self.normalized_anchors[:, 1] = self.pixel_anchors[:, 1] / self.image_h
        self.grid_anchors = self.normalized_anchors.clone()
        self.grid_sizes = self.normalized_anchors.clone()
        for m, nH, nW in zip(anchor_mask, self.nHs, self.nWs):
            self.grid_anchors[m, 0] *= nW
            self.grid_anchors[m, 1] *= nH
            self.grid_sizes[m, 0] = nW
            self.grid_sizes[m, 1] = nH

        self.anchor_mask = anchor_mask
        self.num_anchors = [len(m) for m in anchor_mask]
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh                      # confidence threshold
        self.nms = nms_func if nms_func else batched_nms    # nms function
        self.nms_pre = nms_pre                              # max prediction before nms
        self.nms_post = nms_post                            # max prediction after nms
        self.orien_thresh = orien_thresh                    # distance threshold to center

        self.base_xy = torch.zeros(self.pixel_anchors.size(0), 2, self.image_h, self.image_w, device=self.device)
        self.grid_x, self.grid_y, self.anchor_idx = [], [], []
        for m, nA, nH, nW in zip(anchor_mask, self.num_anchors, self.nHs, self.nWs):
            # base coordinates of orientations (unit size = grid)
            base_y, base_x = torch.meshgrid(
                [torch.arange(self.image_h, device=self.device, dtype=torch.float) / self.image_h * nH,
                 torch.arange(self.image_w, device=self.device, dtype=torch.float) / self.image_w * nW])
            self.base_xy[m] = torch.stack([base_x, base_y], dim=0)

            # base coordinates of grid centers (unit size = grid)
            grid_y, grid_x = torch.meshgrid(
                [torch.arange(nH, device=self.device, dtype=torch.float),
                 torch.arange(nW, device=self.device, dtype=torch.float)])
            self.grid_y.append(grid_y.expand(nA, nH, nW).contiguous())
            self.grid_x.append(grid_x.expand(nA, nH, nW).contiguous())

            # anchor indices of all predictions
            anchor_idx = torch.tensor(m, device=self.device).view(nA, 1, 1)
            self.anchor_idx.append(anchor_idx.expand(nA, nH, nW).contiguous())

        # concat here for convenient usage
        self.dets_grid_x = torch.cat([grid_x.view(-1) for grid_x in self.grid_x], dim=0)
        self.dets_grid_y = torch.cat([grid_y.view(-1) for grid_y in self.grid_y], dim=0)
        self.dets_anchor_idx = torch.cat([anchor_idx.view(-1) for anchor_idx in self.anchor_idx], dim=0)

    def __call__(self, predict):
        return self.apply(predict)

    def apply(self, predict):
        nB = predict[0][0].size(0)
        pred_bbox_batch = [predict_i[0] for predict_i in predict]
        pred_orien_batch = [
            F.interpolate(predict_i[1], scale_factor=4.0, mode='bilinear', align_corners=False)
            for predict_i in predict
        ]

        final_pred = []
        for b in range(nB):
            dets_coord, dets_conf = [], []
            dets_orien = torch.zeros_like(self.base_xy)
            for i in range(self.scales):
                pred_bbox, pred_orien = pred_bbox_batch[i][b], pred_orien_batch[i][b]
                nA = self.num_anchors[i]
                nH, nW = self.nHs[i], self.nWs[i]
                anchor_mask = self.anchor_mask[i]
                anchors = self.normalized_anchors[anchor_mask]
                grid_x, grid_y = self.grid_x[i], self.grid_y[i]

                pred_bbox = pred_bbox.view(nA, -1, nH, nW).permute(0, 2, 3, 1).contiguous()
                pred_orien = pred_orien.view(nA, 2, self.image_h, self.image_w)
                pred_coord, pred_conf = self.get_boxes(pred_bbox, nH, nW, anchors, grid_x, grid_y)

                dets_coord.append(pred_coord)
                dets_conf.append(pred_conf)
                dets_orien[anchor_mask] = pred_orien

            dets_coord = torch.cat(dets_coord, dim=0)
            dets_conf = torch.cat(dets_conf, dim=0)
            dets_grid_y = self.dets_grid_y
            dets_grid_x = self.dets_grid_x
            dets_anchor_idx = self.dets_anchor_idx
            dets_orien = self.get_orien_grid(dets_orien)

            # filter out detections with low-confidence
            selected_inds, dets_cls = torch.nonzero(dets_conf > self.conf_thresh, as_tuple=True)
            selected_inds = selected_inds.view(-1)
            dets_cls = dets_cls.view(-1)
            dets_conf = dets_conf[selected_inds, dets_cls]
            # the maximum number of detections sent to nms
            if selected_inds.numel() > self.nms_pre:
                dets_conf, topk_inds = dets_conf.topk(self.nms_pre)
                selected_inds = selected_inds[topk_inds]
                dets_cls = dets_cls[topk_inds]
            dets_coord = dets_coord[selected_inds]
            dets_anchor_idx = dets_anchor_idx[selected_inds]
            dets_grid_y = dets_grid_y[selected_inds]
            dets_grid_x = dets_grid_x[selected_inds]

            # non maximum suppression
            final_pred.append(
                self.multi_class_nms(
                    dets_coord, dets_conf, dets_cls, dets_anchor_idx,
                    dets_grid_y, dets_grid_x, dets_orien
                )
            )

        return final_pred

    def get_boxes(self, predict, nH, nW, anchors, grid_x, grid_y):
        pred_coord = predict[..., 0:4]
        pred_obj = predict[..., 4].sigmoid().view(-1)
        pred_cls = predict[..., 5:].sigmoid().view(-1, self.num_classes)
        pred_conf = pred_cls * pred_obj.unsqueeze(-1)

        anchor_w, anchor_h = anchors.split(1, dim=1)
        pred_coord[..., 0] = (pred_coord[..., 0].sigmoid() + grid_x) / nW
        pred_coord[..., 1] = (pred_coord[..., 1].sigmoid() + grid_y) / nH
        pred_coord[..., 2] = pred_coord[..., 2].exp() * anchor_w.view(-1, 1, 1)
        pred_coord[..., 3] = pred_coord[..., 3].exp() * anchor_h.view(-1, 1, 1)
        pred_coord = pred_coord.view(-1, 4)

        return pred_coord, pred_conf

    def get_orien_grid(self, pred_orien):
        pixel_orien = pred_orien * self.grid_anchors.view(-1, 2, 1, 1) / 2
        pixel_orien += self.base_xy
        return pixel_orien

    def multi_class_nms(self, pred_coord, pred_conf, pred_cls, pred_anchor_idx, grid_y, grid_x, pred_orien):
        dets = torch.cat([pred_coord, pred_conf.unsqueeze(-1)], dim=1)
        dets, cats, keep = self.nms(dets, pred_cls)

        if keep.numel() > self.nms_post:
            _, topk_inds = dets[:, -1].topk(self.nms_post)
            dets = dets[topk_inds]
            cats = cats[topk_inds]
            keep = keep[topk_inds]

        anchor_idx = pred_anchor_idx[keep]
        x_centers = (self.grid_sizes[anchor_idx, 0] * dets[:, 0]).view(-1, 1, 1)
        y_centers = (self.grid_sizes[anchor_idx, 1] * dets[:, 1]).view(-1, 1, 1)
        det_width = dets[:, 2].view(-1, 1, 1)
        det_height = dets[:, 3].view(-1, 1, 1)
        masks = ((torch.abs(pred_orien[anchor_idx, 0] - x_centers) <
                  self.orien_thresh * det_width * self.grid_sizes[anchor_idx, 0].view(-1, 1, 1)) &
                 (torch.abs(pred_orien[anchor_idx, 1] - y_centers) <
                  self.orien_thresh * det_height * self.grid_sizes[anchor_idx, 1].view(-1, 1, 1)))

        return {'bbox': dets, 'mask': masks, 'cls': cats}
