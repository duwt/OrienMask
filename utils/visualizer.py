import random

import cv2
import torch
import torch.nn.functional as F

import data as data_module


PALETTE = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63,  81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139)
)


class InferenceVisualizer:
    def __init__(self, dataset, device, with_mask=True, conf_thresh=0.3,
                 alpha=0.5, line_thickness=1):
        dataset_module = getattr(data_module, dataset + 'Dataset')
        self.cat2label = torch.tensor(dataset_module.CAT2LABEL, dtype=torch.uint8, device=device)
        self.classes = dataset_module.CLASSES
        self.device = device
        self.with_mask = with_mask
        self.conf_thresh = conf_thresh
        self.alpha = alpha
        self.line_thickness = line_thickness
        self.palette = torch.tensor(PALETTE, dtype=torch.float32, device=device)

    def __call__(self, detections, image, pad_info):
        # image float tensor shape (h, w, 3)
        pred_show = image.clone()
        height, width = pred_show.shape[:2]

        bbox_dets = detections['bbox']
        cls_dets = detections['cls']
        filtered_idx = bbox_dets[:, -1] > self.conf_thresh
        bbox_dets = bbox_dets[filtered_idx]
        cls_dets = cls_dets[filtered_idx]
        if self.with_mask:
            mask_dets = detections['mask'][filtered_idx]
        if bbox_dets.numel() > 0:
            all_xyxy = self._recover_shape_bbox(bbox_dets[:, :4], width, height, pad_info)
            all_score = bbox_dets[:, -1]
            all_cls = [self.classes[cls] for cls in cls_dets]

            colors_idx = torch.arange(bbox_dets.size(0)) * 5 + random.randint(1, self.palette.size(0))
            colors = self.palette[colors_idx % self.palette.size(0)]

            if self.with_mask:
                all_mask = self._recover_shape_segm(mask_dets, width, height, pad_info)
                sorted_indices = all_mask.sum(dim=2).sum(dim=1).argsort()
                all_mask = all_mask[sorted_indices]
                self.plot_all_mask(all_mask, pred_show, colors[sorted_indices])

            pred_show = pred_show.round().to(torch.uint8).cpu().numpy()
            for xyxy, score, cls, color in zip(all_xyxy, all_score, all_cls, colors):
                print(xyxy.tolist(), score.item(), cls)
                text = '%s %.2f' % (cls, score.item())
                self.plot_one_box(xyxy, text, pred_show, color=color.tolist())
        else:
            pred_show = pred_show.round().to(torch.uint8).cpu().numpy()

        return pred_show

    def plot_one_box(self, bbox, text, image, color):
        x1, y1, x2, y2 = bbox.cpu().tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=self.line_thickness)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]
        font_scale = 0.4
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text, font_face, font_scale, font_thickness)[0]
        cv2.rectangle(image, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(image, text, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def plot_all_mask(self, mask, image, colors):
        color_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3) * colors.view(-1, 1, 1, 3) * self.alpha
        alpha_cum = (1 - self.alpha * mask).cumprod(dim=0).unsqueeze(-1)
        image.mul_(alpha_cum[-1]).add_(color_mask[0])
        if image.shape[0] > 1:
            image.add_((color_mask[1:] * alpha_cum[:-1]).sum(dim=0))

    @classmethod
    def _recover_shape_bbox(cls, bbox, width, height, pad_info):
        bx, by, bw, bh = bbox.split(1, dim=-1)

        left, right, top, down, h, w = pad_info
        nh = h - top - down
        nw = w - left - right
        bx = (bx * w - left) / nw
        by = (by * h - top) / nh
        bw = bw * w / nw
        bh = bh * h / nh

        bx1 = (bx - bw / 2) * width
        by1 = (by - bh / 2) * height
        bx2 = (bx + bw / 2) * width
        by2 = (by + bh / 2) * height
        xyxy = torch.cat([bx1, by1, bx2, by2], dim=-1)
        return xyxy.round().long()

    @classmethod
    def _recover_shape_segm(cls, mask, width, height, pad_info):
        left, right, top, down = pad_info[:4]
        mask = mask[:, top:-down if down else None, left:-right if right else None]
        mask = F.interpolate(mask.float().unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
        return mask.squeeze(0)
