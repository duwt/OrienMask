import torch

from . import nms_cpu, nms_cuda


def bbox_ious(bbox1, bbox2):
    """IoUs of bounding boxes with x, y, width and height

    Args:
        bbox1 (n1, 4) / (b, n1, 4)
        bbox2 (n2, 4) / (b, n2, 4)

    Returns:
        iou (n1, n2) / (b, n1, n2)
    """
    b1x1, b1y1 = (bbox1[..., 0:2] - bbox1[..., 2:4] / 2).split(1, -1)
    b1x2, b1y2 = (bbox1[..., 0:2] + bbox1[..., 2:4] / 2).split(1, -1)
    b2x1, b2y1 = (bbox2[..., 0:2] - bbox2[..., 2:4] / 2).split(1, -1)
    b2x2, b2y2 = (bbox2[..., 0:2] + bbox2[..., 2:4] / 2).split(1, -1)

    dx = (b1x2.min(b2x2.squeeze(-1).unsqueeze(-2)) -
          b1x1.max(b2x1.squeeze(-1).unsqueeze(-2))).clamp(min=0)
    dy = (b1y2.min(b2y2.squeeze(-1).unsqueeze(-2)) -
          b1y1.max(b2y1.squeeze(-1).unsqueeze(-2))).clamp(min=0)
    inter = dx * dy

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = (area1 + area2.squeeze(-1).unsqueeze(-2)) - inter

    return inter / union


def anchor_ious(bbox1, bbox2):
    """IoUs of bounding boxes with width and height

    Args:
        bbox1 (n1, 2)
        bbox2 (n2, 2)

    Returns:
        iou (n1, n2)
    """
    dx = bbox1[:, 0:1].min(bbox2[:, 0:1].t())
    dy = bbox1[:, 1:2].min(bbox2[:, 1:2].t())
    inter = dx * dy

    area1 = bbox1[:, 0:1] * bbox1[:, 1:2]
    area2 = bbox2[:, 0:1] * bbox2[:, 1:2]
    union = (area1 + area2.t()) - inter

    return inter / union


def nms(dets, cats, threshold=0.5):
    """Non maximum suppression

    Args:
        dets (n, 5): bounding boxes with x, y, width, height and score
        cats (n,): categories of bounding boxes
        threshold (float): IoU threshold for NMS

    Returns:
        remains dets, cats and their indices
    """
    if dets.size(0) == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
    else:
        if dets.is_cuda:
            keep = nms_cuda.nms(dets, threshold)
        else:
            keep = nms_cpu.nms(dets, threshold)

    return dets[keep], cats[keep], keep


def batched_nms(dets, cats, threshold=0.5, normalized=True):
    """Batched non maximum suppression

    Refer to `torchvision.ops.boxes.batched_nms` for details.

    Args:
        dets (n, 6): bounding boxes with x, y, width, height and score
        cats (n,): cls indices of bounding boxes
        threshold (float): IoU threshold for NMS
        normalized (bool): use normalized coordinates or not

    Returns:
        remains dets, cats and their indices
    """
    if dets.size(0) == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
    else:
        max_coordinate = 1.5 if normalized else dets[:, :2].max() + dets[:, 2:4].max() / 2
        offsets = cats.float().view(-1, 1) * (max_coordinate + 0.5)
        dets_for_nms = dets.clone()
        dets_for_nms[:, :2] += offsets
        if dets.is_cuda:
            keep = nms_cuda.nms(dets_for_nms, threshold)
        else:
            keep = nms_cpu.nms(dets_for_nms, threshold)

    return dets[keep], cats[keep], keep
