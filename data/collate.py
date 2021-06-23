import math

import torch
import torch.nn.functional as F

from utils.envs import get_torch_device


def naive_collate(batch):
    return batch


def collate(batch):
    # cat bbox, cls, mask and store index to recover
    batch_image = torch.stack([sample['image'] for sample in batch], dim=0)
    batch_bbox = torch.cat([sample['bbox'] for sample in batch], dim=0)
    batch_cls = torch.cat([sample['cls'] for sample in batch], dim=0)
    batch_index = torch.tensor([0] + [sample['bbox'].size(0) for sample in batch])
    batch_index = torch.cumsum(batch_index, dim=0)
    batch_anno = (batch_bbox, batch_cls, batch_index)

    if 'mask' in batch[0]:
        batch_mask = torch.cat([sample['mask'] for sample in batch], dim=0)
        batch_anno += (batch_mask,)

    if 'info' in batch[0]:
        batch_info = [sample['info'] for sample in batch]
        return batch_image, batch_anno, batch_info
    else:
        return batch_image, batch_anno


def collate_plus(batch, size_divisor=32, pad_value=0):
    # satisfy the requirements of size divisor for the whole batch
    max_height = max([sample['image'].size(1) for sample in batch])
    max_width = max([sample['image'].size(2) for sample in batch])
    max_height = int(math.ceil(max_height / size_divisor) * size_divisor)
    max_width = int(math.ceil(max_width / size_divisor) * size_divisor)

    device = get_torch_device()
    for sample in batch:
        height, width = sample['image'].shape[-2:]
        # pad_left, pad_top = 0, 0
        pad_left, pad_top = (max_width - width) // 2, (max_height - height) // 2
        pad_right, pad_down = max_width - width - pad_left, max_height - height - pad_top
        padding = [pad_left, pad_right, pad_top, pad_down]

        sample['image'] = F.pad(sample['image'].to(device), padding, value=pad_value)
        sample['bbox'] = sample['bbox'].to(device)
        sample['bbox'][:, 0] = (sample['bbox'][:, 0] * width + pad_left) / max_width
        sample['bbox'][:, 1] = (sample['bbox'][:, 1] * height + pad_top) / max_height
        sample['bbox'][:, 2] = sample['bbox'][:, 2] * width / max_width
        sample['bbox'][:, 3] = sample['bbox'][:, 3] * height / max_height
        if 'mask' in sample:
            sample['mask'] = F.pad(sample['mask'].to(device), padding, value=0)
        if 'info' in sample:
            sample['info']['collate_pad'] = tuple(padding + [max_height, max_width])

    return collate(batch)
