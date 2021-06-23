import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """A base loss built on `torch.nn.Module`

    `_get_loss` calculates loss and metric items and returns them as lists.
    `forward` aggregates loss items into weighted sum and wraps each item
    into a dict for loss and metric respectively.

    Args:
        loss_id (list): id for each loss item
        loss_sum_id (str): id for weighted sum of all loss items
        metric_id (list): id for each metric item
        weight (list, optional): weight of each loss item
    """

    def __init__(self, loss_id, loss_sum_id, metric_id, weight=None):
        super(BaseLoss, self).__init__()
        self.loss_id = loss_id
        self.loss_sum_id = loss_sum_id
        self.metric_id = metric_id if metric_id else tuple()
        self.weight = torch.tensor(weight).float() \
            if weight is not None else torch.ones(len(self.loss_id))

    def forward(self, predict, target, training=True):
        loss_items, metric_items = self._get_loss(predict, target, training)
        loss_cat = torch.cat([loss_item.view(1) for loss_item in loss_items])
        if self.weight.device != loss_cat.device:
            self.weight = self.weight.to(loss_cat.device)
        loss_cat = loss_cat * self.weight
        loss_log = {key: value.item() for key, value in zip(self.loss_id, loss_cat)}
        metric_log = {key: (value[0].item(), value[1].item()) if isinstance(value, (tuple, list)) else value.item()
                      for key, value in zip(self.metric_id, metric_items)}
        loss_sum = loss_cat.sum()
        loss_log[self.loss_sum_id] = loss_sum.item()
        return loss_sum, loss_log, metric_log

    def _get_loss(self, predict, target, training=True):
        raise NotImplementedError


class BaseMultiScaleLoss(nn.Module):
    """A base multi-scale loss built on `torch.nn.Module`

    `_construct_scales_loss` defines a list of `BaseLoss` modules.
    `forward` aggregates loss items of all scales into weighted sum
    and merges loss and metric dicts of all scales respectively.

    Args:
        loss_id (list): id for each loss item
        loss_sum_id (str): id for weighted sum of all loss items
        metric_id (list): id for each metric item
        scales_id (list): prefix for each scale items
        scales_weight (list, optional): weight of each scale loss

    Attributes:
        loss_id (list): {scale_id}_{loss_id} as a single list
        scales_loss_id (list(list)): separate loss_id into scales
        scales_loss_sum_id (list): {scale_id}_{loss_sum_id} as a single list
        metric_id (list): {scale_id}_{metric_id} as a single list
        scales_metric_id (list(list)): separate metric_id into scales
    """

    def __init__(self, loss_id, loss_sum_id, metric_id, scales_id, scales_weight=None):
        super(BaseMultiScaleLoss, self).__init__()
        self.loss_suffix = list(loss_id)
        self.metric_suffix = list(metric_id)
        self.scales_prefix = list(scales_id)
        self.loss_suffix.append(loss_sum_id)

        self.loss_id = []
        self.loss_sum_id = loss_sum_id
        self.metric_id = []
        self.scales_loss_id = []
        self.scales_loss_sum_id = []
        self.scales_metric_id = []
        self.scales_weight = torch.tensor(scales_weight).float() \
            if scales_weight is not None else torch.ones(self.num_scales)

        self.num_scales = len(scales_id)
        for i in range(self.num_scales):
            scale_loss_id = [scales_id[i] + '_' + suffix for suffix in loss_id]
            scale_loss_sum_id = scales_id[i] + '_' + loss_sum_id
            scale_metric_id = [scales_id[i] + '_' + suffix for suffix in metric_id]
            self.loss_id += scale_loss_id
            self.metric_id += scale_metric_id
            self.loss_id.append(scale_loss_sum_id)
            self.scales_loss_id.append(scale_loss_id)
            self.scales_loss_sum_id.append(scale_loss_sum_id)
            self.scales_metric_id.append(scale_metric_id)

        self.cross_scale_loss_id = ['cross_scale_' + suffix for suffix in self.loss_suffix]
        self.loss_id += self.cross_scale_loss_id
        self.cross_scale_metric_id = ['cross_scale_' + suffix for suffix in self.metric_suffix]
        self.metric_id += self.cross_scale_metric_id

        loss = self._construct_scales_loss()
        assert len(loss) == self.num_scales
        for i, scale_loss in enumerate(loss):
            setattr(self, 'scale_loss_%d' % i, scale_loss)

    def _construct_scales_loss(self):
        raise NotImplementedError

    def forward(self, predict, target, training=True):
        loss_list = []
        loss_log, metric_log = {}, {}
        for i in range(self.num_scales):
            scale_loss, scale_loss_log, scale_metric_log = \
                getattr(self, 'scale_loss_%d' % i)(predict[i], target, training)
            loss_list.append(scale_loss)
            loss_log.update(scale_loss_log)
            metric_log.update(scale_metric_log)
        loss_cat = torch.cat([loss_item.view(1) for loss_item in loss_list])
        if self.scales_weight.device != loss_cat.device:
            self.scales_weight = self.scales_weight.to(loss_cat.device)
        loss_sum = (loss_cat * self.scales_weight).sum()
        loss_log[self.loss_sum_id] = loss_sum.item()

        cross_scale_loss = []
        for i in range(self.num_scales):
            scale_loss = [loss_log[loss_id] for loss_id in self.scales_loss_id[i]]
            scale_loss.append(loss_log[self.scales_loss_sum_id[i]])
            cross_scale_loss.append(scale_loss)
        cross_scale_loss = torch.tensor(cross_scale_loss, device=loss_cat.device)
        cross_scale_loss = (cross_scale_loss * self.scales_weight.unsqueeze(-1)).sum(dim=0)
        for key, value in zip(self.cross_scale_loss_id, cross_scale_loss):
            loss_log[key] = value.item()

        if metric_log:
            cross_scale_metric = []
            for i in range(self.num_scales):
                scale_metric = [metric_log[metric_id] for metric_id in self.scales_metric_id[i]]
                cross_scale_metric.append(scale_metric)
            cross_scale_metric = torch.tensor(cross_scale_metric).sum(dim=0)
            for key, value in zip(self.cross_scale_metric_id, cross_scale_metric):
                metric_log[key] = (value[0].item(), value[1].item()) \
                    if value.numel() > 1 else value.item()

        return loss_sum, loss_log, metric_log
