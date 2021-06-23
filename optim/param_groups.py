from typing import Any, Dict, List, Set

import torch


# refer to https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/build.py
def param_groups(model: torch.nn.Module, base_lr=1e-3, weight_decay=1e-4,
                 norm_weight_decay=0.0, bias_lr_factor=1.0, bias_weight_decay=1e-4):
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = base_lr
            weight_decay = weight_decay
            if isinstance(module, norm_module_types):
                weight_decay = norm_weight_decay
            elif key == "bias":
                lr = base_lr * bias_lr_factor
                weight_decay = bias_weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    return params
