import torch
import torch.distributed as dist


__all__ = [
    "get_device_rank", "get_torch_device", "get_world_size",
    "reduce_sum", "reduce_mean"
]


def get_device_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_torch_device():
    return torch.device('cuda', get_device_rank())


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor
