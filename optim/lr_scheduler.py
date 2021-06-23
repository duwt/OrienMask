import math

from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR:
    def __init__(self, warmup_type, warmup_iter, warmup_ratio):
        assert warmup_type in ['const', 'linear', 'power']
        self.type = warmup_type
        self.iter = warmup_iter
        self.ratio = warmup_ratio

    def get_warmup_lr(self, iters, base_lr):
        if self.type == 'const':
            return base_lr * self.ratio
        elif self.type == 'linear':
            # return base_lr * self.ratio * iters / self.iter
            return base_lr * (self.ratio + (1 - self.ratio) * iters / self.iter)
        elif self.type == 'power':
            return base_lr * ((iters / self.iter) ** self.ratio)


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * math.pow(1 - self.last_epoch / self.max_iter, self.power)
                for base_lr in self.base_lrs]


class StepWarmUpLR(MultiStepLR):
    def __init__(self, warmup_type, warmup_iter, warmup_ratio,
                 optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.warmup = WarmupLR(warmup_type, warmup_iter, warmup_ratio)
        self.milestones = milestones
        self.gamma = gamma
        super(StepWarmUpLR, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup.iter:
            return super(StepWarmUpLR, self).get_lr()
        else:
            return [self.warmup.get_warmup_lr(self.last_epoch, base_lr)
                    for base_lr in self.base_lrs]
