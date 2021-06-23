import torch.nn as nn

from ..base import BaseBackbone, conv_bn_leaky, FrozenBatchNorm2d


class _DarkNetBlock(nn.Module):
    def __init__(self, channels):
        super(_DarkNetBlock, self).__init__()
        self.conv = nn.Sequential(
            conv_bn_leaky(channels * 2, channels, 1),
            conv_bn_leaky(channels, channels * 2, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)


class DarkNet53(BaseBackbone):
    def __init__(self, pretrained=None, freeze_backbone=False, batchnorm_eval=False):
        super(DarkNet53, self).__init__(pretrained, freeze_backbone, batchnorm_eval)
        self.conv1 = conv_bn_leaky(3, 32, 3, padding=1)
        self.conv2 = self._build_block(32, 1)
        self.conv3 = self._build_block(64, 2)
        self.conv4 = self._build_block(128, 8)
        self.conv5 = self._build_block(256, 8)
        self.conv6 = self._build_block(512, 4)

        self._load_pretrained_weights()
        self._freeze_network()

    def _freeze_network(self):
        if self.freeze_backbone:
            for i in range(1, 7):
                if self.freeze_backbone >= i:
                    module = getattr(self, 'conv{}'.format(i))
                    for m in module.parameters():
                        self._freeze_module(m)
                    setattr(self, 'conv{}'.format(i), FrozenBatchNorm2d.convert_frozen_batchnorm(module))

    @classmethod
    def _build_block(cls, channels, n_iter):
        layers = [conv_bn_leaky(channels, channels * 2, 3, stride=2, padding=1)]
        for i in range(n_iter):
            layers.append(_DarkNetBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x4 = self.conv3(x)
        x8 = self.conv4(x4)
        x16 = self.conv5(x8)
        x32 = self.conv6(x16)
        return x32, x16, x8, x4

    def get_output_channels(self):
        return 1024, 512, 256, 128
