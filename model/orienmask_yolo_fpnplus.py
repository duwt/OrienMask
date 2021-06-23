import torch
import torch.nn as nn

from .base import BaseModel, NearestUpsample, conv_bn_leaky
from .backbone import DarkNet53


class OrienMaskYOLOFPNPlus(BaseModel):
    def __init__(self, num_anchors, num_classes, pretrained=None,
                 freeze_backbone=False, backbone_batchnorm_eval=False):
        super(OrienMaskYOLOFPNPlus, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = DarkNet53(pretrained, freeze_backbone, backbone_batchnorm_eval)

        self.neck32 = self._build_neck(1024, 512)
        self.neck16 = self._build_neck(768, 256)
        self.neck8 = self._build_neck(384, 128)
        self.neck4 = self._build_neck(256, 128)

        self.route32 = self._build_route(512, 256, 2)
        self.route16 = self._build_route(256, 128, 2)

        bbox_dim = num_anchors * (5 + num_classes)
        self.bbox_head8 = self._build_bbox_head(128, bbox_dim)
        self.bbox_head16 = self._build_bbox_head(256, bbox_dim)
        self.bbox_head32 = self._build_bbox_head(512, bbox_dim)

        orien_dim = num_anchors * 6
        self.skip32 = self._build_route(512, 64, 8)
        self.skip16 = self._build_route(256, 64, 4)
        self.skip8 = self._build_route(128, 64, 2)
        self.skip4 = conv_bn_leaky(128, 64, 1)
        self.orien_head = self._build_orien_head(128, orien_dim)

        self._init_weights()

    @classmethod
    def _build_neck(cls, in_channels, out_channels):
        return nn.Sequential(
            conv_bn_leaky(in_channels, out_channels, 1),
            conv_bn_leaky(out_channels, out_channels * 2, 3, padding=1),
            conv_bn_leaky(out_channels * 2, out_channels, 1),
            conv_bn_leaky(out_channels, out_channels * 2, 3, padding=1),
            conv_bn_leaky(out_channels * 2, out_channels, 1)
        )

    @classmethod
    def _build_route(cls, in_channels, out_channels, upsample=2):
        return nn.Sequential(
            conv_bn_leaky(in_channels, out_channels, 1),
            NearestUpsample(scale_factor=upsample)
        )

    @classmethod
    def _build_bbox_head(cls, in_channels, out_channels):
        return nn.Sequential(
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )

    @classmethod
    def _build_orien_head(cls, in_channels, out_channels):
        return nn.Sequential(
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            conv_bn_leaky(in_channels * 2, in_channels, 1),
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            conv_bn_leaky(in_channels * 2, in_channels, 1),
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )

    def forward(self, x):
        x32, x16, x8, x4 = self.backbone(x)

        neck32 = self.neck32(x32)
        neck16 = self.neck16(torch.cat([self.route32(neck32), x16], dim=1))
        neck8 = self.neck8(torch.cat([self.route16(neck16), x8], dim=1))

        bbox32 = self.bbox_head32(neck32)
        bbox16 = self.bbox_head16(neck16)
        bbox8 = self.bbox_head8(neck8)

        oriens = self.neck4(torch.cat([self.skip32(neck32), self.skip16(neck16),
                                       self.skip8(neck8), self.skip4(x4)], dim=1))
        oriens = self.orien_head(oriens)
        orien32, orien16, orien8 = torch.split(oriens, self.num_anchors * 2, dim=1)

        return (bbox32, orien32), (bbox16, orien16), (bbox8, orien8)
