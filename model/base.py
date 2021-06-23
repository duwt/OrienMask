import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torch.nn.modules.batchnorm import _BatchNorm

from utils.envs import get_device_rank, get_torch_device


class BaseModel(nn.Module):
    """A base model built on `torch.nn.Module`

    'summary' displays the network structure.
    `init_weight` overwrites the default initialization provided by `pytorch`
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def summary(self, input_shape, batch_size=1, device='cpu'):
        if get_device_rank() == 0:
            print('[%s] Network Summary' % self.__class__.__name__)
            torchsummary.summary(self, input_shape, batch_size, device)
            print('--------------------------------------------------------------------')

    def _init_weights(self):
        for name, module in self.named_modules():
            if 'backbone' in name:
                continue
            if isinstance(module, _BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class BaseBackbone(BaseModel):
    """A base backbone built on `BaseModel`

    `_load_pretrained_weights` loads weights to initialize models in all devices.
    `get_output_channels` returns channels of backbone at several places.
    """

    def __init__(self, pretrained=None, freeze_backbone=False, batchnorm_eval=False):
        super(BaseBackbone, self).__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.batchnorm_eval = batchnorm_eval

    def _load_pretrained_weights(self):
        if self.pretrained is not None:
            pretrained_dict = torch.load(self.pretrained, map_location=get_torch_device())
            state_dict = self.state_dict()
            model_dict = {}
            ignore_keys = []
            for k, v in pretrained_dict.items():
                if k in state_dict and v.shape == state_dict[k].shape:
                    model_dict[k] = v
                else:
                    ignore_keys.append(k)
            state_dict.update(model_dict)
            self.load_state_dict(state_dict)
            if get_device_rank() == 0:
                print('[%s] Load pretrained model %s' % (self.__class__.__name__, self.pretrained))
                if len(ignore_keys) > 0:
                    print('Ignore keys:', ignore_keys)

    def _freeze_network(self):
        if self.freeze_backbone:
            for m in self.parameters():
                m.requires_grad = False

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        if mode and self.batchnorm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    def get_output_channels(self):
        raise NotImplementedError


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)


class NearestUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(NearestUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 batchnorm=True, norm_type='BN', activation='relu'):
        super(ConvBNRelu, self).__init__()

        if bias == 'auto':
            bias = False if batchnorm else True

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)]

        if batchnorm is True:
            if norm_type == 'BN':
                layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'GN':
                layers.append(nn.GroupNorm(32, out_channels))
            else:
                raise NotImplementedError

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif activation == 'none':
            pass
        else:
            raise NotImplementedError

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, extra_levels=0,
                 bias='auto', batchnorm=False, activation='none'):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra_levels = extra_levels

        lateral_convs = []
        output_convs = []
        extra_convs = []

        for i in range(len(in_channels)):
            lateral_convs.append(
                ConvBNRelu(in_channels[i], out_channels, 1, bias=bias,
                           batchnorm=batchnorm, activation=activation)
            )
            output_convs.append(
                ConvBNRelu(out_channels, out_channels, 3, padding=1, bias=bias,
                           batchnorm=batchnorm, activation=activation)
            )

        if extra_levels > 0:
            extra_convs.append(
                ConvBNRelu(in_channels[0], out_channels, 3, stride=2, padding=1,
                           bias=bias, batchnorm=batchnorm, activation=activation)
            )
            for _ in range(extra_levels - 1):
                extra_convs.append(
                    ConvBNRelu(out_channels, out_channels, 3, stride=2, padding=1,
                               bias=bias, batchnorm=batchnorm, activation=activation)
                )

        self.lateral_convs = nn.Sequential(*lateral_convs)
        self.output_convs = nn.Sequential(*output_convs)
        self.extra_convs = nn.Sequential(*extra_convs)

        self.upsample = NearestUpsample(scale_factor=2)

    def forward(self, x):
        outputs = []
        prev_lateral = self.lateral_convs[0](x[0])
        outputs.append(self.output_convs[0](prev_lateral))
        for i in range(1, len(x)):
            lateral = self.lateral_convs[i](x[i])
            upsample = self.upsample(prev_lateral)
            prev_lateral = lateral + upsample
            outputs.append(self.output_convs[i](prev_lateral))

        prev_feature = x[0]
        for i in range(self.extra_levels):
            prev_feature = self.extra_convs[i](prev_feature)
            outputs.insert(0, prev_feature)

        return outputs


class SPP(nn.Module):
    def __init__(self, kernel_size, channels, **kwargs):
        super(SPP, self).__init__()
        self.pool = [nn.MaxPool2d(k) for k in kernel_size]
        self.conv = ConvBNRelu(channels * len(kernel_size), channels, 1, **kwargs)

    def forward(self, x):
        pyramid = [x] + [p(x) for p in self.pool]
        x = torch.cat(pyramid, dim=1)
        x = self.conv(x)
        return x


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        if hasattr(bn_module, 'SyncBatchNorm'):
            bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        else:
            bn_module = bn_module.BatchNorm2d
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class Scale(nn.Module):
    def __init__(self, init_value=1.0, learnable=True):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float), requires_grad=learnable)

    def forward(self, x):
        return x * self.scale


def conv_bn_leaky(*args, **kwargs):
    return ConvBNRelu(*args, **kwargs, activation='leaky')
