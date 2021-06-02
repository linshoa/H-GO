from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .backbone.resnet import ResNet, BasicBlock, Bottleneck

from .backbone.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbone.resnet_ibn_a import resnet50_ibn_a


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'ibn_resnet50', 'se_resnet50', 'se_resnext50']


# class ResNet(nn.Module):
#     __factory = {
#         18: torchvision.models.resnet18,
#         34: torchvision.models.resnet34,
#         50: torchvision.models.resnet50,
#         101: torchvision.models.resnet101,
#         152: torchvision.models.resnet152,
#     }
#
#     def __init__(self, depth, pretrained=True):
#         super(ResNet, self).__init__()
#
#         self.depth = depth
#         self.pretrained = pretrained
#
#         # Construct base (pretrained) resnet
#         if depth not in ResNet.__factory:
#             raise KeyError("Unsupported depth:", depth)
#         self.base = ResNet.__factory[depth](pretrained=pretrained)
#
#
#         # todo important!!!!!
#         # remove the final downsample
#
#
#         # avg
#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.avg_td = nn.AdaptiveMaxPool2d((2, 1))
#         self.norm = F.normalize
#
#         #todo
#         # Fix layers [conv1 ~ layer2]
#         # fixed_names = []
#         # for name, module in self.base._modules.items():
#         #     if name == "layer2":
#         #         break  # ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
#         #     fixed_names.append(name)
#         #     for param in module.parameters():
#         #         param.requires_grad = False
#
#         if not self.pretrained:
#             self.reset_params()
#
#     def forward(self, x, output_feature=None):
#         x_ms = []
#         x = self.base.conv1(x)
#         x = self.base.bn1(x)
#         x = self.base.relu(x)
#         x = self.base.maxpool(x)
#         x = self.base.layer1(x)
#         x = self.base.layer2(x)
#         x = self.base.layer3(x)
#         # x_ms.append(x)
#         x = self.base.layer4(x)
#         # x_ms.append(x)
#         # td = self.avg_td(x)
#         # top = td[:,:,0,:].view(x.size(0), -1)
#         # down = td[:,:,1,:].view(x.size(0), -1)
#         # x_ms.append(top)
#         # x_ms.append(down)
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x_ms.append(x)
#         x = self.norm(x)
#         x_ms.append(x)
#         # avg norm
#         return x_ms  # return multi scale feature
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant(m.weight, 1)
#                 init.constant(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant(m.bias, 0)

class Baseline(nn.Module):
    def __init__(self, model_name, num_features=0, dropout=0, num_classes=0):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.base = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
            self.base.load_param('/home/lsc/.torch/models/resnet18-5c106cde.pth')
        elif model_name == 'resnet34':
            self.base = ResNet(last_stride=1,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            self.base.load_param('/home/lsc/.torch/models/resnet34-333f7ec4.pth')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            self.base.load_param('/home/lsc/.torch/models/resnet50-19c8e357.pth')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
            self.base.load_param('/home/lsc/.torch/models/resnet101-5d3b4d8f.pth')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            self.base.load_param('/home/lsc/.torch/models/resnet152-b121ed2d.pth')
        elif model_name == 'ibn_resnet50':
            self.base = resnet50_ibn_a(1, True)
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=1)
            self.base.load_param('/home/lsc/.torch/models/se_resnet50-ce0d4300.pth')
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=1)
            self.base.load_param('/home/lsc/.torch/models/se_resnext50_32x4d-a260b3a4.pth')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.norm = F.normalize
        self.num_features = num_features
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = 2048

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def forward(self, x, feature_withbn=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.has_embedding:
            bn_x = F.relu(bn_x)
        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return Baseline(model_name='resnet50',**kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


def ibn_resnet50(**kwargs):
    return Baseline(model_name='ibn_resnet50',**kwargs)


def se_resnet50(**kwargs):
    return Baseline(model_name='se_resnet50',**kwargs)


def se_resnext50(**kwargs):
    return Baseline(model_name='se_resnext50',**kwargs)
