import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import pdb
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  
        out = F.relu(out)  
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  
            self.in_planes = planes * block.expansion  
        return nn.Sequential(*layers)  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = F.avg_pool2d(out, 4)  
        out = out.view(out.size(0), -1)  
        out = self.linear(out) 
        return out



def customized_resnet18(pretrained: bool = False, class_num=10,progress: bool = True) -> ResNet:
    res18 = ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18


class tiny_ResNet(nn.Module):
    def __init__(self, block, num_blocks, class_num=10):
        super(tiny_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # 
        layers = []
        # 
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 
            self.in_planes = planes * block.expansion  # 
        return nn.Sequential(*layers)  # 

    # 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 
        out = self.layer1(out)  # 
        out = self.layer2(out)  # 
        out = self.layer3(out)  # 
        out = self.layer4(out)  # 
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)  # 
        return out

def tiny_resnet18(pretrained: bool = False, class_num=10,progress: bool = True) -> ResNet:
    res18 = tiny_ResNet(BasicBlock, [2, 2, 2, 2],class_num=class_num)


    # Change BN to GN
    res18.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
    res18.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

    res18.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
    res18.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

    res18.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
    res18.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

    res18.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[0].shortcut[1] = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
    res18.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

    assert len(dict(res18.named_parameters()).keys()) == len(
        res18.state_dict().keys()), 'More BN layers are there...'

    return res18
# def resnet18(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, num_classes=10,**kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=32, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)
#     return model


# def resnet34(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(BasicBlock, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet34'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet50(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet50'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet101(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet101'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnet152(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0, **kwargs: Any):
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#
#     model = ResNet(Bottleneck, [3, 8, 36, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnet152'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext50_32x4d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def resnext101_32x8d(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet50_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                     **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def wide_resnet101_2(pretrained: bool = False, progress: bool = True, group_norm=False, group_channels=0,
#                      **kwargs: Any) -> ResNet:
#     norm_layer = None
#     if group_norm and group_channels > 0:
#         norm_layer = nn.GroupNorm(num_groups=64, num_channels=group_channels)
#     kwargs['width_per_group'] = 64 * 2
#     model = ResNet(Bottleneck, [3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['wide_resnet101_2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)