import torch
import torch.nn as nn

from .base import get_syncbn

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


model_urls = {
    "resnet18": "/path/to/resnet18.pth",
    "resnet34": "/home/user/inchul2/U2PL/resnet34.pth",
    "resnet50": "/home/user/inchul2/U2PL/resnet50-ebb6acbb.pth",
    "resnet101": "/home/user/inchul2/U2PL/resnet101_u2pl.pth",
    "resnet152": "/path/to/resnet152.pth",
}
#/home/user/inchul2/U2PL/u2pl/models/resnet.py
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }

# https://github.com/fregu856/deeplabv3/tree/master/pretrained_models/resnet
# resnet18-5c106cde.pth
# resnet34-333f7ec4.pth
# resnet50-19c8e357.pth


# from AEL
# model_urls = {
#     'resnet18': '/path/to/model_zoo/resnet18-5c106cde.pth',
#     'resnet34': '/path/to/model_zoo/resnet34-333f7ec4.pth',
#     'resnet50': '/path/to/model_zoo/resnet50-ebb6acbb.pth',   https://github.com/ZJULearning/RMI/releases
#     'resnet101': '/path/to/model_zoo/resnet101-2a57e44d.pth',
#     'resnet152': '/path/to/model_zoo/resnet152-0d43d698.pth',
# }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class mlp(nn.Module):
    def __init__(self,in_chan=2048,out_chan=11):
        super(mlp,self).__init__()
        self.fc1_3 = nn.Conv2d(in_chan,out_chan,1,1,0,bias=False)
       
        # self.abf1 = ABF(out_chan,out_chan)
        # self.abf2 = ABF(out_chan,out_chan)
        # self.abf3 = ABF(out_chan,out_chan)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # torch.nn.init.xavier_normal_(self.fc1_1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc1_3.weight)
    def forward(self,x):

        # x = self.fc1_1(x)
        cam = self.fc1_3(x)
        B,C = cam.size()[:2]

        # masks_ = F.softmax(cam,dim=1)
        # features = cam.view(B, C, -1)
        # masks_ = masks_.view(B, C, -1)

        # outs = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # gap_2 = F.interpolate(F.adaptive_avg_pool2d(cam,(2,2)),size=cam.size()[2:],mode='bilinear')
        # gap_4 = F.interpolate(F.adaptive_avg_pool2d(cam,(4,4)),size=cam.size()[2:],mode='bilinear')
        # gap_8 = F.interpolate(F.adaptive_avg_pool2d(cam,(8,8)),size=cam.size()[2:],mode='bilinear')
        # gap_16 = F.interpolate(F.adaptive_avg_pool2d(cam,(16,16)),size=cam.size()[2:],mode='bilinear')
        
        # att1 = self.abf1(gap_2,gap_4)
        # att2 = self.abf2(att1,gap_8)
        # gpp = self.abf3(att2,gap_16)
        # gpp = (gap_2+gap_4+gap_8+gap_16)/3

        
        outs = self.avgpool(cam).squeeze(3).squeeze(2)
        # outs = self.avgpool(F.relu(cam)*F.relu(gpp)).squeeze(3).squeeze(2)
        # outs -= self.avgpool(F.relu(-cam)*F.relu(-gpp)).squeeze(3).squeeze(2)
        
        return cam, outs

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False],
        sync_bn=False,
        multi_grid=False,
        fpn=False,
    ):
        super(ResNet, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128                                                                 ######
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.fpn = fpn                                                                      ######
        self.conv1 = nn.Sequential(                                                         ######  1layer -> 3layer
            conv3x3(3, 64, stride=2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, self.inplanes),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True                              ###### ceil mode added
        )  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            multi_grid=multi_grid,                                                         ###### multigrid added
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_outplanes(self):
        return self.inplanes

    def get_auxplanes(self):
        return self.inplanes // 2

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, multi_grid=False
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        grids = [1] * blocks
        if multi_grid:
            grids = [2, 2, 4]

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation * grids[0],
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation * grids[i],
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        if self.fpn:
            return [x1, x2, x3, x4]
        else:
            return [x3, x4]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_url = model_urls["resnet18"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model

def resnet8(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 2, 2], **kwargs)
    if pretrained:
        model_url = model_urls["resnet18"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet34"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model_url = model_urls["resnet50"]
    #     state_dict = torch.load(model_url)

    #     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    #     print(
    #         f"[Info] Load ImageNet pretrain from '{model_url}'",
    #         "\nmissing_keys: ",
    #         missing_keys,
    #         "\nunexpected_keys: ",
    #         unexpected_keys,
    #     )
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7,\
                    stride=2, padding=3, bias=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model_url = model_urls["resnet101"]
    #     state_dict = torch.load(model_url)

    #     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    #     print(
    #         f"[Info] Load ImageNet pretrain from '{model_url}'",
    #         "\nmissing_keys: ",
    #         missing_keys,
    #         "\nunexpected_keys: ",
    #         unexpected_keys,
    #     )
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_url = model_urls["resnet152"]
        state_dict = torch.load(model_url)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[Info] Load ImageNet pretrain from '{model_url}'",
            "\nmissing_keys: ",
            missing_keys,
            "\nunexpected_keys: ",
            unexpected_keys,
        )
    return model
