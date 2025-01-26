import tfedplat as fp
import torch
from torch import Tensor
import torch.nn as nn
from torch.functional import F
from typing import Type, Any, Callable, Union, List, Optional
from functools import partial


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    Adapted from timm: https://github.com/rwightman/pytorch-image-models/blob/4ea593196414684d2074cbb81d762f3847738484/timm/models/layers/std_conv.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, gain=True, gamma=1.0, eps=1e-5, use_layernorm=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(
            self.out_channels, 1, 1, 1)) if gain else None
        
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                     F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)

ignore_inplace = ['gelu', 'silu', 'softplus', ]

activation_fn = {
    'identity': lambda x, *args, **kwargs: nn.Identity(*args, **kwargs)(x) * _nonlin_gamma['identity'],
    'celu': lambda x, *args, **kwargs: nn.CELU(*args, **kwargs)(x) * _nonlin_gamma['celu'],
    'elu': lambda x, *args, **kwargs: nn.ELU(*args, **kwargs)(x) * _nonlin_gamma['elu'],
    'gelu': lambda x, *args, **kwargs: nn.GELU(*args, **kwargs)(x) * _nonlin_gamma['gelu'],
    'leaky_relu': lambda x, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)(x) * _nonlin_gamma['leaky_relu'],
    'log_sigmoid': lambda x, *args, **kwargs: nn.LogSigmoid(*args, **kwargs)(x) * _nonlin_gamma['log_sigmoid'],
    'log_softmax': lambda x, *args, **kwargs: nn.LogSoftmax(*args, **kwargs)(x) * _nonlin_gamma['log_softmax'],
    'relu': lambda x, *args, **kwargs: nn.ReLU(*args, **kwargs)(x) * _nonlin_gamma['relu'],
    'relu6': lambda x, *args, **kwargs: nn.ReLU6(*args, **kwargs)(x) * _nonlin_gamma['relu6'],
    'selu': lambda x, *args, **kwargs: nn.SELU(*args, **kwargs)(x) * _nonlin_gamma['selu'],
    'sigmoid': lambda x, *args, **kwargs: nn.Sigmoid(*args, **kwargs)(x) * _nonlin_gamma['sigmoid'],
    'silu': lambda x, *args, **kwargs: nn.SiLU(*args, **kwargs)(x) * _nonlin_gamma['silu'],
    'softplus': lambda x, *args, **kwargs: nn.Softplus(*args, **kwargs)(x) * _nonlin_gamma['softplus'],
    'tanh': lambda x, *args, **kwargs: nn.Tanh(*args, **kwargs)(x) * _nonlin_gamma['tanh'],
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return base_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """1x1 convolution"""
    return base_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride, base_conv=base_conv)
        self.activation = activation

        if activation not in ignore_inplace:
            self.act = partial(activation_fn[activation], inplace=True)
        else:
            self.act = partial(activation_fn[activation])
        self.conv2 = conv3x3(planes, planes, base_conv=base_conv)
        self.downsample = downsample
        self.stride = stride
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = activation_fn[self.activation](x=x) * self.beta

        out = self.conv1(out)
        out = self.act(x=out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out *= self.alpha
        out += identity

        return out


class Bottleneck(nn.Module):
    
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width, base_conv=base_conv)
        self.conv2 = conv3x3(width, width, stride, groups,
                             dilation, base_conv=base_conv)
        self.conv3 = conv1x1(
            width, planes * self.expansion, base_conv=base_conv)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.alpha = alpha
        self.beta = beta
        self.activation = activation
        if activation not in ignore_inplace:
            self.act = partial(activation_fn[activation], inplace=True)
        else:
            self.act = partial(activation_fn[activation])

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = activation_fn[self.activation](x) * self.beta

        out = self.conv1(out)
        out = self.act(x=out)

        out = self.conv2(out)
        out = self.act(x=out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out *= self.alpha
        out += identity

        return out

class NFResNet(fp.Module):

    def __init__(self, device, *args, **kwargs):
        super(NFResNet, self).__init__(device)
        self.model = None
        
        self.input_require_shape = [3, -1, -1]
        
        self.ignore_head = False
        self.block = None
        self.layers = None
        self.alpha = None
        self.beta = None
        self.activation = None
        self.base_conv = None

    def generate_model(self, input_data_shape, target_class_num=None, vgg_name='VGG11', *args, **kwargs):
        self.model = NFResNet_Model(self.ignore_head, target_class_num, self.block, self.layers, alpha=self.alpha, beta=self.beta, activation=self.activation, base_conv=self.base_conv).to(self.device)
        self.create_Loc_reshape_list()

    def fix_Loc_list(self):
        
        temp = self.Loc_list
        for i in range(len(temp)):
            if i % 2 == 0:
                temp[i] = torch.hstack([temp[i], temp[i + 1]])
                temp[i + 1] = None

        self.Loc_list = []
        for item in temp:
            if item is not None:
                self.Loc_list.append(item)

class NFResNet_Model(nn.Module):
    def __init__(
            self,
            ignore_head,
            target_class_num,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(NFResNet_Model, self).__init__()

        assert activation in activation_fn.keys()

        self.ignore_head = ignore_head
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            
            
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.base_width = width_per_group

        self.conv1 = base_conv(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 128, layers[0], alpha=alpha, beta=beta, activation=activation, base_conv=base_conv)
        self.layer2 = self._make_layer(block, 256 * block.expansion, layers[1], stride=2,
                                       dilate=self.replace_stride_with_dilation[0], alpha=alpha, beta=beta,
                                       activation=activation, base_conv=base_conv)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.predictor = nn.Linear(256 * block.expansion, target_class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='linear')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride, base_conv=base_conv),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, alpha=alpha, beta=beta, activation=activation, base_conv=base_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                alpha=alpha, beta=beta, activation=activation,
                                base_conv=base_conv))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if not self.ignore_head:
            x = self.predictor(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def NFResNet18(device, alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    module = NFResNet(device)
    module.name = 'resnet18'
    module.block = BasicBlock
    module.layers = [2, 2, 2, 2]
    module.alpha = alpha
    module.beta = beta
    module.activation = activation
    module.base_conv = base_conv
    return module

def NFResNet50(device, alpha: float = 0.2, beta: float = 1.0, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d, **kwargs: Any) -> NFResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    module = NFResNet(device)
    module.name = 'resnet50'
    module.block = BasicBlock
    module.layers = [3, 4, 6, 3]
    module.alpha = alpha
    module.beta = beta
    module.activation = activation
    module.base_conv = base_conv
    return module




































































































