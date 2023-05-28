# Copyright (c) 2022 torch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ezdl.models import ComposedOutput

from wd.models import utils
from wd.models.layers import (DropPath, Identity)
from wd.models.param_init import (constant_init, kaiming_normal_init,
                                         trunc_normal_init)


class RTFormer(nn.Module):
    """
    The RTFormer implementation based on torch.

    The original article refers to "Wang, Jian, Chenhui Gou, Qiman Wu, Haocheng Feng, 
    Junyu Han, Errui Ding, and Jingdong Wang. RTFormer: Efficient Design for Real-Time
    Semantic Segmentation with Transformer. arXiv preprint arXiv:2210.07124 (2022)."

    Args:
        num_classes (int): The unique number of target classes.
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        num_heads (int, optional): The num of heads in EABlock. Default: 8
        head_channels (int, optional): The channels of head in EABlock. Default: 128
        drop_rate (float, optional): The drop rate in EABlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_aux_head (bool, optional): Whether use auxiliary head. Default: True
        use_injection (list[boo], optional): Whether use injection in layer 4 and 5.
            Default: [True, True]
        lr_mult (float, optional): The multiplier of lr for DAPPM and head module. Default: 10
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
        in_channels (int, optional): The channels of input image. Default: 3
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_head=True,
                 use_injection=[True, True],
                 lr_mult=10.,
                 cross_size=12,
                 input_channels=3,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels
        self.lr_mult = lr_mult
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(inplace=False), )

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2,
                                        1)
        self.compression3 = nn.Sequential(
            bn2d(base_chs * 4),
            nn.ReLU(inplace=False),
            conv2d(
                base_chs * 4, base_chs * 2, kernel_size=1), )
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size)
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size)

        self.spp = DAPPM(
            base_chs * 8, spp_channels, base_chs * 2, lr_mult=lr_mult)
        self.seghead = SegHead(
            base_chs * 4, int(head_channels * 2), num_classes, lr_mult=lr_mult)
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.seghead_extra = SegHead(
                base_chs * 2, head_channels, num_classes, lr_mult=lr_mult)

        self.pretrained = pretrained
        self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.layer3_.apply(self._init_weights_kaiming)
        self.compression3.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)
        self.seghead.apply(self._init_weights_kaiming)
        if self.use_aux_head:
            self.seghead_extra.apply(self._init_weights_kaiming)

        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride),
                bn2d(out_channels))

        layers = [block(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(F.relu(x1))  # 2c, 1/8
        x3 = self.layer3(F.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(
            self.compression3(x3), size=x2.shape[2:], mode='bilinear')
        x3_ = self.layer3_(F.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4(
            [F.relu(x3_), F.relu(x3)])  # 2c, 1/8; 8c, 1/16
        x5_, x5 = self.layer5(
            [F.relu(x4_), F.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(
            x6, size=x5_.shape[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(torch.concat([x5_, x6], dim=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.use_aux_head:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list
        ]
        if self.use_aux_head:
            main_out, extra_out = logit_list
            return ComposedOutput(main_out, {'aux': extra_out})
        return logit_list[0]

    # def initialize_param_groups(self, lr, training_params):
    #     def filter_module(param, module_name):
    #         return param[0].split('.')[0] == module_name
    #     spp_params = [{'named_params': list(filter(lambda x: filter_module(x, "spp"), list(self.named_parameters()))), 'lr': lr*self.lr_mult}]
    #     seghead_params = [{'named_params': list(filter(lambda x: filter_module(x, "seghead"), list(self.named_parameters()))), 'lr': lr*self.lr_mult}]
    #     seghead_extra_params = [{'named_params': list(filter(lambda x: filter_module(x, "seghead_extra"), list(self.named_parameters()))), 'lr': lr*self.lr_mult}]
    #     def filter_remaining(x):
    #         return not filter_module(x, "spp") and not filter_module(x, "seghead") and not filter_module(x, "seghead_extra")
    #     remaining = [{'named_params': list(filter(filter_remaining, list(self.named_parameters())))}]
    #     return [*spp_params, *seghead_params, *seghead_extra_params, *remaining]


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           bias_attr=False,
           **kwargs):
    assert bias_attr in [True, False], "bias_attr should be True or False"
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        **kwargs)


def bn2d(in_channels, bn_mom=0.1, **kwargs):
    assert 'bias_attr' not in kwargs, "bias_attr must not in kwargs"
    return nn.BatchNorm2d(
        in_channels,
        momentum=bn_mom,
        **kwargs)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else F.relu(out)


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = bn2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):
    """
    The ExternalAttention implementation based on torch.
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert (
            out_channels % num_heads == 0
        ), f"out_channels ({out_channels}) should be be a multiple of num_heads ({num_heads})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = bn2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = nn.Parameter(
                data=nn.init.trunc_normal_(torch.zeros(inter_channels, in_channels, 1, 1)))
            self.v = nn.Parameter(
                data=nn.init.trunc_normal_(torch.zeros(out_channels, inter_channels, 1, 1)))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)

    def _act_sn(self, x):
        # x = x.reshape([-1, self.inter_channels, 0, 0]) * (self.inter_channels
        #                                                   **-0.5)
        b2 = x.shape[1] // self.inter_channels
        x = rearrange(x, 'b (b2 ch) h w -> (b b2) ch h w', b2=b2, ch=self.inter_channels) * (self.inter_channels ** -0.5)
        x = F.softmax(x, dim=1)
        # x = x.reshape([1, -1, 0, 0])
        x = rearrange(x, '(b b2) ch h w -> b (b2 ch) h w', b2=b2, ch=self.inter_channels)
        return x

    def _act_dn(self, x):
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        ch_per_head = self.inter_channels // self.num_heads
        x = rearrange(x, 'b (heads ch) h w -> b heads ch (h w)', h=h, w=w, heads=self.num_heads, ch=ch_per_head)
        # x = x.reshape(
            # [0, self.num_heads, self.inter_channels // self.num_heads, -1])
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        # x = x.reshape([0, self.inter_channels, h, w])
        x = rearrange(x, 'b heads ch (h w) -> b (heads ch) h w', h=h, w=w, heads=self.num_heads, ch=ch_per_head)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor. 
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(
                x,
                self.k,
                bias=None,
                stride=1 if self.same_in_out_chs else 2,
                padding=0,
            )
            x = self._act_dn(x)  # n,c_inter,h,w
            x = F.conv2d(
                x, self.v, bias=None, stride=1,
                padding=0)  # n,c_inter,h,w -> n,c_out,h,w
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                    "cross_k and cross_v should no be None when use_cross_kv"
            B = x.shape[0]
            assert (
                B > 0
            ), f"The first dim of x ({B}) should be greater than 0, please set input_shape for export.py"
            # x = x.reshape([1, -1, 0, 0])  # n,c_in,h,w -> 1,n*c_in,h,w
            x = rearrange(x, 'b c h w -> 1 (b c) h w')
            x = F.conv2d(
                x, cross_k, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
            x = self._act_sn(x)
            x = F.conv2d(
                x, cross_v, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
            # x = x.reshape([-1, self.in_channels, 0,
            #                0])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)
            x = rearrange(x, '1 (b c) h w -> b c h w', b=B, c=self.in_channels)
        return x


class EABlock(nn.Module):
    """
    The EABlock implementation based on torch.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_injection (bool, optional): Whether inject the high feature into low feature. Default: True
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: True
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_channels_l),
                conv2d(in_channels_l, out_channels_l, 1, 2, 0))
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else Identity()

        # compression
        self.compression = nn.Sequential(
            bn2d(out_channels_l),
            nn.ReLU(inplace=False),
            conv2d(
                out_channels_l, out_channels_h, kernel_size=1))
        self.compression.apply(self._init_weights_kaiming)

        # high resolution
        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv)
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                bn2d(out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size,
                                                  self.cross_size)),
                conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0))
            self.cross_kv.apply(self._init_weights)

        # injection
        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_channels_h),
                nn.ReLU(inplace=False),
                conv2d(
                    out_channels_h,
                    out_channels_l // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                bn2d(out_channels_l // 2),
                nn.ReLU(inplace=False),
                conv2d(
                    out_channels_l // 2,
                    out_channels_l,
                    kernel_size=3,
                    stride=2,
                    padding=1), )
            self.down.apply(self._init_weights_kaiming)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x_h, x_l = x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))  # n,out_chs_l,h,w 

        # compression
        x_h_shape = x_h.shape[2:]
        x_l_cp = self.compression(x_l)
        x_h = x_h + F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_k, cross_v = torch.split(cross_kv, self.out_channels_h, dim=1)
            cross_k = torch.permute(cross_k, [0, 2, 3, 1]).reshape(
                [-1, self.out_channels_h, 1, 1])  # n*144,out_channels_h,1,1
            cross_v = cross_v.reshape(
                [-1, self.cross_size * self.cross_size, 1,
                 1])  # n*out_channels_h,144,1,1
            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k,
                                                   cross_v))  # n,out_chs_h,h,w

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5, stride=2, padding=2, count_include_pad=False),
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9, stride=4, padding=4, count_include_pad=False),
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17, stride=8, padding=8, count_include_pad=False),
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, inter_channels, kernel_size=1))
        self.process1 = nn.Sequential(
            bn2d(
                inter_channels),
            nn.ReLU(inplace=False),
            conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process2 = nn.Sequential(
            bn2d(
                inter_channels),
            nn.ReLU(inplace=False),
            conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process3 = nn.Sequential(
            bn2d(
                inter_channels),
            nn.ReLU(inplace=False),
            conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process4 = nn.Sequential(
            bn2d(
                inter_channels),
            nn.ReLU(inplace=False),
            conv2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.compression = nn.Sequential(
            bn2d(
                inter_channels * 5),
            nn.ReLU(inplace=False),
            conv2d(
                inter_channels * 5,
                out_channels,
                kernel_size=1))
        self.shortcut = nn.Sequential(
            bn2d(
                in_channels),
            nn.ReLU(inplace=False),
            conv2d(
                in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = [self.scale0(x)]

        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=x_shape, mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=x_shape, mode='bilinear') + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=x_shape, mode='bilinear') + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x), size=x_shape, mode='bilinear') + x_list[3])))

        return self.compression(torch.concat(x_list, dim=1)) + self.shortcut(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, lr_mult):
        super().__init__()
        self.lr_mult = lr_mult
        self.bn1 = bn2d(in_channels)
        self.conv1 = conv2d(
            in_channels,
            inter_channels,
            kernel_size=3,
            padding=1)
        self.bn2 = bn2d(inter_channels)
        self.conv2 = conv2d(
            inter_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        return self.conv2(F.relu(self.bn2(x)))