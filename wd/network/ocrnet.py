"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch

from torch import nn
from super_gradients.training import utils as sg_utils
from ezdl.models import ComposedOutput

from wd.network.mynn import initialize_weights, Upsample, scale_as
from wd.network.mynn import ResizeX
from wd.network.utils import get_trunk
from wd.network.utils import BNReLU, get_aspp
from wd.network.utils import make_attn_head
from wd.network.ocr_utils import SpatialGather_Module, SpatialOCR_Module
from wd.network.config import cfg
from wd.utils import fmt_scale, load_checkpoint_module_fix, load_weight_from_clearml



CHANNEL_PRETRAIN = {'R': 0, 'G': 1, 'B': 2}


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, high_level_ch, num_classes):
        super(OCR_block, self).__init__()

        ocr_mid_channels = cfg.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = cfg.MODEL.OCR.KEY_CHANNELS

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_mid_channels = ocr_mid_channels
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        if cfg.OPTIONS.INIT_DECODER:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


class OCRNet(nn.Module):
    """
    OCR net
    """
    def __init__(self, in_channels, num_classes, trunk='hrnetv2', aux_output=False, ocr_output=False, lres_output=False, **kwargs):
        super(OCRNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ocr_output = ocr_output
        self.lres_output = lres_output
        self.backbone, _, _, high_level_ch = get_trunk(trunk, trunk_params={'in_channels': in_channels})
        self.ocr = OCR_block(high_level_ch, num_classes)
        self.aux_output = aux_output

    def forward(self, x):

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_feats = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        hi_res_cls_out = scale_as(cls_out, x)
        aux_dict = {}
        if self.aux_output:
            aux_dict['aux'] = aux_out
        if self.ocr_output:
            aux_dict['ocr'] = ocr_feats
        if self.lres_output:
            aux_dict['lres'] = cls_out
        return ComposedOutput(hi_res_cls_out, aux_dict) if aux_dict else hi_res_cls_out
        
    def _init_from_rgb(self, weights, channel_to_load):
        weights['module.backbone.conv1.weight'] = \
                weights['module.backbone.conv1.weight'][:, channel_to_load]

        # module fix
        weights = {k.lstrip('module.'): v for k, v in weights.items()}
        # name fix
        name_tofix = [k for k in weights if k.startswith('cr')]
        for k in name_tofix:
            weights[f'o{k}'] = weights.pop(k)
        # attn fix
        weights = {k: v for k, v in weights.items() if not k.startswith('scale_attn')}
        # classes fix
        pop_keys = ["ocr.cls_head.weight", "ocr.cls_head.bias", "ocr.aux_head.2.weight", "ocr.aux_head.2.bias"]
        for k in pop_keys:
            weights.pop(k)
        res = self.load_state_dict(weights, strict=False)
        if set(res.missing_keys) != set(pop_keys):
            raise RuntimeError(f"Missing keys not expected: {res.missing_keys}")
        if res.unexpected_keys:
            raise RuntimeError(f"Unexpected keys: {res.unexpected_keys}")
    
    def _init_complete(self, weights):
        weights = load_checkpoint_module_fix(weights)
        self.load_state_dict(weights)
    
    def init_pretrained_weights(self, weights, channel_to_load=None):
        if channel_to_load is None:
            channel_to_load = slice(self.in_channels)
        elif isinstance(channel_to_load, str):
            if channel_to_load == 'complete':
                self._init_complete(weights)
                return
            extra_channels = self.in_channels - 3 # R, G, B
            channel_to_load = [0, 1, 2] + [CHANNEL_PRETRAIN[channel_to_load] for _ in range(extra_channels)
            ]
        else:
            channel_to_load = [CHANNEL_PRETRAIN[x] for x in channel_to_load]
        self._init_from_rgb(weights, channel_to_load)


class OCRNetASPP(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(OCRNetASPP, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8)
        self.ocr = OCR_block(aspp_out_ch)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        aspp = self.aspp(high_level_features)
        cls_out, aux_out, _ = self.ocr(aspp)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            loss = cfg.LOSS.OCR_ALPHA * self.criterion(aux_out, gts) + \
                self.criterion(cls_out, gts)
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict


class MscaleOCR(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)
        self.scale_attn = make_attn_head(
            in_ch=cfg.MODEL.OCR.MID_CHANNELS, out_ch=1)

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        return {'cls_out': cls_out,
                'aux_out': aux_out,
                'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = cfg.LOSS.OCR_ALPHA * self.criterion(aux, gts) + \
                self.criterion(pred, gts)
            return loss
        else:
            output_dict['pred'] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs['gts']
            do_rmi = cfg.LOSS.OCR_AUX_RMI
            aux_loss = self.criterion(joint_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = cfg.LOSS.OCR_ALPHA * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            if cfg.LOSS.SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_lo
                loss += cfg.LOSS.SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
            }
            return output_dict

    def forward(self, inputs):
        
        if cfg.MODEL.N_SCALES and not self.training:
            return self.nscale_forward(inputs, cfg.MODEL.N_SCALES)

        return self.two_scale_forward(inputs)


def HRNet(arch_params):
    pretrained = arch_params.pop('pretrained', False)
    net = OCRNet(**arch_params, trunk='hrnetv2')
    if pretrained:
        if pretrained == True:
            chk = torch.load('checkpoints/best_checkpoint_86.76_PSA_s.pth')
            chk = chk['state_dict']
            channel_to_load = arch_params.get('side_pretrained', None)
            net.init_pretrained_weights(chk, channel_to_load)
        elif isinstance(pretrained, str):
            chk = load_weight_from_clearml(pretrained)
            chk = chk['net']
            net.init_pretrained_weights(chk, "complete")
    return net


class HRNetWeeder(nn.Module):
    def __init__(self, arch_params) -> None:
        super().__init__()
        from wd.network.weeder import WeedLayer # So cc_torch is not necessary
        arch_params['ocr_output'] = True
        self.pretrained = arch_params.get('pretrained') or False
        self.aux_output = arch_params.get('aux_output') if arch_params.get('aux_output') is not None else True
        self.branch_output = arch_params.get('branch_output') if arch_params.get('branch_output') is not None else True
        arch_params['lres_output'] = True
        self.hrnet = HRNet(arch_params)
        in_channels = self.hrnet.ocr.ocr_mid_channels
        embed_channels = in_channels // (arch_params.get('embed_div') or 1)
        num_classes = arch_params['num_classes']
        patch_dim = arch_params.get('patch_dim') or 16
        emb_patch_div = arch_params.get('emb_patch_div') or 1
        num_heads = arch_params.get('num_heads') or 4
        
        self.weeder = WeedLayer(in_channels, embed_channels, patch_dim, emb_patch_div, num_heads, num_classes)

    def forward(self, x):
        probs, other = self.hrnet(x)
        adjusted_probs = self.weeder(other['ocr'], other['lres'])
        adjusted_probs_hr = scale_as(adjusted_probs, x)
        aux_output = {}
        if self.aux_output:
            aux_output['aux_out'] = other['aux_out']
        if self.branch_output:  
            aux_output['branch_out'] = probs
        return ComposedOutput(adjusted_probs_hr, aux_output) if aux_output else adjusted_probs_hr
        
    def initialize_param_groups(self, lr: float, training_params) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """

        def f(x):
            return x[0].startswith('weeder')

        freeze_pretrained = sg_utils.get_param(training_params, 'freeze_pretrained', False)
        if self.pretrained and freeze_pretrained:
            return [{'named_params': list(filter(f, list(self.named_parameters())))}]
        return [{'named_params': self.named_parameters()}]

def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)
