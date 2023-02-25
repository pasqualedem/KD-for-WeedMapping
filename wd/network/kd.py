import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange
from ezdl.models.kd.logits import LogitsDistillationModule
from ezdl.models import KDOutput


class ConvTeacherTargetMerger(nn.Module):
    def __init__(self, num_classes, batch_norm=True, **kwargs):
        super().__init__()
        self.teacher_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        self.teacher_bn = nn.BatchNorm2d(num_classes) if batch_norm else nn.Identity()

        self.target_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        self.target_bn = nn.BatchNorm2d(num_classes) if batch_norm else nn.Identity()

    def forward(self, teacher, target):
        conv_teacher = self.teacher_conv(teacher)
        conv_teacher = self.teacher_bn(teacher)

        target = target.float()
        target = self.target_conv(target)
        target = self.target_bn(target)

        return teacher + target + conv_teacher
    
    
class AttnTeacherTargetMerger(nn.Module):
    def __init__(self, num_classes, class_space, **kwargs):
        super().__init__()
        self.heads = class_space // 2
        self.num_classes = num_classes
        
        self.teacher_conv = nn.Conv2d(num_classes, num_classes * class_space, kernel_size=1, bias=False, groups=num_classes)
        self.target_conv = nn.Conv2d(num_classes, num_classes * class_space, kernel_size=1, bias=False, groups=num_classes)
        
        self.attn = nn.MultiheadAttention(class_space, self.heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(class_space)
        self.classifier = nn.Linear(num_classes * class_space, num_classes)

    def forward(self, teacher, target):
        conv_teacher = self.teacher_conv(teacher)
        target = target.float()
        conv_target = self.target_conv(target)
        
        conv_teacher = rearrange(conv_teacher, "b (c cls) h w -> (b h w) cls c", cls=self.num_classes)
        conv_target = rearrange(conv_target, "b (c cls) h w -> (b h w) cls c", cls=self.num_classes)
        
        attn = self.attn(conv_teacher, conv_target, conv_target)[0]
        res = attn + conv_teacher
        res = self.attn_norm(res)
        res = rearrange(res, "(b h w) cls c -> (b h w) (cls c)", cls=self.num_classes, h=teacher.shape[2], w=teacher.shape[3])
        
        logits = self.classifier(res)
        seg = rearrange(logits, "(b h w) c -> b c h w", h=teacher.shape[2], w=teacher.shape[3])
        return seg
        
        

class FixTeacherDistillationModule(LogitsDistillationModule):
    """
    Fix Teacher Distillation Module
    Uses a module with the targets to fix the output of the teacher
    """
    methods = {
        'conv': ConvTeacherTargetMerger,
        'attn': AttnTeacherTargetMerger,
    }
    def __init__(self, arch_params, student, teacher, **kwargs):
        super().__init__(arch_params, student, teacher, **kwargs)
        merger = arch_params.get('merger', 'conv')
        class_space = arch_params.get('class_space', 2)
        batch_norm = arch_params.get('batch_norm', True)
        
        self.num_classes = arch_params['num_classes']
        self.teacher_fixer = self.methods[merger](self.num_classes, batch_norm=batch_norm, class_space=class_space)

    def forward(self, x, target):
        onehot_target = rearrange(F.one_hot(target, num_classes=self.num_classes), "b h w c -> b c h w")
        kd_output = super().forward(x)
        fixed_teacher = self.teacher_fixer(kd_output.teacher_output, onehot_target)
        return KDOutput(kd_output.student_output, fixed_teacher)