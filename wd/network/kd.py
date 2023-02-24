import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange
from ezdl.models.kd.logits import LogitsDistillationModule


class TeacherTargetMerger(nn.Module):
    def __init__(self, num_classes, batch_norm=True):
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


class FixTeacherDistillationModule(LogitsDistillationModule):
    """
    Fix Teacher Distillation Module
    Uses a module with the targets to fix the output of the teacher
    """
    def __init__(self, arch_params, student, teacher, batch_norm=True, **kwargs):
        super().__init__(arch_params, student, teacher, **kwargs)
        self.num_classes = arch_params['num_classes']
        self.teacher_fixer = TeacherTargetMerger(self.num_classes, batch_norm=batch_norm)

    def forward(self, x, target):
        onehot_target = rearrange(F.one_hot(target, num_classes=self.num_classes), "b h w c -> b c h w")
        kd_output = super().forward(x)
        fixed_teacher = self.teacher_fixer(kd_output.teacher_output, onehot_target)
        kd_output.teacher_output = fixed_teacher
        return kd_output