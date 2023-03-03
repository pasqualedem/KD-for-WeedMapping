import torch.nn as nn
from einops import rearrange
from collections import OrderedDict

settings = {
    "S0": [32],
    "S1": [32, 64, 32],
    "S2": [32, 64, 128, 32],
    "S3": [32, 64, 128, 256, 64, 32],
    "S4": [32, 64, 128, 256, 512, 64, 32],
}


class MLP(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        return x


class ColorSeg(nn.Module):
    def __init__(self, in_channels, num_classes, setting: str = "S0" ,**kwargs) -> None:
        super().__init__()
        self.setting = settings[setting]
        self.projecter = MLP(in_channels, self.setting[0])
        self.feature_extractor = nn.Sequential(OrderedDict([
            (f"block{i}", MLP(self.setting[i], self.setting[i+1])) for i in range(len(self.setting)-1)
        ]))
        self.classifier = MLP(self.setting[-1], num_classes)
        
        
    def forward(self, x):
        _ , _, h, w = x.shape
        x = rearrange(x, "b c h w -> (b h w) c")
        x = self.projecter(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = rearrange(x, "(b h w) c -> b c h w", h=h, w=w)
        return x