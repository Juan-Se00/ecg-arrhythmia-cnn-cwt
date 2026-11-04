"""MobileNetV2 transfer-learning models (paper §III.C.2, Eq. 5-6)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetDerivation(nn.Module):
    def __init__(self, dropout: float, num_classes: int = 5, freeze_base: bool = True):
        super().__init__()
        self.base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.base(x)

    def extract_features(self, x):
        x = self.base.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class FusionModel(nn.Module):
    def __init__(self, dropout: float, num_classes: int = 5):
        super().__init__()
        self.model_mlii = MobileNetDerivation(dropout, num_classes)
        self.model_v5 = MobileNetDerivation(dropout, num_classes)
        in_features = self.model_mlii.base.classifier[1][1].in_features

        self.fusion = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_mlii, x_v5):
        f_mlii = self.model_mlii.extract_features(x_mlii)
        f_v5 = self.model_v5.extract_features(x_v5)
        return self.fusion(torch.cat([f_mlii, f_v5], dim=1))
