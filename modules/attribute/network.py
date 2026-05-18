import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetV2GenderAge(nn.Module):
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        # 마지막 classifier 제거, feature 차원 = 1280
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gender_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2),
        )
        self.age_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 224, 224)
        returns:
            gender_logits: (B, 2)
            age_pred:      (B,)   # 0~100 범위로 학습 가정
        """
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        gender_logits = self.gender_head(feat)
        age_pred = self.age_head(feat).squeeze(-1)
        return gender_logits, age_pred