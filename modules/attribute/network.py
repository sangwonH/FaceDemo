"""
C 모듈 backbone — MobileNetV2 + (gender 2-class, age 101-class) 두 head.

본 구조는 workspace_attribute/model.py 의 MobileNetAgeGender 와 동일하다.
state_dict 키가 일치해야 학습된 가중치를 그대로 로드할 수 있으므로
attribute 이름 (features / pool / head_gender / head_age) 을 변경하지 말 것.

추론 시 age 는 softmax expectation 으로 연속값을 얻는다 (DEX, Rothe et al.):
  E[age] = Σ softmax(age_logits)_i · i,   i = 0..100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class MobileNetV2GenderAge(nn.Module):
    def __init__(self, num_age: int = 101, num_gender: int = 2, dropout: float = 0.2):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        self.features = backbone.features
        feat_dim = backbone.last_channel  # 1280

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_gender = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_gender),
        )
        self.head_age = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_age),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 112, 112)
        returns:
            gender_logits: (B, 2)
            age_logits:    (B, 101)
        """
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        return self.head_gender(feat), self.head_age(feat)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """편의 함수: (gender_class[B], age_float[B]) 반환."""
        lg, la = self.forward(x)
        gender = lg.argmax(dim=1)
        age_idx = torch.arange(la.size(1), device=la.device, dtype=torch.float32)
        age = (F.softmax(la, dim=1) * age_idx).sum(dim=1)
        return gender, age
