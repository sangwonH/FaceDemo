"""
MobileNetV2 기반 age & gender estimator.

- Gender: 2-class classification (0=female, 1=male, IMDB-WIKI 컨벤션)
- Age:    101-class classification (0..100). 추론 시 softmax expectation으로
          연속값 회귀 -> E[age] = sum_i p_i * i  (DEX, Rothe et al.)

torchvision의 ImageNet pretrained MobileNetV2를 백본으로 사용해서
gender/age 두 개의 head만 새로 학습.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetAgeGender(nn.Module):
    def __init__(self,
                 num_age: int = 101,
                 num_gender: int = 2,
                 width_mult: float = 1.0,
                 pretrained: bool = True,
                 dropout: float = 0.2):
        """
        Args:
            num_age:    age 출력 차원 (기본 101 → 0..100)
            num_gender: gender 출력 차원
            width_mult: 1.0=기본, 0.75/0.5는 더 가벼움 (단 pretrained 불가)
            pretrained: ImageNet 가중치 로드 여부 (width_mult=1.0일 때만 가능)
            dropout:    head dropout
        """
        super().__init__()

        if width_mult == 1.0:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
        else:
            if pretrained:
                print(f"[warn] pretrained weights unavailable for width_mult={width_mult}; "
                      "training from scratch.")
            backbone = models.mobilenet_v2(width_mult=width_mult)

        self.features = backbone.features
        feat_dim = backbone.last_channel  # width_mult=1.0 → 1280

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_gender = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_gender),
        )
        self.head_age = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_age),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head_gender(x), self.head_age(x)

    @torch.no_grad()
    def predict(self, x):
        """편의 함수: (gender_class, age_float)을 반환."""
        lg, la = self.forward(x)
        gender = lg.argmax(dim=1)
        age_idx = torch.arange(la.size(1), device=la.device, dtype=torch.float32)
        age = (F.softmax(la, dim=1) * age_idx).sum(dim=1)
        return gender, age


if __name__ == "__main__":
    # sanity check
    for wm in [1.0, 0.75, 0.5]:
        m = MobileNetAgeGender(width_mult=wm, pretrained=(wm == 1.0))
        x = torch.randn(2, 3, 112, 112)
        g, a = m(x)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"width_mult={wm}: gender={tuple(g.shape)}, age={tuple(a.shape)}, "
              f"params={n_params/1e6:.2f}M")
