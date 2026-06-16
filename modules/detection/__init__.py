"""A모듈: RetinaFace 얼굴 검출.

폴더 구조 — 추론과 학습이 공유하는 모델/유틸은 패키지 최상위에 두고,
학습 전용 코드는 `training/` 하위 패키지로 분리한다.

  detector     : 추론 진입점 (BaseFaceDetector 구현, core 인터페이스)
  network      : RetinaFace 모델 (MobileNetV1 0.25 / ResNet50 백본)  [공유]
  config       : cfg_mnet / cfg_re50 하이퍼파라미터                  [공유]
  prior_box    : 앵커 생성                                          [공유]
  box_utils    : 매칭/인코딩/디코딩/NMS                              [공유]
  training/    : 학습 스택 (multibox_loss · data_augment ·
                 prepared_dataset · combined · train)
"""

from .config import cfg_mnet, cfg_re50
from .network import RetinaFace

__all__ = ["RetinaFace", "cfg_mnet", "cfg_re50"]
