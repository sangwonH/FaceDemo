"""RetinaFace 학습 스택 (A모듈).

RetinaFace 원본(`RetinaFace/Pytorch_Retinaface`)과 동일한 학습 구성:
  - multibox_loss   : bbox/cls/landmark 결합 손실 (hard-negative mining)
  - data_augment    : 학습 전처리 / 데이터 증강 (preproc)
  - prepared_dataset: data/prepared 통합 데이터셋 로더 (canonical 포맷)
  - combined        : WIDER+WFLW 혼합 배치 샘플러
  - train           : 학습 진입점 (`python -m modules.detection.training.train`)

모델/설정/앵커/박스 유틸 등 추론과 공유하는 코드는 상위 패키지
(`modules.detection`: network / config / prior_box / box_utils)에 있다.
"""
