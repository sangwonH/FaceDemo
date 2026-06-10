"""RetinaFace 학습/추론 하이퍼파라미터.

`cfg_mnet` (MobileNet0.25), `cfg_re50` (ResNet50) 두 설정을 RetinaFace 원본과
동일하게 유지한다. 주요 필드:
  - batch_size / epoch / decay1,decay2 : 배치 크기, 총 epoch, LR 감쇠 시점(epoch)
  - image_size : 학습 입력 정사각 크기
  - loc_weight : bbox 회귀 손실 가중치
  - ngpu / gpu_train : GPU 개수, GPU 학습 여부
"""

cfg_mnet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    # 백본(ImageNet) 사전학습 가중치 경로 (없으면 자동 스킵). RetinaFace 원본 기본값.
    # 주: 완성된 검출기 가중치(combined_*_final.pth)는 여기가 아니라 추론은
    # RetinaFaceDetector(model_path=...), 학습 이어하기는 --resume_net 으로 사용한다.
    "pretrain_path": "./weights/mobilenetV1X0.25_pretrain.tar",
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

cfg_re50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": True,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}
