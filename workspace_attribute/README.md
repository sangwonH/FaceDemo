# workspace_attribute

C 모듈 (Age & Gender attribute classifier) 학습 작업공간.
AFAD-Full 데이터셋으로 MobileNetV2 기반 멀티헤드(gender 2-class, age 101-class) 분류기를 학습한다.

## 디렉토리

```
workspace_attribute/
├── dataset.bash                  # AFAD-Full 다운로드/압축해제
├── tarball/                      # AFAD 데이터 (gitignore)
├── data/                         # 메타 산출물 (gitignore)
├── checkpoints/                  # 학습 weight (gitignore, .gitkeep만 추적)
├── runs/                         # tensorboard 등 (gitignore)
├── outputs/                      # inference 결과 (gitignore)
│
├── model.py                      # MobileNetAgeGender
├── dataset.py                    # AFADDataset + load_afad_items
├── transforms.py                 # train/val transform
├── train.py                      # 학습 진입점
├── evaluate.py                   # ckpt 평가 (MAE / Gender Acc)
├── inference.py                  # 단독 데모 (MTCNN + 분류)
│
├── scripts/
│   └── export_to_module.py       # best.pth → modules/attribute/weights/
│
├── requirements.txt
└── README.md
```

## 환경 셋업 (최초 1회)

```bash
# conda env (workspace 전용)
conda create -n face-attr python=3.11 -y
conda activate face-attr

# PyTorch 는 CUDA 휠을 index URL 로 직접 설치 (PyTorch conda 채널은 deprecated)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 나머지 의존성
pip install -r requirements.txt

# 동작 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python model.py
```

> 드라이버가 CUDA 13.x 라도 PyTorch cu126 휠은 하위 호환되어 정상 동작.
> 다른 CUDA 버전이 필요하면 `cu118` / `cu124` 등 index URL 만 바꿔주면 된다.

## 빠른 시작

```bash
# 0) 데이터 준비 (한 번만)
bash dataset.bash                          # → tarball/AFAD-Full/<age>/<gender>/*.jpg

# 1) 학습
python train.py --afad_root tarball/AFAD-Full --out_dir checkpoints

# 2) 평가
python evaluate.py --ckpt checkpoints/best.pth --afad_root tarball/AFAD-Full

# 3) 단독 데모
python inference.py --ckpt checkpoints/best.pth --image test.jpg --out outputs/out.jpg

# 4) 메인 파이프라인으로 가중치 배포
python scripts/export_to_module.py --ckpt checkpoints/best.pth
```

## 컨벤션

- **Gender**: `0=female, 1=male` (모델 출력. AFAD 디렉토리 `111=M / 112=F` 에서 변환됨)
- **Age**: 101-class softmax → 추론 시 `E[age] = Σ p_i · i` (DEX, Rothe et al.)
- **이미지 크기**: 112×112 (학습/추론 공통)

## 통합 시 주의

`inference.py` 는 standalone 데모용. 전체 파이프라인(A→B→C) 통합 시에는
얼굴 검출이 A 모듈(RetinaFace)에서 끝나므로, `inference.py` 의 MTCNN 부분은
사용하지 않고 `load_model / build_transform / predict_faces` 만
`modules/attribute/attribute.py` 에서 재사용한다.
