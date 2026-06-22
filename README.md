# FaceDemo
2026-1 Software Development Methodologies

## Prepare checkpoints

체크포인트들을 아래와 같이 위치시킵니다.

'''
modules\attribute\weights\mobilenet_age_gender.pth
modules\detection\weights\combined_100pct_final.pth
modules\landmark\weights\pfld_100_retina_train.pth.tar
'''

## Run Demo from Python
루트 디렉토리의 CMD에서 아래를 실행합니다.

'''
python -m pipeline.webcam_demo --cam 0
'''

## Build Face Demo in window

conda 환경이 활성화된 CMD 터미널에서

'''
build_exe.bat
'''

또는 아래를 실행합니다.

'''
pyinstaller --clean --onedir --name FaceDemo ^
  --paths . ^
  --exclude-module tensorflow ^
  --exclude-module tensorboard ^
  --exclude-module torchaudio ^
  --add-data "modules\detection\weights\combined_100pct_final.pth;modules\detection\weights" ^
  --add-data "modules\landmark\weights\pfld_100_retina_train.pth.tar;modules\landmark\weights" ^
  --add-data "modules\attribute\weights\mobilenet_age_gender.pth;modules\attribute\weights" ^
  --add-data "utils\dataset_config.json;utils" ^
  pipeline\webcam_demo.py
'''