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

