"""
AFAD-Full / AFAD-Lite 데이터셋 로더.

디렉토리 컨벤션:
  <root>/<age>/<gender_code>/*.jpg
    age         : 디렉토리명 = 나이 (보통 15..72)
    gender_code : 111=male(=1), 112=female(=0)

프로젝트 공통 컨벤션 (IMDB-WIKI / core.types.AttributeResult) 에 맞춰
gender 는 0=female, 1=male 로 변환해서 반환한다.
"""

import glob
import os

from PIL import Image
from torch.utils.data import Dataset


def load_afad_items(root: str):
    """
    AFAD-Full/AFAD-Lite 폴더를 스캔해서 (path, age, gender) 튜플 리스트 반환.
    폴더 컨벤션: <root>/<age>/<gender_code>/*.jpg
      gender_code 111 = male (=1),  112 = female (=0)
    """
    items = []
    for age_dir in sorted(os.listdir(root)):
        age_path = os.path.join(root, age_dir)
        if not (age_dir.isdigit() and os.path.isdir(age_path)):
            continue
        age = int(age_dir)
        for g_dir, g_label in [("111", 1), ("112", 0)]:
            gd = os.path.join(age_path, g_dir)
            if not os.path.isdir(gd):
                continue
            for p in glob.glob(os.path.join(gd, "*.jpg")):
                items.append((p, age, g_label))
    return items


class AFADDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, age, gender = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            img = self.transform(img)
        return img, gender, age
