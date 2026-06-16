"""`data/prepared` 통합 데이터셋 로더 (canonical 15-value 포맷).

`utils/prepare_data.py` 가 WIDER FACE 와 WFLW 를 RetinaFace 내부 포맷으로 통일해
`data/prepared/` 에 저장한 산출물을 학습에서 직접 읽는다.

Canonical 텍스트 포맷 (이미지 단위 그룹):

    # <image_path>                                              (이미지 1장)
    x1 y1 x2 y2 l0x l0y l1x l1y l2x l2y l3x l3y l4x l4y label   (얼굴 1개, 15값)
    ...

  - bbox     : xyxy (절대 픽셀)  ← raw WIDER label.txt(xywh) 와 다름
  - landmark : left_eye, right_eye, nose_tip, mouth_left, mouth_right
  - label    : 1 = 랜드마크 유효 / -1 = 랜드마크 없음(bbox-only)

`wider_canonical.txt` 와 `wflw_canonical.txt` 는 포맷이 동일하므로 같은 로더로
읽되, 이미지 루트(`image_root`)만 source 별로 다르게 지정한다.
"""

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


class CanonicalFaceDataset(data.Dataset):
    """canonical 텍스트 1개 + 이미지 루트 1개를 읽는 검출 데이터셋.

    Args:
        canonical_txt : `# path` 헤더로 그룹핑된 15-value annotation 파일.
        image_root    : `image_path` 의 기준 디렉토리.
        preproc       : (image, target) → (image, target) 증강/전처리 콜러블.
        skip_missing  : 실제로 존재하지 않는 이미지 블록은 건너뛴다.
    """

    def __init__(self, canonical_txt, image_root, preproc=None, skip_missing=True):
        self.preproc = preproc
        self.image_root = image_root

        self.imgs_path = []
        self.words = []  # 이미지별 face 리스트(np.ndarray (N,15))
        missing = 0

        cur_path = None
        cur_faces = []

        def flush():
            nonlocal missing
            if cur_path is None:
                return
            if not cur_faces:
                return  # 얼굴 없는 이미지는 학습에서 제외
            full = os.path.join(self.image_root, cur_path)
            if skip_missing and not os.path.exists(full):
                missing += 1
                return
            self.imgs_path.append(full)
            self.words.append(np.asarray(cur_faces, dtype=np.float64))

        with open(canonical_txt) as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("#"):
                    flush()
                    cur_path = line[1:].strip()
                    cur_faces = []
                else:
                    cur_faces.append([float(x) for x in line.split()])
        flush()

        if not self.imgs_path:
            raise RuntimeError(
                f"CanonicalFaceDataset: no usable images "
                f"(txt='{canonical_txt}', root='{image_root}')"
            )
        print(
            "[CanonicalFaceDataset] {} images loaded from {}{}".format(
                len(self.imgs_path),
                canonical_txt,
                f" ({missing} image files missing, skipped)" if missing else "",
            )
        )

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        # canonical 은 이미 [x1,y1,x2,y2, 5*(x,y), label] (15값) 이므로 그대로 사용.
        target = self.words[index].copy()
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target


def detection_collate(batch):
    """이미지마다 face 개수가 다른 배치를 묶는 collate 함수.

    Return:
        (images stacked on dim0, list of per-image annotation tensors)
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
