from typing import Any

import cv2
import numpy as np
import torch

from core.base import BaseFaceAttribute
from core.types import AttributeResult, DetectionResult
from modules.attribute.network import MobileNetV2GenderAge


class MobileNetV2Attribute(BaseFaceAttribute):
    """
    C모듈: MobileNetV2 기반 성별 / 나이 예측

    Input
    -----
    image : np.ndarray
        BGR 원본 이미지
    detection : DetectionResult
        A모듈의 출력. 각 bbox에 대해 속성을 예측

    Output
    ------
    List[AttributeResult]
        bbox 1개당 AttributeResult 1개.
        - gender: "male" | "female"
        - gender_confidence: 0.0 ~ 1.0
        - age: float (예: 32.5)

    TODO (윤호)
    ----------------
    1. _load_model:
       - MobileNetV2 backbone + (gender head, age head) 로드
       - self.device 이동, eval 모드
    2. predict:
       - detection.bboxes 각각에 대해:
         (a) bbox.crop(image)
         (b) 224x224 resize + ImageNet normalize
         (c) forward → gender_logits(2), age_pred(scalar)
         (d) softmax로 gender/confidence 추출
         (e) AttributeResult로 감싸 리스트에 추가
       - 결과 리스트 반환
    3. visualize:
       - 각 결과를 bbox 위에 "M 0.93, 32y" 식으로 표기

    """

    INPUT_SIZE: int = 112
    # 모델 출력 컨벤션: 0=female, 1=male  → index 순서가 그대로 라벨 순서.
    GENDER_LABELS = ("female", "male")
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _load_model(self) -> Any:
        """
        self.model_path 에서 체크포인트를 로드해 MobileNetV2GenderAge 를 만든다.
        체크포인트는 train.py 가 저장한 dict({'state_dict', 'mae', 'gacc', ...})
        형태이거나, state_dict 자체일 수 있다.
        """
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )

        model = MobileNetV2GenderAge()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[attribute] state_dict mismatch — "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        model.to(self.device).eval()
        return model

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR crop → (3, INPUT_SIZE, INPUT_SIZE) float32 텐서."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.IMAGENET_MEAN, dtype=np.float32)) / np.array(
            self.IMAGENET_STD, dtype=np.float32
        )
        return rgb.transpose(2, 0, 1)  # HWC → CHW

    @torch.no_grad()
    def predict(
        self, image: np.ndarray, detection: DetectionResult
    ) -> list[AttributeResult]:
        if not detection.bboxes:
            return []

        tensors, valid_bboxes = [], []
        for bbox in detection.bboxes:
            crop = bbox.crop(image)
            if crop.size == 0:  # bbox 가 이미지 밖이라 빈 crop
                continue
            tensors.append(self._preprocess(crop))
            valid_bboxes.append(bbox)

        if not tensors:
            return []

        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        gender_logits, age_logits = self.model(batch)

        gender_probs = torch.softmax(gender_logits, dim=1)
        gender_idx = gender_probs.argmax(dim=1)
        gender_conf = gender_probs.gather(1, gender_idx.unsqueeze(1)).squeeze(1)

        age_idx_arr = torch.arange(
            age_logits.size(1), device=self.device, dtype=torch.float32
        )
        ages = (torch.softmax(age_logits, dim=1) * age_idx_arr).sum(dim=1)

        gender_idx = gender_idx.cpu().tolist()
        gender_conf = gender_conf.cpu().tolist()
        ages = ages.cpu().tolist()

        return [
            AttributeResult(
                bbox=bbox,
                gender=self.GENDER_LABELS[gi],
                gender_confidence=float(gc),
                age=float(a),
            )
            for bbox, gi, gc, a in zip(valid_bboxes, gender_idx, gender_conf, ages)
        ]

    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    LINE_THICKNESS = 1

    def visualize(
        self, image: np.ndarray, results: list[AttributeResult]
    ) -> np.ndarray:
        out = image.copy()
        for r in results:
            x1, y1, x2, y2 = (
                int(r.bbox.x1),
                int(r.bbox.y1),
                int(r.bbox.x2),
                int(r.bbox.y2),
            )
            initial = "M" if r.gender == "male" else "F"
            label = f"{initial} {r.gender_confidence:.2f}, {r.age:.0f}y"

            #cv2.rectangle(out, (x1, y1), (x2, y2), self.BOX_COLOR, self.LINE_THICKNESS)
            cv2.putText(
                out,
                label,
                (x1, max(12, y1 - 4)),
                self.FONT,
                self.FONT_SCALE,
                self.TEXT_COLOR,
                self.LINE_THICKNESS,
            )
        return out
