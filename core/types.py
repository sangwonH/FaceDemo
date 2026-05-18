from dataclasses import dataclass, field
from typing import List, Literal
import numpy as np


@dataclass
class FaceBBox:
    """얼굴 1개의 바운딩 박스"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def crop(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 이 박스 영역만 잘라서 반환 (B, C 모듈에서 사용)"""
        h, w = image.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return image[y1:y2, x1:x2]


@dataclass
class DetectionResult:
    """A모듈(RetinaFace) 출력 — 한 프레임의 모든 얼굴"""
    bboxes: List[FaceBBox] = field(default_factory=list)


@dataclass
class LandmarkResult:
    """B모듈(PFLD) 출력 — 얼굴 1개당 1개"""
    bbox: FaceBBox
    landmarks: np.ndarray   # shape: (N, 2)


@dataclass
class AttributeResult:
    """C모듈(MobileNetV2) 출력 — 얼굴 1개당 1개"""
    bbox: FaceBBox
    gender: Literal["male", "female"]
    gender_confidence: float
    age: float