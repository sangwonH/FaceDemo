from typing import Any
import numpy as np

from core.base import BaseFaceDetector
from core.types import DetectionResult, FaceBBox

class RetinaFaceDetector(BaseFaceDetector):
    """
    A모듈: RetinaFace 기반 얼굴 검출

    Input
    -----
    image : np.ndarray
        BGR 이미지, shape (H, W, 3)

    Output
    ------
    DetectionResult
        검출된 얼굴들의 FaceBBox 리스트

    TODO (정빈)
    ----------------
    1. _load_model:
       - RetinaFace 가중치(self.model_path)를 로드해서 self.device로 이동
       - eval 모드로 두고 반환
    2. predict:
       - image 전처리 (resize, normalize, BCHW 텐서 변환)
       - forward → bbox, score
       - DetectionResult(bboxes=[...]) 반환
    3. visualize:
       - 입력 image를 copy한 뒤 각 bbox를 사각형으로 그리기
       - confidence를 좌상단에 텍스트로 표시
       - 결과 이미지를 반환 (np.ndarray, BGR)

    """

    CONF_THRESHOLD: float = 0.5
    
    def _load_model(self) -> Any:
        raise NotImplementedError("RetinaFace 가중치 로딩을 구현")

    def predict(self, image: np.ndarray) -> DetectionResult:
        raise NotImplementedError("RetinaFace forward + 후처리를 구현")

    def visualize(self, image: np.ndarray,
                  result: DetectionResult) -> np.ndarray:
        raise NotImplementedError("bbox 그리기를 구현")