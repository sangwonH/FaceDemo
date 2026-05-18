from typing import Any, List
import numpy as np

from core.base import BaseFaceAttribute
from core.types import DetectionResult, AttributeResult
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

    INPUT_SIZE: int = 224
    GENDER_LABELS = ("male", "female")

    def _load_model(self) -> Any:
        raise NotImplementedError("MobileNetV2 가중치 로딩을 구현")

    def predict(self, image: np.ndarray,
                detection: DetectionResult) -> List[AttributeResult]:
        raise NotImplementedError("성별/나이 추론을 구현")

    def visualize(self, image: np.ndarray,
                  results: List[AttributeResult]) -> np.ndarray:
        raise NotImplementedError("텍스트 표시를 구현")