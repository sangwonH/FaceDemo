from typing import Any, List
import numpy as np

from core.base import BaseLandmarkDetector
from core.types import DetectionResult, LandmarkResult


class PFLDLandmark(BaseLandmarkDetector):
    """
    B모듈: PFLD 기반 얼굴 랜드마크 검출

    Input
    -----
    image : np.ndarray
        BGR 원본 이미지, shape (H, W, 3)
    detection : DetectionResult
        A모듈의 출력. 각 bbox에 대해 랜드마크를 추출한다.

    Output
    ------
    List[LandmarkResult]
        bbox 1개당 LandmarkResult 1개. 비어 있을 수 있음.
        landmarks는 원본 이미지 좌표계 기준 (N, 2) ndarray.

    TODO (신웅)
    ----------------
    1. _load_model:
       - PFLD 가중치 로드, self.device 이동, eval 모드
    2. predict:
       - detection.bboxes 각각에 대해:
         (a) bbox.crop(image)으로 얼굴 영역 자르기
         (b) 112x112(또는 모델 입력)로 resize, normalize
         (c) forward → (N, 2) 정규화 좌표
         (d) bbox 크기/위치 기준으로 원본 좌표계로 역변환
         (e) LandmarkResult(bbox=bbox, landmarks=...) 추가
       - 결과 리스트 반환
    3. visualize:
       - 각 LandmarkResult의 landmarks를 점으로 그리기

    """

    INPUT_SIZE: int = 112
    # NUM_LANDMARKS: int = 98   # PFLD-98 기준. 5/68 변형 시 수정.

    def _load_model(self) -> Any:
        raise NotImplementedError("PFLD 가중치 로딩을 구현")

    def predict(self, image: np.ndarray,
                detection: DetectionResult) -> List[LandmarkResult]:
        raise NotImplementedError("PFLD forward + 좌표 역변환을 구현")

    def visualize(self, image: np.ndarray,
                  results: List[LandmarkResult]) -> np.ndarray:
        raise NotImplementedError("랜드마크 점 그리기를 구현")