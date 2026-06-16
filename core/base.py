from abc import ABC, abstractmethod

import numpy as np

from core.types import AttributeResult, DetectionResult, LandmarkResult


class BaseModule(ABC):
    """모든 모듈의 최상위 추상클래스"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        """가중치 로드. 각 모듈에서 구현."""
        ...

    @abstractmethod
    def predict(self, *args, **kwargs):
        """추론. 시그니처는 하위 클래스에서 고정."""
        ...

    @abstractmethod
    def visualize(self, image: np.ndarray, result) -> np.ndarray:
        """결과를 image 위에 그려서 BGR np.ndarray로 반환."""
        ...


class BaseFaceDetector(BaseModule):
    """A모듈 (RetinaFace 등)"""

    @abstractmethod
    def predict(self, image: np.ndarray) -> DetectionResult: ...

    @abstractmethod
    def visualize(self, image: np.ndarray, result: DetectionResult) -> np.ndarray: ...


class BaseLandmarkDetector(BaseModule):
    """B모듈 (PFLD 등)"""

    @abstractmethod
    def predict(
        self, image: np.ndarray, detection: DetectionResult
    ) -> list[LandmarkResult]: ...

    @abstractmethod
    def visualize(
        self, image: np.ndarray, results: list[LandmarkResult]
    ) -> np.ndarray: ...


class BaseFaceAttribute(BaseModule):
    """C모듈 (MobileNetV2 등)"""

    @abstractmethod
    def predict(
        self, image: np.ndarray, detection: DetectionResult
    ) -> list[AttributeResult]: ...

    @abstractmethod
    def visualize(
        self, image: np.ndarray, results: list[AttributeResult]
    ) -> np.ndarray: ...
