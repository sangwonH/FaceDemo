from typing import Any

import cv2
import numpy as np
import torch

from core.base import BaseLandmarkDetector
from core.types import DetectionResult, LandmarkResult
from modules.landmark.network import PFLDInference


class PFLDLandmarkDetector(BaseLandmarkDetector):
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
    NUM_LANDMARKS: int = 98  # PFLD-98 기준. 5/68 변형 시 수정.

    def _load_model(self) -> Any:
        """Base module에서 체크포인트를 경로로 받는 것에 따라,
        생성시 받은 체크포인트 경로를 통해 로드

        Returns:
            Any: Landmark Network를 반환 - self에 반환
        """

        """
        checkpoint는 model 파라미터만 또는 training 정보가 포함될 수 있음
        FLDE_BACKBONE이 추론 파라미터임
        """
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model = PFLDInference(channel_scale=1)
        missing, unexpected = model.load_state_dict(ckpt["pfld_backbone"])

        """check mismatching if strict=False"""
        if missing or unexpected:
            print(
                f"[Landmark] state_dict mismatch — "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        model.to(device=self.device).eval()

        """return to self attribute"""
        return model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """PFLD 학습에 사용한 torchvision ToTensor 변환의 opencv 구현

        Args:
            image (np.ndarray): Module A 가 탐지한 박스로 크롭된 입력 이미지

        Returns:
            np.ndarray: shape(112,112,3), 0-1, np.float32
        """
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # -> 학습코드에서 미적용되어 제거하였습니다.
        image_resized = cv2.resize(image, (self.INPUT_SIZE, self.INPUT_SIZE))
        image_normalized = image_resized.astype(np.float32) / 255

        return image_normalized

    def predict(
        self, image: np.ndarray, detection: DetectionResult
    ) -> list[LandmarkResult]:
        """단일/다수 사람 얼굴이 포함된 이미지와
            Module A 에서 추론한 탐지박스를 활용하여
            각각 얼굴을 크롭하여 추론
            박스의 탐지가 불가능한 경우 빈 리스트를 반환

        Args:
            image (np.ndarray): 단일/다수 기본 입력 이미지
            detection (DetectionResult): Module A의 탐지 결과

        Returns:
            list[LandmarkResult]: 0-1로 정규화된 추론값
        """
        if not detection.bboxes:
            return []

        tensors, valid_bboxes = [], []
        for bbox in detection.bboxes:
            crop = bbox.crop(image)
            if crop.size == 0:
                continue
            tensors.append(self._preprocess(crop))
            valid_bboxes.append(bbox)

        if not tensors:
            return []

        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        batch = batch.permute(0, 3, 1, 2)
        B = len(tensors)
        landmarks = self.model(batch)
        landmarks = landmarks.reshape(B, self.NUM_LANDMARKS, -1)

        return [
            LandmarkResult(
                bbox=valid_bboxes[i], landmarks=landmarks[i].detach().cpu().numpy()
            )
            for i in range(len(valid_bboxes))
        ]

    def denormalize_landmarks_to_global(
        self, results: LandmarkResult
    ) -> LandmarkResult:
        """0-1 공간의 추론 결과를
        bbox를 이용하여 global image의 좌표로 변환

        Args:
            results (LandmarkResult): Module C의 모델 추론

        Returns:
            LandmarkResult: global image 스케일로 변환된 landmarks
        """
        bbox_w = results.bbox.x2 - results.bbox.x1
        bbox_h = results.bbox.y2 - results.bbox.y1
        landmarks_unscale = results.landmarks * np.array([bbox_w, bbox_h])
        landmarks_global = landmarks_unscale + np.array(
            [results.bbox.x1, results.bbox.y1]
        )

        return LandmarkResult(bbox=results.bbox, landmarks=landmarks_global)

    def visualize(self, image: np.ndarray, results: list[LandmarkResult]) -> np.ndarray:
        """Module C의 시각화 코드

        Args:
            image (np.ndarray): 시각화를 위한 crop되지 않은 원본 이미지
            results (LandmarkResult): Module C의 모델 추론

        Returns:
            np.ndarray: visualized image, image shape, BGR
        """
        image_for_vis = image.copy()

        for res in results:
            landmarks_global = self.denormalize_landmarks_to_global(res).landmarks

            for x, y in landmarks_global.astype(np.int16):
                # TODO: Webcam 환경에서 유효한 circle radius가 필요합니다.
                cv2.circle(image_for_vis, (x, y), 3, (255, 0, 0), -1)

        return image_for_vis
