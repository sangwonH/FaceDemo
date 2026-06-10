from typing import Any

import cv2
import numpy as np
import torch

from core.base import BaseFaceDetector
from core.types import DetectionResult, FaceBBox

from .box_utils import decode
from .config import cfg_mnet, cfg_re50
from .network import RetinaFace
from .prior_box import PriorBox


def _safe_torch_load(path: str):
    """체크포인트 로드. 안전한 weights_only=True 를 우선 시도하고,
    텐서 외 객체가 들어있어 실패하면 weights_only=False 로 폴백한다.
    (torch>=2.6 의 기본값 변경 FutureWarning 회피)
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _py_cpu_nms(dets: np.ndarray, thresh: float) -> list[int]:
    """순수 numpy NMS (RetinaFace 원본 utils/nms/py_cpu_nms.py 와 동일)."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


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

    학습한 가중치(`RetinaFace/Pytorch_Retinaface` 또는 modules.detection.training.train
    산출물)를 로드해 추론한다. 백본 종류는 클래스 속성 `NETWORK` 로 지정한다
    ("mobile0.25" 또는 "resnet50").
    """

    # 기본 추론 가중치 (v1 reduction 학습 완료, WIDER+WFLW 통합). FaceDemo 루트 기준.
    DEFAULT_MODEL_PATH: str = "./weights/combined_100pct_final.pth"

    NETWORK: str = "mobile0.25"  # 백본: mobile0.25 / resnet50
    CONF_THRESHOLD: float = 0.5  # 최종 검출로 인정할 점수 임계값
    NMS_THRESHOLD: float = 0.4   # NMS IoU 임계값
    TOP_K: int = 5000            # NMS 이전 상위 K
    KEEP_TOP_K: int = 750        # NMS 이후 상위 K

    # 학습 시 사용한 BGR 평균 (data_augment 와 동일)
    _RGB_MEAN = (104, 117, 123)

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        # model_path 생략 시 DEFAULT_MODEL_PATH(v1 통합 검출기) 사용.
        super().__init__(model_path or self.DEFAULT_MODEL_PATH, device)

    def _load_model(self) -> Any:
        """RetinaFace 가중치를 self.model_path 에서 로드해 eval 모델 반환."""
        if self.NETWORK == "mobile0.25":
            cfg = dict(cfg_mnet)
        elif self.NETWORK == "resnet50":
            cfg = dict(cfg_re50)
        else:
            raise ValueError(f"Unknown network: {self.NETWORK}")
        # 추론 시 백본 사전학습 가중치는 불필요 (전체 가중치를 덮어쓰므로).
        cfg["pretrain"] = False
        self.cfg = cfg

        net = RetinaFace(cfg=cfg, phase="test")
        net = self._load_state_dict(net, self.model_path)
        net = net.to(self.device)
        net.eval()
        return net

    @staticmethod
    def _load_state_dict(net: RetinaFace, path: str) -> RetinaFace:
        """체크포인트 로드 (DataParallel 'module.' prefix / 'state_dict' 키 처리)."""
        ckpt = _safe_torch_load(path)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        def strip(k: str) -> str:
            return k[7:] if k.startswith("module.") else k

        state_dict = {strip(k): v for k, v in ckpt.items()}
        net.load_state_dict(state_dict, strict=False)
        return net

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> DetectionResult:
        """BGR 이미지 → 얼굴 bbox 검출. 전처리 → forward → decode → NMS."""
        im_height, im_width, _ = image.shape

        # 전처리: float32, 평균 차감, CHW, 배치 차원 추가
        img = np.float32(image)
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
        img -= self._RGB_MEAN
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # forward (phase='test' 이므로 conf 는 softmax 적용됨)
        loc, conf, _landms = self.model(img)

        # 앵커 생성 후 박스 디코딩
        priors = PriorBox(self.cfg, image_size=(im_height, im_width)).forward()
        priors = priors.to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # 점수 임계값 필터
        inds = np.where(scores > self.CONF_THRESHOLD)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # NMS 이전 상위 K
        order = scores.argsort()[::-1][: self.TOP_K]
        boxes = boxes[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = _py_cpu_nms(dets, self.NMS_THRESHOLD)
        dets = dets[keep, :]

        # NMS 이후 상위 K
        dets = dets[: self.KEEP_TOP_K, :]

        bboxes = [
            FaceBBox(
                x1=float(d[0]),
                y1=float(d[1]),
                x2=float(d[2]),
                y2=float(d[3]),
                confidence=float(d[4]),
            )
            for d in dets
        ]
        return DetectionResult(bboxes=bboxes)

    def visualize(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """입력 image 복사본에 bbox 와 confidence 를 그려 BGR 이미지로 반환."""
        vis = image.copy()
        for box in result.bboxes:
            x1, y1, x2, y2 = (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{box.confidence:.2f}"
            cv2.putText(
                vis,
                text,
                (x1, y1 + 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
            )
        return vis
