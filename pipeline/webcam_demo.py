"""
웹캠 + 통합 파이프라인 (A → B → C) 데모.

각 모듈의 predict / visualize 를 순서대로 호출해 결과를 누적 시각화한다.
현재 A모듈(RetinaFace), B모듈(PFLD) 이 미완성이라 placeholder 사용.
완성되면 main() 안의 인스턴스 두 줄만 교체.

조작:
  b  bbox (A)        토글
  l  landmark (B)    토글
  a  attribute (C)   토글
  q  종료

실행 (프로젝트 루트):
  python -m pipeline.webcam_demo --cam 0
"""

import argparse
import time

import cv2
import numpy as np
from PIL import Image

from core.types import DetectionResult, FaceBBox
from modules.attribute.attribute import MobileNetV2Attribute

try:
    from facenet_pytorch import MTCNN
except ImportError as e:
    raise SystemExit("facenet-pytorch 필요: pip install facenet-pytorch") from e


# ---------------------------------------------------------------------
# A모듈 placeholder — MTCNN
# RetinaFaceDetector 완성 시 이 클래스 통째로 제거하고 import 로 교체.
# ---------------------------------------------------------------------
class MTCNNDetector:
    BOX_COLOR = (0, 255, 0)

    def __init__(self, device: str):
        self.detector = MTCNN(keep_all=True, device=device)
        self.device = device

    def predict(self, image: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detector.detect(Image.fromarray(rgb))
        if boxes is None:
            return DetectionResult(bboxes=[])

        H, W = image.shape[:2]
        bboxes = []
        for (x1, y1, x2, y2), p in zip(boxes, probs):
            m = int(0.4 * max(x2 - x1, y2 - y1) * 0.5)
            x1 = max(0, int(x1) - m)
            y1 = max(0, int(y1) - m)
            x2 = min(W, int(x2) + m)
            y2 = min(H, int(y2) + m)
            bboxes.append(FaceBBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(p)))
        return DetectionResult(bboxes=bboxes)

    def visualize(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        out = image.copy()
        for b in result.bboxes:
            cv2.rectangle(
                out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), self.BOX_COLOR, 1
            )
        return out


# ---------------------------------------------------------------------
# B모듈 placeholder — landmark 비활성. PFLDLandmark 완성 시 교체.
# ---------------------------------------------------------------------
class NullLandmark:
    def predict(self, image: np.ndarray, detection: DetectionResult):
        return []

    def visualize(self, image: np.ndarray, results) -> np.ndarray:
        return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--ckpt", default="modules/attribute/weights/mobilenet_age_gender.pth"
    )
    p.add_argument(
        "--no_mirror", action="store_true", help="좌우 반전 끄기 (기본은 거울 모드)"
    )
    args = p.parse_args()

    # ---- 모듈 인스턴스 (모델은 한 번만 로드) ----
    print("[init] A: MTCNN (placeholder for RetinaFaceDetector)")
    a_detector = MTCNNDetector(device=args.device)
    print("[init] B: NullLandmark (placeholder for PFLDLandmark)")
    b_landmark = NullLandmark()
    print("[init] C: MobileNetV2Attribute")
    c_attribute = MobileNetV2Attribute(model_path=args.ckpt, device=args.device)
    # 완성 후 교체 예시:
    #   from modules.detection.detector import RetinaFaceDetector
    #   from modules.landmark.landmark import PFLDLandmark
    #   a_detector = RetinaFaceDetector(model_path="modules/detection/weights/...", device=args.device)
    #   b_landmark = PFLDLandmark      (model_path="modules/landmark/weights/...", device=args.device)

    # ---- 웹캠 ----
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.cam} 를 열 수 없습니다.")

    win = "face demo (q to quit)"
    print("[run] b/l/a 키로 bbox/landmark/attribute 토글, q 로 종료")
    t_prev, fps = time.time(), 0.0
    show_bbox = show_landmark = show_attribute = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            # ---- predict (A → B → C) — 항상 모두 실행 ----
            det_result = a_detector.predict(frame)
            lmk_result = b_landmark.predict(frame, det_result)
            att_result = c_attribute.predict(frame, det_result)

            # ---- visualize chain (토글에 따라 선택적으로) ----
            vis = frame.copy()
            if show_bbox:
                vis = a_detector.visualize(vis, det_result)
            if show_landmark:
                vis = b_landmark.visualize(vis, lmk_result)
            if show_attribute:
                vis = c_attribute.visualize(vis, att_result)

            # ---- FPS overlay (EMA) ----
            t_now = time.time()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = t_now

            toggles = (
                f"B[{'on' if show_bbox else 'off'}] "
                f"L[{'on' if show_landmark else 'off'}] "
                f"A[{'on' if show_attribute else 'off'}]"
            )
            cv2.putText(
                vis,
                f"FPS {fps:5.1f}  faces {len(det_result.bboxes)}  {toggles}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                show_bbox = not show_bbox
                print(f"[toggle] bbox      = {show_bbox}")
            elif key == ord("l"):
                show_landmark = not show_landmark
                print(f"[toggle] landmark  = {show_landmark}")
            elif key == ord("a"):
                show_attribute = not show_attribute
                print(f"[toggle] attribute = {show_attribute}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
