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
import sys
import time
from pathlib import Path

import cv2
import torch

from modules.attribute.attribute import MobileNetV2Attribute
from modules.detection.detector import RetinaFaceDetector
from modules.landmark.landmark import PFLDLandmarkDetector


def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / relative_path)
    return str(Path(__file__).resolve().parents[1] / relative_path)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    # p.add_argument("--device", default="cuda")
    p.add_argument("--device", default=default_device())
    p.add_argument(
        "--ckpt_det",
        default=resource_path("modules/detection/weights/combined_100pct_final.pth"),
    )
    p.add_argument(
        "--ckpt_landmark",
        default=resource_path("modules/landmark/weights/pfld_100_retina_train.pth.tar"),
    )
    p.add_argument(
        "--ckpt_attribute",
        default=resource_path("modules/attribute/weights/mobilenet_age_gender.pth"),
    )
    p.add_argument(
        "--no_mirror", action="store_true", help="좌우 반전 끄기 (기본은 거울 모드)"
    )
    args = p.parse_args()

    # ---- 모듈 인스턴스 (모델은 한 번만 로드) ----
    print("[init] A: RetinaFaceDetector")
    a_detector = RetinaFaceDetector(model_path=args.ckpt_det, device=args.device)
    print("[init] B: PFLDLandmark")
    b_landmark = PFLDLandmarkDetector(model_path=args.ckpt_landmark, device=args.device)
    print("[init] C: MobileNetV2Attribute")
    c_attribute = MobileNetV2Attribute(model_path=args.ckpt_attribute,
                                       device=args.device)
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
