"""
A모듈 (RetinaFaceDetector) 단독 데모 — 검출 + 시각화.

입력 이미지에서 얼굴을 검출(predict)하고 bbox/confidence 를 그려(visualize)
결과 이미지를 저장한다.

실행 (프로젝트 루트에서):
  python -m modules.detection._demo --image test.jpg --out outputs/a_demo.jpg

옵션:
  --weights  검출기 가중치 (기본: ./weights/combined_100pct_final.pth)
  --device   cuda | cpu
  --conf     검출 점수 임계값 (기본: RetinaFaceDetector.CONF_THRESHOLD)
"""

import argparse
import os

import cv2

from modules.detection.detector import RetinaFaceDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--out", default="outputs/a_demo.jpg")
    p.add_argument(
        "--weights",
        default=None,
        help="검출기 가중치 (기본: RetinaFaceDetector.DEFAULT_MODEL_PATH)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=None, help="검출 점수 임계값")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    if args.conf is not None:
        RetinaFaceDetector.CONF_THRESHOLD = args.conf

    # ---- A 모듈 ----
    detector = RetinaFaceDetector(model_path=args.weights, device=args.device)
    detection = detector.predict(image)
    print(f"[A·retinaface] detected {len(detection.bboxes)} face(s)")

    out = detector.visualize(image, detection)
    for b in detection.bboxes:
        print(
            f"  → conf={b.confidence:.3f}  "
            f"bbox=({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f})"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
