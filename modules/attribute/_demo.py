"""
C모듈 (MobileNetV2Attribute) 단독 데모 / 통합 시뮬레이션.

A모듈(RetinaFaceDetector)이 완성되기 전, MTCNN 으로 얼굴을 검출해
DetectionResult 를 만들고 C모듈에 넣어 end-to-end 흐름을 검증한다.
A모듈이 준비되면 detect_faces_mtcnn() 호출부만 RetinaFaceDetector 로
교체하면 된다.

실행 (프로젝트 루트에서):
  python -m modules.attribute._demo --image test.jpg --out outputs/c_demo.jpg

의존성:
  pip install facenet-pytorch
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from core.types import DetectionResult, FaceBBox
from modules.attribute.attribute import MobileNetV2Attribute

try:
    from facenet_pytorch import MTCNN
except ImportError as e:
    raise SystemExit(
        "facenet-pytorch 가 필요합니다. `pip install facenet-pytorch` 로 설치하세요."
    ) from e


def detect_faces_mtcnn(image_bgr: np.ndarray, device: str) -> DetectionResult:
    """A모듈 placeholder — MTCNN 검출 결과를 DetectionResult 로 변환."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)

    detector = MTCNN(keep_all=True, device=device)
    boxes, probs = detector.detect(pil)
    if boxes is None:
        return DetectionResult(bboxes=[])

    H, W = image_bgr.shape[:2]
    bboxes = []
    for (x1, y1, x2, y2), p in zip(boxes, probs):
        # 40% margin — 학습 시 face crop 정책에 가까운 padding
        m = int(0.4 * max(x2 - x1, y2 - y1) * 0.5)
        x1 = max(0, int(x1) - m)
        y1 = max(0, int(y1) - m)
        x2 = min(W, int(x2) + m)
        y2 = min(H, int(y2) + m)
        bboxes.append(FaceBBox(x1=x1, y1=y1, x2=x2, y2=y2,
                               confidence=float(p)))
    return DetectionResult(bboxes=bboxes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--out",   default="outputs/c_demo.jpg")
    p.add_argument("--ckpt",
                   default="modules/attribute/weights/mobilenet_age_gender.pth")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    # ---- A 모듈 (placeholder) ----
    detection = detect_faces_mtcnn(image, device=args.device)
    print(f"[A·mtcnn] detected {len(detection.bboxes)} face(s)")
    if not detection.bboxes:
        return

    # ---- C 모듈 ----
    attr = MobileNetV2Attribute(model_path=args.ckpt, device=args.device)
    results = attr.predict(image, detection)
    out = attr.visualize(image, results)

    for r in results:
        print(f"  → {r.gender:>6} (conf={r.gender_confidence:.2f})  "
              f"age={r.age:.1f}  "
              f"bbox=({r.bbox.x1:.0f},{r.bbox.y1:.0f},"
              f"{r.bbox.x2:.0f},{r.bbox.y2:.0f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
