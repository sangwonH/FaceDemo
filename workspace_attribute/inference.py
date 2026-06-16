"""
추론 스크립트 (standalone 데모).

사용 예시:
  python inference.py --ckpt checkpoints/best.pth --image test.jpg --out outputs/out.jpg

얼굴 검출:
  facenet-pytorch (MTCNN) 가 설치되어 있으면 자동으로 사용.
  설치되어 있지 않으면 입력 이미지를 "이미 얼굴이 crop 된 이미지"로 간주.

설치:
  pip install facenet-pytorch

NOTE
----
본 스크립트는 단독 데모용. 전체 파이프라인(A→B→C) 통합 시에는
얼굴 검출이 A 모듈(RetinaFace)에서 끝나므로, 여기 MTCNN 부분은 사용하지 않고
load_model / build_transform / predict_faces 만 modules/attribute/attribute.py
에서 재사용한다.
"""

import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from model import MobileNetAgeGender
from PIL import Image
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN

    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False


def load_model(ckpt_path: str, device: str, width_mult: float = 1.0):
    model = MobileNetAgeGender(width_mult=width_mult, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    model.load_state_dict(state)
    model.eval()
    return model


def build_transform(img_size: int = 112):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predict_faces(model, faces_pil, tf, device):
    """faces_pil: PIL.Image 리스트 (이미 face crop 됨)."""
    if not faces_pil:
        return [], []
    xs = torch.stack([tf(f) for f in faces_pil]).to(device)
    lg, la = model(xs)
    gender = lg.argmax(1).cpu().tolist()
    age_idx = torch.arange(la.size(1), device=device, dtype=torch.float32)
    ages = (F.softmax(la, dim=1) * age_idx).sum(1).cpu().tolist()
    return gender, ages


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device, args.width_mult)
    tf = build_transform(args.img_size)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if HAS_MTCNN:
        detector = MTCNN(keep_all=True, device=device)
        boxes, _ = detector.detect(img_pil)
        if boxes is None:
            print("No face detected.")
            return
        faces, crop_boxes = [], []
        W, H = img_pil.size
        for x1, y1, x2, y2 in boxes.astype(int):
            # 40% margin (IMDB-WIKI crop 정책과 비슷하게)
            m = int(0.4 * max(x2 - x1, y2 - y1) * 0.5)
            x1, y1 = max(0, x1 - m), max(0, y1 - m)
            x2, y2 = min(W, x2 + m), min(H, y2 + m)
            faces.append(img_pil.crop((x1, y1, x2, y2)))
            crop_boxes.append((x1, y1, x2, y2))
    else:
        print(
            "[warn] facenet-pytorch not installed; treating input as pre-cropped face."
        )
        faces = [img_pil]
        crop_boxes = [(0, 0, img_pil.width, img_pil.height)]

    genders, ages = predict_faces(model, faces, tf, device)

    for (x1, y1, x2, y2), g, a in zip(crop_boxes, genders, ages):
        label = f"{'M' if g == 1 else 'F'}, {a:.1f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        print(f"box=({x1},{y1},{x2},{y2}) gender={'M' if g == 1 else 'F'} age={a:.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img_bgr)
    print(f"Saved: {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/out.jpg")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--width_mult", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
