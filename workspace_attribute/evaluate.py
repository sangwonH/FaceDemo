"""
체크포인트 평가 스크립트.

사용:
  python evaluate.py --ckpt checkpoints/best.pth --afad_root tarball/AFAD-Full
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AFADDataset, load_afad_items
from model import MobileNetAgeGender
from transforms import build_transforms
from train import split_train_val


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_afad_items(args.afad_root)
    _, val_items = split_train_val(items, args.val_ratio, args.seed)

    _, val_tf = build_transforms(args.img_size)
    val_ds = AFADDataset(val_items, transform=val_tf)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = MobileNetAgeGender(width_mult=args.width_mult, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    age_idx = torch.arange(0, 101, device=device, dtype=torch.float32)
    mae_sum = gacc_sum = n = 0
    tp = [0, 0]; fp = [0, 0]; fn = [0, 0]

    with torch.no_grad():
        for x, g, a in loader:
            x = x.to(device); g = g.to(device); a = a.to(device)
            logit_g, logit_a = model(x)
            pred_age = (F.softmax(logit_a.float(), dim=1) * age_idx).sum(dim=1)
            pred_g = logit_g.argmax(1)
            mae_sum += (pred_age - a.float()).abs().sum().item()
            gacc_sum += (pred_g == g).sum().item()
            n += x.size(0)
            for cls in (0, 1):
                tp[cls] += ((pred_g == cls) & (g == cls)).sum().item()
                fp[cls] += ((pred_g == cls) & (g != cls)).sum().item()
                fn[cls] += ((pred_g != cls) & (g == cls)).sum().item()

    print(f"[eval] N={n}")
    print(f"  MAE        : {mae_sum / n:.3f}")
    print(f"  Gender Acc : {gacc_sum / n:.4f}")
    for cls, name in enumerate(["female", "male"]):
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        print(f"  Gender[{name}] precision={p:.4f}  recall={r:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--afad_root", default="tarball/AFAD-Full")
    p.add_argument("--img_size",    type=int,   default=112)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--width_mult",  type=float, default=1.0)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
