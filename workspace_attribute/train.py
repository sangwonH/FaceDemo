"""
학습 스크립트 (AFAD-Full).

사용 예시:
  python train.py \
    --afad_root tarball/AFAD-Full \
    --out_dir checkpoints \
    --img_size 112 --batch_size 256 --epochs 40 \
    --lr 1e-3 --width_mult 1.0 --amp
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AFADDataset, load_afad_items
from model import MobileNetAgeGender
from transforms import build_transforms


# ---------------------------------------------------------------------
# Logging — stdout 을 터미널 + 타임스탬프 파일 양쪽에 기록.
# stderr 은 건드리지 않아 tqdm 진행바가 로그 파일을 오염시키지 않는다.
# ---------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(log_dir: str = "runs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")
    log_file = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"[log] writing to {log_path}")
    return log_path


# ---------------------------------------------------------------------
# 데이터 로딩 / split
# ---------------------------------------------------------------------
def collect_items(afad_root):
    if not os.path.isdir(afad_root):
        raise RuntimeError(f"AFAD root not found: {afad_root}")
    items = load_afad_items(afad_root)
    if not items:
        raise RuntimeError(f"No items under {afad_root}. 디렉토리 구조 확인.")
    males   = sum(1 for _, _, g in items if g == 1)
    females = len(items) - males
    print(f"[data] AFAD items: {len(items)}  (male={males}, female={females})")
    return items


def split_train_val(items, val_ratio=0.1, seed=42):
    g = random.Random(seed)
    idxs = list(range(len(items)))
    g.shuffle(idxs)
    n_val = int(val_ratio * len(items))
    val_idx = set(idxs[:n_val])
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    print(f"[data] split: train={len(train)}, val={len(val)}")
    return train, val


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------
def evaluate(model, loader, device, amp, desc="eval"):
    model.eval()
    age_idx = torch.arange(0, 101, device=device, dtype=torch.float32)
    mae_sum, gacc_sum, n = 0.0, 0, 0
    with torch.no_grad():
        for x, g, a in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit_g, logit_a = model(x)
            pred_age = (F.softmax(logit_a.float(), dim=1) * age_idx).sum(dim=1)
            mae_sum  += (pred_age - a.float()).abs().sum().item()
            gacc_sum += (logit_g.argmax(1) == g).sum().item()
            n        += x.size(0)
    return mae_sum / n, gacc_sum / n


def main(args):
    setup_logging(args.log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}, amp={args.amp}")

    # ---- data ----
    train_tf, val_tf = build_transforms(args.img_size)
    items = collect_items(args.afad_root)
    train_items, val_items = split_train_val(items, args.val_ratio, args.seed)

    train_ds = AFADDataset(train_items, transform=train_tf)
    val_ds   = AFADDataset(val_items,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    # ---- model / optim ----
    model = MobileNetAgeGender(num_age=101, num_gender=2,
                               width_mult=args.width_mult,
                               pretrained=args.pretrained,
                               dropout=args.dropout).to(device)

    if args.diff_lr:
        backbone_params = list(model.features.parameters())
        head_params     = list(model.head_gender.parameters()) + \
                          list(model.head_age.parameters())
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ce = nn.CrossEntropyLoss()

    # ---- train ----
    os.makedirs(args.out_dir, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, steps = 0.0, 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{args.epochs}",
                    leave=False)
        for x, g, a in pbar:
            x = x.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logit_g, logit_a = model(x)
                loss_g = ce(logit_g, g)
                loss_a = ce(logit_a, a)
                loss = args.w_gender * loss_g + args.w_age * loss_a

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{loss_sum/steps:.4f}")

        mae, gacc = evaluate(model, val_loader, device, args.amp)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss={loss_sum/steps:.4f}  MAE={mae:.2f}  "
              f"GenderAcc={gacc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.0f}s)")

        ckpt = {
            "state_dict": model.state_dict(),
            "epoch": epoch, "mae": mae, "gacc": gacc,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))
        if mae < best_mae:
            best_mae = mae
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            print(f"  -> new best (MAE={mae:.2f}) saved.")

    print(f"\nDone. Best val MAE = {best_mae:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--afad_root", type=str, default="tarball/AFAD-Full")
    p.add_argument("--out_dir",   type=str, default="checkpoints")
    p.add_argument("--log_dir",   type=str, default="runs",
                   help="train_<YYYYMMDD_HHMMSS>.log 가 저장되는 디렉토리")

    p.add_argument("--img_size",    type=int, default=112)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs",      type=int, default=40)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--w_age",       type=float, default=1.0)
    p.add_argument("--w_gender",    type=float, default=1.0)

    p.add_argument("--width_mult", type=float, default=1.0)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--diff_lr", action="store_true",
                   help="백본은 0.1*lr, head 는 lr 로 차등 학습")

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
