"""
학습 끝난 best.pth 를 메인 파이프라인의 modules/attribute/weights/ 로 복사.

사용:
  python scripts/export_to_module.py --ckpt checkpoints/best.pth
"""
import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DST = REPO_ROOT / "modules" / "attribute" / "weights" / "mobilenet_age_gender.pth"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pth")
    p.add_argument("--dst",  default=str(DEFAULT_DST))
    args = p.parse_args()

    src = Path(args.ckpt).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    dst = Path(args.dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied: {src}\n     -> {dst}")


if __name__ == "__main__":
    main()
