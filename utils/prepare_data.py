"""
WIDER FACE + WFLW 통합 데이터 준비 스크립트 (신규 다운로드용 단일 진입점).

협업 환경에서 "각자 데이터를 받아 변환부터 맞추는" 단계를 멱등(idempotent)하게
재현하기 위한 도구다. raw 다운로드 상태에서 한 줄로 실행하면:

    1) 압축 해제      WFLW_*.tar.gz / *.zip 이 풀려 있지 않으면 푼다 (이미 있으면 skip)
    2) annotation 변환  두 데이터셋을 RetinaFace 내부 포맷(15-value)으로 통일
    3) 통합 manifest    source 컬럼이 붙은 combined_manifest.csv 생성
    4) 체크섬 리포트    산출물 sha256 + source별 통계를 prepare_report.json 으로 출력

prepare_report.json 의 sha256 만 비교하면 "같은 학습셋"임을 검증할 수 있다.

----------------------------------------------------------------------
Canonical(통일) annotation 포맷 — 이미지 단위로 그룹핑된 텍스트 (RetinaFace label.txt 호환)

    # <image_path>                                              (이미지 1장)
    x1 y1 x2 y2 l0x l0y l1x l1y l2x l2y l3x l3y l4x l4y label   (얼굴 1개, 15값)
    ...

  - bbox        : xyxy (절대 픽셀)
  - landmark    : left_eye, right_eye, nose_tip, mouth_left, mouth_right 순서
  - label       :  1 = 랜드마크 유효 / -1 = 랜드마크 없음(bbox-only)
----------------------------------------------------------------------

사용 예:
    python -m utils.prepare_data --config utils/dataset_config.example.json
    python -m utils.prepare_data --data-root /mnt/data/face --out ./data/prepared
    python -m utils.prepare_data --config cfg.json --visualize 8   # 변환 검증용 샘플 N장
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# 0. 설정
# --------------------------------------------------------------------------- #


@dataclass
class PrepareConfig:
    """모든 경로를 한 곳에서 관리한다. 코드에 경로를 박지 않기 위한 단일 출처."""

    # 입력 (raw 다운로드 위치)
    wider_label: str  # WIDER train label.txt (RetinaFace annotated, 5-landmark)
    wider_images: str  # WIDER train images/ 루트
    wflw_ann: str  # WFLW list_98pt_rect_attr_train.txt
    wflw_images: str  # WFLW_images/ 루트

    # 출력
    out_dir: str = "./data/prepared"

    # 압축 해제 후보 (없으면 무시) — (archive, 풀렸는지 확인할 디렉토리)
    archives: list[tuple[str, str]] = field(default_factory=list)

    # 재현성
    seed: int = 42

    @staticmethod
    def from_args(args: argparse.Namespace) -> "PrepareConfig":
        """--config(json) 우선, 없으면 --data-root 기반 기본 경로로 구성."""
        if args.config:
            with open(args.config, encoding="utf-8") as f:
                raw = json.load(f)
            raw["archives"] = [tuple(a) for a in raw.get("archives", [])]
            cfg = PrepareConfig(**raw)
        else:
            root = args.data_root
            if root is None:
                raise SystemExit("--config 또는 --data-root 중 하나는 필요합니다.")
            cfg = PrepareConfig(
                wider_label=os.path.join(root, "widerface/train/label.txt"),
                wider_images=os.path.join(root, "widerface/train/images"),
                wflw_ann=os.path.join(
                    root,
                    "WFLW_annotations/list_98pt_rect_attr_train_test/"
                    "list_98pt_rect_attr_train.txt",
                ),
                wflw_images=os.path.join(root, "WFLW_images"),
                archives=[
                    (os.path.join(root, "WFLW_images.tar.gz"),
                     os.path.join(root, "WFLW_images")),
                    (os.path.join(root, "WFLW_annotations.tar.gz"),
                     os.path.join(root, "WFLW_annotations")),
                ],
            )
        if args.out:
            cfg.out_dir = args.out
        if args.seed is not None:
            cfg.seed = args.seed
        return cfg


# --------------------------------------------------------------------------- #
# 1. Canonical 레코드 + converter (순수 함수, I/O 없음 → 테스트 용이)
# --------------------------------------------------------------------------- #


@dataclass
class CanonicalImage:
    """이미지 1장과 그 안의 얼굴들 (15값 x N)."""

    image_path: str  # out_dir 기준 상대경로 또는 절대경로
    source: str  # "wider" | "wflw"
    faces: np.ndarray  # shape (N, 15)


# WFLW 98점 → RetinaFace 5점 매핑 (methodology §3-2, wflw_dataset.py 와 동일)
_WFLW_LEFT_EYE = list(range(60, 68))
_WFLW_RIGHT_EYE = list(range(68, 76))
_WFLW_NOSE = 54
_WFLW_MOUTH_LEFT = 76
_WFLW_MOUTH_RIGHT = 82
_WFLW_N_TOKENS = 207  # 196 lmk + 4 bbox + 6 attr + 1 name


def wflw_line_to_canonical(line: str) -> tuple[str, np.ndarray] | None:
    """WFLW 1줄 → (image_name, face(1,15)) | None.

    WFLW 는 모든 랜드마크가 유효하므로 label=1 고정.
    """
    parts = line.strip().split()
    if len(parts) < _WFLW_N_TOKENS:
        return None
    coords = np.asarray(parts[:196], dtype=np.float32).reshape(98, 2)
    bbox = np.asarray(parts[196:200], dtype=np.float32)  # x1,y1,x2,y2
    image_name = parts[206]

    five = np.empty((5, 2), dtype=np.float32)
    five[0] = coords[_WFLW_LEFT_EYE].mean(axis=0)
    five[1] = coords[_WFLW_RIGHT_EYE].mean(axis=0)
    five[2] = coords[_WFLW_NOSE]
    five[3] = coords[_WFLW_MOUTH_LEFT]
    five[4] = coords[_WFLW_MOUTH_RIGHT]

    face = np.zeros((1, 15), dtype=np.float32)
    face[0, 0:4] = bbox
    face[0, 4:14] = five.reshape(-1)
    face[0, 14] = 1.0
    return image_name, face


def wider_label_to_canonical(label: list[float]) -> np.ndarray:
    """WIDER label.txt 의 face 1줄 → face(1,15).

    입력: [x, y, w, h, l0x,l0y,score, l1x,l1y,score, ..., l4x,l4y,score, conf]
    (랜드마크가 없는 bbox-only 줄은 길이가 4 일 수 있음 → label=-1)
    """
    face = np.zeros((1, 15), dtype=np.float32)
    # xywh → xyxy
    face[0, 0] = label[0]
    face[0, 1] = label[1]
    face[0, 2] = label[0] + label[2]
    face[0, 3] = label[1] + label[3]

    if len(label) >= 19:
        # score 토큰(6,9,12,15,18)을 건너뛰고 좌표만 추출 (wider_face.py 와 동일)
        idx = [4, 5, 7, 8, 10, 11, 13, 14, 16, 17]
        face[0, 4:14] = [label[i] for i in idx]
        face[0, 14] = -1.0 if face[0, 4] < 0 else 1.0
    else:
        face[0, 14] = -1.0  # bbox-only
    return face


# --------------------------------------------------------------------------- #
# 2. 파서 (raw 파일 → CanonicalImage 리스트)
# --------------------------------------------------------------------------- #


def parse_wider(label_path: str) -> list[CanonicalImage]:
    """RetinaFace 형식 WIDER label.txt 파싱. '# path' 로 이미지 블록 구분."""
    images: list[CanonicalImage] = []
    cur_path: str | None = None
    cur_faces: list[np.ndarray] = []

    def flush() -> None:
        if cur_path is not None:
            faces = (np.concatenate(cur_faces, axis=0)
                     if cur_faces else np.zeros((0, 15), dtype=np.float32))
            images.append(CanonicalImage(cur_path, "wider", faces))

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("#"):
                flush()
                cur_path = line[1:].strip()
                cur_faces = []
            else:
                label = [float(x) for x in line.split()]
                cur_faces.append(wider_label_to_canonical(label))
    flush()
    return images


def parse_wflw(ann_path: str) -> list[CanonicalImage]:
    """WFLW annotation 파싱. 한 이미지에 여러 얼굴이 있으면 병합."""
    grouped: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            rec = wflw_line_to_canonical(line)
            if rec is None:
                continue
            name, face = rec
            if name not in grouped:
                grouped[name] = []
                order.append(name)
            grouped[name].append(face)
    return [
        CanonicalImage(name, "wflw", np.concatenate(grouped[name], axis=0))
        for name in order
    ]


# --------------------------------------------------------------------------- #
# 3. 압축 해제 (멱등)
# --------------------------------------------------------------------------- #


def extract_if_needed(archives: list[tuple[str, str]]) -> list[str]:
    """(archive, expected_dir) 목록을 돌며 풀려 있지 않은 것만 해제."""
    log: list[str] = []
    for archive, expected_dir in archives:
        if os.path.isdir(expected_dir) and os.listdir(expected_dir):
            log.append(f"skip (이미 존재): {expected_dir}")
            continue
        if not os.path.exists(archive):
            log.append(f"warn (압축파일 없음): {archive}")
            continue
        dest = os.path.dirname(os.path.abspath(expected_dir))
        if archive.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive) as tf:
                tf.extractall(dest)  # noqa: S202 (신뢰된 데이터셋 아카이브)
        elif archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
        else:
            log.append(f"warn (알 수 없는 형식): {archive}")
            continue
        log.append(f"extracted: {archive} → {dest}")
    return log


# --------------------------------------------------------------------------- #
# 4. 산출물 작성
# --------------------------------------------------------------------------- #


def write_canonical(images: list[CanonicalImage], out_path: str,
                    image_root: str) -> dict:
    """CanonicalImage 리스트를 통일 텍스트로 저장. 존재하는 이미지만 기록."""
    n_img = n_face = n_missing = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for img in images:
            if not os.path.exists(os.path.join(image_root, img.image_path)):
                n_missing += 1
                continue
            f.write(f"# {img.image_path}\n")
            for face in img.faces:
                f.write(" ".join(f"{v:.3f}" for v in face) + "\n")
            n_img += 1
            n_face += len(img.faces)
    return {"images": n_img, "faces": n_face, "missing_images": n_missing}


def write_manifest(per_source: dict[str, list[CanonicalImage]],
                   roots: dict[str, str], out_path: str) -> None:
    """source/image_path/num_faces/has_landmark 를 가진 통합 인덱스 CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "image_path", "num_faces", "has_landmark"])
        for source, images in per_source.items():
            root = roots[source]
            for img in images:
                if not os.path.exists(os.path.join(root, img.image_path)):
                    continue
                has_lmk = int((img.faces[:, 14] > 0).any()) if len(img.faces) else 0
                w.writerow([source, img.image_path, len(img.faces), has_lmk])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 5. (옵션) 변환 검증용 시각화 훅
# --------------------------------------------------------------------------- #


def visualize_samples(per_source: dict[str, list[CanonicalImage]],
                      roots: dict[str, str], out_dir: str, n: int,
                      seed: int) -> None:
    """변환된 bbox/landmark 를 이미지 위에 그려 눈으로 검증 (cv2 lazy import)."""
    import cv2  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    for source, images in per_source.items():
        pick = rng.choice(len(images), size=min(n, len(images)), replace=False)
        for i in pick:
            img = images[int(i)]
            arr = cv2.imread(os.path.join(roots[source], img.image_path))
            if arr is None:
                continue
            for face in img.faces:
                x1, y1, x2, y2 = face[:4].astype(int)
                cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if face[14] > 0:
                    for k in range(5):
                        px, py = int(face[4 + 2 * k]), int(face[5 + 2 * k])
                        cv2.circle(arr, (px, py), 2, (0, 0, 255), -1)
            name = f"{source}_{os.path.basename(img.image_path)}"
            cv2.imwrite(os.path.join(vis_dir, name), arr)
    print(f"[visualize] 샘플 이미지를 {vis_dir} 에 저장")


# --------------------------------------------------------------------------- #
# 6. 파이프라인
# --------------------------------------------------------------------------- #


def run(cfg: PrepareConfig, visualize: int = 0) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    report: dict = {"config": asdict(cfg), "outputs": {}, "stats": {}}

    # 1) 압축 해제
    report["extract_log"] = extract_if_needed(cfg.archives)

    # 2) 파싱 + 변환
    wider_imgs = parse_wider(cfg.wider_label)
    wflw_imgs = parse_wflw(cfg.wflw_ann)
    per_source = {"wider": wider_imgs, "wflw": wflw_imgs}
    roots = {"wider": cfg.wider_images, "wflw": cfg.wflw_images}

    # 3) canonical 텍스트
    wider_out = os.path.join(cfg.out_dir, "wider_canonical.txt")
    wflw_out = os.path.join(cfg.out_dir, "wflw_canonical.txt")
    report["stats"]["wider"] = write_canonical(wider_imgs, wider_out,
                                               cfg.wider_images)
    report["stats"]["wflw"] = write_canonical(wflw_imgs, wflw_out,
                                              cfg.wflw_images)

    # 4) 통합 manifest
    manifest = os.path.join(cfg.out_dir, "combined_manifest.csv")
    write_manifest(per_source, roots, manifest)

    # 5) 체크섬
    for path in (wider_out, wflw_out, manifest):
        report["outputs"][os.path.basename(path)] = sha256(path)

    # 6) 옵션 시각화
    if visualize > 0:
        visualize_samples(per_source, roots, cfg.out_dir, visualize, cfg.seed)

    report_path = os.path.join(cfg.out_dir, "prepare_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[prepare_data] 완료 → {report_path}")
    print(json.dumps(report["stats"], indent=2, ensure_ascii=False))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="WIDER+WFLW 통합 데이터 준비")
    p.add_argument("--config", help="JSON 설정 파일 경로")
    p.add_argument("--data-root", help="raw 데이터 루트 (config 없을 때 기본 경로 구성)")
    p.add_argument("--out", help="산출물 출력 디렉토리 (config 의 out_dir 덮어씀)")
    p.add_argument("--seed", type=int, help="샘플링/시각화 시드")
    p.add_argument("--visualize", type=int, default=0,
                   help="source 당 N장 변환 결과를 그려 검증")
    args = p.parse_args()
    run(PrepareConfig.from_args(args), visualize=args.visualize)


if __name__ == "__main__":
    main()
