import argparse

import cv2

from modules.detection.detector import RetinaFaceDetector
from modules.landmark.landmark import PFLDLandmarkDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--out", default="./l_demo.png")
    p.add_argument(
        "--ckpt", default="modules/landmark/weights/pfld_100_retina_train.pth.tar"
    )

    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    """run A module (RetinaFace)"""
    model_det = RetinaFaceDetector(
        "modules/detection/weights/combined_100pct_final.pth", device=args.device
    )
    det_results = model_det.predict(image)
    print(f"[A·retinaface] detected {len(det_results.bboxes)} face(s)")
    if len(det_results.bboxes) == 0:
        raise Exception("Module A does not estimate bbox.")

    """run C module (PFLD)"""
    model_landmark = PFLDLandmarkDetector(model_path=args.ckpt)
    landmark_results = model_landmark.predict(image, det_results)
    landmark_vis = model_landmark.visualize(image, landmark_results)

    cv2.imwrite(args.out, landmark_vis)


if __name__ == "__main__":
    main()
