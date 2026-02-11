# ======================================
# Ultralytics YOLO + EasyOCR
# Automatic Number Plate Recognition
# (Works on Roboflow YOLOv8 dataset folders + video)
# ======================================

import os
from glob import glob

import cv2
import torch
import easyocr
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ANPR:
    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR."""

    def __init__(self, model_path: str = "anpr_best.pt", conf: float = 0.25, imgsz: int = 640):
        """
        model_path: path to your trained license-plate detector weights (e.g., best.pt)
        conf: confidence threshold for detections
        imgsz: inference image size for YOLO
        """
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def detect_plates(self, im0: np.ndarray):
        """Detect license plates in an image and return list of xyxy boxes."""
        results = self.model.predict(im0, verbose=False, conf=self.conf, imgsz=self.imgsz, device=self.device)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        return boxes

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """OCR the cropped license plate region; returns text or empty string."""
        h, w = im0.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        # clamp coords to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # skip invalid/tiny crops (avoid OCR noise/crashes)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 25 or (y2 - y1) < 12:
            return ""

        roi = im0[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # light preprocessing to improve OCR stability
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = self.reader.readtext(th, detail=0, paragraph=True)
        return " ".join(text).strip() if text else ""

    def infer_images(self, images_dir: str, output_dir: str = "anpr_preds", max_images: int = 50):
        """Run plate detection + OCR on a folder of images and save annotated images."""
        os.makedirs(output_dir, exist_ok=True)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths = []
        for e in exts:
            paths.extend(glob(os.path.join(images_dir, e)))
        paths = sorted(paths)[:max_images]

        if len(paths) == 0:
            raise ValueError(f"No images found in: {images_dir}")

        print(f"Found {len(paths)} images in {images_dir}")
        print(f"Saving annotated outputs to {output_dir}")

        for p in paths:
            im0 = cv2.imread(p)
            if im0 is None:
                continue

            boxes = self.detect_plates(im0)
            ann = Annotator(im0, line_width=3)

            for bbox in boxes:
                text = self.extract_text(im0, bbox)
                label = text if text else "plate"
                ann.box_label(bbox, label=label, color=colors(17, True))

            out_path = os.path.join(output_dir, os.path.basename(p))
            cv2.imwrite(out_path, im0)

        print("Done ✅")

    def infer_video(self, source: str = 0, output_path: str = None, display: bool = False):
        """
        Real-time/video ANPR inference.
        NOTE: display=True uses cv2.imshow which usually does NOT work in Great Lakes Jupyter.
        Keep display=False and write output video instead.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("🚀 Starting ANPR video inference...")

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            boxes = self.detect_plates(im0)
            ann = Annotator(im0, line_width=3)

            for bbox in boxes:
                text = self.extract_text(im0, bbox)
                label = text if text else "plate"
                ann.box_label(bbox, label=label, color=colors(17, True))

            if writer:
                writer.write(im0)

            if display:
                cv2.imshow("ANPR (press q to quit)", im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 1) Put your model weights here (trained plate detector)
    MODEL_PATH = "anpr_best.pt"  # or "runs/detect/train/weights/best.pt"

    # 2) Set this to your extracted Roboflow dataset root
    # Example: "license-plates-1"
    DATASET_ROOT = "License_plate"

    # 3) Choose split: train / valid / test
    IMAGES_DIR = os.path.join(DATASET_ROOT, "valid", "images")

    anpr = ANPR(model_path=MODEL_PATH, conf=0.25, imgsz=640)

    # Run on a few dataset images and save predictions
    anpr.infer_images(images_dir=IMAGES_DIR, output_dir="anpr_preds", max_images=50)

    # (Optional) Run on a video
    # anpr.infer_video(source="videos/acar.mp4", output_path="anpr_output.mp4", display=False)
