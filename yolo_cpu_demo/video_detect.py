"""Video detection demo for YOLOv8 (CPU-friendly).

Reads a video, runs YOLO per-frame, and writes an annotated video.
Designed to work with the local weights in ``yolo_cpu_demo/models`` and
run on CPU by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 video detection")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default="yolo_cpu_demo/runs/detect/video_output.mp4",
        help="Path to save annotated video (mp4)",
    )
    parser.add_argument(
        "--model",
        default="yolo_cpu_demo/models/yolov8n.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (longest side)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer at: {output_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(
                frame,
                device=args.device,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            annotated = results[0].plot()
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
