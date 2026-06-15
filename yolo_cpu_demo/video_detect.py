"""Video detection demo for YOLOv8 (CPU-friendly).

Reads a video, runs YOLO per-frame, and writes an annotated video.
Supports two modes:

- ``detect`` – object detection only (default model: yolov8n.pt)
- ``pose``   – pose estimation with behaviour analysis (falling / waving)
               (default model: yolov8n-pose.pt)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_cpu_demo.pose_postprocess import PoseAnalyzer, draw_behaviour_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 video detection")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save annotated video (mp4). Defaults to runs/<mode>/video_output.mp4",
    )
    parser.add_argument(
        "--mode",
        choices=["detect", "pose"],
        default="pose",
        help="Run mode: 'detect' for object detection, 'pose' for pose + behaviour analysis",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to YOLO model weights (auto-selected by --mode if omitted)",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (longest side)"
    )
    return parser.parse_args()


_DEFAULT_MODELS = {
    "detect": "yolo_cpu_demo/models/yolov8n.pt",
    "pose": "yolo_cpu_demo/models/yolov8n-pose.pt",
}


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model_path = args.model or _DEFAULT_MODELS[args.mode]
    output_path = Path(
        args.output or f"yolo_cpu_demo/runs/{args.mode}/video_output.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

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

    analyzer = PoseAnalyzer() if args.mode == "pose" else None

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
            r = results[0]
            annotated = r.plot()

            if args.mode == "pose" and analyzer is not None and r.keypoints is not None:
                kps_xy = r.keypoints.xy.cpu().numpy()
                kps_conf = r.keypoints.conf.cpu().numpy()
                boxes = r.boxes.xywh.cpu().numpy()

                behaviour = analyzer.analyze(kps_xy, kps_conf, boxes)
                annotated = draw_behaviour_labels(
                    annotated, behaviour, analyzer.trackers
                )

            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
