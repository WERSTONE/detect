"""Realtime webcam demo for the final three-head ONNX model."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.infer_domain_attr_onnx import (
    COCO_KPT_SKELETON,
    decode_domain_attr_outputs,
    make_session,
    preprocess,
    restore_to_original,
)


CLASS_NAMES = {
    0: "person",
    1: "puddle",
    2: "fire",
    3: "smoke",
    4: "other",
}
ATTR_NAMES = ["smoking", "falling", "waving", "helmet_on"]
COLORS = {
    0: (0, 185, 255),
    1: (255, 160, 30),
    2: (40, 70, 255),
    3: (160, 160, 160),
    4: (80, 220, 80),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run realtime webcam inference with final_three_head ONNX")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model")
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--decode-conf", type=float, default=0.01)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--domain-conf", type=float, default=0.25)
    parser.add_argument("--attr-conf", type=float, default=0.5)
    parser.add_argument(
        "--attr-sampling",
        default="points-max",
        choices=[
            "center",
            "points-mean",
            "points-max",
            "box-mean",
            "box-max",
            "upper-box-mean",
            "upper-box-max",
        ],
        help="How to sample attribute logits from the attr feature maps")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--show-all-attrs", action="store_true")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam display")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save", default=None, help="Optional output video path")
    return parser.parse_args()


def camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def open_capture(source, width, height):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    cap = cv2.VideoCapture(source)
    if isinstance(source, int) and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_label(canvas, text, x, y, color, scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(0, min(x, canvas.shape[1] - tw - 2)))
    y = int(max(th + 3, min(y, canvas.shape[0] - baseline - 2)))
    cv2.rectangle(canvas, (x, y - th - 3), (x + tw + 4, y + baseline + 2), color, -1)
    cv2.putText(canvas, text, (x + 2, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_predictions(frame, pred, args, fps):
    canvas = frame.copy()
    for box, score, cls_id, kpts, attrs in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]):
        cls_id = int(cls_id)
        score = float(score)
        threshold = args.person_conf if cls_id == 0 else args.domain_conf
        if score < threshold:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = COLORS.get(cls_id, (80, 220, 80))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        draw_label(canvas, f"{CLASS_NAMES.get(cls_id, str(cls_id))} {score:.2f}", x1, y1 - 6, color)

        if cls_id != 0:
            continue

        if kpts.size and np.any(kpts[:, 2] > args.kpt_conf):
            for a, b in COCO_KPT_SKELETON:
                if kpts[a, 2] > args.kpt_conf and kpts[b, 2] > args.kpt_conf:
                    pa = tuple(kpts[a, :2].astype(int).tolist())
                    pb = tuple(kpts[b, :2].astype(int).tolist())
                    cv2.line(canvas, pa, pb, (255, 150, 0), 2)
            for x, y, kconf in kpts:
                if kconf > args.kpt_conf:
                    cv2.circle(canvas, (int(x), int(y)), 3, (0, 80, 255), -1)

        attr_lines = []
        for name, prob in zip(ATTR_NAMES, attrs):
            prob = float(prob)
            if args.show_all_attrs or prob >= args.attr_conf:
                attr_lines.append(f"{name}: {prob:.2f}")
        if attr_lines:
            line_h = 20
            top = y2 + 22
            if top + line_h * len(attr_lines) > canvas.shape[0]:
                top = max(22, y1 - 8 - line_h * (len(attr_lines) - 1))
            for idx, line in enumerate(attr_lines):
                draw_label(canvas, line, x1, top + idx * line_h, color, scale=0.5, thickness=1)

    draw_label(canvas, f"FPS {fps:.1f}", 8, 24, (80, 220, 80), scale=0.6, thickness=2)
    return canvas


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = make_session(onnx_path, args.provider)
    input_name = session.get_inputs()[0].name
    print(f"ONNX: {onnx_path}")
    print(f"Providers: {session.get_providers()}")
    print(
        "Thresholds: "
        f"decode={args.decode_conf} person={args.person_conf} "
        f"domain={args.domain_conf} attr={args.attr_conf} "
        f"attr_sampling={args.attr_sampling}")

    source = camera_source(args.camera)
    cap = open_capture(source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {args.camera}")

    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {out_path}")

    fps = 0.0
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror and isinstance(source, int):
                frame = cv2.flip(frame, 1)

            inp, scale, pad = preprocess(frame, args.imgsz)
            outputs = session.run(None, {input_name: inp})
            pred = decode_domain_attr_outputs(
                outputs,
                score_thresh=args.decode_conf,
                iou_thresh=args.iou,
                max_det=args.max_det,
                attr_sampling=args.attr_sampling,
            )
            pred = restore_to_original(pred, scale, pad, frame.shape[:2])

            now = time.perf_counter()
            dt = max(now - last, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            last = now

            vis = draw_predictions(frame, pred, args, fps)
            if writer is not None:
                writer.write(vis)
            cv2.imshow("final_three_head ONNX webcam", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
