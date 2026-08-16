"""PyTorch webcam/image/video demo for the local-branch pose-attr model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final_pose_attr_local_branch.eval import build_model, normalize_device  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
CLASS_NAMES = {0: "person", 1: "puddle", 2: "fire", 3: "smoke", 4: "other"}
ATTR_NAMES = ["smoking", "falling", "waving", "helmet_on"]
COLORS = {
    0: (0, 185, 255),
    1: (255, 160, 30),
    2: (40, 70, 255),
    3: (160, 160, 160),
    4: (80, 220, 80),
}
COCO_KPT_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local-branch final_pose_attr PyTorch inference on images, video, or webcam")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model/final_pose_attr_local_branch/yaml/final_pose_attr_local_branch.yaml"),
    )
    parser.add_argument("--source", default="0", help="Image, directory, video path, or camera index")
    parser.add_argument("--output", default="outputs/final_pose_attr_local_branch_pt", help="Output image/video/directory")
    parser.add_argument("--json", default=None, help="Optional JSON output for image directory mode")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL output for video/camera mode")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cuda, 0, or cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--decode-conf", type=float, default=0.01, help="Low pre-NMS decode threshold")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--domain-conf", type=float, default=0.25)
    parser.add_argument("--attr-conf", "--attr-thresh", dest="attr_conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--show-all-attrs", action="store_true")
    parser.add_argument("--view", action="store_true", help="Show video/camera window")
    parser.add_argument("--no-view", action="store_true", help="Disable default camera preview window")
    parser.add_argument("--mirror", action="store_true", help="Mirror camera input")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop video/camera after N frames")
    return parser.parse_args()


def letterbox_bgr(image_bgr, imgsz=640):
    h, w = image_bgr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = imgsz - new_w
    pad_h = imgsz - new_h
    pad_l, pad_t = pad_w // 2, pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_t,
        pad_h - pad_t,
        pad_l,
        pad_w - pad_l,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (pad_l, pad_t)


def preprocess(image_bgr, imgsz, device, half=False):
    padded_bgr, scale, pad = letterbox_bgr(image_bgr, imgsz)
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)
    if half:
        tensor = tensor.half()
    return tensor, scale, pad


def tensor_pred_to_numpy(pred):
    return {
        key: value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in pred.items()
    }


def restore_to_original(pred, scale, pad, orig_shape):
    if len(pred["scores"]) == 0:
        return pred
    orig_h, orig_w = orig_shape
    pad_l, pad_t = pad
    boxes = pred["boxes"].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    kpts = pred["kpts"].copy()
    kpts[..., 0] = np.clip((kpts[..., 0] - pad_l) / scale, 0, orig_w)
    kpts[..., 1] = np.clip((kpts[..., 1] - pad_t) / scale, 0, orig_h)
    kpts[..., 2] = np.clip(kpts[..., 2], 0, 1)

    out = dict(pred)
    out["boxes"] = boxes
    out["kpts"] = kpts
    return out


@torch.inference_mode()
def predict_frame(model, frame_bgr, args, device):
    tensor, scale, pad = preprocess(frame_bgr, args.imgsz, device, args.half)
    pred = model.predict_val(
        tensor,
        score_thresh=args.decode_conf,
        iou_thresh=args.iou,
        max_det=args.max_det,
    )[0]
    pred = tensor_pred_to_numpy(pred)
    return restore_to_original(pred, scale, pad, frame_bgr.shape[:2])


def draw_label(canvas, text, x, y, color, scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(0, min(x, canvas.shape[1] - tw - 2)))
    y = int(max(th + 3, min(y, canvas.shape[0] - baseline - 2)))
    cv2.rectangle(canvas, (x, y - th - 3), (x + tw + 4, y + baseline + 2), color, -1)
    cv2.putText(canvas, text, (x + 2, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_predictions(frame, pred, args, fps=None):
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

        if kpts.size:
            for a, b in COCO_KPT_SKELETON:
                if kpts[a, 2] > args.kpt_conf and kpts[b, 2] > args.kpt_conf:
                    pa = tuple(kpts[a, :2].astype(int).tolist())
                    pb = tuple(kpts[b, :2].astype(int).tolist())
                    cv2.line(canvas, pa, pb, (255, 150, 0), 2)
            for x, y, conf in kpts:
                if conf > args.kpt_conf:
                    cv2.circle(canvas, (int(x), int(y)), 3, (0, 80, 255), -1)

        attr_lines = [f"{name}: {float(prob):.2f}" for name, prob in zip(ATTR_NAMES, attrs)]
        if attr_lines:
            line_h = 20
            top = y2 + 22
            if top + line_h * len(attr_lines) > canvas.shape[0]:
                top = max(22, y1 - 8 - line_h * (len(attr_lines) - 1))
            for idx, line in enumerate(attr_lines):
                draw_label(canvas, line, x1, top + idx * line_h, color, scale=0.5, thickness=1)

    if fps is not None:
        draw_label(canvas, f"FPS {fps:.1f}", 8, 24, (80, 220, 80), scale=0.6, thickness=2)
    return canvas


def predictions_to_jsonable(pred):
    items = []
    for box, score, cls_id, kpts, attrs in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]):
        attrs_obj = {name: float(prob) for name, prob in zip(ATTR_NAMES, attrs)}
        items.append({
            "box": [float(v) for v in box],
            "score": float(score),
            "class": int(cls_id),
            "class_name": CLASS_NAMES.get(int(cls_id), str(int(cls_id))),
            "keypoints": kpts.astype(float).tolist(),
            "attrs": attrs_obj,
        })
    return items


def iter_images(path):
    path = Path(path)
    if path.is_file():
        yield path
        return
    for item in sorted(path.iterdir()):
        if item.suffix.lower() in IMAGE_EXTS:
            yield item


def is_camera_source(source):
    return str(source).isdigit()


def open_capture(source, width, height):
    if is_camera_source(source):
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, True
    return cv2.VideoCapture(str(source)), False


def run_images(model, args, device):
    image_paths = list(iter_images(args.source))
    if not image_paths:
        raise FileNotFoundError(f"No image found: {args.source}")

    out_path = Path(args.output)
    single_output = len(image_paths) == 1 and out_path.suffix.lower() in IMAGE_EXTS
    if single_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        pred = predict_frame(model, image_bgr, args, device)
        all_results[str(image_path)] = predictions_to_jsonable(pred)
        vis = draw_predictions(image_bgr, pred, args)
        save_path = out_path if single_output else out_path / image_path.name
        if not cv2.imwrite(str(save_path), vis):
            raise RuntimeError(f"Failed to write image: {save_path}")
        shown = sum(
            float(s) >= (args.person_conf if int(c) == 0 else args.domain_conf)
            for s, c in zip(pred["scores"], pred["classes"])
        )
        print(f"{image_path}: {shown} shown predictions -> {save_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON saved: {json_path}")


def run_video_or_camera(model, args, device):
    cap, is_camera = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    out_path = Path(args.output)
    save_video = bool(out_path.suffix)
    writer = None
    if save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {out_path}")

    jsonl_file = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    show_window = (args.view or is_camera) and not args.no_view
    fps = 0.0
    last = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror and is_camera:
                frame = cv2.flip(frame, 1)

            pred = predict_frame(model, frame, args, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.perf_counter()
            dt = max(now - last, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            last = now

            if jsonl_file is not None:
                payload = {"frame": frame_idx, "predictions": predictions_to_jsonable(pred)}
                jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

            vis = draw_predictions(frame, pred, args, fps=fps)
            if writer is not None:
                writer.write(vis)
            if show_window:
                cv2.imshow("final_pose_attr_local_branch PyTorch", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        if show_window:
            cv2.destroyAllWindows()
    print(f"Processed frames: {frame_idx}")


def main():
    args = parse_args()
    weights = Path(args.weights)
    config = Path(args.config)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = normalize_device(args.device)
    torch.backends.cudnn.benchmark = device.startswith("cuda")
    model = build_model(cfg, str(weights), device)
    if args.half and device.startswith("cuda"):
        model.half()
    elif args.half:
        print("--half is ignored because CUDA is not active")
        args.half = False

    print(f"Weights: {weights}")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}")
    print(
        "Thresholds: "
        f"decode={args.decode_conf} person={args.person_conf} "
        f"domain={args.domain_conf} attr={args.attr_conf}")
    print("Press q or Esc to quit the preview window.")

    source = str(args.source)
    source_path = Path(source)
    if source_path.exists() and (source_path.is_dir() or source_path.suffix.lower() in IMAGE_EXTS):
        run_images(model, args, device)
    elif source_path.exists() and source_path.suffix.lower() in VIDEO_EXTS:
        run_video_or_camera(model, args, device)
    elif is_camera_source(source):
        run_video_or_camera(model, args, device)
    else:
        raise FileNotFoundError(f"Unsupported or missing source: {args.source}")


if __name__ == "__main__":
    main()
