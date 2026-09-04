"""Standalone ONNX inference for final_pose_attr_expanded_detect.

Dependencies:
  python -m pip install onnxruntime opencv-python numpy

For GPU runtime, install onnxruntime-gpu instead of onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "onnxruntime is required. Install with:\n"
        "  python -m pip install onnxruntime opencv-python numpy\n"
        "or GPU runtime:\n"
        "  python -m pip install onnxruntime-gpu opencv-python numpy"
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
CLASS_NAMES = ["person", "puddle", "fire", "smoke", "other", "helmet", "head", "cigarette", "falling"]
ATTR_NAMES = ["smoking", "falling", "waving", "helmet_on"]
STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_KPTS = 17
COLORS = {
    0: (0, 185, 255),
    1: (255, 160, 30),
    2: (40, 70, 255),
    3: (160, 160, 160),
    4: (80, 220, 80),
    5: (60, 210, 255),
    6: (220, 120, 255),
    7: (255, 220, 80),
    8: (120, 255, 120),
}
COCO_KPT_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone expanded-detect ONNX inference")
    parser.add_argument("--onnx", required=True, help="Path to exported .onnx model")
    parser.add_argument("--source", default="0", help="Image, directory, video path, or camera index")
    parser.add_argument("--output", default="outputs/expanded_detect_onnx", help="Output image/video/directory")
    parser.add_argument("--json", default=None, help="Optional JSON output for image or image-directory mode")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL output for video/camera mode")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--decode-conf", type=float, default=0.01, help="Low pre-NMS decode threshold")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--domain-conf", type=float, default=0.25)
    parser.add_argument("--attr-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--view", action="store_true", help="Show video/camera preview window")
    parser.add_argument("--no-view", action="store_true", help="Disable default camera preview window")
    parser.add_argument("--mirror", action="store_true", help="Mirror camera input")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop video/camera after N frames")
    parser.add_argument("--hide-low-attrs", action="store_true", help="Only draw attributes above --attr-conf")
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


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


def preprocess(image_bgr, imgsz=640):
    padded_bgr, scale, pad = letterbox_bgr(image_bgr, imgsz)
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None])
    return tensor, scale, pad


def dfl_decode(reg, stride, reg_max=REG_MAX):
    _, _, h, w = reg.shape
    n = h * w
    reg = reg.reshape(1, 4, reg_max, n)
    probs = softmax(reg, axis=2)
    proj = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
    dist = np.sum(probs * proj, axis=2)[0].T * float(stride)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    centers = centers * float(stride) + 0.5 * float(stride)
    left_top = centers - dist[:, 0:2]
    right_bottom = centers + dist[:, 2:4]
    return np.concatenate([left_top, right_bottom], axis=1)


def decode_keypoints(kpt, stride, num_kpts=NUM_KPTS):
    _, _, h, w = kpt.shape
    raw = np.transpose(kpt[0], (1, 2, 0)).reshape(h * w, num_kpts, 3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    xy = (raw[..., :2] * 2.0 + grid[:, None, :]) * float(stride)
    conf = sigmoid(raw[..., 2:3])
    return np.concatenate([xy, conf], axis=2)


def empty_preds(num_attrs=len(ATTR_NAMES)):
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int32),
        "kpts": np.zeros((0, NUM_KPTS, 3), dtype=np.float32),
        "attrs": np.zeros((0, num_attrs), dtype=np.float32),
    }


def decode_det_level(cls_logits, reg_logits, stride, score_thresh, cls_offset=0, max_candidates=30000):
    scores = sigmoid(np.transpose(cls_logits[0], (1, 2, 0)).reshape(-1, cls_logits.shape[1]))
    boxes = dfl_decode(reg_logits, stride)
    anchor_idx, cls_idx = np.where(scores > score_thresh)
    if len(anchor_idx) == 0:
        return empty_preds()

    selected_scores = scores[anchor_idx, cls_idx]
    if max_candidates and len(selected_scores) > max_candidates:
        top = np.argsort(-selected_scores)[:max_candidates]
        anchor_idx = anchor_idx[top]
        cls_idx = cls_idx[top]
        selected_scores = selected_scores[top]

    return {
        "boxes": boxes[anchor_idx].astype(np.float32),
        "scores": selected_scores.astype(np.float32),
        "classes": (cls_idx + cls_offset).astype(np.int32),
        "kpts": np.zeros((len(anchor_idx), NUM_KPTS, 3), dtype=np.float32),
        "attrs": np.zeros((len(anchor_idx), len(ATTR_NAMES)), dtype=np.float32),
    }


def decode_pose_level(cls_logits, reg_logits, kpt_logits, attr_logits, stride, score_thresh, max_candidates=30000):
    scores = sigmoid(cls_logits[0, 0]).reshape(-1)
    keep = np.where(scores > score_thresh)[0]
    if len(keep) == 0:
        num_attrs = attr_logits.shape[1] if attr_logits is not None else len(ATTR_NAMES)
        return empty_preds(num_attrs)

    selected_scores = scores[keep]
    if max_candidates and len(selected_scores) > max_candidates:
        top = np.argsort(-selected_scores)[:max_candidates]
        keep = keep[top]
        selected_scores = selected_scores[top]

    boxes = dfl_decode(reg_logits, stride)
    kpts = decode_keypoints(kpt_logits, stride)
    if attr_logits is not None:
        attrs_all = sigmoid(np.transpose(attr_logits[0], (1, 2, 0)).reshape(-1, attr_logits.shape[1]))
        attrs = attrs_all[keep].astype(np.float32)
    else:
        attrs = np.zeros((len(keep), len(ATTR_NAMES)), dtype=np.float32)

    return {
        "boxes": boxes[keep].astype(np.float32),
        "scores": selected_scores.astype(np.float32),
        "classes": np.zeros(len(keep), dtype=np.int32),
        "kpts": kpts[keep].astype(np.float32),
        "attrs": attrs,
    }


def concat_preds(parts):
    non_empty = [p for p in parts if len(p["scores"]) > 0]
    if not non_empty:
        return empty_preds()
    return {
        "boxes": np.concatenate([p["boxes"] for p in non_empty], axis=0),
        "scores": np.concatenate([p["scores"] for p in non_empty], axis=0),
        "classes": np.concatenate([p["classes"] for p in non_empty], axis=0),
        "kpts": np.concatenate([p["kpts"] for p in non_empty], axis=0),
        "attrs": np.concatenate([p["attrs"] for p in non_empty], axis=0),
    }


def nms_numpy(boxes, scores, iou_thresh):
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-8
        order = rest[inter / union <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def decode_outputs(outputs, output_names, score_thresh=0.01, iou_thresh=0.6, max_det=300):
    names = list(output_names)
    if not names:
        names = [
            "domain_cls_s8", "domain_reg_s8",
            "domain_cls_s16", "domain_reg_s16",
            "domain_cls_s32", "domain_reg_s32",
            "pose_cls_s8", "pose_reg_s8", "pose_kpt_s8",
            "pose_cls_s16", "pose_reg_s16", "pose_kpt_s16",
            "pose_cls_s32", "pose_reg_s32", "pose_kpt_s32",
            "pose_attr_s8", "pose_attr_s16", "pose_attr_s32",
        ]
    out = dict(zip(names, outputs))

    domain_parts = []
    pose_parts = []
    for stride in STRIDES:
        domain_parts.append(decode_det_level(
            out[f"domain_cls_s{stride}"],
            out[f"domain_reg_s{stride}"],
            stride,
            score_thresh,
            cls_offset=1,
        ))
        attr = out.get(f"pose_attr_s{stride}", out.get(f"attr_s{stride}"))
        pose_parts.append(decode_pose_level(
            out[f"pose_cls_s{stride}"],
            out[f"pose_reg_s{stride}"],
            out[f"pose_kpt_s{stride}"],
            attr,
            stride,
            score_thresh,
        ))

    domain = concat_preds(domain_parts)
    pose = concat_preds(pose_parts)
    if len(domain["scores"]) == 0 and len(pose["scores"]) == 0:
        return empty_preds()

    boxes = np.concatenate([pose["boxes"], domain["boxes"]], axis=0)
    scores = np.concatenate([pose["scores"], domain["scores"]], axis=0)
    classes = np.concatenate([pose["classes"], domain["classes"]], axis=0)
    kpts = np.concatenate([pose["kpts"], domain["kpts"]], axis=0)
    attrs = np.concatenate([pose["attrs"], domain["attrs"]], axis=0)

    keep_parts = []
    for cls_id in np.unique(classes):
        idx = np.where(classes == cls_id)[0]
        keep_parts.append(idx[nms_numpy(boxes[idx], scores[idx], iou_thresh)])
    keep = np.concatenate(keep_parts) if keep_parts else np.zeros((0,), dtype=np.int64)
    keep = keep[np.argsort(-scores[keep])]
    if max_det:
        keep = keep[:max_det]

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "classes": classes[keep],
        "kpts": kpts[keep],
        "attrs": attrs[keep],
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
        pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]
    ):
        cls_id = int(cls_id)
        score = float(score)
        threshold = args.person_conf if cls_id == 0 else args.domain_conf
        if score < threshold:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = COLORS.get(cls_id, (80, 220, 80))
        label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        draw_label(canvas, f"{label} {score:.2f}", x1, y1 - 6, color)

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

        attr_lines = []
        for name, prob in zip(ATTR_NAMES, attrs):
            prob = float(prob)
            if args.hide_low_attrs and prob < args.attr_conf:
                continue
            marker = "*" if prob >= args.attr_conf else " "
            attr_lines.append(f"{marker}{name}: {prob:.2f}")
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
        pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]
    ):
        cls_id = int(cls_id)
        item = {
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
            "score": float(score),
            "box_xyxy": [float(v) for v in box],
        }
        if cls_id == 0:
            item["keypoints"] = [[float(x), float(y), float(conf)] for x, y, conf in kpts]
            item["attrs"] = {name: float(prob) for name, prob in zip(ATTR_NAMES, attrs)}
        items.append(item)
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


def make_session(onnx_path, provider="auto"):
    if provider in ("auto", "cuda") and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            print(f"WARNING: onnxruntime CUDA DLL preload failed: {exc}")
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "auto" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def predict_frame(session, input_name, output_names, frame_bgr, args):
    inp, scale, pad = preprocess(frame_bgr, args.imgsz)
    outputs = session.run(None, {input_name: inp})
    pred = decode_outputs(
        outputs,
        output_names,
        score_thresh=args.decode_conf,
        iou_thresh=args.iou,
        max_det=args.max_det,
    )
    return restore_to_original(pred, scale, pad, frame_bgr.shape[:2])


def run_images(session, input_name, output_names, args):
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
        pred = predict_frame(session, input_name, output_names, image_bgr, args)
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


def run_video_or_camera(session, input_name, output_names, args):
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
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width, height))
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

            pred = predict_frame(session, input_name, output_names, frame, args)
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
                cv2.imshow("expanded-detect ONNX", vis)
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
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = make_session(onnx_path, args.provider)
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    print(f"ONNX: {onnx_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Attributes: {', '.join(ATTR_NAMES)}")

    source = str(args.source)
    source_path = Path(source)
    if source_path.exists() and (source_path.is_dir() or source_path.suffix.lower() in IMAGE_EXTS):
        run_images(session, input_name, output_names, args)
    elif source_path.exists() and source_path.suffix.lower() in VIDEO_EXTS:
        run_video_or_camera(session, input_name, output_names, args)
    elif is_camera_source(source):
        run_video_or_camera(session, input_name, output_names, args)
    else:
        raise FileNotFoundError(f"Unsupported or missing source: {args.source}")


if __name__ == "__main__":
    main()
