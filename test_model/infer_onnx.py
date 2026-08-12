"""Run inference with the exported BiFPN dual-head ONNX model.

The ONNX graph exports raw head tensors. This script contains the matching
preprocess, DFL decode, keypoint decode, class-aware NMS, coordinate restore,
and optional visualization.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "onnxruntime is required for ONNX inference. Install it with:\n"
        "  python -m pip install onnxruntime opencv-python numpy\n"
        "or for GPU runtime:\n"
        "  python -m pip install onnxruntime-gpu opencv-python numpy"
    ) from exc


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

COCO_KPT_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def letterbox_bgr(image_bgr, imgsz=640):
    """Resize and pad like the validation dataset: RGB/BGR agnostic padding."""
    h, w = image_bgr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    new_w, new_h = int(w * scale), int(h * scale)
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
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return tensor, scale, pad


def dfl_decode(reg, stride, reg_max=16):
    """Decode [1, 4*reg_max, H, W] DFL distribution to xyxy in 640-space."""
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


def decode_keypoints(kpt, stride, num_kpts=17):
    """Decode [1, 51, H, W] raw keypoints to [N, 17, 3] in 640-space."""
    _, _, h, w = kpt.shape
    raw = np.transpose(kpt[0], (1, 2, 0)).reshape(h * w, num_kpts, 3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    xy = (raw[..., :2] * 2.0 + grid[:, None, :]) * float(stride)
    conf = sigmoid(raw[..., 2:3])
    return np.concatenate([xy, conf], axis=2)


def decode_det_level(cls_logits, reg_logits, stride, score_thresh,
                     cls_offset=0, max_candidates=30000):
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
        "boxes": boxes[anchor_idx],
        "scores": selected_scores.astype(np.float32),
        "classes": (cls_idx + cls_offset).astype(np.int32),
        "kpts": np.zeros((len(anchor_idx), 17, 3), dtype=np.float32),
    }


def decode_pose_level(cls_logits, reg_logits, kpt_logits, stride, score_thresh,
                      max_candidates=30000):
    scores = sigmoid(cls_logits[0, 0]).reshape(-1)
    keep = np.where(scores > score_thresh)[0]
    if len(keep) == 0:
        return empty_preds()

    selected_scores = scores[keep]
    if max_candidates and len(selected_scores) > max_candidates:
        top = np.argsort(-selected_scores)[:max_candidates]
        keep = keep[top]
        selected_scores = selected_scores[top]

    boxes = dfl_decode(reg_logits, stride)
    kpts = decode_keypoints(kpt_logits, stride)
    return {
        "boxes": boxes[keep],
        "scores": selected_scores.astype(np.float32),
        "classes": np.zeros(len(keep), dtype=np.int32),
        "kpts": kpts[keep].astype(np.float32),
    }


def empty_preds():
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int32),
        "kpts": np.zeros((0, 17, 3), dtype=np.float32),
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


def class_aware_nms(pred, iou_thresh=0.6, max_det=300):
    if len(pred["scores"]) == 0:
        return pred
    keep_all = []
    for cls_id in np.unique(pred["classes"]):
        idx = np.where(pred["classes"] == cls_id)[0]
        keep = nms_numpy(pred["boxes"][idx], pred["scores"][idx], iou_thresh)
        keep_all.append(idx[keep])
    keep = np.concatenate(keep_all) if keep_all else np.zeros((0,), dtype=np.int64)
    keep = keep[np.argsort(-pred["scores"][keep])]
    if max_det:
        keep = keep[:max_det]
    return {key: value[keep] for key, value in pred.items()}


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


def decode_outputs(outputs, score_thresh=0.25, iou_thresh=0.6, max_det=300):
    names = [
        "det_cls_s8", "det_reg_s8",
        "det_cls_s16", "det_reg_s16",
        "det_cls_s32", "det_reg_s32",
        "pose_cls_s8", "pose_reg_s8", "pose_kpt_s8",
        "pose_cls_s16", "pose_reg_s16", "pose_kpt_s16",
        "pose_cls_s32", "pose_reg_s32", "pose_kpt_s32",
    ]
    out = dict(zip(names, outputs))
    parts = []
    for stride in (8, 16, 32):
        parts.append(decode_det_level(
            out[f"det_cls_s{stride}"],
            out[f"det_reg_s{stride}"],
            stride,
            score_thresh,
            cls_offset=1,
        ))
        parts.append(decode_pose_level(
            out[f"pose_cls_s{stride}"],
            out[f"pose_reg_s{stride}"],
            out[f"pose_kpt_s{stride}"],
            stride,
            score_thresh,
        ))
    return class_aware_nms(concat_preds(parts), iou_thresh=iou_thresh, max_det=max_det)


def draw_predictions(image_bgr, pred, score_thresh=0.25, kpt_thresh=0.25):
    canvas = image_bgr.copy()
    for box, score, cls_id, kpts in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"]):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int).tolist()
        cls_id = int(cls_id)
        color = (0, 180, 255) if cls_id == 0 else (80, 220, 80)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = COCO80_CLASSES[cls_id] if 0 <= cls_id < len(COCO80_CLASSES) else str(cls_id)
        cv2.putText(
            canvas,
            f"{name} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        if cls_id == 0 and np.any(kpts[:, 2] > kpt_thresh):
            for a, b in COCO_KPT_SKELETON:
                if kpts[a, 2] > kpt_thresh and kpts[b, 2] > kpt_thresh:
                    pa = tuple(kpts[a, :2].astype(int).tolist())
                    pb = tuple(kpts[b, :2].astype(int).tolist())
                    cv2.line(canvas, pa, pb, (255, 160, 0), 2)
            for x, y, conf in kpts:
                if conf > kpt_thresh:
                    cv2.circle(canvas, (int(x), int(y)), 3, (0, 80, 255), -1)
    return canvas


def predictions_to_jsonable(pred):
    items = []
    for box, score, cls_id, kpts in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"]):
        item = {
            "class_id": int(cls_id),
            "class_name": COCO80_CLASSES[int(cls_id)] if 0 <= int(cls_id) < len(COCO80_CLASSES) else str(cls_id),
            "score": float(score),
            "box_xyxy": [float(x) for x in box],
        }
        if int(cls_id) == 0:
            item["keypoints"] = [
                [float(x), float(y), float(conf)]
                for x, y, conf in kpts
            ]
        items.append(item)
    return items


def iter_images(path):
    path = Path(path)
    if path.is_file():
        yield path
        return
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for item in sorted(path.iterdir()):
        if item.suffix.lower() in exts:
            yield item


def parse_args():
    parser = argparse.ArgumentParser(description="BiFPN dual-head ONNX inference")
    parser.add_argument("--onnx", required=True, help="Path to bifpn_dual ONNX file")
    parser.add_argument("--image", required=True, help="Image file or directory")
    parser.add_argument("--output", default="outputs/onnx_infer", help="Output image or directory")
    parser.add_argument("--json", default=None, help="Optional JSON output path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def make_session(onnx_path, provider="auto"):
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "auto" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def main():
    args = parse_args()
    session = make_session(args.onnx, args.provider)
    input_name = session.get_inputs()[0].name
    image_paths = list(iter_images(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No image found: {args.image}")

    out_path = Path(args.output)
    single_file_output = len(image_paths) == 1 and out_path.suffix
    if not single_file_output:
        out_path.mkdir(parents=True, exist_ok=True)
    elif out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        inp, scale, pad = preprocess(image_bgr, args.imgsz)
        outputs = session.run(None, {input_name: inp})
        pred = decode_outputs(
            outputs,
            score_thresh=args.conf,
            iou_thresh=args.iou,
            max_det=args.max_det,
        )
        pred = restore_to_original(pred, scale, pad, image_bgr.shape[:2])
        all_results[str(image_path)] = predictions_to_jsonable(pred)

        vis = draw_predictions(image_bgr, pred, score_thresh=args.conf, kpt_thresh=args.kpt_conf)
        save_path = out_path if single_file_output else out_path / image_path.name
        if not cv2.imwrite(str(save_path), vis):
            raise RuntimeError(f"Failed to write output image: {save_path}")
        print(f"{image_path}: {len(pred['scores'])} predictions -> {save_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
