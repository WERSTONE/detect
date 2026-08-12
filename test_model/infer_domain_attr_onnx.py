"""Run inference with exported bifpn_dual_domain_attr ONNX models.

This script keeps the runtime path independent from PyTorch. The ONNX graph
contains raw head outputs; preprocessing, DFL/keypoint decoding, attribute
sampling, NMS, JSON export, and visualization are implemented in NumPy/OpenCV.
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
    _, _, h, w = kpt.shape
    raw = np.transpose(kpt[0], (1, 2, 0)).reshape(h * w, num_kpts, 3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    xy = (raw[..., :2] * 2.0 + grid[:, None, :]) * float(stride)
    conf = sigmoid(raw[..., 2:3])
    return np.concatenate([xy, conf], axis=2)


def empty_preds():
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int32),
        "kpts": np.zeros((0, 17, 3), dtype=np.float32),
    }


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


def iter_images(path):
    path = Path(path)
    if path.is_file():
        yield path
        return
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for item in sorted(path.iterdir()):
        if item.suffix.lower() in exts:
            yield item


def sample_attrs_for_boxes(attr_maps, boxes, strides=(8, 16, 32), num_attrs=4):
    if len(boxes) == 0:
        return np.zeros((0, num_attrs), dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    max_side = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]).max(axis=1)
    attrs = []
    for i, box in enumerate(boxes):
        level = 0
        if max_side[i] >= strides[1] * 8:
            level = 2
        elif max_side[i] >= strides[0] * 8:
            level = 1
        feat = attr_maps[level][0]
        _, h, w = feat.shape
        gx = int(np.clip(centers[i, 0] / strides[level], 0, w - 1))
        gy = int(np.clip(centers[i, 1] / strides[level], 0, h - 1))
        attrs.append(sigmoid(feat[:, gy, gx]))
    return np.stack(attrs).astype(np.float32) if attrs else np.zeros((0, num_attrs), dtype=np.float32)


def draw_label(canvas, text, x, y, color, scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(0, min(x, canvas.shape[1] - tw - 2)))
    y = int(max(th + 3, min(y, canvas.shape[0] - baseline - 2)))
    cv2.rectangle(canvas, (x, y - th - 3), (x + tw + 4, y + baseline + 2), color, -1)
    cv2.putText(canvas, text, (x + 2, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def draw_predictions(frame, pred, attr_names, conf, person_conf, domain_conf,
                     kpt_conf, attr_conf, show_all_attrs):
    canvas = frame.copy()
    class_names = {0: "person", 1: "fire", 2: "water"}
    colors = {0: (0, 185, 255), 1: (40, 70, 255), 2: (255, 160, 30)}

    for box, score, cls_id, kpts, attrs in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]):
        if float(score) < conf:
            continue
        cls_id = int(cls_id)
        class_conf = person_conf if cls_id == 0 else domain_conf
        if float(score) < class_conf:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = colors.get(cls_id, (80, 220, 80))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        draw_label(canvas, f"{class_names.get(cls_id, str(cls_id))} {float(score):.2f}", x1, y1 - 6, color)

        if cls_id != 0:
            continue
        if kpts.size and np.any(kpts[:, 2] > kpt_conf):
            for a, b in COCO_KPT_SKELETON:
                if kpts[a, 2] > kpt_conf and kpts[b, 2] > kpt_conf:
                    pa = tuple(kpts[a, :2].astype(int).tolist())
                    pb = tuple(kpts[b, :2].astype(int).tolist())
                    cv2.line(canvas, pa, pb, (255, 150, 0), 2)
            for x, y, kconf in kpts:
                if kconf > kpt_conf:
                    cv2.circle(canvas, (int(x), int(y)), 3, (0, 80, 255), -1)

        attr_lines = []
        for name, prob in zip(attr_names, attrs):
            prob = float(prob)
            if show_all_attrs or prob >= attr_conf:
                attr_lines.append(f"{name}: {prob:.2f}")
        if attr_lines:
            line_h = 20
            top = y2 + 22
            if top + line_h * len(attr_lines) > canvas.shape[0]:
                top = max(22, y1 - 8 - line_h * (len(attr_lines) - 1))
            for idx, line in enumerate(attr_lines):
                draw_label(canvas, line, x1, top + idx * line_h, color, scale=0.5, thickness=1)
    return canvas


def decode_domain_attr_outputs(outputs, score_thresh=0.25, iou_thresh=0.6, max_det=300):
    names = [
        "domain_cls_s8", "domain_reg_s8",
        "domain_cls_s16", "domain_reg_s16",
        "domain_cls_s32", "domain_reg_s32",
        "pose_cls_s8", "pose_reg_s8", "pose_kpt_s8",
        "pose_cls_s16", "pose_reg_s16", "pose_kpt_s16",
        "pose_cls_s32", "pose_reg_s32", "pose_kpt_s32",
        "attr_s8", "attr_s16", "attr_s32",
    ]
    out = dict(zip(names, outputs))
    domain_parts = []
    pose_parts = []
    for stride in (8, 16, 32):
        domain_parts.append(decode_det_level(
            out[f"domain_cls_s{stride}"],
            out[f"domain_reg_s{stride}"],
            stride,
            score_thresh,
            cls_offset=1,
        ))
        pose_parts.append(decode_pose_level(
            out[f"pose_cls_s{stride}"],
            out[f"pose_reg_s{stride}"],
            out[f"pose_kpt_s{stride}"],
            stride,
            score_thresh,
        ))

    domain = concat_preds(domain_parts)
    pose = concat_preds(pose_parts)

    attr_maps = [out["attr_s8"], out["attr_s16"], out["attr_s32"]]
    pose_attrs = sample_attrs_for_boxes(attr_maps, pose["boxes"], num_attrs=attr_maps[0].shape[1])
    domain_attrs = np.zeros((domain["boxes"].shape[0], pose_attrs.shape[1] if len(pose_attrs) else attr_maps[0].shape[1]), dtype=np.float32)

    boxes = np.concatenate([pose["boxes"], domain["boxes"]], axis=0) if len(domain["boxes"]) or len(pose["boxes"]) else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate([pose["scores"], domain["scores"]], axis=0) if len(domain["scores"]) or len(pose["scores"]) else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate([pose["classes"], domain["classes"]], axis=0) if len(domain["classes"]) or len(pose["classes"]) else np.zeros((0,), dtype=np.int32)
    kpts = np.concatenate([pose["kpts"], domain["kpts"]], axis=0) if len(domain["kpts"]) or len(pose["kpts"]) else np.zeros((0, 17, 3), dtype=np.float32)
    attrs = np.concatenate([pose_attrs, domain_attrs], axis=0) if len(domain_attrs) or len(pose_attrs) else np.zeros((0, attr_maps[0].shape[1]), dtype=np.float32)

    if len(scores) == 0:
        pred = empty_preds()
        pred["attrs"] = np.zeros((0, attr_maps[0].shape[1]), dtype=np.float32)
        return pred

    class_keep = []
    for cls_id in np.unique(classes):
        idx = np.where(classes == cls_id)[0]
        cls_boxes = boxes[idx]
        cls_scores = scores[idx]
        if len(cls_scores):
            keep_local = nms_numpy(cls_boxes, cls_scores, iou_thresh)
            class_keep.append(idx[keep_local])
    if class_keep:
        keep_idx = np.concatenate(class_keep)
        keep_idx = keep_idx[np.argsort(-scores[keep_idx])]
        if max_det:
            keep_idx = keep_idx[:max_det]
    else:
        keep_idx = np.zeros((0,), dtype=np.int64)

    return {
        "boxes": boxes[keep_idx],
        "scores": scores[keep_idx],
        "classes": classes[keep_idx],
        "kpts": kpts[keep_idx],
        "attrs": attrs[keep_idx],
    }


def predictions_to_jsonable(pred, attr_names):
    items = []
    for box, score, cls_id, kpts, attrs in zip(
            pred["boxes"], pred["scores"], pred["classes"], pred["kpts"], pred["attrs"]):
        item = {
            "class_id": int(cls_id),
            "class_name": {0: "person", 1: "fire", 2: "water"}.get(int(cls_id), str(cls_id)),
            "score": float(score),
            "box_xyxy": [float(x) for x in box],
        }
        if int(cls_id) == 0:
            item["keypoints"] = [[float(x), float(y), float(conf)] for x, y, conf in kpts]
            item["attrs"] = {name: float(prob) for name, prob in zip(attr_names, attrs)}
        items.append(item)
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="BiFPN domain-attr ONNX inference")
    parser.add_argument("--onnx", required=True, help="Path to bifpn_dual_domain_attr ONNX file")
    parser.add_argument("--image", required=True, help="Image file or directory")
    parser.add_argument("--output", default="outputs/onnx_infer_domain_attr", help="Output image or directory")
    parser.add_argument("--json", default=None, help="Optional JSON output path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--person-conf", type=float, default=0.5)
    parser.add_argument("--domain-conf", type=float, default=0.25)
    parser.add_argument("--attr-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--show-all-attrs", action="store_true")
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

    attr_names = ["smoking", "falling", "waving", "helmet_on"]
    all_results = {}
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        inp, scale, pad = preprocess(image_bgr, args.imgsz)
        outputs = session.run(None, {input_name: inp})
        pred = decode_domain_attr_outputs(
            outputs,
            score_thresh=args.conf,
            iou_thresh=args.iou,
            max_det=args.max_det,
        )
        pred = restore_to_original(pred, scale, pad, image_bgr.shape[:2])
        all_results[str(image_path)] = predictions_to_jsonable(pred, attr_names)

        vis = draw_predictions(
            image_bgr,
            pred,
            attr_names=attr_names,
            conf=args.conf,
            person_conf=args.person_conf,
            domain_conf=args.domain_conf,
            kpt_conf=args.kpt_conf,
            attr_conf=args.attr_conf,
            show_all_attrs=args.show_all_attrs,
        )
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
