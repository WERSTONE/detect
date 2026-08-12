"""Run image/video inference for the domain-detection + pose + attribute model."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from test_model.model import create_model


COCO_KPT_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize bifpn_dual_domain_attr predictions on an image or video")
    parser.add_argument("--config", required=True, help="Training yaml used to build the model")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--video", default=None, help="Input video path")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image or video path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25, help="Box confidence threshold")
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Drawing threshold for person boxes; defaults to --conf")
    parser.add_argument(
        "--domain-conf",
        type=float,
        default=None,
        help="Drawing threshold for fire/water boxes; defaults to --conf")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--attr-conf", type=float, default=0.5)
    parser.add_argument(
        "--show-all-attrs",
        action="store_true",
        help="Draw every attribute probability instead of positives only")
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Infer every Nth frame")
    parser.add_argument("--prefer-ema", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_kwargs_from_config(cfg):
    data_cfg = cfg.get("data", {}) or {}
    neck_cfg = cfg.get("neck", {}) or {}
    assigner_cfg = cfg.get("assigner", {}) or {}
    domain_cfg = cfg.get("domain_det", {}) or {}
    attr_cfg = cfg.get("attributes", {}) or {}
    return {
        "num_kpts": cfg.get("num_kpts", 17),
        "num_det_classes": cfg.get("num_det_classes", 79),
        "reg_max": cfg.get("reg_max", 16),
        "input_size": data_cfg.get("input_size", 640),
        "neck_use_p2_context": neck_cfg.get("use_p2_context", False),
        "neck_downsample": neck_cfg.get("downsample", "conv"),
        "neck_out_channels": neck_cfg.get("out_channels", None),
        "assigner_topk": assigner_cfg.get("topk", 10),
        "assigner_alpha": assigner_cfg.get("alpha", 0.5),
        "assigner_beta": assigner_cfg.get("beta", 6.0),
        "assigner_eps": assigner_cfg.get("eps", 1.0e-9),
        "domain_num_classes": domain_cfg.get("num_classes", 2),
        "domain_class_map": domain_cfg.get("class_map", None),
        "num_attrs": attr_cfg.get(
            "num_attrs",
            len(attr_cfg.get("names", [])) if attr_cfg.get("names") else 4,
        ),
        "attr_names": attr_cfg.get("names", None),
    }


def extract_state_dict(checkpoint, prefer_ema=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if prefer_ema and isinstance(checkpoint.get("ema_state"), dict):
        return checkpoint["ema_state"]
    for key in ("model_state_dict", "state_dict", "ema_state"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("Could not find a loadable state dict in checkpoint")


def letterbox_bgr(image_bgr, imgsz):
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


def preprocess(frame_bgr, imgsz, device):
    padded, scale, pad = letterbox_bgr(frame_bgr, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, scale, pad


def restore_prediction(pred, scale, pad, orig_shape):
    h, w = orig_shape[:2]
    pad_l, pad_t = pad
    out = {}
    for key, value in pred.items():
        out[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    if len(out["boxes"]):
        boxes = out["boxes"].astype(np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        out["boxes"] = boxes
    if "kpts" in out and len(out["kpts"]):
        kpts = out["kpts"].astype(np.float32)
        kpts[..., 0] = (kpts[..., 0] - pad_l) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_t) / scale
        kpts[..., 0] = np.clip(kpts[..., 0], 0, w - 1)
        kpts[..., 1] = np.clip(kpts[..., 1], 0, h - 1)
        out["kpts"] = kpts
    return out


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
        name = class_names.get(cls_id, str(cls_id))
        draw_label(canvas, f"{name} {float(score):.2f}", x1, y1 - 6, color)

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
        for name_i, prob in zip(attr_names, attrs):
            prob = float(prob)
            if show_all_attrs or prob >= attr_conf:
                attr_lines.append(f"{name_i}: {prob:.2f}")
        if attr_lines:
            line_h = 20
            top = y2 + 22
            if top + line_h * len(attr_lines) > canvas.shape[0]:
                top = max(22, y1 - 8 - line_h * (len(attr_lines) - 1))
            for idx, line in enumerate(attr_lines):
                draw_label(canvas, line, x1, top + idx * line_h, color, scale=0.5, thickness=1)
    return canvas


def main():
    args = parse_args()
    if bool(args.video) == bool(args.image):
        raise ValueError("Specify exactly one of --video or --image")
    cfg = load_config(args.config)
    imgsz = int(args.imgsz or (cfg.get("data", {}) or {}).get("input_size", 640))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA requested but unavailable; using CPU")

    model_name = cfg.get("model", "bifpn_dual_domain_attr")
    print(f"Creating model: {model_name}")
    model = create_model(model_name, **model_kwargs_from_config(cfg)).to(device)

    print(f"Loading weights: {args.weights}")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint, prefer_ema=args.prefer_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded tensors: {len(state)} | missing={len(missing)} "
        f"unexpected={len(unexpected)} | epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else 'n/a'}"
    )
    if missing:
        print(f"  missing sample: {missing[:5]}")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:5]}")
    model.eval()

    attr_names = list((cfg.get("attributes", {}) or {}).get(
        "names", getattr(model, "attr_names", ["smoking", "falling", "waving", "helmet_on"])))
    person_conf = args.conf if args.person_conf is None else args.person_conf
    domain_conf = args.conf if args.domain_conf is None else args.domain_conf
    print(
        f"Thresholds: decode={args.conf} person={person_conf} "
        f"domain={domain_conf} attr={args.attr_conf}"
    )

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"Cannot open image: {args.image}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            tensor, scale, pad = preprocess(frame, imgsz, device)
            pred = model.predict_val(
                tensor,
                score_thresh=args.conf,
                iou_thresh=args.iou,
                max_det=args.max_det,
            )[0]
            pred = restore_prediction(pred, scale, pad, frame.shape)
            vis = draw_predictions(
                frame,
                pred,
                attr_names=attr_names,
                conf=args.conf,
                person_conf=person_conf,
                domain_conf=domain_conf,
                kpt_conf=args.kpt_conf,
                attr_conf=args.attr_conf,
                show_all_attrs=args.show_all_attrs,
            )
        if not cv2.imwrite(str(output), vis):
            raise RuntimeError(f"Cannot write output image: {output}")
        print(f"Saved: {output}")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    last_vis = None
    frame_idx = 0
    written = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.limit_frames and written >= args.limit_frames:
                break
            if args.stride > 1 and (frame_idx - 1) % args.stride != 0 and last_vis is not None:
                writer.write(last_vis)
                written += 1
                continue

            tensor, scale, pad = preprocess(frame, imgsz, device)
            pred = model.predict_val(
                tensor,
                score_thresh=args.conf,
                iou_thresh=args.iou,
                max_det=args.max_det,
            )[0]
            pred = restore_prediction(pred, scale, pad, frame.shape)
            vis = draw_predictions(
                frame,
                pred,
                attr_names=attr_names,
                conf=args.conf,
                person_conf=person_conf,
                domain_conf=domain_conf,
                kpt_conf=args.kpt_conf,
                attr_conf=args.attr_conf,
                show_all_attrs=args.show_all_attrs,
            )
            writer.write(vis)
            last_vis = vis
            written += 1
            if written == 1 or written % 50 == 0:
                print(f"Processed {written} frames ({frame_idx}/{total or '?'})")

    cap.release()
    writer.release()
    print(f"Saved: {output} ({written} frames)")


if __name__ == "__main__":
    main()
