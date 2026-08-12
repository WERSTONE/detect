"""Sample images and split person-containing frames by predicted attributes."""

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.infer_domain_attr_video import (  # noqa: E402
    extract_state_dict,
    letterbox_bgr,
    model_kwargs_from_config,
)
from test_model.model import create_model  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_ATTRS = {
    "hat": "helmet_on",
    "smoking": "smoking",
    "falling": "falling",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample source images, run bifpn domain-attr inference, and copy original images into attr folders."
    )
    parser.add_argument("--input", required=True, help="Image directory or dataset root containing an images folder")
    parser.add_argument("--weights", required=True, help="bifpn_dual_domain_attr checkpoint")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model/config/bifpn_domain_attr_finetune.yaml"),
        help="Model config YAML",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/domain_attr_filtered"),
        help="Output root. Created with hat/smoking/falling/rest subfolders.",
    )
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of images to sample at most")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--batch", type=int, default=2, help="Inference batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override model input size")
    parser.add_argument("--score-thresh", type=float, default=0.01, help="Decode score threshold")
    parser.add_argument("--person-conf", type=float, default=0.25, help="Minimum person score to keep an image")
    parser.add_argument("--attr-conf", type=float, default=0.5, help="Minimum attribute probability")
    parser.add_argument("--hat-conf", type=float, default=None, help="Override helmet_on probability threshold")
    parser.add_argument("--smoking-conf", type=float, default=None, help="Override smoking probability threshold")
    parser.add_argument("--falling-conf", type=float, default=None, help="Override falling probability threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--prefer-ema", action="store_true")
    parser.add_argument(
        "--exclude-list",
        default=None,
        help="Text file containing source image paths that should not be sampled again.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix for this run's manifest/summary files when appending.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append copied images and manifest rows to an existing output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory. Existing files are not deleted.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_image_dir(path):
    root = Path(path)
    images = root / "images"
    if images.is_dir():
        return images
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Input image directory does not exist: {path}")


def list_images(image_dir):
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def discrete_sample(paths, sample_size):
    if sample_size <= 0 or len(paths) <= sample_size:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, sample_size, dtype=np.int64)
    return [paths[int(i)] for i in indices]


def load_excluded_paths(path):
    if not path:
        return set()
    excluded = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                excluded.add(str(Path(value)))
    return excluded


def normalize_device(device):
    device = str(device or "cuda")
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def load_model(cfg, weights, device, prefer_ema):
    model_name = cfg.get("model", "bifpn_dual_domain_attr")
    model = create_model(model_name, **model_kwargs_from_config(cfg)).to(device)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint, prefer_ema=prefer_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded weights: tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else 'n/a'}",
        flush=True,
    )
    if missing:
        print(f"  missing sample: {missing[:5]}", flush=True)
    if unexpected:
        print(f"  unexpected sample: {unexpected[:5]}", flush=True)
    model.eval()
    return model


def preprocess_batch(image_paths, imgsz, device):
    tensors = []
    frames = []
    readable_paths = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"WARNING: cannot read image, skipped: {path}", flush=True)
            continue
        padded, _scale, _pad = letterbox_bgr(frame, imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensors.append(tensor)
        frames.append(frame)
        readable_paths.append(path)
    if not tensors:
        return None, [], []
    return torch.stack(tensors).to(device, non_blocking=True).contiguous(), frames, readable_paths


def safe_output_name(path, image_dir, used_names):
    rel = path.relative_to(image_dir)
    stem = "__".join(rel.with_suffix("").parts)
    name = f"{stem}{path.suffix.lower()}"
    if name not in used_names:
        used_names.add(name)
        return name
    idx = 2
    while True:
        candidate = f"{stem}__{idx}{path.suffix.lower()}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def copy_to_categories(src, output_name, categories, output_root):
    for category in categories:
        shutil.copy2(src, output_root / category / output_name)


def summarize_person_attrs(pred, attr_names, person_conf, attr_thresholds):
    classes = pred["classes"].detach().cpu().numpy().astype(int)
    scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
    attrs = pred.get("attrs")
    if attrs is None:
        attrs_np = np.zeros((len(classes), len(attr_names)), dtype=np.float32)
    else:
        attrs_np = attrs.detach().cpu().numpy().astype(np.float32)

    person_mask = (classes == 0) & (scores >= person_conf)
    person_indices = np.flatnonzero(person_mask)
    if len(person_indices) == 0:
        return None

    attr_probs = {}
    for attr_name in attr_names:
        attr_probs[attr_name] = 0.0
    for attr_idx, attr_name in enumerate(attr_names):
        if attr_idx < attrs_np.shape[1]:
            attr_probs[attr_name] = float(attrs_np[person_indices, attr_idx].max())

    categories = []
    for category, attr_name in TARGET_ATTRS.items():
        if attr_probs.get(attr_name, 0.0) >= attr_thresholds.get(attr_name, 0.5):
            categories.append(category)
    if not categories:
        categories = ["rest"]

    return {
        "person_count": int(len(person_indices)),
        "max_person_score": float(scores[person_indices].max()),
        "attr_probs": attr_probs,
        "categories": categories,
    }


def prepare_output(output_root, overwrite, append):
    if output_root.exists() and any(output_root.iterdir()) and not (overwrite or append):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use --append, --overwrite, or choose a new --output."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for folder in ["hat", "smoking", "falling", "rest"]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)


def existing_output_names(output_root):
    names = set()
    for folder in ["hat", "smoking", "falling", "rest"]:
        folder_path = output_root / folder
        if folder_path.exists():
            names.update(p.name for p in folder_path.iterdir() if p.is_file())
    return names


def append_csv(path, rows, fieldnames):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    image_dir = resolve_image_dir(args.input)
    all_images = list_images(image_dir)
    excluded_paths = load_excluded_paths(args.exclude_list)
    candidate_images = [path for path in all_images if str(path) not in excluded_paths]
    sampled = discrete_sample(candidate_images, args.sample_size)
    output_root = Path(args.output)
    prepare_output(output_root, args.overwrite, args.append)

    cfg = load_config(args.config)
    imgsz = int(args.imgsz or (cfg.get("data", {}) or {}).get("input_size", 640))
    attr_names = list((cfg.get("attributes", {}) or {}).get("names", ["smoking", "falling", "waving", "helmet_on"]))
    attr_thresholds = {
        "smoking": args.attr_conf if args.smoking_conf is None else args.smoking_conf,
        "falling": args.attr_conf if args.falling_conf is None else args.falling_conf,
        "helmet_on": args.attr_conf if args.hat_conf is None else args.hat_conf,
    }
    device = normalize_device(args.device)
    model = load_model(cfg, args.weights, device, prefer_ema=args.prefer_ema)

    manifest_rows = []
    no_person = []
    used_names = existing_output_names(output_root) if args.append else set()
    counts = {"hat": 0, "smoking": 0, "falling": 0, "rest": 0}
    processed = 0
    kept = 0
    total_batches = math.ceil(len(sampled) / max(args.batch, 1))

    run_suffix = f"_{args.run_name}" if args.run_name else ""
    sampled_file = output_root / f"sampled_images{run_suffix}.txt"
    no_person_file = output_root / f"excluded_no_person{run_suffix}.txt"
    summary_file = output_root / f"summary{run_suffix}.json"
    manifest_file = output_root / f"manifest{run_suffix}.csv"

    with open(sampled_file, "w", encoding="utf-8", newline="\n") as f:
        for path in sampled:
            f.write(str(path) + "\n")

    with torch.inference_mode():
        for batch_idx in range(total_batches):
            start = batch_idx * args.batch
            batch_paths = sampled[start:start + args.batch]
            tensor, _frames, readable_paths = preprocess_batch(batch_paths, imgsz, device)
            if tensor is None:
                continue
            preds = model.predict_val(
                tensor,
                score_thresh=args.score_thresh,
                iou_thresh=args.iou,
                max_det=args.max_det,
            )
            for path, pred in zip(readable_paths, preds):
                processed += 1
                summary = summarize_person_attrs(pred, attr_names, args.person_conf, attr_thresholds)
                if summary is None:
                    no_person.append(str(path))
                    continue

                output_name = safe_output_name(path, image_dir, used_names)
                copy_to_categories(path, output_name, summary["categories"], output_root)
                kept += 1
                for category in summary["categories"]:
                    counts[category] += 1
                manifest_rows.append({
                    "source": str(path),
                    "output_name": output_name,
                    "categories": "|".join(summary["categories"]),
                    "person_count": summary["person_count"],
                    "max_person_score": f"{summary['max_person_score']:.6f}",
                    "smoking": f"{summary['attr_probs'].get('smoking', 0.0):.6f}",
                    "falling": f"{summary['attr_probs'].get('falling', 0.0):.6f}",
                    "helmet_on": f"{summary['attr_probs'].get('helmet_on', 0.0):.6f}",
                    "waving": f"{summary['attr_probs'].get('waving', 0.0):.6f}",
                })

            if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or batch_idx + 1 == total_batches:
                print(
                    f"Batch {batch_idx + 1}/{total_batches}: processed={processed} kept={kept} "
                    f"hat={counts['hat']} smoking={counts['smoking']} "
                    f"falling={counts['falling']} rest={counts['rest']}",
                    flush=True,
                )

    fieldnames = [
        "source",
        "output_name",
        "categories",
        "person_count",
        "max_person_score",
        "smoking",
        "falling",
        "helmet_on",
        "waving",
    ]
    append_csv(manifest_file, manifest_rows, fieldnames)
    if args.append:
        append_csv(output_root / "manifest.csv", manifest_rows, fieldnames)

    with open(no_person_file, "w", encoding="utf-8", newline="\n") as f:
        for path in no_person:
            f.write(path + "\n")

    summary = {
        "input": str(image_dir),
        "weights": str(args.weights),
        "config": str(args.config),
        "output": str(output_root),
        "total_images": len(all_images),
        "candidate_images": len(candidate_images),
        "excluded_from_sampling": len(excluded_paths),
        "sampled_images": len(sampled),
        "processed_images": processed,
        "kept_person_images": kept,
        "excluded_no_person": len(no_person),
        "counts": counts,
        "thresholds": {
            "score_thresh": args.score_thresh,
            "person_conf": args.person_conf,
            "attr_conf": args.attr_conf,
            "attr_thresholds": attr_thresholds,
            "iou": args.iou,
        },
        "note": "Images with multiple positive attributes are copied into each matching attribute folder.",
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.append:
        with open(output_root / "summary_latest.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
