"""Generate comparison plots for beamer. Simple extraction — no split/merge tricks."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CHECKPOINTS = Path(r"D:\AI4PumpRoom\checkpoints")
PLOTS = CHECKPOINTS / "plots"
PLOTS.mkdir(exist_ok=True)

MODELS = [
    ("dual_head_one_stage",  "M-A (single)",  "#1f77b4"),
    ("dual_head_two_stage",  "M-A (2-stage) [*]", "#ff7f0e"),
    ("unified_head",         "M-B Unified",   "#2ca02c"),
    ("dual_neck",            "M-C DualNeck",  "#d62728"),
    ("attn_dual",            "M-D ECA",       "#9467bd"),
    ("bifpn_dual",           "M-E BiFPN",     "#8c564b"),
]


def extract_scalars(tb_path, prefix):
    ea = EventAccumulator(str(tb_path))
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        if tag.startswith(prefix):
            events = ea.Scalars(tag)
            out[tag] = {"steps": [e.step for e in events], "values": [e.value for e in events]}
    return out


def make_train_loss():
    fig, ax = plt.subplots(figsize=(8, 5))
    for tb_dir, label, color in MODELS:
        efs = list((CHECKPOINTS / tb_dir / "tensorboard").glob("events.out.*"))
        if not efs:
            continue
        data = extract_scalars(efs[0], "epoch/train_")
        tag = "epoch/train_total"
        if tag in data:
            s, v = data[tag]["steps"], data[tag]["values"]
            ax.plot(s, v, color=color, lw=1.5, label=label, alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
    ax.set_title("Training Total Loss")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(PLOTS / "train_total_loss.pdf"), dpi=150); plt.close()


def make_val_loss():
    fig, ax = plt.subplots(figsize=(8, 5))
    for tb_dir, label, color in MODELS:
        efs = list((CHECKPOINTS / tb_dir / "tensorboard").glob("events.out.*"))
        if not efs:
            continue
        data = extract_scalars(efs[0], "epoch/val_")
        tag = "epoch/val_total"
        if tag in data:
            s, v = data[tag]["steps"], data[tag]["values"]
            ax.plot(s, v, "o-", color=color, ms=3, lw=1.2, label=label, alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Total Loss")
    ax.set_title("Validation Total Loss")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(str(PLOTS / "val_total_loss.pdf"), dpi=150); plt.close()


def make_det_bar():
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = ["M-A\nsingle","M-A\n2-stage","M-B\nUnified","M-C\nDualNeck","M-D\nECA","M-E\nBiFPN",
              "Y8n-detect","Y8s-detect","Y8m-detect"]
    mAP50   = [57.8, 56.6, 52.6, 54.1, 57.6, 54.7, 54.9, 62.9, 68.7]
    mAP5095 = [38.2, 37.4, 34.0, 34.6, 38.1, 35.5, 38.3, 45.8, 51.4]
    pAP50   = [79.3, 78.3, 78.2, 77.6, 79.2, 78.8, 73.2, 77.7, 80.7]
    x = np.arange(len(labels)); w = 0.25
    ax.bar(x-w, mAP50,  w, label="mAP50",       color="#1f77b4", edgecolor="white")
    ax.bar(x,   mAP5095, w, label="mAP50-95",    color="#ff7f0e", edgecolor="white")
    ax.bar(x+w, pAP50,   w, label="Person AP50", color="#2ca02c", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mAP"); ax.set_title("Detection Performance"); ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 95)
    fig.tight_layout(); fig.savefig(str(PLOTS / "comparison_det.pdf"), dpi=150); plt.close()


def make_pose_bar():
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["M-A\nsingle","M-A\n2-stage","M-B\nUnified","M-C\nDualNeck","M-D\nECA","M-E\nBiFPN",
              "Y8n-pose","Y8s-pose","Y8m-pose"]
    ap50   = [62.6, 59.7, 66.1, 61.8, 63.4, 67.8, 83.7, 87.4, 89.6]
    ap5095 = [40.4, 37.4, 45.8, 38.5, 42.4, 45.9, 62.1, 70.3, 74.6]
    x = np.arange(len(labels)); w = 0.3
    ax.bar(x-w/2, ap50,   w, label="AP50",    color="#d62728", edgecolor="white")
    ax.bar(x+w/2, ap5095, w, label="AP50-95", color="#9467bd", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AP"); ax.set_title("Pose Estimation Performance"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(str(PLOTS / "comparison_pose.pdf"), dpi=150); plt.close()


def main():
    print("Generating plots...")
    make_train_loss()
    make_val_loss()
    make_det_bar()
    make_pose_bar()
    print("Done!")


if __name__ == "__main__":
    main()
