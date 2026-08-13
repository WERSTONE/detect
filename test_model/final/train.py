"""Final three-task training entry.

This entry isolates the new final training flow from the older dual/domain
experiments. It keeps the current three-head model behavior but rebuilds data
routing, task-balanced batches, and four-class detect configuration.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.final.data import build_final_train_loader, build_final_val_loader
from test_model.final.model import create_final_model
from test_model.train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Final three-task trainer")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "test_model" / "final" / "yaml" / "final_three_head.yaml"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-weights", default=None)
    return parser.parse_args()


def normalize_device(device: str | None) -> str:
    device = str(device or "cuda")
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return device


def load_model_weights(model, path, device):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state = None
        for key in ("ema", "model"):
            obj = ckpt.get(key)
            if hasattr(obj, "state_dict"):
                state = obj.state_dict()
                break
        if state is None:
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")
    target = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    model.load_state_dict(matched, strict=False)
    print(f"[init] loaded {len(matched)}/{len(target)} compatible tensors from {path}")
    return ckpt


def set_trainable(model, trainable_roots: list[str] | None):
    if not trainable_roots:
        for param in model.parameters():
            param.requires_grad = True
        model._frozen_module_roots = []
        return
    roots = tuple(trainable_roots)
    for name, param in model.named_parameters():
        param.requires_grad = any(name == root or name.startswith(root + ".") for root in roots)
    model._frozen_module_roots = [
        name for name, _module in model.named_modules()
        if name and not any(name == root or name.startswith(root + ".") for root in roots)
    ]


def apply_stage(model, stage):
    model.train_domain_det = bool(stage.get("train_detect", True))
    model.train_det = model.train_domain_det
    model.train_attr = bool(stage.get("train_attr", True))
    model.train_pose = bool(stage.get("train_pose", True))
    model.set_task_weights(stage.get("det_weight", 1.0), stage.get("pose_weight", 1.0))
    model.set_attr_task_weight(stage.get("attr_weight", 1.0))
    freeze = stage.get("freeze", {}) or {}
    set_trainable(model, freeze.get("trainable_modules") if freeze.get("enabled", False) else None)


def build_model(cfg):
    data_cfg = cfg.get("data", {})
    domain_cfg = cfg["domain_det"]
    attr_cfg = cfg["attributes"]
    neck_cfg = cfg.get("neck", {}) or {}
    assigner_cfg = cfg.get("assigner", {}) or {}
    return create_final_model(
        name=cfg.get("model", "final_three_head"),
        domain_num_classes=domain_cfg.get("num_classes", 4),
        domain_class_map=domain_cfg.get("class_map", {}),
        num_attrs=attr_cfg.get("num_attrs", 4),
        attr_names=attr_cfg.get("names", None),
        num_kpts=cfg.get("num_kpts", 17),
        reg_max=cfg.get("reg_max", 16),
        input_size=data_cfg.get("input_size", 640),
        neck_use_p2_context=neck_cfg.get("use_p2_context", False),
        neck_downsample=neck_cfg.get("downsample", "conv"),
        neck_out_channels=neck_cfg.get("out_channels", None),
        assigner_topk=assigner_cfg.get("topk", 10),
        assigner_alpha=assigner_cfg.get("alpha", 0.5),
        assigner_beta=assigner_cfg.get("beta", 6.0),
        assigner_eps=assigner_cfg.get("eps", 1.0e-9),
    )


def build_trainer(model, cfg, device, stage, save_dir):
    train_cfg = cfg["training"]
    return Trainer(
        model,
        device=device,
        lr=stage.get("lr0", train_cfg.get("lr0", 0.001)),
        optimizer=train_cfg.get("optimizer", "sgd"),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        nesterov=train_cfg.get("nesterov", True),
        param_groups=train_cfg.get("param_groups", "yolo"),
        batch_size=train_cfg.get("batch_size", 32),
        nbs=train_cfg.get("nbs", 64),
        accumulate=train_cfg.get("accumulate", "auto"),
        scale_weight_decay=train_cfg.get("scale_weight_decay", True),
        cos_lr=stage.get("cos_lr", train_cfg.get("cos_lr", True)),
        final_lr_ratio=stage.get("lrf", train_cfg.get("lrf", 0.01)),
        warmup_epochs=stage.get("warmup_epochs", train_cfg.get("warmup_epochs", 0)),
        grad_clip=train_cfg.get("grad_clip", 10.0),
        log_interval=train_cfg.get("log_interval", 50),
        save_interval=train_cfg.get("save_interval", 10),
        val_interval=train_cfg.get("val_interval", 1),
        save_dir=save_dir,
        use_amp=train_cfg.get("amp", True),
        ema_decay=train_cfg.get("ema_decay", 0.9999),
        use_tensorboard=train_cfg.get("tensorboard", True),
        check_finite_loss=train_cfg.get("check_finite_loss", True),
        early_stop_enabled=stage.get(
            "early_stop_enabled", train_cfg.get("early_stop_enabled", False)),
        early_stop_patience=stage.get(
            "early_stop_patience", train_cfg.get("early_stop_patience", 0)),
        early_stop_min_delta=stage.get(
            "early_stop_min_delta", train_cfg.get("early_stop_min_delta", 0.0)),
        early_stop_start_epoch=stage.get(
            "early_stop_start_epoch", train_cfg.get("early_stop_start_epoch", 0)),
    )


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = normalize_device(args.device or cfg.get("device", "cuda"))
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(cfg)
    model.to(device)
    init_weights = args.init_weights or cfg["training"].get("init_weights")
    load_model_weights(model, init_weights, device)

    loss_cfg = cfg.get("loss", {})
    if hasattr(model, "set_attr_pos_weight"):
        names = cfg["attributes"].get("names", [])
        pos_cfg = loss_cfg.get("attr_pos_weight", {})
        pos_weight = [float(pos_cfg.get(name, 1.0)) for name in names]
        model.set_attr_pos_weight(pos_weight)
    model.attr_loss_weight = float(loss_cfg.get("w_attr", 1.0))

    train_loader = build_final_train_loader(cfg)
    val_loader = None
    if cfg.get("data", {}).get("val"):
        try:
            val_loader = build_final_val_loader(cfg)
        except RuntimeError as exc:
            print(f"[val] Skipping validation: {exc}")
    stages = cfg["training"].get("stages", [])
    if not stages:
        stages = [{"name": "single", "epochs": cfg["training"].get("epochs", 100)}]

    base_save = Path(cfg["training"].get("save_dir", "checkpoints/final_three_head"))
    last_best = None
    last_stage_best = None
    for index, stage_in in enumerate(stages, 1):
        stage = copy.deepcopy(stage_in)
        name = stage.get("name", f"stage{index}")
        apply_stage(model, stage)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print()
        print("=" * 70)
        print(
            f"Final stage {index}/{len(stages)}: {name} "
            f"epochs={stage.get('epochs', 1)} trainable={trainable/1e6:.2f}M/{total/1e6:.2f}M"
        )
        print(
            f"  train_detect={model.train_domain_det} train_attr={model.train_attr} "
            f"train_pose={model.train_pose} "
            f"w_det={model.det_task_weight} w_attr={model.attr_task_weight} "
            f"w_pose={model.pose_task_weight}"
        )
        print("=" * 70)

        trainer = build_trainer(model, cfg, device, stage, base_save / name)
        trainer.fit(
            epochs=int(stage.get("epochs", 1)),
            train_loader=train_loader,
            val_loader=val_loader,
            save_prefix=f"final_three_head_{name}",
            close_mosaic_epochs=0,
        )
        best = base_save / name / f"final_three_head_{name}_best.pt"
        if best.exists():
            last_stage_best = best
        if stage.get("load_best_for_next_stage", True) and best.exists():
            load_model_weights(model, best, device)
            last_best = best

    print(f"Final training complete. Last best: {last_best}")

    # Save the final model and run the post-training mAP/attribute report.
    final_out = base_save / "final_three_head_final.pt"
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": sum(int(s.get("epochs", 1)) for s in stages),
        "last_best": str(last_best) if last_best else None,
    }
    torch.save(state, str(final_out))
    print(f"Saved final model: {final_out}")

    if val_loader is not None and cfg.get("eval", {}).get("enabled", True):
        from test_model.final.eval import evaluate_checkpoint

        eval_weights = (
            last_stage_best if last_stage_best else final_out
        )
        evaluate_checkpoint(
            cfg,
            str(eval_weights),
            device=device,
            output_dir=cfg.get("eval", {}).get("output_dir"),
        )


if __name__ == "__main__":
    main()
