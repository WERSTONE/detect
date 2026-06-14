"""
Vigil training entry point.

Usage:
    python -m Vigil.train.train --stage pretrain
    python -m Vigil.train.train --stage finetune
    vigil-train --stage pretrain --model vigil_v2 --config config/train.yaml
"""

import argparse
import copy
import os
import sys

import torch
import yaml

from Vigil.models.registry import create_model
from Vigil.train.dataset import make_dataloaders
from Vigil.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="pretrain",
                        choices=["coco_pose_pretrain", "pretrain", "finetune"])
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--model", default=None, help="覆盖 yaml 中的 model.name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained", default=None, help="覆盖 stage 的 pretrained 权重路径")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    stage_cfg = cfg[args.stage]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    amp_cfg = cfg.get("amp", {})
    cudnn_cfg = cfg.get("cudnn", {})
    dl_cfg = cfg.get("dataloader", {})

    # ── cuDNN ──
    if cudnn_cfg.get("benchmark", False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("  cudnn.benchmark: ON")

    device = args.device or train_cfg.get("device", "cpu")
    epochs = args.epochs or stage_cfg.get("epochs", 50)
    lr = args.lr or stage_cfg.get("optimizer", {}).get("lr", 1e-3)
    batch = args.batch or train_cfg.get("batch_size", 1)
    warmup = stage_cfg.get("optimizer", {}).get("warmup_epochs", 1)
    wd = stage_cfg.get("optimizer", {}).get("weight_decay", 1e-4)
    grad_clip = train_cfg.get("grad_clip", 20.0)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 10)
    val_interval = train_cfg.get("val_interval", 1)
    save_best_by = str(train_cfg.get("save_best_by", "loss")).lower()
    if save_best_by in ("val_loss", "val_total"):
        save_best_by = "loss"
    if save_best_by in ("map", "map@0.5", "map50"):
        save_best_by = "score"
    if save_best_by not in ("loss", "score"):
        raise ValueError("training.save_best_by must be 'loss' or 'score'")
    map_cfg = cfg.get("map", {})
    map_enabled = map_cfg.get("enabled", False)
    if save_best_by == "score" and not map_enabled:
        raise ValueError("training.save_best_by='score' requires map.enabled=true in train.yaml")
    tb_cfg = cfg.get("tensorboard", {})
    use_tb = tb_cfg.get("enabled", False)
    tb_log_dir = tb_cfg.get("log_dir", "logs/train_logs")
    freeze = stage_cfg.get("freeze", [])

    model_name = args.model or model_cfg.get("name", "vigil_v2")
    save_dir = stage_cfg.get("output", f"checkpoints/{model_name}")
    model_kwargs = copy.deepcopy(model_cfg.get("kwargs", {}))
    model_kwargs.update(stage_cfg.get("model_kwargs", {}))

    pretrained = args.pretrained or stage_cfg.get("pretrained")
    # 自动串联训练阶段: pretrain 接 coco_pose_pretrain, finetune 接 pretrain.
    auto_pretrained = {
        "pretrain": f"checkpoints/{model_name}/coco_pose_pretrain_best.pt",
        "finetune": f"checkpoints/{model_name}/pretrain_best.pt",
    }
    if pretrained is None and args.stage in auto_pretrained:
        default_pt = auto_pretrained[args.stage]
        if os.path.exists(default_pt):
            pretrained = default_pt
            print(f"  auto pretrained: {default_pt}")

    print(f"Vigil Training: {args.stage}")
    print(f"  model={model_name} device={device} epochs={epochs} lr={lr}")

    model = create_model(model_name, pretrained=pretrained, **model_kwargs)
    print(f"  params: {model.num_params/1e6:.2f}M")
    if save_best_by == "score" and not hasattr(model, "predict_val_full"):
        raise ValueError(f"model {model_name!r} does not implement predict_val_full, cannot save best by score")

    for part in freeze:
        if hasattr(model, part):
            for p in getattr(model, part).parameters():
                p.requires_grad = False
            print(f"  frozen: {part}")

    datasets = stage_cfg["datasets"]
    aug_cfg = cfg.get("augmentation", {})
    val_ratio = train_cfg.get("val_ratio", 0.2)
    print(f"  datasets: {list(datasets.keys())} | val_ratio={val_ratio}")

    result = make_dataloaders(
        datasets, batch_size=batch, augment=aug_cfg, val_ratio=val_ratio,
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=dl_cfg.get("pin_memory", False),
        persistent_workers=dl_cfg.get("persistent_workers", False),
        verbose=False)
    if val_ratio > 0:
        train_loaders, val_loaders = result
    else:
        train_loaders, val_loaders = result, {}
    for name, dl in train_loaders.items():
        n_train = len(dl.dataset)
        n_val = len(val_loaders[name].dataset) if name in val_loaders else 0
        print(f"  [{name}] {n_train} train / {n_val} val samples")

    if not train_loaders:
        print("ERROR: no datasets found")
        sys.exit(1)

    trainer = Trainer(
        model, device=device,
        lr=lr, weight_decay=wd, warmup_epochs=warmup,
        grad_clip=grad_clip, log_interval=log_interval,
        save_interval=save_interval, val_interval=val_interval,
        save_dir=save_dir,
        use_tensorboard=use_tb, tb_log_dir=tb_log_dir,
        use_amp=amp_cfg.get("enabled", False),
        amp_dtype=amp_cfg.get("dtype", "float16"),
        map_enabled=map_enabled,
        map_samples=map_cfg.get("val_samples", 500),
        save_best_by=save_best_by,
    )

    if args.resume:
        trainer.load(args.resume)

    # 提取数据集采样权重 (解决小样本数据集学习不充分问题)
    dataset_weights = {}
    for ds_name, ds_spec in datasets.items():
        w = ds_spec.get("weight", None)
        if w is not None:
            dataset_weights[ds_name] = float(w)

    trainer.fit(epochs, train_loaders, val_loaders, save_prefix=args.stage,
                dataset_weights=dataset_weights if dataset_weights else None)


if __name__ == "__main__":
    main()

