"""Training entry point for multi-head verification.

Usage:
    python -m test_model.train --model dual_head --data /data/coco2017
    python -m test_model.train --config test_model/config.yaml
    python -m test_model.train --config test_model/config.yaml --model dual_head --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.models import create_model
from test_model.dataset import create_dataloader
from test_model.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description='Train multi-head verification model')
    p.add_argument('--config', type=str, default=None,
                   help='Path to YAML config file')
    p.add_argument('--model', type=str, default=None,
                   choices=['dual_head', 'unified_head', 'dual_neck', 'attn_dual', 'bifpn_dual'],
                   help='Model variant (overrides config)')
    p.add_argument('--data', type=str, default=None,
                   help='Dataset root directory (overrides config)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--save-dir', type=str, default=None)
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from checkpoint')
    p.add_argument('--no-mosaic', action='store_true', default=None)
    p.add_argument('--no-amp', action='store_true', default=None)
    p.add_argument('--debug', action='store_true', default=None)
    return p.parse_args()


def load_config(args):
    """Load config and merge with CLI args."""
    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f)

    # Merge: CLI > config > defaults
    def _get(key, default, cfg_section=None):
        section = cfg.get(cfg_section, {}) if cfg_section else cfg
        return getattr(args, key) if getattr(args, key) is not None else section.get(key, default)

    model_name = args.model or cfg.get('model', 'dual_head')
    data_root = args.data or cfg.get('data', {}).get('root', '/data/coco2017')
    epochs = _get('epochs', 300, 'training')
    batch = _get('batch', 16, 'training')
    lr = _get('lr', 0.01, 'training')
    device = args.device or cfg.get('device', 'cuda')
    workers = _get('workers', 4, 'training')
    save_dir = _get('save_dir', 'checkpoints', 'training')
    no_mosaic = _get('no_mosaic', False, 'training')
    no_amp = _get('no_amp', False, 'training')
    debug = args.debug if args.debug is not None else cfg.get('debug', False)

    return {
        'model': model_name,
        'data_root': data_root,
        'epochs': epochs,
        'batch': batch,
        'lr': lr,
        'device': device,
        'workers': workers,
        'save_dir': save_dir,
        'no_mosaic': no_mosaic,
        'no_amp': no_amp,
        'debug': debug,
        'config': cfg,
    }, args.resume


def main():
    args = parse_args()
    opts, resume = load_config(args)
    cfg = opts['config']
    t_cfg = cfg.get('training', {})
    l_cfg = cfg.get('loss', {})

    # Device setup
    device = opts['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        gpu_id = cfg.get('gpu_id', 0)
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(gpu_id % torch.cuda.device_count())
        print(f"GPU {gpu_id}: cudnn.benchmark=True")

    # Create model with loss weights from config
    model_name = opts['model']
    print(f"Creating model: {model_name}")
    model_kwargs = {
        'num_kpts': cfg.get('num_kpts', 17),
        'reg_max': cfg.get('reg_max', 16),
    }
    if model_name == 'unified_head':
        model_kwargs['num_classes'] = cfg.get('num_classes', 20)
    else:
        model_kwargs['num_det_classes'] = cfg.get('num_det_classes', 19)

    model = create_model(model_name, **model_kwargs)
    print(f"Parameters: {model.num_params / 1e6:.2f}M")

    # Set loss weights (override defaults)
    if hasattr(model, 'det_loss'):
        model.det_loss.w_box = l_cfg.get('w_box', 7.5)
        model.det_loss.w_cls = l_cfg.get('w_cls', 0.5)
        model.det_loss.w_dfl = l_cfg.get('w_dfl', 1.5)
    if hasattr(model, 'pose_loss'):
        model.pose_loss.w_box = l_cfg.get('w_box', 7.5)
        model.pose_loss.w_cls = l_cfg.get('w_cls', 0.5)
        model.pose_loss.w_dfl = l_cfg.get('w_dfl', 1.5)
        model.pose_loss.w_pose = l_cfg.get('w_pose', 12.0)
        model.pose_loss.w_kobj = l_cfg.get('w_kobj', 1.0)
    if hasattr(model, 'loss_fn'):
        model.loss_fn.w_box = l_cfg.get('w_box', 7.5)
        model.loss_fn.w_cls = l_cfg.get('w_cls', 0.5)
        model.loss_fn.w_dfl = l_cfg.get('w_dfl', 1.5)
        model.loss_fn.w_pose = l_cfg.get('w_pose', 12.0)
        model.loss_fn.w_kobj = l_cfg.get('w_kobj', 1.0)

    # Create dataloaders
    data_root = Path(opts['data_root'])
    train_loader = create_dataloader(
        data_dir=data_root,
        img_dir=cfg.get('data', {}).get('train_img', 'images/train2017'),
        label_dir=cfg.get('data', {}).get('train_label', 'labels/train2017'),
        batch_size=opts['batch'],
        use_mosaic=not opts['no_mosaic'],
        augment=True,
        shuffle=True,
        num_workers=opts['workers'],
    )

    val_loader = create_dataloader(
        data_dir=data_root,
        img_dir=cfg.get('data', {}).get('val_img', 'images/val2017'),
        label_dir=cfg.get('data', {}).get('val_label', 'labels/val2017'),
        batch_size=opts['batch'],
        use_mosaic=False,
        augment=False,
        shuffle=False,
        num_workers=opts['workers'],
    )

    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # Create trainer
    save_path = Path(opts['save_dir']) / model_name
    trainer = Trainer(
        model=model,
        device=device,
        lr=opts['lr'],
        momentum=t_cfg.get('momentum', 0.937),
        weight_decay=t_cfg.get('weight_decay', 5e-4),
        warmup_epochs=t_cfg.get('warmup_epochs', 3),
        grad_clip=t_cfg.get('grad_clip', 10.0),
        log_interval=20 if not opts['debug'] else 1,
        save_interval=t_cfg.get('save_interval', 50),
        val_interval=t_cfg.get('val_interval', 5),
        save_dir=str(save_path),
        use_amp=(not opts['no_amp']) and device == 'cuda',
        ema_decay=t_cfg.get('ema_decay', 0.9999),
        save_best_by=t_cfg.get('save_best_by', 'loss'),
        use_tensorboard=t_cfg.get('tensorboard', True),
    )

    if resume:
        trainer.load(resume)

    # Train
    epochs = 3 if opts['debug'] else opts['epochs']
    close_mosaic = t_cfg.get('close_mosaic_epochs', 10) if not opts['no_mosaic'] else 0

    trainer.fit(
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        save_prefix=model_name,
        close_mosaic_epochs=close_mosaic,
    )

    print(f"\nTraining complete for {model_name}!")


if __name__ == '__main__':
    main()
