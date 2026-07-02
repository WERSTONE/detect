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
    p.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'adamw'])
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
    def _get(key, default, cfg_section=None, aliases=()):
        section = cfg.get(cfg_section, {}) if cfg_section else cfg
        arg_val = getattr(args, key)
        if arg_val is not None:
            return arg_val
        for name in (key, *aliases):
            if name in section:
                return section[name]
        return default

    model_name = args.model or cfg.get('model', 'dual_head')
    data_root = args.data or cfg.get('data', {}).get('root', '/data/coco2017')
    epochs = _get('epochs', 300, 'training')
    batch = _get('batch', 16, 'training', aliases=('batch_size',))
    lr = _get('lr', 0.01, 'training', aliases=('lr0',))
    optimizer = _get('optimizer', 'sgd', 'training')
    device = args.device or cfg.get('device', 'cuda')
    workers = _get('workers', 4, 'training')
    save_dir = _get('save_dir', 'checkpoints', 'training')
    no_mosaic = _get('no_mosaic', False, 'training')
    no_amp = args.no_amp if args.no_amp is not None else (
        cfg.get('training', {}).get('no_amp', False) or
        not cfg.get('training', {}).get('amp', True)
    )
    debug = args.debug if args.debug is not None else cfg.get('debug', False)

    return {
        'model': model_name,
        'data_root': data_root,
        'epochs': epochs,
        'batch': batch,
        'lr': lr,
        'optimizer': optimizer,
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
    a_cfg = cfg.get('augmentation', {})
    d_cfg = cfg.get('data', {})

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

    # Create val dataloader (shared)
    data_root = Path(opts['data_root'])
    val_loader = create_dataloader(
        data_dir=data_root,
        img_dir=d_cfg.get('val_img', 'images/val2017'),
        label_dir=d_cfg.get('val_label', 'labels/val2017'),
        input_size=d_cfg.get('input_size', 640),
        batch_size=opts['batch'],
        use_mosaic=False,
        augment=False,
        shuffle=False,
        num_workers=opts['workers'],
        drop_last=False,
        class_id_format=d_cfg.get('class_id_format', 'yolo80'),
    )
    print(f"Val: {len(val_loader.dataset)} samples")

    close_mosaic = t_cfg.get('close_mosaic_epochs', 10) if not opts['no_mosaic'] else 0

    def _make_trainer(lr, save_dir_suffix=''):
        save_path = Path(opts['save_dir']) / (model_name + save_dir_suffix)
        return Trainer(
            model=model,
            device=device,
            lr=lr,
            optimizer=opts['optimizer'],
            momentum=t_cfg.get('momentum', 0.937),
            weight_decay=t_cfg.get('weight_decay', 5e-4),
            nesterov=t_cfg.get('nesterov', True),
            final_lr_ratio=t_cfg.get('lrf', 0.01),
            cos_lr=t_cfg.get('cos_lr', True),
            warmup_epochs=t_cfg.get('warmup_epochs', 3),
            grad_clip=t_cfg.get('grad_clip', 10.0),
            log_interval=t_cfg.get('log_interval', 20 if not opts['debug'] else 1),
            save_interval=t_cfg.get('save_interval', 50),
            val_interval=t_cfg.get('val_interval', 5),
            save_dir=str(save_path),
            use_amp=(not opts['no_amp']) and device == 'cuda',
            ema_decay=t_cfg.get('ema_decay', 0.9999),
            save_best_by=t_cfg.get('save_best_by', 'loss'),
            use_tensorboard=t_cfg.get('tensorboard', True),
        )

    def _make_train_loader(person_only=False):
        return create_dataloader(
            data_dir=data_root,
            img_dir=d_cfg.get('train_img', 'images/train2017'),
            label_dir=d_cfg.get('train_label', 'labels/train2017'),
            input_size=d_cfg.get('input_size', 640),
            batch_size=opts['batch'],
            use_mosaic=not opts['no_mosaic'],
            augment=True,
            shuffle=True,
            num_workers=opts['workers'],
            drop_last=True,
            class_id_format=d_cfg.get('class_id_format', 'yolo80'),
            hsv_h=a_cfg.get('hsv_h', 0.015),
            hsv_s=a_cfg.get('hsv_s', 0.7),
            hsv_v=a_cfg.get('hsv_v', 0.4),
            flip_lr=a_cfg.get('flip_lr', 0.5),
            mosaic_prob=a_cfg.get('mosaic_prob', 0.5),
            person_only=person_only,
        )

    # Two-stage training for dual-head models
    two_stage = t_cfg.get('two_stage', {})
    is_dual_head = model_name in ('dual_head', 'dual_neck', 'attn_dual', 'bifpn_dual')
    if two_stage.get('enabled') and is_dual_head:
        s1 = two_stage['stage1']
        s2 = two_stage['stage2']

        # ---- Stage 1: pose head only (person-only data) ----
        model.train_det = False
        model.freeze_head('det')

        train_loader_s1 = _make_train_loader(person_only=True)
        print(f"Stage1 train (person-only): {len(train_loader_s1.dataset)} samples")

        # Quick debug: test one batch
        model.train()
        dbg_batch = next(iter(train_loader_s1))
        dbg_images = dbg_batch['image'].to(device, non_blocking=True)
        dbg_gt = [{'boxes': dbg_batch['boxes'][i], 'classes': dbg_batch['classes'][i],
                    'kpts': dbg_batch['kpts'][i]} for i in range(len(dbg_images))]
        dbg_losses = model.compute_loss(dbg_images, dbg_gt)
        print(f"  [DEBUG] test batch loss: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(dbg_losses.items())))

        s1_epochs = min(3, s1.get('epochs', 80)) if opts['debug'] else s1.get('epochs', 80)
        s1_lr = s1.get('lr0', opts['lr'])
        if s1.get('freeze_backbone', False):
            for p in model.backbone.parameters():
                p.requires_grad = False

        print(f"\n{'='*60}")
        print(f"Stage 1: pose head only | Epochs: {s1_epochs} | LR: {s1_lr}")
        print(f"{'='*60}")

        trainer1 = _make_trainer(s1_lr)
        trainer1.fit(
            epochs=s1_epochs,
            train_loader=train_loader_s1,
            val_loader=val_loader,
            save_prefix=model_name + '_stage1',
            close_mosaic_epochs=close_mosaic,
        )

        # Release stage1 loader workers before creating full loader
        if hasattr(train_loader_s1, '_iterator') and train_loader_s1._iterator is not None:
            train_loader_s1._iterator._shutdown_workers()
        del train_loader_s1

        # ---- Stage 2: both heads (full data) ----
        model.train_det = True
        model.det_weight_warmup_epochs = s2.get('det_weight_warmup_epochs', 5)
        model.det_weight_mult = 0.0
        model.unfreeze_all()

        train_loader = _make_train_loader(person_only=False)
        print(f"Stage2 train (full): {len(train_loader.dataset)} samples")

        def _on_epoch_start(epoch):
            model.update_det_weight(epoch)

        s2_epochs = min(3, s2.get('epochs', 200)) if opts['debug'] else s2.get('epochs', 200)
        s2_lr = s2.get('lr0', opts['lr'] * 0.4)

        print(f"\n{'='*60}")
        print(f"Stage 2: both heads | Epochs: {s2_epochs} | LR: {s2_lr}")
        print(f"  det_weight warmup over {model.det_weight_warmup_epochs} epochs (0→1)")
        print(f"{'='*60}")

        trainer2 = _make_trainer(s2_lr)
        trainer2.fit(
            epochs=s2_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            save_prefix=model_name,
            close_mosaic_epochs=close_mosaic,
            on_epoch_start=_on_epoch_start,
        )

        print(f"\nTwo-stage training complete for {model_name}!")
    else:
        # Single-stage training (original flow)
        train_loader = _make_train_loader()
        print(f"Train: {len(train_loader.dataset)} samples")

        trainer = _make_trainer(opts['lr'])
        if resume:
            trainer.load(resume)

        epochs = 3 if opts['debug'] else opts['epochs']

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
