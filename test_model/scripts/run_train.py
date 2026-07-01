#!/usr/bin/env python
"""Single model training launcher with auto-evaluation.

Usage:
    python test_model/scripts/run_train.py --config test_model/config.yaml
    python test_model/scripts/run_train.py --config test_model/config.yaml --model dual_head
    python test_model/scripts/run_train.py --config test_model/config.yaml --epochs 100 --batch 32

CLI args override config file values.
After training completes, automatically evaluates best.pt and last.pt.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def deep_merge(base, override):
    """Merge override dict into base dict (nested)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def parse_args():
    p = argparse.ArgumentParser(description='Train multi-head verification model')
    p.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'test_model/config.yaml'),
                   help='Path to YAML config')
    # CLI overrides
    p.add_argument('--model', type=str, default=None,
                   choices=['dual_head', 'unified_head', 'dual_neck', 'attn_dual', 'bifpn_dual'])
    p.add_argument('--data', type=str, default=None, help='Dataset root directory')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--optimizer', type=str, default=None, choices=['sgd', 'adamw'])
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--gpu-id', type=int, default=None)
    p.add_argument('--save-dir', type=str, default=None)
    p.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    p.add_argument('--no-mosaic', action='store_true', default=None,
                   help='Disable mosaic augmentation')
    p.add_argument('--no-amp', action='store_true', default=None,
                   help='Disable AMP mixed precision')
    p.add_argument('--debug', action='store_true', default=None,
                   help='Quick smoke test (overrides config)')
    return p.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}, using defaults")
        cfg = {}
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config: {config_path}")

    # Apply CLI overrides (non-None values)
    overrides = {}
    if args.model is not None:
        cfg['model'] = args.model
    if args.epochs is not None:
        overrides.setdefault('training', {})['epochs'] = args.epochs
    if args.batch is not None:
        overrides.setdefault('training', {})['batch_size'] = args.batch
    if args.lr is not None:
        overrides.setdefault('training', {})['lr0'] = args.lr
    if args.optimizer is not None:
        overrides.setdefault('training', {})['optimizer'] = args.optimizer
    if args.device is not None:
        cfg['device'] = args.device
    if args.gpu_id is not None:
        cfg['gpu_id'] = args.gpu_id
    if args.save_dir is not None:
        overrides.setdefault('training', {})['save_dir'] = args.save_dir
    if args.debug is not None:
        cfg['debug'] = args.debug
    if args.no_mosaic is not None:
        overrides.setdefault('training', {})['no_mosaic'] = args.no_mosaic
    if args.no_amp is not None:
        overrides.setdefault('training', {})['amp'] = not args.no_amp
    if args.data is not None:
        cfg.setdefault('data', {})['root'] = args.data

    if overrides:
        deep_merge(cfg, overrides)

    return cfg, args.resume


def print_config(cfg):
    """Pretty-print config summary."""
    t = cfg.get('training', {})
    l = cfg.get('loss', {})
    ts = t.get('two_stage', {})
    lines = [f"""
{'='*60}
Training Config
{'='*60}
Model:       {cfg['model']}
Data:        {cfg['data']['root']}
Epochs:      {t.get('epochs', 300)} | Batch: {t.get('batch_size', 16)}
Optimizer:   {t.get('optimizer', 'sgd')} | LR: {t.get('lr0', 0.01)} → {t.get('lr0', 0.01) * t.get('lrf', 0.01):.0e}
Warmup:      {t.get('warmup_epochs', 3)} epochs | Cosine LR: {t.get('cos_lr', True)}
EMA:         {t.get('ema_decay', 0)} | AMP: {t.get('amp', True)}
Mosaic:      close_mosaic={t.get('close_mosaic_epochs', 10)}
Loss:        box={l.get('w_box', 7.5)} cls={l.get('w_cls', 0.5)} dfl={l.get('w_dfl', 1.5)}
             pose={l.get('w_pose', 12.0)} kobj={l.get('w_kobj', 1.0)}
Save:        {t.get('save_dir', 'checkpoints')}/{cfg['model']}
Device:      {cfg.get('device', 'cuda')}:{cfg.get('gpu_id', 0)}
Debug:       {cfg.get('debug', False)}"""]
    if ts.get('enabled'):
        s1 = ts.get('stage1', {})
        s2 = ts.get('stage2', {})
        lines.append(f"""Two-Stage:   enabled
  Stage1:    epochs={s1.get('epochs', 50)} lr={s1.get('lr0', 0.005)} freeze_backbone={s1.get('freeze_backbone', False)}
  Stage2:    epochs={s2.get('epochs', 150)} lr={s2.get('lr0', 0.002)} det_weight_warmup={s2.get('det_weight_warmup_epochs', 5)}""")
    lines.append(f"{'='*60}")
    print('\n'.join(lines))


def main():
    args = parse_args()
    cfg, resume_path = load_config(args)
    print_config(cfg)

    # Setup GPU
    gpu_id = cfg.get('gpu_id', 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = cfg.get('device', 'cuda')

    model_name = cfg['model']
    t = cfg['training']
    data_root = cfg['data']['root']
    save_dir = Path(t.get('save_dir', 'checkpoints'))
    epochs = 3 if cfg.get('debug') else t.get('epochs', 300)
    batch = t.get('batch_size', 16)

    # Build train command
    config_path = args.config
    train_args = [
        sys.executable, '-m', 'test_model.train',
        '--model', model_name,
        '--config', config_path,
        '--data', data_root,
        '--epochs', str(epochs),
        '--batch', str(batch),
        '--optimizer', t.get('optimizer', 'sgd'),
        '--device', device,
        '--workers', str(t.get('workers', 8)),
        '--save-dir', str(save_dir),
    ]

    if resume_path:
        train_args.extend(['--resume', resume_path])
    if t.get('no_mosaic'):
        train_args.append('--no-mosaic')
    if not t.get('amp', True):
        train_args.append('--no-amp')
    if cfg.get('debug'):
        train_args.append('--debug')

    print(f"\nRunning: {' '.join(train_args)}\n")
    result = subprocess.run(train_args, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Auto-evaluation
    if not cfg.get('debug') and cfg.get('eval', {}).get('auto_after_train', True):
        eval_cfg = cfg.get('eval', {})
        eval_batch = eval_cfg.get('batch_size', batch)
        model_dir = save_dir / model_name

        for ckpt_type in eval_cfg.get('checkpoints', ['last', 'best']):
            ckpt_path = model_dir / f'{model_name}_{ckpt_type}.pt'
            if not ckpt_path.exists():
                print(f"\nCheckpoint not found, skipping: {ckpt_path}")
                continue

            metrics_path = model_dir / f'{model_name}_{ckpt_type}_metrics.json'
            eval_args = [
                sys.executable, '-m', 'test_model.eval',
                '--model', model_name,
                '--config', config_path,
                '--weights', str(ckpt_path),
                '--data', data_root,
                '--device', device,
                '--batch', str(eval_batch),
                '--img-dir', cfg.get('data', {}).get('val_img', 'images/val2017'),
                '--label-dir', cfg.get('data', {}).get('val_label', 'labels/val2017'),
                '--input-size', str(cfg.get('data', {}).get('input_size', 640)),
                '--class-id-format', cfg.get('data', {}).get('class_id_format', 'yolo80'),
                '--score-thresh', str(eval_cfg.get('score_thresh', 0.01)),
                '--iou-thresh', str(eval_cfg.get('iou_thresh', 0.6)),
                '--output', str(metrics_path),
            ]

            print(f"\n{'='*60}")
            print(f"Evaluating: {ckpt_path}")
            print(f"{'='*60}")
            eval_result = subprocess.run(eval_args, cwd=str(PROJECT_ROOT))
            if eval_result.returncode != 0:
                print(f"  Evaluation failed with exit code {eval_result.returncode}")
                continue

            # Print key metrics
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                print(f"\n  {ckpt_type}.pt results:")
                for k in ['mAP@0.5', 'mAP@0.5_no_person', 'AP_person_box@0.5',
                          'mAP@0.5:0.95', 'AP_pose@0.5']:
                    if k in m and m[k] is not None:
                        print(f"    {k}: {m[k]:.4f}")

    print(f"\nDone! Model: {model_name}")
    return 0


if __name__ == '__main__':
    main()
