"""Training engine for the BiFPN detector-pose model.

Usage:
    python -m test_model.main --config test_model/config/bifpn_dual.yaml
    python -m test_model.main --config test_model/config/bifpn_det_only.yaml
    python -m test_model.main --config test_model/config/bifpn_pose_only.yaml
"""

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test_model.model import create_model
from test_model.train.dataset import create_dataloader, collate_fn
from test_model.train.trainer import Trainer

OFFICIAL_BACKBONE_PREFIX_MAP = {
    'model.0': 'stem.0',
    'model.1': 'stem.1',
    'model.2': 'stem.2',
    'model.3': 'stage3.0',
    'model.4': 'stage3.1',
    'model.5': 'stage4.0',
    'model.6': 'stage4.1',
    'model.7': 'stage5_down',
    'model.8': 'stage5_c2f',
    'model.9': 'stage5_sppf',
}

YOLOV8_SCALE_BY_OUT_CHANNELS = {
    (32, 64, 128, 256): 'n',
    (64, 128, 256, 512): 's',
    (96, 192, 384, 768): 'm',
    (128, 256, 512, 512): 'l',
    (160, 320, 512, 512): 'x',
}


def _normalize_device(device, gpu_id=0):
    device = str(device or 'cuda').strip()
    if device.isdigit():
        device = f'cuda:{device}'
    if device == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = f'cuda:{int(gpu_id) % torch.cuda.device_count()}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return 'cpu'
    return device


def _is_cuda_device(device):
    return torch.device(device).type == 'cuda'


def _config_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def parse_args():
    p = argparse.ArgumentParser(description='Train multi-head verification model')
    p.add_argument('--config', type=str, default=None,
                   help='Path to YAML config file')
    p.add_argument('--model', type=str, default=None,
                   choices=['bifpn', 'bifpn_dual', 'bifpn_detect', 'bifpn_det',
                            'bifpn_pose', 'bifpn_pose_only'],
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
    p.add_argument('--backbone-weights', type=str, default=None,
                   help='Path to backbone pretrained weights (YOLO official or compatible checkpoint)')
    p.add_argument('--backbone-strict', action='store_true', default=None,
                   help='Require a near-complete backbone match when loading pretrained weights')
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

    model_name = args.model or cfg.get('model', 'bifpn_dual')
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
    pretrained_cfg = cfg.setdefault('pretrained', {}).setdefault('backbone', {})
    if args.backbone_weights is not None:
        pretrained_cfg['enabled'] = True
        pretrained_cfg['weights'] = args.backbone_weights
    if args.backbone_strict is not None:
        pretrained_cfg['strict'] = args.backbone_strict

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


def _extract_checkpoint_state_dict(checkpoint):
    """Extract a state dict from common checkpoint layouts."""
    if hasattr(checkpoint, 'state_dict'):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    for key in ('ema', 'model'):
        module = checkpoint.get(key)
        if hasattr(module, 'state_dict'):
            return module.state_dict()

    for key in ('state_dict', 'model_state_dict'):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise KeyError("Could not find a compatible state_dict in checkpoint")


def _map_backbone_key(source_key):
    """Map checkpoint key names to local backbone key names."""
    if source_key.startswith('backbone.'):
        return source_key[len('backbone.'):]

    for official_prefix, local_prefix in OFFICIAL_BACKBONE_PREFIX_MAP.items():
        if source_key == official_prefix:
            return local_prefix
        if source_key.startswith(official_prefix + '.'):
            return local_prefix + source_key[len(official_prefix):]

    return source_key


def _infer_yolov8_scale(model):
    """Infer YOLOv8 scale from backbone channel dimensions."""
    out_channels = tuple(int(v) for v in getattr(model.backbone, 'out_channels', ()))
    return YOLOV8_SCALE_BY_OUT_CHANNELS.get(out_channels)


def _official_backbone_assets(model, backbone_cfg, explicit_name=None):
    """Return preferred official asset filenames for this backbone."""
    assets = []
    if explicit_name:
        assets.append(explicit_name)

    scale = backbone_cfg.get('scale', 'auto')
    if scale in (None, '', 'auto'):
        scale = _infer_yolov8_scale(model)
    if not scale:
        scale = 'm'

    pose_asset = f'yolov8{scale}-pose.pt'
    detect_asset = f'yolov8{scale}.pt'
    ordered = [pose_asset, detect_asset] if backbone_cfg.get('prefer_pose', True) else [detect_asset, pose_asset]
    for asset in ordered:
        if asset not in assets:
            assets.append(asset)
    return assets


def _candidate_weight_paths(weights_value, cache_dir, asset_names):
    """Return candidate local paths to try before downloading."""
    paths = []
    seen = set()

    search_dirs = [
        PROJECT_ROOT,
        cache_dir,
        PROJECT_ROOT / 'checkpoints',
        PROJECT_ROOT / 'checkpoints' / 'yolo_pose',
        PROJECT_ROOT / 'weights',
    ]

    def _add(path_obj):
        path_obj = Path(path_obj)
        key = str(path_obj).lower()
        if key not in seen:
            seen.add(key)
            paths.append(path_obj)

    if weights_value and str(weights_value).strip().lower() not in ('auto', 'none', 'null'):
        raw_path = Path(str(weights_value))
        _add(raw_path)
        if not raw_path.is_absolute():
            for base in search_dirs:
                _add(base / raw_path)

    for asset in asset_names:
        for base in search_dirs:
            _add(base / asset)

    return paths


def _download_ultralytics_asset(asset_name, cache_dir):
    """Download an official Ultralytics asset into the project cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.ultralytics'))

    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception as exc:
        raise RuntimeError(f"Ultralytics download helper is unavailable: {exc}") from exc

    cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        downloaded = Path(attempt_download_asset(asset_name))
    finally:
        os.chdir(cwd)

    if not downloaded.is_absolute():
        downloaded = cache_dir / downloaded
    downloaded = downloaded.resolve()

    if not downloaded.exists():
        raise FileNotFoundError(f"Ultralytics reported {downloaded}, but the file does not exist")

    target = (cache_dir / asset_name).resolve()
    if downloaded != target:
        if not target.exists():
            shutil.copy2(downloaded, target)
        return target
    return downloaded


def _resolve_backbone_weights_path(model, cfg):
    """Resolve local or downloadable official backbone weights."""
    backbone_cfg = cfg.get('pretrained', {}).get('backbone', {})
    weights_value = backbone_cfg.get('weights', 'auto')
    cache_dir = Path(backbone_cfg.get('cache_dir', 'checkpoints/yolo_pose'))
    if not cache_dir.is_absolute():
        cache_dir = (PROJECT_ROOT / cache_dir).resolve()

    explicit_name = None
    if weights_value and str(weights_value).strip().lower() not in ('auto', 'none', 'null'):
        explicit_name = Path(str(weights_value)).name

    asset_names = _official_backbone_assets(model, backbone_cfg, explicit_name=explicit_name)
    candidate_paths = _candidate_weight_paths(weights_value, cache_dir, asset_names)
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    if backbone_cfg.get('auto_download', True):
        download_errors = []
        for asset_name in asset_names:
            try:
                print(f"[Backbone preload] Local file missing, downloading official asset: {asset_name}")
                return _download_ultralytics_asset(asset_name, cache_dir)
            except Exception as exc:
                download_errors.append(f"{asset_name}: {exc}")
        joined = "; ".join(download_errors)
        raise FileNotFoundError(
            f"Backbone weights were not found locally and auto-download failed. Tried assets: {asset_names}. "
            f"Details: {joined}"
        )

    searched = ", ".join(str(p) for p in candidate_paths[:8])
    raise FileNotFoundError(
        f"Backbone weights not found. Searched candidates such as: {searched}. "
        f"Enable pretrained.backbone.auto_download or provide a valid weights path."
    )


def _load_pretrained_backbone(model, cfg, resume_path=None):
    """Load backbone weights from a compatible checkpoint if configured."""
    backbone_cfg = cfg.get('pretrained', {}).get('backbone', {})
    if not backbone_cfg.get('enabled', False):
        return None

    if resume_path:
        print(f"[Backbone preload] Resume is set ({resume_path}); skipping backbone preload.")
        return None

    weights_path = _resolve_backbone_weights_path(model, cfg)

    print(f"[Backbone preload] Loading checkpoint: {weights_path}")
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    source_state = _extract_checkpoint_state_dict(checkpoint)
    target_state_full = model.backbone.state_dict()
    target_state = {
        key: value for key, value in target_state_full.items()
        if not key.endswith('num_batches_tracked')
    }

    matched_state = {}
    matched_keys = []
    mismatched = []
    candidate_keys = 0

    for source_key, tensor in source_state.items():
        if source_key.endswith('num_batches_tracked'):
            continue
        local_key = _map_backbone_key(source_key)
        if local_key not in target_state:
            continue
        candidate_keys += 1
        if tuple(target_state[local_key].shape) != tuple(tensor.shape):
            mismatched.append({
                'source_key': source_key,
                'target_key': local_key,
                'source_shape': tuple(tensor.shape),
                'target_shape': tuple(target_state[local_key].shape),
            })
            continue
        matched_state[local_key] = tensor
        matched_keys.append((source_key, local_key))

    missing_keys = [key for key in target_state.keys() if key not in matched_state]
    optional_missing_keys = [key for key in missing_keys if key.startswith('eca_')]
    required_missing_keys = [key for key in missing_keys if key not in optional_missing_keys]
    strict = bool(backbone_cfg.get('strict', False))

    if strict and (mismatched or required_missing_keys):
        mismatch_msg = ""
        if mismatched:
            first = mismatched[0]
            mismatch_msg = (
                f"; first mismatch {first['source_key']} -> {first['target_key']} "
                f"{first['source_shape']} != {first['target_shape']}"
            )
        raise RuntimeError(
            f"Backbone strict preload failed: loaded {len(matched_state)}/{len(target_state)} keys, "
            f"missing_required {len(required_missing_keys)}, optional_missing {len(optional_missing_keys)}, "
            f"mismatched {len(mismatched)}{mismatch_msg}"
        )

    if not matched_state:
        print(
            "[Backbone preload] No compatible backbone tensors were loaded. "
            "Check whether the checkpoint scale matches this YOLOv8m-style backbone."
        )
        if mismatched:
            first = mismatched[0]
            print(
                f"  First mismatch: {first['source_key']} -> {first['target_key']} "
                f"{first['source_shape']} != {first['target_shape']}"
            )
        return {
            'loaded': 0,
            'total': len(target_state),
            'missing': len(missing_keys),
            'missing_required': len(required_missing_keys),
            'mismatched': len(mismatched),
            'weights_path': str(weights_path),
        }

    model.backbone.load_state_dict(matched_state, strict=False)
    print(
        f"[Backbone preload] Loaded {len(matched_state)}/{len(target_state)} backbone tensors "
        f"from {weights_path.name}; matched candidates={candidate_keys}, "
        f"missing={len(missing_keys)} (required={len(required_missing_keys)}), "
        f"mismatched={len(mismatched)}"
    )
    if matched_keys:
        preview = ", ".join(f"{src}->{dst}" for src, dst in matched_keys[:5])
        print(f"  Sample matches: {preview}")
    if mismatched:
        first = mismatched[0]
        print(
            f"  First shape mismatch: {first['source_key']} -> {first['target_key']} "
            f"{first['source_shape']} != {first['target_shape']}"
        )

    return {
        'loaded': len(matched_state),
        'total': len(target_state),
        'missing': len(missing_keys),
        'missing_required': len(required_missing_keys),
        'mismatched': len(mismatched),
        'weights_path': str(weights_path),
    }


def _as_name_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _candidate_path(path_value):
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _matches_roots(name, roots):
    return any(name == root or name.startswith(root + '.') for root in roots)


def _load_pretrained_modules(model, cfg, resume_path=None):
    """Load exact-name compatible module tensors from a checkpoint."""
    modules_cfg = cfg.get('pretrained', {}).get('modules', {})
    if not modules_cfg or not modules_cfg.get('enabled', False):
        return None
    if resume_path:
        print(f"[Module preload] Resume is set ({resume_path}); skipping module preload.")
        return None

    weights_value = modules_cfg.get('weights', None)
    if not weights_value:
        raise ValueError("pretrained.modules.enabled=true requires pretrained.modules.weights")
    weights_path = _candidate_path(weights_value)
    if not weights_path.exists():
        raise FileNotFoundError(f"Module preload checkpoint not found: {weights_path}")

    include = _as_name_list(modules_cfg.get('include', modules_cfg.get('modules', [])))
    exclude = _as_name_list(modules_cfg.get('exclude', []))
    if not include:
        raise ValueError("pretrained.modules.include must list modules such as ['backbone', 'neck']")

    print(f"[Module preload] Loading checkpoint: {weights_path}")
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    source_state = _extract_checkpoint_state_dict(checkpoint)
    target_state = {
        key: value for key, value in model.state_dict().items()
        if not key.endswith('num_batches_tracked')
    }

    matched_state = {}
    mismatched = []
    skipped = 0
    candidate_keys = 0
    for source_key, tensor in source_state.items():
        if source_key.endswith('num_batches_tracked'):
            continue
        if not _matches_roots(source_key, include):
            continue
        if exclude and _matches_roots(source_key, exclude):
            skipped += 1
            continue
        candidate_keys += 1
        if source_key not in target_state:
            skipped += 1
            continue
        if tuple(target_state[source_key].shape) != tuple(tensor.shape):
            mismatched.append({
                'key': source_key,
                'source_shape': tuple(tensor.shape),
                'target_shape': tuple(target_state[source_key].shape),
            })
            continue
        matched_state[source_key] = tensor

    strict = bool(modules_cfg.get('strict', False))
    expected_keys = [
        key for key in target_state
        if _matches_roots(key, include) and not (exclude and _matches_roots(key, exclude))
    ]
    missing = [key for key in expected_keys if key not in matched_state]
    if strict and (missing or mismatched):
        msg = ""
        if mismatched:
            first = mismatched[0]
            msg = f"; first mismatch {first['key']} {first['source_shape']} != {first['target_shape']}"
        raise RuntimeError(
            f"Module strict preload failed: loaded {len(matched_state)}/{len(expected_keys)} "
            f"expected tensors, missing={len(missing)}, mismatched={len(mismatched)}{msg}")

    if not matched_state:
        raise RuntimeError(
            f"Module preload found no compatible tensors in {weights_path} for include={include}")

    model.load_state_dict(matched_state, strict=False)
    print(
        f"[Module preload] Loaded {len(matched_state)}/{len(expected_keys)} tensors "
        f"for include={include}; candidates={candidate_keys}, skipped={skipped}, "
        f"missing={len(missing)}, mismatched={len(mismatched)}")
    if matched_state:
        print("  Sample module keys: " + ", ".join(list(matched_state.keys())[:5]))
    if mismatched:
        first = mismatched[0]
        print(
            f"  First shape mismatch: {first['key']} "
            f"{first['source_shape']} != {first['target_shape']}")
    return {
        'loaded': len(matched_state),
        'expected': len(expected_keys),
        'missing': len(missing),
        'mismatched': len(mismatched),
        'weights_path': str(weights_path),
    }


def _module_roots_from_flags(cfg):
    roots = []
    if cfg.get('freeze_backbone', False):
        roots.append('backbone')
    if cfg.get('freeze_neck', False):
        roots.append('neck')
    if cfg.get('freeze_det_head', False):
        roots.extend(['det_adapter', 'det_head'])
    if cfg.get('freeze_pose_head', False):
        roots.extend(['pose_adapter', 'pose_head'])
    roots.extend(_as_name_list(cfg.get('freeze_modules', [])))
    roots.extend(_as_name_list(cfg.get('modules', [])))
    return list(dict.fromkeys(roots))


def _apply_module_freeze(model, freeze_cfg=None, reset=False, label=''):
    """Apply generic module freeze/trainable rules and remember frozen BN modules."""
    freeze_cfg = freeze_cfg or {}
    if reset:
        for p in model.parameters():
            p.requires_grad = True
        setattr(model, '_frozen_module_roots', [])

    trainable_roots = _as_name_list(freeze_cfg.get('trainable_modules', []))
    if trainable_roots:
        for p in model.parameters():
            p.requires_grad = False
        for name, param in model.named_parameters():
            if _matches_roots(name, trainable_roots):
                param.requires_grad = True

    freeze_roots = _module_roots_from_flags(freeze_cfg)
    for name, param in model.named_parameters():
        if _matches_roots(name, freeze_roots):
            param.requires_grad = False

    freeze_bn = bool(freeze_cfg.get('freeze_bn', True))
    frozen_for_eval = freeze_roots if freeze_bn else []
    if trainable_roots and freeze_bn:
        frozen_for_eval = [
            name for name, _module in model.named_children()
            if not _matches_roots(name, trainable_roots)
        ]
    setattr(model, '_frozen_module_roots', list(dict.fromkeys(frozen_for_eval)))

    if trainable_roots or freeze_roots:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        prefix = f"[Freeze:{label}]" if label else "[Freeze]"
        print(
            f"{prefix} trainable_modules={trainable_roots or 'all except frozen'} "
            f"freeze_modules={freeze_roots or 'none'} freeze_bn={freeze_bn} "
            f"trainable={trainable / 1e6:.2f}M / {total / 1e6:.2f}M")


def _freeze_cfg_active(freeze_cfg):
    if not freeze_cfg:
        return False
    if 'enabled' in freeze_cfg and not bool(freeze_cfg.get('enabled')):
        return False
    return bool(
        _as_name_list(freeze_cfg.get('trainable_modules', [])) or
        _module_roots_from_flags(freeze_cfg)
    )


def _apply_frozen_module_eval(model):
    roots = getattr(model, '_frozen_module_roots', None) or []
    if not roots:
        return
    for module_name, module in model.named_modules():
        if module_name and _matches_roots(module_name, roots):
            module.eval()


def _load_model_weights_for_staged_resume(model, path, device):
    """Load model weights only; staged training creates a fresh optimizer per stage."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt if isinstance(ckpt, dict) else None)
    if not isinstance(state_dict, dict):
        raise KeyError(f"Could not find model_state_dict in resume checkpoint: {path}")
    model.load_state_dict(state_dict, strict=True)
    print(f"[Resume] Loaded model weights from {path} (epoch {ckpt.get('epoch', 'unknown')})")
    return ckpt


def main():
    args = parse_args()
    opts, resume = load_config(args)
    cfg = opts['config']
    t_cfg = cfg.get('training', {})
    l_cfg = cfg.get('loss', {})
    a_cfg = cfg.get('augmentation', {})
    d_cfg = cfg.get('data', {})

    # Device setup
    device = _normalize_device(opts['device'], cfg.get('gpu_id', 0))

    if _is_cuda_device(device):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_num_threads(1)  # Limit main-process CPU threads
        device_index = torch.device(device).index
        gpu_id = cfg.get('gpu_id', 0) if device_index is None else device_index
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(gpu_id % torch.cuda.device_count())
        print(f"GPU {gpu_id}: cudnn.benchmark=True")

    # Create model with loss weights from config
    model_name = opts['model']
    print(f"Creating model: {model_name}")
    model_kwargs = {
        'reg_max': cfg.get('reg_max', 16),
    }
    if model_name in ('bifpn_detect', 'bifpn_det'):
        neck_cfg = cfg.get('neck', {}) or {}
        assigner_cfg = cfg.get('assigner', {}) or {}
        detect_kwargs = {
            'num_det_classes': cfg.get('num_det_classes', cfg.get('num_classes', 80)),
            'input_size': d_cfg.get('input_size', 640),
            'assigner_topk': assigner_cfg.get('topk', 10),
            'assigner_alpha': assigner_cfg.get('alpha', 0.5),
            'assigner_beta': assigner_cfg.get('beta', 6.0),
            'assigner_eps': assigner_cfg.get('eps', 1.0e-9),
        }
        detect_kwargs.update({
            'neck_use_p2_context': neck_cfg.get('use_p2_context', False),
            'neck_downsample': neck_cfg.get('downsample', 'conv'),
            'neck_out_channels': neck_cfg.get('out_channels', None),
        })
        model_kwargs.update(detect_kwargs)
    elif model_name in ('bifpn_pose', 'bifpn_pose_only'):
        neck_cfg = cfg.get('neck', {}) or {}
        assigner_cfg = cfg.get('assigner', {}) or {}
        pose_kwargs = {
            'num_kpts': cfg.get('num_kpts', 17),
            'input_size': d_cfg.get('input_size', 640),
            'assigner_topk': assigner_cfg.get('topk', 10),
            'assigner_alpha': assigner_cfg.get('alpha', 0.5),
            'assigner_beta': assigner_cfg.get('beta', 6.0),
            'assigner_eps': assigner_cfg.get('eps', 1.0e-9),
        }
        pose_kwargs.update({
            'neck_use_p2_context': neck_cfg.get('use_p2_context', False),
            'neck_downsample': neck_cfg.get('downsample', 'conv'),
            'neck_out_channels': neck_cfg.get('out_channels', None),
        })
        model_kwargs.update(pose_kwargs)
    else:
        neck_cfg = cfg.get('neck', {}) or {}
        assigner_cfg = cfg.get('assigner', {}) or {}
        model_kwargs.update({
            'num_kpts': cfg.get('num_kpts', 17),
            'num_det_classes': cfg.get('num_det_classes', 80),
            'input_size': d_cfg.get('input_size', 640),
            'neck_use_p2_context': neck_cfg.get('use_p2_context', False),
            'neck_downsample': neck_cfg.get('downsample', 'conv'),
            'neck_out_channels': neck_cfg.get('out_channels', None),
            'assigner_topk': assigner_cfg.get('topk', 10),
            'assigner_alpha': assigner_cfg.get('alpha', 0.5),
            'assigner_beta': assigner_cfg.get('beta', 6.0),
            'assigner_eps': assigner_cfg.get('eps', 1.0e-9),
        })

    model = create_model(model_name, **model_kwargs)
    print(f"Parameters: {model.num_params / 1e6:.2f}M")
    _load_pretrained_backbone(model, cfg, resume_path=resume)
    _load_pretrained_modules(model, cfg, resume_path=resume)

    # Set loss weights (override defaults)
    if hasattr(model, 'det_loss'):
        model.det_loss.w_box = l_cfg.get('w_box', 7.5)
        model.det_loss.w_cls = l_cfg.get('w_cls', 0.5)
        model.det_loss.w_dfl = l_cfg.get('w_dfl', 1.5)
    if hasattr(model, 'pose_loss'):
        if model_name in ('bifpn', 'bifpn_dual',
                          'bifpn_pose', 'bifpn_pose_only'):
            model.pose_loss.w_box = l_cfg.get('w_box', 7.5)
            model.pose_loss.w_cls = l_cfg.get('w_cls', 0.5)
            model.pose_loss.w_dfl = l_cfg.get('w_dfl', 1.5)
        else:
            model.pose_loss.w_box = 0.0
            model.pose_loss.w_cls = 0.0
            model.pose_loss.w_dfl = 0.0
        model.pose_loss.w_pose = l_cfg.get('w_pose', 12.0)
        model.pose_loss.w_kobj = l_cfg.get('w_kobj', 1.0)
    if hasattr(model, 'disable_pose_proposal_training'):
        model.disable_pose_proposal_training()
    if hasattr(model, 'loss_fn'):
        model.loss_fn.w_box = l_cfg.get('w_box', 7.5)
        model.loss_fn.w_cls = l_cfg.get('w_cls', 0.5)
        model.loss_fn.w_dfl = l_cfg.get('w_dfl', 1.5)
        model.loss_fn.w_pose = l_cfg.get('w_pose', 12.0)
        model.loss_fn.w_kobj = l_cfg.get('w_kobj', 1.0)

    # Create val dataloader (shared)
    data_root = Path(opts['data_root'])
    class_id_format = d_cfg.get('class_id_format', 'yolo80')
    print(f"Label format: {class_id_format} (internal COCO ids 0..79)")
    keep_classes = d_cfg.get('keep_classes', None)
    if keep_classes is None and cfg.get('num_det_classes', 80) == 1:
        keep_classes = [0]
    if keep_classes is not None:
        keep_classes = [int(c) for c in keep_classes]
        print(f"Target class filter: keep_classes={keep_classes}")
    data_person_only = bool(d_cfg.get('person_only', False))
    data_require_keypoints = bool(d_cfg.get('require_keypoints', False))
    val_person_only = bool(d_cfg.get('val_person_only', data_person_only))
    val_require_keypoints = bool(d_cfg.get('val_require_keypoints', data_require_keypoints))
    if data_person_only or data_require_keypoints or val_person_only or val_require_keypoints:
        print(
            "Sample filters: "
            f"train_person_only={data_person_only} "
            f"train_require_keypoints={data_require_keypoints} "
            f"val_person_only={val_person_only} "
            f"val_require_keypoints={val_require_keypoints}"
        )
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
        class_id_format=class_id_format,
        keep_classes=keep_classes,
        person_only=val_person_only,
        require_keypoints=val_require_keypoints,
    )
    print(f"Val: {len(val_loader.dataset)} samples")

    validation_cfg = cfg.get('validation', {})
    score_cfg = validation_cfg.get('final_eval', {})
    if not isinstance(score_cfg, dict):
        score_cfg = {'enabled': bool(score_cfg)}
    score_eval_cfg = validation_cfg.get('score_eval', {})
    if not isinstance(score_eval_cfg, dict):
        score_eval_cfg = {'enabled': bool(score_eval_cfg)}
    final_eval_enabled = bool(score_cfg.get('enabled', validation_cfg.get('backend') == 'cocoeval'))
    per_epoch_score_enabled = bool(score_eval_cfg.get('enabled', False))
    score_loader = None
    if final_eval_enabled or per_epoch_score_enabled:
        score_loader = val_loader
        score_samples = int(
            score_eval_cfg.get('max_samples', score_cfg.get('max_samples', 0)) or 0)
        if score_samples > 0 and score_samples < len(val_loader.dataset):
            score_loader = DataLoader(
                Subset(val_loader.dataset, range(score_samples)),
                batch_size=cfg.get('eval', {}).get('batch_size', opts['batch']),
                shuffle=False,
                num_workers=opts['workers'],
                collate_fn=collate_fn,
                pin_memory=_is_cuda_device(device),
            )
            print(f"COCOeval: first {score_samples} val samples")
        else:
            print("COCOeval: full val set")

    score_eval_kwargs = {
        'score_thresh': cfg.get('eval', {}).get('score_thresh', 0.01),
        'iou_thresh': cfg.get('eval', {}).get('iou_thresh', 0.6),
        'max_det': cfg.get('eval', {}).get('max_det', 300),
        'num_classes': cfg.get('num_classes', 80),
        'keep_classes': keep_classes,
        'data_root': data_root,
        'task': validation_cfg.get('task', 'both'),
        'instances_json': validation_cfg.get('instances_json', None),
        'keypoints_json': validation_cfg.get('keypoints_json', None),
        'coco_max_det': cfg.get('eval', {}).get('coco_max_det', 100),
    }

    resume_cfg = t_cfg.get('resume', {}) or {}
    if resume and _config_bool(resume_cfg.get('keep_mosaic_closed', False)):
        opts['no_mosaic'] = True
        print("[Resume] keep_mosaic_closed=true; training mosaic is disabled.")
    close_mosaic = t_cfg.get('close_mosaic_epochs', 10) if not opts['no_mosaic'] else 0

    gradient_projection_cfg = t_cfg.get('gradient_projection', {})
    distillation_cfg = t_cfg.get('distillation', {}) or {}

    def _build_distillation_teacher():
        if not distillation_cfg.get('enabled', False):
            return None
        teacher_model_name = distillation_cfg.get('teacher_model', 'bifpn_detect')
        teacher_weights = distillation_cfg.get('teacher_weights', None)
        if not teacher_weights:
            raise ValueError("training.distillation.teacher_weights is required when distillation is enabled")
        teacher_path = _candidate_path(teacher_weights)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Distillation teacher checkpoint not found: {teacher_path}")
        neck_cfg = cfg.get('neck', {}) or {}
        assigner_cfg = cfg.get('assigner', {}) or {}
        teacher_kwargs = {
            'reg_max': cfg.get('reg_max', 16),
            'num_det_classes': int(distillation_cfg.get('teacher_num_classes', 80)),
            'input_size': d_cfg.get('input_size', 640),
            'neck_use_p2_context': neck_cfg.get('use_p2_context', False),
            'neck_downsample': neck_cfg.get('downsample', 'conv'),
            'neck_out_channels': neck_cfg.get('out_channels', None),
            'assigner_topk': assigner_cfg.get('topk', 10),
            'assigner_alpha': assigner_cfg.get('alpha', 0.5),
            'assigner_beta': assigner_cfg.get('beta', 6.0),
            'assigner_eps': assigner_cfg.get('eps', 1.0e-9),
        }
        teacher = create_model(teacher_model_name, **teacher_kwargs).to(device)
        ckpt = torch.load(str(teacher_path), map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        teacher.load_state_dict(state, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print(f"[Distill] Loaded frozen teacher: {teacher_path}")
        return teacher

    distill_teacher = _build_distillation_teacher()

    def _make_trainer(lr, save_dir_suffix='', gradient_projection_enabled=None,
                      stage_cfg=None):
        save_path = Path(opts['save_dir']) / (model_name + save_dir_suffix)
        use_gradient_projection = (
            gradient_projection_cfg.get('enabled', False)
            if gradient_projection_enabled is None
            else gradient_projection_enabled
        )
        lr_groups_cfg = t_cfg.get('lr_groups', {}) or {}
        stage_cfg = stage_cfg or {}
        backbone_lr = stage_cfg.get(
            'backbone_lr',
            lr_groups_cfg.get('backbone_lr', None))
        backbone_lr_mult = stage_cfg.get(
            'backbone_lr_mult',
            lr_groups_cfg.get('backbone_lr_mult', 1.0))
        return Trainer(
            model=model,
            device=device,
            lr=lr,
            optimizer=opts['optimizer'],
            momentum=t_cfg.get('momentum', 0.937),
            weight_decay=t_cfg.get('weight_decay', 5e-4),
            nesterov=t_cfg.get('nesterov', True),
            final_lr_ratio=t_cfg.get('lrf', 0.01),
            backbone_lr=backbone_lr,
            backbone_lr_mult=backbone_lr_mult,
            param_groups=t_cfg.get('param_groups', 'basic'),
            batch_size=opts['batch'],
            nbs=t_cfg.get('nbs', 64),
            accumulate=t_cfg.get('accumulate', 'auto'),
            scale_weight_decay=t_cfg.get('scale_weight_decay', False),
            cos_lr=t_cfg.get('cos_lr', True),
            warmup_epochs=t_cfg.get('warmup_epochs', 3),
            warmup_momentum=t_cfg.get('warmup_momentum', 0.8),
            warmup_bias_lr=t_cfg.get('warmup_bias_lr', 0.1),
            yolo_warmup=t_cfg.get('yolo_warmup', False),
            grad_clip=t_cfg.get('grad_clip', 10.0),
            log_interval=t_cfg.get('log_interval', 20 if not opts['debug'] else 1),
            save_interval=t_cfg.get('save_interval', 50),
            val_interval=t_cfg.get('val_interval', 5),
            save_dir=str(save_path),
            use_amp=(not opts['no_amp']) and _is_cuda_device(device),
            ema_decay=t_cfg.get('ema_decay', 0.9999),
            save_best_by=t_cfg.get('save_best_by', 'loss'),
            use_tensorboard=t_cfg.get('tensorboard', True),
            early_stop_enabled=t_cfg.get('early_stop', {}).get('enabled', False),
            early_stop_patience=t_cfg.get('early_stop', {}).get('patience', 0),
            early_stop_min_delta=t_cfg.get('early_stop', {}).get('min_delta', 0.0),
            early_stop_start_epoch=t_cfg.get('early_stop', {}).get('start_epoch', 0),
            score_interval=1,
            score_det_baseline=score_eval_cfg.get(
                'det_baseline_mAP50_95',
                score_cfg.get('det_baseline_mAP50_95', 1.0)),
            score_pose_baseline=score_eval_cfg.get(
                'pose_baseline_mAP50_95',
                score_cfg.get('pose_baseline_mAP50_95', 1.0)),
            score_det_metric=score_eval_cfg.get(
                'det_metric', score_cfg.get('det_metric', 'bbox/AP')),
            score_pose_metric=score_eval_cfg.get(
                'pose_metric', score_cfg.get('pose_metric', 'keypoints/AP')),
            score_eval_enabled=per_epoch_score_enabled,
            score_eval_interval=score_eval_cfg.get(
                'interval', t_cfg.get('val_interval', 1)),
            score_joint_metric=score_eval_cfg.get('joint_metric', 'score'),
            gradient_projection_enabled=use_gradient_projection,
            gradient_projection_method=gradient_projection_cfg.get('method', 'pcgrad'),
            gradient_projection_scope=gradient_projection_cfg.get('scope', 'all'),
            gradient_projection_eps=gradient_projection_cfg.get('eps', 1.0e-12),
            cagrad_c=gradient_projection_cfg.get('cagrad_c', 0.5),
            gradnorm_alpha=gradient_projection_cfg.get('gradnorm_alpha', 1.0),
            distill_teacher=distill_teacher,
            distill_cfg=distillation_cfg,
        )

    def _make_train_loader(person_only=None, require_keypoints=None):
        effective_person_only = data_person_only or bool(person_only)
        effective_require_keypoints = data_require_keypoints or bool(require_keypoints)
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
            class_id_format=class_id_format,
            hsv_h=a_cfg.get('hsv_h', 0.015),
            hsv_s=a_cfg.get('hsv_s', 0.7),
            hsv_v=a_cfg.get('hsv_v', 0.4),
            degrees=a_cfg.get('degrees', 0.0),
            translate=a_cfg.get('translate', 0.1),
            scale=a_cfg.get('scale', 0.5),
            shear=a_cfg.get('shear', 0.0),
            perspective=a_cfg.get('perspective', 0.0),
            flip_lr=a_cfg.get('flip_lr', 0.5),
            flip_ud=a_cfg.get('flip_ud', 0.0),
            bgr=a_cfg.get('bgr', 0.0),
            mosaic_prob=a_cfg.get('mosaic_prob', 0.5),
            mixup_prob=a_cfg.get('mixup_prob', 0.0),
            cutmix_prob=a_cfg.get('cutmix_prob', 0.0),
            copy_paste_prob=a_cfg.get('copy_paste_prob', 0.0),
            copy_paste_ioa=a_cfg.get('copy_paste_ioa', 0.3),
            copy_paste_max_objects=a_cfg.get('copy_paste_max_objects', 8),
            keep_classes=keep_classes,
            person_only=effective_person_only,
            require_keypoints=effective_require_keypoints,
        )

    dynamic_cfg = t_cfg.get('dynamic_weights', {})
    task_weights_cfg = t_cfg.get('task_weights', {}) or {}
    base_det_weight = float(task_weights_cfg.get('det', task_weights_cfg.get('detection', 1.0)))
    base_pose_weight = float(task_weights_cfg.get('pose', 1.0))
    if base_det_weight < 0 or base_pose_weight < 0:
        raise ValueError("training.task_weights det/pose must be >= 0")
    print(f"Base task weights: det={base_det_weight:.4g} pose={base_pose_weight:.4g}")
    task_loss_history = []
    dynamic_enabled = bool(dynamic_cfg.get('enabled', False))
    dynamic_method = str(dynamic_cfg.get('method', 'dwa')).lower()
    if dynamic_method in ('none', 'fixed', 'off'):
        dynamic_method = 'fixed'
        dynamic_enabled = False
    if dynamic_method not in ('fixed', 'uncertainty', 'dwa'):
        raise ValueError(
            f"Unsupported dynamic weight method: {dynamic_method}. "
            "Use one of: fixed, uncertainty, dwa.")

    if dynamic_enabled and dynamic_method == 'uncertainty':
        if not hasattr(model, 'enable_uncertainty_weighting'):
            raise ValueError("uncertainty dynamic weighting is only supported by dual-head models")
        if hasattr(model, 'set_uncertainty_weight_bounds'):
            model.set_uncertainty_weight_bounds(
                det_min=dynamic_cfg.get('det_min', None),
                det_max=dynamic_cfg.get('det_max', None),
                pose_min=dynamic_cfg.get('pose_min', None),
                pose_max=dynamic_cfg.get('pose_max', None),
            )
        model.enable_uncertainty_weighting(True)
        print("Dynamic weights: uncertainty weighting enabled")
    elif dynamic_enabled and dynamic_method == 'dwa':
        print(
            "Dynamic weights: DWA enabled "
            f"(temperature={dynamic_cfg.get('temperature', 2.0)}, "
            f"warmup_epochs={dynamic_cfg.get('warmup_epochs', 2)}, "
            f"source={dynamic_cfg.get('source', 'train')})")
    elif hasattr(model, 'enable_uncertainty_weighting'):
        model.enable_uncertainty_weighting(False)

    def _apply_dynamic_weights(epoch):
        if hasattr(model, 'disable_pose_proposal_training'):
            model.disable_pose_proposal_training()
        if hasattr(model, 'train_det') and hasattr(model, 'train_pose'):
            if not model.train_det or not model.train_pose:
                if hasattr(model, 'set_task_weights'):
                    model.set_task_weights(base_det_weight if model.train_det else 0.0,
                                           base_pose_weight if model.train_pose else 0.0)
                return
        if dynamic_enabled and dynamic_method == 'uncertainty':
            return
        if not dynamic_enabled:
            if hasattr(model, 'set_task_weights'):
                model.set_task_weights(base_det_weight, base_pose_weight)
            return
        method = dynamic_method
        if method != 'dwa':
            raise ValueError(f"Unsupported dynamic weight method: {method}")

        num_tasks = 2.0
        temperature = float(dynamic_cfg.get('temperature', 2.0))
        if temperature <= 0:
            raise ValueError("training.dynamic_weights.temperature must be > 0 for DWA")
        warmup_epochs = int(dynamic_cfg.get('warmup_epochs', 2))
        eps = float(dynamic_cfg.get('eps', 1e-8))
        ratio_min = float(dynamic_cfg.get('ratio_min', 0.0))
        ratio_max = float(dynamic_cfg.get('ratio_max', 5.0))

        if epoch < warmup_epochs or len(task_loss_history) < 2:
            det_weight = 1.0
            pose_weight = 1.0
        else:
            prev2 = task_loss_history[-2]
            prev1 = task_loss_history[-1]
            det_ratio = prev1['det_total'] / max(prev2['det_total'], eps)
            pose_ratio = prev1['pose_total'] / max(prev2['pose_total'], eps)
            det_ratio = float(min(max(det_ratio, ratio_min), ratio_max))
            pose_ratio = float(min(max(pose_ratio, ratio_min), ratio_max))
            det_score = torch.exp(torch.tensor(det_ratio / temperature, dtype=torch.float32)).item()
            pose_score = torch.exp(torch.tensor(pose_ratio / temperature, dtype=torch.float32)).item()
            norm = max(det_score + pose_score, eps)
            det_weight = num_tasks * det_score / norm
            pose_weight = num_tasks * pose_score / norm

        det_weight *= base_det_weight
        pose_weight *= base_pose_weight
        det_weight = float(min(max(det_weight, dynamic_cfg.get('det_min', 0.2)), dynamic_cfg.get('det_max', 5.0)))
        pose_weight = float(min(max(pose_weight, dynamic_cfg.get('pose_min', 0.2)), dynamic_cfg.get('pose_max', 5.0)))
        if hasattr(model, 'set_task_weights'):
            model.set_task_weights(det_weight, pose_weight)

    def _record_task_losses(epoch, train_metrics, _val_metrics=None):
        source = str(dynamic_cfg.get('source', 'train')).lower()
        if source == 'val':
            if not _val_metrics:
                return
            det_total = float(_val_metrics.get('val_det_total', 0.0))
            pose_total = float(_val_metrics.get('val_pose_total', 0.0))
        elif source == 'train':
            det_total = float(train_metrics.get('det_total', 0.0))
            pose_total = float(train_metrics.get('pose_total', 0.0))
        else:
            raise ValueError("training.dynamic_weights.source must be 'train' or 'val'")
        task_loss_history.append({
            'epoch': int(epoch),
            'det_total': max(det_total, 1e-8),
            'pose_total': max(pose_total, 1e-8),
        })

    def _compose_epoch_callbacks(*callbacks):
        callbacks = [cb for cb in callbacks if cb]
        if not callbacks:
            return None

        def _callback(epoch):
            for cb in callbacks:
                cb(epoch)
        return _callback

    def _dispose_trainer(trainer):
        if trainer is None:
            return
        writer = getattr(trainer, 'writer', None)
        if writer is not None:
            writer.close()
            trainer.writer = None
        if hasattr(trainer, '_ema_state'):
            trainer._ema_state.clear()
        trainer.optimizer = None
        trainer.scaler = None
        trainer.model = None

    def _release_cuda_cache():
        gc.collect()
        if _is_cuda_device(device):
            torch.cuda.empty_cache()

    def _preserve_resume_lr(trainer, total_epochs):
        """Adjust base_lr so the first resumed epoch keeps checkpoint LR."""
        if not resume or not _config_bool(resume_cfg.get('keep_current_lr', False)):
            return
        groups = trainer.optimizer.param_groups
        if not groups:
            return
        reference_group = next(
            (g for g in groups if abs(float(g.get('lr_mult', 1.0)) - 1.0) < 1e-12),
            groups[0],
        )
        lr_mult = max(float(reference_group.get('lr_mult', 1.0)), 1e-12)
        current_lr = float(reference_group.get('lr', trainer.base_lr * lr_mult)) / lr_mult
        scale = float(resume_cfg.get('lr_scale', 1.0))
        target_lr = current_lr * scale
        factor = max(float(trainer._lr_factor(trainer.current_epoch, total_epochs)), 1e-12)
        old_base_lr = trainer.base_lr
        trainer.base_lr = target_lr / factor
        for group in groups:
            group['initial_lr'] = trainer.base_lr * float(group.get('lr_mult', 1.0))
        print(
            "[Resume] keep_current_lr=true: "
            f"checkpoint_lr={current_lr:.6g}, scale={scale:.4g}, "
            f"epoch={trainer.current_epoch}, total_epochs={total_epochs}, "
            f"base_lr {old_base_lr:.6g}->{trainer.base_lr:.6g}, "
            f"next_lr={target_lr:.6g}"
        )

    def _init_best_from_resume_loss(trainer, val_loader_to_check):
        """Use the resumed checkpoint's validation loss as the initial best."""
        if not resume or not _config_bool(resume_cfg.get('init_best_from_resume', False)):
            return
        if val_loader_to_check is None:
            raise RuntimeError("resume.init_best_from_resume=true requires a validation loader")
        print("[Resume] Computing validation loss for the resumed checkpoint...")
        val_metrics = trainer.validate(val_loader_to_check)
        current = val_metrics.get(
            'val_loss_total',
            val_metrics.get('val_total', float('inf')),
        )
        if not torch.isfinite(torch.tensor(float(current))).item():
            raise RuntimeError(f"Resume validation loss is not finite: {current}")
        trainer.best_metric = float(current)
        print(f"[Resume] Initial best loss set to resumed checkpoint val loss: {trainer.best_metric:.4f}")
        for line in Trainer.format_metric_lines(
                val_metrics,
                label='resume_val',
                indent='  ',
                prefix='val_'):
            print(line)

    def _init_score_bests_from_resume(trainer, score_loader_to_check,
                                      save_prefix):
        """Use the resumed checkpoint's COCO metrics as score-best baselines."""
        if not resume or not _config_bool(
                resume_cfg.get('init_score_best_from_resume', False)):
            return
        if not per_epoch_score_enabled:
            return
        if score_loader_to_check is None:
            raise RuntimeError(
                "resume.init_score_best_from_resume=true requires a COCOeval loader")
        print("[Resume] Computing COCOeval metrics for the resumed checkpoint...")
        score_metrics = trainer.validate_score(
            score_loader_to_check,
            **score_eval_kwargs,
        )
        trainer.init_score_bests(
            score_metrics,
            save_prefix=save_prefix if _config_bool(
                resume_cfg.get('save_score_best_aliases', True)) else None,
        )
        for key, value in score_metrics.items():
            if isinstance(value, float):
                print(f"  resume_coco/{key}: {value:.4f}")

    def _run_preflight(train_loader, val_loader_to_check=None, tag='train'):
        print(f"\n[Preflight] Checking one {tag} batch...")
        model.to(device)
        model.train()
        _apply_frozen_module_eval(model)
        batch = next(iter(train_loader))
        images = batch['image'].to(device, non_blocking=True)
        gt_list = [{
            'boxes': batch['boxes'][i],
            'classes': batch['classes'][i],
            'kpts': batch['kpts'][i],
        } for i in range(len(images))]
        with torch.no_grad():
            if (not opts['no_amp']) and _is_cuda_device(device):
                with torch.amp.autocast('cuda'):
                    losses = model.compute_loss(images, gt_list)
            else:
                losses = model.compute_loss(images, gt_list)
        print("  loss ok:")
        visible_losses = {
            key: value for key, value in losses.items()
            if isinstance(value, torch.Tensor) and not str(key).startswith('_')
        }
        for line in Trainer.format_metric_lines(
                visible_losses,
                label='preflight',
                indent='    '):
            print(line)
        del losses, images, gt_list, batch
        _release_cuda_cache()

        if val_loader_to_check is not None:
            print(f"[Preflight] Checking one {tag} val decode batch...")
            model.eval()
            val_batch = next(iter(val_loader_to_check))
            val_images = val_batch['image'].to(device, non_blocking=True)
            predictions = model.predict_val(
                val_images,
                score_thresh=cfg.get('eval', {}).get('score_thresh', 0.01),
                iou_thresh=cfg.get('eval', {}).get('iou_thresh', 0.6),
                max_det=cfg.get('eval', {}).get('max_det', 300),
            )
            if len(predictions) != len(val_images):
                raise RuntimeError(
                    f"predict_val returned {len(predictions)} predictions for {len(val_images)} images")
            print(f"  decode ok: batch={len(val_images)} preds={len(predictions)}")
            del predictions, val_images, val_batch
            _release_cuda_cache()

    def _set_trainable_stage(stage_cfg):
        for p in model.parameters():
            p.requires_grad = True
        setattr(model, '_frozen_module_roots', [])

        model.train_det = bool(stage_cfg.get('train_det', True))
        model.train_pose = bool(stage_cfg.get('train_pose', True))
        model.det_weight_mult = float(stage_cfg.get('det_weight_mult', 1.0))
        if hasattr(model, 'set_task_weights'):
            model.set_task_weights(
                stage_cfg.get('det_weight', 1.0 if model.train_det else 0.0),
                stage_cfg.get('pose_weight', 1.0 if model.train_pose else 0.0),
            )

        if stage_cfg.get('freeze_backbone', False):
            for p in model.backbone.parameters():
                p.requires_grad = False
        elif stage_cfg.get('trainable_backbone_layers') is not None:
            trainable_layers = stage_cfg.get('trainable_backbone_layers') or []
            if isinstance(trainable_layers, str):
                trainable_layers = [trainable_layers]
            for p in model.backbone.parameters():
                p.requires_grad = False
            for layer_name in trainable_layers:
                module = getattr(model.backbone, str(layer_name), None)
                if module is None:
                    raise ValueError(
                        f"Unknown backbone layer '{layer_name}' in stage "
                        f"'{stage_cfg.get('name', 'unnamed')}'. "
                        f"Available examples: stem, stage3, stage4, "
                        f"stage5_down, stage5_c2f, stage5_sppf."
                    )
                for p in module.parameters():
                    p.requires_grad = True
        if stage_cfg.get('freeze_neck', False):
            for module_name in ('neck', 'det_neck', 'pose_neck'):
                module = getattr(model, module_name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = False
        if stage_cfg.get('freeze_det_head', False):
            model.freeze_head('det')
        if stage_cfg.get('freeze_pose_head', False):
            model.freeze_head('pose')
        stage_freeze_cfg = stage_cfg.get('freeze', stage_cfg)
        if _freeze_cfg_active(stage_freeze_cfg):
            _apply_module_freeze(
                model,
                stage_freeze_cfg,
                reset=False,
                label=stage_cfg.get('name', 'stage'))
        if hasattr(model, 'disable_pose_proposal_training'):
            model.disable_pose_proposal_training()

        use_stage_uncertainty = (
            dynamic_enabled and
            dynamic_method == 'uncertainty' and
            stage_cfg.get('use_dynamic_weights', False)
        )
        if hasattr(model, 'enable_uncertainty_weighting'):
            model.enable_uncertainty_weighting(use_stage_uncertainty)

    def _stage_epoch_callback(stage_cfg):
        def _callback(epoch):
            _set_trainable_stage(stage_cfg)
            if stage_cfg.get('det_weight_warmup_epochs', 0) and hasattr(model, 'det_weight_warmup_epochs'):
                model.det_weight_warmup_epochs = int(stage_cfg.get('det_weight_warmup_epochs', 0))
                model.update_det_weight(epoch)
            if stage_cfg.get('use_dynamic_weights', False):
                _apply_dynamic_weights(epoch)
        return _callback

    staged_cfg = t_cfg.get('staged_training', {})
    is_bifpn_model = model_name in ('bifpn', 'bifpn_dual')
    if staged_cfg.get('enabled') and is_bifpn_model:
        stages = staged_cfg.get('stages', [])
        if not stages:
            raise ValueError("training.staged_training.enabled=true requires a non-empty stages list")

        start_stage = int(staged_cfg.get('start_stage', 1))
        if resume:
            _load_model_weights_for_staged_resume(model, resume, device='cpu')
            resume_name = Path(resume).name.lower()
            resume_parent = Path(resume).parent.name.lower()
            for idx, candidate_stage in enumerate(stages, start=1):
                candidate_name = str(candidate_stage.get('name', f'stage{idx}')).lower()
                if candidate_name in resume_name or candidate_name in resume_parent:
                    start_stage = max(start_stage, idx + 1)
                    break
            print(f"[Resume] Staged training will start from stage {start_stage}")

        print(f"\nStaged training: {len(stages)} stages")
        trainer_stage = None
        for stage_index, stage_cfg in enumerate(stages, start=1):
            if stage_index < start_stage:
                print(f"Skipping stage {stage_index}: {stage_cfg.get('name', f'stage{stage_index}')}")
                continue

            task_loss_history.clear()
            stage_name = stage_cfg.get('name', f'stage{stage_index}')
            stage_epochs = min(3, stage_cfg.get('epochs', 1)) if opts['debug'] else int(stage_cfg.get('epochs', 1))
            stage_lr = float(stage_cfg.get('lr0', opts['lr']))
            stage_person_only = stage_cfg.get('person_only', None)
            stage_require_keypoints = stage_cfg.get('require_keypoints', None)
            effective_person_only = data_person_only or bool(stage_person_only)
            effective_require_keypoints = data_require_keypoints or bool(stage_require_keypoints)

            _set_trainable_stage(stage_cfg)
            train_loader_stage = _make_train_loader(
                person_only=stage_person_only,
                require_keypoints=stage_require_keypoints,
            )
            print(f"\n{'='*60}")
            print(f"Stage {stage_index}: {stage_name} | Epochs: {stage_epochs} | LR: {stage_lr}")
            print(
                f"  train_det={getattr(model, 'train_det', True)} "
                f"train_pose={getattr(model, 'train_pose', True)} "
                f"freeze_backbone={stage_cfg.get('freeze_backbone', False)} "
                f"trainable_backbone_layers={stage_cfg.get('trainable_backbone_layers', 'all')} "
                f"person_only={effective_person_only} "
                f"require_keypoints={effective_require_keypoints} "
                f"dynamic={stage_cfg.get('use_dynamic_weights', False)} "
                f"grad_proj={stage_cfg.get('use_gradient_projection', gradient_projection_cfg.get('enabled', False))}"
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  trainable={trainable / 1e6:.2f}M / {total / 1e6:.2f}M")
            print(f"  samples={len(train_loader_stage.dataset)}")
            print(f"{'='*60}")

            _run_preflight(train_loader_stage, val_loader, tag=stage_name)
            trainer_stage = _make_trainer(
                stage_lr,
                save_dir_suffix='' if stage_index == len(stages) else f'_{stage_name}',
                gradient_projection_enabled=stage_cfg.get(
                    'use_gradient_projection',
                    gradient_projection_cfg.get('enabled', False),
                ),
                stage_cfg=stage_cfg)
            trainer_stage.fit(
                epochs=stage_epochs,
                train_loader=train_loader_stage,
                val_loader=val_loader,
                save_prefix=model_name if stage_index == len(stages) else f"{model_name}_{stage_name}",
                close_mosaic_epochs=stage_cfg.get('close_mosaic_epochs', close_mosaic),
                on_epoch_start=_stage_epoch_callback(stage_cfg),
                on_epoch_end=_record_task_losses,
                score_loader=score_loader,
                score_eval_kwargs=score_eval_kwargs,
                final_score_eval=final_eval_enabled and stage_index == len(stages),
            )

            if stage_index < len(stages) and stage_cfg.get('load_best_for_next_stage', True):
                stage_save_dir = Path(opts['save_dir']) / (
                    model_name if stage_index == len(stages) else f"{model_name}_{stage_name}")
                stage_save_prefix = (
                    model_name if stage_index == len(stages) else f"{model_name}_{stage_name}")
                best_ckpt = stage_save_dir / f"{stage_save_prefix}_best.pt"
                if best_ckpt.exists():
                    _load_model_weights_for_staged_resume(model, best_ckpt, device=device)
                    print(f"[Stage handoff] Loaded best checkpoint for next stage: {best_ckpt}")
                else:
                    print(f"[Stage handoff] Best checkpoint not found, keeping last weights: {best_ckpt}")

            if hasattr(train_loader_stage, '_iterator') and train_loader_stage._iterator is not None:
                train_loader_stage._iterator._shutdown_workers()
            _dispose_trainer(trainer_stage)
            del train_loader_stage
            del trainer_stage
            trainer_stage = None
            _release_cuda_cache()

        print(f"\nStaged training complete for {model_name}!")
    else:
        # Single-stage training (original flow)
        single_freeze_cfg = t_cfg.get('freeze', {}) or {}
        if _freeze_cfg_active(single_freeze_cfg):
            _apply_module_freeze(model, single_freeze_cfg, reset=True, label='single-stage')
            if hasattr(model, 'disable_pose_proposal_training'):
                model.disable_pose_proposal_training()
        train_loader = _make_train_loader()
        print(f"Train: {len(train_loader.dataset)} samples")
        _run_preflight(train_loader, val_loader, tag='single-stage')

        trainer = _make_trainer(opts['lr'])
        if resume:
            reset_ema_from_model = _config_bool(
                resume_cfg.get(
                    'reset_ema_from_model',
                    resume_cfg.get('init_best_from_resume', False),
                )
            )
            trainer.load(resume, reset_ema_from_model=reset_ema_from_model)

        epochs = 3 if opts['debug'] else opts['epochs']
        _preserve_resume_lr(trainer, epochs)
        _init_best_from_resume_loss(trainer, val_loader)
        _init_score_bests_from_resume(trainer, score_loader, model_name)

        trainer.fit(
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            save_prefix=model_name,
            close_mosaic_epochs=close_mosaic,
            on_epoch_start=_apply_dynamic_weights,
            on_epoch_end=_record_task_losses,
            score_loader=score_loader,
            score_eval_kwargs=score_eval_kwargs,
            final_score_eval=final_eval_enabled,
        )

        print(f"\nTraining complete for {model_name}!")


if __name__ == '__main__':
    main()





