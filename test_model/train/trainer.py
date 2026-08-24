"""Training loop with SGD/AdamW + cosine LR + EMA + AMP.

Key features:
- SGD optimizer with momentum or AdamW
- Cosine LR schedule with linear warmup
- EMA (exponential moving average)
- AMP mixed precision
- Mosaic scheduling (disable in last N epochs)
- Optional final COCOeval after training
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    """Generic trainer for YOLOv8-based multi-head models.

    Args:
        model: nn.Module with compute_loss(images, gt_dict_list) -> dict
        device: 'cuda' or 'cpu'
        lr: Base learning rate
        momentum: SGD momentum
        weight_decay: L2 weight decay
        warmup_epochs: Linear LR warmup duration
        grad_clip: Max gradient norm (0 = disabled)
        log_interval: Steps between logging
        save_interval: Epochs between saving intermediate checkpoints
        val_interval: Epochs between validation
        save_dir: Checkpoint output directory
        use_amp: Enable AMP mixed precision
        ema_decay: EMA decay rate (0 = disabled)
        save_best_by: 'loss'. Checkpoint selection is intentionally loss-only.
    """

    def __init__(self, model, device='cuda',
                 lr=0.01, momentum=0.937, weight_decay=5e-4,
                 optimizer='sgd', nesterov=True, final_lr_ratio=0.01,
                 backbone_lr=None, backbone_lr_mult=1.0,
                 param_groups='basic', batch_size=16, nbs=64,
                 accumulate='auto', scale_weight_decay=False,
                 cos_lr=True,
                 warmup_epochs=3, warmup_momentum=0.8,
                 warmup_bias_lr=0.1, yolo_warmup=False,
                 grad_clip=10.0,
                 log_interval=20, save_interval=20, val_interval=5,
                 save_dir='checkpoints', use_amp=True,
                 ema_decay=0.9999, save_best_by='loss',
                 use_tensorboard=False, check_finite_loss=False,
                 early_stop_enabled=False, early_stop_patience=0,
                 early_stop_min_delta=0.0, early_stop_start_epoch=0,
                 score_interval=1, score_det_baseline=1.0,
                 score_pose_baseline=1.0, score_det_metric='mAP@0.5:0.95',
                 score_pose_metric='AP_pose@0.5:0.95',
                 score_eval_enabled=False, score_eval_interval=1,
                 score_joint_metric='score',
                 gradient_projection_enabled=False,
                 gradient_projection_method='pcgrad',
                 gradient_projection_scope='all',
                 gradient_projection_eps=1.0e-12,
                 pomsi_alpha=1.0, pomsi_alpha_lr=1.0e-3,
                 pomsi_alpha_min=0.0, pomsi_alpha_max=5.0,
                 pomsi_static=False,
                 cagrad_c=0.5, gradnorm_alpha=1.0,
                 distill_teacher=None, distill_cfg=None):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.save_dir / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.optimizer_name = str(optimizer).lower()
        self.base_lr = lr
        self.momentum = momentum
        self.batch_size = max(int(batch_size), 1)
        self.nbs = max(int(nbs), 1)
        if accumulate in (None, 'auto'):
            self.accumulate = max(round(self.nbs / self.batch_size), 1)
        else:
            self.accumulate = max(int(accumulate), 1)
        self.param_group_mode = str(param_groups or 'basic').lower()
        self.scale_weight_decay = bool(scale_weight_decay)
        self.weight_decay = float(weight_decay)
        if self.scale_weight_decay:
            self.weight_decay = self.weight_decay * self.batch_size * self.accumulate / self.nbs
        self.backbone_lr = None if backbone_lr is None else float(backbone_lr)
        self.backbone_lr_mult = float(backbone_lr_mult)
        param_groups = self._build_param_groups()
        self._validate_param_groups(param_groups)
        if self.optimizer_name in ('adamw', 'adam'):
            opt_decay = 0.0 if self.param_group_mode == 'yolo' else self.weight_decay
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=opt_decay)
        elif self.optimizer_name == 'sgd':
            # Use regular SGD. Fused SGD can fail when multi-head batches leave
            # some head parameters without gradients.
            opt_decay = 0.0 if self.param_group_mode == 'yolo' else self.weight_decay
            optim_kw = dict(lr=lr, momentum=momentum, weight_decay=opt_decay, nesterov=nesterov)
            self.optimizer = torch.optim.SGD(param_groups, **optim_kw)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', self.base_lr * float(group.get('lr_mult', 1.0)))
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = float(warmup_momentum)
        self.warmup_bias_lr = float(warmup_bias_lr)
        self.yolo_warmup = bool(yolo_warmup)
        self.final_lr_ratio = final_lr_ratio
        self.cos_lr = cos_lr

        self.current_epoch = 0
        self.global_step = 0
        self.save_best_by = save_best_by
        self.best_metric = float('inf') if save_best_by == 'loss' else -float('inf')
        self.check_finite_loss = check_finite_loss
        self.early_stop_enabled = early_stop_enabled and early_stop_patience > 0
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.early_stop_start_epoch = int(early_stop_start_epoch)
        self.early_stop_bad_epochs = 0
        self.score_interval = max(1, int(score_interval))
        self.score_det_baseline = max(float(score_det_baseline), 1e-8)
        self.score_pose_baseline = max(float(score_pose_baseline), 1e-8)
        self.score_det_metric = score_det_metric
        self.score_pose_metric = score_pose_metric
        self.score_eval_enabled = bool(score_eval_enabled)
        self.score_eval_interval = max(1, int(score_eval_interval))
        self.score_joint_metric = str(score_joint_metric or 'score')
        self.best_score_metrics = {
            'joint': float('-inf'),
            'det': float('-inf'),
            'pose': float('-inf'),
        }
        self.gradient_projection_enabled = bool(gradient_projection_enabled)
        self.gradient_projection_method = str(gradient_projection_method or 'pcgrad').lower()
        self.gradient_projection_scope = str(gradient_projection_scope or 'all').lower()
        self.gradient_projection_eps = float(gradient_projection_eps)
        self.pomsi_alpha = torch.tensor(float(pomsi_alpha), device=self.device, requires_grad=True)
        self.pomsi_alpha_lr = float(pomsi_alpha_lr)
        self.pomsi_alpha_min = float(pomsi_alpha_min)
        self.pomsi_alpha_max = float(pomsi_alpha_max)
        self.pomsi_static = bool(pomsi_static)
        self.cagrad_c = float(cagrad_c)
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.distill_teacher = distill_teacher
        self.distill_cfg = distill_cfg or {}

        # EMA
        self.ema_decay = ema_decay
        self.ema_enabled = ema_decay > 0
        self._ema_state = {}
        if self.ema_enabled:
            self._build_ema()

        # AMP
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def _apply_frozen_module_eval(self):
        roots = getattr(self.model, '_frozen_module_roots', None) or []
        if not roots:
            return
        for module_name, module in self.model.named_modules():
            if module_name and any(
                    module_name == root or module_name.startswith(root + '.')
                    for root in roots):
                module.eval()

    def _build_param_groups(self):
        if self.param_group_mode == 'yolo':
            norm_types = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
            backbone_lr_mult = float(self._configured_backbone_lr_mult())
            buckets = {
                ('other', 'weight'): [],
                ('other', 'bn'): [],
                ('other', 'bias'): [],
                ('backbone', 'weight'): [],
                ('backbone', 'bn'): [],
                ('backbone', 'bias'): [],
            }
            for module_name, module in self.model.named_modules():
                scope = 'backbone' if module_name == 'backbone' or module_name.startswith('backbone.') else 'other'
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    if module_name == '' and param.ndim <= 1:
                        kind = 'bias'
                    elif 'bias' in param_name:
                        kind = 'bias'
                    elif isinstance(module, norm_types):
                        kind = 'bn'
                    else:
                        kind = 'weight'
                    buckets[(scope, kind)].append(param)

            groups = []
            for scope, lr_mult in (('other', 1.0), ('backbone', backbone_lr_mult)):
                for kind, wd in (('weight', self.weight_decay), ('bn', 0.0), ('bias', 0.0)):
                    params = buckets[(scope, kind)]
                    if params:
                        groups.append({
                            'params': params,
                            'name': f'{scope}_{kind}' if scope == 'backbone' else kind,
                            'param_group': kind,
                            'weight_decay': wd,
                            'lr_mult': lr_mult,
                        })
            if not groups:
                raise ValueError("No trainable parameters found for optimizer")
            return groups

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.'):
                backbone_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({
                'params': other_params,
                'name': 'other',
                'lr_mult': 1.0,
            })
        if backbone_params:
            lr_mult = self.backbone_lr_mult
            if self.backbone_lr is not None:
                lr_mult = self.backbone_lr / max(self.base_lr, 1e-12)
            groups.append({
                'params': backbone_params,
                'name': 'backbone',
                'lr_mult': float(lr_mult),
            })

        if not groups:
            raise ValueError("No trainable parameters found for optimizer")
        return groups

    def _validate_param_groups(self, param_groups):
        expected = {
            id(param): name
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        seen = {}
        duplicates = []
        for group in param_groups:
            for param in group.get('params', []):
                param_id = id(param)
                if param_id in seen:
                    duplicates.append(expected.get(param_id, '<unnamed>'))
                seen[param_id] = True

        missing = [
            name for param_id, name in expected.items()
            if param_id not in seen
        ]
        extra = [
            expected.get(param_id, '<unnamed>')
            for param_id in seen
            if param_id not in expected
        ]
        if missing or duplicates or extra:
            details = []
            if missing:
                details.append(
                    "missing=" + ", ".join(missing[:20]) +
                    ("..." if len(missing) > 20 else ""))
            if duplicates:
                details.append(
                    "duplicates=" + ", ".join(duplicates[:20]) +
                    ("..." if len(duplicates) > 20 else ""))
            if extra:
                details.append(
                    "extra=" + ", ".join(extra[:20]) +
                    ("..." if len(extra) > 20 else ""))
            raise RuntimeError(
                "Optimizer parameter groups do not exactly cover trainable "
                "parameters: " + "; ".join(details))

    def _configured_backbone_lr_mult(self):
        if self.backbone_lr is not None:
            return self.backbone_lr / max(self.base_lr, 1e-12)
        return self.backbone_lr_mult

    def _apply_configured_lr_groups(self):
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get('name')
            if name is None and len(self.optimizer.param_groups) == 2:
                name = 'other' if idx == 0 else 'backbone'
                group['name'] = name
            if name == 'backbone' or str(name).startswith('backbone_'):
                group['lr_mult'] = float(self._configured_backbone_lr_mult())
            elif name == 'other' or name in ('weight', 'bn', 'bias'):
                group['lr_mult'] = 1.0

    def _clamp_dynamic_model_state(self):
        if hasattr(self.model, 'clamp_uncertainty_parameters'):
            self.model.clamp_uncertainty_parameters()
        if not (self.ema_enabled and self._ema_state):
            return
        if not hasattr(self.model, 'uncertainty_log_var_bounds'):
            return
        for name, (min_log_var, max_log_var) in self.model.uncertainty_log_var_bounds().items():
            tensor = self._ema_state.get(name)
            if tensor is None:
                continue
            if min_log_var is None and max_log_var is None:
                continue
            tensor.clamp_(
                min=-float('inf') if min_log_var is None else min_log_var,
                max=float('inf') if max_log_var is None else max_log_var,
            )

    @staticmethod
    def _should_log_loss_key(key):
        return not str(key).startswith('_')

    @staticmethod
    def _format_metric_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    @classmethod
    def _format_metric_pairs(cls, metrics, keys, prefix=''):
        parts = []
        for key, label in keys:
            full_key = prefix + key
            if full_key in metrics:
                parts.append(f"{label}={cls._format_metric_value(metrics[full_key])}")
        return parts

    @classmethod
    def format_metric_lines(cls, metrics, label='train', indent='  ', prefix='',
                            extra=None):
        """Return compact grouped console lines for train/val loss metrics."""
        extra = extra or {}
        groups = [
            ('det', [
                ('det_ciou', 'ciou'),
                ('det_cls', 'cls'),
                ('det_dfl', 'dfl'),
                ('det_total', 'total'),
                ('det_num_pos', 'pos'),
                ('target_scores_sum_det', 'score_sum'),
            ]),
            ('pose', [
                ('pose_det_ciou', 'det_ciou'),
                ('pose_det_cls', 'det_cls'),
                ('pose_det_dfl', 'det_dfl'),
                ('pose_det_total', 'det_total'),
                ('pose_kobj', 'kobj'),
                ('pose_kpt', 'kpt'),
                ('pose_kpt_total', 'kpt_total'),
                ('pose_total', 'total'),
                ('pose_num_pos', 'pos'),
                ('target_scores_sum_pose', 'score_sum'),
            ]),
            ('attr', [
                ('attr_bce', 'bce'),
                ('attr_smoking', 'smoking'),
                ('attr_falling', 'falling'),
                ('attr_waving', 'waving'),
                ('attr_helmet_on', 'helmet'),
                ('attr_consistency', 'cons'),
                ('attr_total', 'total'),
                ('attr_count', 'count'),
                ('task_w_attr', 'w'),
            ]),
            ('dyn', [
                ('dyn_w_det', 'w_det'),
                ('dyn_w_pose', 'w_pose'),
            ]),
            ('distill', [
                ('distill_total', 'total'),
                ('distill_cls', 'cls'),
                ('distill_reg', 'reg'),
                ('distill_feat', 'feat'),
            ]),
            ('misc', [
                ('loss_total', 'loss'),
                ('total', 'obj'),
                ('num_pos', 'num_pos'),
                ('target_scores_sum', 'score_sum'),
                ('pomsi_alpha', 'pomsi_alpha'),
                ('skipped', 'skipped'),
            ]),
        ]

        lines = []
        for group_name, keys in groups:
            parts = cls._format_metric_pairs(metrics, keys, prefix=prefix)
            if group_name == 'misc':
                for key, value in extra.items():
                    parts.append(f"{key}={cls._format_metric_value(value)}")
            if parts:
                lines.append(f"{indent}{label}/{group_name}: " + " ".join(parts))

        class_names = [
            key[len(prefix + 'det_cls_'):]
            for key in metrics
            if key.startswith(prefix + 'det_cls_')
        ]
        if class_names:
            parts = []
            for name in class_names:
                base = prefix + f'det_cls_{name}'
                valid = prefix + f'det_valid_images_{name}'
                positive = prefix + f'det_pos_anchors_{name}'
                parts.append(
                    f"{name}(bce={cls._format_metric_value(metrics[base])},"
                    f"img={cls._format_metric_value(metrics.get(valid, 0))},"
                    f"pos={cls._format_metric_value(metrics.get(positive, 0))})"
                )
            lines.append(f"{indent}{label}/det_classes: " + " ".join(parts))
        return lines

    def _collect_task_grads(self, task_loss, params, retain_graph):
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler:
            self.scaler.scale(task_loss).backward(retain_graph=retain_graph)
        else:
            task_loss.backward(retain_graph=retain_graph)
        grads = []
        for p in params:
            grads.append(None if p.grad is None else p.grad.detach().clone())
        return grads

    def _grad_dot(self, grads_a, grads_b):
        dot = None
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            value = torch.sum(ga * gb)
            dot = value if dot is None else dot + value
        if dot is None:
            return torch.tensor(0.0, device=self.device)
        return dot

    def _grad_norm_sq(self, grads):
        norm = None
        for grad in grads:
            if grad is None:
                continue
            value = torch.sum(grad * grad)
            norm = value if norm is None else norm + value
        if norm is None:
            return torch.tensor(0.0, device=self.device)
        return norm

    def _grad_cosine(self, grads_a, grads_b):
        denom = torch.sqrt(
            self._grad_norm_sq(grads_a) * self._grad_norm_sq(grads_b)
            + self.gradient_projection_eps
        )
        if denom.item() <= self.gradient_projection_eps:
            return torch.tensor(0.0, device=self.device)
        return self._grad_dot(grads_a, grads_b) / denom

    def _gradient_param_groups(self):
        scoped = []
        other = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            in_scope = True
            if self.gradient_projection_scope == 'shared':
                in_scope = (
                    name.startswith('backbone.') or
                    name.startswith('neck.')
                )
            elif self.gradient_projection_scope != 'all':
                roots = [
                    root.strip()
                    for root in self.gradient_projection_scope.split(',')
                    if root.strip()
                ]
                in_scope = any(name == root or name.startswith(root + '.') for root in roots)
            (scoped if in_scope else other).append(param)
        return scoped, other

    @staticmethod
    def _clone_param_grads(params):
        return [None if p.grad is None else p.grad.detach().clone() for p in params]

    @staticmethod
    def _scale_grads(grads, scale):
        return [None if g is None else g * scale for g in grads]

    def _pcgrad(self, task_grads):
        projected = []
        for i, grads_i in enumerate(task_grads):
            grads_proj = [None if g is None else g.clone() for g in grads_i]
            for j, grads_j in enumerate(task_grads):
                if i == j:
                    continue
                dot = self._grad_dot(grads_proj, grads_j)
                if dot.item() >= 0:
                    continue
                denom = self._grad_norm_sq(grads_j) + self.gradient_projection_eps
                scale = dot / denom
                for k, (g_proj, g_ref) in enumerate(zip(grads_proj, grads_j)):
                    if g_proj is not None and g_ref is not None:
                        grads_proj[k] = g_proj - scale * g_ref
            projected.append(grads_proj)
        return projected

    def _pomsi_phi(self, grads_i, grads_j):
        norm_i = torch.sqrt(self._grad_norm_sq(grads_i) + self.gradient_projection_eps)
        norm_j = torch.sqrt(self._grad_norm_sq(grads_j) + self.gradient_projection_eps)
        base = (2.0 * norm_i * norm_j) / (
            norm_i.pow(2) + norm_j.pow(2) + self.gradient_projection_eps
        )
        base = base.clamp(min=self.gradient_projection_eps, max=1.0)
        return base.pow(self.pomsi_alpha)

    def _update_pomsi_alpha(self, loss_norm):
        if self.pomsi_static or self.pomsi_alpha_lr <= 0.0:
            return
        grad = torch.autograd.grad(
            loss_norm,
            self.pomsi_alpha,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grad is None or not torch.isfinite(grad.detach()).item():
            return
        with torch.no_grad():
            self.pomsi_alpha.sub_(self.pomsi_alpha_lr * grad)
            self.pomsi_alpha.clamp_(self.pomsi_alpha_min, self.pomsi_alpha_max)
        self.pomsi_alpha = self.pomsi_alpha.detach().requires_grad_(True)

    def _pomsi_grads(self, task_grads):
        """POMSI: PCGrad followed by MSI with a learnable alpha."""
        if len(task_grads) < 2:
            return task_grads

        projected = self._pcgrad(task_grads)
        reference_norms = [
            torch.sqrt(self._grad_norm_sq(grads) + self.gradient_projection_eps)
            for grads in projected
        ]
        mean_reference_norm = torch.stack(reference_norms).mean().detach()

        scaled_grads = []
        scaled_norms = []
        for i, grads_i in enumerate(projected):
            grads_pm = [None if g is None else g.clone() for g in grads_i]
            for j, grads_j in enumerate(projected):
                if i == j:
                    continue
                scale = self._pomsi_phi(grads_pm, grads_j)
                current_norm = torch.sqrt(self._grad_norm_sq(grads_pm) + self.gradient_projection_eps)
                ref_norm = reference_norms[j]
                if current_norm.item() >= ref_norm.item():
                    grads_pm = self._scale_grads(grads_pm, scale)
                else:
                    grads_pm = self._scale_grads(
                        grads_pm,
                        1.0 / scale.clamp(min=self.gradient_projection_eps),
                    )
            scaled_grads.append(grads_pm)
            scaled_norms.append(torch.sqrt(self._grad_norm_sq(grads_pm) + self.gradient_projection_eps))

        loss_norm = torch.stack([
            torch.abs(norm - mean_reference_norm)
            for norm in scaled_norms
        ]).sum()
        self._update_pomsi_alpha(loss_norm)
        return [[None if g is None else g.detach() for g in grads] for grads in scaled_grads]

    def _gradnorm_grads(self, task_grads):
        norms = [
            torch.sqrt(self._grad_norm_sq(grads) + self.gradient_projection_eps)
            for grads in task_grads
        ]
        avg_norm = torch.stack(norms).mean()
        normalized = []
        for grads, norm in zip(task_grads, norms):
            scale = (avg_norm / (norm + self.gradient_projection_eps)).pow(self.gradnorm_alpha)
            normalized.append(self._scale_grads(grads, scale.detach()))
        return normalized

    def _cagrad_grads(self, task_grads):
        if len(task_grads) != 2:
            return task_grads
        g1, g2 = task_grads
        dot12 = self._grad_dot(g1, g2)
        n1 = self._grad_norm_sq(g1)
        n2 = self._grad_norm_sq(g2)
        denom = n1 + n2 - 2.0 * dot12
        if denom.abs().item() <= self.gradient_projection_eps:
            alpha = torch.tensor(0.5, device=self.device)
        else:
            alpha = ((n2 - dot12) / (denom + self.gradient_projection_eps)).clamp(0.0, 1.0)
        avg = []
        min_norm = []
        for ga, gb in zip(g1, g2):
            if ga is None and gb is None:
                avg.append(None)
                min_norm.append(None)
            elif ga is None:
                avg.append(gb * 0.5)
                min_norm.append(gb * (1.0 - alpha))
            elif gb is None:
                avg.append(ga * 0.5)
                min_norm.append(ga * alpha)
            else:
                avg.append((ga + gb) * 0.5)
                min_norm.append(alpha * ga + (1.0 - alpha) * gb)
        c = min(max(self.cagrad_c, 0.0), 1.0)
        combined = []
        for ga, gm in zip(avg, min_norm):
            if ga is None and gm is None:
                combined.append(None)
            elif ga is None:
                combined.append(gm)
            elif gm is None:
                combined.append(ga)
            else:
                combined.append((1.0 - c) * ga + c * gm)
        return [combined]

    def _aux_to_pose_grads(self, task_grads):
        """Protect pose by projecting only conflicting auxiliary gradients.

        The last gradient list is treated as the reference pose gradient and
        kept unchanged. Every preceding task gradient (detection, attributes,
        or any future auxiliary task) has only its pose-conflicting component
        removed. Helpful auxiliary components are retained.
        """
        if len(task_grads) < 2:
            return task_grads
        pose_grads = task_grads[-1]
        pose_norm_sq = self._grad_norm_sq(pose_grads)
        safe_aux_grads = []
        for aux_grads in task_grads[:-1]:
            dot = self._grad_dot(aux_grads, pose_grads)
            if dot.item() < 0 and pose_norm_sq.item() > self.gradient_projection_eps:
                scale = dot / (pose_norm_sq + self.gradient_projection_eps)
                aux_grads = [
                    None if ga is None else (
                        ga - scale * gp if gp is not None else ga
                    )
                    for ga, gp in zip(aux_grads, pose_grads)
                ]
            safe_aux_grads.append(aux_grads)

        combined = [None if g is None else g.clone() for g in pose_grads]
        for aux_grads in safe_aux_grads:
            for idx, grad in enumerate(aux_grads):
                if grad is None:
                    continue
                combined[idx] = grad if combined[idx] is None else combined[idx] + grad
        return [combined]

    def _combine_task_grads(self, task_grads):
        method = self.gradient_projection_method
        if method == 'pcgrad':
            return self._pcgrad(task_grads)
        if method == 'pomsi':
            return self._pomsi_grads(task_grads)
        if method == 'gradnorm':
            return self._gradnorm_grads(task_grads)
        if method == 'cagrad':
            return self._cagrad_grads(task_grads)
        if method in ('det_to_pose', 'attr_det_to_pose', 'aux_to_pose', 'pose_guard', 'asymmetric_pcgrad'):
            return self._aux_to_pose_grads(task_grads)
        raise ValueError(
            f"Unsupported gradient projection method: {method}. "
            "Use pcgrad, pomsi, gradnorm, cagrad, or aux_to_pose.")

    def _apply_gradient_projection(self, losses, total_loss):
        det_loss = losses.get('_gp_det_loss')
        attr_loss = losses.get('_gp_attr_loss')
        pose_loss = losses.get('_gp_pose_loss')
        method = self.gradient_projection_method
        pose_available = isinstance(pose_loss, torch.Tensor) and pose_loss.requires_grad
        aux_to_pose = method in ('det_to_pose', 'attr_det_to_pose', 'aux_to_pose', 'pose_guard', 'asymmetric_pcgrad')
        if aux_to_pose:
            if not pose_available:
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                return False
            task_losses = [
                loss for loss in (det_loss, attr_loss)
                if isinstance(loss, torch.Tensor) and loss.requires_grad
            ]
            task_losses.append(pose_loss)
        else:
            task_losses = [
                loss for loss in (det_loss, attr_loss, pose_loss)
                if isinstance(loss, torch.Tensor) and loss.requires_grad
            ]
        if len(task_losses) < 2:
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            return False

        params, other_params = self._gradient_param_groups()
        if not params:
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            return False

        # Preserve gradients from earlier micro-batches. The projection itself
        # is computed for the current batch, then merged back into the
        # accumulation buffer below.
        previous_scoped_grads = self._clone_param_grads(params)
        previous_other_grads = self._clone_param_grads(other_params)

        other_grads = []
        if other_params:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(total_loss).backward(retain_graph=True)
            else:
                total_loss.backward(retain_graph=True)
            other_grads = self._clone_param_grads(other_params)

        task_grads = [
            self._collect_task_grads(loss, params, retain_graph=(idx < len(task_losses) - 1))
            for idx, loss in enumerate(task_losses)
        ]
        projected = self._combine_task_grads(task_grads)

        self.optimizer.zero_grad(set_to_none=True)
        for param_index, p in enumerate(params):
            grad_sum = None
            for grads in projected:
                grad = grads[param_index]
                if grad is None:
                    continue
                grad_sum = grad if grad_sum is None else grad_sum + grad
            previous = previous_scoped_grads[param_index]
            if grad_sum is None:
                grad_sum = previous
            elif previous is not None:
                grad_sum = previous + grad_sum
            if grad_sum is not None:
                p.grad = grad_sum
        for p, grad, previous in zip(other_params, other_grads, previous_other_grads):
            if grad is None:
                grad = previous
            elif previous is not None:
                grad = previous + grad
            if grad is not None:
                p.grad = grad
        return True

    def _build_ema(self):
        self._ema_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
            if torch.is_floating_point(tensor)
        }

    def reset_ema_from_model(self):
        """Reset EMA shadow to the currently loaded model weights."""
        if self.ema_enabled:
            self._build_ema()

    def _update_ema(self):
        d = self.ema_decay
        current_state = self.model.state_dict()
        for name, ema_tensor in self._ema_state.items():
            if name in current_state:
                ema_tensor.mul_(d).add_(current_state[name].detach(), alpha=1 - d)

    def _swap_ema(self, to_ema=True):
        """Swap model weights with EMA shadow."""
        if not self.ema_enabled:
            return
        current_state = self.model.state_dict()
        for name, ema_tensor in self._ema_state.items():
            if name in current_state:
                tmp = current_state[name].detach().clone()
                current_state[name].copy_(ema_tensor)
                self._ema_state[name] = tmp

    def _lr_factor(self, epoch, max_epochs):
        if self.cos_lr:
            progress = epoch / max(max_epochs, 1)
            return self.final_lr_ratio + (1.0 - self.final_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress))
        return max(1 - epoch / max(max_epochs, 1), 0) * (1.0 - self.final_lr_ratio) + self.final_lr_ratio

    def _get_lr(self, epoch, max_epochs):
        if not self.yolo_warmup and epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs)
            return self.base_lr * (0.1 + 0.9 * progress)  # start at 0.1*lr
        return self.base_lr * self._lr_factor(epoch, max_epochs)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * float(pg.get('lr_mult', 1.0))

    def _set_yolo_warmup(self, ni, nw, epoch, max_epochs):
        target_factor = self._lr_factor(epoch, max_epochs)
        for pg in self.optimizer.param_groups:
            target_lr = float(pg.get('initial_lr', self.base_lr)) * target_factor
            start_lr = self.warmup_bias_lr if pg.get('param_group') == 'bias' else 0.0
            pg['lr'] = float(np.interp(ni, [0, nw], [start_lr, target_lr]))
            if 'momentum' in pg:
                pg['momentum'] = float(np.interp(
                    ni, [0, nw], [self.warmup_momentum, self.momentum]))

    def _is_improved(self, current):
        if self.save_best_by == 'loss':
            return current < self.best_metric - self.early_stop_min_delta
        return current > self.best_metric + self.early_stop_min_delta

    @staticmethod
    def _finite_float(value, default=float('-inf')):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def init_score_bests(self, metrics, save_prefix=None):
        """Initialize COCOeval best trackers from an existing checkpoint."""
        if not metrics:
            return
        joint = self._finite_float(metrics.get(self.score_joint_metric))
        if joint == float('-inf'):
            joint = self._finite_float(metrics.get('score'))
        if joint == float('-inf'):
            joint = self._finite_float(metrics.get('mean_mAP50_95'))
        det = self._finite_float(metrics.get(self.score_det_metric))
        pose = self._finite_float(metrics.get(self.score_pose_metric))
        self.best_score_metrics.update({
            'joint': joint,
            'det': det,
            'pose': pose,
        })
        print(
            "[Resume] Initial COCO bests from resumed checkpoint: "
            f"joint={joint:.4f} det={det:.4f} pose={pose:.4f}"
        )
        if save_prefix:
            if joint != float('-inf'):
                self.save(self.save_dir / f"{save_prefix}_coco_best.pt", metrics)
            if det != float('-inf'):
                self.save(self.save_dir / f"{save_prefix}_det_best.pt", metrics)
            if pose != float('-inf'):
                self.save(self.save_dir / f"{save_prefix}_pose_best.pt", metrics)

    def _format_score_line(self, metrics, label='val/coco', indent='  '):
        keys = [
            (self.score_joint_metric, 'joint'),
            ('score', 'score'),
            (self.score_det_metric, 'det'),
            (self.score_pose_metric, 'pose'),
            ('det_ratio', 'det_ratio'),
            ('pose_ratio', 'pose_ratio'),
            ('mean_mAP50_95', 'mean'),
            ('joint_mAP50_95', 'min_ap'),
        ]
        seen = set()
        parts = []
        for key, alias in keys:
            if key in seen or key not in metrics:
                continue
            seen.add(key)
            parts.append(f"{alias}={self._format_metric_value(metrics[key])}")
        if not parts:
            return None
        return f"{indent}{label}: " + " ".join(parts)

    def _save_score_bests(self, score_metrics, save_prefix):
        statuses = []
        joint = self._finite_float(score_metrics.get(self.score_joint_metric))
        if joint == float('-inf'):
            joint = self._finite_float(score_metrics.get('score'))
        if joint == float('-inf'):
            joint = self._finite_float(score_metrics.get('mean_mAP50_95'))
        det = self._finite_float(score_metrics.get(self.score_det_metric))
        pose = self._finite_float(score_metrics.get(self.score_pose_metric))

        if joint > self.best_score_metrics['joint'] + self.early_stop_min_delta:
            self.best_score_metrics['joint'] = joint
            self.save(self.save_dir / f"{save_prefix}_coco_best.pt", score_metrics)
            statuses.append('COCO_BEST')
        if det > self.best_score_metrics['det'] + self.early_stop_min_delta:
            self.best_score_metrics['det'] = det
            self.save(self.save_dir / f"{save_prefix}_det_best.pt", score_metrics)
            statuses.append('DET_BEST')
        if pose > self.best_score_metrics['pose'] + self.early_stop_min_delta:
            self.best_score_metrics['pose'] = pose
            self.save(self.save_dir / f"{save_prefix}_pose_best.pt", score_metrics)
            statuses.append('POSE_BEST')
        return statuses

    @staticmethod
    def _score_from_metrics(metrics, det_metric, pose_metric, det_baseline,
                            pose_baseline, task='both'):
        out = {}
        has_det = det_metric in metrics
        has_pose = pose_metric in metrics
        if has_det:
            det_value = float(metrics.get(det_metric, 0.0) or 0.0)
            out['score_det_value'] = det_value
        if has_pose:
            pose_value = float(metrics.get(pose_metric, 0.0) or 0.0)
            out['score_pose_value'] = pose_value
        if str(task).lower() == 'both' and has_det and has_pose:
            det_ratio = out['score_det_value'] / det_baseline
            pose_ratio = out['score_pose_value'] / pose_baseline
            out.update({
                'score': min(det_ratio, pose_ratio),
                'joint_mAP50_95': min(out['score_det_value'], out['score_pose_value']),
                'mean_mAP50_95': 0.5 * (out['score_det_value'] + out['score_pose_value']),
                'det_ratio': det_ratio,
                'pose_ratio': pose_ratio,
            })
        return out

    @torch.no_grad()
    def validate_score(self, loader, score_thresh=0.01, iou_thresh=0.6,
                       max_det=300, num_classes=80, keep_classes=None,
                       data_root=None, task='both', instances_json=None,
                       keypoints_json=None, coco_max_det=100):
        """Run official COCOeval and return metrics plus a joint target score."""
        from test_model.train.cocoeval import evaluate_model

        self._swap_ema(to_ema=True)
        try:
            metrics = evaluate_model(
                self.model, loader, self.device,
                task=task,
                data_root=data_root,
                instances_json=instances_json,
                keypoints_json=keypoints_json,
                num_classes=num_classes,
                keep_classes=keep_classes,
                score_thresh=score_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
                coco_max_det=coco_max_det,
            )
        finally:
            self._swap_ema(to_ema=True)

        metrics['task'] = task
        metrics.update(self._score_from_metrics(
            metrics, self.score_det_metric, self.score_pose_metric,
            self.score_det_baseline, self.score_pose_baseline, task=task))
        return metrics

    @staticmethod
    def _as_loss_weight(cfg, key, default=0.0):
        try:
            return float((cfg or {}).get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _non_person_teacher_cls(teacher_cls, student_cls):
        if teacher_cls.shape[1] == student_cls.shape[1] + 1:
            return teacher_cls[:, 1:, :, :]
        return teacher_cls

    def _distill_detect_outputs(self, student_out, teacher_out, cfg):
        cls_w = self._as_loss_weight(cfg, 'cls_weight', 0.0)
        reg_w = self._as_loss_weight(cfg, 'dfl_weight', cfg.get('reg_weight', 0.0))
        if cls_w <= 0.0 and reg_w <= 0.0:
            device = student_out['cls'][0].device
            zero = torch.zeros((), device=device)
            return zero, zero, zero

        cls_loss = None
        reg_loss = None
        for s_cls, t_cls in zip(student_out['cls'], teacher_out['cls']):
            t_cls = self._non_person_teacher_cls(t_cls.detach(), s_cls)
            if t_cls.shape == s_cls.shape and cls_w > 0.0:
                loss = F.mse_loss(s_cls.float(), t_cls.float())
                cls_loss = loss if cls_loss is None else cls_loss + loss
        for s_reg, t_reg in zip(student_out['reg'], teacher_out['reg']):
            if t_reg.shape == s_reg.shape and reg_w > 0.0:
                loss = F.mse_loss(s_reg.float(), t_reg.detach().float())
                reg_loss = loss if reg_loss is None else reg_loss + loss

        device = student_out['cls'][0].device
        zero = torch.zeros((), device=device)
        cls_loss = zero if cls_loss is None else cls_loss / max(len(student_out['cls']), 1)
        reg_loss = zero if reg_loss is None else reg_loss / max(len(student_out['reg']), 1)
        total = cls_w * cls_loss + reg_w * reg_loss
        return total, cls_loss.detach(), reg_loss.detach()

    def _distill_features(self, student_feats, teacher_feats, cfg):
        feat_w = self._as_loss_weight(cfg, 'feature_weight', 0.0)
        if feat_w <= 0.0:
            return torch.zeros((), device=student_feats[0].device)
        losses = []
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if s_feat.shape == t_feat.shape:
                losses.append(F.mse_loss(s_feat, t_feat.detach().to(dtype=s_feat.dtype)))
        if not losses:
            return torch.zeros((), device=student_feats[0].device)
        return feat_w * sum(losses) / len(losses)

    def _apply_distillation(self, images, losses):
        cfg = self.distill_cfg or {}
        if not cfg.get('enabled', False) or self.distill_teacher is None:
            return losses
        if not hasattr(self.model, 'forward_det_outputs'):
            return losses

        det_cfg = cfg.get('det', {}) or {}
        if not det_cfg.get('enabled', True):
            return losses

        teacher = self.distill_teacher
        teacher.eval()
        with torch.no_grad():
            teacher_data = teacher.forward_det_outputs(images, return_features=True)
        student_data = {
            'out': losses.get('_distill_det_out'),
            'det_feats': losses.get('_distill_det_feats', []),
        }
        if student_data['out'] is None:
            student_data = self.model.forward_det_outputs(images, return_features=True)
        out_total, cls_loss, reg_loss = self._distill_detect_outputs(
            student_data['out'], teacher_data['out'], det_cfg)
        feat_loss = self._distill_features(
            student_data.get('det_feats', []),
            teacher_data.get('det_feats', []),
            det_cfg,
        )
        total = (out_total + feat_loss) * max(int(images.shape[0]), 1)
        if total.detach().abs().item() == 0.0:
            return losses

        losses['total'] = losses['total'] + total
        if isinstance(losses.get('_gp_det_loss'), torch.Tensor):
            losses['_gp_det_loss'] = losses['_gp_det_loss'] + total
        losses['distill_total'] = total.detach()
        losses['distill_cls'] = cls_loss.detach()
        losses['distill_reg'] = reg_loss.detach()
        losses['distill_feat'] = feat_loss.detach()
        return losses

    def train_epoch(self, loader, max_epochs, epoch, close_mosaic=None):
        """Train one epoch.

        Args:
            loader: DataLoader
            max_epochs: Total epochs (for LR schedule)
            epoch: Current epoch (0-indexed)
            close_mosaic: If True, disable mosaic for this epoch
        """
        self.model.train()
        self._apply_frozen_module_eval()

        # Set mosaic mode
        if close_mosaic is not None:
            if hasattr(loader.dataset, 'set_close_mosaic'):
                loader.dataset.set_close_mosaic(close_mosaic)
            elif hasattr(loader.dataset, 'use_mosaic'):
                loader.dataset.use_mosaic = not close_mosaic

        metrics = {}
        running = {}
        n_batches = len(loader)
        warmup_iters = (
            max(round(self.warmup_epochs * n_batches), 100)
            if self.yolo_warmup and self.warmup_epochs > 0 else -1
        )
        step = 0
        seen_step = 0
        skipped = 0
        last_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.zero_grad(set_to_none=True)

        for batch in loader:
            seen_step += 1
            ni = epoch * n_batches + (seen_step - 1)
            if self.yolo_warmup and ni <= warmup_iters:
                self._set_yolo_warmup(ni, warmup_iters, epoch, max_epochs)
            else:
                lr = self._get_lr(epoch, max_epochs)
                self._set_lr(lr)
            last_lr = self.optimizer.param_groups[0]['lr']

            images = batch['image'].to(self.device, non_blocking=True)

            # Build gt_dict_list from batch
            gt_list = []
            for i in range(len(images)):
                gt_list.append({
                    'boxes': batch['boxes'][i],
                    'classes': batch['classes'][i],
                    'kpts': batch['kpts'][i],
                    'attrs': batch.get('attrs', [None] * len(images))[i],
                    'attr_mask': batch.get('attr_mask', [None] * len(images))[i],
                    'domain_valid_mask': batch.get('domain_valid_mask', [None] * len(images))[i],
                    'task': batch.get('task', [None] * len(images))[i],
                })

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    losses = self.model.compute_loss(images, gt_list)
                    losses = self._apply_distillation(images, losses)
            else:
                losses = self.model.compute_loss(images, gt_list)
                losses = self._apply_distillation(images, losses)

            total_loss = losses['total']

            # Check for NaN
            if self.check_finite_loss and not torch.isfinite(total_loss.detach()).item():
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                print(f"  WARNING: NaN/Inf loss at batch {seen_step}/{n_batches}, skipping batch")
                continue

            optimizer_stepped = False
            if self.gradient_projection_enabled:
                self._apply_gradient_projection(losses, total_loss)
                if seen_step % self.accumulate == 0 or seen_step == n_batches:
                    if self.scaler:
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_stepped = True
            elif self.scaler:
                self.scaler.scale(total_loss).backward()
                if seen_step % self.accumulate == 0 or seen_step == n_batches:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_stepped = True
            else:
                total_loss.backward()
                if seen_step % self.accumulate == 0 or seen_step == n_batches:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_stepped = True

            if optimizer_stepped:
                self._clamp_dynamic_model_state()
                self.global_step += 1
                if self.ema_enabled:
                    self._update_ema()
            step += 1

            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and self._should_log_loss_key(k):
                    value = v.detach()
                    running[k] = running.get(k, torch.zeros_like(value)) + value
                    metrics[k] = metrics.get(k, torch.zeros_like(value)) + value
            if self.gradient_projection_enabled and self.gradient_projection_method == 'pomsi':
                value = self.pomsi_alpha.detach()
                running['pomsi_alpha'] = running.get('pomsi_alpha', torch.zeros_like(value)) + value
                metrics['pomsi_alpha'] = metrics.get('pomsi_alpha', torch.zeros_like(value)) + value

            if step % self.log_interval == 0:
                pct = step / n_batches * 100
                avg_running = {
                    k: (running[k] / self.log_interval).item()
                    for k in running
                }
                print(f"  [{step}/{n_batches} {pct:.0f}%]")
                for line in self.format_metric_lines(
                        avg_running,
                        label='train',
                        indent='    ',
                        extra={'lr': f"{last_lr:.2e}"}):
                    print(line)
                if self.writer:
                    for k in running:
                        self.writer.add_scalar(f'train/{k}', (running[k] / self.log_interval).item(), self.global_step)
                    self.writer.add_scalar('train/lr', last_lr, self.global_step)
                running.clear()

        if step == 0:
            return metrics
        out = {k: (v / step).item() for k, v in metrics.items()}
        if skipped:
            out['skipped'] = float(skipped)
        return out

    @torch.no_grad()
    def validate(self, loader):
        """Validation loop."""
        self.model.eval()
        self._swap_ema(to_ema=True)
        try:
            metrics = {}

            for batch in loader:
                images = batch['image'].to(self.device, non_blocking=True)
                gt_list = []
                for i in range(len(images)):
                    gt_list.append({
                        'boxes': batch['boxes'][i],
                        'classes': batch['classes'][i],
                        'kpts': batch['kpts'][i],
                        'attrs': batch.get('attrs', [None] * len(images))[i],
                        'attr_mask': batch.get('attr_mask', [None] * len(images))[i],
                        'domain_valid_mask': batch.get('domain_valid_mask', [None] * len(images))[i],
                        'task': batch.get('task', [None] * len(images))[i],
                    })

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        losses = self.model.compute_loss(images, gt_list)
                else:
                    losses = self.model.compute_loss(images, gt_list)

                for k, v in losses.items():
                    if isinstance(v, torch.Tensor) and self._should_log_loss_key(k):
                        key = 'val_' + k
                        value = v.detach()
                        metrics[key] = metrics.get(key, torch.zeros_like(value)) + value
        finally:
            self._swap_ema(to_ema=True)
        n = max(len(loader), 1)
        return {k: (v / n).item() for k, v in metrics.items()}

    def save(self, path, metrics=None):
        self._swap_ema(to_ema=True)
        try:
            state = {
                'epoch': self.current_epoch + 1,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_groups': [
                    {
                        'name': group.get('name', f'group{i}'),
                        'lr_mult': float(group.get('lr_mult', 1.0)),
                        'params': sum(p.numel() for p in group['params']),
                    }
                    for i, group in enumerate(self.optimizer.param_groups)
                ],
                'pomsi_alpha': float(self.pomsi_alpha.detach()),
                'metrics': metrics,
            }
            if self.scaler:
                state['scaler_state_dict'] = self.scaler.state_dict()
            if self.ema_enabled:
                state['ema_state'] = {name: t.clone() for name, t in self._ema_state.items()}
            torch.save(state, str(path))
        finally:
            self._swap_ema(to_ema=True)
        print(f"  Saved: {path}")

    def load(self, path, reset_ema_from_model=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self._apply_configured_lr_groups()
            except ValueError as exc:
                print(
                    "  WARNING: optimizer state was not loaded "
                    f"({exc}). Model weights were loaded; optimizer was rebuilt."
                )
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'pomsi_alpha' in ckpt:
            self.pomsi_alpha = torch.tensor(
                float(ckpt['pomsi_alpha']),
                device=self.device,
                requires_grad=True,
            )
        if self.ema_enabled and reset_ema_from_model:
            self.reset_ema_from_model()
            print("  EMA reset from loaded model weights")
        elif self.ema_enabled and 'ema_state' in ckpt:
            self._ema_state = {name: t.to(self.device)
                               for name, t in ckpt['ema_state'].items()}
        self._clamp_dynamic_model_state()
        self.current_epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        print(f"  Loaded: {path} (epoch {self.current_epoch})")

    def load_eval_weights(self, path):
        """Load checkpoint model weights for final evaluation.

        Saved checkpoints already store EMA model weights in model_state_dict.
        Reset EMA state to the same tensors so validate_score() does not swap
        raw training weights back in during evaluation.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state)
        self._clamp_dynamic_model_state()
        if self.ema_enabled:
            self._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()
                if torch.is_floating_point(tensor)
            }
        print(f"  Loaded eval weights: {path} (epoch {ckpt.get('epoch', 'unknown')})")

    def fit(self, epochs, train_loader, val_loader=None,
            save_prefix='model', close_mosaic_epochs=10,
            on_epoch_start=None, on_epoch_end=None,
            score_loader=None, score_eval_kwargs=None,
            final_score_eval=False):
        """Main training loop.

        Args:
            epochs: Total epochs
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            save_prefix: Prefix for checkpoint filenames
            close_mosaic_epochs: Disable mosaic for last N epochs
            on_epoch_start: Optional callback(epoch) called at start of each epoch
            on_epoch_end: Optional callback(epoch, train_metrics, val_metrics) called at end of each epoch
        """
        if self.save_best_by != 'loss':
            raise ValueError(
                "Training checkpoint selection is loss-only. "
                "Set training.save_best_by: loss and run COCOeval after training.")

        print(f"\n{'='*60}")
        print(f"Training: {save_prefix} | Epochs: {epochs} | "
              f"Device: {self.device} | Base LR: {self.base_lr}")
        print(f"Optimizer: {self.optimizer_name} | AMP: {self.use_amp} | EMA: {self.ema_enabled} "
              f"(decay={self.ema_decay})")
        print(
            f"Batch: {self.batch_size} | nbs={self.nbs} | accumulate={self.accumulate} | "
            f"param_groups={self.param_group_mode} | weight_decay={self.weight_decay:.6g} | "
            f"yolo_warmup={self.yolo_warmup}")
        if self.gradient_projection_enabled:
            print(
                "Gradient strategy: "
                f"method={self.gradient_projection_method} "
                f"scope={self.gradient_projection_scope}")
            if self.gradient_projection_method == 'pomsi':
                print(
                    "POMSI: "
                    f"alpha={float(self.pomsi_alpha.detach()):.4g} "
                    f"alpha_lr={self.pomsi_alpha_lr:.4g} "
                    f"static={self.pomsi_static}")
        if self.distill_cfg.get('enabled', False):
            print("Distillation: enabled")
        group_parts = []
        for group in self.optimizer.param_groups:
            n_params = sum(p.numel() for p in group['params'])
            group_parts.append(
                f"{group.get('name', 'group')}: lr_mult={float(group.get('lr_mult', 1.0)):.4g} "
                f"params={n_params / 1e6:.2f}M")
        print("LR groups: " + " | ".join(group_parts))
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            if on_epoch_start:
                on_epoch_start(epoch)
            t0 = time.time()

            close_mosaic = (close_mosaic_epochs > 0 and
                            epoch >= epochs - close_mosaic_epochs)

            train_m = self.train_epoch(train_loader, epochs, epoch,
                                       close_mosaic=close_mosaic)
            train_elapsed = time.time() - t0

            do_val = val_loader and (epoch + 1) % self.val_interval == 0
            val_m = None
            val_elapsed = 0.0
            status = ''
            if do_val:
                val_t0 = time.time()
                val_m = self.validate(val_loader)
                val_elapsed = time.time() - val_t0

                metric_payload = dict(val_m)
                current = metric_payload.get(
                    'val_loss_total',
                    metric_payload.get('val_total', float('inf')),
                )
                improved = self._is_improved(current)
                if improved:
                    self.best_metric = current
                    self.early_stop_bad_epochs = 0
                    self.save(self.save_dir / f"{save_prefix}_best.pt", metric_payload)
                    self.save(self.save_dir / f"{save_prefix}_loss_best.pt", metric_payload)
                    status = ' [BEST]'
                elif (self.early_stop_enabled and
                      epoch + 1 >= self.early_stop_start_epoch):
                    self.early_stop_bad_epochs += 1
                    status = f" [NO_IMPROVE {self.early_stop_bad_epochs}/{self.early_stop_patience}]"

            score_m = None
            score_statuses = []
            do_score = (
                do_val and self.score_eval_enabled and score_loader is not None and
                (epoch + 1) % self.score_eval_interval == 0
            )
            if do_score:
                print(f"Running COCOeval at epoch {epoch + 1}...")
                score_m = self.validate_score(score_loader, **(score_eval_kwargs or {}))
                if val_m is not None:
                    score_m.update({k: v for k, v in val_m.items() if k not in score_m})
                score_statuses = self._save_score_bests(score_m, save_prefix)

            total_elapsed = train_elapsed + val_elapsed
            if do_val:
                time_msg = (
                    f"train={train_elapsed:.0f}s val={val_elapsed:.0f}s "
                    f"total={total_elapsed:.0f}s"
                )
            else:
                time_msg = f"train={train_elapsed:.0f}s total={total_elapsed:.0f}s"
            print(f"Epoch {epoch + 1:3d}/{epochs} | {time_msg}{status}")
            for line in self.format_metric_lines(train_m, label='train', indent='  '):
                print(line)
            if do_val:
                for line in self.format_metric_lines(
                        val_m,
                        label='val',
                        indent='  ',
                        prefix='val_'):
                    print(line)
            if score_m is not None:
                score_line = self._format_score_line(score_m, label='val/coco', indent='  ')
                if score_line:
                    suffix = f" [{' '.join(score_statuses)}]" if score_statuses else ""
                    print(score_line + suffix)

            if self.writer:
                for k, v in train_m.items():
                    self.writer.add_scalar(f'epoch/train_{k}', v, epoch)
                if do_val:
                    for k, v in val_m.items():
                        self.writer.add_scalar(f'epoch/{k}', v, epoch)
                if score_m is not None:
                    for k, v in score_m.items():
                        if isinstance(v, (int, float, np.floating)):
                            self.writer.add_scalar(f'coco/{k}', float(v), epoch)

            if (epoch + 1) % self.save_interval == 0:
                self.save(self.save_dir / f"{save_prefix}_epoch{epoch + 1}.pt")

            if on_epoch_end:
                on_epoch_end(epoch, train_m, val_m if do_val else None)

            if (do_val and self.early_stop_enabled and
                    self.early_stop_bad_epochs >= self.early_stop_patience):
                print(f"Early stopping at epoch {epoch + 1}: "
                      f"best {self.save_best_by}={self.best_metric:.4f}")
                break

        # Save last checkpoint
        last_path = self.save_dir / f"{save_prefix}_last.pt"
        self.save(last_path)
        print(f"\nBest {self.save_best_by}: {self.best_metric:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")

        if final_score_eval:
            if score_loader is None:
                raise RuntimeError("Final COCOeval requires a score_loader")
            best_path = self.save_dir / f"{save_prefix}_best.pt"
            eval_path = best_path if best_path.exists() else last_path
            if eval_path != best_path:
                print(f"\nBest checkpoint not found; final COCOeval will use: {eval_path}")
            else:
                print(f"\nRunning final COCOeval on loss-best checkpoint: {eval_path}")
            self.load_eval_weights(eval_path)
            metrics = self.validate_score(score_loader, **(score_eval_kwargs or {}))
            metrics['checkpoint'] = str(eval_path)
            metrics['checkpoint_selection'] = 'loss_best' if eval_path == best_path else 'last_fallback'
            metrics_path = self.save_dir / f"{save_prefix}_final_cocoeval_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Final COCOeval metrics saved to: {metrics_path}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
        if self.writer:
            self.writer.close()

