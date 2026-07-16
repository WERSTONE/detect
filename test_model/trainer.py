"""Training loop with SGD/AdamW + cosine LR + EMA + AMP.

Key features:
- SGD optimizer with momentum or AdamW
- Cosine LR schedule with linear warmup
- EMA (exponential moving average)
- AMP mixed precision
- Mosaic scheduling (disable in last N epochs)
- Automatic evaluation on best + last checkpoint after training
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
        save_best_by: 'loss' or custom metric
    """

    def __init__(self, model, device='cuda',
                 lr=0.01, momentum=0.937, weight_decay=5e-4,
                 optimizer='sgd', nesterov=True, final_lr_ratio=0.01,
                 backbone_lr=None, backbone_lr_mult=1.0,
                 cos_lr=True,
                 warmup_epochs=3, grad_clip=10.0,
                 log_interval=20, save_interval=20, val_interval=5,
                 save_dir='checkpoints', use_amp=True,
                 ema_decay=0.9999, save_best_by='loss',
                 use_tensorboard=False, check_finite_loss=False,
                 early_stop_enabled=False, early_stop_patience=0,
                 early_stop_min_delta=0.0, early_stop_start_epoch=0,
                 score_interval=1, score_det_baseline=1.0,
                 score_pose_baseline=1.0, score_det_metric='mAP@0.5:0.95',
                 score_pose_metric='AP_pose@0.5:0.95',
                 gradient_projection_enabled=False,
                 gradient_projection_eps=1.0e-12):
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
        self.backbone_lr = None if backbone_lr is None else float(backbone_lr)
        self.backbone_lr_mult = float(backbone_lr_mult)
        param_groups = self._build_param_groups()
        if self.optimizer_name in ('adamw', 'adam'):
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        elif self.optimizer_name == 'sgd':
            # Use regular SGD. Fused SGD can fail when multi-head batches leave
            # some head parameters without gradients.
            optim_kw = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
            self.optimizer = torch.optim.SGD(param_groups, **optim_kw)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        self.warmup_epochs = warmup_epochs
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
        self.gradient_projection_enabled = bool(gradient_projection_enabled)
        self.gradient_projection_eps = float(gradient_projection_eps)

        # EMA
        self.ema_decay = ema_decay
        self.ema_enabled = ema_decay > 0
        self._ema_state = {}
        if self.ema_enabled:
            self._build_ema()

        # AMP
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def _build_param_groups(self):
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
            if name == 'backbone':
                group['lr_mult'] = float(self._configured_backbone_lr_mult())
            elif name == 'other':
                group['lr_mult'] = 1.0

    @staticmethod
    def _should_log_loss_key(key):
        return not str(key).startswith('_')

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

    def _apply_gradient_projection(self, losses, total_loss):
        det_loss = losses.get('_gp_det_loss')
        pose_loss = losses.get('_gp_pose_loss')
        task_losses = [
            loss for loss in (det_loss, pose_loss)
            if isinstance(loss, torch.Tensor) and loss.requires_grad
        ]
        if len(task_losses) < 2:
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            return False

        params = [p for p in self.model.parameters() if p.requires_grad]
        task_grads = [
            self._collect_task_grads(loss, params, retain_graph=(idx < len(task_losses) - 1))
            for idx, loss in enumerate(task_losses)
        ]

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

        self.optimizer.zero_grad(set_to_none=True)
        for param_index, p in enumerate(params):
            grad_sum = None
            for grads in projected:
                grad = grads[param_index]
                if grad is None:
                    continue
                grad_sum = grad if grad_sum is None else grad_sum + grad
            if grad_sum is not None:
                p.grad = grad_sum
        return True

    def _build_ema(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self._ema_state[name] = p.data.clone().detach()

    def _update_ema(self):
        d = self.ema_decay
        for name, p in self.model.named_parameters():
            if name in self._ema_state:
                self._ema_state[name].mul_(d).add_(p.data, alpha=1 - d)

    def _swap_ema(self, to_ema=True):
        """Swap model weights with EMA shadow."""
        if not self.ema_enabled:
            return
        for name, p in self.model.named_parameters():
            if name in self._ema_state:
                if to_ema:
                    tmp = p.data.clone()
                    p.data.copy_(self._ema_state[name])
                    self._ema_state[name] = tmp
                else:
                    tmp = p.data.clone()
                    p.data.copy_(self._ema_state[name])
                    self._ema_state[name] = tmp

    def _get_lr(self, epoch, max_epochs):
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs)
            return self.base_lr * (0.1 + 0.9 * progress)  # start at 0.1*lr
        progress = (epoch - self.warmup_epochs) / max(1, max_epochs - self.warmup_epochs)
        final_lr = self.base_lr * self.final_lr_ratio
        if self.cos_lr:
            return final_lr + (self.base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return self.base_lr + (final_lr - self.base_lr) * progress

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * float(pg.get('lr_mult', 1.0))

    def _is_improved(self, current):
        if self.save_best_by == 'loss':
            return current < self.best_metric - self.early_stop_min_delta
        return current > self.best_metric + self.early_stop_min_delta

    @staticmethod
    def _score_from_metrics(metrics, det_metric, pose_metric, det_baseline, pose_baseline):
        det_value = float(metrics.get(det_metric, 0.0) or 0.0)
        pose_value = float(metrics.get(pose_metric, 0.0) or 0.0)
        det_ratio = det_value / det_baseline
        pose_ratio = pose_value / pose_baseline
        return {
            'score': min(det_ratio, pose_ratio),
            'joint_mAP50_95': min(det_value, pose_value),
            'mean_mAP50_95': 0.5 * (det_value + pose_value),
            'det_ratio': det_ratio,
            'pose_ratio': pose_ratio,
            'score_det_value': det_value,
            'score_pose_value': pose_value,
        }

    @torch.no_grad()
    def validate_score(self, loader, score_thresh=0.01, iou_thresh=0.6,
                       max_det=300, num_classes=80):
        """Run AP validation and return metrics plus a joint target score."""
        from test_model.eval import compute_all_metrics, evaluate

        self._swap_ema(to_ema=True)
        all_preds, all_gts = evaluate(
            self.model, loader, self.device, score_thresh=score_thresh,
            iou_thresh=iou_thresh, max_det=max_det)
        self._swap_ema(to_ema=True)

        metrics = compute_all_metrics(all_preds, all_gts, num_classes=num_classes)
        metrics.update(self._score_from_metrics(
            metrics, self.score_det_metric, self.score_pose_metric,
            self.score_det_baseline, self.score_pose_baseline))
        return metrics

    def train_epoch(self, loader, max_epochs, epoch, close_mosaic=None):
        """Train one epoch.

        Args:
            loader: DataLoader
            max_epochs: Total epochs (for LR schedule)
            epoch: Current epoch (0-indexed)
            close_mosaic: If True, disable mosaic for this epoch
        """
        self.model.train()

        # Set mosaic mode
        if close_mosaic is not None:
            if hasattr(loader.dataset, 'set_close_mosaic'):
                loader.dataset.set_close_mosaic(close_mosaic)
            elif hasattr(loader.dataset, 'use_mosaic'):
                loader.dataset.use_mosaic = not close_mosaic

        metrics = {}
        running = {}
        n_batches = len(loader)
        step = 0
        seen_step = 0
        skipped = 0

        for batch in loader:
            seen_step += 1
            lr = self._get_lr(epoch, max_epochs)
            self._set_lr(lr)

            images = batch['image'].to(self.device, non_blocking=True)

            # Build gt_dict_list from batch
            gt_list = []
            for i in range(len(images)):
                gt_list.append({
                    'boxes': batch['boxes'][i],
                    'classes': batch['classes'][i],
                    'kpts': batch['kpts'][i],
                })

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    losses = self.model.compute_loss(images, gt_list)
            else:
                losses = self.model.compute_loss(images, gt_list)

            total_loss = losses['total']

            # Check for NaN
            if self.check_finite_loss and not torch.isfinite(total_loss.detach()).item():
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                print(f"  WARNING: NaN/Inf loss at batch {seen_step}/{n_batches}, skipping batch")
                continue

            self.optimizer.zero_grad(set_to_none=True)
            if self.gradient_projection_enabled:
                self._apply_gradient_projection(losses, total_loss)
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
            elif self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.global_step += 1
            step += 1

            if self.ema_enabled:
                self._update_ema()

            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and self._should_log_loss_key(k):
                    value = v.detach()
                    running[k] = running.get(k, torch.zeros_like(value)) + value
                    metrics[k] = metrics.get(k, torch.zeros_like(value)) + value

            if step % self.log_interval == 0:
                pct = step / n_batches * 100
                parts = [f"{k}={(running[k] / self.log_interval).item():.4f}" for k in sorted(running)]
                parts.append(f"lr={lr:.2e}")
                print(f"  [{step}/{n_batches} {pct:.0f}%] " + " ".join(parts))
                if self.writer:
                    for k in running:
                        self.writer.add_scalar(f'train/{k}', (running[k] / self.log_interval).item(), self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
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
        metrics = {}

        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            gt_list = []
            for i in range(len(images)):
                gt_list.append({
                    'boxes': batch['boxes'][i],
                    'classes': batch['classes'][i],
                    'kpts': batch['kpts'][i],
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

        self._swap_ema(to_ema=True)
        n = max(len(loader), 1)
        return {k: (v / n).item() for k, v in metrics.items()}

    def save(self, path, metrics=None):
        self._swap_ema(to_ema=True)
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
            'metrics': metrics,
        }
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
        if self.ema_enabled:
            state['ema_state'] = {name: t.clone() for name, t in self._ema_state.items()}
        torch.save(state, str(path))
        self._swap_ema(to_ema=True)
        print(f"  Saved: {path}")

    def load(self, path):
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
        if self.ema_enabled and 'ema_state' in ckpt:
            self._ema_state = {name: t.to(self.device)
                               for name, t in ckpt['ema_state'].items()}
        self.current_epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        print(f"  Loaded: {path} (epoch {self.current_epoch})")

    def fit(self, epochs, train_loader, val_loader=None,
            save_prefix='model', close_mosaic_epochs=10,
            on_epoch_start=None, on_epoch_end=None,
            score_loader=None, score_eval_kwargs=None):
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
        print(f"\n{'='*60}")
        print(f"Training: {save_prefix} | Epochs: {epochs} | "
              f"Device: {self.device} | Base LR: {self.base_lr}")
        print(f"Optimizer: {self.optimizer_name} | AMP: {self.use_amp} | EMA: {self.ema_enabled} "
              f"(decay={self.ema_decay})")
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
            elapsed = time.time() - t0

            log = f"Epoch {epoch + 1:3d}/{epochs} | {elapsed:.0f}s | "
            log += " ".join(f"{k}={v:.4f}" for k, v in sorted(train_m.items()))

            do_val = val_loader and (epoch + 1) % self.val_interval == 0
            if do_val:
                val_m = self.validate(val_loader)
                log += " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(val_m.items()))

                metric_payload = dict(val_m)
                needs_ap_metrics = self.save_best_by != 'loss'
                if needs_ap_metrics:
                    if score_loader is None:
                        raise RuntimeError(f"save_best_by='{self.save_best_by}' requires a score_loader")
                    if (epoch + 1) % self.score_interval == 0:
                        score_kwargs = score_eval_kwargs or {}
                        score_m = self.validate_score(score_loader, **score_kwargs)
                        metric_payload.update(score_m)
                        metric_name = self.save_best_by
                        metric_value = metric_payload.get(metric_name)
                        if metric_value is None:
                            raise RuntimeError(
                                f"save_best_by='{metric_name}' was not produced by AP validation. "
                                f"Available keys include: {sorted(metric_payload)[:20]}"
                            )
                        log += f" | {metric_name}={float(metric_value):.4f}"
                    else:
                        score_m = None
                else:
                    score_m = None

                current = (metric_payload.get(self.save_best_by, -float('inf'))
                           if self.save_best_by != 'loss'
                           else metric_payload.get('val_total', float('inf')))
                improved = self._is_improved(current)
                if improved:
                    self.best_metric = current
                    self.early_stop_bad_epochs = 0
                    self.save(self.save_dir / f"{save_prefix}_best.pt", metric_payload)
                    log += " [BEST]"
                elif (self.early_stop_enabled and
                      epoch + 1 >= self.early_stop_start_epoch):
                    self.early_stop_bad_epochs += 1
                    log += f" [NO_IMPROVE {self.early_stop_bad_epochs}/{self.early_stop_patience}]"

            print(log)

            if self.writer:
                for k, v in train_m.items():
                    self.writer.add_scalar(f'epoch/train_{k}', v, epoch)
                if do_val:
                    for k, v in val_m.items():
                        self.writer.add_scalar(f'epoch/{k}', v, epoch)

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
        self.save(self.save_dir / f"{save_prefix}_last.pt")
        print(f"\nBest {self.save_best_by}: {self.best_metric:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        if self.writer:
            self.writer.close()
