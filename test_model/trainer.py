"""Training loop with SGD + cosine LR + EMA + AMP.

Key features:
- SGD optimizer with momentum (matching YOLOv8 official recipe)
- Cosine LR schedule with linear warmup
- EMA (exponential moving average)
- AMP mixed precision
- Mosaic scheduling (disable in last N epochs)
- Automatic evaluation on best + last checkpoint after training
"""

import math
import time
from collections import defaultdict
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
                 warmup_epochs=3, grad_clip=10.0,
                 log_interval=20, save_interval=20, val_interval=5,
                 save_dir='checkpoints', use_amp=True,
                 ema_decay=0.9999, save_best_by='loss'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Fused SGD for ~20% faster optimizer step (CUDA only)
        optim_kw = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        if device == 'cuda':
            try:
                self.optimizer = torch.optim.SGD(model.parameters(), fused=True, **optim_kw)
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.SGD(model.parameters(), **optim_kw)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), **optim_kw)
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr

        self.current_epoch = 0
        self.global_step = 0
        self.save_best_by = save_best_by
        self.best_metric = float('inf') if save_best_by == 'loss' else -float('inf')

        # EMA
        self.ema_decay = ema_decay
        self.ema_enabled = ema_decay > 0
        self._ema_state = {}
        if self.ema_enabled:
            self._build_ema()

        # AMP
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

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
            return self.base_lr * max(progress, 0.01)
        progress = (epoch - self.warmup_epochs) / max(1, max_epochs - self.warmup_epochs)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

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
        if hasattr(loader.dataset, 'use_mosaic'):
            if close_mosaic is not None:
                loader.dataset.use_mosaic = not close_mosaic

        metrics = defaultdict(float)
        running = defaultdict(float)
        n_batches = len(loader)
        step = 0

        for batch in loader:
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
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  WARNING: NaN/Inf loss at step {step}, skipping batch")
                continue

            self.optimizer.zero_grad()
            if self.scaler:
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
                if isinstance(v, torch.Tensor):
                    running[k] += v.item()
                    metrics[k] += v.item()

            if step % self.log_interval == 0:
                pct = step / n_batches * 100
                parts = [f"{k}={running[k] / self.log_interval:.4f}" for k in sorted(running)]
                parts.append(f"lr={lr:.2e}")
                print(f"  [{step}/{n_batches} {pct:.0f}%] " + " ".join(parts))
                running.clear()

        for k in metrics:
            metrics[k] /= step
        return metrics

    @torch.no_grad()
    def validate(self, loader):
        """Validation loop."""
        self.model.eval()
        self._swap_ema(to_ema=True)
        metrics = defaultdict(float)

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
                if isinstance(v, torch.Tensor):
                    metrics['val_' + k] += v.item()

        self._swap_ema(to_ema=True)
        n = max(len(loader), 1)
        for k in metrics:
            metrics[k] /= n
        return metrics

    def save(self, path, metrics=None):
        self._swap_ema(to_ema=True)
        state = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        if self.ema_enabled and 'ema_state' in ckpt:
            self._ema_state = {name: t.to(self.device)
                               for name, t in ckpt['ema_state'].items()}
        self.current_epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        print(f"  Loaded: {path} (epoch {self.current_epoch})")

    def fit(self, epochs, train_loader, val_loader=None,
            save_prefix='model', close_mosaic_epochs=10):
        """Main training loop.

        Args:
            epochs: Total epochs
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            save_prefix: Prefix for checkpoint filenames
            close_mosaic_epochs: Disable mosaic for last N epochs
        """
        print(f"\n{'='*60}")
        print(f"Training: {save_prefix} | Epochs: {epochs} | "
              f"Device: {self.device} | Base LR: {self.base_lr}")
        print(f"AMP: {self.use_amp} | EMA: {self.ema_enabled} "
              f"(decay={self.ema_decay})")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
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

                current = val_m.get('val_total', float('inf'))
                if self.save_best_by == 'loss' and current < self.best_metric:
                    self.best_metric = current
                    self.save(self.save_dir / f"{save_prefix}_best.pt", val_m)
                    log += " [BEST]"

            print(log)

            if (epoch + 1) % self.save_interval == 0:
                self.save(self.save_dir / f"{save_prefix}_epoch{epoch + 1}.pt")

        # Save last checkpoint
        self.save(self.save_dir / f"{save_prefix}_last.pt")
        print(f"\nBest val_loss: {self.best_metric:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
