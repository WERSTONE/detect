"""通用训练器 — 模型无关，只依赖模型的 compute_loss 接口."""

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class Trainer:
    """通用训练器.

    模型契约:
        model.compute_loss(sample) → {"total": Tensor, ...}
        训练器只调用这个方法和标准的 PyTorch 接口 (parameters, train, eval,
        state_dict, load_state_dict).

    AMP:
        use_amp=True 时启用自动混合精度. amp_dtype="float16" 使用 GradScaler,
        "bfloat16" 不需要 scaler (仅 Ampere+ GPU).
    """

    def __init__(self, model, device="cpu",
                 lr=1e-3, weight_decay=1e-4, warmup_epochs=3,
                 grad_clip=20.0, log_interval=10,
                 save_interval=10, val_interval=1,
                 save_dir="checkpoints",
                 use_tensorboard=False, tb_log_dir="logs/train_logs",
                 use_amp=False, amp_dtype="float16",
                 map_enabled=True, map_samples=500,
                 save_best_by="loss",
                 ema_decay=0.9999):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr

        self.current_epoch = 0
        self.global_step = 0
        if save_best_by not in ("loss", "score"):
            raise ValueError("save_best_by must be 'loss' or 'score'")
        self.save_best_by = save_best_by
        self.best_score = -float("inf") if save_best_by == "score" else float("inf")

        self.map_enabled = map_enabled
        self.map_samples = map_samples

        # ── EMA ──
        self.ema_decay = ema_decay
        self.ema_enabled = ema_decay > 0
        self._ema_state = {}  # {name: shadow_tensor}
        if self.ema_enabled:
            self._build_ema()

        # ── AMP ──
        self.use_amp = use_amp and device.startswith("cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and amp_dtype != "bfloat16" else None
        if self.use_amp:
            print(f"  AMP: {amp_dtype}" + (" (GradScaler)" if self.scaler else ""))

        self.writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"  TensorBoard: {tb_log_dir}")

    def _build_ema(self):
        """创建 EMA 影子参数 (仅 trainable params)."""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self._ema_state[name] = p.data.clone().detach()

    def _update_ema(self):
        """EMA 更新: shadow = decay * shadow + (1-decay) * current."""
        d = self.ema_decay
        for name, p in self.model.named_parameters():
            if name in self._ema_state:
                self._ema_state[name].mul_(d).add_(p.data, alpha=1 - d)

    def _swap_ema(self, to_ema=True):
        """交换模型权重与 EMA 影子 (to_ema=True → 用 EMA 替换当前权重)."""
        if not self.ema_enabled:
            return
        for name, p in self.model.named_parameters():
            if name in self._ema_state:
                if to_ema:
                    # 保存当前到 _ema_state 的临时备份 → 这个不行，会破坏 EMA
                    # 正确的做法：swap
                    tmp = p.data.clone()
                    p.data.copy_(self._ema_state[name])
                    self._ema_state[name] = tmp
                else:
                    # swap back
                    tmp = p.data.clone()
                    p.data.copy_(self._ema_state[name])
                    self._ema_state[name] = tmp

    def _get_lr(self, epoch, max_epochs):
        if epoch < self.warmup_epochs:
            # 线性 warmup: epoch 0 从 30% base_lr 起步, 4 epoch 内爬升到完整 lr
            warmup_bias = 0.3
            progress = epoch / max(1, self.warmup_epochs)
            return self.base_lr * (warmup_bias + (1 - warmup_bias) * progress)
        progress = (epoch - self.warmup_epochs) / max(1, max_epochs - self.warmup_epochs)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train_epoch(self, loaders, max_epochs, dataset_weights=None):
        """训练一个 epoch.

        dataset_weights 不为 None 时, 指定各数据集的重复倍数.
        例如 {'smoking': 3.0} 表示 smoking 的 dataloader 在 epoch 内循环 3 遍.
        """
        self.model.train()
        metrics = defaultdict(float)

        # epoch 总 batch 数 = 每个数据集的样本数 × 权重
        n_batches = 0
        for name, dl in loaders.items():
            w = dataset_weights.get(name, 1.0) if dataset_weights else 1.0
            n_batches += int(len(dl) * w)

        iters = {name: iter(dl) for name, dl in loaders.items()}
        restart_count = {name: 0 for name in loaders}
        done = set()

        # 构建带权重的轮询名册: weight=3 → 名册中出现 3 次同名条目
        if dataset_weights is not None:
            weighted_names = []
            for name in loaders:
                n_slots = max(1, int(round(dataset_weights.get(name, 1.0))))
                weighted_names.extend([name] * n_slots)
        else:
            weighted_names = list(loaders.keys())

        step = 0
        idx = 0
        running = defaultdict(float)

        while len(done) < len(loaders):
            name = weighted_names[idx % len(weighted_names)]
            idx += 1
            if name in done:
                continue

            dl_iter = iters[name]
            try:
                batch = next(dl_iter)
            except StopIteration:
                # 加权数据集: 耗尽后重启, 直到达到权重指定的遍数
                max_repeats = int(dataset_weights.get(name, 1.0)) if dataset_weights else 1
                if restart_count[name] + 1 < max_repeats:
                    iters[name] = iter(loaders[name])
                    restart_count[name] += 1
                    batch = next(iters[name])
                else:
                    done.add(name)
                    continue

            lr = self._get_lr(self.current_epoch, max_epochs)
            self._set_lr(lr)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    losses = self.model.compute_loss(batch)
            else:
                losses = self.model.compute_loss(batch)
            total_loss = losses["total"]
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    running[k] += v.item()

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.global_step += 1
            metrics["loss"] += total_loss.item()
            step += 1

            if self.ema_enabled:
                self._update_ema()

            if step % self.log_interval == 0:
                pct = step / n_batches * 100
                parts = []
                for k in sorted(running):
                    avg = running[k] / self.log_interval
                    parts.append(f"{k}={avg:.4f}")
                    if self.writer:
                        self.writer.add_scalar(f"train/{k}", avg, self.global_step)
                if self.writer:
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                parts.append(f"lr={lr:.2e}")
                print(f"  [{step}/{n_batches} {pct:.0f}%] " + " ".join(parts))
                running.clear()

        for k in metrics:
            metrics[k] /= step
        return metrics

    @torch.no_grad()
    def validate(self, loaders):
        self.model.eval()
        self._swap_ema(to_ema=True)  # 用 EMA 权重做验证
        metrics = defaultdict(float)
        n = 0

        for dl in loaders.values():
            for batch in dl:
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        losses = self.model.compute_loss(batch)
                else:
                    losses = self.model.compute_loss(batch)
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        metrics["val_" + k] += v.item()
                n += 1

        self._swap_ema(to_ema=True)  # swap back (对称操作)
        for k in metrics:
            metrics[k] /= max(n, 1)
        return metrics

    # ── mAP 计算 ──

    @torch.no_grad()
    def _compute_map(self, val_loaders, max_samples=500):
        """Compute mAP@0.5 across validation datasets (subset for speed).

        Requires model.predict_val(sample) → (boxes[K,4], scores[K], classes[K]).
        Models without predict_val are skipped by the caller.
        max_samples limits total samples across all loaders to keep val fast.
        """
        import numpy as np
        self.model.eval()
        self._swap_ema(to_ema=True)

        all_preds = []  # (boxes, scores, classes) per image
        all_gts = []    # (boxes, classes) per image
        collected = 0

        for dl in val_loaders.values():
            for batch in dl:
                for sample in batch:
                    if collected >= max_samples:
                        break
                    p_boxes, p_scores, p_cls = self.model.predict_val(sample)
                    all_preds.append((
                        p_boxes.cpu().numpy().astype(np.float32),
                        p_scores.cpu().numpy().astype(np.float32),
                        p_cls.cpu().numpy().astype(np.int32)))

                    gt_list_b, gt_list_c = [], []
                    if sample.person_boxes.numel() > 0:
                        n = len(sample.person_boxes)
                        gt_list_b.append(sample.person_boxes.numpy())
                        gt_list_c.append(np.zeros(n, dtype=np.int32))
                    if sample.detect_boxes.numel() > 0:
                        for j in range(len(sample.detect_boxes)):
                            gt_list_b.append(sample.detect_boxes[j].numpy().reshape(1, 4))
                            gt_list_c.append(
                                np.array([sample.detect_classes[j].item()], dtype=np.int32))

                    if gt_list_b:
                        all_gts.append((np.concatenate(gt_list_b, axis=0),
                                        np.concatenate(gt_list_c, axis=0)))
                    else:
                        all_gts.append((np.zeros((0, 4), dtype=np.float32),
                                        np.zeros(0, dtype=np.int32)))
                    collected += 1
                if collected >= max_samples:
                    break
            if collected >= max_samples:
                break

        aps = {}
        for cls_idx, cls_name in [(0, "person"), (1, "fire"), (2, "water")]:
            aps[cls_name] = self._compute_ap(all_preds, all_gts, cls_idx)

        valid = [v for v in aps.values() if v is not None]
        mAP = float(np.mean(valid)) if valid else 0.0
        self._swap_ema(to_ema=True)  # swap back
        return mAP, aps

    @torch.no_grad()
    def _compute_score_metrics(self, val_loaders, max_samples=500):
        """Compute composite validation score.

        score is the mean of available metrics among:
        mAP@0.5, keypoint PCK accuracy, helmet accuracy, and smoke accuracy.
        Keypoints are counted correct when visible GT keypoints fall within
        5% of the matched person box max side. Person matching uses IoU >= 0.5.
        """
        self.model.eval()
        self._swap_ema(to_ema=True)

        all_preds = []
        all_gts = []
        kpt_correct = kpt_total = 0
        helmet_correct = helmet_total = 0
        smoke_correct = smoke_total = 0
        collected = 0

        try:
            for dl in val_loaders.values():
                for batch in dl:
                    for sample in batch:
                        if collected >= max_samples:
                            break
                        pred = self.model.predict_val_full(sample)

                        p_boxes = pred["boxes"].detach().cpu().numpy().astype(np.float32)
                        p_scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
                        p_cls = pred["classes"].detach().cpu().numpy().astype(np.int32)
                        all_preds.append((p_boxes, p_scores, p_cls))

                        gt_list_b, gt_list_c = [], []
                        if sample.person_boxes.numel() > 0:
                            n = len(sample.person_boxes)
                            gt_list_b.append(sample.person_boxes.numpy())
                            gt_list_c.append(np.zeros(n, dtype=np.int32))
                        if sample.detect_boxes.numel() > 0:
                            gt_list_b.append(sample.detect_boxes.numpy())
                            gt_list_c.append(sample.detect_classes.numpy().astype(np.int32))

                        if gt_list_b:
                            all_gts.append((np.concatenate(gt_list_b, axis=0),
                                            np.concatenate(gt_list_c, axis=0)))
                        else:
                            all_gts.append((np.zeros((0, 4), dtype=np.float32),
                                            np.zeros(0, dtype=np.int32)))

                        if sample.person_boxes.numel() > 0:
                            gt_boxes = sample.person_boxes.numpy().astype(np.float32)
                            gt_kpts = sample.person_kpts.numpy().astype(np.float32)
                            gt_helmet = sample.person_helmet.numpy().astype(np.float32)
                            gt_smoke = sample.person_smoke.numpy().astype(np.float32)

                            pred_boxes = pred["person_boxes"].detach().cpu().numpy().astype(np.float32)
                            pred_scores = pred["person_scores"].detach().cpu().numpy().astype(np.float32)
                            pred_kpts = pred["person_kpts"].detach().cpu().numpy().astype(np.float32)
                            pred_helmet_logits = pred["person_helmet"].detach().cpu()
                            pred_smoke_logits = pred["person_smoke"].detach().cpu()

                            matched_gt = set()
                            for pi in np.argsort(-pred_scores):
                                if len(gt_boxes) == 0:
                                    break
                                ious = self._box_iou_batch(pred_boxes[pi], gt_boxes)
                                for gi in np.argsort(-ious):
                                    if int(gi) in matched_gt:
                                        continue
                                    if ious[gi] < 0.5:
                                        continue
                                    matched_gt.add(int(gi))

                                    visible = gt_kpts[gi, :, 2] > 0
                                    if visible.any():
                                        box = gt_boxes[gi]
                                        scale = max(box[2] - box[0], box[3] - box[1], 1.0)
                                        dist = np.linalg.norm(
                                            pred_kpts[pi, :, :2] - gt_kpts[gi, :, :2], axis=1)
                                        kpt_correct += int((dist[visible] <= 0.05 * scale).sum())
                                        kpt_total += int(visible.sum())

                                    if gi < len(gt_helmet) and gt_helmet[gi] >= 0:
                                        helmet_prob = torch.sigmoid(pred_helmet_logits[pi]).item()
                                        pred_helmet = 0 if helmet_prob > 0.5 else 1
                                        helmet_correct += int(pred_helmet == int(gt_helmet[gi]))
                                        helmet_total += 1

                                    if gi < len(gt_smoke) and gt_smoke[gi] >= 0:
                                        smoke_prob = torch.sigmoid(pred_smoke_logits[pi]).item()
                                        pred_smoke = 1 if smoke_prob > 0.5 else 0
                                        smoke_correct += int(pred_smoke == int(gt_smoke[gi]))
                                        smoke_total += 1
                                    break

                        collected += 1
                    if collected >= max_samples:
                        break
                if collected >= max_samples:
                    break

            aps = {}
            for cls_idx, cls_name in [(0, "person"), (1, "fire"), (2, "water")]:
                aps[cls_name] = self._compute_ap(all_preds, all_gts, cls_idx)
            valid_ap = [v for v in aps.values() if v is not None]
            mAP = float(np.mean(valid_ap)) if valid_ap else 0.0

            metrics = {"mAP@0.5": mAP}
            for k, v in aps.items():
                if v is not None:
                    metrics[f"AP_{k}"] = v

            if kpt_total > 0:
                metrics["kpt_acc"] = kpt_correct / kpt_total
            if helmet_total > 0:
                metrics["helmet_acc"] = helmet_correct / helmet_total
            if smoke_total > 0:
                metrics["smoke_acc"] = smoke_correct / smoke_total

            score_parts = [metrics["mAP@0.5"]]
            for key in ("kpt_acc", "helmet_acc", "smoke_acc"):
                if key in metrics:
                    score_parts.append(metrics[key])
            metrics["score"] = float(np.mean(score_parts)) if score_parts else 0.0
            return metrics
        finally:
            self._swap_ema(to_ema=True)

    def _compute_ap(self, all_preds, all_gts, cls_idx, iou_thresh=0.5):
        """Compute AP@iou_thresh for a single class (101-point interpolation)."""
        import numpy as np

        # Flatten all detections with image index, sort by confidence
        detections = []
        for img_idx, (boxes, scores, classes) in enumerate(all_preds):
            for i in np.where(classes == cls_idx)[0]:
                detections.append((img_idx, boxes[i], float(scores[i])))
        detections.sort(key=lambda x: x[2], reverse=True)

        # GT matched flags
        gt_matched = [np.zeros(len(gts[0]), dtype=bool) for gts in all_gts]
        gt_counts = [int(np.sum(gts[1] == cls_idx)) for gts in all_gts]
        total_gt = sum(gt_counts)

        if total_gt == 0:
            return None  # AP undefined for this class

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for det_idx, (img_idx, det_box, _) in enumerate(detections):
            gt_boxes, gt_cls = all_gts[img_idx]
            mask = gt_cls == cls_idx
            gt_boxes_cls = gt_boxes[mask]

            if len(gt_boxes_cls) == 0:
                fp[det_idx] = 1
                continue

            # Map local indices → global GT indices for matching
            local_to_global = np.where(mask)[0]
            ious = self._box_iou_batch(det_box, gt_boxes_cls)

            best_iou, best_local = 0.0, -1
            for li in range(len(gt_boxes_cls)):
                gi = local_to_global[li]
                if not gt_matched[img_idx][gi] and ious[li] > best_iou:
                    best_iou = float(ious[li])
                    best_local = li

            if best_iou >= iou_thresh:
                tp[det_idx] = 1
                gt_matched[img_idx][local_to_global[best_local]] = True
            else:
                fp[det_idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / total_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            ap += (np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0) / 101.0
        return float(ap)

    @staticmethod
    def _box_iou_batch(box, boxes):
        """IoU between one box [4] and multiple boxes [N,4]."""
        import numpy as np
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter = w * h
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-16)

    def save(self, path, metrics=None):
        # 保存时使用 EMA 权重 (若启用)
        self._swap_ema(to_ema=True)
        state = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        if self.scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if self.ema_enabled:
            state["ema_state"] = {
                name: t.clone() for name, t in self._ema_state.items()}
        torch.save(state, str(path))
        self._swap_ema(to_ema=True)  # swap back
        print(f"  Saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema_enabled and "ema_state" in ckpt:
            self._ema_state = {
                name: t.to(self.device)
                for name, t in ckpt["ema_state"].items()}
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        print(f"  Loaded: {path} (epoch {self.current_epoch})")

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def fit(self, epochs, train_loaders, val_loaders=None, save_prefix="model",
            dataset_weights=None):
        print(f"\n{'='*50}")
        print(f"Stage: {save_prefix} | Epochs: {epochs} (start from {self.current_epoch}) | "
              f"Datasets: {list(train_loaders.keys())}")
        if dataset_weights:
            print(f"  dataset_weights={dataset_weights}")
        print(f"  save_interval={self.save_interval} val_interval={self.val_interval} "
              f"save_best_by={self.save_best_by}")
        print(f"{'='*50}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)
            t0 = time.time()
            train_m = self.train_epoch(train_loaders, epochs, dataset_weights)
            elapsed = time.time() - t0

            log = f"Epoch {epoch+1:3d}/{epochs} | {elapsed:.0f}s | "
            log += " ".join(f"{k}={v:.4f}" for k, v in sorted(train_m.items()))

            if self.writer:
                for k, v in train_m.items():
                    self.writer.add_scalar(f"epoch/train_{k}", v, epoch)

            do_val = val_loaders and (epoch + 1) % self.val_interval == 0
            if do_val:
                val_m = self.validate(val_loaders)
                val_parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(val_m.items()))
                log += " | " + val_parts
                if self.writer:
                    for k, v in val_m.items():
                        self.writer.add_scalar(f"epoch/{k}", v, epoch)

                score_metrics = None
                if self.map_enabled and hasattr(self.model, "predict_val_full"):
                    try:
                        score_metrics = self._compute_score_metrics(
                            val_loaders, max_samples=self.map_samples)
                        val_m.update(score_metrics)
                        metric_parts = " ".join(
                            f"{k}={v:.4f}" for k, v in sorted(score_metrics.items()))
                        log += " | " + metric_parts
                        if self.writer:
                            for k, v in score_metrics.items():
                                self.writer.add_scalar(f"epoch/{k}", v, epoch)
                    except Exception as e:
                        if self.save_best_by == "score":
                            raise RuntimeError("save_best_by='score' requires successful score computation") from e
                        log += f" | score=err({e})"

                if self.save_best_by == "score":
                    if score_metrics is None:
                        raise RuntimeError("save_best_by='score' requires map.enabled=true and model.predict_val_full")
                    current = score_metrics["score"]
                    improved = current > self.best_score
                else:
                    current = val_m.get("val_total", float("inf"))
                    improved = current < self.best_score
                if improved:
                    self.best_score = current
                    self.save(self.save_dir / f"{save_prefix}_best.pt", val_m)
            elif not val_loaders:
                if self.save_best_by == "score":
                    raise RuntimeError("save_best_by='score' requires a validation set")
                if train_m["loss"] < self.best_score:
                    self.best_score = train_m["loss"]

            print(log)

            if (epoch + 1) % self.save_interval == 0:
                self.save(self.save_dir / f"{save_prefix}_epoch{epoch+1}.pt")

        self.save(self.save_dir / f"{save_prefix}_last.pt")
        print(f"Best {self.save_best_by}: {self.best_score:.4f}")

        self.close()

