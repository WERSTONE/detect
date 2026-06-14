"""模型基类 — 推理 / 训练 / 注册的统一契约."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class VigilModelBase(ABC):
    """所有模型必须实现此基类.

    推理契约: detect(frame) → dict
    训练契约: compute_loss(sample) → {"total": Tensor, ...}
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> dict:
        """单帧推理.

        Args:
            frame: (H, W, 3) uint8 RGB numpy 数组.

        Returns:
            {"person": {"boxes": [N,4], "scores": [N], "kpts": [N,17,3],
                        "helmet": [N], "smoking": [N]},
             "fire":   {"boxes": [M,4], "scores": [M]},
             "water":  {"boxes": [K,4], "scores": [K]}}
        """
        ...

    @abstractmethod
    def compute_loss(self, samples) -> dict[str, torch.Tensor | float]:
        """从一批 VigilSample 计算损失.

        模型内部负责: forward → GT构建 → 正负样本分配 → 设备迁移 → 损失计算.
        训练器只负责传 batch 和反向传播，不关心模型内部结构.

        Args:
            samples: List[VigilSample] — 一个 batch 的样本列表.

        Returns:
            dict 必须包含 "total" (Tensor, 用于 backward).
            其余键用于日志输出，模型可自由定义.
        """
        ...

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """模型期望的输入尺寸 (width, height)."""
        ...

    @property
    @abstractmethod
    def num_params(self) -> int:
        """模型可训练参数总数."""
        ...

