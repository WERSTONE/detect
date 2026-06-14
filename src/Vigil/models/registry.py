"""模型注册表 — 按名称注册/查找模型工厂, 自动关联训练权重."""

from collections.abc import Callable
from pathlib import Path

from Vigil.models.base import VigilModelBase

MODEL_REGISTRY: dict[str, Callable[..., VigilModelBase]] = {}

CHECKPOINT_ROOT = "checkpoints"


def register_model(name: str):
    """注册模型工厂 (可用作装饰器或直接调用).

    用法:
        @register_model("vigil_v1")
        def create_model(pretrained=None, **kwargs): ...

        register_model("onnx_v1", lambda **kw: ONNXModel(**kw))
    """
    def decorator(factory):
        MODEL_REGISTRY[name] = factory
        return factory
    return decorator


def _resolve_weights(name: str, pretrained: bool | str | None) -> str | None:
    """解析权重路径.

    Args:
        name: 模型注册名.
        pretrained:
            - True (默认):  自动查找 checkpoints/{name}/best.pt
            - str:          显式路径
            - None / False: 跳过加载

    Returns:
        权重文件路径, 或 None 表示跳过.
    """
    if pretrained is None or pretrained is False:
        return None
    if isinstance(pretrained, str):
        return pretrained
    # pretrained=True -> 自动按统一 checkpoint 约定查找。
    root = Path(CHECKPOINT_ROOT) / name
    for filename in ("finetune_best.pt", "pretrain_best.pt", "finetune_last.pt", "pretrain_last.pt", "best.pt"):
        auto = root / filename
        if auto.exists():
            return str(auto)
    return None


def create_model(name: str, pretrained: bool | str | None = True, **kwargs
                 ) -> VigilModelBase:
    """从注册表创建模型实例, 自动加载对应权重.

    Args:
        name:       注册的模型名 (如 "vigil_v1").
        pretrained: True=自动加载 checkpoints/{name}/best.pt,
                    str=显式路径, None/False=跳过.
        **kwargs:   传递给模型工厂的参数.

    Raises:
        KeyError: 模型名未注册.
    """
    if name not in MODEL_REGISTRY:
        # 尝试自动导入 models/{name}/model.py 触发注册
        try:
            import importlib
            importlib.import_module(f"Vigil.models.{name}.model")
        except ImportError:
            pass
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"未知模型 '{name}'. 可用: {list(MODEL_REGISTRY.keys())}"
        )
    weights = _resolve_weights(name, pretrained)
    return MODEL_REGISTRY[name](pretrained=weights, **kwargs)


def list_models() -> list[str]:
    """列出所有已注册模型名称."""
    return list(MODEL_REGISTRY.keys())

