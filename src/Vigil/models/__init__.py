# 导入当前仓库内存在的模型模块以触发注册。
import Vigil.models.vigil_v2.model  # noqa: F401
import Vigil.models.yolov8.model  # noqa: F401
import Vigil.models.yolov8_pose.model  # noqa: F401
from Vigil.models.base import VigilModelBase
from Vigil.models.registry import MODEL_REGISTRY, create_model, list_models, register_model

