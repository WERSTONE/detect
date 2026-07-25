"""Model construction helpers."""

from test_model.model.bifpn import ModelE_BiFPN
from test_model.model.detect import BifpnDetectModel
from test_model.model.pose import BifpnPoseModel, YOLOv8MPoseModel
from test_model.model.yolov8nano import YOLOv8NanoModel


def create_model(name="bifpn_dual", **kwargs):
    if name in ("bifpn", "bifpn_dual"):
        model = ModelE_BiFPN(**kwargs)
    elif name in ("bifpn_detect", "bifpn_det"):
        model = BifpnDetectModel(**kwargs)
    elif name in ("yolov8n", "yolov8nano"):
        model = YOLOv8NanoModel(**kwargs)
    elif name in ("yolov8m_pose", "yolov8m-pose"):
        model = YOLOv8MPoseModel(**kwargs)
    elif name in ("bifpn_pose", "bifpn_pose_only"):
        model = BifpnPoseModel(**kwargs)
    else:
        raise ValueError(
            'Supported models: "bifpn_dual", "bifpn_detect", "bifpn_pose", "yolov8n", "yolov8m_pose".')
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
