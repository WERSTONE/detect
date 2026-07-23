"""Model construction helpers."""

from test_model.model.bifpn import ModelE_BiFPN
from test_model.model.detect import BifpnDetectModel
from test_model.model.yolov8nano import YOLOv8NanoModel


def create_model(name="bifpn_dual", **kwargs):
    if name in ("bifpn", "bifpn_dual"):
        model = ModelE_BiFPN(**kwargs)
    elif name in ("bifpn_detect", "bifpn_det"):
        model = BifpnDetectModel(**kwargs)
    elif name in ("yolov8n", "yolov8nano"):
        model = YOLOv8NanoModel(**kwargs)
    else:
        raise ValueError(
            'Supported models: "bifpn_dual", "bifpn_detect", "yolov8n".')
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
