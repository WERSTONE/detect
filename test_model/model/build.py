"""Model construction helpers."""

from test_model.model.bifpn import ModelE_BiFPN
from test_model.model.detect import BifpnDetectModel


def create_model(name="bifpn_dual", **kwargs):
    if name in ("bifpn", "bifpn_dual"):
        model = ModelE_BiFPN(**kwargs)
    elif name in ("bifpn_detect", "bifpn_det"):
        model = BifpnDetectModel(**kwargs)
    else:
        raise ValueError(
            'Only BiFPN models are supported: "bifpn_dual" or "bifpn_detect".')
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
