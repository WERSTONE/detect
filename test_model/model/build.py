"""Model construction helpers."""

from test_model.model.bifpn import DomainAttrBiFPN, ModelE_BiFPN
from test_model.model.detect import BifpnDetectModel
from test_model.model.pose import BifpnPoseModel


def create_model(name="bifpn_dual", **kwargs):
    if name in ("bifpn", "bifpn_dual"):
        model = ModelE_BiFPN(**kwargs)
    elif name in ("bifpn_dual_domain_attr", "bifpn_domain_attr"):
        model = DomainAttrBiFPN(**kwargs)
    elif name in ("bifpn_detect", "bifpn_det"):
        model = BifpnDetectModel(**kwargs)
    elif name in ("bifpn_pose", "bifpn_pose_only"):
        model = BifpnPoseModel(**kwargs)
    else:
        raise ValueError(
            'Supported models: "bifpn_dual", "bifpn_detect", '
            '"bifpn_pose", "bifpn_dual_domain_attr".')
    print(f"  Created {name}: {model.num_params / 1e6:.2f}M params")
    return model
