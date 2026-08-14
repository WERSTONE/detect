"""Export trained BiFPN models to inference-only ONNX graphs."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_model.model import create_model
from test_model.final.model import create_final_model
from test_model.final_pose_attr.model import create_final_pose_attr_model


class RawBiFPNDualExport(nn.Module):
    """ONNX wrapper that returns raw multi-scale head tensors only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        det_out, pose_out = self.model(images)
        return (
            det_out["cls"][0], det_out["reg"][0],
            det_out["cls"][1], det_out["reg"][1],
            det_out["cls"][2], det_out["reg"][2],
            pose_out["cls"][0], pose_out["reg"][0], pose_out["kpt"][0],
            pose_out["cls"][1], pose_out["reg"][1], pose_out["kpt"][1],
            pose_out["cls"][2], pose_out["reg"][2], pose_out["kpt"][2],
        )


class RawBiFPNDomainAttrExport(nn.Module):
    """ONNX wrapper for the domain-detection + pose + attribute model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outs = self.model(images)
        domain_out = outs["domain_det"]
        pose_out = outs["pose"]
        attr_out = outs.get("attr")
        attr_feats = attr_out["attr"] if attr_out is not None else pose_out["attr"]
        return (
            domain_out["cls"][0], domain_out["reg"][0],
            domain_out["cls"][1], domain_out["reg"][1],
            domain_out["cls"][2], domain_out["reg"][2],
            pose_out["cls"][0], pose_out["reg"][0], pose_out["kpt"][0],
            pose_out["cls"][1], pose_out["reg"][1], pose_out["kpt"][1],
            pose_out["cls"][2], pose_out["reg"][2], pose_out["kpt"][2],
            attr_feats[0], attr_feats[1], attr_feats[2],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained BiFPN checkpoint to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "test_model/config/bifpn_dual.yaml"))
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--model", default=None, help="Override model name from config")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size")
    parser.add_argument("--batch", type=int, default=1, help="Dummy export batch size")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier after export")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export dynamic batch axis")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_kwargs_from_config(model_name, cfg):
    data_cfg = cfg.get("data", {}) or {}
    neck_cfg = cfg.get("neck", {}) or {}
    assigner_cfg = cfg.get("assigner", {}) or {}

    common = {
        "input_size": data_cfg.get("input_size", 640),
        "assigner_topk": assigner_cfg.get("topk", 10),
        "assigner_alpha": assigner_cfg.get("alpha", 0.5),
        "assigner_beta": assigner_cfg.get("beta", 6.0),
        "assigner_eps": assigner_cfg.get("eps", 1.0e-9),
    }

    if model_name in ("bifpn", "bifpn_dual"):
        common.update({
            "num_kpts": cfg.get("num_kpts", 17),
            "num_det_classes": cfg.get("num_det_classes", 79),
            "neck_use_p2_context": neck_cfg.get("use_p2_context", False),
            "neck_downsample": neck_cfg.get("downsample", "conv"),
            "neck_out_channels": neck_cfg.get("out_channels", None),
        })
        return common

    if model_name in (
        "bifpn_dual_domain_attr",
        "bifpn_domain_attr",
        "final_three_head",
        "bifpn_final_three_head",
    ):
        domain_cfg = cfg.get("domain_det", {}) or {}
        attr_cfg = cfg.get("attributes", {}) or {}
        common.update({
            "num_kpts": cfg.get("num_kpts", 17),
            "num_det_classes": cfg.get("num_det_classes", 79),
            "reg_max": cfg.get("reg_max", 16),
            "neck_use_p2_context": neck_cfg.get("use_p2_context", False),
            "neck_downsample": neck_cfg.get("downsample", "conv"),
            "neck_out_channels": neck_cfg.get("out_channels", None),
            "domain_num_classes": domain_cfg.get("num_classes", 2),
            "domain_class_map": domain_cfg.get("class_map", None),
            "num_attrs": attr_cfg.get(
                "num_attrs",
                len(attr_cfg.get("names", [])) if attr_cfg.get("names") else 4,
            ),
            "attr_names": attr_cfg.get("names", None),
        })
        return common

    if model_name in ("final_pose_attr", "bifpn_final_pose_attr"):
        domain_cfg = cfg.get("domain_det", {}) or {}
        attr_cfg = cfg.get("pose_attr", {}) or {}
        loss_cfg = cfg.get("loss", {}) or {}
        common.update({
            "num_kpts": cfg.get("num_kpts", 17),
            "reg_max": cfg.get("reg_max", 16),
            "neck_use_p2_context": neck_cfg.get("use_p2_context", False),
            "neck_downsample": neck_cfg.get("downsample", "conv"),
            "neck_out_channels": neck_cfg.get("out_channels", None),
            "domain_num_classes": domain_cfg.get("num_classes", 4),
            "domain_class_map": domain_cfg.get("class_map", None),
            "num_attrs": attr_cfg.get(
                "num_attrs",
                len(attr_cfg.get("names", [])) if attr_cfg.get("names") else 4,
            ),
            "attr_names": attr_cfg.get("names", None),
            "attr_dropout": attr_cfg.get("attr_dropout", 0.0),
            "attr_consistency_weight": loss_cfg.get("attr_consistency_weight", 0.0),
        })
        return common

    raise ValueError(
        "This exporter supports bifpn_dual, bifpn_dual_domain_attr, "
        "final_three_head, and final_pose_attr only.")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise KeyError("Could not find model_state_dict/state_dict in checkpoint")


def main():
    args = parse_args()
    weights = Path(args.weights)
    cfg = load_config(args.config)
    model_name = args.model or cfg.get("model", "bifpn_dual")

    if args.imgsz is not None:
        cfg.setdefault("data", {})["input_size"] = int(args.imgsz)
    imgsz = int(cfg.get("data", {}).get("input_size", 640))

    output = Path(args.output) if args.output else weights.with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Creating model: {model_name}")
    if model_name in ("final_three_head", "bifpn_final_three_head"):
        model = create_final_model(model_name, **model_kwargs_from_config(model_name, cfg))
    elif model_name in ("final_pose_attr", "bifpn_final_pose_attr"):
        model = create_final_pose_attr_model(model_name, **model_kwargs_from_config(model_name, cfg))
    else:
        model = create_model(model_name, **model_kwargs_from_config(model_name, cfg))

    print(f"Loading checkpoint: {weights}")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys: {len(missing)}; first: {missing[:5]}")
    if unexpected:
        print(f"WARNING: unexpected keys: {len(unexpected)}; first: {unexpected[:5]}")

    model.eval().to(device)
    if model_name in (
        "bifpn_dual_domain_attr",
        "bifpn_domain_attr",
        "final_three_head",
        "bifpn_final_three_head",
        "final_pose_attr",
        "bifpn_final_pose_attr",
    ):
        wrapper = RawBiFPNDomainAttrExport(model).eval().to(device)
        attr_prefix = "pose_attr" if model_name in ("final_pose_attr", "bifpn_final_pose_attr") else "attr"
        output_names = [
            "domain_cls_s8", "domain_reg_s8",
            "domain_cls_s16", "domain_reg_s16",
            "domain_cls_s32", "domain_reg_s32",
            "pose_cls_s8", "pose_reg_s8", "pose_kpt_s8",
            "pose_cls_s16", "pose_reg_s16", "pose_kpt_s16",
            "pose_cls_s32", "pose_reg_s32", "pose_kpt_s32",
            f"{attr_prefix}_s8", f"{attr_prefix}_s16", f"{attr_prefix}_s32",
        ]
    else:
        wrapper = RawBiFPNDualExport(model).eval().to(device)
        output_names = [
            "det_cls_s8", "det_reg_s8",
            "det_cls_s16", "det_reg_s16",
            "det_cls_s32", "det_reg_s32",
            "pose_cls_s8", "pose_reg_s8", "pose_kpt_s8",
            "pose_cls_s16", "pose_reg_s16", "pose_kpt_s16",
            "pose_cls_s32", "pose_reg_s32", "pose_kpt_s32",
        ]
    dummy = torch.zeros(args.batch, 3, imgsz, imgsz, device=device)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    with torch.no_grad():
        print(f"Exporting ONNX: {output}")
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            dynamo=False,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
        except ImportError as exc:
            raise SystemExit(
                "onnxsim is required for --simplify. Install it with:\n"
                "  uv sync --extra export\n"
                "or:\n"
                "  python -m pip install onnxsim"
            ) from exc
        print(f"Simplifying ONNX: {output}")
        onnx_model = onnx.load(str(output))
        input_shapes = {"images": [args.batch, 3, imgsz, imgsz]}
        onnx_model, ok = simplify(onnx_model, input_shapes=input_shapes)
        if not ok:
            raise RuntimeError("onnxsim validation failed")
        onnx.save(onnx_model, str(output))

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"ONNX saved: {output} ({size_mb:.1f} MB)")
    print("Outputs are raw logits/distributions; decode/NMS is not embedded.")


if __name__ == "__main__":
    main()
