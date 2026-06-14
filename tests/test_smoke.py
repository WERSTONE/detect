from pathlib import Path

import yaml


def test_vigil_package_registers_models():
    import Vigil
    from Vigil.models.registry import list_models

    assert Vigil.__name__ == "Vigil"
    assert {"vigil_v2", "yolov8", "yolov8_pose"}.issubset(set(list_models()))


def test_train_config_dataset_paths_exist():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "train.yaml").read_text(encoding="utf-8"))

    dataset_paths = set()
    for stage in ("coco_pose_pretrain", "pretrain", "finetune"):
        for spec in cfg[stage]["datasets"].values():
            dataset_paths.add(root / spec["path"])

    missing = [str(path.relative_to(root)) for path in sorted(dataset_paths) if not path.exists()]
    assert not missing


def test_vigil_v2_auto_weight_path_exists():
    from Vigil.models.registry import _resolve_weights

    weight_path = _resolve_weights("vigil_v2", pretrained=True)
    assert weight_path is not None
    assert Path(weight_path).exists()
    assert Path(weight_path).name in {"finetune_best.pt", "pretrain_best.pt", "finetune_last.pt", "pretrain_last.pt", "best.pt"}
