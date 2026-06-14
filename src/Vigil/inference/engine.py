"""推理引擎 — 模型无关.

模型契约 (VigilModelBase): model.detect(frame: np.ndarray) → dict:
    {"person": {"boxes": [N,4], "scores": [N], "kpts": [N,17,3],
                "helmet": [N], "smoking": [N]},
     "fire":   {"boxes": [M,4], "scores": [M]},
     "water":  {"boxes": [K,4], "scores": [K]}}
    所有坐标为原始帧像素.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from Vigil.models.base import VigilModelBase


@dataclass
class Person:
    bbox: list[float]               # xyxy, 原始帧坐标
    confidence: float               # 检测置信度
    helmet_status: int              # 0=佩戴, 1=未佩戴
    helmet_conf: float              # 安全帽属性置信度
    smoking_status: int             # 0=未吸烟, 1=吸烟
    smoking_conf: float             # 吸烟属性置信度
    keypoints: list[list[float]]    # [17, 3] xyv


@dataclass
class Anomaly:
    bbox: list[float]
    class_name: str
    confidence: float


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    persons: list[Person]
    anomalies: list[Anomaly]
    events: list[dict]
    latency_ms: float


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i.item())
        box_i = boxes[i]
        rest = boxes[order[1:]]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        lt = torch.max(box_i[:2], rest[:, :2])
        rb = torch.min(box_i[2:], rest[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / (area_i + area_rest - inter + 1e-8)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


class InferenceEngine:
    """模型无关推理引擎. 模型只需实现 VigilModelBase 协议."""

    def __init__(self, model: VigilModelBase, config: dict):
        inf = config.get("inference", {})
        pp = config.get("postprocess", {})

        self.device = torch.device(
            "cuda" if inf.get("device", "cpu") == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = model
        if hasattr(self.model, 'eval'):
            self.model = self.model.eval().to(self.device)

        self.conf_person  = inf.get("conf_threshold_person", 0.25)
        self.conf_anomaly = inf.get("conf_threshold_anomaly", 0.15)
        self.iou_thresh   = inf.get("iou_threshold", 0.45)
        self.frame_count  = 0

        from Vigil.postprocess.temporal import PostProcessor
        self.postprocessor = PostProcessor(pp)
        self._warmup()

    def _warmup(self):
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        self.model.detect(dummy)

    def infer(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_count += 1

        detections = self.model.detect(frame)

        persons   = self._extract_persons(detections)
        anomalies = self._extract_anomalies(detections)
        events    = self.postprocessor.process_frame(persons, anomalies, *frame.shape[:2])
        latency   = (time.perf_counter() - t0) * 1000

        return FrameResult(
            frame_id=self.frame_count, timestamp=time.time(),
            persons=persons, anomalies=anomalies,
            events=events, latency_ms=latency)

    # ── 阈值过滤 + NMS ──

    def _extract_persons(self, detections):
        entry = detections.get("person")
        if entry is None:
            return []
        boxes, scores = entry["boxes"], entry["scores"]
        kpts, helmet, smoking = entry["kpts"], entry["helmet"], entry["smoking"]

        keep = scores > self.conf_person
        if not keep.any():
            return []
        boxes, scores = boxes[keep], scores[keep]
        kpts, helmet, smoking = kpts[keep], helmet[keep], smoking[keep]

        keep_nms = _nms(boxes, scores, self.iou_thresh)
        boxes, scores = boxes[keep_nms], scores[keep_nms]
        kpts, helmet, smoking = kpts[keep_nms], helmet[keep_nms], smoking[keep_nms]

        results = []
        for i in range(len(boxes)):
            h = torch.sigmoid(helmet[i]).item()
            s = torch.sigmoid(smoking[i]).item()
            if not np.isfinite(h):
                helmet_status, helmet_conf = -1, 0.0
            else:
                helmet_status = 0 if h > 0.5 else 1
                helmet_conf = h if helmet_status == 0 else 1 - h
            if not np.isfinite(s):
                smoking_status, smoking_conf = -1, 0.0
            else:
                smoking_status = 1 if s > 0.5 else 0
                smoking_conf = s if smoking_status == 1 else 1 - s
            results.append(Person(
                bbox=boxes[i].clamp(min=0).tolist(),
                confidence=scores[i].item(),
                helmet_status=helmet_status,
                helmet_conf=helmet_conf,
                smoking_status=smoking_status,
                smoking_conf=smoking_conf,
                keypoints=kpts[i].cpu().tolist(),
            ))
        return results

    def _extract_anomalies(self, detections):
        results = []
        for cls_name in ("fire", "water"):
            entry = detections.get(cls_name)
            if entry is None:
                continue
            boxes, scores = entry["boxes"], entry["scores"]
            keep = scores > self.conf_anomaly
            if not keep.any():
                continue
            boxes, scores = boxes[keep], scores[keep]
            keep_nms = _nms(boxes, scores, self.iou_thresh)
            boxes, scores = boxes[keep_nms], scores[keep_nms]
            for box, score in zip(boxes, scores):
                results.append(Anomaly(
                    bbox=box.clamp(min=0).tolist(),
                    class_name=cls_name,
                    confidence=score.item(),
                ))
        return results


def create_engine(model_name: str = "vigil_v2",
                  config_path: str = "config/config.yaml",
                  device: str = None,
                  pretrained=None,
                  **model_kwargs) -> InferenceEngine:
    """便捷工厂: 从注册表创建模型 + 从 YAML 加载配置 → 引擎.

    Args:
        model_name: 注册的模型名 (如 "vigil_v2").
        config_path: YAML 配置文件路径.
        device: 推理设备.
        pretrained: None=随机初始化, str=指定路径, True=自动查找.
        **model_kwargs: 传递给模型工厂的参数.
    """
    import os

    from Vigil.models.registry import create_model as create_registered_model

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    inf = config.setdefault("inference", {})
    inf.setdefault("device", "cpu")
    if device:
        inf["device"] = device

    # 从 config 读取模型参数 (优先级低于显式传入的 kwargs)
    yaml_model_kwargs = config.get("model", {}).get("kwargs", {})
    yaml_model_kwargs.update(model_kwargs)
    model_kwargs = yaml_model_kwargs

    # Resolve inference weights explicitly. The repository does not ship checkpoints,
    # so CLI inference should fail loudly instead of using random initialization.
    if pretrained is True:
        candidates = [
            f"checkpoints/{model_name}/finetune_best.pt",
            f"checkpoints/{model_name}/pretrain_best.pt",
            f"checkpoints/{model_name}/finetune_last.pt",
            f"checkpoints/{model_name}/pretrain_last.pt",
            f"checkpoints/{model_name}/best.pt",
        ]
        if model_name == "yolov8":
            candidates.append("checkpoints/yolov8/yolov8n.pt")
        elif model_name in ("yolov8_pose", "yolov8-pose"):
            candidates.append("checkpoints/yolov8_pose/yolov8n-pose.pt")

        pretrained = next((c for c in candidates if os.path.exists(c)), None)
        if pretrained is None:
            searched = ", ".join(candidates)
            raise FileNotFoundError(
                f"No weights found for model {model_name!r}. Searched: {searched}. "
                "Place weights under checkpoints/{model_name}/ or pass --weights. "
                "Use pretrained=None only for deliberate random-initialized smoke tests."
            )

    model = create_registered_model(model_name, pretrained=pretrained, **model_kwargs)
    return InferenceEngine(model, config)

