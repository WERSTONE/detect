"""
统一标签结构: person 行自带 helmet/smoking 属性，移除独立的头盔框/烟蒂框。

person 行格式: 0 cx cy w h [17×3 kpt] helmet_attr smoke_attr  (共58字段)
非 person 行:   保持不变

属性含义:
  helmet_attr: 0=戴了, 1=没戴
  smoke_attr:  0=没抽, 1=在抽

默认值:
  person:     helmet=1(没戴), smoke=0(没抽)
  helmet:     匹配覆盖 → helmet属性, smoke=0
  fire_smoke: helmet=0(戴了), smoke=0(没抽)
  smoking:    匹配覆盖 → smoke属性, helmet=1
  water_leak: helmet=1(没戴), smoke=0(没抽)

用法: python scripts/unify_labels.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path("data/processed")


def box_iou(b1, b2):
    """两个 xyxy 框的 IoU"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def xywh_to_xyxy(cx, cy, w_n, h_n, img_w=640, img_h=640):
    """归一化 cxcywh → xyxy 像素坐标 (img_w/h 仅用于缩放，不影响 IoU)"""
    x1 = (cx - w_n / 2) * img_w
    y1 = (cy - h_n / 2) * img_h
    x2 = (cx + w_n / 2) * img_w
    y2 = (cy + h_n / 2) * img_h
    return x1, y1, x2, y2


def process_dataset(name, helmet_default, smoke_default, match_config):
    """
    match_config: dict or None
      helmet: {"class_ids": [1,2], "attr_map": {1:0, 2:1}, "iou_thresh": 0.15}
      smoking: {"class_ids": [1], "attr_map": {1:1}, "dist_ratio": 0.5}
    """
    d = ROOT / name
    img_dir = d / "images"
    lbl_dir = d / "labels"

    new_lbls = {}  # stem → list of lines (strings)

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_path.stem
        person_lines = []   # [(raw_line, xyxy)]
        other_lines = []    # [(cls_id, raw_line, xyxy or None)]
        obj_lines = []      # non-person detection lines to keep as-is

        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                vals = list(map(float, parts[1:]))
                cx, cy, w_n, h_n = vals[0], vals[1], vals[2], vals[3]

                if cls_id == 0 and len(vals) >= 55:
                    # person + keypoints: keep raw data
                    person_lines.append((line, xywh_to_xyxy(cx, cy, w_n, h_n), vals))
                elif match_config and cls_id in match_config.get("class_ids", []):
                    if not match_config.get("discard"):
                        other_lines.append((cls_id, xywh_to_xyxy(cx, cy, w_n, h_n)))
                elif cls_id != 0:
                    # fire/water detection: keep as-is
                    obj_lines.append(line)

        # ── 属性匹配 ──
        person_attrs = []  # [(helmet, smoke) per person]

        if match_config and other_lines:
            for p_line, p_xyxy, p_vals in person_lines:
                best_helmet = helmet_default  # use default if no match
                best_smoke = smoke_default

                for o_cls, o_xyxy in other_lines:
                    match_type = match_config.get("match_type", "iou")
                    matched = False

                    if match_type == "center_inside":
                        # 小目标(烟蒂): 检查目标中心是否在人物框内
                        ocx = (o_xyxy[0] + o_xyxy[2]) / 2
                        ocy = (o_xyxy[1] + o_xyxy[3]) / 2
                        matched = (p_xyxy[0] <= ocx <= p_xyxy[2] and
                                   p_xyxy[1] <= ocy <= p_xyxy[3])
                    else:
                        iou = box_iou(p_xyxy, o_xyxy)
                        thresh = match_config.get("iou_thresh", 0.1)
                        matched = iou > thresh

                    if matched:
                        attr_val = match_config.get("attr_map", {}).get(o_cls)
                        if attr_val is not None:
                            target = match_config.get("target", "helmet")
                            if target == "smoke":
                                best_smoke = attr_val
                            else:
                                best_helmet = attr_val

                person_attrs.append((best_helmet, best_smoke))
        else:
            # 无匹配配置 → 全用默认值
            person_attrs = [(helmet_default, smoke_default)] * len(person_lines)

        # ── 生成新行 ──
        new_lines = []
        for i, ((line, _, _), (h_attr, s_attr)) in enumerate(
            zip(person_lines, person_attrs)):
            parts = line.strip().split()
            # parts[0]=class, parts[1:5]=bbox, parts[5:56]=51 kpt values
            # 追加 helmet + smoke
            parts.append(str(h_attr))   # helmet
            parts.append(str(s_attr))   # smoke
            new_lines.append(" ".join(parts))

        # 保留非 person 检测行 (fire/water)
        new_lines.extend(obj_lines)
        new_lbls[stem] = new_lines

    # ── 写入临时目录 ──
    tmp_img = d / "_u_img"
    tmp_lbl = d / "_u_lbl"
    for td in (tmp_img, tmp_lbl):
        if td.exists():
            shutil.rmtree(td)
        td.mkdir()

    for i, (stem, lines) in enumerate(sorted(new_lbls.items())):
        # 找对应图片
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        new_stem = f"{name}_{i:05d}"
        shutil.copy2(img_path, tmp_img / f"{new_stem}{img_path.suffix}")
        with open(tmp_lbl / f"{new_stem}.txt", "w") as f:
            for line in lines:
                f.write(line + "\n")

    # ── 替换 ──
    shutil.rmtree(img_dir)
    shutil.rmtree(lbl_dir)
    tmp_img.rename(img_dir)
    tmp_lbl.rename(lbl_dir)

    # 统计
    total_persons = sum(
        sum(1 for l in lines if l.startswith("0 ")) for lines in new_lbls.values())
    total_objects = sum(
        sum(1 for l in lines if not l.startswith("0 ")) for lines in new_lbls.values())
    print(f"  {len(new_lbls)} images, {total_persons} persons, {total_objects} other detections")


def main():
    configs = {
        "person":     (1, 0, None),    # helmet=off(1), smoke=no(0)
        "helmet":     (1, 0, {         # 匹配 helmet 框: 中心在人物框内
            "class_ids": [1, 2],
            "attr_map": {1: 0, 2: 1},  # class1=helmet_on→0, class2=off→1
            "target": "helmet",
            "match_type": "center_inside",
        }),
        "fire_smoke": (0, 0, None),    # helmet=on(0), smoke=no(0)
        "smoking":    (1, 1, {         # 全是吸烟者，丢弃烟蒂框
            "class_ids": [1],
            "discard": True,
        }),
        "water_leak": (1, 0, None),    # helmet=off(1), smoke=no(0)
    }

    for name, (helm_def, smok_def, match_cfg) in configs.items():
        print(f"[{name}]")
        process_dataset(name, helm_def, smok_def, match_cfg)

    print("\n=== 验证 ===")
    for name in configs:
        d = ROOT / name
        n_img = len(list((d / "images").glob("*")))
        # 采样一行 person
        sample = None
        for lbl in sorted((d / "labels").glob("*.txt")):
            with open(lbl) as f:
                for line in f:
                    if line.startswith("0 "):
                        sample = line.strip()
                        break
            if sample:
                break
        n_fields = len(sample.split()) if sample else 0
        print(f"  {name:15s}: {n_img:4d} imgs, person line {n_fields} fields" +
              (f", ends with: ...{sample.rsplit(' ', 2)}" if sample else ""))

    # 更新 data.yaml names
    yaml_names = {
        "person":     {"0": "person"},
        "helmet":     {"0": "person"},
        "fire_smoke": {"0": "person", "1": "fire"},
        "smoking":    {"0": "person"},
        "water_leak": {"0": "person", "1": "water"},
    }
    for name, names in yaml_names.items():
        yaml_path = ROOT / name / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {json.dumps(str((ROOT/name).resolve()))}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            names_str = json.dumps({str(k): v for k, v in names.items()})
            f.write(f"names: {names_str}\n")

    print("\nDone. Backups at data/processed/*_bak/")


if __name__ == "__main__":
    main()

