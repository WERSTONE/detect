"""
Vigil pump-room multi-task monitoring CLI.

Usage:
    vigil demo <image> --model vigil_v2 --weights checkpoints/vigil_v2/pretrain_best.pt
    vigil live --cam 0 --model yolov8_pose --weights checkpoints/yolov8_pose/yolov8n-pose.pt --show
    python -m Vigil.main live --video test.mp4 --model vigil_v2 --weights checkpoints/vigil_v2/pretrain_best.pt
"""
import argparse
import sys
import time

import cv2
import yaml
from loguru import logger

COCO_SKELETON = [
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 6),                # shoulders
    (5, 11), (6, 12),      # torso
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
    (0, 1), (0, 2),        # nose to eyes
    (1, 3), (2, 4),        # eyes to ears
]

LEFT_KPTS = {5, 7, 9, 11, 13, 15}
RIGHT_KPTS = {6, 8, 10, 12, 14, 16}


def _build_engine(args):
    from Vigil.inference.engine import create_engine
    pretrained = args.weights if args.weights else True
    return create_engine(model_name=args.model,
                         config_path=args.config, device=args.device,
                         pretrained=pretrained)


def _clip_box(box, width, height):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    ]


def _draw_label(frame, lines, origin, fg=(255, 255, 255), bg=(35, 35, 35)):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_h = 17
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + 10
    box_h = line_h * len(lines) + 8
    x = max(0, min(frame.shape[1] - box_w - 1, x))
    y = max(box_h + 2, y)
    cv2.rectangle(frame, (x, y - box_h), (x + box_w, y), bg, -1)
    cv2.rectangle(frame, (x, y - box_h), (x + box_w, y), fg, 1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + 5, y - box_h + 15 + i * line_h),
                    font, scale, fg, thickness, cv2.LINE_AA)


def _draw_pose(frame, keypoints, conf_thresh=0.3):
    if not keypoints or len(keypoints) < 17:
        return

    pts = []
    for kp in keypoints[:17]:
        if len(kp) < 3:
            pts.append(None)
            continue
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf <= conf_thresh or x <= 0 or y <= 0:
            pts.append(None)
        else:
            pts.append((int(round(x)), int(round(y)), conf))

    for a, b in COCO_SKELETON:
        if pts[a] is None or pts[b] is None:
            continue
        color = (80, 220, 80)
        if a in LEFT_KPTS or b in LEFT_KPTS:
            color = (255, 170, 60)
        elif a in RIGHT_KPTS or b in RIGHT_KPTS:
            color = (60, 180, 255)
        cv2.line(frame, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)

    for idx, pt in enumerate(pts):
        if pt is None:
            continue
        color = (80, 220, 80)
        if idx in LEFT_KPTS:
            color = (255, 170, 60)
        elif idx in RIGHT_KPTS:
            color = (60, 180, 255)
        cv2.circle(frame, pt[:2], 4, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, pt[:2], 3, color, -1, cv2.LINE_AA)


def _draw_results(frame, result, latency_ms):
    """在画面上绘制检测框和事件"""
    h, w = frame.shape[:2]

    for p in result.persons:
        bx = _clip_box(p.bbox, w, h)
        helmet_unknown = int(p.helmet_status) < 0
        smoke_unknown = int(p.smoking_status) < 0
        helmet_bad = int(p.helmet_status) == 1
        smoke_bad = int(p.smoking_status) == 1
        box_color = (0, 165, 255) if helmet_bad or smoke_bad else (40, 220, 90)
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), box_color, 2, cv2.LINE_AA)

        helmet_text = "HELMET N/A" if helmet_unknown else ("NO_HELMET" if helmet_bad else "HELMET")
        smoke_text = "SMOKE N/A" if smoke_unknown else ("SMOKE" if smoke_bad else "NO_SMOKE")
        label_lines = [
            f"PERSON {p.confidence:.2f}",
            helmet_text if helmet_unknown else f"{helmet_text} {p.helmet_conf:.2f}",
            smoke_text if smoke_unknown else f"{smoke_text} {p.smoking_conf:.2f}",
        ]
        _draw_label(frame, label_lines, (bx[0], bx[1] - 6), bg=(28, 35, 30) if not helmet_bad and not smoke_bad else (35, 25, 15))
        _draw_pose(frame, p.keypoints)

    for a in result.anomalies:
        bx = [int(a.bbox[0]), int(a.bbox[1]), int(a.bbox[2]), int(a.bbox[3])]
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
        cv2.putText(frame, f"{a.class_name} {a.confidence:.2f}",
                    (bx[0], bx[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 延迟 & FPS
    cv2.putText(frame, f"{latency_ms:.0f}ms", (w - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 告警事件计数
    y0 = h - 20
    for ev in result.events:
        txt = f"[{ev.get('type', '?')}] t{ev.get('task_id', '?')}"
        cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y0 -= 16
        if y0 < h - 120: break


def cmd_demo(args):
    engine = _build_engine(args)
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot read: {args.image}")
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = engine.infer(rgb)

    logger.info(f"Frame: {len(result.persons)} persons, {len(result.anomalies)} anomalies, "
                f"{len(result.events)} events, latency={result.latency_ms:.1f}ms")
    for p in result.persons:
        logger.info(f"  Person: conf={p.confidence:.3f} helmet={p.helmet_status} smoke={p.smoking_status}")
    for a in result.anomalies:
        logger.info(f"  Anomaly: {a.class_name} conf={a.confidence:.3f}")
    for ev in result.events:
        logger.info(f"  Event: [{ev.get('type','?')}] task={ev.get('task_id','?')} conf={ev.get('confidence',0):.3f}")

    if args.show:
        _draw_results(img, result, result.latency_ms)
        cv2.imshow("Vigil — Demo", img)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_live(args):
    engine = _build_engine(args)

    if args.cam is not None:
        cap = cv2.VideoCapture(int(args.cam))
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        from Vigil.pipeline.gst_pipeline import VideoPipeline
        logger.info("RTSP mode (no --video/--cam)")
        config = yaml.safe_load(open(args.config, encoding="utf-8"))

        def on_frame(frame):
            result = engine.infer(frame)
            for ev in result.events:
                logger.info(f"[{ev.get('type','?')}] conf={ev.get('confidence',0):.3f}")

        pipeline = VideoPipeline(source_uri=config["pipeline"]["source"],
                                  inference_callback=on_frame,
                                  fps=config["pipeline"]["fps"])
        pipeline.start()
        try:
            while pipeline.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pipeline.stop()
        return

    if not cap.isOpened():
        logger.error("Cannot open video source")
        sys.exit(1)

    logger.info("Running... press ESC to stop")
    frame_count = 0
    total_latency = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = engine.infer(rgb)
        frame_count += 1
        total_latency += result.latency_ms

        if args.show:
            _draw_results(frame, result, result.latency_ms)
            cv2.imshow("Vigil — Live", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count}: {len(result.persons)} persons, "
                        f"{len(result.anomalies)} anomalies, "
                        f"avg latency={total_latency/frame_count:.1f}ms")

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Done. {frame_count} frames, avg latency={total_latency/max(frame_count,1):.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Vigil 泵房监控系统")
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="单图推理")
    p_demo.add_argument("image")
    p_demo.add_argument("--show", action="store_true", help="显示检测结果画面")

    p_live = sub.add_parser("live", help="实时推理")
    src = p_live.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="本地视频文件")
    src.add_argument("--cam", default=None, help="摄像头索引 (0=默认摄像头)")
    p_live.add_argument("--show", action="store_true", help="显示实时画面")

    for p in [p_demo, p_live]:
        p.add_argument("--model", default="vigil_v2", help="注册的模型名称")
        p.add_argument("--config", default="config/config.yaml")
        p.add_argument("--weights", default=None, help="权重路径 (省略则自动查找)")
        p.add_argument("--device", default=None)

    args = parser.parse_args()
    if args.cmd == "demo": cmd_demo(args)
    elif args.cmd == "live": cmd_live(args)
    else: parser.print_help()


if __name__ == "__main__":
    main()

