# 已完成任务

## TASK-20260609-001：搭建项目骨架结构

- [x] 已完成

## TASK-20260609-002：在 NVIDIA GPU 环境中测试 YOLO pose 与 Vigil-v2

- [x] 已完成

当前 uv 环境：

```text
torch 2.3.1+cu121
torchvision 0.18.1+cu121
CUDA 12.1
GPU: NVIDIA GeForce MX450
```

测试视频：

```text
D:\Vigil\test.mp4
```

抽样帧：

```text
0, 3853, 7706, 11559, 15407
```

测试模型：

| 模型 | 权重来源 | 说明 |
|------|----------|------|
| YOLOv8n-pose | `checkpoints/yolo_pose/yolov8n-pose.pt` | Ultralytics pose 小模型 |
| YOLOv8s-pose | `checkpoints/yolo_pose/yolov8s-pose.pt` | Ultralytics pose 小型模型 |
| YOLO11n-pose | `checkpoints/yolo_pose/yolo11n-pose.pt` | Ultralytics YOLO11 pose 小模型 |
| Vigil-v2 | `checkpoints/vigil_v2/pretrain_best.pt` | 当前项目训练好的多任务模型 |

测试结果图：

```text
docs/assets/pose_video_sample_comparison.jpg
```

结果图内容：每一行对应一个模型，每一列对应一个抽样帧；图中展示了人体框、关键点或 Vigil-v2 的多任务叠加结果，并标注了每帧检测人数与单帧推理耗时。

抽样结果摘要：

| 模型 | frame 0 | frame 3853 | frame 7706 | frame 11559 | frame 15407 |
|------|---------|------------|------------|-------------|-------------|
| YOLOv8n-pose | 2 persons | 1 person | 3 persons | 2 persons | 3 persons |
| YOLOv8s-pose | 3 persons | 1 person | 2 persons | 2 persons | 3 persons |
| YOLO11n-pose | 3 persons | 1 person | 2 persons | 2 persons | 2 persons |
| Vigil-v2 | 4 persons | 2 persons | 3 persons | 4 persons | 3 persons |

结论：YOLO pose 系列和 Vigil-v2 均能在当前 uv + CUDA 环境中完成 `test.mp4` 抽样推理，Vigil-v2 使用项目内权重自动加载路径工作正常。

## TASK-20260609-003：分析 YOLOv8 与 YOLOv8-pose 是否共用 backbone

- [x] 已完成

依据当前环境中 Ultralytics 官方模型配置文件：

```text
.venv/Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml
.venv/Lib/site-packages/ultralytics/cfg/models/v8/yolov8-pose.yaml
.venv/Lib/site-packages/ultralytics/cfg/models/11/yolo11-pose.yaml
```

### YOLOv8 Detect 官方架构

参数：

```text
nc: 80
scales: n, s, m, l, x
```

backbone：

```text
Conv -> Conv -> C2f -> Conv -> C2f -> Conv -> C2f -> Conv -> C2f -> SPPF
```

head：

```text
Upsample + Concat + C2f
Upsample + Concat + C2f
Conv + Concat + C2f
Conv + Concat + C2f
Detect(P3, P4, P5)
```

### YOLOv8-pose 官方架构

参数：

```text
nc: 1
kpt_shape: [17, 3]
scales: n, s, m, l, x
```

backbone：

```text
Conv -> Conv -> C2f -> Conv -> C2f -> Conv -> C2f -> Conv -> C2f -> SPPF
```

head：

```text
Upsample + Concat + C2f
Upsample + Concat + C2f
Conv + Concat + C2f
Conv + Concat + C2f
Pose(P3, P4, P5)
```

### YOLO11-pose 官方架构

参数：

```text
nc: 80
kpt_shape: [17, 3]
scales: n, s, m, l, x
```

backbone：

```text
Conv -> Conv -> C3k2 -> Conv -> C3k2 -> Conv -> C3k2 -> Conv -> C3k2 -> SPPF -> C2PSA
```

head：

```text
Upsample + Concat + C3k2
Upsample + Concat + C3k2
Conv + Concat + C3k2
Conv + Concat + C3k2
Pose(P3, P4, P5)
```

### backbone 结论

- YOLOv8 Detect 与 YOLOv8-pose 的 backbone 相同，均为 `Conv/C2f/SPPF` 结构，并输出 P3、P4、P5 多尺度特征。
- YOLOv8 Detect 与 YOLOv8-pose 的 neck/head 主体路径也基本相同，区别在最终任务头：Detect 使用 `Detect(P3, P4, P5)`，pose 使用 `Pose(P3, P4, P5)`。
- YOLOv8 Detect 默认 `nc=80`，YOLOv8-pose 默认 `nc=1` 且额外定义 `kpt_shape=[17, 3]`。
- YOLO11-pose 与 YOLOv8-pose 同属 Ultralytics pose 模型，但 backbone 不相同：YOLO11-pose 使用 `C3k2` 和 `C2PSA`，而 YOLOv8-pose 使用 `C2f` 和 `SPPF`。
- 因此，若只比较 YOLOv8 Detect 和 YOLOv8-pose，可以认为二者共用同类 backbone；区别主要在任务头和任务输出。如果跨到 YOLO11-pose，则 backbone 已经发生变化，不能视为与 YOLOv8-pose 相同。