# AI4PumpRoom

AI4PumpRoom 是泵房多任务智能监控系统，当前核心实现迁移自 Vigil，主代码包位于 `src/Vigil`。系统面向泵房场景，同时覆盖人体检测、火焰检测、积水检测、人体 17 关键点、安全帽属性、吸烟属性、跌倒检测和挥手检测。

## 项目结构

```text
AI4PumpRoom/
├── src/Vigil/          # 核心 Python 包
│   ├── models/         # 模型、注册表、检测头和损失
│   ├── train/          # 训练入口、Trainer、数据集
│   ├── inference/      # 推理引擎
│   ├── postprocess/    # 时序后处理
│   ├── pipeline/       # 视频流管道
│   └── main.py         # 命令行入口
├── config/             # 训练和推理配置
├── data/               # 处理后的数据集
├── checkpoints/        # 本地权重目录，已被 gitignore 忽略
├── scripts/            # 数据准备与标注辅助脚本
├── docs/               # 项目文档
└── tasks/              # 任务记录
```


## 运行

推理：

```powershell
uv run vigil live --video test.mp4 --model vigil_v2 --show
```

训练：

```powershell
uv run vigil-train --stage coco_pose_pretrain
uv run vigil-train --stage pretrain
uv run vigil-train --stage finetune
```

也可以使用模块方式：

```powershell
uv run python -m Vigil.main --help
uv run python -m Vigil.train.train --stage pretrain
```

## 权重位置

训练权重保存在：

```text
checkpoints/vigil_v2/
```

当前本地已放置源项目训练好的权重：

```text
coco_pose_pretrain_best.pt
coco_pose_pretrain_last.pt
pretrain_best.pt
pretrain_last.pt
```

推理自动读取顺序：

```text
finetune_best.pt
pretrain_best.pt
finetune_last.pt
pretrain_last.pt
best.pt
```
