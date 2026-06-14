# AGENT.md

> 本文件供 AI 编程助手自动读取，用于理解 AI4PumpRoom 的项目背景、目录约定和协作规则。

## 项目定位

AI4PumpRoom 是泵房多任务智能监控系统。当前核心实现来自 Vigil，单模型同时覆盖人体检测、火焰检测、积水检测、人体 17 关键点、安全帽属性、吸烟属性、跌倒检测和挥手检测。

## 目录结构

```text
AI4PumpRoom/
├── src/Vigil/          # 项目主 Python 包，所有核心业务代码放在这里
│   ├── models/         # 模型、注册表、检测头、损失函数
│   ├── train/          # 训练入口、Trainer、数据集
│   ├── inference/      # 推理引擎
│   ├── postprocess/    # 时序后处理
│   ├── pipeline/       # 视频流管道
│   └── main.py         # 命令行入口
├── config/             # 训练和推理配置
├── data/               # 已处理数据集
├── scripts/            # 数据准备和标注辅助脚本
├── tests/              # pytest 测试
├── docs/               # Markdown 文档
├── tasks/              # 任务管理记录
├── pyproject.toml      # 项目配置、依赖、命令入口
├── uv.lock             # uv 锁文件
└── README.md           # 项目说明
```

## 代码约定

- 新增核心 Python 代码放在 `src/Vigil/` 下。
- 测试代码放在 `tests/` 下，文件名以 `test_` 开头。
- 文档放在 `docs/` 下，修改文档后同步 `docs/index.md`。
- 任务记录只操作 `tasks/` 下的文件。
- 运行产物、权重、日志、缓存不提交：`checkpoints/`、`logs/`、`runs/`、`outputs/`、`__pycache__/`、`.ultralytics/`。

## 环境与命令

项目使用 uv 管理 Python 环境和依赖：

```powershell
uv sync
uv run vigil --help
uv run vigil-train --help
uv run pytest
uv run ruff check .
```

模块方式也可用：

```powershell
uv run python -m Vigil.main --help
uv run python -m Vigil.train.train --stage pretrain
```

GStreamer/RTSP 依赖为可选项，需要时执行：

```powershell
uv sync --extra gstreamer
```

## 权重约定

仓库不提交 `checkpoints/`。推理必须提供本地权重，或把权重放到约定路径：

```text
checkpoints/{model_name}/pretrain_best.pt
checkpoints/{model_name}/finetune_best.pt
checkpoints/{model_name}/pretrain_last.pt
checkpoints/{model_name}/best.pt
```

若未找到权重，推理入口应明确报错，不应静默随机初始化。

## Git 提交

使用 Conventional Commits，例如：

- `feat: migrate Vigil monitoring system`
- `fix: require explicit inference weights`
- `test: add migration smoke tests`