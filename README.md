# AI4PumpRoom

AI 驱动的泵房智能管理系统。

## 项目结构

```
AI4PumpRoom/
├── src/ai4pumproom/    # 项目主代码包
├── tests/              # 单元测试（pytest）
├── docs/               # 项目文档
├── tasks/              # 任务管理
│   ├── backlog.md      # 待办任务
│   ├── in-progress.md  # 进行中任务
│   └── done.md         # 已完成任务
├── scripts/            # 辅助脚本
└── pyproject.toml      # 项目配置
```

## 快速开始

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境与依赖。

```bash
# 安装依赖（自动创建虚拟环境）
uv sync

# 运行测试
uv run pytest

# 代码检查
uv run ruff check .
```

## 许可证

待定
