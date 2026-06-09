# AGENT.md

> 本文件供 AI 编程助手（Cursor / Windsurf / Copilot Chat / opencode 等）自动读取，以便理解项目背景与规则，给出更精准的建议。

## 项目定位

**AI4PumpRoom** — AI 驱动的泵房智能管理系统。基于 Python 构建，目标是实现泵房设备的智能监控、故障诊断与运维决策。

## 目录结构

```
AI4PumpRoom/
├── src/ai4pumproom/    # 项目主代码包，所有 Python 模块位于此目录下
├── tests/              # 单元测试，使用 pytest，文件以 test_ 开头
├── docs/               # 项目文档（Markdown），含 index.md 索引
├── tasks/              # 任务管理
│   ├── backlog.md      # 未开始的任务，无序列表
│   ├── in-progress.md  # 进行中的任务，- [ ] 复选框
│   └── done.md          # 已完成任务，- [x]，最新在上
├── scripts/            # 构建、部署等辅助脚本
├── pyproject.toml      # 项目配置（依赖、ruff、pytest）
├── uv.lock             # 依赖锁定文件
└── README.md           # 仓库整体说明
```

- **所有新建 Python 代码**必须放在 `src/ai4pumproom/` 内。
- **测试代码**必须放在 `tests/` 内，文件名以 `test_` 开头。
- **文档**统一放在 `docs/` 下，使用 GitHub Flavored Markdown，更新后同步 `docs/index.md` 索引。
- **任务记录**仅操作 `tasks/` 下的三个文件，不要写入 `docs/`。

## 环境与依赖

- 使用 **uv** 管理 Python 环境与依赖，Python >= 3.10。
- 安装依赖：`uv sync`
- 运行命令：`uv run <command>`
- 添加依赖：`uv add <package>`，开发依赖：`uv add --dev <package>`
- 新增第三方库后，需确认 `pyproject.toml` 和 `uv.lock` 同步更新。

## 编码规范

- 遵循 **PEP 8**，使用 **type hints**。
- 使用 **ruff** 进行代码检查与格式化，配置见 `pyproject.toml`。
  - `uv run ruff check .` — 检查
  - `uv run ruff check --fix .` — 自动修复
  - `uv run ruff format .` — 格式化
- 行宽上限 120，目标 Python 3.10+。
- 所有模块和公开函数须添加简要 **docstring**（Google 风格）。
- **不添加多余注释**，代码应自解释。
- 测试使用 **pytest**：`uv run pytest`。

## 任务管理

- **新建任务**：添加到 `tasks/backlog.md`，分配 ID 格式 `TASK-YYYYMMDD-序号`。
- **开始任务**：从 `backlog.md` 移到 `in-progress.md`，转为 `- [ ]` 并记录开始时间。
- **完成任务**：勾选为 `- [x]`，写入完成日期，移到 `done.md` 顶部。
- 任务状态变更时，建议单独 Git 提交以保持记录清晰。

## 文档约定

- 文档统一使用 Markdown，放在 `docs/` 下。
- 新建或修改文档后，更新 `docs/index.md` 的索引表。
- API 文档 docstring 遵循 Google 风格：

```python
def example(param: str) -> bool:
    """简述功能。

    Args:
        param: 参数说明。

    Returns:
        返回值说明。
    """
```

## Git 提交规范

使用 **Conventional Commits**：

| 前缀 | 用途 |
|-------|------|
| `feat:` | 新功能 |
| `fix:` | 修复缺陷 |
| `docs:` | 文档变更 |
| `refactor:` | 重构（不加功能、不修缺陷） |
| `test:` | 测试相关 |
| `chore:` | 构建/工具/依赖变更 |
| `task:` | 任务状态变更 |

示例：
- `feat: add pump status monitor`
- `docs: update architecture overview`
- `task: move TASK-20260609-001 to in-progress`

## AI 协作规则

1. **始终以仓库结构为前提**，引用具体文件路径。
2. **修改代码后主动运行** `uv run ruff check .` 和 `uv run pytest` 确保无报错。
3. **新增第三方库时**，提醒更新 `pyproject.toml` 并运行 `uv sync`。
4. **不要自行提交或推送**，除非用户明确要求。
5. **任务和文档分开**，任务仅操作 `tasks/`，文档仅操作 `docs/`。
6. **指令不明确时**，先基于结构假设合理位置并请求确认。
7. **回复简洁**，避免不必要的解释，优先给出可直接使用的代码或内容。
