#!/bin/bash
# AutoDL environment setup script for multi-head verification training.
#
# Usage on AutoDL:
#   bash test_model/scripts/setup_autodl.sh
#
# What this does:
#   1. Installs uv package manager
#   2. Creates a dedicated conda env or uses uv venv
#   3. Installs project dependencies (PyTorch, OpenCV, etc.)
#   4. Downloads and prepares COCO 2017 20-class dataset
#
# Customize via env vars:
#   DATA_DIR=/root/autodl-tmp/coco2017     # fast SSD for data
#   SAVE_DIR=/root/autodl-fs/checkpoints   # persistent storage for checkpoints

set -e

echo "========================================"
echo "AutoDL Setup for YOLOv8 Multi-Head Test"
echo "========================================"

# ---- Paths ----
# autodl-tmp: fast local SSD (data here for speed)
# autodl-fs:  persistent network disk (checkpoints here for safety)
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/coco2017}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-fs/checkpoints}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Project: $PROJECT_DIR"
echo "Data:    $DATA_DIR"
echo "Save:    $SAVE_DIR"

# ---- Step 1: Install uv ----
if ! command -v uv &> /dev/null; then
    echo ""
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/4] uv already installed: $(uv --version)"
fi

# ---- Step 2: Install dependencies ----
echo ""
echo "[2/4] Installing dependencies..."
cd "$PROJECT_DIR"

# Pin PyTorch version to match CUDA on AutoDL (usually CUDA 12.1 or 12.4)
# AutoDL standard images: CUDA 12.1 with PyTorch 2.3.x
# Use an updated torch constraint for cloud GPUs
uv sync --group dev 2>/dev/null || uv pip install -e ".[dev]"

# Verify
uv run python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ---- Step 3: Download & prepare COCO ----
echo ""
echo "[3/4] Preparing COCO 2017 20-class dataset..."
uv run python test_model/data/prepare_coco20.py \
    --data-dir "$DATA_DIR" \
    --download

# ---- Step 4: Verify dataset ----
echo ""
echo "[4/4] Verifying dataset..."
uv run python test_model/data/prepare_coco20.py \
    --data-dir "$DATA_DIR" \
    --verify-only

mkdir -p "$SAVE_DIR"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  # Edit config"
echo "  vi test_model/config.yaml"
echo ""
echo "  # Single model"
echo "  uv run python test_model/scripts/run_train.py --config test_model/config.yaml"
echo ""
echo "  # Smoke test (3 epochs)"
echo "  uv run python test_model/scripts/run_train.py --config test_model/config.yaml --debug"
echo ""
echo "  # Five models on 5 GPUs (separate terminals)"
echo "  CUDA_VISIBLE_DEVICES=0 uv run python test_model/scripts/run_train.py \\"
echo "    --config test_model/config.yaml --model dual_head &"
echo "========================================"
