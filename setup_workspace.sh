#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/.venv}"
REPO_URL="${REPO_URL:-https://github.com/xiaosi0517/nano-vllm.git}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/nano-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$WORKSPACE/models}"
MODEL_LINK="${MODEL_LINK:-/root/huggingface/Qwen3-0.6B}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "error: root privileges are required to install system packages." >&2
    exit 1
  fi
}

echo "== Create workspace =="
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

echo "== Check Python and venv support =="
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN is not installed or is not on PATH." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "Installing python${PYTHON_VERSION}-venv..."
    run_as_root apt-get update
    run_as_root apt-get install -y "python${PYTHON_VERSION}-venv"
  else
    echo "error: ensurepip is unavailable and this system does not provide apt-get." >&2
    echo "Install the venv package for $PYTHON_BIN, then rerun this script." >&2
    exit 1
  fi
fi

echo "== Create Python 3.10 venv =="
if [ ! -x "$VENV_DIR/bin/python" ]; then
  if [ -e "$VENV_DIR" ]; then
    echo "Removing incomplete virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtual environment at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "== Upgrade build tools =="
pip install --upgrade pip setuptools wheel packaging ninja

echo "== Install PyTorch =="
pip install torch==2.5.1 torchvision torchaudio

echo "== Install FlashAttention =="
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "== Clone nano-vllm =="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "== Install nano-vllm =="
pip install -e .

echo "== Install ModelScope for model download =="
pip install modelscope

echo "== Download Qwen3-0.6B from ModelScope =="
MODEL_DIR="$(
python - <<PY
from modelscope import snapshot_download

model_dir = snapshot_download(
    "$MODEL_ID",
    cache_dir="$MODEL_CACHE_DIR",
)

print(model_dir)
PY
)"

echo "Downloaded model to: $MODEL_DIR"

echo "== Create easy symlink =="
mkdir -p "$(dirname "$MODEL_LINK")"
if [ ! -e "$MODEL_LINK" ]; then
  ln -s "$MODEL_DIR" "$MODEL_LINK"
fi

echo "== Verify environment =="
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)

import flash_attn

print("flash-attn ok")
PY

echo "== Verify nano-vllm import =="
python - <<'PY'
from nanovllm import LLM, SamplingParams

print("nano-vllm import ok")
PY

echo ""
echo "Setup finished."
echo "Activate env later with:"
echo "source $VENV_DIR/bin/activate"
echo ""
echo "Model path:"
echo "$MODEL_LINK"
echo ""
echo "Example benchmark:"
echo "cd $REPO_DIR"
echo "python benchmark_quant_profile.py --model-path $MODEL_LINK"
