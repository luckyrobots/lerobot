#!/usr/bin/env bash
set -euo pipefail

# Requires: active conda env with Torch 2.9.0+cu128 and torchvision 0.24.0+cu128 installed

# 1) Ensure CUDA 12.8 toolchain is present (includes headers)
conda install -y -c conda-forge \
  cuda-toolkit=12.8 \
  cuda-nvcc=12.8

# 2) Point build to conda CUDA and expose headers/libs
export CUDA_HOME="${CONDA_PREFIX:?Activate your conda env first}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# 3) Build a LeRobot-compatible flash-attn (v2.x) for Torch 2.9.0+cu128 and RTX 5090
pip uninstall -y flash-attn || true
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"    # PTX covers newer Blackwell SMs (5090)
export FLASH_ATTENTION_SKIP_CUDA_ARCH_CHECK=1
pip install --no-build-isolation --force-reinstall \
  "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3"

# 4) Sanity check flash-attn
python - << 'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
from flash_attn.flash_attn_interface import flash_attn_func
print('flash-attn OK')
PY

# 5) Train with local dataset (GPU)
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=so100_shaver_insert \
  --dataset.root=./dataset/so100_shaver_insert \
  --policy.type=diffusion \
  --policy.push_to_hub=False \
  --dataset.video_backend=torchcodec