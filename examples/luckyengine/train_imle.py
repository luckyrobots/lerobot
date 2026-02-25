#!/usr/bin/env python
"""Train IMLE policy on local dataset (v2 — larger dataset, batch=512, val split).

Usage (from WSL):
    source /home/zero/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
    python /mnt/d/FInal_Setup/train_imle.py

Train/val split: 80/20 by episode (160 train, 40 val).
Epoch math (train split: 58,087 frames):
    batch_size=512  →  steps_per_epoch = ceil(58087/512) = 114
    1000 epochs     →  114,000 steps
    log  every  1 epoch  =  114 steps
    save every 10 epochs =  1,140 steps  (~88 checkpoints total)

LR scaling: sqrt rule for batch_size=512 vs reference=64.
    lr = 1e-4 * sqrt(512/64) ≈ 2.83e-4

Pre-decode frames first (run once):
    python /mnt/d/FInal_Setup/preprocess_frames.py
"""

import sys
import os
import math

# Ensure our lerobot is on the path
lerobot_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot", "src")
if lerobot_src not in sys.path:
    sys.path.insert(0, lerobot_src)

# Import policies FIRST to register all config subclasses with draccus
import lerobot.policies  # noqa: F401

# ── Fast frame cache (pre-decoded 96×96 numpy mmap) ──────────────
# Falls back to normal video decode if cache doesn't exist.
_CACHE_META = "/home/zero/imle_training/frames_cache/meta.json"
if os.path.exists(_CACHE_META):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fast_patch import install_fast_patch
    install_fast_patch()
else:
    print("[train_imle] WARNING: frames cache not found — run preprocess_frames.py first for 10× speedup")

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs.configs import PiperRoomEnvConfig
from lerobot.policies.imle.configuration_imle import IMLEConfig
from lerobot.scripts.lerobot_train import train

# ---------- Train/val split ----------
NUM_EPISODES    = 200
TRAIN_EPISODES  = list(range(160))      # episodes 0-159 (80%)
VAL_EPISODES    = list(range(160, 200)) # episodes 160-199 (20%)

# ---------- Epoch-aligned constants (train split only) ----------
TRAIN_FRAMES    = 58_087                # exact frame count for 160 episodes
BATCH_SIZE      = 512
STEPS_PER_EPOCH = -(-TRAIN_FRAMES // BATCH_SIZE)  # ceil div = 114

STEPS     = 1000 * STEPS_PER_EPOCH           # 1000 epochs = 114,000 steps
LOG_FREQ  = 1 * STEPS_PER_EPOCH              # log  every 1 epoch  = 114 steps
SAVE_FREQ = 10 * STEPS_PER_EPOCH             # save every 10 epochs = 1,140 steps

# ---------- LR scaling (sqrt rule) ----------
BASE_LR         = 1e-4
REF_BATCH_SIZE  = 64
SCALED_LR       = BASE_LR * math.sqrt(BATCH_SIZE / REF_BATCH_SIZE)  # ≈ 2.83e-4
WARMUP_STEPS    = 1000  # slightly longer warmup for larger batch

# ---------- Configuration ----------

DATASET_ROOT = "/home/zero/imle_training/dataset"

dataset_cfg = DatasetConfig(
    repo_id="session_2026-02-24_18-55-44",
    root=DATASET_ROOT,
    episodes=TRAIN_EPISODES,
    video_backend="pyav",
    image_transforms=ImageTransformsConfig(enable=True),
)

policy_cfg = IMLEConfig(
    push_to_hub=False,
    use_amp=True,
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    use_group_norm=False,   # must be False with pretrained weights
    optimizer_lr=SCALED_LR,
    scheduler_warmup_steps=WARMUP_STEPS,
)

wandb_cfg = WandBConfig(
    enable=True,
    project="imle-piper",
    disable_artifact=True,
)

# Mid-training eval via LuckyEngine gRPC.
# Requires LuckyEngine running with Piper-room in Play mode.
# Set eval_freq=0 to disable eval when LuckyEngine is not available.
env_cfg = PiperRoomEnvConfig(
    host="192.168.240.1",
    port=50055,
    fps=30,
    episode_length=300,
    success_dist_m=0.05,
)

eval_cfg = EvalConfig(
    n_episodes=5,
    batch_size=1,
    use_async_envs=False,
)

# ---------- Resume from checkpoint (set to None for fresh start) ----------
RESUME_CHECKPOINT = "outputs/train/2026-02-24/23-55-59_piper_room_imle/checkpoints/005700"

from pathlib import Path

if RESUME_CHECKPOINT is not None:
    _ckpt = Path(RESUME_CHECKPOINT)
    policy_cfg.pretrained_path = _ckpt / "pretrained_model"
    _wandb_run_id = "nol4y6nt"  # resume same W&B run
else:
    _wandb_run_id = None

wandb_cfg.run_id = _wandb_run_id

cfg = TrainPipelineConfig(
    dataset=dataset_cfg,
    policy=policy_cfg,
    wandb=wandb_cfg,
    env=env_cfg,
    eval=eval_cfg,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    eval_freq=0,            # set to SAVE_FREQ when LuckyEngine is running
    save_freq=SAVE_FREQ,
    log_freq=LOG_FREQ,
    num_workers=8,
    seed=1000,
    save_checkpoint=True,
    tolerance_s=0.1,
    resume=False,           # we handle resume manually below
)

# Manual resume: set checkpoint_path and output_dir after construction
# (bypass validate()'s CLI arg parsing for resume)
if RESUME_CHECKPOINT is not None:
    cfg.resume = True
    cfg.checkpoint_path = Path(RESUME_CHECKPOINT)
    cfg.output_dir = Path(RESUME_CHECKPOINT).parent.parent

if __name__ == "__main__":
    train(cfg)
