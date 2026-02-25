#!/usr/bin/env python
"""Train ACT policy on 200-episode Piper pick-and-place dataset.

Usage (from WSL):
    source /home/zero/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
    python /mnt/d/FInal_Setup/train_act.py

Dataset: 200 episodes, 72,607 frames, 2 cameras (CameraGripper + CameraLeft)
Train/val split: 80/20 by episode (160 train, 40 val).

ACT typically converges in 100k-200k steps with batch_size=8.
Epoch math (train split: 58,087 frames):
    batch_size=8  ->  steps_per_epoch = ceil(58087/8) = 7261
    100k steps    ->  ~14 passes through data
"""

import sys
import os

# Ensure our lerobot is on the path
lerobot_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot", "src")
if lerobot_src not in sys.path:
    sys.path.insert(0, lerobot_src)

# Import policies FIRST to register all config subclasses with draccus
import lerobot.policies  # noqa: F401

# Fast frame cache (same as IMLE)
_CACHE_META = "/home/zero/imle_training/frames_cache/meta.json"
if os.path.exists(_CACHE_META):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fast_patch import install_fast_patch
    install_fast_patch()
else:
    print("[train_act] WARNING: frames cache not found — run preprocess_frames.py first for 10x speedup")

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs.configs import PiperRoomEnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

# ---------- Train/val split ----------
TRAIN_EPISODES = list(range(160))       # episodes 0-159 (80%)
VAL_EPISODES   = list(range(160, 200))  # episodes 160-199 (20%)

# ---------- Training constants ----------
BATCH_SIZE = 8        # ACT standard (small batch, low LR)
STEPS      = 100_000  # typically sufficient for ACT
LOG_FREQ   = 100
SAVE_FREQ  = 5_000    # ~20 checkpoints

# ---------- Dataset ----------
DATASET_ROOT = "/home/zero/imle_training/dataset"

dataset_cfg = DatasetConfig(
    repo_id="session_2026-02-24_18-55-44",
    root=DATASET_ROOT,
    episodes=TRAIN_EPISODES,
    video_backend="pyav",
    image_transforms=ImageTransformsConfig(enable=True),
)

# ---------- ACT Policy ----------
policy_cfg = ACTConfig(
    push_to_hub=False,

    # Action chunking
    chunk_size=100,          # predict 100 steps ahead (~3.3s at 30Hz)
    n_action_steps=100,      # execute full chunk before re-planning

    # Vision backbone
    vision_backbone="resnet18",
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",

    # Transformer
    dim_model=512,
    n_heads=8,
    dim_feedforward=3200,
    n_encoder_layers=4,
    n_decoder_layers=1,

    # VAE (standard ACT)
    use_vae=True,
    latent_dim=32,
    n_vae_encoder_layers=4,
    kl_weight=10.0,

    # Optimizer (ACT defaults)
    optimizer_lr=1e-5,
    optimizer_weight_decay=1e-4,
    optimizer_lr_backbone=1e-5,

    dropout=0.1,
)

# ---------- W&B ----------
wandb_cfg = WandBConfig(
    enable=True,
    project="act-piper",
    disable_artifact=True,
)

# ---------- Env (for mid-training eval, set eval_freq>0 when LuckyEngine available) ----------
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

# ---------- Pipeline ----------
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
)

if __name__ == "__main__":
    train(cfg)
