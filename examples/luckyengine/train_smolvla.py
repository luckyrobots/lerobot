#!/usr/bin/env python
"""Train SmolVLA policy on local Piper pick-and-place dataset.

Usage (from RunPod H100 SXM):
    pip install -e ".[smolvla]"
    python train_smolvla.py

Train/val split: 80/20 by episode (160 train, 40 val).
Epoch math (train split: 58,087 frames):
    batch_size=128  →  steps_per_epoch = ceil(58087/128) = 454
    ~440 epochs     →  200,000 steps
    log  every  1 epoch  ≈  454 steps
    save every ~11 epochs =  5000 steps  (~40 checkpoints total)

Target GPU: H100 SXM (80GB VRAM, ~$2.69/hr RunPod community cloud)
    - ~990 TFLOPS bf16 tensor — 3x RTX 4090, fastest single-GPU option
    - SmolVLA peak VRAM ~7 GiB at batch 128 (trainable expert is only ~28.5M params)
    - Estimated wall time: 4-7 hrs for 200k steps

SmolVLA fine-tuning strategy:
    - Freeze vision encoder (SigLIP) and VLM backbone
    - Train only action expert + state projection (~28.5M params)
    - Batch 128: saturates H100 compute while preserving gradient noise for generalization
    - No image augmentation — frozen SigLIP expects clean images
"""

import sys
import os

# Ensure our lerobot is on the path
lerobot_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerobot", "src")
if lerobot_src not in sys.path:
    sys.path.insert(0, lerobot_src)

# Import policies FIRST to register all config subclasses with draccus
import lerobot.policies  # noqa: F401

from pathlib import Path

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

# ---------- Train/val split ----------
NUM_EPISODES    = 200
TRAIN_EPISODES  = list(range(160))      # episodes 0-159 (80%)
VAL_EPISODES    = list(range(160, 200)) # episodes 160-199 (20%)

# ---------- Epoch-aligned constants (train split only) ----------
TRAIN_FRAMES    = 58_087                # exact frame count for 160 episodes
BATCH_SIZE      = 128
STEPS_PER_EPOCH = -(-TRAIN_FRAMES // BATCH_SIZE)  # ceil div = 454

STEPS     = 200_000
LOG_FREQ  = STEPS_PER_EPOCH             # log every epoch ≈ 454 steps
SAVE_FREQ = 5000                        # save every ~5000 steps

# ---------- Configuration ----------

DATASET_ROOT = os.environ.get("SMOLVLA_DATASET_ROOT", "/home/zero/imle_training/dataset")

dataset_cfg = DatasetConfig(
    repo_id="session_2026-02-24_18-55-44",
    root=DATASET_ROOT,
    episodes=TRAIN_EPISODES,
    video_backend="torchcodec",
    # No image transforms — SmolVLA's frozen SigLIP handles its own preprocessing
)

policy_cfg = SmolVLAConfig(
    push_to_hub=False,
    # Fine-tuning: freeze vision encoder + VLM, train only action expert
    freeze_vision_encoder=True,
    train_expert_only=True,
    train_state_proj=True,
    # Action chunking
    chunk_size=50,
    n_action_steps=50,
    # Optimizer (sqrt-scaled for batch_size=128: 1e-4 * sqrt(128/32) = 2e-4)
    optimizer_lr=2e-4,
    scheduler_warmup_steps=1000,
    scheduler_decay_steps=60_000,
    # VLM backbone
    vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    load_vlm_weights=False,  # False: VLM weights come from pretrained SmolVLA checkpoint
)

# Point to pretrained SmolVLA base weights (HuggingFace Hub ID)
policy_cfg.pretrained_path = Path("lerobot/smolvla_base")

wandb_cfg = WandBConfig(
    enable=True,
    project="smolvla-piper",
    disable_artifact=True,
)

eval_cfg = EvalConfig(
    n_episodes=0,
    batch_size=0,
    use_async_envs=False,
)

# ---------- Resume from checkpoint (set to None for fresh start) ----------
RESUME_CHECKPOINT = None

if RESUME_CHECKPOINT is not None:
    _ckpt = Path(RESUME_CHECKPOINT)
    policy_cfg.pretrained_path = _ckpt / "pretrained_model"
    _wandb_run_id = None  # set to W&B run ID to resume same run
else:
    _wandb_run_id = None

wandb_cfg.run_id = _wandb_run_id

cfg = TrainPipelineConfig(
    dataset=dataset_cfg,
    policy=policy_cfg,
    wandb=wandb_cfg,
    eval=eval_cfg,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    eval_freq=0,            # no gym eval — eval runs locally via luckyengine_eval_smolvla.py
    save_freq=SAVE_FREQ,
    log_freq=LOG_FREQ,
    num_workers=12,
    seed=1000,
    save_checkpoint=True,
    tolerance_s=10.0,  # some episodes have videos shorter than parquet — use nearest frame
    resume=False,
)

# Manual resume: set checkpoint_path and output_dir after construction
if RESUME_CHECKPOINT is not None:
    cfg.resume = True
    cfg.checkpoint_path = Path(RESUME_CHECKPOINT)
    cfg.output_dir = Path(RESUME_CHECKPOINT).parent.parent

if __name__ == "__main__":
    train(cfg)
