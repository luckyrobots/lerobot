#!/usr/bin/env python
"""Dry-run LuckyEngine eval on the latest checkpoint to see what's failing."""

import sys, os

# Force UTF-8 on Windows consoles that default to legacy charmap encodings.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
lerobot_src = os.path.join(BASE, "lerobot", "src")
if lerobot_src not in sys.path:
    sys.path.insert(0, lerobot_src)

# Register policy configs
import lerobot.policies  # noqa: F401

from pathlib import Path
from luckyengine_eval import run_luckyengine_eval


def _find_latest_checkpoint(base: Path) -> Path | None:
    """Return the pretrained_model dir of the highest-numbered checkpoint."""
    train_root = base / "outputs" / "train"
    if not train_root.exists():
        return None
    # Walk all run dirs, collect checkpoint dirs
    best_ckpt = None
    best_num = -1
    for run_dir in sorted(train_root.rglob("checkpoints")):
        try:
            entries = sorted(run_dir.iterdir())
        except OSError:
            continue
        for ckpt_dir in entries:
            try:
                is_dir = ckpt_dir.is_dir()
            except OSError:
                continue
            if not is_dir:
                continue
            try:
                num = int(ckpt_dir.name)
            except ValueError:
                continue
            try:
                pretrained = ckpt_dir / "pretrained_model"
                exists = pretrained.exists()
            except OSError:
                continue
            if exists and num > best_num:
                best_num = num
                best_ckpt = pretrained
    return best_ckpt


CHECKPOINT = _find_latest_checkpoint(Path(BASE))

print(f"Base dir:          {BASE}")
print(f"Checkpoint found:  {CHECKPOINT}")
print(f"Checkpoint exists: {CHECKPOINT.exists() if CHECKPOINT else False}")
if CHECKPOINT and CHECKPOINT.exists():
    print(f"Contents:          {[p.name for p in CHECKPOINT.iterdir()]}")
print()

if CHECKPOINT is None or not CHECKPOINT.exists():
    print("ERROR: No checkpoint found. Is training running and has it saved at least one checkpoint?")
    sys.exit(1)

print(f"Running LuckyEngine eval (1 episode, max 200 steps) on {CHECKPOINT.parent.name}...")
result = run_luckyengine_eval(
    checkpoint_dir=CHECKPOINT,
    host="192.168.240.1",
    port=50055,
    n_episodes=1,
    max_steps=200,
)

print()
print("=== RESULT ===")
for k, v in result.items():
    print(f"  {k}: {v}")
