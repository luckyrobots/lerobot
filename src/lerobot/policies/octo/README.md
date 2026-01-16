# Octo finetuning in LeRobot (what we do + why)

This folder contains a thin **LeRobot → OctoPt** adapter:
- `configuration_octo.py`: config + dataset sampling (history + action chunk)
- `modeling_octo.py`: observation/action shaping, masks, normalization, finetune wrapper

The goal is to finetune Octo **the recommended way**: feed Octo a **temporal window** of observations and supervise a **future action chunk**, while using the dataset’s stats to normalize/unnormalize actions.

---

## Octo architecture (high level)

Octo is a **tokenize → transformer → action head** policy with optional multi-modal inputs.

```
                 +------------------------------+
 images/state -->| Observation tokenizers       |--> obs_tokens
 goals/lang  --->| Task tokenizers (optional)   |--> task_tokens
                 +------------------------------+
                               |
                               v
                 +------------------------------+
                 | Blockwise-causal transformer |
                 | (attend to <= current time)  |
                 +------------------------------+
                               |
                               v
                 +------------------------------+
                 | Action head                  |
                 | (often diffusion)            |
                 +------------------------------+
                               |
                               v
                    action_chunk (H steps)
```

Key concepts we must respect:
- **Windowing**: Octo consumes `T` past timesteps at once (history window).
- **Masking**: `timestep_pad_mask` and `pad_mask_dict` tell Octo which timesteps/modalities are real vs padding.
- **Action chunking**: Octo predicts a horizon of `H` future actions; training supervises that chunk.
- **Normalization**: Octo’s `sample_actions()` produces *normalized* actions unless you pass unnormalization stats.

---

## What we finetune (our contract)

### Inputs we feed Octo (shapes)

We build an `observations` dict like OctoPt expects:

```
observations = {
  "image_primary":    uint8  (B, T, C, H, W),
  "image_wrist":      uint8  (B, T, C, H, W),
  "timestep_pad_mask":bool   (B, T),
  "pad_mask_dict": {
      "image_primary":bool   (B, T),
      "image_wrist":  bool   (B, T),
      "timestep":     bool   (B, T),
  },
  "timestep":         int64  (B, T),   # from dataset if available
  "task_completed":   bool   (B, T, *) # broadcast to match checkpoint example_batch
}
```

Where:
- `B` = batch size
- `T` = `policy.window_size` (history window length)
- images are resized to Octo’s expected spatial sizes
- masks come from LeRobotDataset’s `*_is_pad` keys when present

### Targets we supervise (shapes)

LeRobot provides `batch["action"]` as the **future chunk** using delta-indices:

```
batch["action"]          float (B, H, A)   # H=action_horizon, A=action_dim
batch["action_is_pad"]   bool  (B, H)      # optional
```

We convert that into Octo’s training label format (supervise only the **last timestep** in the window):

```
gt_actions      float (B, T, H, A)   # only gt_actions[:, -1] is filled
action_pad_mask bool  (B, T, H, A)   # only action_pad_mask[:, -1] is True where valid
```

---

## Why each step exists (reasoning)

### 1) Windowing (history) via `delta_indices`
**What**: `OctoConfig.observation_delta_indices = [-(T-1), ..., 0]`

**Why**: Octo is trained to reason over short histories. Without history, you lose temporal context and training drifts from the intended API.

### 2) Action chunk supervision (future horizon)
**What**: `OctoConfig.action_delta_indices = [0, 1, ..., H-1]`

**Why**: Octo predicts an action sequence (chunk). Supervising only a single action or repeating a single target across `H` breaks learning.

### 3) Correct masking (`timestep_pad_mask`, `pad_mask_dict`)
**What**: build masks from LeRobotDataset padding keys `f"{key}_is_pad"` and AND them for `timestep_pad_mask`.

**Why**: near episode boundaries the history window includes padded frames; masking prevents the transformer from learning artifacts from padding.

### 4) Image format and dtype
**What**: always feed Octo **uint8** images in **(B,T,C,H,W)**.

**Why**: Octo’s vision encoders normalize images internally (e.g. uint8 → [-1,1]). Wrong layout/dtype silently corrupts inputs.

### 5) Action normalization for training
**What**: normalize GT actions with dataset stats:

```
norm = (action - mean) / std   (masked dims left unchanged)
```

**Why**: Octo’s head/loss assumes actions are in its normalized space. Training on raw units can destabilize optimization.

### 6) Action unnormalization for inference
**What**: pass `unnormalization_statistics` to `sample_actions()` so it returns actions in your embodiment’s units.

**Why**: normalized actions are not directly executable on the robot; you need real units/scales.

### 7) Finetune modes (freezing)
**What**: `finetune_mode ∈ {head_only, head_mlp_only, full}` controls which Octo weights are frozen.

**Why**:
- `head_only`: adapt output mapping cheaply while keeping the backbone stable
- `full`: maximum adaptation, highest compute and overfitting risk

---

## Minimal training command (example)

```
PYTHONPATH=lerobot/src python -m lerobot.scripts.train \
  --dataset.repo_id=<your_dataset_id> \
  --dataset.root=<your_dataset_root> \
  --policy.type=octo \
  --policy.base_checkpoint=hf://rail-berkeley/octo-base-1.5 \
  --policy.finetune_mode=head_only \
  --policy.window_size=2 \
  --policy.action_horizon=4 \
  --policy.primary_image_key=observation.images.CameraLeft \
  --policy.wrist_image_key=observation.images.CameraSide \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false \
  --num_workers=0
```

Notes:
- Camera keys must exist in your dataset features (config validates this).
- If you change `window_size` or `action_horizon`, the dataset sampling changes automatically via delta-indices.


