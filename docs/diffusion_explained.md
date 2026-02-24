# Diffusion Policy — Theory, Training, and Data-loading (Detailed)

This document explains how diffusion models work at a high level, how the `DiffusionPolicy` in this repository is implemented and trained, and how the dataset / dataloader (the `LeRobotDataset`) is structured and used. It is written to be actionable for someone who wants to understand or modify training in `lerobot`.

---

## 1) Diffusion models — core idea and math intuition

- Goal: learn a generative model p(x0) by defining a noising (forward) process q(x_t | x_{t-1}) that gradually adds Gaussian noise to data x0 until it becomes near-isotropic Gaussian, and learning a reverse process p_theta(x_{t-1} | x_t) that denoises.

- Forward (fixed) diffusion (discrete-time DDPM notation):
  - q(x_t | x_{t-1}) = N(x_t ; sqrt(1 - beta_t) x_{t-1}, beta_t I)
  - You can sample x_t at arbitrary timestep t directly from x_0 via closed form:
    - x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, where eps ~ N(0, I)
    - alpha_bar_t = Π_{s=1..t} (1 - beta_s)

- Reverse (learned) process:
  - p_theta(x_{t-1} | x_t) is parameterized (commonly Gaussian) where the network predicts either:
    - epsilon (the noise) — i.e., predict eps_theta(x_t, t) and reconstruct x_{t-1}, or
    - x_0 (the original sample), sometimes called "sample" prediction.
  - Many implementations predict epsilon; training loss reduces to MSE between predicted and true epsilon.

- Training objective (standard DDPM):
  - Sample x_0 from dataset, sample timestep t uniformly in [1..T], sample eps ~ N(0, I).
  - Compute x_t = q(x_t | x_0, eps) (closed form above).
  - Let pred = model(x_t, t, cond) and target = eps (if using epsilon prediction).
  - Loss = E_{x0, t, eps}[ || pred - target ||^2 ] (MSE).

- Sampling:
  - Start from x_T ~ N(0, I) and iteratively apply learned reverse updates from T -> 0.
  - Different schedulers (DDPM vs DDIM) implement different deterministic/stochastic update formulas to step backward.

- Implementation details important for policies:
  - Conditioning: model must condition on observations (images, states). Conditioning is often concatenated to timestep embeddings or injected via FiLM.
  - Temporal dimension: here the model operates on a 1D sequence of actions (trajectory) and denoises the entire trajectory conditioned on recent observations.

## 2) Where diffusion is implemented in this repo

- Core implementation files:
  - `src/lerobot/policies/diffusion/modeling_diffusion.py` — main `DiffusionPolicy`, `DiffusionModel`, noise scheduler creation, UNet, encoders.
  - `src/lerobot/policies/diffusion/configuration_diffusion.py` — `DiffusionConfig` (all hyperparameters and feature shapes).
  - `src/lerobot/policies/diffusion/processor_diffusion.py` — pre/post processors for normalization and transforms.

- High-level responsibilities:
  - `DiffusionPolicy` is the policy wrapper; it exposes `forward`, `select_action`, and `predict_action_chunk`.
  - `DiffusionModel` contains:
    - image encoders (`DiffusionRgbEncoder`), state concatenation to form `global_cond`
    - `DiffusionConditionalUnet1d` — a 1D UNet over the action trajectory (FiLM conditioning using timestep embedding + global_cond)
    - a noise scheduler (DDPM or DDIM) from `diffusers`
    - `compute_loss(batch)` and `conditional_sample()` for sampling

## 3) DiffusionPolicy training & loss (how the repo computes the loss)

- Expected input fields for compute_loss:
  - `observation.state`: (B, n_obs_steps, state_dim)
  - `observation.images`: (B, n_obs_steps, num_cameras, C, H, W) OR `observation.environment_state`
  - `action`: (B, horizon, action_dim)
  - `action_is_pad`: (B, horizon) boolean mask for padded actions

- Steps in `DiffusionModel.compute_loss`:
  1. Validate shapes: horizon == config.horizon and n_obs_steps == config.n_obs_steps.
  2. Build `global_cond` by encoding images (via `DiffusionRgbEncoder`) and concatenating state/env state across the sequence, then flatten to (B, global_cond_dim).
  3. Take `trajectory = batch["action"]` and sample `eps ~ N(0, I)` with same shape.
  4. Sample timesteps per example uniformly: `timesteps = randint(0, num_train_timesteps)`.
  5. Create `noisy_trajectory = noise_scheduler.add_noise(trajectory, eps, timesteps)`.
  6. Run the UNet: `pred = unet(noisy_trajectory, timesteps, global_cond)`.
  7. Choose target depending on `prediction_type`:
     - `epsilon` → target = eps
     - `sample` → target = trajectory
  8. Compute MSE loss and optionally mask padded steps with `action_is_pad`.
  9. Return mean loss.

- Notes:
  - The scheduler used (DDPM vs DDIM) and the `prediction_type` are configuration options in `DiffusionConfig`.
  - Timestep embeddings use a small sinusoidal MLP (`DiffusionSinusoidalPosEmb`) followed by linear layers.
  - Conditioning uses FiLM-like modulation (bias modulation; optional scale modulation) applied inside residual blocks.

## 4) Inference / Sampling (how action chunks are produced)

- `DiffusionModel.conditional_sample`:
  - Starts from prior sample (noise) or provided noise.
  - Sets timesteps on the scheduler (`set_timesteps(num_inference_steps)`).
  - Iteratively runs the UNet and the scheduler `.step(...)` from t = timesteps to 0.
  - Returns a full trajectory of length `horizon`.

- `DiffusionPolicy.generate_actions`:
  - Prepares `global_cond` from the last `n_obs_steps` observations (images + state).
  - Calls `conditional_sample(batch_size, global_cond, noise)`.
  - Extracts the `n_action_steps` chunk to execute starting at the current timestep (start = n_obs_steps - 1).

## 5) The training script and pipeline orchestration

- Entrypoint(s):
  - CLI wrapper: `lerobot/scripts/lerobot_train.py` is the main orchestrator (`train(cfg: TrainPipelineConfig)`).
  - Example/scripted usage in `examples/...` (several example scripts instantiate `DiffusionConfig` and `DiffusionPolicy` and run a small loop).

- Training pipeline responsibilities (`train`):
  1. Parse and validate `TrainPipelineConfig`.
  2. Create an `Accelerator` (from `accelerate`) to support single / multi-GPU training and mixed precision.
  3. Create dataset(s) via `make_dataset(cfg)` (factory uses `LeRobotDataset` or `StreamingLeRobotDataset`).
  4. Create the policy via `make_policy(cfg.policy, ds_meta=dataset.meta)`.
  5. Build preprocessor and postprocessor pipelines (`make_pre_post_processors`). These encapsulate normalization, device placement, and any required transformations.
  6. Create optimizer and scheduler via `make_optimizer_and_scheduler(cfg, policy)`.
  7. Create a PyTorch `DataLoader` (with `EpisodeAwareSampler` for certain policies) and wrap everything with `accelerator.prepare(...)`.
  8. Enter step loop:
     - fetch batch from dataloader iterator
     - apply `preprocessor(batch)` to normalize and collate into tensors proper dtype/device
     - call `update_policy(...)` which:
       - policy.train(); with `accelerator.autocast()` run forward and compute loss
       - `accelerator.backward(loss)` (handles mixed precision)
       - gradient clipping (`accelerator.clip_grad_norm_`)
       - `optimizer.step()` then `optimizer.zero_grad()`
       - step LR scheduler if present
     - bookkeeping: increment `step`, log metrics, optionally evaluate, save checkpoints, push to hub

- Special features:
  - RA-BC (relabeling / reweighting) support: optional per-sample weights applied to loss.
  - Checkpoint resume via `load_training_state`.
  - Environment-based evaluation (`eval_env`) when training on simulation.
  - `push_to_hub` support for policy + processors when training finishes.

## 6) Dataset & dataloader details (LeRobotDataset)

- Dataset layout:
  - Root contains `data/` (parquet files chunked), `meta/` (info.json, episodes parquet), and `videos/` (mp4 per camera).
  - The dataset is wrapped with a `LeRobotDataset` object that lazily downloads (via HF snapshot) and loads an `hf_dataset` of frames.

- Querying and `delta_timestamps`:
  - `delta_timestamps` is a mapping such as:
    - `"observation.state": [-0.1, 0.0]`
    - `"action": [-0.1, 0.0, 0.1, 0.2, ...]`
  - The factory converts timestamps to integer frame offsets (`delta_indices`) relative to the dataset fps.
  - For an absolute index `abs_idx` and an episode, `_get_query_indices` computes indices to fetch for each feature, plus `action_is_pad` masks for padding when indices are outside episode bounds.
  - Video frames are decoded from mp4 files on demand using timestamps adjusted to episode start.

- `__getitem__` behavior:
  - Loads a row from HF dataset with index `idx` (or maps to a particular episode + file when using MultiLeRobotDataset).
  - If `delta_indices` set: it queries additional rows (previous/future frames) and their padding flags.
  - If `video_keys` present: decode frames from video file(s) for requested timestamps.
  - Apply `image_transforms` (if provided).
  - Adds string `task` and optional `subtask`.
  - Returns a dict of tensors keyed by feature names suitable for the policy preprocessor.

- Dataloader specifics:
  - `torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=..., sampler=..., pin_memory=device.type == "cuda", prefetch_factor=2 if num_workers>0 else None)`
  - When `cfg.policy` supports `drop_n_last_frames`, an `EpisodeAwareSampler` is used to avoid sampling last frames that lack future actions.
  - For streaming datasets, `StreamingLeRobotDataset` is available with similar API but streaming download semantics.
  - Preprocessing is applied to batches via `preprocessor(batch)` (this handles normalization, stacking multiple camera views into `observation.images`, dtype conversion, device transfer).

## 7) Configuration highlights (`DiffusionConfig` and training config)

- Important policy parameters (from `configuration_diffusion.py`):
  - `n_obs_steps` — how many past steps of observations are used as conditioning.
  - `horizon` — total predicted trajectory length produced by the diffusion model.
  - `n_action_steps` — how many of those predicted steps are to be executed (action chunking).
  - `input_features` / `output_features` — shapes and types the policy expects.
  - `vision_backbone`, `crop_shape`, `spatial_softmax_num_keypoints` — visual encoder options.
  - `down_dims`, `kernel_size`, `n_groups` — UNet architecture.
  - `diffusion_step_embed_dim` — timestep embedding size.
  - `noise_scheduler_type`, `num_train_timesteps`, `beta_schedule`, `prediction_type` — diffusion hyperparameters.
  - `clip_sample` and `clip_sample_range` — clipping behavior during sampling.

- Training pipeline config (`TrainPipelineConfig`):
  - `batch_size`, `num_workers`, `steps`, `save_freq`, `log_freq`, `eval_freq`
  - `optimizer` and `scheduler` presets — created by `make_optimizer_and_scheduler`.
  - `wandb` support — enabled via `cfg.wandb`.
  - `resume`, `checkpoint_path`, `policy.pretrained_path` — for resuming/initializing.

## 8) Practical advice & pointers for experiments

- Batch size and effective batch size:
  - `effective_bs = cfg.batch_size * accelerator.num_processes`. Many logging / epoch computations use effective batch size.

- Mixed precision:
  - The pipeline uses `accelerator.autocast()` and `accelerator.backward(loss)` to safely train with AMP.

- Gradient clipping:
  - `cfg.optimizer.grad_clip_norm` is applied per-step. If your model diverges, try lowering lr or enabling clipping.

- Handling padded actions:
  - When training on episodes, early/late frames may be copy-padded. Set `do_mask_loss_for_padding` to True if you want to mask such regions (but the default follows original Diffusion Policy).

- Scheduler choices:
  - Default training uses `DDPMScheduler` from `diffusers`. For faster / deterministic sampling, try `DDIMScheduler` at inference time.

- Debugging tips:
  - Inspect `batch` outputs from dataset to validate shapes: `observation.state` dims, `action`, `action_is_pad`.
  - Start with small `num_train_timesteps` and short `horizon` for quick experiments.

## 9) Example: key functions / files to inspect (quick map)

```text
src/lerobot/policies/diffusion/modeling_diffusion.py    # DiffusionPolicy, DiffusionModel, UNet, encoders
src/lerobot/policies/diffusion/configuration_diffusion.py  # DiffusionConfig (all policy hyperparams)
src/lerobot/policies/diffusion/processor_diffusion.py   # processors (normalization / transforms)
src/lerobot/scripts/lerobot_train.py                    # training loop, accelerator, dataloader, checkpointing
src/lerobot/datasets/lerobot_dataset.py                 # LeRobotDataset implementation, parquet/video handling
src/lerobot/datasets/factory.py                         # make_dataset helper and delta timestamp resolution
```

## 10) Example CLI (from repository examples / terminal)

Typical launch uses `lerobot-train` CLI. Example arguments (adapt to your dataset, device, and GPU count):

```bash
lerobot-train --dataset.repo_id=session_2026-02-16_19-20-16 \
              --dataset.root=/path/to/dataset/root \
              --policy.type=diffusion \
              --policy.device=cuda \
              --batch_size=256 \
              --steps=100000 \
              --save_freq=2000 \
              --log_freq=20 \
              --policy.n_obs_steps=2 \
              --policy.horizon=16 \
              --policy.n_action_steps=8 \
              --policy.vision_backbone=resnet18 \
              --policy.diffusion_step_embed_dim=128 \
              --policy.num_train_timesteps=100 \
              --output_dir=outputs/train/my_diffusion_run
```

Make sure the `--dataset.repo_id` or the `--dataset.root` points to an existing `LeRobotDataset` (parquet + meta + videos) with matching feature keys configured in `DiffusionConfig.input_features`.

---

If you want, I can:
- add inline call-outs to the exact lines in `modeling_diffusion.py` / `lerobot_train.py` referenced above (code references),
- include a short "quick-start" example config file,
- or expand the "debugging checklist" with exact commands and minimal reproducible experiment steps.


