# ACT (Action Chunking Transformer) — Theory, Implementation, Training, Data-loading

This document explains the ACT policy implemented in this repository: architecture, training objective, inference behavior (action chunking and temporal ensembling), and how it integrates with the dataset/dataloader. It mirrors the level of detail in `diffusion_explained.md` but focused on ACT.

---

## 1) Short overview / intuition

- ACT (Action Chunking Transformer) predicts action "chunks" (a sequence of future actions) in one forward pass and then executes a subset of them (`n_action_steps`) in the environment. This reduces policy query frequency and improves sample-efficiency for sequential control tasks.
- The implementation optionally uses a VAE-style encoder to produce a latent z that conditions the transformer; during inference z is set to zero (or sampled deterministically). The transformer decoder produces the chunk of actions.
- Temporal ensembling is an optional online averaging mechanism used at inference to smooth and improve action stability.

## 2) Where ACT is implemented in the repo

- Core implementation files:
  - `src/lerobot/policies/act/modeling_act.py` — `ACTPolicy`, `ACT` model, encoder/decoder modules, temporal ensembler.
  - `src/lerobot/policies/act/configuration_act.py` — `ACTConfig` (hyperparameters).
  - `src/lerobot/policies/act/*` other helper modules (encoder layers, positional embeddings).

Code references (key spots):

```41:69:d:/FInal_Setup/lerobot/src/lerobot/policies/act/modeling_act.py
class ACTPolicy(PreTrainedPolicy):
    ...
    def __init__(...):
        self.model = ACT(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(...)
```

```142:161:d:/FInal_Setup/lerobot/src/lerobot/policies/act/modeling_act.py
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        l1_loss = F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        if self.config.use_vae:
            mean_kld = ...  # KL divergence
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
```

```448:456:d:/FInal_Setup/lerobot/src/lerobot/policies/act/modeling_act.py
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
```

```165:176:d:/FInal_Setup/lerobot/src/lerobot/policies/act/modeling_act.py
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
```

And configuration highlights:

```84:88:d:/FInal_Setup/lerobot/src/lerobot/policies/act/configuration_act.py
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
```

```114:121:d:/FInal_Setup/lerobot/src/lerobot/policies/act/configuration_act.py
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    temporal_ensemble_coeff: float | None = None
```

## 3) Architecture (detailed)

- High-level modules:
  - VAE encoder (optional): encodes the target action chunk and robot state into a latent z (BERT-style with a CLS token that parameterizes μ and log σ²).
  - Vision backbone: ResNet (typically `resnet18`) produces spatial feature maps for each camera view; features are projected into transformer token embeddings.
  - Transformer encoder: processes tokens consisting of [latent, robot_state?, env_state?, image_feature_pixels...] into context.
  - Transformer decoder: uses learnable decoder queries (one per chunk position) with cross-attention to generate chunk outputs.
  - Action head: linear layer mapping decoder features → action_dim per time step.

- Key points:
  - The VAE encoder is used only at training time (when `use_vae=True`) to learn a latent distribution over action chunks, enabling a richer generative objective. At inference, the model runs without sampling from the VAE (latent is set to zeros unless other behavior is implemented).
  - Image features are flattened per-pixel from the backbone feature map and appended as tokens to the encoder sequence; 2D sinusoidal pos-embeddings are used for spatial information.
  - Decoder positional embeddings are learnable (like DETR queries) and the decoder predicts the full chunk (length `chunk_size`) which is then trimmed to `n_action_steps` for execution.

## 4) Training objective & forward pass

- Expected batch keys:
  - `observation.state` (robot joint states) — optional depending on config
  - `observation.images` — list of camera tensors (each: B, C, H, W)
  - `action`: (B, chunk_size, action_dim) — required when training with VAE (for reconstructive objective)
  - `action_is_pad`: (B, chunk_size) — boolean mask for padding

- Forward / loss (implemented in `ACTPolicy.forward`):
  1. Model returns `actions_hat, (mu, log_sigma_x2)` where mu/log_sigma are latent params (or (None, None) if VAE disabled).
  2. Reconstruction loss: L1 (mean absolute) between `actions_hat` and `batch[ACTION]`, masked by `action_is_pad`.
  3. If `use_vae` is True, compute KL divergence of latent PDF to standard normal and add `kl_weight * KLD` to loss.
  4. Return `loss` and a diagnostic `loss_dict` containing `l1_loss` and `kld_loss` (when present).

Practical notes:
  - L1 is chosen as reconstruction for robustness to outliers and because it encourages sharper motor outputs.
  - The KL weight (`kl_weight`) is a key hyperparameter: too large → collapsed reconstructions; too small → poor latent regularization.

## 5) Inference: action chunking & temporal ensembling

- Action chunking:
  - The model predicts a sequence of length `chunk_size`. At each policy invocation the agent executes only the first `n_action_steps` of that sequence and discards/overwrites the rest.
  - This reduces the frequency the model must be queried and can stabilize control.
  - `ACTPolicy.select_action` manages an action queue: when the queue is empty it calls `predict_action_chunk` to refill it and then pops a single action for execution.

- Temporal ensembling:
  - If `temporal_ensemble_coeff` is set, `ACTTemporalEnsembler` performs an online exponential-weighted average of predicted chunks and returns the ensembled action at each step instead of a raw predicted action.
  - This smooths noisy predictions and can improve stability; it requires `n_action_steps == 1` because the ensemble must be updated every environment step.

Code reference for queue and ensemble usage:

```114:122:d:/FInal_Setup/lerobot/src/lerobot/policies/act/modeling_act.py
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
```

## 6) Integration with dataset & dataloader

- The `LeRobotDataset` used by the training script provides `action` sequences and `action_is_pad` for chunked training — the ACT policy expects the chunk-length `chunk_size` to align with how dataset `delta_timestamps` were configured (factory converts timestamps → delta indices).
- Typical training pipeline (same orchestration as other policies):
  - `train` in `lerobot/scripts/lerobot_train.py` uses `make_dataset` → `LeRobotDataset`.
  - DataLoader constructed with `batch_size`, `num_workers`, optional `EpisodeAwareSampler`.
  - `preprocessor(batch)` converts dataset dicts to tensors and stacks multiple camera views into `observation.images`.
  - In training loop: `batch = next(dl_iter); batch = preprocessor(batch); loss, _ = policy.forward(batch)` and then backprop via `update_policy`.

Important: ACTConfig enforces `n_obs_steps == 1` (current code) so dataset delta timestamps must reflect a single observation snapshot per training sample.

## 7) Key config knobs (from `ACTConfig`)

- Chunking parameters:
  - `chunk_size` — length of predicted chunk (default 100).
  - `n_action_steps` — how many of the chunk are executed (must be <= chunk_size).
  - `n_obs_steps` — currently required to be 1.

- VAE & latent:
  - `use_vae` — enable variational objective (default True).
  - `latent_dim` — VAE latent dimension.
  - `kl_weight` — KL weight added to reconstruction loss.

- Transformer:
  - `dim_model`, `n_heads`, `n_encoder_layers`, `n_decoder_layers`, `pre_norm`, `dropout`, `dim_feedforward`, etc.

- Vision:
  - `vision_backbone`, `pretrained_backbone_weights`, and `replace_final_stride_with_dilation`.

- Inference smoothing:
  - `temporal_ensemble_coeff` — if set, enables online temporal ensembling (default None).

## 8) Practical advice & debugging tips

- When to use VAE:
  - VAE helps capture multimodal action distributions. If dataset is unimodal or you don't need stochasticity, try disabling `use_vae` to simplify training.

- Chunk size vs action steps:
  - Larger `chunk_size` means the transformer must model longer trajectories; this can increase memory/compute but provides richer lookahead. `n_action_steps` controls how often you query the model.

- Temporal ensembling:
  - Use with caution: it requires `n_action_steps == 1`. It smooths but also biases older predictions.

- Loss balancing:
  - `kl_weight` is sensitive. Start with the default and run ablations if reconstructions look poor or the latent collapses.

- Check batch shapes:
  - Ensure `batch[ACTION]` matches `(B, chunk_size, action_dim)` and `action_is_pad` matches `(B, chunk_size)`.

## 9) Example CLI snippet

```bash
lerobot-train --policy.type=act \
              --dataset.repo_id=my_dataset \
              --policy.chunk_size=100 \
              --policy.n_action_steps=10 \
              --policy.use_vae=true \
              --policy.latent_dim=32 \
              --batch_size=64 \
              --steps=50000 \
              --output_dir=outputs/train/act_run
```

## 10) Where to look next in code

- `src/lerobot/policies/act/modeling_act.py` — most of the model & inference logic.
- `src/lerobot/policies/act/configuration_act.py` — default hyperparameters and validation.
- `src/lerobot/scripts/lerobot_train.py` — training orchestration / accelerator / dataloader usage (same as other policies).
- `src/lerobot/datasets/lerobot_dataset.py` — how actions and `action_is_pad` are provided by the dataset.

---

If you'd like, I can:
- add inline code references to exact, annotated lines in `modeling_act.py` (more than the snippets above), or
- create a minimal runnable `TrainPipelineConfig` example for a small ACT experiment, or
- run a quick static check for any config mismatches between default `ACTConfig` and an example dataset you provide.


