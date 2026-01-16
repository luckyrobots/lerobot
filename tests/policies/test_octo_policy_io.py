import torch
import numpy as np


class _DummyOcto:
    """Minimal OctoPt stub for unit-testing LeRobot's OctoPolicy glue code.

    - `sample_actions` returns CPU tensors (matches OctoModelPt behavior).
    - `__call__` records inputs so tests can assert shapes/masks.
    """

    def __init__(self, *, action_dim: int, action_horizon: int):
        self.last_call = None
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        # Mimic OctoModelPt carrying an example_batch with a 3D task_completed (B,T,H).
        self.example_batch = {
            "observation": {
                "task_completed": torch.zeros((1, 2, self.action_horizon), dtype=torch.bool),
            },
            "task": {},
        }

    def create_tasks(self, *, texts, device):
        # LeRobot policy doesn't inspect task contents; just needs a dict.
        return {"language_instruction": texts, "pad_mask_dict": {"language_instruction": torch.ones(len(texts), dtype=torch.bool)}}

    def __call__(
        self,
        *,
        observations,
        tasks,
        timestep_pad_mask,
        action_pad_mask,
        gt_actions,
        train,
        **kwargs,
    ):
        self.last_call = {
            "observations": observations,
            "tasks": tasks,
            "timestep_pad_mask": timestep_pad_mask,
            "action_pad_mask": action_pad_mask,
            "gt_actions": gt_actions,
            "train": train,
        }
        # Match OctoModulePt head_outputs structure: head_outputs["action"] = (loss, metrics)
        loss = torch.tensor(0.123, device=timestep_pad_mask.device)
        return None, {"action": (loss, {})}

    def sample_actions(
        self,
        *,
        observations,
        tasks,
        unnormalization_statistics,
        timestep_pad_mask,
        train,
        **kwargs,
    ):
        B = int(observations["image_primary"].shape[0])
        A = self.action_dim
        H = self.action_horizon
        action = torch.zeros((B, H, A), dtype=torch.float32, device="cpu")
        if unnormalization_statistics is None:
            return action
        mean = unnormalization_statistics["mean"].float()
        std = unnormalization_statistics["std"].float()
        mask = unnormalization_statistics.get("mask", torch.ones_like(mean, dtype=torch.bool))
        # Apply NORMAL unnormalization: action*std + mean where mask is True.
        action = action[..., : len(mask)]
        action = torch.where(mask, (action * std) + mean, action)
        return action


def _make_policy_stub():
    from lerobot.policies.octo.configuration_octo import OctoConfig
    from lerobot.policies.octo.modeling_octo import OctoPolicy

    policy = OctoPolicy.__new__(OctoPolicy)
    # Initialize nn.Module internals (we bypassed __init__).
    torch.nn.Module.__init__(policy)
    policy.config = OctoConfig(device="cpu", window_size=2, action_horizon=4)
    # Use numpy stats to mirror what LeRobot v3 stats often provide.
    policy.dataset_stats = {"action": {"mean": np.array([1.0, 2.0, 3.0]), "std": np.array([2.0, 4.0, 8.0])}}
    policy.octo = _DummyOcto(action_dim=3, action_horizon=4)
    return policy


def test_octo_config_delta_indices():
    from lerobot.policies.octo.configuration_octo import OctoConfig

    cfg = OctoConfig(device="cpu", window_size=3, action_horizon=4)
    assert cfg.observation_delta_indices == [-2, -1, 0]
    assert cfg.action_delta_indices == [0, 1, 2, 3]


def test_build_observations_layout_and_masks():
    policy = _make_policy_stub()
    cfg = policy.config

    B, T, C, H, W = 2, 2, 3, 64, 80
    # Test uint8 path (common when decoding videos): values should round-trip after scaling.
    prim = torch.full((B, T, C, H, W), 128, dtype=torch.uint8)
    wrist = torch.full((B, T, C, H, W), 64, dtype=torch.uint8)
    batch = {
        cfg.primary_image_key: prim,
        cfg.wrist_image_key: wrist,
        f"{cfg.primary_image_key}_is_pad": torch.tensor([[True, False], [False, False]]),
        f"{cfg.wrist_image_key}_is_pad": torch.tensor([[False, False], [False, True]]),
        # Prefer dataset-provided timestep + episode end.
        "frame_index": torch.tensor([[10, 11], [20, 21]]),
        "next.done": torch.tensor([[False, False], [False, True]]),
        "task": ["do thing", "do other thing"],
    }

    obs, texts, timestep_pad_mask, pad_mask_dict = policy._build_observations(batch)

    assert texts == ["do thing", "do other thing"]
    assert obs["image_primary"].dtype == torch.uint8
    assert obs["image_wrist"].dtype == torch.uint8
    assert obs["image_primary"].shape[:2] == (B, T)
    assert obs["image_wrist"].shape[:2] == (B, T)
    assert obs["image_primary"].shape[2] == 3  # channel-first
    # Spot-check scaling: uint8 inputs should not get saturated to 255.
    assert int(obs["image_primary"][0, 0, 0, 0, 0].item()) == 128
    assert int(obs["image_wrist"][0, 0, 0, 0, 0].item()) == 64

    # timestep_pad_mask is AND of both modalities
    expected_primary = torch.tensor([[False, True], [True, True]])
    expected_wrist = torch.tensor([[True, True], [True, False]])
    expected = expected_primary & expected_wrist
    assert torch.equal(timestep_pad_mask.cpu(), expected)
    assert torch.equal(pad_mask_dict["image_primary"].cpu(), expected_primary)
    assert torch.equal(pad_mask_dict["image_wrist"].cpu(), expected_wrist)
    # Optional keys to match Octo example_batch (suppress OctoPt warnings).
    assert "timestep" in obs and obs["timestep"].shape == (B, T)
    assert torch.equal(obs["timestep"].cpu(), batch["frame_index"])
    assert "task_completed" in obs and obs["task_completed"].shape == (B, T, policy.config.action_horizon)
    # Broadcasted from per-timestep done signal.
    expected_done = batch["next.done"].to(dtype=torch.bool)[:, :, None].expand(B, T, policy.config.action_horizon)
    assert torch.equal(obs["task_completed"].cpu(), expected_done)
    assert "timestep" in pad_mask_dict and pad_mask_dict["timestep"].shape == (B, T)


def test_forward_wires_action_chunk_only_on_last_timestep():
    policy = _make_policy_stub()
    cfg = policy.config

    B, T, C, H, W = 2, 2, 3, 64, 80
    A = 3
    batch = {
        cfg.primary_image_key: torch.rand((B, T, C, H, W), dtype=torch.float32),
        cfg.wrist_image_key: torch.rand((B, T, C, H, W), dtype=torch.float32),
        "task": ["t1", "t2"],
        "action": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [13.0, 14.0, 15.0]],
                [[2.0, 4.0, 8.0], [1.0, 3.0, 5.0], [7.0, 9.0, 11.0], [13.0, 15.0, 17.0]],
            ],
            dtype=torch.float32,
        ),
        "action_is_pad": torch.tensor([[False, False, True, True], [False, True, False, True]]),
    }

    loss, metrics = policy.forward(batch)
    assert isinstance(loss, torch.Tensor)
    assert "action_mse0" in metrics

    call = policy.octo.last_call
    assert call is not None
    gt_actions = call["gt_actions"]
    action_pad_mask = call["action_pad_mask"]

    assert gt_actions.shape == (B, T, cfg.action_horizon, A)
    assert action_pad_mask.shape == (B, T, cfg.action_horizon, A)
    # Only last timestep should be supervised (mask True where not padded)
    assert not action_pad_mask[:, 0].any()
    expected_valid = (~batch["action_is_pad"]).to(dtype=torch.bool)
    assert torch.equal(action_pad_mask[:, -1, :, 0].cpu(), expected_valid.cpu())

    # Check normalization at last timestep, first horizon step
    # mean=[1,2,3], std=[2,4,8] => (a-mean)/std
    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([2.0, 4.0, 8.0])
    expected_norm0 = (batch["action"][:, 0, :] - mean) / std
    assert torch.allclose(gt_actions[:, -1, 0, :].cpu(), expected_norm0.cpu())


def test_predict_action_chunk_uses_unnormalization_stats():
    policy = _make_policy_stub()
    cfg = policy.config

    B, T, C, H, W = 2, 2, 3, 64, 80
    batch = {
        cfg.primary_image_key: torch.rand((B, T, C, H, W), dtype=torch.float32),
        cfg.wrist_image_key: torch.rand((B, T, C, H, W), dtype=torch.float32),
        "task": ["t1", "t2"],
    }
    actions = policy.predict_action_chunk(batch)
    # Dummy returns zeros then unnormalizes -> should equal mean per-dim.
    expected_mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert actions.shape[-1] == 3
    assert torch.allclose(actions[:, 0, :], expected_mean[None, :].repeat(B, 1))


