import numpy as np
import torch


def test_piper_action7_targets_matches_numpy_reference():
    # Torch implementation under test
    from lerobot.policies.octo.piper_action7_targets import PiperAction7TargetConfig, piper_joint_to_action7_targets

    # Numpy reference from the Piper shim
    from piper_sdk.vla_shim.joint_to_eef_delta import JointToEefDeltaConfig, JointToEefDeltaConverter

    rng = np.random.default_rng(0)
    B = 32

    # Keep joints in a moderate range (radians).
    q_curr = rng.uniform(low=-1.5, high=1.5, size=(B, 6)).astype(np.float32)
    q_tgt = q_curr + rng.normal(scale=0.05, size=(B, 6)).astype(np.float32)
    gripper = rng.uniform(low=0.0, high=0.04, size=(B,)).astype(np.float32)

    state = np.concatenate([q_curr, gripper[:, None]], axis=1)
    action = np.concatenate([q_tgt, gripper[:, None]], axis=1)

    cfg_np = JointToEefDeltaConfig(max_translation_m=0.02, max_rotation_rad=float(np.deg2rad(5.0)))
    ref = JointToEefDeltaConverter(dh_is_offset=1, cfg=cfg_np)
    ref_out = []
    for i in range(B):
        a7, _ = ref.joint_to_action7(q_curr_rad=q_curr[i], q_tgt_rad=q_tgt[i], gripper_tgt_m=float(gripper[i]))
        ref_out.append(a7)
    ref_out = np.stack(ref_out, axis=0)

    cfg_t = PiperAction7TargetConfig(
        dh_is_offset=1,
        max_translation_m=0.02,
        max_rotation_rad=float(np.deg2rad(5.0)),
        gripper_open_threshold_m=0.0175,
    )
    out = piper_joint_to_action7_targets(
        state=torch.tensor(state, dtype=torch.float32),
        action=torch.tensor(action, dtype=torch.float32),
        cfg=cfg_t,
    ).cpu().numpy()

    np.testing.assert_allclose(out, ref_out, atol=5e-4, rtol=0)


