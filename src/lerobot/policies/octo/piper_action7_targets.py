from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class PiperAction7TargetConfig:
    """Torch implementation of the Piper joint->EEF-delta action7 target.

    Matches the numpy reference implementation used in the Piper shim:
    - translation delta in EEF/body frame
    - rotation delta as body-frame rotvec (log map of R_err = R_curr^T R_tgt)
    - clamp by norm then normalize into ~[-1, 1]
    - gripper is binary (+1 open, -1 close) using threshold in meters
    """

    dh_is_offset: Literal[0, 1] = 1
    max_translation_m: float = 0.02
    max_rotation_rad: float = math.radians(5.0)
    gripper_open_threshold_m: float = 0.0175


def _dh_params(dh_is_offset: int, *, device: torch.device, dtype: torch.dtype):
    # Extracted from `piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics`.
    if int(dh_is_offset) == 1:
        a_mm = [0.0, 0.0, 285.03, -21.98, 0.0, 0.0]
        alpha = [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2]
        theta0 = [0.0, -math.pi * 172.22 / 180.0, -102.78 / 180.0 * math.pi, 0.0, 0.0, 0.0]
        d_mm = [123.0, 0.0, 0.0, 250.75, 0.0, 91.0]
    else:
        a_mm = [0.0, 0.0, 285.03, -21.98, 0.0, 0.0]
        alpha = [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2]
        theta0 = [0.0, -math.pi * 174.22 / 180.0, -100.78 / 180.0 * math.pi, 0.0, 0.0, 0.0]
        d_mm = [123.0, 0.0, 0.0, 250.75, 0.0, 91.0]

    a_mm_t = torch.tensor(a_mm, device=device, dtype=dtype)
    alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
    theta0_t = torch.tensor(theta0, device=device, dtype=dtype)
    d_mm_t = torch.tensor(d_mm, device=device, dtype=dtype)
    return a_mm_t, alpha_t, theta0_t, d_mm_t


def _fk_T_world_ee(q_rad: torch.Tensor, *, dh_is_offset: int) -> torch.Tensor:
    """Batched DH forward kinematics -> SE(3) matrix with translation in meters.

    Args:
        q_rad: (B,6) joint angles in radians
    Returns:
        T: (B,4,4)
    """
    if q_rad.ndim != 2 or q_rad.shape[-1] != 6:
        raise ValueError(f"q_rad must have shape (B,6), got {tuple(q_rad.shape)}")
    B = q_rad.shape[0]
    device = q_rad.device
    dtype = q_rad.dtype
    a_mm, alpha, theta0, d_mm = _dh_params(dh_is_offset, device=device, dtype=dtype)

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    for i in range(6):
        theta = q_rad[:, i] + theta0[i]
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha[i])
        sa = torch.sin(alpha[i])
        Ti = torch.zeros((B, 4, 4), device=device, dtype=dtype)
        Ti[:, 0, 0] = ct
        Ti[:, 0, 1] = -st
        Ti[:, 0, 2] = 0.0
        Ti[:, 0, 3] = a_mm[i]

        Ti[:, 1, 0] = st * ca
        Ti[:, 1, 1] = ct * ca
        Ti[:, 1, 2] = -sa
        Ti[:, 1, 3] = -sa * d_mm[i]

        Ti[:, 2, 0] = st * sa
        Ti[:, 2, 1] = ct * sa
        Ti[:, 2, 2] = ca
        Ti[:, 2, 3] = ca * d_mm[i]

        Ti[:, 3, 3] = 1.0

        T = torch.bmm(T, Ti)

    # Convert mm -> m
    T = T.clone()
    T[:, :3, 3] = T[:, :3, 3] * 1e-3
    return T


def _rot_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) log map.

    Args:
        R: (B,3,3)
    Returns:
        rotvec: (B,3)
    """
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (B,3,3), got {tuple(R.shape)}")
    B = R.shape[0]
    device = R.device
    dtype = R.dtype

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    rotvec = torch.zeros((B, 3), device=device, dtype=dtype)

    small = theta < 1e-6
    if small.any():
        # For very small angles, rotvec ~ 0.
        rotvec[small] = 0.0

    near_pi = (torch.abs(theta - math.pi) < 1e-4) & (~small)
    general = (~small) & (~near_pi)

    if general.any():
        th = theta[general]
        denom = 2.0 * torch.sin(th)
        w_hat = (R[general] - R[general].transpose(1, 2)) / denom[:, None, None]
        axis = torch.stack([w_hat[:, 2, 1], w_hat[:, 0, 2], w_hat[:, 1, 0]], dim=1)
        rotvec[general] = axis * th[:, None]

    if near_pi.any():
        Rpi = R[near_pi]
        th = theta[near_pi]
        axis = torch.empty((Rpi.shape[0], 3), device=device, dtype=dtype)
        axis[:, 0] = torch.sqrt(torch.clamp((Rpi[:, 0, 0] + 1.0) / 2.0, min=0.0))
        axis[:, 1] = torch.sqrt(torch.clamp((Rpi[:, 1, 1] + 1.0) / 2.0, min=0.0))
        axis[:, 2] = torch.sqrt(torch.clamp((Rpi[:, 2, 2] + 1.0) / 2.0, min=0.0))
        axis[:, 0] = torch.where((Rpi[:, 2, 1] - Rpi[:, 1, 2]) < 0, -axis[:, 0], axis[:, 0])
        axis[:, 1] = torch.where((Rpi[:, 0, 2] - Rpi[:, 2, 0]) < 0, -axis[:, 1], axis[:, 1])
        axis[:, 2] = torch.where((Rpi[:, 1, 0] - Rpi[:, 0, 1]) < 0, -axis[:, 2], axis[:, 2])
        axis_norm = torch.linalg.norm(axis, dim=1).clamp(min=1e-12)
        axis = axis / axis_norm[:, None]
        rotvec[near_pi] = axis * th[:, None]

    return rotvec


def _clamp_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    max_norm_t = torch.tensor(float(max_norm), device=v.device, dtype=v.dtype)
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    scale = torch.where((n > max_norm_t) & (n > 1e-12), max_norm_t / n, torch.ones_like(n))
    return v * scale


def piper_joint_to_action7_targets(
    *,
    state: torch.Tensor,
    action: torch.Tensor,
    cfg: PiperAction7TargetConfig = PiperAction7TargetConfig(),
) -> torch.Tensor:
    """Compute normalized action7 target from Piper joint state + joint-target action.

    Args:
        state: (B,7) [joint1..joint6, gripper]
        action: (B,7) [joint1..joint6, gripper]
    Returns:
        action7: (B,7) [dx,dy,dz, drotvec_x,y,z, gripper(+1/-1)]
    """
    if state.ndim != 2 or action.ndim != 2 or state.shape[-1] != 7 or action.shape[-1] != 7:
        raise ValueError(f"state/action must have shape (B,7), got {tuple(state.shape)} and {tuple(action.shape)}")

    q_curr = state[:, :6]
    q_tgt = action[:, :6]
    gripper_tgt_m = action[:, 6]

    T_curr = _fk_T_world_ee(q_curr, dh_is_offset=int(cfg.dh_is_offset))
    T_tgt = _fk_T_world_ee(q_tgt, dh_is_offset=int(cfg.dh_is_offset))
    R_curr = T_curr[:, :3, :3]
    t_curr = T_curr[:, :3, 3]
    R_tgt = T_tgt[:, :3, :3]
    t_tgt = T_tgt[:, :3, 3]

    # EEF/body frame deltas
    delta_xyz_eef = torch.bmm(R_curr.transpose(1, 2), (t_tgt - t_curr).unsqueeze(-1)).squeeze(-1)
    R_err = torch.bmm(R_curr.transpose(1, 2), R_tgt)
    rotvec_eef = _rot_to_rotvec(R_err)

    # Clamp + normalize
    delta_xyz_clamped = _clamp_norm(delta_xyz_eef, float(cfg.max_translation_m))
    rotvec_clamped = _clamp_norm(rotvec_eef, float(cfg.max_rotation_rad))
    a_xyz = delta_xyz_clamped / float(cfg.max_translation_m)
    a_rot = rotvec_clamped / float(cfg.max_rotation_rad)

    a_grip = torch.where(
        gripper_tgt_m > float(cfg.gripper_open_threshold_m),
        torch.ones_like(gripper_tgt_m),
        -torch.ones_like(gripper_tgt_m),
    )
    action7 = torch.cat([a_xyz, a_rot, a_grip[:, None]], dim=1).to(dtype=torch.float32)
    return action7


