#!/usr/bin/env python
"""Standalone LuckyEngine pick-and-place eval for in-training checkpoints.

Called from the training loop after each checkpoint save. Connects to a running
LuckyEngine instance via HazelBackend gRPC, loads the just-saved checkpoint,
runs a few pick-and-place episodes, and returns success metrics.

Everything is wrapped in try/except — any failure returns {success: False, error: str}
so training never stops.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.luckyengine.hazel_backend import HazelBackend, HazelStreamConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action
from lerobot.policies.utils import prepare_observation_for_inference



# ── Constants (match piper_room.py / sweep defaults) ──────────────
HOME_ACTION = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035]
BLOCK_POS = (0.317096353, 0.0464101955, 0.000183301046)
BLOCK_QUAT = (0.0, 0.0, 0.0, 1.0)  # identity
CAMERAS = ("CameraGripper", "CameraLeft", "CameraTop")
AGENT_NAME = "agent_0"
ROBOT_NAME = ""
# Render directly at the policy input size to reduce LuckyEngine GPU load ~8×.
# (Training data was also resized to 96×96 before the network, so the domain
# gap from skipping the 320×240→96×96 downscale is minimal.)
CAMERA_W, CAMERA_H, CAMERA_FPS = 96, 96, 30


def _autodetect_hazel_port(
    host: str, start: int = 50055, end: int = 50100, timeout_s: float = 0.5
) -> int:
    import socket
    import grpc
    from lerobot.luckyengine.grpc_stubs import default_proto_path, generate_python_stubs

    st = generate_python_stubs(str(default_proto_path()))
    pb2, pb2_grpc = st.pb2, st.pb2_grpc

    for port in range(start, end + 1):
        try:
            with socket.create_connection((host, port), timeout=0.05):
                pass
        except OSError:
            continue
        ch = grpc.insecure_channel(f"{host}:{port}")
        scene = pb2_grpc.SceneServiceStub(ch)
        try:
            scene.GetSceneInfo(pb2.GetSceneInfoRequest(), timeout=timeout_s)
            return port
        except Exception:
            continue
    raise RuntimeError(f"No Hazel gRPC server found on {host}:{start}-{end}")


def _set_entity_pose(backend, entity_id, transform, pos, quat):
    t = type(transform)()
    t.CopyFrom(transform)
    t.position.x, t.position.y, t.position.z = float(pos[0]), float(pos[1]), float(pos[2])
    if quat is not None:
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = (
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        )
    backend.set_entity_transform(entity_id=entity_id, transform=t)


def _randomize_block_pos(rng: np.random.Generator) -> tuple[float, float, float]:
    """Return a random block position within ±3 cm of the canonical spot."""
    x = BLOCK_POS[0] + rng.uniform(-0.06, 0.06)
    y = BLOCK_POS[1] + rng.uniform(-0.06, 0.06)
    z = BLOCK_POS[2]  # keep on table surface
    return (x, y, z)


def _reset_robot_and_block(
    backend: HazelBackend,
    block_id: int,
    block_transform,
    rng: np.random.Generator,
    step_timeout_ms: int = 2000,
):
    """Simplified reset: reset agent, drive to home, set block to random nearby pose."""
    # 1) Reset MuJoCo state
    try:
        backend.reset_agent(agent_name=AGENT_NAME)
    except Exception:
        pass

    # 2) Hold home pose for 1.5s
    dt = 1.0 / 30.0
    t_end = time.perf_counter() + 1.5
    while time.perf_counter() < t_end:
        backend.step(list(map(float, HOME_ACTION)), agent_name=AGENT_NAME, timeout_ms=step_timeout_ms)
        time.sleep(dt)

    # 3) Randomize block position and set pose (twice for robustness)
    pos = _randomize_block_pos(rng)
    for _ in range(2):
        _set_entity_pose(backend, block_id, block_transform, pos, BLOCK_QUAT)
        time.sleep(0.03)

    # 4) Hold block pose for 0.25s against scene randomization
    t_end = time.perf_counter() + 0.25
    while time.perf_counter() < t_end:
        _set_entity_pose(backend, block_id, block_transform, pos, BLOCK_QUAT)
        time.sleep(dt)

    time.sleep(0.1)


def _coerce_state(state: np.ndarray, expected_dim: int, backend: HazelBackend) -> np.ndarray:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.size == expected_dim:
        return np.ascontiguousarray(s)
    if s.size == expected_dim * 2:
        return np.ascontiguousarray(s[:expected_dim])
    qpos, _ = backend.get_joint_state(robot_name=ROBOT_NAME)
    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if q.size == expected_dim:
        return np.ascontiguousarray(q)
    raise RuntimeError(f"State dim mismatch: got {s.size}, qpos {q.size}, expected {expected_dim}")


def _extract_camera_keys(pretrained_dir: Path) -> list[str]:
    import json
    cfg_path = pretrained_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    feats = cfg.get("input_features") or {}
    return [k[len("observation.images."):] for k in feats if k.startswith("observation.images.")]


def _run_inference(obs, policy, device, preproc, postproc, _dbg=False) -> tuple[list[float], float]:
    """Run policy inference. Returns (action, elapsed_s).

    Pass _dbg=True to print a per-phase breakdown:
      preproc – numpy→tensor conversion + normalisation (CPU+GPU copy)
      forward – policy.select_action (GPU forward pass when queue empty, else ~0ms queue pop)
      cpu cpy – D2H tensor copy

    n_action_steps>1 means only every n-th call runs the GPU forward; the rest
    are cheap queue pops.  elapsed_s is wall time after GPU sync.
    """
    from contextlib import nullcontext
    from copy import copy as _copy

    if _dbg:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    if not _dbg:
        # Fast path
        action_t = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preproc,
            postprocessor=postproc,
            use_amp=True,
        )
        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
        return action, time.perf_counter() - t0

    # -- Debug path: inline predict_action so we can synchronize between phases --
    # Phase 1: numpy -> tensor + normalise  (mirrors predict_action internals)
    _t = time.perf_counter()
    with torch.inference_mode():
        _obs_t = prepare_observation_for_inference(_copy(obs), device)
        _obs_norm = preproc(_obs_t)
    torch.cuda.synchronize(device)
    print(f"  [dbg] preproc   {1e3*(time.perf_counter()-_t):.1f}ms", flush=True)

    # Phase 2: policy.select_action + postproc
    # (queue pop if action queue non-empty, else full GPU forward pass)
    _t = time.perf_counter()
    _amp = torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), _amp:
        action_t = policy.select_action(_obs_norm)
        action_t = postproc(action_t)
    torch.cuda.synchronize(device)
    print(f"  [dbg] forward   {1e3*(time.perf_counter()-_t):.1f}ms", flush=True)

    # Phase 3: D2H copy
    _t = time.perf_counter()
    action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
    print(f"  [dbg] cpu cpy   {1e3*(time.perf_counter()-_t):.1f}ms", flush=True)

    return action, time.perf_counter() - t0


def run_luckyengine_eval(
    checkpoint_dir: Path,
    host: str = "127.0.0.1",
    port: int = 0,
    n_episodes: int = 3,
    max_steps: int = 1200,
    success_dist_m: float = 0.08,
    control_hz: float = 30.0,
    save_video: bool = True,
    video_camera: str = "CameraLeft",
    step_timeout_ms: int = 2000,
) -> dict:
    """Run pick-and-place eval against LuckyEngine. Returns metrics dict.

    Control loop (sequential, fresh obs):

        for each step:
            step(prev_action)        # physics frame, ~25ms
            obs  = get_observation() # read streaming buffer, ~0.5ms
            action = infer(obs)      # GPU forward (only every n_action_steps), ~6ms avg
            prev_action = action

        With n_action_steps=8, the GPU forward pass runs only once every 8 steps;
        the other 7 steps are cheap queue pops (~0.1ms).  Expected avg: ~30Hz.

    Args:
        checkpoint_dir: Path to the pretrained_model directory of the checkpoint.
        host: Hazel gRPC host.
        port: Hazel gRPC port (0 = auto-detect).
        n_episodes: Number of episodes to run.
        max_steps: Max control steps per episode.
        success_dist_m: Distance threshold for success (block to dropbox).
        control_hz: Control frequency in Hz.
        step_timeout_ms: How long (ms) the server waits for a physics frame per Step() call.
            Default 2000ms handles background-throttled LuckyEngine (Windows deprioritises
            background windows; frames can take >100ms). Increase if you see timeout errors.

    Returns:
        dict with keys: success, success_rate, min_dist, episodes_run, error (if any).
    """
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return {"success": False, "error": f"Checkpoint not found: {checkpoint_dir}"}

        # Auto-detect port
        if port == 0:
            port = _autodetect_hazel_port(host)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load policy
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
        policy_camera_keys = _extract_camera_keys(checkpoint_dir)
        if not policy_camera_keys:
            policy_camera_keys = list(CAMERAS)

        state_feat = cfg.input_features.get("observation.state")
        state_dim = state_feat.shape[0] if state_feat is not None else 7

        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(pretrained_name_or_path=str(checkpoint_dir))
        preproc, postproc = make_pre_post_processors(cfg, pretrained_path=str(checkpoint_dir))
        policy.to(device)
        policy.eval()

        # Let cuDNN benchmark and cache the fastest kernel for our exact input shapes.
        # The cost is a few extra forward passes during warmup — amortised immediately.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Warm up CUDA kernels before connecting to the sim so first-call
        # JIT overhead doesn't pollute episode timing.
        print("[eval] Warming up CUDA (8 passes)...", flush=True)
        _dummy_obs = {"observation.state": np.zeros(state_dim, dtype=np.float32)}
        for _cam in policy_camera_keys:
            _dummy_obs[f"observation.images.{_cam}"] = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        for _ in range(8):
            _run_inference(_dummy_obs, policy, device, preproc, postproc)
        print("[eval] Warmup done.", flush=True)

        # Connect to LuckyEngine
        backend = HazelBackend(host=host, port=port)
        backend.connect()
        backend.get_scene_info()  # fail-fast

        # Snapshot block entity
        block_entity = backend.get_entity(tag="Red Block")
        block_id = int(block_entity.id.id)
        block_transform = block_entity.transform

        # Camera map (identity)
        scene_camera_map = {cam: cam for cam in policy_camera_keys}
        scene_cameras = sorted(set(policy_camera_keys))

        # Start streams
        cam_cfg = HazelStreamConfig(
            target_fps=CAMERA_FPS, width=CAMERA_W, height=CAMERA_H, format="raw",
        )
        for cam in scene_cameras:
            backend.start_camera_stream(camera_name=cam, cfg=cam_cfg)
        backend.start_agent_stream(agent_name=AGENT_NAME, target_fps=CAMERA_FPS)
        backend.wait_for_first_obs(camera_names=scene_cameras, timeout_s=10.0)

        # Wait for non-black frames
        time.sleep(1.0)

        # Thresholds for granular stage detection
        PICK_Z_THRESH = 0.02   # block Z > 0.02m → picked off table
        MOVE_XY_THRESH = 0.05  # block moved > 5cm XY from start → moved toward goal

        rng = np.random.default_rng()
        successes = []
        min_dists = []
        picked_list = []
        moved_list = []
        spawn_positions = []
        episodes_run = 0

        # Video recording setup
        video_dir = None
        if save_video:
            video_dir = checkpoint_dir.parent / "eval_videos"
            video_dir.mkdir(parents=True, exist_ok=True)

        try:
            for ep in range(n_episodes):
                policy.reset()
                _reset_robot_and_block(backend, block_id, block_transform, rng, step_timeout_ms=step_timeout_ms)

                # Record block start position for move detection and logging
                block_start = backend.get_entity_position(tag="Red Block").copy()
                spawn_positions.append(block_start.tolist())
                # Cache goal position — Dropbox never moves during an episode
                goal_pos = backend.get_entity_position(tag="Dropbox")

                ep_min_dist = math.inf
                ep_success = False
                ep_picked = False
                ep_moved = False
                ep_frames = []
                dt_target = 1.0 / max(1e-6, control_hz)
                next_t = time.perf_counter()

                # Timing accumulators (ep0 only)
                _t_obs = _t_step = _t_infer = _t_total = 0.0

                prev_action = list(map(float, HOME_ACTION))

                for step in range(max_steps):
                    _step_start = time.perf_counter()

                    # -- 1. Step physics with previous action --
                    _t0 = time.perf_counter()
                    backend.step(prev_action, agent_name=AGENT_NAME, timeout_ms=step_timeout_ms)
                    _t_step += time.perf_counter() - _t0

                    # -- 2. Get FRESH observation (strictly post-step) --
                    # step() returns only after the physics frame completes, so
                    # the streaming buffer here reflects the new robot state.
                    _t0 = time.perf_counter()
                    obs_scene = backend.get_observation(camera_map=scene_camera_map)
                    state = _coerce_state(obs_scene["observation.state"], state_dim, backend)

                    # Capture video frame
                    if save_video:
                        raw_frame = obs_scene.get(f"observation.images.{video_camera}")
                        if raw_frame is not None:
                            raw_frame = np.ascontiguousarray(np.flip(raw_frame, axis=0))
                            ep_frames.append(raw_frame)

                    obs = {"observation.state": state}
                    for cam in policy_camera_keys:
                        img = obs_scene.get(f"observation.images.{cam}")
                        if img is not None:
                            # Images already at 96×96 (CAMERA_W/H set to policy input size).
                            # Only flip vertically to correct GPU readback orientation.
                            img = np.ascontiguousarray(np.flip(img, axis=0))
                        obs[f"observation.images.{cam}"] = img
                    _t_obs += time.perf_counter() - _t0

                    # -- 3. Infer next action --
                    _t0 = time.perf_counter()
                    _dbg_this_step = (ep == 0 and step < 10)
                    if _dbg_this_step:
                        print(f"  [dbg] -- step {step} --", flush=True)
                    prev_action, _infer_elapsed = _run_inference(
                        obs, policy, device, preproc, postproc, _dbg=_dbg_this_step
                    )
                    _t_infer += _infer_elapsed

                    _t_total += time.perf_counter() - _step_start

                    # Check for success every 50 steps (early termination)
                    if step % 50 == 49:
                        block_pos = backend.get_entity_position(tag="Red Block")
                        dist = float(np.linalg.norm(block_pos - goal_pos))
                        ep_min_dist = min(ep_min_dist, dist)
                        if block_pos[2] > PICK_Z_THRESH:
                            ep_picked = True
                        xy_disp = float(np.linalg.norm(block_pos[:2] - block_start[:2]))
                        if ep_picked and xy_disp > MOVE_XY_THRESH:
                            ep_moved = True
                        if dist <= success_dist_m:
                            ep_success = True
                            break

                        # If block fell flat (tipped over), stand it back up in place
                        if block_pos[2] < PICK_Z_THRESH:
                            blk_ent = backend.get_entity(tag="Red Block")
                            rot = blk_ent.transform.rotation
                            tilt = abs(rot.x) + abs(rot.y) + abs(rot.z)
                            if tilt > 0.1:
                                upright_pos = (float(block_pos[0]), float(block_pos[1]), float(BLOCK_POS[2]))
                                _set_entity_pose(backend, block_id, block_transform, upright_pos, BLOCK_QUAT)
                                time.sleep(0.05)

                    # Rate limit
                    next_t += dt_target
                    sleep_s = next_t - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                # Print timing breakdown after first episode
                if ep == 0 and step > 0:
                    n = step + 1
                    actual_hz = n / _t_total if _t_total > 0 else 0.0
                    print(
                        f"\n[timing] ep0 over {n} steps  target={control_hz:.0f}Hz  "
                        f"actual={actual_hz:.1f}Hz  "
                        f"(step={1e3*_t_step/n:.1f}ms  "
                        f"obs+preproc={1e3*_t_obs/n:.1f}ms  "
                        f"infer={1e3*_t_infer/n:.1f}ms  "
                        f"total={1e3*_t_total/n:.1f}ms/step)"
                    )
                    if actual_hz < control_hz * 0.85:
                        print(
                            f"[timing] WARNING: actual Hz ({actual_hz:.1f}) is "
                            f"{100*(1-actual_hz/control_hz):.0f}% below target ({control_hz:.0f}). "
                            f"Policy was trained at {control_hz:.0f}Hz — temporal mismatch may hurt performance."
                        )

                # Single block-position query per episode (outside hot path)
                block_pos_final = backend.get_entity_position(tag="Red Block")
                dist_final = float(np.linalg.norm(block_pos_final - goal_pos))
                ep_min_dist = dist_final
                ep_success = dist_final <= success_dist_m
                ep_picked = block_pos_final[2] > PICK_Z_THRESH
                xy_displacement = float(np.linalg.norm(block_pos_final[:2] - block_start[:2]))
                ep_moved = ep_picked and xy_displacement > MOVE_XY_THRESH

                # Save episode video
                if save_video and ep_frames and video_dir is not None:
                    ckpt_name = checkpoint_dir.parent.name
                    video_path = video_dir / f"ep{ep}_{ckpt_name}.mp4"
                    h, w = ep_frames[0].shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        control_hz, (w, h),
                    )
                    for frame in ep_frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()

                successes.append(ep_success)
                picked_list.append(ep_picked)
                moved_list.append(ep_moved)
                min_dists.append(ep_min_dist if ep_min_dist != math.inf else 999.0)
                episodes_run += 1

                status = "SUCCESS" if ep_success else ("moved" if ep_moved else ("picked" if ep_picked else "fail"))
                print(f"  ep{ep}: {status}  dist={ep_min_dist:.3f}  spawn=({block_start[0]:.3f}, {block_start[1]:.3f}, {block_start[2]:.3f})", flush=True)

        finally:
            try:
                backend.close()
            except Exception:
                pass

        n = max(len(successes), 1)
        any_success = any(successes)
        success_rate = sum(successes) / n
        pick_rate = sum(picked_list) / n
        move_rate = sum(moved_list) / n
        overall_min_dist = min(min_dists) if min_dists else 999.0

        return {
            "success": any_success,
            "success_rate": success_rate,
            "pick_rate": pick_rate,
            "move_rate": move_rate,
            "min_dist": overall_min_dist,
            "episodes_run": episodes_run,
            "spawn_positions": spawn_positions,
        }

    except Exception as e:
        return {
            "success": False,
            "success_rate": 0.0,
            "min_dist": 0.0,
            "episodes_run": 0,
            "error": f"{type(e).__name__}: {e}",
        }
