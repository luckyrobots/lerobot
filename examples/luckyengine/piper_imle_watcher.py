"""
Checkpoint watcher for IMLE training → LuckyEngine evaluation.

Runs alongside training in a separate terminal.  Polls for new checkpoints,
evaluates them against the live LuckyEngine sim (same logic as
piper_imle_sweep.py), prioritises newest checkpoints, and backfills older ones.

Output is written to the same piper_imle_sweep.jsonl / .csv / videos_left/
format as the sweep script so the two can be mixed and matched.

Usage:
    python piper_imle_watcher.py \
        --checkpoints_root "path/to/session" \
        --port 50053 \
        --trials_per_checkpoint 3 \
        --poll_interval 30 \
        --output_dir "."
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

# ── Reuse everything possible from the sweep script ──────────────────────────
from piper_imle_sweep import (
    CheckpointResult,
    DEFAULT_RED_BLOCK_RESET_POS_CSV,
    _build_policy_observation_from_scene_observation,
    _checkpoint_step,
    _coerce_state_to_policy_dim,
    _extract_policy_camera_keys_from_config,
    _flip_image_hwc_u8,
    _format_exception_short,
    _frame_is_all_black_u8_rgb,
    _parse_csv_list,
    _parse_policy_to_scene_camera_map,
    _parse_quat_csv,
    _parse_vec3_csv,
    _random_yaw_quat,
    _reset_robot_and_block,
    _sample_randomized_block_pos,
    _set_entity_pose_keep_scale,
    _wait_for_nonblack_camera_set,
    autodetect_hazel_port,
    discover_pretrained_models,
)

from lerobot.luckyengine.contracts import (
    DEFAULT_PIPER_CONTRACT,
    LuckyEnginePolicyContract,
    validate_checkpoint_contract,
)
from lerobot.luckyengine.hazel_backend import HazelBackend, HazelStreamConfig, make_dummy_rgb_image
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig


# ── Watcher-specific helpers ─────────────────────────────────────────────────

def _is_checkpoint_ready(pm_dir: Path, min_age_s: float = 5.0) -> bool:
    """Return True when the checkpoint has all required files and is old enough.

    Avoids reading a checkpoint that is still being written by the training
    process.  We check:
      1. config.json exists
      2. model.safetensors exists
      3. model.safetensors was last modified at least *min_age_s* seconds ago
    """
    config_f = pm_dir / "config.json"
    model_f = pm_dir / "model.safetensors"
    if not config_f.is_file() or not model_f.is_file():
        return False
    try:
        age = time.time() - model_f.stat().st_mtime
        return age >= min_age_s
    except OSError:
        return False


def _load_completed_set(
    jsonl_path: Path,
    trials_per_checkpoint: int,
    *,
    include_errors: bool = False,
) -> set[tuple[str, int]]:
    """Read the JSONL log and return {(checkpoint_name, trial_num)} for done trials.

    By default, errored trials are *not* included (they will be retried).
    Pass ``include_errors=True`` to skip retrying them.
    """
    done: set[tuple[str, int]] = set()
    if not jsonl_path.is_file():
        return done
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ckpt = rec.get("checkpoint", "")
                trial = rec.get("trial")
                has_error = bool(rec.get("error"))
                if trial is None:
                    continue
                if has_error and not include_errors:
                    continue
                done.add((ckpt, int(trial)))
    except OSError:
        pass
    return done


def _prioritized_eval_queue(
    all_dirs: list[Path],
    completed: set[tuple[str, int]],
    trials_per_checkpoint: int,
) -> list[tuple[Path, int]]:
    """Return ``(pm_dir, trial_num)`` pairs sorted newest-checkpoint-first.

    ``trial_num`` is 1-based (matches sweep convention).  Pairs already in
    *completed* are skipped.
    """
    # Sort by step number descending (newest first).
    def _step_key(d: Path) -> int:
        s = _checkpoint_step(d)
        return s if s is not None else 0

    dirs_newest_first = sorted(all_dirs, key=_step_key, reverse=True)

    queue: list[tuple[Path, int]] = []
    for pm_dir in dirs_newest_first:
        ckpt_name = pm_dir.parent.name
        for trial in range(1, trials_per_checkpoint + 1):
            if (ckpt_name, trial) not in completed:
                queue.append((pm_dir, trial))
    return queue


def _regenerate_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Re-read the full JSONL and write a ranked CSV (same ordering as sweep)."""
    results: list[dict] = []
    if not jsonl_path.is_file():
        return
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return
    if not results:
        return

    # Rank: success first → fewer steps_to_success → smaller min_dist
    def _rank(r: dict):
        return (
            0 if r.get("success") else 1,
            r.get("steps_to_success") if r.get("steps_to_success") is not None else 10**9,
            r.get("min_dist_m", math.inf),
        )

    results.sort(key=_rank)
    fieldnames = list(results[0].keys())
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r)
    except OSError as e:
        print(f"[warn] Could not write CSV: {e}")


def _reconnect_backend(
    host: str,
    port: int,
    proto_path: str | None,
    *,
    max_retries: int = 10,
) -> HazelBackend:
    """Reconnect to Hazel with exponential backoff."""
    delay = 5.0
    for attempt in range(1, max_retries + 1):
        print(f"  [reconnect] Attempt {attempt}/{max_retries} (waiting {delay:.1f}s)...")
        time.sleep(delay)
        try:
            backend = HazelBackend(host=host, port=port, proto_path=proto_path)
            backend.connect()
            backend.get_scene_info()
            print(f"  [reconnect] Success on attempt {attempt}.")
            return backend
        except Exception as e:
            print(f"  [reconnect] Failed: {e}")
        delay = min(delay * 2, 60.0)
    raise RuntimeError(f"Could not reconnect to Hazel after {max_retries} attempts.")


# ── Single-trial evaluation (extracted from sweep inner loop) ────────────────

def _evaluate_single_trial(
    *,
    policy,
    preproc,
    postproc,
    backend: HazelBackend,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_name: str,
    pm_dir: Path,
    attempt_num: int,
    # Scene state
    policy_to_scene_camera: dict[str, str],
    scene_camera_map: dict[str, str],
    scene_camera_names: list[str],
    policy_camera_keys: list[str],
    red_block_id: int,
    red_block_initial_transform,
    forced_object_pos: tuple[float, float, float] | None,
    forced_object_quat: tuple[float, float, float, float] | None,
    goal_center_offset: tuple[float, float, float],
    spawn_rng: np.random.Generator | None,
    base_pos_np: np.ndarray | None,
    dropbox_pos_np: np.ndarray | None,
    output_dir: Path,
    trials_per_ckpt: int,
) -> CheckpointResult:
    """Run one trial episode and return the result."""
    do_reset = not args.no_reset
    do_video = not args.no_video and not args.dummy_images

    policy.reset()

    # ── Per-trial reset ──
    if do_reset:
        trial_object_pos = forced_object_pos
        trial_object_quat = forced_object_quat
        if args.randomize_block and spawn_rng is not None and forced_object_pos is not None:
            trial_object_pos = _sample_randomized_block_pos(
                base_pos=base_pos_np,
                dropbox_pos=dropbox_pos_np,
                default_y=forced_object_pos[1],
                rng=spawn_rng,
            )
            trial_object_quat = _random_yaw_quat(spawn_rng)
            print(
                f"  Randomized block pos: "
                f"({trial_object_pos[0]:.4f}, {trial_object_pos[1]:.4f}, {trial_object_pos[2]:.4f})"
            )

        print("  Resetting robot and block...")
        _reset_robot_and_block(
            backend,
            red_block_id=red_block_id,
            red_block_initial_transform=red_block_initial_transform,
            forced_object_pos=trial_object_pos,
            forced_object_quat=trial_object_quat,
            home_action=args.home_action,
            robot_name=args.robot_name,
            reset_settle_s=float(args.reset_settle_s),
            reset_hz=float(args.reset_hz),
            agent_name=args.agent_name,
            object_tag=args.object_tag,
            debug_reset=bool(args.debug_reset),
            object_reset_hold_s=float(args.object_reset_hold_s),
            object_reset_hold_hz=float(args.object_reset_hold_hz),
        )

    min_dist = math.inf
    success = False
    steps_to_success = None
    err: str | None = None
    steps_taken = 0
    last_dist: float | None = None
    state_shape_warned = False

    dt_target = 1.0 / max(1e-6, float(args.control_hz))
    next_t = time.perf_counter()
    video_writer = None
    black_frame_warned = False

    if do_video:
        videos_dir = output_dir / "videos_left"
        videos_dir.mkdir(parents=True, exist_ok=True)
        trial_suffix = f"_trial{attempt_num}" if trials_per_ckpt > 1 else ""
        video_path = videos_dir / f"{ckpt_name}{trial_suffix}_left.mp4"
        try:
            video_writer = imageio.get_writer(str(video_path), fps=float(args.control_hz))
        except Exception as e:
            print(f"[warn] Could not open video writer for {video_path}: {e}")
            video_writer = None

    try:
        prev_qpos = None
        for step in range(int(args.max_steps)):
            steps_taken = step + 1
            if args.dummy_images:
                raw_state = backend.get_observation(camera_map={})["observation.state"]
                state_fixed, state_note = _coerce_state_to_policy_dim(
                    state=raw_state,
                    expected_dim=DEFAULT_PIPER_CONTRACT.state_dim,
                    backend=backend,
                    robot_name=args.robot_name,
                )
                if state_note is not None and not state_shape_warned:
                    print(f"  [warn] {state_note}")
                    state_shape_warned = True
                obs = {"observation.state": state_fixed}
                for cam in policy_camera_keys:
                    obs[f"observation.images.{cam}"] = make_dummy_rgb_image(
                        int(args.camera_width), int(args.camera_height)
                    )
            else:
                obs_scene = backend.get_observation(camera_map=scene_camera_map)
                obs = _build_policy_observation_from_scene_observation(
                    obs_scene=obs_scene,
                    policy_to_scene_camera=policy_to_scene_camera,
                    flip_mode=args.flip_for_policy,
                )
                state_fixed, state_note = _coerce_state_to_policy_dim(
                    state=obs_scene["observation.state"],
                    expected_dim=DEFAULT_PIPER_CONTRACT.state_dim,
                    backend=backend,
                    robot_name=args.robot_name,
                )
                obs["observation.state"] = state_fixed
                if state_note is not None and not state_shape_warned:
                    print(f"  [warn] {state_note}")
                    state_shape_warned = True
                if not bool(args.allow_black_images):
                    if not _wait_for_nonblack_camera_set(
                        backend,
                        camera_map=scene_camera_map,
                        camera_keys=scene_camera_names,
                        timeout_s=float(args.black_image_timeout_s),
                        poll_s=float(args.black_image_poll_s),
                    ):
                        raise RuntimeError(
                            "Camera frames are all-black (all zeros) and did not recover "
                            "within timeout. Aborting trial to avoid evaluating on invalid observations."
                        )
                # Record left camera frame for video
                if video_writer is not None:
                    frame_key = f"observation.images.{args.left_camera_name}"
                    frame = obs.get(frame_key)
                    if frame is not None:
                        if not black_frame_warned and _frame_is_all_black_u8_rgb(frame):
                            black_frame_warned = True
                            print(
                                "  [warn] Camera frames are all-black (all zeros). "
                                "This usually means LuckyEngine isn't rendering / camera "
                                "streaming is stale."
                            )
                        if bool(args.allow_black_images) or not _frame_is_all_black_u8_rgb(frame):
                            frame_flipped = np.flip(frame, axis=0)
                            video_writer.append_data(frame_flipped)

            # Resize images to 96x96 to match training resolution
            for _k in list(obs.keys()):
                if "image" in _k:
                    _img = torch.from_numpy(obs[_k]).permute(2, 0, 1).unsqueeze(0).float()
                    _img = torch.nn.functional.interpolate(
                        _img, size=(96, 96), mode="bilinear", align_corners=False
                    )
                    obs[_k] = _img.squeeze(0).permute(1, 2, 0).byte().numpy()

            action_t = predict_action(
                observation=obs,
                policy=policy,
                device=device,
                preprocessor=preproc,
                postprocessor=postproc,
                use_amp=bool(args.use_amp),
            )
            action = action_t.squeeze(0).detach().to("cpu").numpy().astype(np.float32).tolist()
            backend.step(action, agent_name=args.agent_name)

            if (
                bool(args.debug_joint_state)
                and int(args.debug_joint_every) > 0
                and (step % int(args.debug_joint_every) == 0)
            ):
                qpos, qvel = backend.get_joint_state(robot_name=args.robot_name)
                if prev_qpos is None:
                    prev_qpos = qpos
                    dq = np.zeros_like(qpos)
                else:
                    dq = qpos - prev_qpos
                    prev_qpos = qpos
                dq_max = float(np.max(np.abs(dq))) if dq.size else 0.0
                qvel_max = float(np.max(np.abs(qvel))) if qvel.size else 0.0
                a_max = (
                    float(np.max(np.abs(np.asarray(action, dtype=np.float32))))
                    if action
                    else 0.0
                )
                print(
                    f"  [debug] step={step:04d} | max|dqpos|={dq_max:.6f}  "
                    f"max|qvel|={qvel_max:.6f}  max|action|={a_max:.4f}"
                )

            obj = backend.get_entity_position(tag=args.object_tag)
            goal = backend.get_entity_position(tag=args.goal_tag) + np.asarray(
                goal_center_offset, dtype=np.float32
            )
            dist = float(np.linalg.norm(obj - goal))
            last_dist = dist
            min_dist = min(min_dist, dist)
            if dist <= float(args.success_dist_m):
                success = True
                steps_to_success = step + 1
                break

            next_t += dt_target
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as e:
        err = _format_exception_short(e)
        print(
            f"  [ERROR] Checkpoint {ckpt_name} attempt {attempt_num} "
            f"crashed at step {steps_taken}: {err}"
        )
    finally:
        with contextlib.suppress(Exception):
            if video_writer is not None:
                video_writer.close()

    return CheckpointResult(
        checkpoint=ckpt_name,
        pretrained_model_dir=str(pm_dir),
        trial=attempt_num,
        success=success,
        steps=steps_taken,
        steps_to_success=steps_to_success,
        min_dist_m=float(min_dist if min_dist != math.inf else 0.0),
        final_dist_m=last_dist,
        error=err,
    )


# ── Main watcher loop ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Checkpoint watcher: continuously evaluates new IMLE checkpoints "
            "against LuckyEngine Piper-room via Hazel gRPC."
        ),
    )

    # ── All sweep-compatible arguments ──
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="Hazel gRPC port (0 = auto-detect)")
    ap.add_argument("--proto_path", type=str, default=None)

    ap.add_argument(
        "--checkpoints_root", type=str, required=True,
        help="Either <session>/checkpoints or <session> containing checkpoints/",
    )

    ap.add_argument("--agent_name", default="agent_0")
    ap.add_argument("--robot_name", default="")

    ap.add_argument("--control_hz", type=float, default=30.0)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--camera_width", type=int, default=320)
    ap.add_argument("--camera_height", type=int, default=240)
    ap.add_argument("--camera_fps", type=int, default=30)
    ap.add_argument("--policy_cameras", type=str, default="")
    ap.add_argument("--camera_map", type=str, default="")
    ap.add_argument(
        "--flip_for_policy", choices=["none", "h", "v", "hv"], default="v",
    )
    ap.add_argument("--dummy_images", action="store_true")

    ap.add_argument("--object_tag", default="Red Block")
    ap.add_argument("--object_reset_pos", type=str, default=DEFAULT_RED_BLOCK_RESET_POS_CSV)
    ap.add_argument("--object_reset_quat", type=str, default="0,0,0,1")
    ap.add_argument("--goal_tag", default="Dropbox")
    ap.add_argument("--goal_center_offset", type=str, default="0,0,0")
    ap.add_argument("--success_dist_m", type=float, default=0.05)

    ap.add_argument("--no_reset", action="store_true")
    ap.add_argument(
        "--home_action", type=float, nargs=7,
        default=[0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035],
    )
    ap.add_argument("--reset_settle_s", type=float, default=1.5)
    ap.add_argument("--reset_hz", type=float, default=30.0)

    ap.add_argument("--randomize_block", action="store_true")

    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument(
        "--step_multiple", type=int, default=0,
        help="If >0, only evaluate checkpoints whose step is a multiple of this value.",
    )
    ap.add_argument(
        "--min_step", type=int, default=0,
        help="If >0, only evaluate checkpoints whose step >= this value.",
    )
    ap.add_argument("--trials_per_checkpoint", type=int, default=1)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--debug_reset", action="store_true")
    ap.add_argument("--object_reset_hold_s", type=float, default=0.25)
    ap.add_argument("--object_reset_hold_hz", type=float, default=30.0)
    ap.add_argument("--left_camera_name", type=str, default="CameraLeft")
    ap.add_argument("--allow_black_images", action="store_true")
    ap.add_argument("--black_image_timeout_s", type=float, default=5.0)
    ap.add_argument("--black_image_poll_s", type=float, default=0.05)
    ap.add_argument("--startup_grace_s", type=float, default=2.0)
    ap.add_argument("--debug_joint_state", action="store_true")
    ap.add_argument("--debug_joint_every", type=int, default=30)

    # ── Watcher-specific arguments ──
    ap.add_argument(
        "--poll_interval", type=float, default=30.0,
        help="Seconds between directory polls when no pending work.",
    )
    ap.add_argument(
        "--min_checkpoint_age_s", type=float, default=5.0,
        help="Min file age (seconds) before considering a checkpoint ready.",
    )
    ap.add_argument(
        "--max_reconnect_retries", type=int, default=10,
        help="Max gRPC reconnection attempts before giving up.",
    )
    ap.add_argument(
        "--retry_errors", action="store_true",
        help="Re-attempt previously errored trials (default: skip them).",
    )

    args = ap.parse_args()

    # ── Graceful shutdown on Ctrl+C ──
    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n[watcher] Force quit.")
            sys.exit(1)
        shutdown_requested = True
        print("\n[watcher] Shutdown requested (Ctrl+C again to force)...")

    signal.signal(signal.SIGINT, _signal_handler)

    device = torch.device(args.device)
    trials_per_ckpt = max(1, int(args.trials_per_checkpoint))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "piper_imle_sweep.jsonl"
    csv_path = output_dir / "piper_imle_sweep.csv"

    # ── Connect to Hazel ──
    port = int(args.port)
    if port == 0:
        port = autodetect_hazel_port(args.host)
        print(f"[watcher] Auto-detected Hazel gRPC port: {port}")

    backend = HazelBackend(host=args.host, port=port, proto_path=args.proto_path)
    backend.connect()

    try:
        backend.get_scene_info()
    except Exception as e:
        msg = str(e)
        if "unknown service hazel.rpc.v1.SceneService" in msg:
            raise RuntimeError(
                f"Connected to {args.host}:{port} but it does not expose Hazel SceneService. "
                "Use --port 0 (auto-detect) or set --port to the one shown in LuckyEditor gRPC panel."
            ) from e
        raise

    # ── Snapshot initial scene state ──
    red_block_entity = backend.get_entity(tag=args.object_tag)
    red_block_id = int(red_block_entity.id.id)
    red_block_initial_transform = red_block_entity.transform

    forced_object_pos: tuple[float, float, float] | None = None
    forced_object_quat: tuple[float, float, float, float] | None = None
    if str(args.object_reset_pos).strip():
        forced_object_pos = _parse_vec3_csv(str(args.object_reset_pos))
    if str(args.object_reset_quat).strip():
        forced_object_quat = _parse_quat_csv(str(args.object_reset_quat))

    if forced_object_pos is not None:
        for _ in range(2):
            _set_entity_pose_keep_scale(
                backend,
                entity_id=red_block_id,
                transform=red_block_initial_transform,
                position_xyz=forced_object_pos,
                quat_xyzw=forced_object_quat,
            )
            time.sleep(0.03)
        red_block_initial_transform = backend.get_entity(tag=args.object_tag).transform

    goal_center_offset = _parse_vec3_csv(str(args.goal_center_offset))

    # ── Block randomization setup ──
    spawn_rng: np.random.Generator | None = None
    base_pos_np: np.ndarray | None = None
    dropbox_pos_np: np.ndarray | None = None
    if args.randomize_block:
        spawn_rng = np.random.default_rng()
        try:
            base_pos_np = backend.get_entity_position(tag="piper")
        except Exception:
            base_pos_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            print("  [warn] Could not find 'piper' entity; using origin as robot base.")
        try:
            dropbox_pos_np = backend.get_entity_position(tag=args.goal_tag)
        except Exception:
            dropbox_pos_np = None
            print(f"  [warn] Could not find '{args.goal_tag}' entity; dropbox avoidance disabled.")

    # ── Camera setup (done once, reused across all checkpoints) ──
    # We need at least one checkpoint to auto-detect cameras.  Wait for it.
    print(f"[watcher] Waiting for first checkpoint under {args.checkpoints_root}...")
    first_pm_dir: Path | None = None
    while not shutdown_requested:
        ckpt_dirs = discover_pretrained_models(Path(args.checkpoints_root))
        ready = [d for d in ckpt_dirs if _is_checkpoint_ready(d, args.min_checkpoint_age_s)]
        if ready:
            first_pm_dir = ready[-1]  # newest ready
            break
        time.sleep(min(5.0, args.poll_interval))

    if shutdown_requested or first_pm_dir is None:
        print("[watcher] Shutting down (no checkpoints found).")
        backend.close()
        return 0

    policy_camera_keys = (
        _parse_csv_list(args.policy_cameras)
        if str(args.policy_cameras).strip()
        else _extract_policy_camera_keys_from_config(first_pm_dir)
    )
    if not policy_camera_keys:
        policy_camera_keys = list(DEFAULT_PIPER_CONTRACT.camera_keys)

    policy_to_scene_camera = _parse_policy_to_scene_camera_map(
        policy_camera_keys=policy_camera_keys,
        mapping_csv=args.camera_map,
    )
    scene_camera_names = sorted(set(policy_to_scene_camera.values()))
    scene_camera_map = {cam: cam for cam in scene_camera_names}

    cam_cfg = HazelStreamConfig(
        target_fps=int(args.camera_fps),
        width=int(args.camera_width),
        height=int(args.camera_height),
        format="raw",
    )
    if not args.dummy_images:
        for cam in scene_camera_names:
            backend.start_camera_stream(camera_name=cam, cfg=cam_cfg)
    backend.start_agent_stream(agent_name=args.agent_name, target_fps=int(args.camera_fps))

    if not args.dummy_images:
        backend.wait_for_first_obs(camera_names=scene_camera_names, timeout_s=15.0)
        if not bool(args.allow_black_images):
            grace_s = max(0.0, float(args.startup_grace_s))
            if grace_s > 0.0:
                print(f"  [info] Startup grace: sleeping {grace_s:.2f}s to let rendering stabilize...")
                time.sleep(grace_s)
            if not _wait_for_nonblack_camera_set(
                backend,
                camera_map=scene_camera_map,
                camera_keys=scene_camera_names,
                timeout_s=float(args.black_image_timeout_s),
                poll_s=float(args.black_image_poll_s),
            ):
                raise RuntimeError(
                    "Initial camera frames are all-black. "
                    "Fix LuckyEngine rendering, increase --startup_grace_s / --black_image_timeout_s, "
                    "or rerun with --allow_black_images."
                )

    # ── Print configuration ──
    do_reset = not args.no_reset
    do_video = not args.no_video and not args.dummy_images
    print(f"\n{'='*80}")
    print(f"  IMLE Checkpoint Watcher")
    print(f"{'='*80}")
    print(f"  Checkpoints root : {args.checkpoints_root}")
    print(f"  Poll interval    : {args.poll_interval}s")
    print(f"  Min checkpoint age: {args.min_checkpoint_age_s}s")
    print(f"  Control Hz       : {args.control_hz}")
    print(f"  Max steps        : {args.max_steps}")
    print(f"  Trials/checkpoint: {trials_per_ckpt}")
    print(f"  Device           : {device}")
    print(f"  Reset each trial : {do_reset}")
    print(f"  Record video     : {do_video}")
    print(f"  Flip for policy  : {args.flip_for_policy}")
    print(f"  Policy cameras   : {policy_camera_keys}")
    print(f"  Scene cameras    : {scene_camera_names}")
    print(f"  Retry errors     : {args.retry_errors}")
    print(f"  Output dir       : {output_dir}")
    print(f"  JSONL            : {jsonl_path}")
    print(f"  CSV              : {csv_path}")
    print(f"{'='*80}\n")

    # ── Main poll loop ──
    total_evaluated = 0
    total_successes = 0

    while not shutdown_requested:
        # 1. Discover all checkpoints
        ckpt_dirs = discover_pretrained_models(Path(args.checkpoints_root))

        # 2. Filter by step_multiple, min_step
        if args.step_multiple and args.step_multiple > 0:
            m = int(args.step_multiple)
            ckpt_dirs = [
                d for d in ckpt_dirs
                if ((s := _checkpoint_step(d)) is not None and s % m == 0)
            ]
        if args.min_step and args.min_step > 0:
            ms = int(args.min_step)
            ckpt_dirs = [
                d for d in ckpt_dirs
                if ((s := _checkpoint_step(d)) is not None and s >= ms)
            ]

        # 3. Filter by readiness
        ckpt_dirs = [
            d for d in ckpt_dirs
            if _is_checkpoint_ready(d, args.min_checkpoint_age_s)
        ]

        if not ckpt_dirs:
            print(f"[watcher] No ready checkpoints found. Sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)
            continue

        # 4. Load completed set from JSONL
        completed = _load_completed_set(
            jsonl_path, trials_per_ckpt,
            include_errors=not args.retry_errors,
        )

        # 5. Build prioritised queue (newest first)
        queue = _prioritized_eval_queue(ckpt_dirs, completed, trials_per_ckpt)

        if not queue:
            print(
                f"[watcher] All {len(ckpt_dirs)} checkpoints evaluated "
                f"({len(completed)} trials done). Sleeping {args.poll_interval}s..."
            )
            time.sleep(args.poll_interval)
            continue

        print(f"[watcher] {len(queue)} pending trial(s) across {len(ckpt_dirs)} checkpoints.")

        # 6. Evaluate pending trials
        current_policy_path: str | None = None

        for pm_dir, attempt_num in queue:
            if shutdown_requested:
                break

            ckpt_name = pm_dir.parent.name
            step_num = _checkpoint_step(pm_dir)
            step_str = f"step {step_num}" if step_num is not None else "?"

            print(f"\n{'='*80}")
            print(
                f"  EVAL  |  {ckpt_name} ({step_str})  |  "
                f"trial {attempt_num}/{trials_per_ckpt}"
            )
            print(f"{'='*80}")

            # Load policy (only reload if checkpoint changed)
            try:
                if str(pm_dir) != current_policy_path:
                    # Validate checkpoint contract
                    ckpt_camera_keys = _extract_policy_camera_keys_from_config(pm_dir)
                    if set(ckpt_camera_keys) != set(policy_camera_keys):
                        print(
                            f"  [skip] Camera mismatch: checkpoint expects {ckpt_camera_keys}, "
                            f"watcher configured with {policy_camera_keys}."
                        )
                        continue
                    validate_checkpoint_contract(
                        pm_dir,
                        contract=LuckyEnginePolicyContract(
                            action_dim=DEFAULT_PIPER_CONTRACT.action_dim,
                            state_dim=DEFAULT_PIPER_CONTRACT.state_dim,
                            camera_keys=tuple(policy_camera_keys),
                        ),
                    )

                    cfg = PreTrainedConfig.from_pretrained(str(pm_dir))
                    policy_cls = get_policy_class(cfg.type)
                    policy = policy_cls.from_pretrained(pretrained_name_or_path=str(pm_dir))
                    preproc, postproc = make_pre_post_processors(cfg, pretrained_path=str(pm_dir))
                    policy.to(device)
                    policy.eval()
                    current_policy_path = str(pm_dir)
                    print(f"  Loaded policy from {pm_dir}")
            except Exception as e:
                err_msg = _format_exception_short(e)
                print(f"  [ERROR] Failed to load checkpoint {ckpt_name}: {err_msg}")
                # Record error result so we don't retry immediately
                res = CheckpointResult(
                    checkpoint=ckpt_name,
                    pretrained_model_dir=str(pm_dir),
                    trial=attempt_num,
                    success=False,
                    steps=0,
                    steps_to_success=None,
                    min_dist_m=0.0,
                    final_dist_m=None,
                    error=err_msg,
                )
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(res)) + "\n")
                _regenerate_csv(jsonl_path, csv_path)
                continue

            # Run trial
            try:
                res = _evaluate_single_trial(
                    policy=policy,
                    preproc=preproc,
                    postproc=postproc,
                    backend=backend,
                    device=device,
                    args=args,
                    ckpt_name=ckpt_name,
                    pm_dir=pm_dir,
                    attempt_num=attempt_num,
                    policy_to_scene_camera=policy_to_scene_camera,
                    scene_camera_map=scene_camera_map,
                    scene_camera_names=scene_camera_names,
                    policy_camera_keys=policy_camera_keys,
                    red_block_id=red_block_id,
                    red_block_initial_transform=red_block_initial_transform,
                    forced_object_pos=forced_object_pos,
                    forced_object_quat=forced_object_quat,
                    goal_center_offset=goal_center_offset,
                    spawn_rng=spawn_rng,
                    base_pos_np=base_pos_np,
                    dropbox_pos_np=dropbox_pos_np,
                    output_dir=output_dir,
                    trials_per_ckpt=trials_per_ckpt,
                )
            except Exception as e:
                # Connection-level failure → try reconnecting
                err_msg = _format_exception_short(e)
                print(f"  [ERROR] Trial-level crash: {err_msg}")
                print("  [watcher] Attempting to reconnect to Hazel...")
                try:
                    backend.close()
                except Exception:
                    pass
                try:
                    backend = _reconnect_backend(
                        args.host, port, args.proto_path,
                        max_retries=args.max_reconnect_retries,
                    )
                    # Re-snapshot scene state
                    red_block_entity = backend.get_entity(tag=args.object_tag)
                    red_block_id = int(red_block_entity.id.id)
                    red_block_initial_transform = red_block_entity.transform
                    if forced_object_pos is not None:
                        for _ in range(2):
                            _set_entity_pose_keep_scale(
                                backend,
                                entity_id=red_block_id,
                                transform=red_block_initial_transform,
                                position_xyz=forced_object_pos,
                                quat_xyzw=forced_object_quat,
                            )
                            time.sleep(0.03)
                        red_block_initial_transform = backend.get_entity(tag=args.object_tag).transform

                    # Restart streams
                    if not args.dummy_images:
                        for cam in scene_camera_names:
                            backend.start_camera_stream(camera_name=cam, cfg=cam_cfg)
                    backend.start_agent_stream(
                        agent_name=args.agent_name, target_fps=int(args.camera_fps)
                    )
                    if not args.dummy_images:
                        backend.wait_for_first_obs(
                            camera_names=scene_camera_names, timeout_s=15.0
                        )
                    print("  [watcher] Reconnected successfully. Resuming evaluation loop.")
                except Exception as re_err:
                    print(f"  [FATAL] Could not reconnect: {re_err}")
                    break
                # Record the error for this trial so it's not lost
                res = CheckpointResult(
                    checkpoint=ckpt_name,
                    pretrained_model_dir=str(pm_dir),
                    trial=attempt_num,
                    success=False,
                    steps=0,
                    steps_to_success=None,
                    min_dist_m=0.0,
                    final_dist_m=None,
                    error=err_msg,
                )
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(res)) + "\n")
                _regenerate_csv(jsonl_path, csv_path)
                continue

            # Append result
            total_evaluated += 1
            if res.success:
                total_successes += 1

            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res)) + "\n")
            _regenerate_csv(jsonl_path, csv_path)

            # Per-trial result log
            status_icon = "SUCCESS" if res.success else ("ERROR" if res.error else "FAIL")
            final_dist_s = "n/a" if res.final_dist_m is None else f"{res.final_dist_m:.4f}m"
            print(f"  -------------------------------------------------------")
            print(
                f"  RESULT  |  {ckpt_name} ({step_str})  "
                f"|  trial {attempt_num}/{trials_per_ckpt}"
            )
            print(f"    Status      : {status_icon}")
            print(
                f"    Steps       : {res.steps}/{args.max_steps}"
                + (f"  (reached goal at step {res.steps_to_success})" if res.steps_to_success else "")
            )
            print(f"    Min distance: {res.min_dist_m:.4f}m   (threshold: {float(args.success_dist_m):.4f}m)")
            print(f"    Final dist  : {final_dist_s}")
            if res.error:
                print(f"    Error       : {res.error}")
            print(f"  -------------------------------------------------------")

            # Free GPU memory between checkpoints
            if str(pm_dir) == current_policy_path:
                # Check if next item in queue is a different checkpoint
                # (no need to free if same checkpoint, different trial)
                pass

        # End of queue pass — sleep before next poll
        if not shutdown_requested:
            print(
                f"\n[watcher] Poll pass done. "
                f"Evaluated {total_evaluated} trial(s) total ({total_successes} successes). "
                f"Sleeping {args.poll_interval}s..."
            )
            time.sleep(args.poll_interval)

    # ── Clean shutdown ──
    print(f"\n{'='*80}")
    print(f"  WATCHER SHUTDOWN")
    print(f"    Total trials evaluated: {total_evaluated}")
    print(f"    Total successes       : {total_successes}")
    print(f"    JSONL: {jsonl_path}")
    print(f"    CSV  : {csv_path}")
    print(f"{'='*80}")

    try:
        backend.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
