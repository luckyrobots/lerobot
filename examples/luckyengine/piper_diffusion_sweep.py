from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.luckyengine.contracts import DEFAULT_PIPER_CONTRACT, validate_checkpoint_contract
from lerobot.luckyengine.hazel_backend import HazelBackend, HazelStreamConfig, make_dummy_rgb_image
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig


# Block spawn position used for per-trial resets.
# Keep this in sync with your LuckyEditor Inspector (Translation X/Y/Z).
DEFAULT_RED_BLOCK_RESET_POS = (0.32, 0.05, 0.0)
DEFAULT_RED_BLOCK_RESET_POS_CSV = ",".join(str(float(x)) for x in DEFAULT_RED_BLOCK_RESET_POS)


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint: str
    pretrained_model_dir: str
    success: bool
    steps: int
    steps_to_success: int | None
    min_dist_m: float
    final_dist_m: float | None = None
    error: str | None = None


def _frame_is_all_black_u8_rgb(frame: np.ndarray) -> bool:
    """
    Heuristic: returns True if an HWC uint8 RGB frame is entirely zero.
    This catches the common failure mode where the camera stream returns a black buffer.
    """
    try:
        return frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3 and int(frame.max()) == 0
    except Exception:
        return False


def _wait_for_nonblack_camera_frame(
    backend: HazelBackend,
    *,
    camera_map: dict[str, str],
    camera_key: str,
    timeout_s: float = 3.0,
    poll_s: float = 0.05,
) -> bool:
    """
    Best-effort warmup: wait until a specific camera frame is not entirely black.
    Returns True if a non-black frame was observed before timeout, else False.
    """
    t0 = time.perf_counter()
    frame_key = f"observation.images.{camera_key}"
    while time.perf_counter() - t0 < float(timeout_s):
        obs = backend.get_observation(camera_map=camera_map)
        frame = obs.get(frame_key)
        if frame is not None and not _frame_is_all_black_u8_rgb(frame):
            return True
        time.sleep(float(poll_s))
    return False


def _wait_for_nonblack_camera_set(
    backend: HazelBackend,
    *,
    camera_map: dict[str, str],
    camera_keys: list[str],
    timeout_s: float = 2.0,
    poll_s: float = 0.05,
) -> bool:
    """
    Wait until *all* requested cameras produce a non-black frame.
    Returns True if all are non-black before timeout; otherwise False.
    """
    t0 = time.perf_counter()
    frame_keys = [f"observation.images.{k}" for k in camera_keys]
    while time.perf_counter() - t0 < float(timeout_s):
        obs = backend.get_observation(camera_map=camera_map)
        ok = True
        for fk in frame_keys:
            fr = obs.get(fk)
            if fr is None or _frame_is_all_black_u8_rgb(fr):
                ok = False
                break
        if ok:
            return True
        time.sleep(float(poll_s))
    return False


def _numeric_sort_key(p: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in p.name if ch.isdigit())
    return (int(digits) if digits else 10**18, p.name)


def _checkpoint_step(pm_dir: Path) -> int | None:
    """
    Extract numeric step from a checkpoint directory path like:
      .../checkpoints/002000/pretrained_model  -> 2000
    Returns None if no digits are present.
    """
    digits = "".join(ch for ch in pm_dir.parent.name if ch.isdigit())
    return int(digits) if digits else None


def discover_pretrained_models(checkpoints_root: Path) -> list[Path]:
    # Accept either:
    # - <session>/checkpoints
    # - <session> (containing checkpoints/)
    root = checkpoints_root
    if (root / "checkpoints").is_dir():
        root = root / "checkpoints"

    dirs = []
    for ckpt in sorted(root.glob("*"), key=_numeric_sort_key):
        # Many training runs create a `last` symlink/junction. On Windows this can raise
        # WinError 1920 (or similar) when pathlib tries to stat it. We treat it as optional.
        if ckpt.name.lower() == "last":
            continue
        pm = ckpt / "pretrained_model"
        try:
            if pm.is_dir():
                dirs.append(pm)
        except OSError as e:
            # Skip unreadable/broken entries instead of crashing the whole sweep.
            print(f"[warn] Skipping checkpoint entry due to filesystem error: {pm} ({e})")
            continue
    return dirs


def autodetect_hazel_port(host: str, start_port: int = 50051, end_port: int = 50100, timeout_s: float = 0.5) -> int:
    """
    Find a port that responds to Hazel's SceneService.GetSceneInfo.
    Useful on Windows where 50051 is often taken by another service.
    """
    import socket

    import grpc  # type: ignore

    from lerobot.luckyengine.grpc_stubs import default_proto_path, generate_python_stubs

    st = generate_python_stubs(str(default_proto_path()))
    pb2, pb2_grpc = st.pb2, st.pb2_grpc

    for port in range(int(start_port), int(end_port) + 1):
        # Fast-path: skip ports that aren't even accepting TCP connections.
        try:
            with socket.create_connection((host, int(port)), timeout=0.05):
                pass
        except OSError:
            continue

        ch = grpc.insecure_channel(f"{host}:{port}")
        scene = pb2_grpc.SceneServiceStub(ch)
        try:
            scene.GetSceneInfo(pb2.GetSceneInfoRequest(), timeout=float(timeout_s))
            return port
        except Exception:
            continue

    raise RuntimeError(
        f"Could not find a Hazel gRPC server on {host}:{start_port}-{end_port}. "
        "Ensure LuckyEditor gRPC Server is running and SceneService is enabled."
    )


def _parse_vec3_csv(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'x,y,z' format, got: {value!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_quat_csv(value: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 'x,y,z,w' format, got: {value!r}")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def _format_exception_short(e: Exception) -> str:
    """
    Short single-line error for logs/JSONL.
    """
    name = type(e).__name__
    msg = str(e).replace("\r", " ").replace("\n", " ").strip()
    if not msg:
        return name
    return f"{name}: {msg}"


def _set_entity_pose_keep_scale(
    backend: HazelBackend,
    *,
    entity_id: int,
    transform,
    position_xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float] | None,
) -> None:
    t = type(transform)()
    t.CopyFrom(transform)
    t.position.x = float(position_xyz[0])
    t.position.y = float(position_xyz[1])
    t.position.z = float(position_xyz[2])
    if quat_xyzw is not None:
        t.rotation.x = float(quat_xyzw[0])
        t.rotation.y = float(quat_xyzw[1])
        t.rotation.z = float(quat_xyzw[2])
        t.rotation.w = float(quat_xyzw[3])
    backend.set_entity_transform(entity_id=entity_id, transform=t)


def _reset_robot_and_block(
    backend: HazelBackend,
    *,
    red_block_id: int,
    red_block_initial_transform,
    forced_object_pos: tuple[float, float, float] | None,
    forced_object_quat: tuple[float, float, float, float] | None,
    home_action: list[float],
    robot_name: str,
    reset_settle_s: float,
    reset_hz: float,
    agent_name: str,
    object_tag: str,
    debug_reset: bool,
    object_reset_hold_s: float,
    object_reset_hold_hz: float,
) -> None:
    """Reset the block position and drive the robot to home pose."""
    # 1) Reset agent via gRPC first.
    # IMPORTANT: ResetAgent can reset MuJoCo state and may reposition scene entities.
    # If we set the block pose before ResetAgent, the reset may immediately overwrite it.
    try:
        backend.reset_agent(agent_name=agent_name)
    except Exception as e:
        print(f"  [warn] ResetAgent failed (non-fatal): {e}")

    # 2) Drive robot to home for settle window (reduces chances of the arm bumping the block on spawn).
    hold_dt = 1.0 / max(1e-6, float(reset_hz))
    t_end = time.perf_counter() + max(0.0, float(reset_settle_s))
    while time.perf_counter() < t_end:
        backend.send_control(list(map(float, home_action)), robot_name=robot_name)
        time.sleep(hold_dt)

    # 3) Reset block pose AFTER agent reset + home settle.
    # Do it twice (with a short delay) to avoid a race with the first physics step after reset.
    if forced_object_pos is not None:
        _set_entity_pose_keep_scale(
            backend,
            entity_id=red_block_id,
            transform=red_block_initial_transform,
            position_xyz=forced_object_pos,
            quat_xyzw=forced_object_quat,
        )
        time.sleep(0.03)
        _set_entity_pose_keep_scale(
            backend,
            entity_id=red_block_id,
            transform=red_block_initial_transform,
            position_xyz=forced_object_pos,
            quat_xyzw=forced_object_quat,
        )

        # 3b) If the scene has its own episode/block randomization, it may overwrite the pose
        # shortly after reset. Re-assert the desired pose for a brief hold window.
        hold_s = max(0.0, float(object_reset_hold_s))
        if hold_s > 0.0:
            dt = 1.0 / max(1e-6, float(object_reset_hold_hz))
            t_end = time.perf_counter() + hold_s
            while time.perf_counter() < t_end:
                _set_entity_pose_keep_scale(
                    backend,
                    entity_id=red_block_id,
                    transform=red_block_initial_transform,
                    position_xyz=forced_object_pos,
                    quat_xyzw=forced_object_quat,
                )
                time.sleep(dt)
    else:
        backend.set_entity_transform(entity_id=red_block_id, transform=red_block_initial_transform)

    # 4) Optional debug: confirm actual block pose after reset
    if debug_reset:
        try:
            p = backend.get_entity_position(tag=object_tag)
            print(f"  [debug] '{object_tag}' position after reset: {p.tolist()}")
        except Exception as e:
            print(f"  [debug] Could not query '{object_tag}' position after reset: {e}")

    # 5) Brief pause to let observations stabilize
    time.sleep(0.1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep diffusion checkpoints against LuckyEngine Piper-room via Hazel gRPC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="Hazel gRPC port (0 = auto-detect)")
    ap.add_argument("--proto_path", type=str, default=None, help="Optional explicit path to hazel_rpc.proto")

    ap.add_argument(
        "--checkpoints_root",
        type=str,
        required=True,
        help="Either <session>/checkpoints or <session> containing checkpoints/",
    )

    ap.add_argument("--agent_name", default="agent_0")
    ap.add_argument("--robot_name", default="")

    # Defaults match training dataset: 30 Hz, 320x240
    ap.add_argument("--control_hz", type=float, default=30.0)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_amp", action="store_true", help="Enable AMP for CUDA inference")

    ap.add_argument("--camera_width", type=int, default=320)
    ap.add_argument("--camera_height", type=int, default=240)
    ap.add_argument("--camera_fps", type=int, default=30)
    ap.add_argument(
        "--dummy_images",
        action="store_true",
        help="Use black dummy images instead of streaming cameras (smoke-test only).",
    )

    ap.add_argument("--object_tag", default="Red Block")
    ap.add_argument(
        "--object_reset_pos",
        type=str,
        default=DEFAULT_RED_BLOCK_RESET_POS_CSV,
        help=(
            "Force object XYZ right after connect and at each checkpoint reset. "
            "Default: script constant DEFAULT_RED_BLOCK_RESET_POS."
        ),
    )
    ap.add_argument(
        "--object_reset_quat",
        type=str,
        default="",
        help=(
            "Force object quaternion (x,y,z,w) right after connect and at each checkpoint reset. "
            "Set empty string to keep the scene snapshot orientation (recommended)."
        ),
    )
    # In Piper-room / Piper-Lego scenes, `Dropbox` is the physical box parent (made of 5 walls).
    # `BoxTarget` is typically just a marker. We default to the physical dropbox center.
    ap.add_argument("--goal_tag", default="Dropbox")
    ap.add_argument(
        "--goal_center_offset",
        type=str,
        default="0,0,0",
        help="Optional XYZ offset (meters) added to the goal entity position when computing distance, format 'x,y,z'.",
    )
    ap.add_argument("--success_dist_m", type=float, default=0.05)

    ap.add_argument(
        "--no_reset",
        action="store_true",
        help="Disable per-trial reset of robot and block (NOT recommended).",
    )
    ap.add_argument(
        "--home_action",
        type=float,
        nargs=7,
        default=[0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035],
        help="7D absolute home action for Piper actuators: joint1..joint6, gripper (meters).",
    )
    ap.add_argument(
        "--reset_settle_s",
        type=float,
        default=1.5,
        help="How long to hold home_action after reset (seconds).",
    )
    ap.add_argument(
        "--reset_hz",
        type=float,
        default=30.0,
        help="Send rate while holding home_action during reset.",
    )

    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N checkpoints")
    ap.add_argument(
        "--step_multiple",
        type=int,
        default=0,
        help="If >0, only evaluate checkpoints whose numeric step is a multiple of this value (e.g. 1000).",
    )
    ap.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="If >0, only evaluate checkpoints whose numeric step is >= this value (e.g. 18000).",
    )
    ap.add_argument(
        "--no_video",
        action="store_true",
        help="Disable per-trial left camera video recording.",
    )
    ap.add_argument(
        "--debug_reset",
        action="store_true",
        help="Print the block position after each trial reset (diagnostic).",
    )
    ap.add_argument(
        "--object_reset_hold_s",
        type=float,
        default=0.25,
        help="After each reset, keep re-applying the forced object pose for this many seconds (robust against scene-side randomization).",
    )
    ap.add_argument(
        "--object_reset_hold_hz",
        type=float,
        default=30.0,
        help="Rate used while re-applying the forced object pose during --object_reset_hold_s.",
    )
    ap.add_argument(
        "--left_camera_name",
        type=str,
        default="CameraLeft",
        help="Camera key to record for per-trial video.",
    )
    ap.add_argument(
        "--allow_black_images",
        action="store_true",
        help="Allow all-black camera frames (NOT recommended). If not set, black frames abort the trial.",
    )
    ap.add_argument(
        "--black_image_timeout_s",
        type=float,
        default=5.0,
        help="If camera frames are all-black, wait up to this many seconds for non-black frames before aborting.",
    )
    ap.add_argument(
        "--black_image_poll_s",
        type=float,
        default=0.05,
        help="Polling interval while waiting for non-black frames.",
    )
    ap.add_argument(
        "--startup_grace_s",
        type=float,
        default=2.0,
        help=(
            "Extra grace period (seconds) to wait after the first observations arrive before enforcing the "
            "non-black camera requirement. Useful while LuckyEngine finishes renderer/shader initialization."
        ),
    )
    ap.add_argument(
        "--debug_joint_state",
        action="store_true",
        help="Print MuJoCo joint state deltas (diagnostic: helps confirm robot is actually moving).",
    )
    ap.add_argument(
        "--debug_joint_every",
        type=int,
        default=30,
        help="If --debug_joint_state, print every N control steps.",
    )
    args = ap.parse_args()

    ckpt_dirs = discover_pretrained_models(Path(args.checkpoints_root))

    if args.step_multiple and args.step_multiple > 0:
        m = int(args.step_multiple)
        ckpt_dirs = [d for d in ckpt_dirs if ((s := _checkpoint_step(d)) is not None and s % m == 0)]

    if args.min_step and args.min_step > 0:
        ms = int(args.min_step)
        ckpt_dirs = [d for d in ckpt_dirs if ((s := _checkpoint_step(d)) is not None and s >= ms)]

    if args.limit and args.limit > 0:
        ckpt_dirs = ckpt_dirs[: int(args.limit)]
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found under {args.checkpoints_root}")

    device = torch.device(args.device)
    do_reset = not args.no_reset
    do_video = not args.no_video and not args.dummy_images

    # Print configuration summary
    print(f"\n  Configuration:")
    print(f"    Control Hz     : {args.control_hz}")
    print(f"    Camera         : {args.camera_width}x{args.camera_height} @ {args.camera_fps}fps")
    print(f"    Max steps      : {args.max_steps}")
    print(f"    Reset each trial: {do_reset}")
    print(f"    Record video   : {do_video}")
    print(f"    Device         : {device}")
    print(f"    Checkpoints    : {len(ckpt_dirs)}")
    if bool(args.debug_joint_state):
        print(f"    Debug joint state: True (every {int(args.debug_joint_every)} steps)")

    port = int(args.port)
    if port == 0:
        port = autodetect_hazel_port(args.host)
        print(f"[info] Auto-detected Hazel gRPC port: {port}")

    backend = HazelBackend(host=args.host, port=port, proto_path=args.proto_path)
    backend.connect()

    # Fail fast with a clear message if user points at a non-Hazel gRPC server (common on Windows with 50051).
    try:
        backend.get_scene_info()
    except Exception as e:
        msg = str(e)
        if "unknown service hazel.rpc.v1.SceneService" in msg:
            raise RuntimeError(
                f"Connected to {args.host}:{port} but it does not expose Hazel SceneService. "
                "This usually means you're hitting another local service on that port. "
                "Use --port 0 (auto-detect) or set --port to the one shown in LuckyEditor gRPC panel."
            ) from e
        raise

    # Snapshot initial transforms so we can restore them before each checkpoint evaluation.
    # This is the "default" for this sweep run (i.e., the scene state at sweep start).
    red_block_entity = backend.get_entity(tag=args.object_tag)
    red_block_id = int(red_block_entity.id.id)
    red_block_initial_transform = red_block_entity.transform

    forced_object_pos: tuple[float, float, float] | None = None
    forced_object_quat: tuple[float, float, float, float] | None = None
    if str(args.object_reset_pos).strip():
        forced_object_pos = _parse_vec3_csv(str(args.object_reset_pos))
    if str(args.object_reset_quat).strip():
        forced_object_quat = _parse_quat_csv(str(args.object_reset_quat))
    print(f"    Object reset pos: {forced_object_pos if forced_object_pos is not None else 'disabled'}")
    print(f"    Object reset quat: {forced_object_quat if forced_object_quat is not None else 'keep scene snapshot'}")
    if forced_object_pos is not None:
        # Robustness: some scenes snap dynamic objects on initial Play.
        # Force once now and again per-checkpoint reset so every trial starts from the same pose.
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

    # Map Hazel camera entity tags -> LeRobot observation key suffix.
    # We keep them identical for this Piper setup.
    camera_map = {k: k for k in DEFAULT_PIPER_CONTRACT.camera_keys}

    cam_cfg = HazelStreamConfig(
        target_fps=int(args.camera_fps),
        width=int(args.camera_width),
        height=int(args.camera_height),
        format="raw",
    )
    if not args.dummy_images:
        for cam in DEFAULT_PIPER_CONTRACT.camera_keys:
            backend.start_camera_stream(camera_name=cam, cfg=cam_cfg)
    backend.start_agent_stream(agent_name=args.agent_name, target_fps=int(args.camera_fps))

    if not args.dummy_images:
        backend.wait_for_first_obs(camera_names=list(DEFAULT_PIPER_CONTRACT.camera_keys), timeout_s=15.0)
        # Warmup: some renderer/device states yield an initial all-black buffer; avoid recording
        # an entire trial of black frames if the stream recovers shortly after startup.
        if not bool(args.allow_black_images):
            grace_s = max(0.0, float(args.startup_grace_s))
            if grace_s > 0.0:
                print(f"  [info] Startup grace: sleeping {grace_s:.2f}s to let rendering stabilize...")
                time.sleep(grace_s)
            if not _wait_for_nonblack_camera_set(
                backend,
                camera_map=camera_map,
                camera_keys=list(DEFAULT_PIPER_CONTRACT.camera_keys),
                timeout_s=float(args.black_image_timeout_s),
                poll_s=float(args.black_image_poll_s),
            ):
                raise RuntimeError(
                    "Initial camera frames are all-black (all zeros). "
                    "LuckyEngine likely isn't producing valid camera pixels (missing resources/shaders, "
                    "not in Play/runtime, or a renderer/device issue). "
                    "Fix LuckyEngine rendering, increase --startup_grace_s / --black_image_timeout_s, "
                    "or rerun with --allow_black_images (diagnostic only)."
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "piper_diffusion_sweep.jsonl"
    csv_path = output_dir / "piper_diffusion_sweep.csv"

    results: list[CheckpointResult] = []

    for idx, pm_dir in enumerate(ckpt_dirs):
        ckpt_name = pm_dir.parent.name
        trial_num = idx + 1
        print(f"\n{'='*80}")
        print(f"  TRIAL {trial_num}/{len(ckpt_dirs)}  |  Checkpoint: {ckpt_name}  |  Path: {pm_dir}")
        print(f"{'='*80}")

        # ── Per-trial reset: robot to home + block to start pose ──
        if do_reset:
            print("  Resetting robot and block...")
            _reset_robot_and_block(
                backend,
                red_block_id=red_block_id,
                red_block_initial_transform=red_block_initial_transform,
                forced_object_pos=forced_object_pos,
                forced_object_quat=forced_object_quat,
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

        validate_checkpoint_contract(pm_dir, contract=DEFAULT_PIPER_CONTRACT)

        cfg = PreTrainedConfig.from_pretrained(str(pm_dir))
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(pretrained_name_or_path=str(pm_dir))
        preproc, postproc = make_pre_post_processors(cfg, pretrained_path=str(pm_dir))
        policy.to(device)
        policy.eval()
        policy.reset()

        min_dist = math.inf
        success = False
        steps_to_success = None
        err: str | None = None
        steps_taken = 0
        last_dist: float | None = None

        dt_target = 1.0 / max(1e-6, float(args.control_hz))
        next_t = time.perf_counter()
        video_writer = None
        black_frame_warned = False
        if do_video:
            videos_dir = output_dir / "videos_left"
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / f"{ckpt_name}_left.mp4"
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
                    obs = {"observation.state": backend.get_observation(camera_map={})["observation.state"]}
                    for cam in DEFAULT_PIPER_CONTRACT.camera_keys:
                        obs[f"observation.images.{cam}"] = make_dummy_rgb_image(int(args.camera_width), int(args.camera_height))
                else:
                    obs = backend.get_observation(camera_map=camera_map)
                    if not bool(args.allow_black_images):
                        # Ensure the policy never runs on all-black camera buffers.
                        if not _wait_for_nonblack_camera_set(
                            backend,
                            camera_map=camera_map,
                            camera_keys=list(DEFAULT_PIPER_CONTRACT.camera_keys),
                            timeout_s=float(args.black_image_timeout_s),
                            poll_s=float(args.black_image_poll_s),
                        ):
                            raise RuntimeError(
                                "Camera frames are all-black (all zeros) and did not recover within timeout. "
                                "Aborting trial to avoid evaluating on invalid observations."
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
                                    "This usually means LuckyEngine isn't rendering / camera streaming is stale. "
                                    "Common causes: not in Play mode, GPU/render device issue, or a camera stream crash."
                                )
                            # Never write all-black frames unless explicitly allowed.
                            if bool(args.allow_black_images) or not _frame_is_all_black_u8_rgb(frame):
                                video_writer.append_data(frame)

                action_t = predict_action(
                    observation=obs,
                    policy=policy,
                    device=device,
                    preprocessor=preproc,
                    postprocessor=postproc,
                    use_amp=bool(args.use_amp),
                )
                action = action_t.squeeze(0).detach().to("cpu").numpy().astype(np.float32).tolist()
                backend.send_control(action, robot_name=args.robot_name)

                if bool(args.debug_joint_state) and int(args.debug_joint_every) > 0 and (step % int(args.debug_joint_every) == 0):
                    qpos, qvel = backend.get_joint_state(robot_name=args.robot_name)
                    if prev_qpos is None:
                        prev_qpos = qpos
                        dq = np.zeros_like(qpos)
                    else:
                        dq = qpos - prev_qpos
                        prev_qpos = qpos
                    dq_max = float(np.max(np.abs(dq))) if dq.size else 0.0
                    qvel_max = float(np.max(np.abs(qvel))) if qvel.size else 0.0
                    a_max = float(np.max(np.abs(np.asarray(action, dtype=np.float32)))) if action else 0.0
                    print(f"  [debug] step={step:04d} | max|dqpos|={dq_max:.6f}  max|qvel|={qvel_max:.6f}  max|action|={a_max:.4f}")

                obj = backend.get_entity_position(tag=args.object_tag)
                goal = backend.get_entity_position(tag=args.goal_tag) + np.asarray(goal_center_offset, dtype=np.float32)
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
            # Don't crash the entire sweep due to a transient gRPC failure; record it and move on.
            err = _format_exception_short(e)
            print(f"  [ERROR] Trial {trial_num} crashed at step {steps_taken}: {err}")
        finally:
            with contextlib.suppress(Exception):
                if video_writer is not None:
                    video_writer.close()

        res = CheckpointResult(
            checkpoint=ckpt_name,
            pretrained_model_dir=str(pm_dir),
            success=success,
            steps=steps_taken,
            steps_to_success=steps_to_success,
            min_dist_m=float(min_dist if min_dist != math.inf else 0.0),
            final_dist_m=last_dist,
            error=err,
        )
        results.append(res)

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(res)) + "\n")

        # ── Per-checkpoint result log ──
        status_icon = "SUCCESS" if res.success else ("ERROR" if res.error else "FAIL")
        final_dist_s = "n/a" if res.final_dist_m is None else f"{res.final_dist_m:.4f}m"
        print(f"  -------------------------------------------------------")
        print(f"  RESULT  Trial {trial_num}/{len(ckpt_dirs)}  |  Checkpoint: {ckpt_name}")
        print(f"    Status      : {status_icon}")
        print(f"    Steps       : {res.steps}/{args.max_steps}" + (f"  (reached goal at step {res.steps_to_success})" if res.steps_to_success else ""))
        print(f"    Min distance: {res.min_dist_m:.4f}m   (threshold: {float(args.success_dist_m):.4f}m)")
        print(f"    Final dist  : {final_dist_s}")
        if res.error:
            print(f"    Error       : {res.error}")
        print(f"  -------------------------------------------------------")

    # Rank: success first, then fewer steps_to_success, then smaller min_dist
    def rank_key(r: CheckpointResult):
        return (0 if r.success else 1, r.steps_to_success if r.steps_to_success is not None else 10**9, r.min_dist_m)

    ranked = sorted(results, key=rank_key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(asdict(ranked[0]).keys()) if ranked else ["checkpoint", "pretrained_model_dir", "success"],
        )
        w.writeheader()
        for r in ranked:
            w.writerow(asdict(r))

    # ── Final sweep summary ──
    n_success = sum(1 for r in results if r.success)
    n_error = sum(1 for r in results if r.error)
    n_fail = len(results) - n_success - n_error
    print(f"\n{'='*80}")
    print(f"  SWEEP COMPLETE  |  {len(results)} checkpoints evaluated")
    print(f"    SUCCESS: {n_success}   FAIL: {n_fail}   ERROR: {n_error}")
    if ranked:
        top = ranked[0]
        print(f"    Best checkpoint: {top.checkpoint}  success={top.success}  min_dist={top.min_dist_m:.4f}m  steps_to_success={top.steps_to_success}")
    print(f"  Output files:")
    print(f"    {jsonl_path}")
    print(f"    {csv_path}")
    if do_video:
        print(f"    {output_dir / 'videos_left'}/")
    print(f"{'='*80}")

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
