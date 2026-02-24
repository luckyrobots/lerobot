from __future__ import annotations

import argparse
import random
import time
from typing import Sequence

import cv2
import numpy as np

from lerobot.luckyengine.hazel_backend import HazelBackend, HazelStreamConfig


def _uniform_controls(nu: int, scale: float) -> list[float]:
    return [random.uniform(-scale, scale) for _ in range(max(0, int(nu)))]


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 50053,
    agent_name: str = "agent_0",
    robot_name: str = "",
    camera_names: Sequence[str] = ("CameraLeft",),
    fps: int = 10,
    width: int = 320,
    height: int = 240,
    seconds: float = 30.0,
    control_scale: float = 1.0,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    backend = HazelBackend(host=host, port=int(port))
    backend.connect()

    # Verify server is reachable.
    try:
        info = backend.get_scene_info()
        print(f"Scene: {getattr(info, 'scene_name', '?')}  entities={getattr(info, 'entity_count', '?')}")
    except Exception as e:
        backend.close()
        raise RuntimeError(f"Could not reach Hazel gRPC on {host}:{port}.") from e

    # ── 1. Start agent stream and wait for it (this is fast & reliable) ──
    print(f"Starting agent stream ({fps} fps)...")
    backend.start_agent_stream(agent_name=str(agent_name), target_fps=int(fps))

    print("Waiting for agent state...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 15.0:
        with backend._lock:
            if backend._latest_state is not None:
                break
            err = backend._agent_stream_error
        if err is not None:
            backend.close()
            raise RuntimeError(f"Agent stream failed: {err}") from err
        time.sleep(0.02)
    else:
        backend.close()
        raise TimeoutError("Agent state never arrived (is PiperExternalEnv in the scene?)")
    print("Agent state OK!")

    # ── 2. Discover actuators ──
    mj_info = backend.get_mujoco_info(robot_name=str(robot_name))
    nu = int(getattr(mj_info, "nu", 0))
    if nu <= 0:
        backend.close()
        raise RuntimeError("GetMujocoInfo reported nu == 0.")

    # Start cameras (best-effort — don't block startup if slow to warm up).
    unique_cams = list(dict.fromkeys(camera_names))
    cam_cfg = HazelStreamConfig(target_fps=int(fps), width=int(width), height=int(height), format="raw")
    for cam in unique_cams:
        print(f"Starting camera stream: {cam} ({width}x{height} @ {fps}fps)...")
        backend.start_camera_stream(camera_name=str(cam), cfg=cam_cfg)
        time.sleep(0.3)

    print("Waiting up to 10s for camera frames...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        with backend._lock:
            have_all = all(c in backend._latest_images for c in unique_cams)
        if have_all:
            break
        time.sleep(0.05)
    with backend._lock:
        for c in unique_cams:
            print(f"  {c}: {'OK' if c in backend._latest_images else 'NOT YET (will keep trying)'}")

    print(f"\nConnected to {host}:{port}")
    print(f"  Actuators: nu={nu}  Cameras: {unique_cams}")
    print(f"  Loop rate: {fps} Hz")
    print("  Press 'q' in preview window or Ctrl+C to stop.\n")

    camera_map = {n: n for n in unique_cams}
    dt = 1.0 / max(1, int(fps))
    t_end = time.perf_counter() + float(seconds) if seconds > 0 else float("inf")
    next_t = time.perf_counter()
    step = 0

    try:
        while time.perf_counter() < t_end:
            step += 1

            # Get agent state (always available).
            with backend._lock:
                state = backend._latest_state.copy() if backend._latest_state is not None else None
                images = {k: v.copy() for k, v in backend._latest_images.items()}

            # Joint state from MuJoCo.
            qpos, qvel = backend.get_joint_state(robot_name=str(robot_name))

            # Random control action.
            controls = _uniform_controls(nu=nu, scale=float(control_scale))
            backend.send_control(controls, robot_name=str(robot_name))

            # Camera preview (best-effort).
            frames = [images[c] for c in unique_cams if c in images]
            if frames:
                target_h = frames[0].shape[0]
                resized = []
                for f in frames:
                    if f.shape[0] != target_h:
                        sf = target_h / f.shape[0]
                        f = cv2.resize(f, (int(f.shape[1] * sf), target_h))
                    resized.append(f)
                mosaic = np.concatenate(resized, axis=1)
                mosaic = np.flip(mosaic, axis=0)  # fix vertical flip from GPU readback
                cv2.imshow("Hazel Camera Preview", cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("'q' pressed — stopping.")
                    break

            if step % max(1, int(fps)) == 0:
                frame_shape = tuple(frames[0].shape) if frames else None
                print(
                    f"[step {step:05d}] "
                    f"state={None if state is None else tuple(state.shape)}  "
                    f"qpos={tuple(qpos.shape)}  qvel={tuple(qvel.shape)}  "
                    f"frame={frame_shape}  "
                    f"ctrl=[{', '.join(f'{c:+.2f}' for c in controls[:4])}...]"
                )

            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        backend.close()
        print("Done.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal Hazel gRPC demo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50053)
    ap.add_argument("--agent-name", default="agent_0")
    ap.add_argument("--robot-name", default="")
    ap.add_argument("--camera", action="append", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    run(
        host=args.host, port=args.port,
        agent_name=args.agent_name, robot_name=args.robot_name,
        camera_names=args.camera or ["CameraLeft"],
        fps=args.fps, width=args.width, height=args.height,
        seconds=args.seconds, control_scale=args.control_scale, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
