from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lerobot.luckyengine.grpc_stubs import default_proto_path, generate_python_stubs


@dataclass(frozen=True)
class HazelStreamConfig:
    target_fps: int = 30
    width: int = 640
    height: int = 480
    format: str = "raw"


class HazelBackend:
    """
    Minimal Hazel ScriptCore gRPC client for LuckyEngine simulation control.

    - Streams cameras via CameraService.StreamCamera(name=...)
    - Streams agent observations via AgentService.StreamAgent(agent_name=...)
    - Sends actions via MujocoService.SendControl(controls=...)
    - Queries entity transforms via SceneService.GetEntity(name=<tag>) (tags, not entity names)
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50051,
        proto_path: str | Path | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)

        proto = Path(proto_path) if proto_path is not None else default_proto_path()
        self._stubs = generate_python_stubs(str(proto))
        self._pb2 = self._stubs.pb2
        self._pb2_grpc = self._stubs.pb2_grpc

        self._channel = None
        self._scene = None
        self._mujoco = None
        self._agent = None
        self._camera = None

        self._lock = threading.Lock()
        self._latest_state: np.ndarray | None = None
        self._latest_images: dict[str, np.ndarray] = {}

        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        # Stream error capture (otherwise errors are swallowed and callers only see timeouts).
        self._agent_stream_error: Exception | None = None
        self._camera_stream_errors: dict[str, Exception] = {}

    def connect(self) -> None:
        try:
            import grpc  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing grpcio. Install it (and grpcio-tools) to use LuckyEngine gRPC:\n"
                "  python -m pip install grpcio grpcio-tools"
            ) from e

        target = f"{self._host}:{self._port}"
        self._channel = grpc.insecure_channel(target)

        self._scene = self._pb2_grpc.SceneServiceStub(self._channel)
        self._mujoco = self._pb2_grpc.MujocoServiceStub(self._channel)
        self._agent = self._pb2_grpc.AgentServiceStub(self._channel)
        self._camera = self._pb2_grpc.CameraServiceStub(self._channel)

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.2)
        self._threads.clear()

        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
        self._channel = None
        self._agent_stream_error = None
        self._camera_stream_errors.clear()

    # ----------------------------
    # Streams
    # ----------------------------
    def start_agent_stream(self, *, agent_name: str = "agent_0", target_fps: int = 30) -> None:
        if self._agent is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        req = self._pb2.StreamAgentRequest(agent_name=agent_name, target_fps=int(target_fps))

        def _run() -> None:
            try:
                for frame in self._agent.StreamAgent(req, timeout=None):
                    if self._stop.is_set():
                        return
                    obs = np.asarray(frame.observations, dtype=np.float32)
                    with self._lock:
                        self._latest_state = obs
            except Exception as e:
                with self._lock:
                    self._agent_stream_error = e
                return

        t = threading.Thread(target=_run, name=f"hazel-agent-{agent_name}", daemon=True)
        t.start()
        self._threads.append(t)

    def start_camera_stream(self, *, camera_name: str, cfg: HazelStreamConfig) -> None:
        if self._camera is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        req = self._pb2.StreamCameraRequest(
            name=camera_name,
            target_fps=int(cfg.target_fps),
            width=int(cfg.width),
            height=int(cfg.height),
            format=str(cfg.format),
        )

        def _run() -> None:
            try:
                for frame in self._camera.StreamCamera(req, timeout=None):
                    if self._stop.is_set():
                        return
                    img = _decode_image_frame(frame)
                    with self._lock:
                        self._latest_images[camera_name] = img
            except Exception as e:
                with self._lock:
                    self._camera_stream_errors[camera_name] = e
                return

        t = threading.Thread(target=_run, name=f"hazel-cam-{camera_name}", daemon=True)
        t.start()
        self._threads.append(t)

    def wait_for_first_obs(
        self,
        *,
        camera_names: Sequence[str],
        timeout_s: float = 10.0,
        poll_s: float = 0.01,
    ) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            with self._lock:
                agent_err = self._agent_stream_error
                cam_errs = dict(self._camera_stream_errors)
                have_state = self._latest_state is not None
                have_imgs = all(name in self._latest_images for name in camera_names)
            if agent_err is not None and not have_state:
                raise RuntimeError(f"Agent stream failed before first observation: {agent_err}") from agent_err
            for cam_name, err in cam_errs.items():
                if cam_name in camera_names and cam_name not in self._latest_images:
                    raise RuntimeError(f"Camera stream failed before first frame for '{cam_name}': {err}") from err
            if have_state and have_imgs:
                return
            time.sleep(poll_s)
        missing = []
        with self._lock:
            if self._latest_state is None:
                missing.append("agent_state")
            for name in camera_names:
                if name not in self._latest_images:
                    missing.append(f"camera:{name}")
        raise TimeoutError(f"Timed out waiting for initial observations: {missing}")

    # ----------------------------
    # RPC helpers
    # ----------------------------
    def send_control(self, controls: Sequence[float], *, robot_name: str = "") -> None:
        if self._mujoco is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._mujoco.SendControl(
            self._pb2.SendControlRequest(robot_name=robot_name, controls=list(map(float, controls))),
            timeout=self._timeout_s,
        )
        if not getattr(resp, "success", False):
            raise RuntimeError(f"SendControl rejected: {getattr(resp, 'message', '')}")

    def get_joint_state(self, *, robot_name: str = "") -> tuple[np.ndarray, np.ndarray]:
        """
        Fetch MuJoCo joint positions/velocities for the currently registered agent.
        Returns (qpos, qvel) as float32 numpy arrays.
        """
        if self._mujoco is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._mujoco.GetJointState(
            self._pb2.GetJointStateRequest(robot_name=str(robot_name)),
            timeout=self._timeout_s,
        )
        if not getattr(resp, "success", False):
            raise RuntimeError(f"GetJointState failed: {getattr(resp, 'message', '')}")
        st = resp.state
        qpos = np.asarray(list(getattr(st, "positions", [])), dtype=np.float32)
        qvel = np.asarray(list(getattr(st, "velocities", [])), dtype=np.float32)
        return qpos, qvel

    def get_mujoco_info(self, *, robot_name: str = ""):
        """
        Fetch MuJoCo metadata for the currently registered agent (nq/nv/nu + names).
        Returns the raw protobuf response.
        """
        if self._mujoco is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._mujoco.GetMujocoInfo(
            self._pb2.GetMujocoInfoRequest(robot_name=str(robot_name)),
            timeout=self._timeout_s,
        )
        if not getattr(resp, "success", False):
            raise RuntimeError(f"GetMujocoInfo failed: {getattr(resp, 'message', '')}")
        return resp

    def get_entity_position(self, *, tag: str) -> np.ndarray:
        if self._scene is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._scene.GetEntity(self._pb2.GetEntityRequest(name=tag), timeout=self._timeout_s)
        if not getattr(resp, "found", False):
            raise KeyError(f"Entity with tag '{tag}' not found")
        pos = resp.entity.transform.position
        return np.asarray([pos.x, pos.y, pos.z], dtype=np.float32)

    def get_entity(self, *, tag: str):
        """
        Fetch full entity info (id + transform) by tag.
        """
        if self._scene is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._scene.GetEntity(self._pb2.GetEntityRequest(name=tag), timeout=self._timeout_s)
        if not getattr(resp, "found", False):
            raise KeyError(f"Entity with tag '{tag}' not found")
        return resp.entity

    def set_entity_transform(self, *, entity_id: int, transform: Any) -> None:
        """
        Set an entity transform by id using SceneService.SetEntityTransform.
        """
        if self._scene is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        req = self._pb2.SetEntityTransformRequest(id=self._pb2.EntityId(id=int(entity_id)), transform=transform)
        resp = self._scene.SetEntityTransform(req, timeout=self._timeout_s)
        if not getattr(resp, "success", False):
            raise RuntimeError(f"SetEntityTransform failed: {getattr(resp, 'message', '')}")

    def reset_agent(self, agent_name: str = "agent_0") -> None:
        """
        Reset a specific agent via AgentService.ResetAgent.
        Clears observation/action buffers and resets MuJoCo state for the agent.
        """
        if self._agent is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        resp = self._agent.ResetAgent(
            self._pb2.ResetAgentRequest(agent_name=agent_name),
            timeout=self._timeout_s,
        )
        if not getattr(resp, "success", False):
            raise RuntimeError(f"ResetAgent failed: {getattr(resp, 'message', '')}")

    def get_scene_info(self):
        """
        Fetch basic scene metadata (requires SceneService to be enabled on the server).
        """
        if self._scene is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._scene.GetSceneInfo(self._pb2.GetSceneInfoRequest(), timeout=self._timeout_s)

    # ----------------------------
    # Observation assembly (LeRobot keys)
    # ----------------------------
    def get_observation(self, *, camera_map: Mapping[str, str] | None = None) -> dict[str, np.ndarray]:
        """
        Build a LeRobot-style observation dict using keys like:
        - observation.state
        - observation.images.CameraLeft
        """
        with self._lock:
            state = None if self._latest_state is None else self._latest_state.copy()
            images = {k: v.copy() for k, v in self._latest_images.items()}
            agent_err = self._agent_stream_error
            cam_errs = dict(self._camera_stream_errors)

        if state is None:
            # Prefer surfacing the underlying stream error if it exists.
            if agent_err is not None:
                raise RuntimeError(f"Agent stream failed: {agent_err}") from agent_err
            raise RuntimeError("No agent state received yet.")

        obs: dict[str, np.ndarray] = {"observation.state": state}
        if camera_map:
            # If a camera stream died after the first frame, callers would otherwise keep getting
            # a stale last image forever. Surface the error so higher-level code can retry/abort.
            for cam_name in camera_map.keys():
                err = cam_errs.get(cam_name)
                if err is not None:
                    raise RuntimeError(f"Camera stream failed for '{cam_name}': {err}") from err
            for cam_name, obs_key_suffix in camera_map.items():
                if cam_name not in images:
                    raise RuntimeError(f"No image received yet for camera '{cam_name}'")
                obs[f"observation.images.{obs_key_suffix}"] = images[cam_name]

        return obs


def make_dummy_rgb_image(width: int, height: int) -> np.ndarray:
    """
    Create an HWC uint8 RGB image filled with zeros.
    Useful for smoke-testing inference when camera streaming isn't available.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dummy image size: {width}x{height}")
    return np.zeros((height, width, 3), dtype=np.uint8)


def _decode_image_frame(frame: Any) -> np.ndarray:
    """
    Convert Hazel gRPC ImageFrame to HWC uint8 numpy array.

    The server usually streams raw RGBA; we drop alpha to get RGB.
    """
    if frame.format != "raw":
        raise ValueError(f"Only raw frames are supported for now (got format={frame.format!r})")
    w = int(frame.width)
    h = int(frame.height)
    c = int(frame.channels)
    buf = bytes(frame.data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size != w * h * c:
        raise ValueError(f"Invalid frame byte size: got {arr.size}, expected {w*h*c} ({w=} {h=} {c=})")
    img = arr.reshape((h, w, c))
    if c == 4:
        img = img[:, :, :3]
    return img


