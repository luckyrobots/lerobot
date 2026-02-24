"""PiperRoom Gymnasium environment wrapping the LuckyEngine HazelBackend gRPC.

Only n_envs=1 is meaningfully supported with a single Hazel server. Multiple
instances of this environment all connect to the same gRPC endpoint and will
receive the same observations.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class PiperRoomEnv(gym.Env):
    """Gymnasium environment for the Piper arm in LuckyEngine's Piper-room scene.

    Wraps the HazelBackend gRPC client with lazy connection: gRPC streams are
    only established on the first call to reset().

    Observation keys (required by lerobot eval pipeline preprocess_observation):
        ``"agent_pos"``: float32 array of shape (7,) — joint qpos
        ``"pixels"``:    dict of {cam_name: uint8 HWC ndarray}
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50051,
        agent_name: str = "agent_0",
        robot_name: str = "",
        camera_names: tuple[str, ...] = ("CameraGripper", "CameraLeft", "CameraTop"),
        camera_width: int = 320,
        camera_height: int = 240,
        fps: int = 30,
        episode_length: int = 300,
        success_dist_m: float = 0.05,
        object_tag: str = "Red Block",
        goal_tag: str = "Dropbox",
        goal_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        home_action: list[float] | None = None,
        object_reset_pos: tuple[float, float, float] = (0.317096353, 0.0464101955, 0.000183301046),
        object_reset_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        reset_settle_s: float = 1.5,
        reset_hz: float = 30.0,
        object_reset_hold_s: float = 0.25,
        object_reset_hold_hz: float = 30.0,
        startup_grace_s: float = 2.0,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()

        self._host = host
        self._port = port
        self._agent_name = agent_name
        self._robot_name = robot_name
        self._camera_names = tuple(camera_names)
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._fps = fps
        self._success_dist_m = float(success_dist_m)
        self._object_tag = object_tag
        self._goal_tag = goal_tag
        self._goal_center_offset = np.asarray(goal_center_offset, dtype=np.float32)
        self._home_action = (
            list(home_action)
            if home_action is not None
            else [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0, 0.035]
        )
        self._object_reset_pos = tuple(float(x) for x in object_reset_pos)
        self._object_reset_quat = tuple(float(x) for x in object_reset_quat)
        self._reset_settle_s = float(reset_settle_s)
        self._reset_hz = float(reset_hz)
        self._object_reset_hold_s = float(object_reset_hold_s)
        self._object_reset_hold_hz = float(object_reset_hold_hz)
        self._startup_grace_s = float(startup_grace_s)
        self.render_mode = render_mode

        # Required by the lerobot eval pipeline (add_envs_task / check_env_attributes_and_types)
        self.task = "PiperRoom-v0"
        self.task_description = "Pick up the red block and place it in the Dropbox."
        # _max_episode_steps is read by rollout() via env.call("_max_episode_steps")[0].
        # The TimeLimit wrapper also exposes this attribute; our copy is a fallback.
        self._max_episode_steps = episode_length

        # Build observation and action spaces
        pixel_spaces = {
            cam: spaces.Box(0, 255, shape=(camera_height, camera_width, 3), dtype=np.uint8)
            for cam in self._camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "pixels": spaces.Dict(pixel_spaces),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

        # Internal state — set during _connect()
        self._backend: Any = None
        self._initialized: bool = False
        self._block_id: int | None = None
        self._block_initial_transform: Any = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._initialized:
            self._connect()
        self._do_reset()
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._backend.send_control(action.tolist(), robot_name=self._robot_name)
        self._step_count += 1
        obs = self._get_obs()

        block_pos = self._backend.get_entity_position(tag=self._object_tag)
        goal_pos = self._backend.get_entity_position(tag=self._goal_tag) + self._goal_center_offset
        dist = float(np.linalg.norm(block_pos - goal_pos))

        reward = -dist
        terminated = dist <= self._success_dist_m
        truncated = False  # TimeLimit wrapper handles step-count truncation
        info: dict[str, Any] = {"is_success": terminated, "dist_to_goal": dist}
        return obs, reward, terminated, truncated, info

    def render(self):
        obs = self._get_obs()
        cam = "CameraLeft" if "CameraLeft" in self._camera_names else self._camera_names[0]
        return obs["pixels"][cam]

    def close(self):
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        self._backend = None
        self._initialized = False
        self._block_id = None
        self._block_initial_transform = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish gRPC connection, start streams, and snapshot the block entity."""
        from lerobot.luckyengine.hazel_backend import HazelBackend, HazelStreamConfig

        backend = HazelBackend(host=self._host, port=self._port)
        backend.connect()
        backend.get_scene_info()  # fail-fast verify — raises if server not reachable

        cam_cfg = HazelStreamConfig(
            target_fps=self._fps,
            width=self._camera_width,
            height=self._camera_height,
            format="raw",
        )
        for cam in self._camera_names:
            backend.start_camera_stream(camera_name=cam, cfg=cam_cfg)
        backend.start_agent_stream(agent_name=self._agent_name, target_fps=self._fps)
        backend.wait_for_first_obs(camera_names=list(self._camera_names), timeout_s=15.0)

        if self._startup_grace_s > 0.0:
            time.sleep(self._startup_grace_s)

        # Snapshot block entity id and initial transform (used for per-episode resets)
        entity = backend.get_entity(tag=self._object_tag)
        self._block_id = int(entity.id.id)  # entity.id is an EntityId message; .id.id extracts the int
        self._block_initial_transform = entity.transform

        self._backend = backend
        self._initialized = True

    def _do_reset(self) -> None:
        """Execute the full 5-step reset sequence matching piper_diffusion_sweep.py."""
        # Step 1: reset_agent (non-fatal — may reposition scene entities)
        try:
            self._backend.reset_agent(agent_name=self._agent_name)
        except Exception as exc:
            logger.warning("ResetAgent failed (non-fatal): %s", exc)

        # Step 2: drive robot to home pose for reset_settle_s at reset_hz
        hold_dt = 1.0 / max(1e-6, self._reset_hz)
        t_end = time.perf_counter() + max(0.0, self._reset_settle_s)
        while time.perf_counter() < t_end:
            self._backend.send_control(
                list(map(float, self._home_action)), robot_name=self._robot_name
            )
            time.sleep(hold_dt)

        # Step 3: teleport block ×2 with 30 ms gap (physics race fix)
        self._set_block_pose()
        time.sleep(0.03)
        self._set_block_pose()

        # Step 4: hold block pose to fight scene randomisation
        hold_s = max(0.0, self._object_reset_hold_s)
        if hold_s > 0.0:
            dt = 1.0 / max(1e-6, self._object_reset_hold_hz)
            t_end = time.perf_counter() + hold_s
            while time.perf_counter() < t_end:
                self._set_block_pose()
                time.sleep(dt)

        # Step 5: brief pause so observations stabilise
        time.sleep(0.1)

    def _set_block_pose(self) -> None:
        """Teleport the block to the canonical reset position, preserving scale."""
        t = type(self._block_initial_transform)()
        t.CopyFrom(self._block_initial_transform)
        pos = self._object_reset_pos
        quat = self._object_reset_quat
        t.position.x = float(pos[0])
        t.position.y = float(pos[1])
        t.position.z = float(pos[2])
        t.rotation.x = float(quat[0])
        t.rotation.y = float(quat[1])
        t.rotation.z = float(quat[2])
        t.rotation.w = float(quat[3])
        self._backend.set_entity_transform(entity_id=self._block_id, transform=t)

    def _get_obs(self) -> dict[str, Any]:
        """Build the gymnasium observation dict from the latest backend streams."""
        camera_map = {cam: cam for cam in self._camera_names}
        raw = self._backend.get_observation(camera_map=camera_map)

        # Coerce agent state to 7D qpos
        state = np.asarray(raw["observation.state"], dtype=np.float32).reshape(-1)
        if state.size == 7:
            agent_pos = np.ascontiguousarray(state)
        elif state.size == 14:
            # Agent stream returns [qpos, qvel] concatenated — take only qpos
            agent_pos = np.ascontiguousarray(state[:7])
        else:
            # Last resort: pull qpos directly from MuJoCo GetJointState
            qpos, _ = self._backend.get_joint_state(robot_name=self._robot_name)
            q = np.asarray(qpos, dtype=np.float32).reshape(-1)
            if q.size >= 7:
                agent_pos = np.ascontiguousarray(q[:7])
            else:
                logger.warning(
                    "Unexpected state size %d and qpos size %d; padding with zeros.",
                    state.size,
                    q.size,
                )
                agent_pos = np.zeros(7, dtype=np.float32)

        # Build per-camera image dict.
        # The gRPC capture path (GrpcCapture.cpp) applies a vertical flip assuming
        # OpenGL bottom-to-top readback, but the engine actually uses D3D12/Vulkan
        # (top-to-bottom). This erroneous C++ flip makes images upside-down.
        # We flip again here to restore the correct orientation and match the
        # dataset videos (which have no flip applied in FFmpegWriter).
        pixels: dict[str, np.ndarray] = {}
        for cam in self._camera_names:
            img = raw[f"observation.images.{cam}"]
            pixels[cam] = np.ascontiguousarray(np.flip(img, axis=0))

        return {"agent_pos": agent_pos, "pixels": pixels}


def create_piper_room_envs(
    cfg: Any,
    n_envs: int = 1,
    env_cls: Any = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create a vectorized PiperRoom environment and return it in lerobot's standard shape.

    Args:
        cfg: ``PiperRoomEnvConfig`` instance.
        n_envs: Number of parallel environments. **Only n_envs=1 is fully supported.**
            All instances share the same gRPC server and will receive identical observations.
        env_cls: Vector env class (SyncVectorEnv or AsyncVectorEnv). Defaults to SyncVectorEnv.

    Returns:
        ``{"piper_room": {0: <VectorEnv>}}``
    """
    from gymnasium.wrappers import TimeLimit

    if env_cls is None:
        env_cls = gym.vector.SyncVectorEnv

    if n_envs > 1:
        logger.warning(
            "PiperRoomEnv: n_envs=%d requested but only n_envs=1 is fully supported "
            "with a single Hazel gRPC server. All %d instances connect to the same server "
            "and will receive identical observations.",
            n_envs,
            n_envs,
        )

    gym_kwargs = cfg.gym_kwargs
    episode_length = cfg.episode_length

    def _make_one() -> gym.Env:
        env = PiperRoomEnv(**gym_kwargs)
        return TimeLimit(env, max_episode_steps=episode_length)

    vec = env_cls([_make_one for _ in range(n_envs)])
    return {cfg.type: {0: vec}}
