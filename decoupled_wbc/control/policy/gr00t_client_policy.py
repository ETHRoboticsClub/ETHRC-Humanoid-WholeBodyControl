"""VLA client policy: queries an Isaac-GR00T PolicyServer over ZMQ and
produces upper-body goals in the format expected by
:class:`decoupled_wbc.control.policy.g1_decoupled_whole_body_policy.G1DecoupledWholeBodyPolicy`.

Action chunks returned by the server are unrolled step-by-step; a new chunk is
requested once the previous one is exhausted (or forced by ``refresh_every``).
"""
import time as time_module
from typing import Optional

import numpy as np

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.main.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.robot_model.robot_model import RobotModel
from gr00t.policy.server_client import PolicyClient


# Must match Isaac-GR00T's pre-registered ``unitree_g1`` modality.
# See Isaac-GR00T/gr00t/configs/data/embodiment_configs.py.
STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
]
# Action keys the VLA returns. All are shape (B=1, T=horizon, D).
ACTION_UPPER_BODY_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
VIDEO_KEY = "ego_view"
# MuJoCo sim image publisher uses MJ camera ids as dict keys (see BottleEnv / DefaultEnv).
VIDEO_KEY_FALLBACKS = ("ego_view", "egoview", "global_view")
LANGUAGE_KEY = "annotation.human.task_description"


class Gr00tClientPolicy(Policy):
    """Streams upper-body goals derived from a remote Isaac-GR00T VLA.

    The policy is pull-based: call :meth:`get_action` each control tick. It
    lazily fetches a new action chunk from the server whenever its cursor
    catches up to the chunk length or after ``refresh_every`` steps, whichever
    comes first.

    Args:
        robot_model: Instantiated G1 robot model (from
            :mod:`decoupled_wbc.control.robot_model.instantiation.g1`).
        camera: Image source exposing ``read() -> {"images": {VIDEO_KEY: np.uint8[H,W,3]}, ...}``
            (e.g. ``ComposedCameraClientSensor``).
        task_prompt: Natural-language instruction sent with every request.
        server_host: Hostname/IP of ``run_gr00t_server.py``.
        server_port: Port it's listening on (matches ``DEFAULT_MODEL_SERVER_PORT``).
        refresh_every: Re-query the server after this many dispatched steps even
            if the chunk isn't exhausted. ``None`` means "only when exhausted".
        dt: Nominal control period, used for timestamp targets in the goal dict.
        camera_endpoint: ``host:port`` string for logs (composed camera ZMQ PUB).
    """

    is_active = True

    def __init__(
        self,
        robot_model: RobotModel,
        camera,
        task_prompt: str,
        server_host: str = "localhost",
        server_port: int = 5555,
        refresh_every: Optional[int] = None,
        dt: float = 1.0 / 20.0,
        camera_endpoint: Optional[str] = None,
    ):
        self.robot_model = robot_model
        self.camera = camera
        self.task_prompt = task_prompt
        self.dt = dt
        self.refresh_every = refresh_every
        self._server_endpoint = f"{server_host}:{server_port}"
        self._camera_endpoint = camera_endpoint or "camera_host:camera_port"

        self.client = PolicyClient(host=server_host, port=server_port, strict=False)

        # Cache joint-group indices once. These are *global* joint indices.
        self._group_idx = {
            g: np.asarray(self.robot_model.get_joint_group_indices(g), dtype=int)
            for g in ("left_leg", "right_leg", "waist", "left_arm",
                       "right_arm", "left_hand", "right_hand", "upper_body")
        }
        self._upper_body_idx = self._group_idx["upper_body"]

        # Precompute, for each VLA upper-body action key, the slice within the
        # upper_body vector where its values should land. A VLA key maps to
        # zero positions within upper_body if that sub-group isn't part of the
        # upper body in the current robot_model (e.g. waist when
        # ``waist_location == "lower_body"``).
        self._ub_slots = {}
        for key in ACTION_UPPER_BODY_KEYS:
            global_idx = self._group_idx[key]
            # Position of each global index within upper_body (or -1 if absent).
            positions = np.searchsorted(self._upper_body_idx, global_idx)
            mask = (positions < len(self._upper_body_idx)) & (
                self._upper_body_idx[np.clip(positions, 0, len(self._upper_body_idx) - 1)]
                == global_idx
            )
            self._ub_slots[key] = (positions[mask], mask)

        # Proprio snapshot (filled by ``set_observation``); required to build
        # the VLA state input.
        self._latest_q: Optional[np.ndarray] = None

        # Last-seen camera frame, kept around so a transient ``None`` from
        # ``camera.read()`` doesn't stall inference.
        self._latest_img: Optional[np.ndarray] = None

        # Action chunk state.
        self._chunk: Optional[dict] = None
        self._cursor = 0
        self._steps_since_refresh = 0

        self._logged_proprio_wait = False
        self._logged_proprio_ok = False
        self._logged_camera_block = False
        self._logged_first_ego_view = False
        self._chunk_request_seq = 0

    # --- Policy interface ------------------------------------------------- #

    def set_observation(self, observation: dict):
        """Expected keys: ``q`` (full joint-position vector)."""
        if observation is None:
            return
        q = observation.get("q")
        if q is not None:
            self._latest_q = np.asarray(q, dtype=np.float32)
            if not self._logged_proprio_ok:
                print(
                    "[gr00t_client] proprio received (q); will build VLA observations "
                    f"(camera tcp://{self._camera_endpoint}, then server "
                    f"tcp://{self._server_endpoint})."
                )
                self._logged_proprio_ok = True

    def reset(self):
        self._chunk = None
        self._cursor = 0
        self._steps_since_refresh = 0
        try:
            self.client.reset()
        except Exception as e:  # noqa: BLE001 — non-fatal
            print(f"[gr00t_client] reset() failed: {e}")

    def get_action(self, time: Optional[float] = None) -> dict:
        """Return the next control-goal dict for the decoupled WBC."""
        if self._latest_q is None:
            if not self._logged_proprio_wait:
                print(
                    f"[gr00t_client] waiting for proprio: no q yet (subscribe to "
                    f"'{STATE_TOPIC_NAME}', e.g. from run_g1_control_loop). "
                    "VLA get_action is not called until q arrives."
                )
                self._logged_proprio_wait = True
            return self._safe_goal(time)

        if self._chunk is None or self._cursor >= self._chunk_len() or self._should_refresh():
            self._refresh_chunk()

        return self._build_goal_from_cursor(time)

    # --- internals -------------------------------------------------------- #

    def _chunk_len(self) -> int:
        return self._chunk[ACTION_UPPER_BODY_KEYS[0]].shape[1] if self._chunk else 0

    def _should_refresh(self) -> bool:
        return self.refresh_every is not None and self._steps_since_refresh >= self.refresh_every

    def _safe_goal(self, time: Optional[float]) -> dict:
        t_now = time if time is not None else time_module.monotonic()
        return {
            "target_upper_body_pose": self.robot_model.get_initial_upper_body_pose(),
            "base_height_command": DEFAULT_BASE_HEIGHT,
            "navigate_cmd": np.array(DEFAULT_NAV_CMD, dtype=np.float64),
            "target_time": t_now + self.dt,
            "timestamp": t_now,
        }

    def _build_observation(self) -> dict:
        # Image. ComposedCameraClientSensor uses blocking ZMQ recv; first frame
        # can wait indefinitely if no publisher is running.
        if not self._logged_camera_block:
            print(
                f"[gr00t_client] blocking on camera.read() until a ZMQ frame arrives "
                f"from tcp://{self._camera_endpoint} "
                f"(need images with one of {list(VIDEO_KEY_FALLBACKS)})…"
            )
            self._logged_camera_block = True
        img_msg = self.camera.read()
        if img_msg is not None:
            images = img_msg.get("images", {})
            for k in VIDEO_KEY_FALLBACKS:
                if k in images:
                    self._latest_img = images[k]
                    if not self._logged_first_ego_view:
                        print(
                            f"[gr00t_client] first video frame received from camera "
                            f"(images['{k}'] → VLA video key '{VIDEO_KEY}')."
                        )
                        self._logged_first_ego_view = True
                    break
        if self._latest_img is None:
            raise RuntimeError(
                f"No frame with any of keys {list(VIDEO_KEY_FALLBACKS)} in images yet. "
                "For sim, run run_sim_loop.py with --enable-image-publish and "
                "--enable-offscreen; use env pnp_bottle (egoview) or default/scene_29dof "
                "(global_view). Cube/box MuJoCo envs ship no render cameras."
            )
        img = self._latest_img  # (H, W, 3) uint8
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3, (
            f"Expected uint8 HxWx3, got dtype={img.dtype} shape={img.shape}"
        )
        video_batched = img[None, None, ...]  # (B=1, T=1, H, W, 3)

        # State slices from the current joint vector.
        q = self._latest_q
        state = {}
        for key in STATE_KEYS:
            idx = self._group_idx[key]
            if idx.size == 0:
                # Group absent from this robot_model configuration. Isaac-GR00T
                # will reject the observation; the user needs to adjust the
                # embodiment config or the robot_model to line up.
                raise RuntimeError(
                    f"State group '{key}' has no indices in this robot_model. "
                    "The VLA unitree_g1 modality expects all seven groups."
                )
            state[key] = q[idx][None, None, :].astype(np.float32)  # (1, 1, D)

        language = {LANGUAGE_KEY: [[self.task_prompt]]}  # (B=1, T=1)

        return {"video": {VIDEO_KEY: video_batched},
                "state": state,
                "language": language}

    def _refresh_chunk(self):
        obs = self._build_observation()
        self._chunk_request_seq += 1
        print(
            f"[gr00t_client] chunk request #{self._chunk_request_seq}: "
            f"calling PolicyClient.get_action → tcp://{self._server_endpoint} "
            "(blocked until server returns)…"
        )
        t0 = time_module.monotonic()
        action, _info = self.client.get_action(obs)
        latency = time_module.monotonic() - t0
        # Sanity-check shape: each key should be (B=1, T, D).
        for k in ACTION_UPPER_BODY_KEYS + ["base_height_command", "navigate_command"]:
            if k not in action:
                raise RuntimeError(f"VLA server response missing action key '{k}'")
            if action[k].ndim != 3 or action[k].shape[0] != 1:
                raise RuntimeError(
                    f"Action '{k}' has unexpected shape {action[k].shape}; expected (1, T, D)"
                )
        self._chunk = action
        self._cursor = 0
        self._steps_since_refresh = 0
        print(
            f"[gr00t_client] new chunk len={self._chunk_len()} "
            f"server_latency={latency * 1000:.1f}ms"
        )

    def _build_goal_from_cursor(self, time: Optional[float]) -> dict:
        t_now = time if time is not None else time_module.monotonic()
        t = self._cursor

        # Assemble upper-body joint-space target from the chunk's slice.
        upper_pose = self.robot_model.get_initial_upper_body_pose().astype(np.float64)
        for key in ACTION_UPPER_BODY_KEYS:
            positions, mask = self._ub_slots[key]
            if positions.size == 0:
                continue
            vals = self._chunk[key][0, t, :]  # (D,)
            upper_pose[positions] = vals[mask]

        base_height = float(self._chunk["base_height_command"][0, t, 0])
        navigate = np.asarray(
            self._chunk["navigate_command"][0, t, :], dtype=np.float64
        )

        goal = {
            "target_upper_body_pose": upper_pose,
            "base_height_command": base_height,
            "navigate_cmd": navigate,
            "target_time": t_now + self.dt,
            "timestamp": t_now,
        }

        self._cursor += 1
        self._steps_since_refresh += 1
        return goal
