"""Stereolabs ZED camera driver (ZED SDK capture).

Thin adapter that lets the gear_sonic composed-camera orchestrator drive a
ZED camera the same way it drives RealSense / OAK / USB cameras. Modeled on
``decoupled_wbc.control.sensor.zed.ZedSensor``.

Emits rectified left and right RGB frames. On the wire, the keys are::

    {mount_position}        -> left  RGB
    {mount_position}_right  -> right RGB
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyzed.sl as sl

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


_RES_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}

_DEPTH_MAP = {
    "NONE": sl.DEPTH_MODE.NONE,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
}


class ZEDConfig:
    """Configuration for the Stereolabs ZED camera."""

    resolution: str = "HD720"
    output_image_dim: tuple[int, int] = (640, 360)  # (width, height)
    fps: int = 30
    depth_mode: str = "NONE"  # NONE saves GPU when depth isn't used


class ZEDSensor(Sensor):
    """Sensor for Stereolabs ZED cameras (e.g., ZED-M on Unitree G1)."""

    def __init__(
        self,
        config: ZEDConfig = ZEDConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        serial_number: int | None = None,
    ):
        self.config = config
        self.mount_position = mount_position

        init = sl.InitParameters()
        init.camera_resolution = _RES_MAP[config.resolution]
        init.camera_fps = config.fps
        init.depth_mode = _DEPTH_MAP[config.depth_mode]
        init.coordinate_units = sl.UNIT.METER
        if serial_number is not None:
            init.set_from_serial_number(serial_number)

        self.camera = sl.Camera()
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        info = self.camera.get_camera_information()
        print(
            f"[{mount_position}] Connected to ZED: {info.camera_model} "
            f"(S/N {info.serial_number}) @ {config.resolution} {config.fps} FPS"
        )

        self._runtime = sl.RuntimeParameters()
        self._left_mat = sl.Mat()
        self._right_mat = sl.Mat()

    def read(self) -> dict[str, Any] | None:
        if self.camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        capture_time = time.time()
        self.camera.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        self.camera.retrieve_image(self._right_mat, sl.VIEW.RIGHT)

        left_bgra = self._left_mat.get_data()
        right_bgra = self._right_mat.get_data()
        left_rgb = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2RGB)
        right_rgb = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2RGB)

        w, h = self.config.output_image_dim
        if (left_rgb.shape[1], left_rgb.shape[0]) != (w, h):
            left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)
            right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)

        right_key = f"{self.mount_position}_right"
        return {
            "timestamps": {self.mount_position: capture_time, right_key: capture_time},
            "images": {self.mount_position: left_rgb, right_key: right_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        return ImageMessageSchema(
            timestamps=data["timestamps"], images=data["images"]
        ).serialize()

    def observation_space(self):
        if gym is None:
            return None
        w, h = self.config.output_image_dim
        box = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        return gym.spaces.Dict(
            {self.mount_position: box, f"{self.mount_position}_right": box}
        )

    def close(self):
        if hasattr(self, "camera") and self.camera.is_opened():
            self.camera.close()
