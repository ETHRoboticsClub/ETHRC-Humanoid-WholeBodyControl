"""Stereolabs ZED camera driver (ZED SDK capture).

Thin adapter that lets the gear_sonic composed-camera orchestrator drive a
ZED camera the same way it drives RealSense / OAK / USB cameras. Modeled on
``decoupled_wbc.control.sensor.zed.ZedSensor``.

Emits the rectified left RGB frame as ``{mount_position}``. The right eye is
opt-in (``ZEDConfig.emit_right_view``) and, when enabled, is published as
``{mount_position}_right``. Data collection records only the mono ego view, so
the right eye is off by default to avoid wasting encode / USB / ZMQ bandwidth
on the Orin for an unrecorded stream.
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
    # (width, height). 640x480 to match the ego_view dataset schema
    # (features_sonic_vla.EGO_VIEW_HEIGHT/WIDTH); the native 16:9 HD720 frame is
    # resized to 4:3 in read(), consistent for both collection and inference.
    output_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    depth_mode: str = "NONE"  # NONE saves GPU when depth isn't used
    emit_right_view: bool = False  # also publish right eye as {mount}_right; off = mono ego only


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
        w, h = self.config.output_image_dim

        self.camera.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        left_rgb = cv2.cvtColor(self._left_mat.get_data(), cv2.COLOR_BGRA2RGB)
        if (left_rgb.shape[1], left_rgb.shape[0]) != (w, h):
            left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)

        timestamps = {self.mount_position: capture_time}
        images = {self.mount_position: left_rgb}

        if self.config.emit_right_view:
            self.camera.retrieve_image(self._right_mat, sl.VIEW.RIGHT)
            right_rgb = cv2.cvtColor(self._right_mat.get_data(), cv2.COLOR_BGRA2RGB)
            if (right_rgb.shape[1], right_rgb.shape[0]) != (w, h):
                right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)
            right_key = f"{self.mount_position}_right"
            timestamps[right_key] = capture_time
            images[right_key] = right_rgb

        return {"timestamps": timestamps, "images": images}

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
        spaces = {self.mount_position: box}
        if self.config.emit_right_view:
            spaces[f"{self.mount_position}_right"] = box
        return gym.spaces.Dict(spaces)

    def close(self):
        if hasattr(self, "camera") and self.camera.is_opened():
            self.camera.close()
