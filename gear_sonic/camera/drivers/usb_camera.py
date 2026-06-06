"""Generic USB webcam driver using OpenCV.

No hardware SDK needed — works with any UVC-compatible camera visible as
``/dev/video*``.  Only requires ``opencv-python``.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


class USBCameraConfig:
    """Configuration for generic USB camera."""

    image_dim: tuple = (640, 480)
    fps: int = 30
    device_index: int = 0


class USBCameraSensor(Sensor):
    """Sensor for generic USB cameras using OpenCV VideoCapture."""

    def __init__(
        self,
        config: USBCameraConfig = USBCameraConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_index: int | None = None,
    ):
        self.config = config
        self.mount_position = mount_position

        idx = device_index if device_index is not None else config.device_index

        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {idx}")

        # Force on-camera MJPEG. The default UVC format is raw YUYV, which at
        # 640x480x30 is ~18 MB/s per camera and saturates the shared USB bus on
        # the Orin (ZED + 2 wrist cams), starving the other streams. FOURCC must
        # be set before resolution for the V4L2 backend to honor it.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        # NOT 1: a single V4L2 buffer can't double-buffer, so the driver drops
        # every other frame and these UVC cams stream at ~15fps instead of 30
        # (verified: bufsize 1 -> 14.7fps, bufsize>=2 -> 29.5fps; v4l2-ctl reads
        # the same device at ~30). The composed-camera worker drains to the
        # latest frame downstream, so a few buffers add no real latency.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

        print(f"[{mount_position}] Warming up USB camera...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        print(f"[{mount_position}] USB camera opened at index {idx}")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  FOURCC: {fourcc_str}")
        if fourcc_str != "MJPG":
            print(
                f"[WARN] USB camera {mount_position} (idx {idx}): expected=MJPG, "
                f"got={fourcc_str}, fallback=uncompressed — raw frames eat shared "
                f"USB bandwidth and may starve the other cameras",
                flush=True,
            )

    def read(self) -> dict[str, Any] | None:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[{self.mount_position}] USB camera read failed: ret={ret}")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: frame_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.config.image_dim[1], self.config.image_dim[0], 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
