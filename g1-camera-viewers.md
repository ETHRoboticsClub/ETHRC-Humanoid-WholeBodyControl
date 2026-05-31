# G1 camera viewers (ZED + 2 wrist cams)

How to put the G1's cameras on screen from the workstation. Assumes the
**servers are already running on the G1 Orin** (`192.168.123.164`).

## TL;DR

```bash
cd ~/ETHRC-Humanoid-WholeBodyControl

# ZED stereo (ego view, left+right)
DISPLAY=:1 .venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 --camera-port 5556

# Both wrist cams (left + right, composed)
DISPLAY=:1 .venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 --camera-port 5555
```

Two OpenCV windows titled **"SONIC Camera Viewer"**. Stop with `pkill -f run_camera_viewer.py`.

## Which server is on which port

The G1 runs **ZMQ composed camera servers** (started from the
`✳ Start Zed and wrist camera servers` terminal):

| Server                | Port                 | Streams                        | Source                              |
| --------------------- | -------------------- | ------------------------------ | ----------------------------------- |
| ZED (ego, L+R RGB)    | 5556 ZMQ, 8080 MJPEG | `ego_view`, `ego_view_right`   | `/dev/video0,1` — ZED-M @ HD720/30  |
| Left wrist (composed) | 5555 ZMQ             | `left_wrist`                   | `/dev/video4` — Innomaker 640×480   |
| Right wrist (composed)| 5555 ZMQ             | `right_wrist`                  | `/dev/video2` — Innomaker 640×480   |

One `run_camera_viewer.py` instance connects to **one** port, so the ZED and
the wrists need **two** instances (ports 5556 and 5555 respectively).

The ZED also exposes an **MJPEG** stream — open in any browser, no venv needed:

```
http://192.168.123.164:8080
```

## Controls (OpenCV window must be focused)

- **R** — start/stop recording → MP4 under `camera_recordings/rec_<timestamp>/`
- **Q** — quit that viewer

## Gotcha: two different camera transports in this repo

There are **two unrelated** camera-streaming paths. Don't mix them up:

1. **ZMQ composed server → `run_camera_viewer.py`** (OpenCV).
   This is what the G1 servers above use. **Use this one.**
2. **Rerun gRPC: `run_rerun_camera_publisher.py` (G1) → `run_rerun_viewer.py` (PC)**, port 9876.
   The PC viewer only shows frames if the G1 is running the *Rerun publisher*,
   which opens the cameras directly via V4L2/pyzed. It does **not** consume the
   ZMQ/MJPEG servers — point it at those and you get an empty window.

So: if the G1 is running the ZMQ servers (5555/5556/8080), use
`run_camera_viewer.py`, **not** `run_rerun_viewer.py`.

## Verifying a port before launching

To see exactly what a ZMQ port serves without opening a window:

```bash
.venv_data_collection/bin/python - <<'PY'
import time
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
c = ComposedCameraClientSensor(server_ip="192.168.123.164", port=5556)
for _ in range(50):
    s = c.read(blocking=False)
    if s and s.get("images"):
        print({k: (None if v is None else v.shape) for k, v in s["images"].items()})
        break
    time.sleep(0.1)
c.close()
PY
```

## Notes

- Viewer venv is **`.venv_data_collection`** (has `rerun`, `opencv`, the
  `gear_sonic.camera` package).
- `DISPLAY=:1` is the graphical session on this workstation — the OpenCV
  windows render there.
- **Port caveat:** CLAUDE.md documents 5556 as the SONIC SMPL/pose stream and
  5555 as *the* camera server. This setup puts a second composed camera server
  on 5556 instead. Keep that in mind so the SMPL stream during data collection
  doesn't get pointed at 5556.
