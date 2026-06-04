# G1 camera-server fix for SONIC data collection

**Audience:** the Claude session running *on the G1 Orin* (`unitree@192.168.123.164`).
**Author:** workstation Claude session. I traced the SONIC data exporter on the
workstation; the camera servers on the G1 are currently laid out in a way the
exporter cannot consume. This doc tells you what to change and how to verify it.

> ⚠️ This is the **ETH Robotics Club fork of NVIDIA GR00T-WholeBodyControl**, a
> real humanoid (Unitree G1). The camera server itself does not command motors,
> but it feeds the data-collection pipeline. Don't touch motor/deploy paths.
> Make camera-server changes, verify with the probes below, and don't disable
> anything silently — if a camera is missing, surface it.

---

## TL;DR of the fix

The G1 is currently running **two** composed camera servers:

| Port | Streams it exposes        | Problem |
| ---- | ------------------------- | ------- |
| 5556 | `ego_view`, `ego_view_right` (ZED) | (a) exporter needs ego on the *same* port as the wrists; (b) **5556 is the exporter's SMPL-pose port** — collision |
| 5555 | `left_wrist`, `right_wrist`         | missing `ego_view` |

The exporter opens **one** `ComposedCameraClientSensor` on **one** port and
requires all needed image keys to come from that single server. You must run a
**single composed camera server on port 5555** that exposes **`ego_view` +
`left_wrist` + `right_wrist` together**, and leave **5556 free** for the SMPL
pose stream.

---

## Why — evidence from the exporter (workstation paths)

`gear_sonic/scripts/run_data_exporter.py`:

- **Line 240** — one client, one port:
  ```python
  self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)
  ```
- **Lines 526-538** — each dataset image feature is looked up *by its key suffix*
  in that one server's `images` dict, and a **missing key raises** (loud, good):
  ```python
  image_key = feature_name.split(".")[-1]          # "observation.images.ego_view" -> "ego_view"
  if image_key not in images:
      raise ValueError(f"Required image '{image_key}' ... not found. Available: {list(images.keys())}")
  ```
- **Line 86** — `sonic_zmq_port: int = 5556` → the exporter subscribes to the
  **SMPL pose** stream on **5556**. A camera server on 5556 collides with this.

`gear_sonic/data/features_sonic_vla.py`:

- `get_features_sonic_vla()` **always** registers `observation.images.ego_view`
  → requires key **`ego_view`**.
- `--record-wrist-cameras` adds `observation.images.left_wrist` /
  `observation.images.right_wrist` → requires keys **`left_wrist`**, **`right_wrist`**.

### What the exporter therefore demands from ONE server on `--camera-port`

| Run mode                  | Required `images` keys (all from one port) |
| ------------------------- | ------------------------------------------ |
| default (no wrist record) | `ego_view`                                 |
| `--record-wrist-cameras`  | `ego_view`, `left_wrist`, `right_wrist`    |

Extra keys (e.g. `ego_view_right`) are harmless — they're ignored, not errors.

---

## The fix — one composed server on 5555

The composed server is `gear_sonic/camera/composed_camera.py` (run as a module).
Its `ComposedCameraConfig` registers each camera under a fixed mount key:
`ego_view`, `left_wrist`, `right_wrist` (see `_get_camera_configs`, ~line 164).
The published `images` keys are exactly those mount names, so configuring all
three on one server gives the exporter everything it needs on one port.

**Device mapping (from the workstation's probe of your current servers):**

| Mount       | Camera type | Device                                   |
| ----------- | ----------- | ---------------------------------------- |
| `ego_view`  | `zed`       | ZED-M S/N 14069382 (`/dev/video0,1`)     |
| `left_wrist`| `usb`       | Innomaker `/dev/video4` → device-id `4`  |
| `right_wrist`| `usb`      | Innomaker `/dev/video2` → device-id `2`  |

**Command (run on the Orin, repo is `/home/unitree/GR00T-WholeBodyControl`, venv `.venv_camera`):**

```bash
cd /home/unitree/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera \
    --ego-view-camera zed \
    --left-wrist-camera usb  --left-wrist-device-id 4 \
    --right-wrist-camera usb --right-wrist-device-id 2 \
    --port 5555
```

Notes / things for you to verify on the box (don't assume):
- **First stop the two existing servers** so they release the cameras and free
  the ports (the ZED on 5556 and the wrist server on 5555). Find them with
  `pgrep -af "composed_camera|realsense.*--server|videohub"` and the systemd
  unit `sudo systemctl status composed_camera_server.service`. If the systemd
  unit is the one serving cameras, edit its `ExecStart` to the command above
  rather than running a second copy by hand (two servers fighting for the ZED
  will fail).
- **ZED device-id:** the `zed` camera type opens via `pyzed`. `--ego-view-device-id`
  may be optional (single ZED) or take the serial `14069382`. Try without first;
  if it grabs the wrong camera, pass the serial.
- **USB device-id is the `/dev/videoN` index** (the integer `N`), per the
  `ComposedCameraConfig` docstring. `4` and `2` are the *capture* nodes your
  current wrist server already uses — confirm with `v4l2-ctl --list-devices`.
- Confirm the `zed` and `usb` camera types are actually implemented in *this*
  build of `gear_sonic/camera/composed_camera.py` (grep the worker for
  `camera_type == "zed"` / `"usb"`). If `zed` isn't supported there, that's why
  the current setup runs the ZED as a separate server — tell the workstation
  session and we'll rethink (e.g. keep ZED separate but move it **off 5556**,
  and point the exporter's camera port at the wrist+? server — but the exporter
  can only read one port, so a single combined server is the clean answer).

---

## Verify before declaring success

**1. One server exposes all three keys** (run on the Orin or workstation):

```bash
.venv_camera/bin/python - <<'PY'
import time
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
c = ComposedCameraClientSensor(server_ip="127.0.0.1", port=5555)   # use 192.168.123.164 from the workstation
for _ in range(50):
    s = c.read(blocking=False)
    if s and s.get("images"):
        print(sorted(s["images"].keys()))
        break
    time.sleep(0.1)
c.close()
PY
```

Expect: `['ego_view', 'ego_view_right', 'left_wrist', 'right_wrist']`
(`ego_view_right` is fine/ignored). **Required:** `ego_view`, `left_wrist`, `right_wrist`.

**2. Port 5556 is NOT held by a camera** (so the SMPL pose stream can bind it):

```bash
pgrep -af "composed_camera|realsense" ; ss -ltnp | grep -E ':5555|:5556'
```

Expect a camera listener on **5555 only**.

**3. (Optional) Exporter dry-run** — from the workstation, with the launcher's
default `--camera-port 5555`, the exporter should connect and (with
`--record-wrist-cameras`) **not** raise the `Required image '...' not found`
error. If it raises, the key listed in `Available:` tells you which mount didn't
come up.

---

## Hand back

When done, reply to the workstation session with:
- the output of verify step 1 (the sorted keys),
- which ports are now held (step 2),
- whether `--ego-view-device-id` was needed,
- and whether `zed`/`usb` camera types were present in this build.

That's enough for us to launch `launch_data_collection.py` with confidence.

---

## ⚠️ USB index fragility — wrist `device-id`s drift on reboot/replug (2026-06-04)

**This is the #1 recurring failure for this server.** The `usb` wrist cameras are
addressed by their `/dev/videoN` *capture-node index*, but those indices are
**not stable** — they shuffle whenever USB re-enumerates (reboot, replug, hub
change, even the ZED grabbing nodes first). In one session the wrist capture node
was observed at `0`→`2`→`3`→`4` across a handful of replugs, and the ZED-M took
`/dev/video0,1`. A `device-id` baked into the systemd unit that was correct
yesterday can silently point at the **wrong device** (or at the ZED) today.

### Symptom
Service is `active` and a listener is bound on `5555`, **but no frames publish**
(`check_camera_server.sh` → "server published NO frames"). The journal shows the
worker spinning on `can't open camera by index` / `Camera index out of range`
(index points at the ZED or a gone node) or `USB camera read failed: ret=False` /
`VIDIOC_REQBUFS ... No such device (errno 19)` (device vanished). Because publish
is all-or-nothing, one bad wrist index kills *all three* streams.

### Fix recipe (the one that worked, 2026-06-04)
1. **Find the real capture indices** — each Innomaker exposes two nodes; the
   **lower** one is the capture node:
   ```bash
   v4l2-ctl --list-devices    # ZED-M => video0,1 ; each Innomaker => e.g. video2,3 and video4,5
   ```
   Confirm BOTH wrist cams open *simultaneously* (the real USB-bandwidth test —
   they must negotiate MJPG, not uncompressed YUYV, on a shared hub):
   ```bash
   .venv_camera/bin/python - <<'PY'
   import cv2
   for i in (2,4):
       c=cv2.VideoCapture(i,cv2.CAP_V4L2); ok,_=c.read(); print(i, c.isOpened(), ok); c.release()
   PY
   ```
2. **Update the device-ids** in the systemd **drop-in** (NOT the base unit — the
   drop-in's `ExecStart=` override wins). On this G1 the live config lives at:
   ```
   /etc/systemd/system/composed_camera_server.service.d/zed.conf
   ```
   Set `--left-wrist-device-id <N>` / `--right-wrist-device-id <M>` to the capture
   indices found above. (Convention here: left=4, right=2 — swap if mirrored.)
3. **Reload + HARD restart.** A plain `restart` is not enough: editing `ExecStart`
   then `daemon-reload` makes systemd log *"Current command vanished from the unit
   file"* and **detach from the still-running old PID without killing it** — the
   stale process keeps holding port 5555 and the cameras, so nothing changes. The
   old worker also ignores `SIGTERM` (it just loops on read errors). Force it:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl stop composed_camera_server.service
   sudo pkill -9 -f 'gear_sonic.camera.composed_camera'   # kill the stuck old PID
   sudo systemctl start composed_camera_server.service
   ```
4. **Verify** the process really cycled and frames flow:
   ```bash
   systemctl show composed_camera_server.service -p MainPID -p ExecMainStartTimestamp
   pgrep -af composed_camera     # confirm the args show the NEW device-ids
   bash check_camera_server.sh   # expect: ego_view:OK left_wrist:OK right_wrist:OK
   ```

### Hardware notes
- Keep the **ZED-M on a USB 3.0 path** (must enumerate as `2b03:f682`; `2b03:f681`
  alone = USB 2.0 / unseated → ZED stream fails). The preflight checks this.
- Two 1080p wrist cams on a single shared hub can exceed USB-2.0 bandwidth if a
  cam falls back to uncompressed YUYV; splitting them across two hubs (Apple +
  Realtek here) resolved the "second camera won't enumerate" case.
- A more robust long-term fix than raw indices would be **udev by-id symlinks**
  (or addressing the wrist cams by USB serial/path) so the mapping survives
  re-enumeration — not yet implemented in this build.
