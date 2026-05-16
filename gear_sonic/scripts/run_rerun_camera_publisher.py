"""
Publish live frames from the G1's cameras to a Rerun viewer running on a
LAN-connected machine.

Cameras handled:
  - ZED Mini stereo (via the ZED SDK, depth_mode=NONE so it starts instantly)
  - Any Innomaker U20CAM (the wrist cameras) — auto-discovered via V4L2

Setup on the G1 (Orin NX):
  pip install -e "gear_sonic[viz]"
  # NOTE: pyzed is NOT a pip dep — install the Stereolabs ZED SDK separately.
  # Without pyzed the script still runs (wrist-only).

Usage on the G1:
  sg zed -c "python3 gear_sonic/scripts/run_rerun_camera_publisher.py --rerun-host <laptop-ip>"

  The `sg zed -c` prefix puts the process in the `zed` group so it has
  access to the ZED's udev nodes.

Usage on the laptop:
  pip install -e "gear_sonic[viz]"
  python gear_sonic/scripts/run_rerun_viewer.py   # binds 0.0.0.0:9876, prints the IP to use

  The `--session` value must match what this publisher passes for the panes
  to appear in the same recording.

Stop with Ctrl+C on the G1.
"""

import argparse
import glob
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr


def discover_innomaker_nodes() -> List[Tuple[str, int]]:
    """Return [(friendly_name, /dev/videoN integer index), ...] for each
    Innomaker U20CAM-1080p capture node currently enumerated by V4L2.

    UVC webcams expose two /dev/videoN nodes — one for image, one for
    metadata. We keep only the image node (the one whose v4l2 capability
    includes `video_capture`), which is conventionally the lower-numbered
    of each pair.
    """
    found: List[Tuple[str, int]] = []
    for name_path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            name = open(name_path).read().strip()
        except OSError:
            continue
        if "Innomaker" not in name:
            continue
        idx = int(os.path.basename(os.path.dirname(name_path)).replace("video", ""))
        # Quick capability check: image capture node opens, metadata node doesn't
        # respond to MJPG fourcc set — but the cheaper heuristic is "even index
        # of each pair." The kernel always numbers them as (capture, metadata)
        # in pairs starting at the first index UVC enumerates.
        if idx % 2 == 0:
            found.append((name, idx))
    return found


def open_v4l2_mjpg(dev_index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        return None
    return cap


def open_zed():
    """Returns (camera, runtime, left_mat, right_mat) or None on failure."""
    try:
        import pyzed.sl as sl
    except Exception as e:
        print(f"[zed] pyzed import failed: {e}")
        return None
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE  # critical: avoid 6-minute TRT optimisation
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[zed] open failed: {err}")
        return None
    info = cam.get_camera_information()
    print(f"[zed] opened {info.camera_model}  S/N {info.serial_number}  HD720@30")
    return cam, sl.RuntimeParameters(), sl.Mat(), sl.Mat(), sl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-host", required=True, help="IP or hostname of the laptop running the rerun viewer")
    parser.add_argument("--rerun-port", type=int, default=9876)
    parser.add_argument("--session", default="g1_cameras",
                        help="must match the --session passed to run_rerun_viewer.py on the PC")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0, help="max publish rate per camera")
    parser.add_argument("--no-zed", action="store_true")
    args = parser.parse_args()

    # 1. Open cameras (skip silently if unavailable so this works during physical setup).
    zed = None if args.no_zed else open_zed()

    innomakers: List[Tuple[str, cv2.VideoCapture]] = []
    for name, idx in discover_innomaker_nodes():
        cap = open_v4l2_mjpg(idx, args.width, args.height)
        if cap is None:
            print(f"[wrist] /dev/video{idx} ({name}) failed to open — skipping")
            continue
        # The two Innomakers are differentiated by their USB port (visible in
        # v4l2-ctl --list-devices). Without a stable mapping we just label them
        # in discovery order: wrist_0 (first found), wrist_1 (second).
        slot = f"wrist_{len(innomakers)}"
        print(f"[wrist] /dev/video{idx} -> cameras/{slot}")
        innomakers.append((slot, cap))

    if zed is None and not innomakers:
        print("ERROR: no cameras opened. Aborting.")
        return 1

    # 2. Connect to the rerun viewer on the laptop.
    rr.init(args.session)
    url = f"rerun+http://{args.rerun_host}:{args.rerun_port}/proxy"
    print(f"[rerun] connecting to {url}")
    rr.connect_grpc(url=url)

    # 3. Stream forever.
    period = 1.0 / args.fps
    frame_idx = 0
    print("[run] streaming — Ctrl+C to stop")
    try:
        while True:
            t_loop = time.time()
            rr.set_time_seconds("camera_clock", t_loop)
            rr.set_time_sequence("frame", frame_idx)

            if zed is not None:
                cam, runtime, left, right, sl = zed
                if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(left, sl.VIEW.LEFT)
                    cam.retrieve_image(right, sl.VIEW.RIGHT)
                    rr.log("cameras/zed/left",  rr.Image(cv2.cvtColor(left.get_data(),  cv2.COLOR_BGRA2RGB)))
                    rr.log("cameras/zed/right", rr.Image(cv2.cvtColor(right.get_data(), cv2.COLOR_BGRA2RGB)))

            for slot, cap in innomakers:
                ok, frame = cap.read()
                if ok:
                    rr.log(f"cameras/{slot}", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            frame_idx += 1
            dt = time.time() - t_loop
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[run] stopping")
    finally:
        if zed is not None:
            zed[0].close()
        for _, cap in innomakers:
            cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
