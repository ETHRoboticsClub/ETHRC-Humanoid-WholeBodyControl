"""
PC-side companion for run_rerun_camera_publisher.py running on the G1 Orin.

Spawns a Rerun viewer that listens on a gRPC port for incoming camera streams,
preconfigures a 4-pane layout (ZED left, ZED right, two wrist cameras), and
stays alive so you can Ctrl+C cleanly.

This file is meant to live on the *PC* (the LAN-connected machine), not on
the G1.

Usage (on the PC):

  pip install -e "gear_sonic[viz]"
  python gear_sonic/scripts/run_rerun_viewer.py                # binds 0.0.0.0:9876
  python gear_sonic/scripts/run_rerun_viewer.py --port 9876    # explicit port

Then SSH to the G1 and launch the publisher:

  sg zed -c "python3 gear_sonic/scripts/run_rerun_camera_publisher.py --rerun-host <THIS-PC-IP>"

Notes:
  - `rr.spawn()` needs a local display. If you SSH'd into this workstation
    without X forwarding, the viewer will fail to open — run this script
    from a graphical session (or run the standalone `rerun --port 9876`
    binary locally and skip this helper).
  - The Rerun gRPC port (default 9876) must be reachable from the G1. On a
    fresh Ubuntu workstation `ufw` is usually off; if it's on, allow
    inbound from the robot subnet: `sudo ufw allow from 192.168.123.0/24 to any port 9876 proto tcp`.
  - The `--session` value must match what the publisher passes for the
    panes to appear in the same recording.
"""

import argparse
import os
import socket
import sys
import time

import rerun as rr
import rerun.blueprint as rrb


def _ensure_rerun_bin_on_path() -> None:
    """`rr.spawn()` shells out to the `rerun` viewer binary that pip installs
    alongside the rerun_sdk Python package (typically <venv>/bin/rerun). When
    this script is invoked via an absolute interpreter path (e.g.
    `.venv_data_collection/bin/python gear_sonic/scripts/run_rerun_viewer.py`)
    the venv's bin/ is NOT on PATH, so the binary is invisible and rr.spawn()
    fails with "Failed to find Rerun Viewer executable in PATH".

    Prepend sys.executable's directory before spawning. Idempotent if the
    user already activated the venv.
    """
    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[listener] prepended {bin_dir} to PATH for rr.spawn", flush=True)


def get_local_lan_ips() -> list[str]:
    """Best-effort: pick LAN-routable IPs to show the user.
    Returns IPs the G1 should be able to reach.
    """
    ips: list[str] = []
    try:
        # Trick: opening a UDP socket to a public IP forces the kernel to
        # populate the local address with the IP it would use for that route,
        # without actually sending anything. Repeat for both common LAN
        # gateways so we surface both the wired and the wifi address.
        for probe in ("8.8.8.8", "192.168.123.1", "10.5.12.1"):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.1)
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                s.close()
                if ip and ip not in ips and not ip.startswith("127."):
                    ips.append(ip)
            except OSError:
                pass
    except Exception:
        pass
    return ips


def build_blueprint() -> rrb.Blueprint:
    """A 2×2 grid: ZED left | ZED right / Wrist 0 | Wrist 1.

    `Spatial2DView` is the right view type for plain camera frames. The
    `origin` is the entity-path the G1 publisher logs each image under
    (see run_rerun_camera_publisher.py).
    """
    return rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin="cameras/zed/left",  name="ZED Left"),
            rrb.Spatial2DView(origin="cameras/zed/right", name="ZED Right"),
            rrb.Spatial2DView(origin="cameras/wrist_0",   name="Wrist 0"),
            rrb.Spatial2DView(origin="cameras/wrist_1",   name="Wrist 1"),
            grid_columns=2,
        ),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9876,
                        help="gRPC port the viewer listens on for the G1 publisher (default: 9876)")
    parser.add_argument("--session", default="g1_cameras",
                        help="must match the --session passed to run_rerun_camera_publisher.py on the G1")
    args = parser.parse_args()

    # 1. Initialise a recording stream with the session name. The viewer
    #    groups all publishers that share this name under a single recording.
    rr.init(args.session)

    # 2. Spawn the rerun viewer in a detached subprocess. It listens on
    #    0.0.0.0:<port> for incoming gRPC connections from any publisher on
    #    the LAN, and our Python here is also auto-connected to it so we can
    #    push the blueprint.
    _ensure_rerun_bin_on_path()
    print(f"[listener] spawning viewer on port {args.port} ...", flush=True)
    rr.spawn(port=args.port)

    # 3. Push the layout so the panes appear named and pre-arranged before
    #    the G1's first frame lands.
    rr.send_blueprint(build_blueprint())

    # 4. Tell the user where the G1 should point.
    ips = get_local_lan_ips()
    print(f"[listener] viewer is up, listening on 0.0.0.0:{args.port}", flush=True)
    if ips:
        for ip in ips:
            print(f"             on the G1, set --rerun-host {ip} --rerun-port {args.port}", flush=True)
    else:
        print("             (couldn't auto-detect LAN IPs; run `ip addr` and use the one on the G1's subnet)", flush=True)
    print(f"[listener] session name = {args.session!r} (must match --session on the G1)", flush=True)
    print("[listener] Ctrl+C exits this script; the viewer stays open until you close its window.", flush=True)

    # 5. Heartbeat. The viewer is a detached subprocess, so we just need
    #    something keeping this script alive so the user can stop it cleanly.
    try:
        n = 0
        while True:
            time.sleep(10)
            n += 1
            print(f"[listener] alive ({n*10}s)", flush=True)
    except KeyboardInterrupt:
        print("\n[listener] exiting — viewer window left running", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
