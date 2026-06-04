#!/usr/bin/env bash
# check_camera_server.sh
# Read-only preflight for the SONIC unified camera server (ego ZED + 2 wrist USB)
# on the G1 Orin. Run AFTER a boot / before trusting the server for data
# collection. systemctl reports the unit "active" even when it publishes NOTHING
# (publish is all-or-nothing across cameras, and a dead camera does not crash the
# process), so "active" cannot be trusted — this script verifies frames actually
# flow on port 5555.
#
# Usage:  bash check_camera_server.sh        (run on the Orin)
# Exit:   0 = all required checks pass, 1 = at least one required check failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY="$REPO_ROOT/.venv_camera/bin/python"
PORT=5555
SMPL_PORT=5556
FAILS=0

pass() { echo "[OK]   $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; FAILS=$((FAILS + 1)); }

echo "── camera-server preflight ($(hostname)) ──"

# 1. ZED-M video interface enumerated (only happens on USB 3.0; on USB 2.0 just
#    the HID interface 2b03:f681 shows, and the SDK fails to start the stream).
if lsusb | grep -q "2b03:f682"; then
    pass "ZED-M camera interface enumerated (2b03:f682) — on USB 3.0."
elif lsusb | grep -q "2b03:f681"; then
    fail "ZED-M only shows its HID interface (2b03:f681), no video (2b03:f682). It is on USB 2.0 / not seated — replug into a USB 3.0 port."
else
    fail "No ZED-M detected on USB at all (expected 2b03:f682). Check the cable."
fi

# 2. pyzed importable in the camera venv (gets wiped when .venv_camera is
#    recreated by install_camera_server.sh — it is not part of gear_sonic[camera]).
if [ ! -x "$VPY" ]; then
    fail ".venv_camera python not found at $VPY (venv missing?)."
else
    if SDK=$("$VPY" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null); then
        pass "pyzed importable in .venv_camera (ZED SDK $SDK)."
    else
        fail "pyzed NOT importable in .venv_camera — ZED ego_view will fail. Restore it (offline: copy cp310 pyzed from ~/.cache/uv into the venv; online: run /usr/local/zed/get_python_api.py with .venv_camera/bin/python)."
    fi
fi

# 3. Wrist video nodes present (left=/dev/video6, right=/dev/video2).
for node in /dev/video0 /dev/video2; do
    if [ -e "$node" ]; then
        pass "wrist node $node present."
    else
        fail "wrist node $node missing (USB re-enumeration? check the Innomaker cams)."
    fi
done

# 4. Service active + listening on 5555 only, 5556 left free for the SMPL stream.
if [ "$(systemctl is-active composed_camera_server.service 2>/dev/null)" = "active" ]; then
    pass "composed_camera_server.service is active."
else
    fail "composed_camera_server.service is not active."
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    pass "listener bound on $PORT."
else
    fail "nothing listening on $PORT."
fi
if ss -ltn 2>/dev/null | grep -q ":$SMPL_PORT "; then
    warn "something is listening on $SMPL_PORT — must be free for the SMPL pose stream. Verify it is not a camera server."
else
    pass "$SMPL_PORT free for the SMPL pose stream."
fi

# 5. THE definitive check: the server actually publishes all required image keys
#    with non-blank frames. This is what "active" cannot tell you.
if [ -x "$VPY" ]; then
    echo "[INFO] probing published streams on tcp://127.0.0.1:$PORT (up to ~20s) …"
    # 2>&1 so a probe crash (import/ZMQ error) is captured for the diagnostic below.
    PROBE_RAW=$(timeout 30 "$VPY" - "$PORT" <<'PY' 2>&1
import sys, time
import numpy as np
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
port = int(sys.argv[1])
c = ComposedCameraClientSensor(server_ip="127.0.0.1", port=port)
out = "NOFRAME"
for _ in range(200):
    s = c.read(blocking=False)
    if s and s.get("images"):
        parts = []
        for k in ("ego_view", "left_wrist", "right_wrist"):
            v = s["images"].get(k)
            if v is None:
                parts.append(f"{k}:MISSING")
            else:
                a = np.asarray(v)
                parts.append(f"{k}:{'OK' if a.any() else 'BLANK'}")
        out = " ".join(parts)
        break
    time.sleep(0.1)
c.close()
print("RESULT " + out)   # marker so the caller ignores the client's own stdout
PY
)
    # The client sensor prints a banner to stdout on init; pull out only our marked line.
    PROBE=$(printf '%s\n' "$PROBE_RAW" | sed -n 's/^RESULT //p')
    if [ "$PROBE" = "ego_view:OK left_wrist:OK right_wrist:OK" ]; then
        pass "all three required streams publishing, non-blank: $PROBE"
    elif [ -z "$PROBE" ]; then
        # No RESULT line at all → the probe itself failed (broken venv / import error),
        # NOT a camera problem. Don't send the operator to replug USB cables.
        fail "camera probe produced no result — likely a .venv_camera / import error, not the cameras. Last output: $(printf '%s' "$PROBE_RAW" | tail -n1)"
    elif [ "$PROBE" = "NOFRAME" ]; then
        fail "server published NO frames in ~20s (all-or-nothing — at least one camera is down)."
    else
        fail "required streams incomplete/blank: $PROBE"
    fi
fi

echo "────────────────────────────────────────────"
if [ "$FAILS" -eq 0 ]; then
    echo "[OK] camera server preflight PASSED — safe to start data collection."
    exit 0
else
    echo "[FAIL] $FAILS check(s) failed — do NOT trust the camera server until fixed."
    exit 1
fi
