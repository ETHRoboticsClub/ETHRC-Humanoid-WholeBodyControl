#!/usr/bin/env bash
# install_camera_server.sh
# Sets up the .venv_camera venv for running the composed camera server
# on the robot computer.
#
# Installs gear_sonic[camera] which includes the ZMQ-based camera server
# framework and the depthai SDK (OAK cameras). For other camera SDKs
# (e.g. pyrealsense2), install them into the venv after setup.
#
# Usage:  bash install_scripts/install_camera_server.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 0. Print detected architecture ───────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 ─────────────────────────────────────
echo "[INFO] Installing uv-managed Python 3.10 …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project --managed-python 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ─────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_camera (if present) …"
rm -rf .venv_camera

# ── 4. Create venv & install camera extra ────────────────────────────────────
echo "[INFO] Creating .venv_camera with uv-managed Python 3.10 …"
uv venv .venv_camera --python "$MANAGED_PY" --prompt gear_sonic_camera
# shellcheck disable=SC1091
source .venv_camera/bin/activate
echo "[INFO] Installing gear_sonic[camera] …"
uv pip install -e "gear_sonic[camera]"

# ── 4b. Restore pyzed (ZED Python SDK) — only if the ZED SDK is installed ─────
# gear_sonic[camera] does NOT include pyzed; it is an out-of-band ZED SDK install
# that the `rm -rf .venv_camera` above wipes on every re-run. The robot subnet
# usually cannot reach download.stereolabs.com, so prefer the local uv wheel
# cache (offline), then fall back to the ZED SDK's get_python_api.py, and WARN
# loudly if neither works (fine for OAK/RealSense ego cameras).
restore_pyzed() {
    local vpy="$REPO_ROOT/.venv_camera/bin/python"
    local sp
    sp="$("$vpy" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)" || return 1

    # Installed ZED SDK version that pyzed MUST match (e.g. 5.3.0). A pyzed built
    # for a different SDK can import fine and then garble/return wrong frames at
    # runtime — the exact silent-blank trap we are guarding against.
    local sdk_ver=""
    if [ -f /usr/local/zed/zed-config-version.cmake ]; then
        sdk_ver="$(grep -oE 'PACKAGE_VERSION "[0-9.]+"' /usr/local/zed/zed-config-version.cmake \
                   | grep -oE '[0-9.]+' | head -n1 || true)"
    fi
    matches() { [ -z "$sdk_ver" ] || [ "$1" = "$sdk_ver" ]; }  # dynamic scope sees $sdk_ver

    # Already importable and version-matching? Done.
    local cur
    cur="$("$vpy" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null || true)"
    if [ -n "$cur" ] && matches "$cur"; then
        echo "[OK] pyzed already importable (SDK $cur)."; return 0
    fi
    [ -n "$cur" ] && echo "[WARN] pyzed present but reports SDK '$cur' != installed '$sdk_ver' — replacing."

    # Offline: pick the cp310 cache build whose SDK version matches the installed SDK.
    local so src chosen="" v
    while IFS= read -r so; do
        [ -n "$so" ] || continue
        src="$(dirname "$(dirname "$so")")"   # .../<archive>/pyzed/sl…so -> <archive>
        [ -d "$src/pyzed" ] || continue
        v="$(PYTHONPATH="$src" "$vpy" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null || true)"
        if [ -n "$v" ] && matches "$v"; then chosen="$src"; break; fi
    done < <(find "$HOME/.cache/uv" -path '*pyzed*' -name 'sl.cpython-310-*-linux-gnu.so' 2>/dev/null || true)

    if [ -n "$chosen" ]; then
        rm -rf "$sp/pyzed" "$sp"/pyzed-*.dist-info 2>/dev/null || true
        cp -r "$chosen/pyzed" "$sp/" 2>/dev/null || true
        cp -r "$chosen"/pyzed-*.dist-info "$sp/" 2>/dev/null || true
        local nv
        nv="$("$vpy" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null || true)"
        if [ -n "$nv" ] && matches "$nv"; then
            echo "[OK] pyzed restored from uv cache (SDK $nv): $chosen"; return 0
        fi
    fi

    # Online fallback: ZED SDK's installer (needs pip + reachable Stereolabs).
    if [ -f /usr/local/zed/get_python_api.py ]; then
        "$vpy" -m ensurepip >/dev/null 2>&1 || true
        if "$vpy" /usr/local/zed/get_python_api.py >/dev/null 2>&1; then
            nv="$("$vpy" -c 'import pyzed.sl as sl; print(sl.Camera.get_sdk_version())' 2>/dev/null || true)"
            if [ -n "$nv" ] && matches "$nv"; then
                echo "[OK] pyzed installed via /usr/local/zed/get_python_api.py (SDK $nv)."; return 0
            fi
        fi
    fi
    return 1
}

if [ -d /usr/local/zed ]; then
    echo "[INFO] ZED SDK present — ensuring pyzed is in .venv_camera (for a ZED ego camera) …"
    if ! restore_pyzed; then
        echo "[WARN] pyzed: expected=importable in .venv_camera, got=missing, fallback=skipped."
        echo "       A ZED ego_view camera will FAIL until pyzed is installed. Offline:"
        echo "       copy a cp310 pyzed from ~/.cache/uv into the venv site-packages;"
        echo "       online: run /usr/local/zed/get_python_api.py with .venv_camera/bin/python."
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Camera server venv setup complete!"
echo "  depthai (OAK cameras) is included by default."
echo ""
echo "  Activate the venv with:"
echo "    source .venv_camera/bin/activate"
echo ""
echo "  For other camera SDKs, install into the venv:"
echo "    pip install pyrealsense2     # Intel RealSense"
echo ""
echo "  See docs/source/tutorials/data_collection.md for full setup."
echo "══════════════════════════════════════════════════════════════"

# ── 5. Optionally install the systemd service ────────────────────────────────
SERVICE_TEMPLATE="$REPO_ROOT/systemd/composed_camera_server.service"
SERVICE_NAME="composed_camera_server.service"

if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo ""
    echo "[WARN] systemd template not found at $SERVICE_TEMPLATE — skipping."
    exit 0
fi

echo ""
read -rp "Install the camera server as a systemd service (auto-start on boot)? [y/N] " INSTALL_SERVICE
if [[ ! "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "  Skipped systemd install. You can run the camera server manually:"
    echo "    source .venv_camera/bin/activate"
    echo "    python -m gear_sonic.camera.composed_camera --ego-view-camera oak --port 5555"
    echo ""
    exit 0
fi

# Gather configuration
echo ""
echo "── Camera service configuration ──"
echo "  Each camera needs a type and a device ID so the server knows"
echo "  which physical camera maps to each mount position (ego, wrist, etc.)."
echo ""

detect_oak_cameras() {
    "${REPO_ROOT}/.venv_camera/bin/python" -c "
import depthai as dai
devices = dai.Device.getAllAvailableDevices()
if not devices:
    exit(1)
for i, d in enumerate(devices):
    # Try to get actual MxID; fall back to string representation
    mxid = None
    for attr in ['mxid', 'getMxId']:
        if hasattr(d, attr):
            val = getattr(d, attr)
            mxid = val() if callable(val) else val
            if mxid:
                break
    state = d.state.name if hasattr(d, 'state') else 'N/A'
    name = getattr(d, 'name', '')
    if mxid and mxid != name:
        print(f'    [{i}] MxId: {mxid}  port: {name}  state: {state}')
    else:
        # MxID not available; show all useful attributes
        print(f'    [{i}] device: {d}  state: {state}')
        print(f'         attributes: {[a for a in dir(d) if not a.startswith(\"_\")]}')
" 2>&1
}

while true; do
    echo "  Detecting connected OAK cameras …"
    OAK_DEVICES="$(detect_oak_cameras)" && OAK_FOUND=true || OAK_FOUND=false

    if $OAK_FOUND && [ -n "$OAK_DEVICES" ]; then
        echo "$OAK_DEVICES"
        break
    else
        echo "  (no OAK devices detected)"
        if [ -n "$OAK_DEVICES" ]; then
            echo "  depthai output: $OAK_DEVICES"
        fi
        echo ""
        read -rp "  Retry detection? [Y/n] (or 'n' to enter device IDs manually): " RETRY
        if [[ "$RETRY" =~ ^[Nn]$ ]]; then
            break
        fi
        echo ""
    fi
done
echo ""

# Build ExecStart args incrementally
CAMERA_ARGS=""

# --- Ego-view camera (required) ---
read -rp "  Ego-view camera type (oak, oak_mono, realsense, usb) [oak]: " EGO_TYPE
EGO_TYPE="${EGO_TYPE:-oak}"
read -rp "  Ego-view device ID (MxID or /dev/video index): " EGO_DEVICE_ID
CAMERA_ARGS="--ego-view-camera ${EGO_TYPE}"
if [ -n "$EGO_DEVICE_ID" ]; then
    CAMERA_ARGS="${CAMERA_ARGS} --ego-view-device-id ${EGO_DEVICE_ID}"
fi

# --- Left wrist camera (optional) ---
echo ""
read -rp "  Add a left-wrist camera? [y/N]: " ADD_LEFT
if [[ "$ADD_LEFT" =~ ^[Yy]$ ]]; then
    read -rp "  Left-wrist camera type (oak, oak_mono, realsense, usb) [oak]: " LEFT_TYPE
    LEFT_TYPE="${LEFT_TYPE:-oak}"
    read -rp "  Left-wrist device ID (MxID or /dev/video index): " LEFT_DEVICE_ID
    CAMERA_ARGS="${CAMERA_ARGS} --left-wrist-camera ${LEFT_TYPE}"
    if [ -n "$LEFT_DEVICE_ID" ]; then
        CAMERA_ARGS="${CAMERA_ARGS} --left-wrist-device-id ${LEFT_DEVICE_ID}"
    fi
fi

# --- Right wrist camera (optional) ---
echo ""
read -rp "  Add a right-wrist camera? [y/N]: " ADD_RIGHT
if [[ "$ADD_RIGHT" =~ ^[Yy]$ ]]; then
    read -rp "  Right-wrist camera type (oak, oak_mono, realsense, usb) [oak]: " RIGHT_TYPE
    RIGHT_TYPE="${RIGHT_TYPE:-oak}"
    read -rp "  Right-wrist device ID (MxID or /dev/video index): " RIGHT_DEVICE_ID
    CAMERA_ARGS="${CAMERA_ARGS} --right-wrist-camera ${RIGHT_TYPE}"
    if [ -n "$RIGHT_DEVICE_ID" ]; then
        CAMERA_ARGS="${CAMERA_ARGS} --right-wrist-device-id ${RIGHT_DEVICE_ID}"
    fi
fi

echo ""
read -rp "  ZMQ port [5555]: " CFG_PORT
CFG_PORT="${CFG_PORT:-5555}"
CAMERA_ARGS="${CAMERA_ARGS} --port ${CFG_PORT}"

EXEC_START="${REPO_ROOT}/.venv_camera/bin/python -m gear_sonic.camera.composed_camera ${CAMERA_ARGS}"
echo ""
echo "  ExecStart command:"
echo "    $EXEC_START"
echo ""
read -rp "  Look correct? [Y/n]: " CONFIRM
if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
    echo "  Aborted. Edit systemd/composed_camera_server.service manually."
    exit 0
fi

# Generate unit file directly (avoids fragile sed on multi-line ExecStart)
TMPUNIT="$(mktemp)"
cat > "$TMPUNIT" <<UNIT
[Unit]
Description=SONIC Composed Camera Server (ZMQ)
After=network.target

[Service]
Type=simple
User=$USER
Environment="HOME=$HOME"
Environment="REPO_DIR=$REPO_ROOT"
WorkingDirectory=$REPO_ROOT
ExecStart=$EXEC_START
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

echo ""
echo "[INFO] Installing systemd service …"
sudo cp "$TMPUNIT" "/etc/systemd/system/$SERVICE_NAME"
rm -f "$TMPUNIT"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  systemd service installed and started!"
echo ""
echo "  Check status:"
echo "    sudo systemctl status $SERVICE_NAME"
echo ""
echo "  View logs:"
echo "    journalctl -u $SERVICE_NAME -f"
echo ""
echo "  To reconfigure, edit and re-run this script, or:"
echo "    sudo systemctl edit $SERVICE_NAME"
echo "    sudo systemctl restart $SERVICE_NAME"
echo "══════════════════════════════════════════════════════════════"
