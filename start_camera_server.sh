#!/usr/bin/env bash
# Start a generic composed camera ZMQ server on the robot.
# Usage:
#   ./start_camera_server.sh [--port PORT] [--ego-camera TYPE] [--ego-device-id ID]
#                        [--left-camera TYPE] [--left-device-id ID]
#                        [--right-camera TYPE] [--right-device-id ID]
#                        [--head-camera TYPE] [--head-device-id ID]
# Supported types: oak, oak_mono, realsense, zed, usb

set -euo pipefail

REPO_DIR="/home/unitree/GR00T-WholeBodyControl"
PORT=5555
EGO_CAMERA="oak"
EGO_DEVICE_ID=""
LEFT_CAMERA=""
LEFT_DEVICE_ID=""
RIGHT_CAMERA=""
RIGHT_DEVICE_ID=""
HEAD_CAMERA=""
HEAD_DEVICE_ID=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --port PORT                 Camera server TCP port (default: 5555)
  --ego-camera TYPE           Ego-view camera type (oak, oak_mono, realsense, zed, usb)
  --ego-device-id ID          Ego-view device ID (MxID or /dev/video index)
  --left-camera TYPE          Left wrist camera type
  --left-device-id ID         Left wrist device ID
  --right-camera TYPE         Right wrist camera type
  --right-device-id ID        Right wrist device ID
  --head-camera TYPE          Head camera type
  --head-device-id ID         Head camera device ID
  --repo-dir PATH             Repository root (default: $REPO_DIR)
  -h, --help                  Show this help message

Example:
  $0 --ego-camera zed --left-camera oak --left-device-id 18443010.. \
     --right-camera oak --right-device-id 18443010.. --port 5555
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --ego-camera)
            EGO_CAMERA="$2"
            shift 2
            ;;
        --ego-device-id)
            EGO_DEVICE_ID="$2"
            shift 2
            ;;
        --left-camera)
            LEFT_CAMERA="$2"
            shift 2
            ;;
        --left-device-id)
            LEFT_DEVICE_ID="$2"
            shift 2
            ;;
        --right-camera)
            RIGHT_CAMERA="$2"
            shift 2
            ;;
        --right-device-id)
            RIGHT_DEVICE_ID="$2"
            shift 2
            ;;
        --head-camera)
            HEAD_CAMERA="$2"
            shift 2
            ;;
        --head-device-id)
            HEAD_DEVICE_ID="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

PYTHON_CMD="$REPO_DIR/.venv_camera/bin/python"
if [[ ! -x "$PYTHON_CMD" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="$(command -v python)"
    else
        echo "ERROR: Python not found. Install Python 3 or create .venv_camera."
        exit 1
    fi
fi

reset_realsense() {
    echo "[2/3] Resetting RealSense hardware..."
    "$PYTHON_CMD" - <<'PY'
import pyrealsense2 as rs
import time
ctx = rs.context()
devs = ctx.query_devices()
if len(devs) == 0:
    print('  No RealSense device found! Check USB connection.')
    raise SystemExit(1)
for d in devs:
    name = d.get_info(rs.camera_info.name)
    print(f'  Resetting {name}...')
    d.hardware_reset()
time.sleep(4)
print('  Reset complete.')
PY
}

if [[ "$EGO_CAMERA" == "realsense" || "$LEFT_CAMERA" == "realsense" || "$RIGHT_CAMERA" == "realsense" || "$HEAD_CAMERA" == "realsense" ]]; then
    echo "[1/3] Releasing RealSense camera processes..."
    pkill -9 -f videohub_pc4 2>/dev/null && echo "  Killed videohub_pc4" || true
    pkill -9 -f "realsense.*--server" 2>/dev/null && echo "  Killed old realsense server" || true
    sleep 1
    reset_realsense
fi

cd "$REPO_DIR"
echo "[3/3] Starting composed camera server on port $PORT"
EXEC_ARGS=(--ego-view-camera "$EGO_CAMERA")
if [[ -n "$EGO_DEVICE_ID" ]]; then
    EXEC_ARGS+=(--ego-view-device-id "$EGO_DEVICE_ID")
fi
if [[ -n "$LEFT_CAMERA" ]]; then
    EXEC_ARGS+=(--left-wrist-camera "$LEFT_CAMERA")
    if [[ -n "$LEFT_DEVICE_ID" ]]; then
        EXEC_ARGS+=(--left-wrist-device-id "$LEFT_DEVICE_ID")
    fi
fi
if [[ -n "$RIGHT_CAMERA" ]]; then
    EXEC_ARGS+=(--right-wrist-camera "$RIGHT_CAMERA")
    if [[ -n "$RIGHT_DEVICE_ID" ]]; then
        EXEC_ARGS+=(--right-wrist-device-id "$RIGHT_DEVICE_ID")
    fi
fi
if [[ -n "$HEAD_CAMERA" ]]; then
    EXEC_ARGS+=(--head-camera "$HEAD_CAMERA")
    if [[ -n "$HEAD_DEVICE_ID" ]]; then
        EXEC_ARGS+=(--head-device-id "$HEAD_DEVICE_ID")
    fi
fi
EXEC_ARGS+=(--port "$PORT")

printf '  %s\n' "${EXEC_ARGS[@]}"

echo "  Press Ctrl+C to stop."

exec "$PYTHON_CMD" -m gear_sonic.camera.composed_camera "${EXEC_ARGS[@]}"
