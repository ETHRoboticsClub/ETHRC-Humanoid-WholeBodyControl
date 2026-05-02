#!/bin/bash
set -e  # Exit on error

# Reinstall robocasa from ETHRC's copy so that `import robocasa` resolves to
# the volume-mounted local files. The pre-built image has GR00T's robocasa
# installed; by switching to an editable install of ETHRC's copy, all local
# changes to gr00trobocasa are immediately reflected without container restarts.
ETHRC_ROBOCASA="/root/Projects/ETHRC-Humanoid-WholeBodyControl/decoupled_wbc/dexmg/gr00trobocasa"
if [ -d "$ETHRC_ROBOCASA" ]; then
    echo "Installing robocasa from ETHRC (editable)..."
    /root/.cargo/bin/uv pip install -e "$ETHRC_ROBOCASA" --quiet
fi

echo "Dependencies installed successfully. Starting interactive bash shell..."
exec /bin/bash
