#!/usr/bin/env bash
# Creates the lerobot-trim conda environment used by the trim dashboard.
# Run once from any directory:
#   bash trim_dashboard/setup_env.sh

set -e

ENV_NAME="lerobot-trim"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — skipping creation."
    echo "To rebuild it from scratch: conda env remove -n ${ENV_NAME} && bash trim_dashboard/setup_env.sh"
    exit 0
fi

echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -y -n "${ENV_NAME}" python=3.10

echo "Installing dependencies..."
conda run -n "${ENV_NAME}" pip install \
    flask \
    pyarrow \
    numpy \
    huggingface_hub

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
