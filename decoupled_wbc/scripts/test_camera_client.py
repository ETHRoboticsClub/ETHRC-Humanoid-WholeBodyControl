"""
Test script to verify the RealSense camera ZMQ connection from the client side.
Run this inside the Docker container on the main PC.

Usage:
    python decoupled_wbc/scripts/test_camera_client.py --host 192.168.123.164
"""
import argparse
import os
import time

import cv2

from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor


def main():
    parser = argparse.ArgumentParser(description="Test camera client connection")
    parser.add_argument("--host", type=str, default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    print(f"Connecting to camera server at {args.host}:{args.port}...")
    client = ComposedCameraClientSensor(server_ip=args.host, port=args.port)

    print("Waiting for first frame...")
    data = client.read()

    if data is None or not data.get("images"):
        print("ERROR: No image received.")
        client.close()
        return

    for key, img in data["images"].items():
        print(f"Received '{key}': shape={img.shape}, dtype={img.dtype}")
        latency_ms = (time.time() - data["timestamps"].get(key, 0)) * 1000
        print(f"Latency: {latency_ms:.0f}ms")

    # Save first image
    first_key = next(iter(data["images"]))
    img = data["images"][first_key]
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image.png")
    cv2.imwrite(save_path, img[..., ::-1])
    print(f"Saved: {save_path}")

    client.close()
    print("OK")


if __name__ == "__main__":
    main()
