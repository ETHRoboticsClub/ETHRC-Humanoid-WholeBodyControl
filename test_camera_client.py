"""
Test script to verify the RealSense camera ZMQ connection from the client side.
Run this inside the Docker container on the main PC.

Usage:
    # Print stats only:
    python decoupled_wbc/scripts/test_camera_client.py --host 192.168.123.164

    # Show live video feed (requires display, press 'q' to quit):
    python decoupled_wbc/scripts/test_camera_client.py --host 192.168.123.164 --show
"""
import argparse
import time

import cv2
import numpy as np

from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor


def main():
    parser = argparse.ArgumentParser(description="Test camera client connection")
    parser.add_argument("--host", type=str, default="192.168.123.164",
                        help="Camera server IP (Orin)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Number of frames to read (ignored with --show)")
    parser.add_argument("--show", action="store_true",
                        help="Show live video feed (press 'q' to quit)")
    args = parser.parse_args()

    print(f"Connecting to camera server at {args.host}:{args.port}...")
    client = ComposedCameraClientSensor(server_ip=args.host, port=args.port)

    if args.show:
        print("Showing live feed (press 'q' to quit)...")
        try:
            while True:
                data = client.read()
                if data is None:
                    continue
                for key, img in data["images"].items():
                    bgr = img[..., ::-1]  # RGB -> BGR for OpenCV display
                    latency_ms = (time.time() - data["timestamps"].get(key, 0)) * 1000
                    cv2.putText(bgr, f"FPS: {client.fps():.1f} | Latency: {latency_ms:.0f}ms",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(key, bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            client.close()
        return

    # Default: print stats for N frames
    print(f"Reading {args.num_frames} frames...\n")
    for i in range(args.num_frames):
        data = client.read()
        if data is None:
            print(f"  Frame {i}: No data received")
            continue

        parts = []
        for key, img in data["images"].items():
            parts.append(f"{key}: {img.shape} {img.dtype}")
        ts_parts = []
        for key, t in data.get("timestamps", {}).items():
            latency_ms = (time.time() - t) * 1000
            ts_parts.append(f"{key}: {latency_ms:.0f}ms latency")

        print(f"  Frame {i:3d} | {', '.join(parts)} | {', '.join(ts_parts)} | "
              f"client FPS: {client.fps():.1f}")

    print(f"\nDone. Average client FPS: {client.fps():.1f}")
    client.close()


if __name__ == "__main__":
    main()
