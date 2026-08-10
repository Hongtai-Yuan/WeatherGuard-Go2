#!/usr/bin/env python3
"""Wait until the Go2 map->base_link TF is available."""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_frame", default="map")
    parser.add_argument("--source_frame", default="unitree_go2_0/base_link")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("wait_for_go2_tf")
    buffer = Buffer()
    TransformListener(buffer, node)

    deadline = time.time() + args.timeout
    ok = False
    while rclpy.ok() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            buffer.lookup_transform(
                args.target_frame,
                args.source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            ok = True
            break
        except Exception:
            pass

    node.destroy_node()
    rclpy.shutdown()
    if ok:
        print(f"TF ready: {args.target_frame} <- {args.source_frame}", flush=True)
        return 0
    print(f"Timed out waiting for TF: {args.target_frame} <- {args.source_frame}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
