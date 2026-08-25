#!/usr/bin/env python3
"""Camera-only tag sanity check — no arm, no planner, nothing can move.

    # shell 1: the D405 driver (aligned depth on)
    ros2 launch realsense2_camera rs_launch.py camera_namespace:=d405 \\
        camera_name:=d405 align_depth.enable:=true

    # shell 2 (sourced chain):
    python3 scripts/tag_check.py

Detects the configured tag (id + measured size from the container yaml)
on the live color stream and prints, per sighting: RGB-solve range,
depth-refined range, and the tag's pixel size. Everything is reported
in the CAMERA frame, so this works with the arm off (no TF needed) —
hold the camera or the container by hand. Use it to verify the printed
tag + measured `tag.size_m` before any bench session: the two ranges
should agree within a few mm once the size is right (the RGB range
scales directly with the configured edge; depth does not).

Ctrl+C exits.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402

from rammp_box_opening.models.container import load_press_demo  # noqa: E402
from rammp_box_opening.perception.tag_source import refine_point  # noqa: E402

CONTAINER_YAML = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def main():
    import cv2

    from rammp_curobo_ros.seek_core import D405Grabber
    from rammp_curobo_ros.tags import tag_pose_from_frame

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--container", default=str(CONTAINER_YAML))
    args = ap.parse_args()
    cfg = load_press_demo(args.container)

    rclpy.init()
    node = rclpy.create_node("tag_check")
    grab = D405Grabber(node, need_depth=True)
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        aruco.DetectorParameters(),
    )
    s = cfg.tag_size_m / 2.0
    objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    print(
        "tag_check: id %d, configured edge %.0f mm — Ctrl+C to stop"
        % (cfg.tag_id, cfg.tag_size_m * 1000)
    )
    last_stamp = None
    frames = hits = 0
    last_report = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if grab.color is None or grab.k is None or grab.color_stamp is None:
                if time.monotonic() - last_report > 2.0:
                    last_report = time.monotonic()
                    print("waiting for camera: %s" % ", ".join(grab.missing()))
                continue
            stamp = (grab.color_stamp.sec, grab.color_stamp.nanosec)
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            frames += 1
            hit = tag_pose_from_frame(
                grab.color, detector, objp, grab.k, grab.dist, cfg.tag_id
            )
            if hit is None:
                if time.monotonic() - last_report > 2.0:
                    last_report = time.monotonic()
                    print("no tag id %d (%d frames seen)" % (cfg.tag_id, frames))
                continue
            hits += 1
            _tid, _rot, tvec = hit
            rgb_range = float(tvec[2])
            refined = refine_point(tvec, grab.k, grab.depth)
            px = grab.k[0, 0] * cfg.tag_size_m / rgb_range
            if time.monotonic() - last_report > 0.5:
                last_report = time.monotonic()
                print(
                    "tag id %d: rgb %.3f m | depth %s | ~%.0f px across  "
                    "(%d/%d frames)"
                    % (
                        cfg.tag_id,
                        rgb_range,
                        "%.3f m" % refined[2] if refined is not None else "-- ",
                        px,
                        hits,
                        frames,
                    )
                )
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    print(
        "\n%d/%d frames saw the tag. RGB vs depth range agreeing within a "
        "few mm = printed size and config match." % (hits, frames)
    )


if __name__ == "__main__":
    main()
