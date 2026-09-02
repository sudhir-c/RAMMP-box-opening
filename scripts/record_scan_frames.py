#!/usr/bin/env python3
"""Record wrist-camera frames (+ camera pose) for offline perception work.

    # terminal A (this): record for 40 s
    python3 scripts/record_scan_frames.py
    # terminal B: park the camera over the bench
    ros2 run rammp_box_opening press_demo --execute --detect-only

Nothing here moves the arm — it only subscribes. Frames land in
~/.ros/rammp_box_opening/captures/<stamp>/frame_*.npz with color, depth,
K, and the frame-stamp camera pose (base_link <- camera), i.e. exactly
the inputs top_face_from_depth takes, so the depth detector can be
developed and tuned entirely offline.

    # replay a capture through the CURRENT detector:
    python3 scripts/record_scan_frames.py --analyze <capture-dir>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening")
)

CAPTURES = Path.home() / ".ros" / "rammp_box_opening" / "captures"


def record(seconds, period):
    import rclpy
    from rclpy.node import Node

    from rammp_box_opening.perception.depth_source import camera_pose_at

    rclpy.init()
    node = Node("scan_recorder")
    from rammp_curobo_ros.seek_core import D405Grabber

    grab = D405Grabber(node, need_depth=True)
    out = CAPTURES / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(
        "recording to %s for %.0f s — run --detect-only in another shell"
        % (out, seconds)
    )

    n, last_stamp = 0, None
    t_end = time.monotonic() + seconds
    t_next = 0.0
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        if now < t_next:
            continue
        if grab.depth is None or grab.color is None or grab.k is None:
            continue
        stamp = (grab.color_stamp.sec, grab.color_stamp.nanosec)
        if stamp == last_stamp:
            continue
        cam = camera_pose_at(grab)
        if cam is None:
            continue  # no TF yet (bringup still coming up)
        rot, trans = cam
        np.savez_compressed(
            out / ("frame_%03d.npz" % n),
            color=grab.color,
            depth=grab.depth,
            k=grab.k,
            rot_cam=rot,
            trans_cam=trans,
            stamp=np.array(stamp),
        )
        last_stamp = stamp
        n += 1
        t_next = now + period
        print(
            "  frame %3d  cam_t [%.3f %.3f %.3f]" % (n, trans[0], trans[1], trans[2]),
            flush=True,
        )
    print("done: %d frames in %s" % (n, out))
    rclpy.shutdown()


def analyze(cap_dir):
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception.depth_source import top_face_from_depth
    from rammp_box_opening.worlds import WorldStore

    repo = Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening"
    model = ContainerModel.load(str(repo / "config/containers/oxo_pop.yaml"))
    table_z = WorldStore(str(repo / "config/world_bench.yaml")).table_top_z
    frames = sorted(Path(cap_dir).glob("frame_*.npz"))
    print("%d frames | table_z %.3f" % (len(frames), table_z))
    hits = 0
    for f in frames:
        d = np.load(f)
        fix, why = top_face_from_depth(
            d["depth"], d["k"], d["rot_cam"], d["trans_cam"], table_z, model
        )
        if fix is None:
            print("  %-14s -                     (%s)" % (f.stem, why))
            continue
        hits += 1
        print(
            "  %-14s top [%.3f %.3f %.3f] yaw %5.1f  %.3fx%.3f m  %4d px"
            % (
                f.stem,
                *fix.center,
                np.degrees(fix.yaw),
                *fix.footprint,
                fix.n_px,
            )
        )
    print("hit rate: %d/%d" % (hits, len(frames)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--period", type=float, default=0.4)
    ap.add_argument("--analyze", metavar="DIR", help="replay a capture offline")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        record(args.seconds, args.period)


if __name__ == "__main__":
    main()
