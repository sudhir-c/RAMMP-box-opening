#!/usr/bin/env python3
"""Run Chris's depth box detector live, with nothing that can move the arm.

No planner, no executor, no GPU, no OWL: just BoxTopWatcher (depth plateau ->
TF lift -> footprint gates -> button circle) on the live camera, printing
every committed fix and, while it refuses, WHY. The arm must simply hold a
tool-down pose ~35-45 cm above the box; position it beforehand with whatever
you like (teleop, presets) -- this script never commands it.

Needs on the ROS graph:
  - colour + aligned depth + camera_info on the topics named in
    ~/RAMMP-CuRobo/rammp_curobo_ros/config/camera_d405_wrist.yaml
  - a TF from base_link to that file's parent_frame at the frame stamp
    (any robot_state_publisher + joint states; the feeding stack's
    microwave_bringup.launch.py provides both, plus the hand-eye TF)

Copy into ~/RAMMP-box-opening/scripts/ and run from the repo root with the
venv active and the CuRobo + box-opening installs sourced:

    python3 scripts/detect_standalone.py --table-z -0.03          # your tape-measured table top, base_link metres
    python3 scripts/detect_standalone.py --table-z -0.03 --seconds 60
    python3 scripts/detect_standalone.py --table-z -0.03 --no-circle   # accept plateau fixes without the button circle

Read the "reject:" lines: "camera moving" = TF jitter or the arm is not
still; "no points at container height" = table_z wrong or box out of view;
"footprint" = a plateau of the wrong size; "no circle" = lid found, button
seam not readable (lighting) -- try --no-circle to see the plateau fix.
"""

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-z", type=float, required=True, help="table TOP height in base_link (m); tape-measure it")
    ap.add_argument("--container", default=str(REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no-circle", action="store_true", help="let plateau-only sightings commit (diagnostic)")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node

    from rammp_box_opening.models.container import ContainerModel, load_press_demo
    from rammp_box_opening.perception.depth_source import BoxTopWatcher

    model = ContainerModel.load(args.container)
    cfg = load_press_demo(args.container)
    print("box model: %.3f x %.3f x %.3f m, button %.0f mm; table_z %.3f -> expect lid top at z %.3f"
          % (*model.dims, model.button_diameter_m * 1000, args.table_z, args.table_z + model.dims[2]))
    print("commit rule: %d agreeing frames within %.0f mm, fresh < %.1f s%s"
          % (cfg.min_hits, cfg.tol_m * 1000, cfg.fresh_s, "" if not args.no_circle else "  (circle NOT required)"))

    rclpy.init()
    node = Node("detect_standalone")
    watcher = BoxTopWatcher(node, cfg, model, args.table_z)
    if args.no_circle:
        watcher.require_circle = False
    watcher.active = True

    t0 = time.monotonic()
    last_fix, last_reject, last_report = None, None, 0.0
    try:
        while time.monotonic() - t0 < args.seconds and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            got = watcher.fix()
            if got is not None:
                pos, yaw = got
                key = tuple(round(float(v), 4) for v in pos)
                if key != last_fix:
                    last_fix = key
                    f = watcher.last_debug
                    print("FIX  top [%.3f %.3f %.3f]  yaw %5.1f deg  footprint %.3fx%.3f m  %d px  | lid top %+.1f mm vs nominal"
                          % (pos[0], pos[1], pos[2], math.degrees(float(yaw)), f.footprint[0], f.footprint[1], f.n_px,
                             (pos[2] - (args.table_z + model.dims[2])) * 1000))
            now = time.monotonic()
            if now - last_report > 2.0:
                last_report = now
                missing = watcher.grab.missing()
                if missing:
                    print("waiting for camera: %s" % ", ".join(missing))
                elif watcher.frames == 0:
                    print("frames arriving? none processed yet (TF base_link -> %s missing?)" % watcher.grab.parent)
                elif watcher.last_reject != last_reject or got is None:
                    last_reject = watcher.last_reject
                    print("status: %s | last reject: %s" % (watcher.status(), watcher.last_reject))
    except KeyboardInterrupt:
        pass
    finally:
        watcher.active = False
        print("\ndone: %s" % watcher.status())
        rclpy.shutdown()


if __name__ == "__main__":
    main()
