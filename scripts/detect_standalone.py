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
    ap.add_argument("--overlay", action="store_true",
                    help="publish the colour image with the detection drawn on it to /detect_standalone/overlay "
                         "(view: ros2 run rqt_image_view rqt_image_view /detect_standalone/overlay)")
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

    overlay_pub = None
    if args.overlay:
        import cv2
        import numpy as np
        from sensor_msgs.msg import Image

        overlay_pub = node.create_publisher(Image, "/detect_standalone/overlay", 1)

        def project(p_base, rot_cam, trans_cam, k):
            p = np.asarray(rot_cam).T @ (np.asarray(p_base, dtype=float) - np.asarray(trans_cam))
            if p[2] <= 0.01:
                return None, p[2]
            return (int(k[0, 0] * p[0] / p[2] + k[0, 2]), int(k[1, 1] * p[1] / p[2] + k[1, 2])), p[2]

        def draw(img, fix_pos, fix_yaw, sight, committed, status, reject):
            out = img.copy()
            cam = watcher._last_cam
            k = watcher.grab.k
            for label, pos, yaw, color in (("sight", sight, None, (0, 200, 255)), ("FIX", fix_pos, fix_yaw, (0, 255, 0))):
                if pos is None or cam is None or k is None:
                    continue
                rot_cam, trans_cam = cam
                uv, z = project(pos, rot_cam, trans_cam, k)
                if uv is None:
                    continue
                r_px = max(3, int(k[0, 0] * (model.button_diameter_m / 2) / z))
                cv2.circle(out, uv, r_px, color, 2)
                cv2.drawMarker(out, uv, color, cv2.MARKER_CROSS, 12, 1)
                if yaw is not None:
                    c, s_ = math.cos(float(yaw)), math.sin(float(yaw))
                    hx, hy = model.dims[0] / 2, model.dims[1] / 2
                    pts = []
                    for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
                        corner = [pos[0] + c * dx - s_ * dy, pos[1] + s_ * dx + c * dy, pos[2]]
                        cuv, _ = project(corner, rot_cam, trans_cam, k)
                        if cuv is not None:
                            pts.append(cuv)
                    if len(pts) == 4:
                        cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, color, 2)
                cv2.putText(out, label, (uv[0] + 10, uv[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            lines = [status or "", ("reject: %s" % reject) if reject else "", "FIX committed" if committed else "no committed fix"]
            for i, t in enumerate(l for l in lines if l):
                cv2.putText(out, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(out, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return out

        def publish_overlay(img):
            m = Image()
            m.header.stamp = node.get_clock().now().to_msg()
            m.height, m.width = img.shape[:2]
            m.encoding = "bgr8"
            m.step = img.shape[1] * 3
            m.data = np.ascontiguousarray(img).tobytes()
            overlay_pub.publish(m)

    t0 = time.monotonic()
    last_fix, last_reject, last_report, last_overlay = None, None, 0.0, 0.0
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
            if overlay_pub is not None and now - last_overlay > 0.2 and watcher.grab.color is not None:
                last_overlay = now
                try:
                    sight = tuple(watcher.last_debug.center) if watcher.last_debug is not None else None
                    fix_pos = tuple(got[0]) if got is not None else None
                    fix_yaw = got[1] if got is not None else None
                    publish_overlay(draw(watcher.grab.color, fix_pos, fix_yaw, sight, got is not None,
                                         watcher.status(), watcher.last_reject))
                except Exception as e:  # the overlay is a diagnostic; detection must not die for it
                    print("overlay error: %s" % e)
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
