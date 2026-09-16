#!/usr/bin/env python3
"""Run Chris's depth box detector live, with nothing that can move the arm.

No planner, no executor, no motion: BoxTopWatcher (depth plateau -> TF lift ->
footprint gates -> button circle) on the live camera, printing every committed
fix and, while it refuses, WHY. The arm (or a fake TF) must hold a tool-down
pose ~35-45 cm above the box; this script never commands it.

    python3 scripts/detect_standalone.py --table-z 0.05 --overlay
    python3 scripts/detect_standalone.py --table-z 0.05 --no-circle     # accept plateau fixes without the button circle

--overlay publishes /detect_standalone/overlay (view with rqt_image_view):
  * every frame's best CANDIDATE plateau, found with the footprint gate
    relaxed so rejected candidates are drawn too, coloured by which gate
    it fails: green = passes all, amber = height residual, red = footprint
  * next to it: residual mm (limit 12), footprint mm (model 75x75), pixel
    count, circle found or not, and a 0..1 "geometry margin" = the worst
    gate's headroom (this is the honest stand-in for a confidence score;
    the detector is a chain of pass/fail gates, not a classifier)
  * the committed FIX (thick green) when the 3-frame agreement rule passes
  * the OWL node's bbox + score in blue, if `owl_detector` is running
"""

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

RELAXED_FOOT_TOL_M = 0.10  # draw candidates up to +/-10 cm off the model footprint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-z", type=float, default=None, help="table TOP height in base_link (m); needs a TF source")
    ap.add_argument("--table-from-depth", action="store_true",
                    help="no TF at all: fit the table plane in each depth frame (RANSAC), derive the camera's height and "
                         "tilt from it, and detect in a table-aligned frame (table top = z 0). Prototype of runtime table "
                         "measurement; x/y are relative to the point under the camera, not the arm base")
    ap.add_argument("--container", default=str(REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no-circle", action="store_true", help="let plateau-only sightings commit (diagnostic)")
    ap.add_argument("--overlay", action="store_true", help="publish /detect_standalone/overlay")
    ap.add_argument("--window", action="store_true",
                    help="open a diagnostic window (needs the non-headless opencv-contrib-python): colour+candidate | "
                         "height-above-table map with the search band | gate numbers. Implies --overlay rendering.")
    args = ap.parse_args()
    if args.table_z is None and not args.table_from_depth:
        sys.exit("give --table-z <m> (with a TF source) or --table-from-depth (no TF needed)")
    if args.table_from_depth:
        args.table_z = 0.0
    if args.window:
        args.overlay = True

    import numpy as np
    import rclpy
    from rclpy.node import Node

    from rammp_box_opening.models.container import ContainerModel, load_press_demo
    from rammp_box_opening.perception import depth_source as ds
    from rammp_box_opening.perception.depth_source import BoxTopWatcher, camera_pose_at

    model = ContainerModel.load(args.container)
    cfg = load_press_demo(args.container)
    expected_top = args.table_z + model.dims[2]
    print("box model: %.3f x %.3f x %.3f m, button %.0f mm; table_z %.3f -> expect lid top at z %.3f"
          % (*model.dims, model.button_diameter_m * 1000, args.table_z, expected_top))
    print("commit rule: %d agreeing frames within %.0f mm, fresh < %.1f s%s"
          % (cfg.min_hits, cfg.tol_m * 1000, cfg.fresh_s, "" if not args.no_circle else "  (circle NOT required)"))

    def table_pose_from_depth(depth, k, stride=4, iters=120, thresh=0.006, min_range=0.15, max_range=1.2):
        """RANSAC-fit the dominant plane (the table) in the depth image; return the camera pose
        (rot_cam, trans_cam) in a TABLE-ALIGNED base frame: +z = table normal, table top at z=0,
        origin at the foot of the camera's perpendicular. Also the inlier fraction and tilt (deg)."""
        h, w = depth.shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[vs, us]
        ok = (z > min_range) & (z < max_range) & np.isfinite(z)
        if ok.sum() < 300:
            return None
        x = (us[ok] - k[0, 2]) / k[0, 0] * z[ok]
        y = (vs[ok] - k[1, 2]) / k[1, 1] * z[ok]
        pts = np.stack([x, y, z[ok]], axis=1)
        rng = np.random.default_rng(0)
        best_n, best_d, best_cnt = None, 0.0, 0
        for _ in range(iters):
            tri = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -float(n @ tri[0])
            cnt = int((np.abs(pts @ n + d) < thresh).sum())
            if cnt > best_cnt:
                best_n, best_d, best_cnt = n, d, cnt
        if best_n is None or best_cnt < 0.2 * len(pts):
            return None
        inl = pts[np.abs(pts @ best_n + best_d) < thresh]
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c)
        n = vt[2]
        d = -float(n @ c)
        if d < 0:            # orient the normal toward the camera (camera is at the origin, above the table)
            n, d = -n, -d
        z_b = n
        x_b = np.array([1.0, 0.0, 0.0]) - (n[0]) * n
        x_b /= np.linalg.norm(x_b)
        y_b = np.cross(z_b, x_b)
        rot = np.stack([x_b, y_b, z_b])           # v_base = rot @ v_cam
        trans = np.array([0.0, 0.0, d])           # camera origin sits d above the table
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        return rot, trans, best_cnt / len(pts), tilt

    class PlaneWatcher:
        """TF-free stand-in for BoxTopWatcher: same detector, same commit rules, camera pose from the
        fitted table plane instead of TF. Exposes the attributes the loop and overlay read."""

        def __init__(self, node, cfg, model):
            from rammp_curobo_ros.seek_core import D405Grabber

            self.cfg, self.model, self.table_z = cfg, model, 0.0
            self.require_circle = bool(getattr(cfg, "require_button_circle", True))
            self.grab = D405Grabber(node, need_depth=True)
            self.window = ds.FixWindow(cfg.min_hits, cfg.tol_m, cfg.window_s, cfg.fresh_s)
            self.frames = self.hits = self.circle_hits = 0
            self.last_debug = self.last_reject = self._last_stamp = self._last_cam = None
            self.plane = None       # (inlier fraction, tilt deg, height)
            self.active = True
            node.create_timer(float(getattr(cfg, "detect_period_s", 0.15)), self._tick)

        def _tick(self):
            g = self.grab
            if g.depth is None or g.k is None or g.color_stamp is None:
                return
            stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
            if stamp == self._last_stamp:
                return
            self._last_stamp = stamp
            self.frames += 1
            fit = table_pose_from_depth(g.depth, g.k)
            if fit is None:
                self.last_reject = "no table plane in the depth image"
                return
            rot, trans, frac, tilt = fit
            self.plane = (frac, tilt, trans[2])
            self._last_cam = (rot, trans)
            fix, why = ds.top_face_from_depth(g.depth, g.k, rot, trans, 0.0, self.model)
            if fix is None:
                self.last_reject = why
                return
            bad = ds.top_residual_reject(fix.center[2], 0.0, self.model)
            if bad is not None:
                self.last_reject = bad
                return
            self.hits += 1
            circle = ds.button_circle_refine(g.color, g.depth, g.k, rot, trans, fix.center, self.model.button_diameter_m)
            if circle is not None:
                fix = ds.TopFaceFix(center=(circle[0], circle[1], fix.center[2]), yaw=fix.yaw,
                                    footprint=fix.footprint, n_px=fix.n_px)
                self.circle_hits += 1
            elif self.require_circle:
                self.last_reject = "lid found but no button circle"
                return
            self.last_debug = fix
            self.last_reject = None
            self.window.add(np.asarray(fix.center), fix.yaw, time.monotonic())

        def fix(self, now=None):
            got = self.window.fix(time.monotonic() if now is None else now)
            return None if got is None else (got[0], got[1])

        def status(self):
            missing = self.grab.missing()
            if missing:
                return "camera streams missing: %s" % ", ".join(missing)
            pl = "" if self.plane is None else " | table plane: %.0f%% inliers, tilt %.1f deg, camera %.3f m up" % (
                self.plane[0] * 100, self.plane[1], self.plane[2])
            return "%d/%d frames found a container top, %d button-circle%s" % (self.hits, self.frames, self.circle_hits, pl)

    rclpy.init()
    node = Node("detect_standalone")
    watcher = PlaneWatcher(node, cfg, model) if args.table_from_depth else BoxTopWatcher(node, cfg, model, args.table_z)
    if args.no_circle:
        watcher.require_circle = False
    watcher.active = True

    # ---- overlay ------------------------------------------------------------------
    overlay = None
    owl = {"msg": None, "t": 0.0}
    if args.overlay:
        import cv2
        from sensor_msgs.msg import Image

        if args.window and not hasattr(cv2, "imshow"):
            sys.exit("--window needs a GUI OpenCV: pip uninstall -y opencv-contrib-python-headless && "
                     "pip install opencv-contrib-python==4.10.0.84")
        from std_msgs.msg import Float32MultiArray

        pub = node.create_publisher(Image, "/detect_standalone/overlay", 1)

        def _owl_cb(m):
            owl["msg"], owl["t"] = list(m.data), time.monotonic()

        node.create_subscription(Float32MultiArray, "/rammp_box_opening/owl_bbox", _owl_cb, 1)

        def project(p_base, rot_cam, trans_cam, k):
            p = np.asarray(rot_cam).T @ (np.asarray(p_base, dtype=float) - np.asarray(trans_cam))
            if p[2] <= 0.01:
                return None, p[2]
            return (int(k[0, 0] * p[0] / p[2] + k[0, 2]), int(k[1, 1] * p[1] / p[2] + k[1, 2])), p[2]

        def square(out, center, yaw, rot_cam, trans_cam, k, color, thick):
            c, s_ = math.cos(float(yaw)), math.sin(float(yaw))
            hx, hy = model.dims[0] / 2, model.dims[1] / 2
            pts = []
            for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
                uv, _ = project([center[0] + c * dx - s_ * dy, center[1] + s_ * dx + c * dy, center[2]],
                                rot_cam, trans_cam, k)
                if uv is not None:
                    pts.append(uv)
            if len(pts) == 4:
                cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, color, thick)

        def label(out, uv, lines, color):
            for i, t in enumerate(lines):
                y = uv[1] + 18 * (i + 1)
                cv2.putText(out, t, (uv[0] + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(out, t, (uv[0] + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        def candidate(g, cam):
            """Relaxed pass: same detector, footprint gate widened, so a rejected
            plateau still comes back with its numbers."""
            saved = ds.FOOT_TOL_M
            ds.FOOT_TOL_M = RELAXED_FOOT_TOL_M
            try:
                fix, why = ds.top_face_from_depth(g.depth, g.k, cam[0], cam[1], args.table_z, model)
            finally:
                ds.FOOT_TOL_M = saved
            if fix is None:
                return None, why
            residual = fix.center[2] - expected_top
            foot_err = max(abs(fix.footprint[0] - model.dims[0]), abs(fix.footprint[1] - model.dims[1]))
            circle = ds.button_circle_refine(g.color, g.depth, g.k, cam[0], cam[1], fix.center, model.button_diameter_m)
            # headroom of each gate, 0..1, worst wins: the honest "confidence"
            margin = max(0.0, min(1.0 - abs(residual) / ds.TOP_RESIDUAL_MAX_M,
                                  1.0 - foot_err / ds.FOOT_TOL_M,
                                  min(1.0, fix.n_px / 400.0)))
            return dict(fix=fix, residual=residual, foot_err=foot_err, circle=circle, margin=margin), None

        def height_map(g, cam):
            """Every depth pixel's height above the table (base z), colour-mapped 0..0.25 m, with the
            detector's search band (expected lid top +/- BAND_TOL) drawn as a bright mask."""
            depth, k = g.depth, g.k
            h, w = depth.shape
            st = 2
            vs, us = np.mgrid[0:h:st, 0:w:st]
            z = depth[vs, us]
            ok = (z > 0.1) & (z < 1.5)
            x = (us - k[0, 2]) / k[0, 0] * z
            y = (vs - k[1, 2]) / k[1, 1] * z
            pts = np.stack([x, y, z], axis=-1) @ np.asarray(cam[0]).T + np.asarray(cam[1])
            zb = pts[..., 2]
            norm = np.clip(zb / 0.25, 0.0, 1.0)
            img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            img[~ok] = (0, 0, 0)
            band = ok & (np.abs(zb - expected_top) < ds.BAND_TOL_M)
            img[band] = (0.35 * img[band] + 0.65 * np.array([255, 255, 255])).astype(np.uint8)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, "height above table; white = in lid band (%.0f+/-%.0f mm)" % (expected_top * 1000, ds.BAND_TOL_M * 1000),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, "height above table; white = in lid band (%.0f+/-%.0f mm)" % (expected_top * 1000, ds.BAND_TOL_M * 1000),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return img

        def render(g, cam, got):
            out = g.color.copy()
            k = g.k
            hud = [watcher.status()]
            cand, why = candidate(g, cam)
            if cand is None:
                hud.append("candidate: none (%s)" % why)
            else:
                f = cand["fix"]
                ok_res = abs(cand["residual"]) <= ds.TOP_RESIDUAL_MAX_M
                ok_foot = cand["foot_err"] <= ds.FOOT_TOL_M
                color = (0, 200, 0) if (ok_res and ok_foot) else ((0, 200, 255) if ok_foot else (0, 0, 255))
                uv, z = project(f.center, cam[0], cam[1], k)
                if uv is not None:
                    square(out, f.center, f.yaw, cam[0], cam[1], k, color, 2)
                    cv2.drawMarker(out, uv, color, cv2.MARKER_CROSS, 14, 1)
                    if cand["circle"] is not None:
                        cuv, cz = project((cand["circle"][0], cand["circle"][1], f.center[2]), cam[0], cam[1], k)
                        if cuv is not None:
                            cv2.circle(out, cuv, max(3, int(k[0, 0] * model.button_diameter_m / 2 / cz)), color, 2)
                    label(out, uv, [
                        "margin %.2f" % cand["margin"],
                        "height %+.0f mm (lim 12)%s" % (cand["residual"] * 1000, "" if ok_res else "  X"),
                        "foot %.0fx%.0f mm (75x75)%s" % (f.footprint[0] * 1000, f.footprint[1] * 1000, "" if ok_foot else "  X"),
                        "%d px  circle %s" % (f.n_px, "yes" if cand["circle"] is not None else "no"),
                    ], color)
            if got is not None:
                pos, yaw = got
                square(out, pos, yaw, cam[0], cam[1], k, (0, 255, 0), 4)
                hud.append("FIX committed: [%.3f %.3f %.3f]" % tuple(pos))
            m = owl["msg"]
            if m is not None and time.monotonic() - owl["t"] < 3.0 and m[4] >= 0.0:
                x0, y0, x1, y1 = (int(v) for v in m[:4])
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 128, 0), 2)
                label(out, (x0, y0 - 22), ["OWL %.2f" % m[4]], (255, 128, 0))

            # ---- height-above-table map: the scene as the plateau detector sees it -------------
            hmap = height_map(g, cam)
            # ---- text panel ----------------------------------------------------------------
            h_img = out.shape[0]
            panel = np.full((h_img, 360, 3), 30, dtype=np.uint8)
            lines = ["DETECTOR", " " + watcher.status()[:52]]
            if getattr(watcher, "plane", None):
                fr, tilt, hh = watcher.plane
                lines += ["", "TABLE PLANE (from depth)", "  inliers %.0f %%   tilt %.1f deg" % (fr * 100, tilt),
                          "  camera %.3f m above table" % hh]
            lines += ["", "CANDIDATE (footprint gate relaxed)"]
            if cand is None:
                lines += ["  none: %s" % (why or "")]
            else:
                f = cand["fix"]
                lines += ["  margin      %.2f" % cand["margin"],
                          "  height      %+.0f mm   (limit +/-12)" % (cand["residual"] * 1000),
                          "  footprint   %.0f x %.0f mm (75 x 75, +/-35)" % (f.footprint[0] * 1000, f.footprint[1] * 1000),
                          "  pixels      %d" % f.n_px,
                          "  circle      %s" % ("found" if cand["circle"] is not None else "not found"),
                          "  yaw         %.0f deg" % math.degrees(float(f.yaw))]
            lines += ["", "STRICT DETECTOR SAYS", "  " + (watcher.last_reject or "ok")[:52]]
            lines += ["", "FIX: %s" % ("[%.3f %.3f %.3f]" % tuple(got[0]) if got is not None else "none committed")]
            for i, t in enumerate(lines):
                bold = t.isupper() and t.strip() != ""
                cv2.putText(panel, t, (10, 22 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255) if bold else (200, 200, 200), 1)
            for i, t in enumerate(hud):
                cv2.putText(out, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(out, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            composite = np.hstack([out, hmap, panel])
            if args.window:
                cv2.imshow("box detector  (q quits)", composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            if pub.get_subscription_count() > 0 or not args.window:
                msg = Image()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.height, msg.width = composite.shape[:2]
                msg.encoding = "bgr8"
                msg.step = composite.shape[1] * 3
                msg.data = np.ascontiguousarray(composite).tobytes()
                pub.publish(msg)

        overlay = render

    # ---- main loop ----------------------------------------------------------------
    t0 = time.monotonic()
    last_fix, last_reject, last_report, last_stamp = None, None, 0.0, None
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
                             (pos[2] - expected_top) * 1000))
            g = watcher.grab
            if overlay is not None and g.color is not None and g.depth is not None and g.k is not None:
                stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
                if stamp != last_stamp:
                    last_stamp = stamp
                    cam = watcher._last_cam if args.table_from_depth else camera_pose_at(g)
                    if cam is not None:
                        try:
                            overlay(g, cam, got)
                        except Exception as e:  # diagnostic only; detection must not die for it
                            print("overlay error: %s" % e)
            now = time.monotonic()
            if now - last_report > 2.0:
                last_report = now
                missing = g.missing()
                if missing:
                    print("waiting for camera: %s" % ", ".join(missing))
                elif watcher.frames == 0:
                    print("frames arriving? none processed yet (TF base_link -> %s missing?)" % g.parent)
                elif watcher.last_reject != last_reject or got is None:
                    last_reject = watcher.last_reject
                    print("status: %s | last reject: %s" % (watcher.status(), watcher.last_reject))
    except (KeyboardInterrupt, RuntimeError):  # RuntimeError: rclpy take-during-shutdown artifact on Ctrl+C
        pass
    finally:
        watcher.active = False
        print("\ndone: %s" % watcher.status())
        if args.window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
