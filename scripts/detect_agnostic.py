#!/usr/bin/env python3
"""Size-agnostic box detection: any box-shaped thing standing on a measured table.

The alternative-1 detector from the 2026-09-16 design review. Nothing typed in:
not the table height, not the box height, not the box side. Per frame:

  1. deproject the aligned depth; RANSAC-fit the dominant plane = the table;
     express every point in a table-aligned frame (table top = z 0)
  2. keep points standing 4..15 cm above the table (--min-h / --max-h)
  3. clean: despeckle, local-smoothness gate, connected blobs
  4. per blob, in METRIC table-frame xy: min-area rectangle -> side lengths,
     yaw, centre; gates: side 6..13 cm, aspect 0.8..1.2, fill >= 0.35,
     not touching the image border, RAISED above its surroundings (a step,
     not a slope), enough cells
  5. if several survive, prefer the one nearest the image centre (the scan
     axis) and say how many there were
  6. aim = the rectangle's centre (OXO centres the button on the lid);
     a FIX commits when the last 3 still-camera sightings agree within
     15 mm and the newest is < 1 s old. Height and side are MEASURED and
     reported, not checked against a config.

Reuses only Chris's camera grabber (topics + intrinsics). No TF, no planner,
no motion, no colour dependence (the colour image is only drawn on).

    python3 scripts/detect_agnostic.py --window
    python3 scripts/detect_agnostic.py --window --min-side-mm 50 --max-side-mm 160
"""

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--window", action="store_true", help="diagnostic window (needs GUI OpenCV in the venv)")
    ap.add_argument("--overlay", action="store_true", help="publish /detect_agnostic/overlay for rqt_image_view")
    ap.add_argument("--min-h", type=float, default=0.04, help="lowest lid height above the table to consider (m)")
    ap.add_argument("--max-h", type=float, default=0.15, help="highest (m)")
    ap.add_argument("--min-side-mm", type=float, default=60.0)
    ap.add_argument("--max-side-mm", type=float, default=130.0)
    ap.add_argument("--aspect-tol", type=float, default=0.2, help="|w/h - 1| allowed (square lid)")
    ap.add_argument("--min-fill", type=float, default=0.35)
    ap.add_argument("--surface-std-mm", type=float, default=12.0, help="local height scatter allowed on a real surface")
    ap.add_argument("--raise-mm", type=float, default=25.0, help="blob must stand this far above the ring around it")
    ap.add_argument("--min-cells", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--agree-mm", type=float, default=15.0)
    ap.add_argument("--n-agree", type=int, default=3)
    args = ap.parse_args()
    if args.window:
        args.overlay = True

    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node

    if args.window and not hasattr(cv2, "imshow"):
        sys.exit("--window needs GUI OpenCV: pip uninstall -y opencv-contrib-python-headless && pip install opencv-contrib-python==4.10.0.84")

    # ------------------------------------------------------------------ geometry helpers
    def table_pose_from_depth(depth, k, stride, iters=120, thresh=0.006, min_range=0.15, max_range=1.2):
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
        if d < 0:
            n, d = -n, -d
        z_b = n
        x_b = np.array([1.0, 0.0, 0.0]) - n[0] * n
        x_b /= np.linalg.norm(x_b)
        y_b = np.cross(z_b, x_b)
        rot = np.stack([x_b, y_b, z_b])
        trans = np.array([0.0, 0.0, d])
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        return rot, trans, best_cnt / len(pts), tilt

    def still(prev, cur, dt=0.004, da=0.02):
        if prev is None:
            return False
        dp = float(np.linalg.norm(cur[1] - prev[1]))
        r = prev[0].T @ cur[0]
        ang = math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))
        return dp <= dt and ang <= da

    def project(p_base, cam, k):
        p = cam[0].T @ (np.asarray(p_base, dtype=float) - cam[1])
        if p[2] <= 0.01:
            return None
        return int(k[0, 0] * p[0] / p[2] + k[0, 2]), int(k[1, 1] * p[1] / p[2] + k[1, 2])

    # ------------------------------------------------------------------ the detector
    def detect(depth, k, cam):
        """Return (chosen candidate or None, all candidates, grid arrays for drawing, why)."""
        rot, trans = cam[0], cam[1]
        st = args.stride
        h, w = depth.shape
        vs, us = np.mgrid[0:h:st, 0:w:st]
        z = depth[vs, us]
        valid = (z > 0.1) & (z < 1.5) & np.isfinite(z)
        x = (us - k[0, 2]) / k[0, 0] * z
        y = (vs - k[1, 2]) / k[1, 1] * z
        p = np.stack([x, y, z], axis=-1) @ rot.T + trans          # table frame
        zb = p[..., 2]
        # local smoothness on the height grid, valid cells only
        vf = valid.astype(np.float32)
        zf = np.where(valid, zb, 0.0).astype(np.float32)
        cnt = cv2.boxFilter(vf, -1, (5, 5), normalize=False)
        s1 = cv2.boxFilter(zf, -1, (5, 5), normalize=False)
        s2 = cv2.boxFilter(zf * zf, -1, (5, 5), normalize=False)
        mean = s1 / np.maximum(cnt, 1)
        var = np.maximum(s2 / np.maximum(cnt, 1) - mean * mean, 0.0)
        smooth = (np.sqrt(var) < args.surface_std_mm * 1e-3) & (cnt >= 6)
        band = valid & (zb > args.min_h) & (zb < args.max_h)
        mask = (band & smooth).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        grids = dict(zb=zb, valid=valid, band=band, mask=mask.astype(bool))
        if mask.sum() == 0:
            return None, [], grids, "nothing standing %.0f..%.0f mm above the table" % (args.min_h * 1e3, args.max_h * 1e3)

        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        gh, gw = mask.shape
        cands, reasons = [], []
        for lab in range(1, n_lab):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < args.min_cells:
                continue
            bx, by, bw, bh = (int(stats[lab, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            blob = labels == lab
            if bx <= 1 or by <= 1 or bx + bw >= gw - 1 or by + bh >= gh - 1:
                reasons.append("a blob touches the image border")
                continue
            pxy = p[blob][:, :2].astype(np.float32)               # metric xy on the table
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(pxy)
            side_a, side_b = max(rw, rh), min(rw, rh)
            z_top = float(np.median(zb[blob]))
            if not (args.min_side_mm * 1e-3 <= side_a <= args.max_side_mm * 1e-3) or side_b < args.min_side_mm * 1e-3 * 0.7:
                reasons.append("footprint %.0fx%.0f mm outside %.0f..%.0f" % (side_a * 1e3, side_b * 1e3, args.min_side_mm, args.max_side_mm))
                continue
            aspect = side_b / side_a if side_a > 0 else 0.0
            if abs(aspect - 1.0) > args.aspect_tol:
                reasons.append("footprint %.0fx%.0f mm not square (aspect %.2f)" % (side_a * 1e3, side_b * 1e3, aspect))
                continue
            zmed = float(np.median(z[blob]))
            cell = (st * zmed / k[0, 0]) ** 2                        # table area per grid cell at that range
            fill = area * cell / max(side_a * side_b, 1e-6)
            if fill < args.min_fill:
                reasons.append("footprint %.0fx%.0f mm but fill %.2f < %.2f" % (side_a * 1e3, side_b * 1e3, fill, args.min_fill))
                continue
            ring = cv2.dilate(blob.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~blob & valid
            if ring.sum() >= 10:
                rise = z_top - float(np.median(zb[ring]))
                if rise < args.raise_mm * 1e-3:
                    reasons.append("footprint %.0fx%.0f mm but only %.0f mm above its surroundings" % (side_a * 1e3, side_b * 1e3, rise * 1e3))
                    continue
            else:
                rise = float("nan")
            cu, cv_ = project((cx, cy, z_top), cam, k) or (w // 2, h // 2)
            cands.append(dict(center=(float(cx), float(cy), z_top), side=(side_a, side_b), yaw=math.radians(ang) % (math.pi / 2),
                              height=z_top, fill=fill, cells=area, rise=rise, blob=blob,
                              off_axis=math.hypot(cu - w / 2, cv_ - h / 2)))
        if not cands:
            return None, [], grids, (reasons[0] if reasons else "no blob big enough (min %d cells)" % args.min_cells)
        cands.sort(key=lambda c: c["off_axis"])
        return cands[0], cands, grids, None

    # ------------------------------------------------------------------ ROS plumbing
    rclpy.init()
    node = Node("detect_agnostic")
    from rammp_curobo_ros.seek_core import D405Grabber

    grab = D405Grabber(node, need_depth=True)
    pub = None
    if args.overlay:
        from sensor_msgs.msg import Image

        pub = node.create_publisher(Image, "/detect_agnostic/overlay", 1)

    window = deque()      # (center xyz, yaw, side, t)
    state = dict(frames=0, hits=0, last_stamp=None, last_cam=None, plane=None, chosen=None, cands=[], grids=None,
                 why="no frame yet", fix=None, n_cands=0)

    def try_fix(now):
        while window and now - window[0][3] > 2.0:
            window.popleft()
        if len(window) < args.n_agree or now - window[-1][3] > 1.0:
            return None
        recent = list(window)[-args.n_agree:]
        pts = np.array([r[0][:2] for r in recent])
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.linalg.norm(pts[i] - pts[j]) > args.agree_mm * 1e-3:
                    return None
        med = np.median(np.array([r[0] for r in recent]), axis=0)
        sides = np.median(np.array([r[2] for r in recent]), axis=0)
        return dict(center=tuple(float(v) for v in med), yaw=recent[-1][1], side=tuple(float(v) for v in sides))

    def tick():
        g = grab
        if g.depth is None or g.k is None or g.color_stamp is None:
            return
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == state["last_stamp"]:
            return
        state["last_stamp"] = stamp
        state["frames"] += 1
        fit = table_pose_from_depth(g.depth, g.k, args.stride)
        if fit is None:
            state["why"] = "no table plane in the depth image"
            state["chosen"] = None
            return
        rot, trans, frac, tilt = fit
        cam = (rot, trans)
        state["plane"] = (frac, tilt, float(trans[2]))
        is_still = still(state["last_cam"], cam)
        state["last_cam"] = cam
        chosen, cands, grids, why = detect(g.depth, g.k, cam)
        state.update(chosen=chosen, cands=cands, grids=grids, cam=cam, n_cands=len(cands))
        if chosen is None:
            state["why"] = why
            return
        state["why"] = None if is_still else "camera moving (sighting not counted)"
        if is_still:
            state["hits"] += 1
            window.append((chosen["center"], chosen["yaw"], chosen["side"], time.monotonic()))

    node.create_timer(0.05, tick)

    # ------------------------------------------------------------------ drawing
    def render(got):
        g = grab
        out = g.color.copy()
        cam, k = state.get("cam"), g.k
        h, w = out.shape[:2]

        def square(center, yaw, side, color, thick):
            c, s_ = math.cos(yaw), math.sin(yaw)
            hx, hy = side[0] / 2, side[1] / 2
            pts = []
            for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
                uv = project((center[0] + c * dx - s_ * dy, center[1] + s_ * dx + c * dy, center[2]), cam, k)
                if uv is not None:
                    pts.append(uv)
            if len(pts) == 4:
                cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, color, thick)

        if cam is not None:
            for c in state["cands"]:
                square(c["center"], c["yaw"], c["side"], (0, 200, 255), 1)
            ch = state["chosen"]
            if ch is not None:
                square(ch["center"], ch["yaw"], ch["side"], (0, 220, 0), 2)
                uv = project(ch["center"], cam, k)
                if uv is not None:
                    cv2.drawMarker(out, uv, (0, 220, 0), cv2.MARKER_CROSS, 16, 2)
                    for i, t in enumerate(["h %.0f mm  side %.0fx%.0f" % (ch["height"] * 1e3, ch["side"][0] * 1e3, ch["side"][1] * 1e3),
                                           "fill %.2f  rise %.0f mm  %d cells" % (ch["fill"], ch["rise"] * 1e3, ch["cells"])]):
                        cv2.putText(out, t, (uv[0] + 12, uv[1] + 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                        cv2.putText(out, t, (uv[0] + 12, uv[1] + 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            if got is not None:
                square(got["center"], got["yaw"], got["side"], (0, 255, 0), 4)

        # height map with the standing-band mask
        grids = state.get("grids")
        if grids is not None and cam is not None:
            zb, valid = grids["zb"], grids["valid"]
            img = cv2.applyColorMap((np.clip(zb / 0.25, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            img[~valid] = (0, 0, 0)
            band = grids["band"] & valid
            img[band] = (0.5 * img[band] + 0.5 * np.array([200, 200, 200])).astype(np.uint8)
            img[grids["mask"]] = (255, 255, 255)
            if state["chosen"] is not None:
                img[state["chosen"]["blob"]] = (0, 220, 0)
            hmap = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            hmap = np.zeros_like(out)
        cv2.putText(hmap, "height above table; grey = %.0f..%.0f mm band, white = smooth, green = chosen" % (args.min_h * 1e3, args.max_h * 1e3),
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(hmap, "height above table; grey = %.0f..%.0f mm band, white = smooth, green = chosen" % (args.min_h * 1e3, args.max_h * 1e3),
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        panel = np.full((h, 380, 3), 30, dtype=np.uint8)
        lines = ["SIZE-AGNOSTIC DETECTOR", " %d/%d frames had a standing box" % (state["hits"], state["frames"])]
        if state["plane"]:
            fr, tilt, hh = state["plane"]
            lines += ["", "TABLE PLANE", "  inliers %.0f %%   tilt %.1f deg" % (fr * 100, tilt), "  camera %.3f m above table" % hh]
        lines += ["", "CANDIDATES: %d" % state["n_cands"]]
        ch = state["chosen"]
        if ch is None:
            lines += ["  none: %s" % (state["why"] or "")]
        else:
            lines += ["  MEASURED height  %.0f mm" % (ch["height"] * 1e3),
                      "  MEASURED side    %.0f x %.0f mm" % (ch["side"][0] * 1e3, ch["side"][1] * 1e3),
                      "  fill %.2f   cells %d" % (ch["fill"], ch["cells"]),
                      "  rise over ring   %.0f mm" % (ch["rise"] * 1e3),
                      "  yaw %.0f deg   %s" % (math.degrees(ch["yaw"]), "" if state["why"] is None else "(" + state["why"] + ")")]
        lines += ["", "FIX (aim = lid centre):"]
        if got is None:
            lines += ["  none yet (%d/%d agreeing)" % (len(window), args.n_agree)]
        else:
            lines += ["  [%.3f %.3f] on table, lid z %.3f" % (got["center"][0], got["center"][1], got["center"][2]),
                      "  box %.0f x %.0f x %.0f mm" % (got["side"][0] * 1e3, got["side"][1] * 1e3, got["center"][2] * 1e3)]
        for i, t in enumerate(lines):
            bold = t.isupper() and t.strip() != ""
            cv2.putText(panel, t, (10, 22 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255) if bold else (200, 200, 200), 1)
        composite = np.hstack([out, hmap, panel])
        if args.window:
            cv2.imshow("size-agnostic box detector  (q quits)", composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        if pub is not None and (pub.get_subscription_count() > 0 or not args.window):
            from sensor_msgs.msg import Image

            m = Image()
            m.header.stamp = node.get_clock().now().to_msg()
            m.height, m.width = composite.shape[:2]
            m.encoding = "bgr8"
            m.step = composite.shape[1] * 3
            m.data = np.ascontiguousarray(composite).tobytes()
            pub.publish(m)

    # ------------------------------------------------------------------ main loop
    print("gates: standing %.0f..%.0f mm | side %.0f..%.0f mm | aspect +/-%.2f | fill >= %.2f | surface std %.0f mm | rise >= %.0f mm"
          % (args.min_h * 1e3, args.max_h * 1e3, args.min_side_mm, args.max_side_mm, args.aspect_tol, args.min_fill, args.surface_std_mm, args.raise_mm))
    print("commit: %d still sightings within %.0f mm, newest < 1 s" % (args.n_agree, args.agree_mm))
    t0 = time.monotonic()
    last_fix_key, last_report, last_drawn = None, 0.0, None
    try:
        while time.monotonic() - t0 < args.seconds and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            got = try_fix(now)
            if got is not None:
                key = tuple(round(v, 3) for v in got["center"])
                if key != last_fix_key:
                    last_fix_key = key
                    print("FIX  aim [%.3f %.3f] lid z %.3f  yaw %.0f deg  box %.0f x %.0f x %.0f mm"
                          % (got["center"][0], got["center"][1], got["center"][2], math.degrees(got["yaw"]),
                             got["side"][0] * 1e3, got["side"][1] * 1e3, got["center"][2] * 1e3))
            if args.overlay and grab.color is not None and state.get("cam") is not None and state["last_stamp"] != last_drawn:
                last_drawn = state["last_stamp"]
                try:
                    render(got)
                except Exception as e:
                    print("overlay error: %s" % e)
            if now - last_report > 2.0:
                last_report = now
                missing = grab.missing()
                if missing:
                    print("waiting for camera: %s" % ", ".join(missing))
                else:
                    pl = state["plane"]
                    ch = state["chosen"]
                    print("status: %d/%d frames | plane %s | %s"
                          % (state["hits"], state["frames"],
                             "n/a" if pl is None else "%.0f%% tilt %.1f deg cam %.3f m" % (pl[0] * 100, pl[1], pl[2]),
                             ("box h %.0f side %.0fx%.0f mm (%d cand)" % (ch["height"] * 1e3, ch["side"][0] * 1e3, ch["side"][1] * 1e3, state["n_cands"]))
                             if ch is not None else "none: %s" % state["why"]))
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        print("\ndone: %d/%d frames had a standing box" % (state["hits"], state["frames"]))
        if args.window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
