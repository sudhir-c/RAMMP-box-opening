"""Tag-driven container pose: continuous wrist-camera detection.

Consumes the RAMMP-CuRobo pipeline — `tag_pose_from_frame` (pure ArUco
IPPE_SQUARE solve), `D405Grabber` (color + aligned depth + intrinsics +
TF-composed camera pose), `stable_fix` (n-agreeing-frames gate) — and
adds what this task needs on top: depth-refined range (what makes a
1-inch hover trustworthy: the RGB solve's z is the noisy axis, the
aligned depth at the tag's pixel is mm-scale), the tag→container-origin
mapping, and a rolling fix window with freshness so the flow only ever
acts on CURRENT sightings.

TagWatcher runs on a node timer, so detection proceeds during every
spin the CLI does — planner waits, execution monitoring, the detect
wait itself. The camera is never "off" (owner decision 2026-08-24); a
tick with no new frame returns immediately.
"""

import math
import time

import numpy as np

from rammp_box_opening.models.container import ContainerPose


def container_pose_from_tag(tag_pos, tag_rot, model, tag_offset):
    """Tag pose (base_link) -> ContainerPose (bottom-center origin + yaw).

    tag_offset is tag center -> button top center in the CONTAINER frame
    ((0,0,0) = tag stuck on the button); yaw is read from the tag's +x
    axis projected to the world xy plane."""
    r = np.asarray(tag_rot, dtype=float)
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)

    def spin(v):
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

    p = (
        np.asarray(tag_pos, dtype=float)
        + spin(np.asarray(tag_offset, dtype=float))
        - spin(np.asarray(model.button_offset, dtype=float))
    )
    return ContainerPose(xyz=tuple(float(v) for v in p), yaw=yaw)


def refine_point(tvec, k, depth, half_px=3, min_m=0.07, max_m=0.9):
    """Tag center re-derived from the aligned depth at its pixel, or None.

    Projects the RGB solve's tvec through K to find the tag's pixel,
    medians a small depth window there, and rebuilds the camera-frame
    point on that pixel's ray at the measured depth. None when depth is
    missing, out of the D405's trusted range, or off-image — callers
    fall back to the RGB solve and say so."""
    if depth is None:
        return None
    t = np.asarray(tvec, dtype=float)
    kk = np.asarray(k, dtype=float)
    if t[2] <= 0:
        return None
    u = kk[0, 0] * t[0] / t[2] + kk[0, 2]
    v = kk[1, 1] * t[1] / t[2] + kk[1, 2]
    ui, vi = int(round(u)), int(round(v))
    h, w = depth.shape
    if not (half_px <= ui < w - half_px and half_px <= vi < h - half_px):
        return None
    win = depth[vi - half_px : vi + half_px + 1, ui - half_px : ui + half_px + 1]
    valid = win[(win > min_m) & (win < max_m) & np.isfinite(win)]
    if valid.size < 3:
        return None
    d = float(np.median(valid))
    return np.array([(u - kk[0, 2]) / kk[0, 0] * d, (v - kk[1, 2]) / kk[1, 1] * d, d])


def servo_step(p_cam, rot_cam, k, tol_px, min_step_m, max_step_m):
    """One center-the-tag step: (base displacement | None-if-centered, px).

    px is the tag center's pixel distance from the image center. The
    displacement translates the CAMERA (and rigidly the tool) so the
    optical axis lands on the tag: rot_cam @ (x, y, 0) — camera-frame
    lateral offset lifted to base. Centering is robust to camera-mount
    calibration error: the direction may be a few degrees off, the
    iteration converges anyway, and 'tag on the optical axis' is true
    regardless of where the mount THINKS the camera is (owner design
    2026-08-25)."""
    p = np.asarray(p_cam, dtype=float)
    kk = np.asarray(k, dtype=float)
    px = math.hypot(kk[0, 0] * p[0] / p[2], kk[1, 1] * p[1] / p[2])
    disp = np.asarray(rot_cam, dtype=float) @ np.array([p[0], p[1], 0.0])
    n = float(np.linalg.norm(disp))
    if px <= tol_px or n < min_step_m:
        return None, px
    if n > max_step_m:
        disp = disp * (max_step_m / n)
    return [float(v) for v in disp], px


class FixWindow:
    """Rolling sightings -> a fresh, stable (position, rotation) fix.

    Position stability goes through the consumed `stable_fix` (median of
    the last min_hits when they agree pairwise within tol_m); rotation is
    the latest sighting's (yaw is joint-7-absorbed, spec'd second-order).
    A fix older than fresh_s never commits — the arm only acts on what
    the camera is seeing NOW."""

    def __init__(self, min_hits, tol_m, window_s, fresh_s):
        self.min_hits = int(min_hits)
        self.tol_m = float(tol_m)
        self.window_s = float(window_s)
        self.fresh_s = float(fresh_s)
        self.samples = []  # [(pos ndarray, t)]
        self.rot = None
        self.last_seen = None

    def add(self, pos, rot, t):
        t = float(t)
        self.samples = [(p, ts) for p, ts in self.samples if t - ts < self.window_s]
        self.samples.append((np.asarray(pos, dtype=float), t))
        self.rot = np.asarray(rot, dtype=float)
        self.last_seen = t

    def fix(self, now):
        from rammp_curobo_ros.seek_core import stable_fix

        if self.last_seen is None or now - self.last_seen > self.fresh_s:
            return None
        self.samples = [(p, ts) for p, ts in self.samples if now - ts < self.window_s]
        pos = stable_fix(self.samples, tol=self.tol_m, n=self.min_hits)
        if pos is None:
            return None
        return pos, self.rot


class TagWatcher:
    """Continuous detection on a node timer (fires during any spin).

    Reads the D405Grabber's frame buffers non-blockingly (never calls
    its spinning shot()); each NEW color frame is detected, lifted to
    base_link with TF at that frame's stamp (grabber-composed mount),
    depth-refined, and fed to the FixWindow."""

    def __init__(self, node, cfg, period_s=0.15):
        import cv2

        from rammp_curobo_ros.seek_core import D405Grabber

        self.node = node
        self.cfg = cfg
        self.grab = D405Grabber(node, need_depth=True)
        aruco = cv2.aruco
        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
            aruco.DetectorParameters(),
        )
        s = cfg.tag_size_m / 2.0
        # tag-frame corner order for IPPE_SQUARE (tags.py convention)
        self.objp = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64
        )
        self.window = FixWindow(cfg.min_hits, cfg.tol_m, cfg.window_s, cfg.fresh_s)
        self.frames = 0
        self.hits = 0
        self.refined_hits = 0
        self.depth_refined = None  # last sighting: True/False/None
        self.last_debug = None  # (p_cam, rot_cam, trans_cam) of last sighting
        self._last_stamp = None
        node.create_timer(period_s, self._tick)

    def reset(self):
        """Purge the window. Called when the arm settles at the scan pose:
        sightings gathered DURING motion carry TF/depth timing skew and a
        parked-only fix costs under half a second of fresh frames."""
        self.window.samples = []
        self.window.last_seen = None

    def _tick(self):
        from rammp_curobo_ros.tags import tag_pose_from_frame

        g = self.grab
        if g.color is None or g.k is None or g.color_stamp is None:
            return
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp
        self.frames += 1
        hit = tag_pose_from_frame(
            g.color, self.detector, self.objp, g.k, g.dist, self.cfg.tag_id
        )
        if hit is None:
            return
        _tid, r_tag_cam, tvec = hit
        self.hits += 1
        cam = self._camera_pose(g)
        if cam is None:
            return
        rot_cam, trans_cam = cam
        refined = refine_point(tvec, g.k, g.depth)
        self.depth_refined = refined is not None
        if refined is None:
            # the 1-inch hover premise REQUIRES mm-scale z: an RGB-only
            # sighting (2-3 cm z noise) never enters the window, so a
            # depth-starved run fails honestly as NO TAG instead of
            # pressing on a guess (2026-08-24 review)
            return
        self.refined_hits += 1
        pos = rot_cam @ refined + trans_cam
        rot = rot_cam @ r_tag_cam
        # raw ingredients of the last sighting, for mount diagnostics
        self.last_debug = (refined, rot_cam, trans_cam)
        self.window.add(pos, rot, time.monotonic())

    def _camera_pose(self, g):
        """base_link <- camera AT THE FRAME'S STAMP (mount composition as
        in D405Grabber.shot(), which cannot be used here: it spins).

        No latest-TF fallback: upstream documents that fallback as safe
        only while parked, and this watcher runs during motion — at
        continuous frame rates a dropped frame costs nothing, a
        wrong-pose frame poisons the fix (2026-08-24 review)."""
        import rclpy.time as rt

        from rammp_curobo.perception import quat_to_mat

        try:
            tr = g.tf_buffer.lookup_transform(
                "base_link", g.parent, rt.Time.from_msg(g.color_stamp)
            )
        except Exception:
            return None
        q, t = tr.transform.rotation, tr.transform.translation
        r_p = quat_to_mat(q.x, q.y, q.z, q.w)
        qx, qy, qz, qw = g.mount_quat
        rot = r_p @ quat_to_mat(qx, qy, qz, qw)
        trans = r_p @ np.asarray(g.mount_xyz, dtype=float) + np.array([t.x, t.y, t.z])
        return rot, trans

    def fix(self, now=None):
        return self.window.fix(time.monotonic() if now is None else now)

    def status(self):
        missing = self.grab.missing()
        if missing:
            return "camera streams missing: %s" % ", ".join(missing)
        return "%d/%d frames saw tag id %d, %d depth-refined%s" % (
            self.hits,
            self.frames,
            self.cfg.tag_id,
            self.refined_hits,
            " (RGB-only sightings never commit)"
            if self.hits and not self.refined_hits
            else "",
        )
