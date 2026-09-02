"""Container pose from depth: the box found by GEOMETRY, not by a print.

The fiducial path this replaced died twice at the bench in one week —
the gripper knuckles strike a printed tag on every press, and a 30 mm
print at scan height has almost no detection margin. This source needs
no print at all: from the scan pose the lid top is the only plateau at
container height above the MEASURED table. Aligned-depth pixels are
deprojected, lifted to base_link with frame-stamp TF (camera_pose_at),
banded by height, and the most box-like in-band blob whose footprint
matches the model becomes the sighting:

    plateau median z  = the lid top, MEASURED
    plateau centroid  = the button center (the button is centered),
                        refined to the button's CIRCLE when one is seen
    min-area-rect     = yaw, mod 90 deg — the box is square, and every
                        consumer is yaw-mod-90 invariant: button_offset
                        is purely vertical, press attitude is steered by
                        BEARING, and both collision cuboids are square

BoxTopWatcher ticks on a node timer (detection proceeds during every
spin the CLI does), commits through FixWindow's n-agreeing-fresh-frames
rule, and exposes fix()/status()/reset() to the mission. What this
source cannot do: tell one box from another (the VLM/OWL roi does that),
or see a box whose lid is off — fine here, the mission starts lid-on.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.models.container import ContainerPose

# Height band around the EXPECTED container top (table + dims.z, which
# is itself derived from live tag fixes 2026-08-25). Generous banding let
# table stereo-noise sheets at +3..+7 cm into play (capture
# 20260901-130610); the physical box top only ever reads within ~2 cm of
# expectation, so +/-0.035 keeps calibration slack without the noise.
BAND_TOL_M = 0.035
# COMMIT gate on the measured-vs-nominal top height. The band above only
# SEARCHES; with origin z pinned to the calibrated table, a fix whose
# measured top sits far from table+dims is either not our box resting on
# our table or a stale calibration — pressing on nominal geometry would
# then stroke air (below) or leave lid unmodelled (above). Tighter than
# travel_m (15 mm) so a committed fix always reaches real material;
# field residuals to date: -3 mm, +8 mm.
TOP_RESIDUAL_MAX_M = 0.012
# a real surface is locally SMOOTH; passive-stereo speckle on the blank
# table is locally wild. Local z-std above this is not a surface.
SURFACE_STD_M = 0.006
# a real top face is solidish: plateau cells / min-area-rect cells. 0.35,
# not higher: the core erosion opens holes in a real (white, low-texture)
# lid, and the smoothness gate already killed the noise archipelagos
MIN_FILL = 0.35
# second-stage tightening around the found plateau's own median
PLATEAU_TOL_M = 0.02
# blob footprint sanity vs the model's xy dims (per side)
FOOT_TOL_M = 0.035
# blobs touching the image border are cut off — their centroid is biased
BORDER_PX = 6
# depth trust range (D405 close-range envelope, same spirit as refine_point)
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 0.90
# pixel subsampling stride: 848x480/4 -> ~25k rays, milliseconds on CPU
STRIDE = 4
# a sample may only VOTE when the camera was still between consecutive
# processed frames: in-flight samples carry a systematic TF-vs-exposure
# lag bias that made three of them agree ~10-15 mm off the truth and the
# press land on the button's EDGE (field 2026-09-01). Parked frames at
# 20 Hz commit within ~150 ms of arrival — speed comes from committing
# early on STILL frames, not from letting moving ones vote.
STILL_TRANS_M = 0.004
STILL_ROT_RAD = 0.02


def camera_pose_at(g):
    """base_link <- camera AT THE FRAME'S STAMP (mount composition as in
    D405Grabber.shot(), which cannot be used here: it spins).

    No latest-TF fallback: upstream documents that fallback as safe only
    while parked, and the watcher runs during motion — at continuous frame
    rates a dropped frame costs nothing, a wrong-pose frame poisons the fix
    (2026-08-24 review)."""
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


class FixWindow:
    """Rolling sightings -> a fresh, stable (position, yaw) fix.

    Position stability goes through the consumed `stable_fix` (median of
    the last min_hits when they agree pairwise within tol_m); the yaw is
    the latest sighting's (joint-7-absorbed, spec'd second-order). A fix
    older than fresh_s never commits — the arm only acts on what the
    camera is seeing NOW."""

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


def camera_is_still(prev, cur, trans_tol=STILL_TRANS_M, rot_tol=STILL_ROT_RAD):
    """True when two consecutive camera poses are effectively identical.
    prev/cur: (rot 3x3, trans 3). Pure, testable."""
    if prev is None:
        return False
    dp = float(np.linalg.norm(np.asarray(cur[1]) - np.asarray(prev[1])))
    r = np.asarray(prev[0]).T @ np.asarray(cur[0])
    cos_a = (float(np.trace(r)) - 1.0) / 2.0
    ang = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return dp <= trans_tol and ang <= rot_tol


@dataclass(frozen=True)
class TopFaceFix:
    center: tuple  # button-top center, base frame (x, y, z=plateau median)
    yaw: float  # footprint yaw, mod pi/2
    footprint: tuple  # min-area-rect (w, h) in metres
    n_px: int  # plateau pixels (at STRIDE) backing the fix


def top_face_from_depth(depth, k, rot_cam, trans_cam, table_z, model, roi=None):
    """One depth frame -> TopFaceFix, or None with honesty about why not.

    Returns (fix, why). The candidate is the most BOX-LIKE in-band blob,
    not the largest: the bench table is featureless and the D405 is
    passive stereo, so a blank table speckles +/-5 cm of depth noise into
    the height band (capture 20260901-130610 — 24%% of all pixels). The
    speckle is morphologically opened away, every surviving blob is
    scored against the model footprint, and ambiguity (two box-sized
    tops) is refused rather than guessed — unless `roi` (a pixel-space
    bbox from the VLM source) says which one is meant.
    """
    if depth is None or k is None:
        return None, "no depth frame"
    kk = np.asarray(k, dtype=float)
    fx, fy, cx, cy = kk[0, 0], kk[1, 1], kk[0, 2], kk[1, 2]
    d = np.asarray(depth, dtype=np.float32)[::STRIDE, ::STRIDE]
    gh, gw = d.shape
    us = np.arange(gw, dtype=np.float32) * STRIDE
    vs = np.arange(gh, dtype=np.float32) * STRIDE
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(d) & (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
    if not valid.any():
        return None, "no valid depth"

    x = (uu - cx) / fx * d
    y = (vv - cy) / fy * d
    pts_cam = np.stack([x, y, d], axis=-1).reshape(-1, 3)
    rot = np.asarray(rot_cam, dtype=float)
    pts = pts_cam @ rot.T + np.asarray(trans_cam, dtype=float)
    z_base = pts[:, 2].reshape(gh, gw)

    expected_top = table_z + float(model.dims[2])
    band = (
        valid
        & (z_base > expected_top - BAND_TOL_M)
        & (z_base < expected_top + BAND_TOL_M)
    )
    # surface test: local smoothness of the height field. cv2.blur gives
    # E[z] and E[z^2] in one pass each; speckle fails the std gate even
    # where it lands inside the band.
    import cv2

    zf = np.where(valid, z_base, 0.0).astype(np.float32)
    wt = valid.astype(np.float32)
    ez = cv2.blur(zf, (3, 3)) / np.maximum(cv2.blur(wt, (3, 3)), 1e-6)
    ez2 = cv2.blur(zf * zf, (3, 3)) / np.maximum(cv2.blur(wt, (3, 3)), 1e-6)
    local_std = np.sqrt(np.maximum(ez2 - ez * ez, 0.0))
    band &= local_std < SURFACE_STD_M
    if roi is not None:
        u0, v0, u1, v1 = (int(v) for v in roi)
        gate = np.zeros_like(band)
        gate[
            max(0, v0 // STRIDE) : max(0, v1 // STRIDE) + 1,
            max(0, u0 // STRIDE) : max(0, u1 // STRIDE) + 1,
        ] = True
        band &= gate
    if not band.any():
        return None, "no points at container height"

    # open away what little speckle survives the smoothness gate
    opened = cv2.morphologyEx(
        band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_blobs < 2:
        return None, "no plateau after despeckle"

    edge = np.zeros((gh, gw), dtype=bool)
    b = max(1, BORDER_PX // STRIDE)
    edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True
    ex, ey = float(model.dims[0]), float(model.dims[1])
    lo, hi = min(ex, ey) - FOOT_TOL_M, max(ex, ey) + FOOT_TOL_M

    candidates, reasons = [], []
    order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    for label in order[:8]:
        blob = labels == label
        if int(blob.sum()) < 12:
            continue
        if (blob & edge).any():
            reasons.append("a top touches the image border")
            continue
        zb = z_base[blob]
        top_z0 = float(np.median(zb))
        plateau = blob & (np.abs(z_base - top_z0) < PLATEAU_TOL_M)
        # measure the CORE: depth discontinuities smear a ~1-cell ring of
        # flying pixels around the true face (capture 20260901-130610 read
        # a 75 mm box as 90-120 mm); one erosion shaves the ring and the
        # centroid/footprint come from real surface only
        plateau = cv2.erode(plateau.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
            bool
        )
        if plateau.sum() < 12:
            reasons.append("plateau too small (%d px)" % int(plateau.sum()))
            continue
        sel = pts.reshape(gh, gw, 3)[plateau]
        xy = sel[:, :2].astype(np.float32)
        (rcx, rcy), (w, h), angle = cv2.minAreaRect(xy)
        w, h = float(w), float(h)
        if not (lo <= min(w, h) and max(w, h) <= hi):
            reasons.append("footprint %.2fx%.2f m vs model %.2fx%.2f" % (w, h, ex, ey))
            continue
        # a real top face is SOLID; a noise archipelago that happens to
        # span a box-sized rect is not (fill = cells / rect area in cells)
        ys, xs = np.nonzero(plateau)
        rect_px = cv2.minAreaRect(np.stack([xs, ys], axis=-1).astype(np.float32))[1]
        rect_cells = max(1.0, float(rect_px[0]) * float(rect_px[1]))
        fill = float(plateau.sum()) / rect_cells
        if fill < MIN_FILL:
            reasons.append("top not solid (fill %.2f)" % fill)
            continue
        candidates.append(
            TopFaceFix(
                center=(float(rcx), float(rcy), float(np.median(sel[:, 2]))),
                yaw=math.radians(angle) % (math.pi / 2),
                footprint=(w, h),
                n_px=int(plateau.sum()),
            )
        )

    if not candidates:
        return None, (reasons[0] if reasons else "no plateau")
    if len(candidates) > 1:
        # two box-sized tops and no roi to disambiguate: guessing which
        # to press is exactly the mistake this gate exists to refuse
        return None, "%d container-sized tops in view — ambiguous" % len(candidates)
    return candidates[0], "ok"


def button_circle_refine(color_rgb, depth, k, rot_cam, trans_cam, center, button_d_m):
    """The round button's centre in base frame, or None.

    The plateau centroid finds the LID; this finds the BUTTON — the
    thing the press must actually hit (2026-09-01: centroid presses
    landed slightly off-centre). Hough circles on a crop around the
    plateau centre, radius-windowed from the button's physical diameter
    and the measured range, winner = the circle nearest the centroid.
    None on any doubt: the centroid fallback is always sane.
    """
    import cv2

    if color_rgb is None:
        return None
    kk = np.asarray(k, dtype=float)
    rot = np.asarray(rot_cam, dtype=float)
    tr = np.asarray(trans_cam, dtype=float)
    p_cam = rot.T @ (np.asarray(center, dtype=float) - tr)
    if p_cam[2] <= 0.05:
        return None
    u0 = kk[0, 0] * p_cam[0] / p_cam[2] + kk[0, 2]
    v0 = kk[1, 1] * p_cam[1] / p_cam[2] + kk[1, 2]
    r_px = kk[0, 0] * (float(button_d_m) / 2.0) / float(p_cam[2])
    if not 6.0 <= r_px <= 200.0:
        return None
    half = int(4.0 * r_px)
    h, w = color_rgb.shape[:2]
    x0, y0 = int(u0) - half, int(v0) - half
    x1, y1 = int(u0) + half, int(v0) + half
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None  # crop truncated: centroid fallback beats a biased circle
    # the grabber stores BGR (verified 2026-09-02); grey from the right
    # channel order — the seam is achromatic so the Hough is insensitive,
    # but the buffer is what it is
    gray = cv2.cvtColor(
        np.ascontiguousarray(color_rgb[y0:y1, x0:x1]), cv2.COLOR_BGR2GRAY
    )
    # blur 3, NOT 5: the heavier blur smeared the button seam and
    # Hough hit 1/37 capture frames; at 3 it hits 36/37 with 2.6 px
    # centre std (parameter sweep on capture 20260901-130610)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2 * r_px,
        param1=90,
        param2=14,
        minRadius=int(0.7 * r_px),
        maxRadius=int(1.4 * r_px),
    )
    if circles is None:
        return None
    cx = cy = None
    best = None
    for c in circles[0]:
        d = math.hypot(c[0] - half, c[1] - half)
        if best is None or d < best[0]:
            best = (d, c)
    d_px, c = best
    # the circle must be NEAR the plateau centroid — a far circle is some
    # other round thing (>25 mm at range is not this button)
    if d_px * p_cam[2] / kk[0, 0] > 0.025:
        return None
    u, v = float(c[0]) + x0, float(c[1]) + y0
    # range at the button's own pixel (handles the popped 15 mm knob);
    # falls back to the plateau's height when depth is holey there
    zwin = depth[int(v) - 2 : int(v) + 3, int(u) - 2 : int(u) + 3]
    valid = zwin[(zwin > DEPTH_MIN_M) & (zwin < DEPTH_MAX_M) & np.isfinite(zwin)]
    z = float(np.median(valid)) if valid.size >= 3 else float(p_cam[2])
    pc = np.array([(u - kk[0, 2]) / kk[0, 0] * z, (v - kk[1, 2]) / kk[1, 1] * z, z])
    out = rot @ pc + tr
    return (float(out[0]), float(out[1]))


def top_residual_reject(z_top, table_z, model):
    """None when the measured top is commit-close to nominal, else the
    honest refusal reason. Pure, unit-testable."""
    residual = float(z_top) - (float(table_z) + model.dims[2])
    if abs(residual) <= TOP_RESIDUAL_MAX_M:
        return None
    return (
        "top %.0f mm from nominal (limit %.0f) — not a table-resting box "
        "here, or table_z needs recalibrating"
        % (residual * 1000, TOP_RESIDUAL_MAX_M * 1000)
    )


def container_pose_from_top(center, yaw, model, table_z=None):
    """Top-face fix -> ContainerPose (bottom-center origin + yaw).

    Origin z comes from the CALIBRATED table when one is known: the box
    rests on the table, its moulded height is constant, and the bench
    yaml's table_z is surveyed — while the 6-frame passive-stereo
    plateau median wandered -3 mm one run (grip grazed the lid) and
    +8 mm the next (press geometry shifted a stroke-end delta under the
    drift gate) (field 2026-09-01/02). The measured top still gates
    detection (plateau band) and its residual vs nominal is reported by
    the watcher so calibration drift stays visible."""
    z = float(center[2]) - model.dims[2] if table_z is None else float(table_z)
    return ContainerPose(
        xyz=(float(center[0]), float(center[1]), z),
        yaw=float(yaw),
    )


class BoxTopWatcher:
    """Continuous detection on a node timer, FixWindow commit rules, and
    the fix()/status()/reset() surface the mission's detect flow uses.

    In-flight samples are KEPT across a detect wait (wait_for_fix never
    purges): geometry is lifted with frame-stamp TF, the still-camera
    filter drops moving frames, the 1 s freshness window means only the
    scan's parked tail can support a commit, and the 3-agreeing gate
    stands — so the fix is often ready the moment the arm parks (owner:
    detect during the flip, 2026-09-01)."""

    def __init__(self, node, cfg, model, table_z, period_s=None):
        from rammp_curobo_ros.seek_core import D405Grabber

        if period_s is None:
            period_s = float(getattr(cfg, "detect_period_s", 0.15))
        self.cfg = cfg
        self.model = model
        self.table_z = float(table_z)
        self.grab = D405Grabber(node, need_depth=True)
        self.window = FixWindow(cfg.min_hits, cfg.tol_m, cfg.window_s, cfg.fresh_s)
        self.frames = 0
        self.hits = 0
        self.refined_hits = 0  # == hits: every depth sighting IS refined
        self.circle_hits = 0  # sightings where the BUTTON circle aimed
        self.last_debug = None  # the last TopFaceFix (bench diagnostics)
        # pixel-space gate from the VLM source; None = whole frame. Set
        # while PARKED at the scan pose and cleared before any re-fix
        # from a different pose — a bbox is only valid where it was taken.
        self.roi = None
        self.last_reject = None  # why the last non-hit frame was refused
        self._last_stamp = None
        self._last_cam = None  # previous frame's camera pose (still filter)
        # detection runs only inside a detect window (wait_for_fix and the
        # detect-only report set it); the idle 20 Hz tick otherwise costs a
        # TF lookup per frame for nothing
        self.active = False
        # an offered roi is applied only once the camera has been still
        # since BEFORE the bbox's frame: a box seen while decelerating must
        # not gate the parked frames (review 2026-09-02)
        self._still_since = None  # frame stamp when the camera became still
        self._pending_roi = None  # (roi, frame_t) waiting for a still epoch
        node.create_timer(period_s, self._tick)

    def offer_roi(self, roi, frame_t):
        self._pending_roi = (roi, float(frame_t))

    def _apply_pending_roi(self, still, frame_t):
        if not still:
            self._still_since = None
            return
        if self._still_since is None:
            self._still_since = float(frame_t)
        if self._pending_roi is not None:
            roi, t = self._pending_roi
            self._pending_roi = None
            if t >= self._still_since:
                self.roi = roi

    def _tick(self):
        if not self.active:
            return
        g = self.grab
        if g.depth is None or g.k is None or g.color_stamp is None:
            return
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp
        self.frames += 1
        cam = camera_pose_at(g)
        if cam is None:
            return
        rot_cam, trans_cam = cam
        still = camera_is_still(self._last_cam, cam)
        self._last_cam = cam
        self._apply_pending_roi(still, g.color_stamp.sec + g.color_stamp.nanosec * 1e-9)
        if not still:
            self.last_reject = "camera moving"
            return
        fix, why = top_face_from_depth(
            g.depth, g.k, rot_cam, trans_cam, self.table_z, self.model, roi=self.roi
        )
        if fix is None:
            self.last_reject = why
            return
        bad = top_residual_reject(fix.center[2], self.table_z, self.model)
        if bad is not None:
            self.last_reject = bad
            return
        self.hits += 1
        self.refined_hits += 1
        circle = button_circle_refine(
            g.color,
            g.depth,
            g.k,
            rot_cam,
            trans_cam,
            fix.center,
            self.model.button_diameter_m,
        )
        if circle is not None:
            fix = TopFaceFix(
                center=(circle[0], circle[1], fix.center[2]),
                yaw=fix.yaw,
                footprint=fix.footprint,
                n_px=fix.n_px,
            )
            self.circle_hits += 1
        self.last_debug = fix
        self.window.add(np.asarray(fix.center), fix.yaw, time.monotonic())

    def reset(self):
        """Purge the window AND the counters: status() must describe the
        window it is asked about, not the mission's whole history (the
        pre-grip re-look printed scan-pose hit counts and hid its own
        reject reason for two days, 2026-09-02)."""
        self.window.samples = []
        self.window.last_seen = None
        self.frames = 0
        self.hits = 0
        self.refined_hits = 0
        self.circle_hits = 0
        self.last_reject = None

    def fix(self, now=None):
        return self.window.fix(time.monotonic() if now is None else now)

    def to_container_pose(self, got):
        pos, yaw = got
        top_residual_mm = (pos[2] - (self.table_z + self.model.dims[2])) * 1000
        print(
            "[depth] measured top %.1f mm from nominal — origin z pinned "
            "to the table" % top_residual_mm
        )
        return container_pose_from_top(pos, float(yaw), self.model, self.table_z)

    def status(self):
        missing = self.grab.missing()
        if missing:
            return "camera streams missing: %s" % ", ".join(missing)
        why = " (last reject: %s)" % self.last_reject if self.last_reject else ""
        return "%d/%d frames found a container top, %d button-circle%s" % (
            self.hits,
            self.frames,
            self.circle_hits,
            why,
        )
