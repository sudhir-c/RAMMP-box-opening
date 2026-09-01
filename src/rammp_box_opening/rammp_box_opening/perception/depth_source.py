"""Tag-free container pose: the box found by GEOMETRY, not by a print.

The fiducial path died twice at the bench in one week — the gripper
knuckles strike the printed tag on every press, and a 30 mm print at
scan height has almost no detection margin. This source needs no print
at all: from the scan pose the lid top is the only plateau at container
height above the MEASURED table. Aligned-depth pixels are deprojected,
lifted to base_link with the same frame-stamp TF the tag path used
(camera_pose_at), banded by height, and the largest in-band blob whose
footprint matches the model becomes the sighting:

    plateau median z  = the lid top, MEASURED (the tag path only ever
                        derived it from nominal dims)
    plateau centroid  = the button center (the button is centered)
    min-area-rect     = yaw, mod 90 deg — the box is square, and every
                        consumer is yaw-mod-90 invariant: button_offset
                        is purely vertical, press attitude is steered by
                        BEARING, and both collision cuboids are square

BoxTopWatcher mirrors TagWatcher's whole surface (tick on a node timer,
FixWindow commit rules, status/reset/fix), so the mission swaps sources
without touching its detect flow. What this source cannot do: tell one
box from another, or see a box whose lid is off — both fine here, the
mission starts lid-on with one container on the bench by definition.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from rammp_box_opening.models.container import ContainerPose
from rammp_box_opening.perception.tag_source import FixWindow, camera_pose_at

# Height band above the table that could be a container top. Wide on
# purpose: dims.z is nominal, the plateau refinement below re-tightens
# around what it actually finds.
BAND_LO_M = 0.05
BAND_HI_M = 0.16
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


@dataclass(frozen=True)
class TopFaceFix:
    center: tuple  # button-top center, base frame (x, y, z=plateau median)
    yaw: float  # footprint yaw, mod pi/2
    footprint: tuple  # min-area-rect (w, h) in metres
    n_px: int  # plateau pixels (at STRIDE) backing the fix


def top_face_from_depth(depth, k, rot_cam, trans_cam, table_z, model):
    """One depth frame -> TopFaceFix, or None with honesty about why not.

    Returns (fix, why): fix is None when nothing container-like is seen;
    why is a short reason for the status line ("no plateau", "footprint
    0.31 m", "touches border", ...).
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

    # deproject every subsampled pixel and lift to base_link
    x = (uu - cx) / fx * d
    y = (vv - cy) / fy * d
    pts_cam = np.stack([x, y, d], axis=-1).reshape(-1, 3)
    rot = np.asarray(rot_cam, dtype=float)
    pts = pts_cam @ rot.T + np.asarray(trans_cam, dtype=float)
    z_base = pts[:, 2].reshape(gh, gw)

    band = valid & (z_base > table_z + BAND_LO_M) & (z_base < table_z + BAND_HI_M)
    if not band.any():
        return None, "no points at container height"

    import cv2

    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(
        band.astype(np.uint8), connectivity=8
    )
    if n_blobs < 2:
        return None, "no plateau"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = labels == biggest

    # border truncation = biased centroid: refuse rather than guess
    edge = np.zeros_like(blob)
    b = max(1, BORDER_PX // STRIDE)
    edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True
    if (blob & edge).any():
        return None, "container top touches the image border"

    # plateau refinement: the coarse band also catches side-wall points
    # seen at grazing angles — re-tighten around the blob's own top so
    # walls cannot drag the centroid sideways
    zb = z_base[blob]
    top_z = float(np.median(zb))
    plateau = blob & (np.abs(z_base - top_z) < PLATEAU_TOL_M)
    if plateau.sum() < 12:
        return None, "plateau too small (%d px)" % int(plateau.sum())

    sel = pts.reshape(gh, gw, 3)[plateau]
    xy = sel[:, :2].astype(np.float32)
    (rcx, rcy), (w, h), angle = cv2.minAreaRect(xy)
    w, h = float(w), float(h)
    ex, ey = float(model.dims[0]), float(model.dims[1])
    lo, hi = min(ex, ey) - FOOT_TOL_M, max(ex, ey) + FOOT_TOL_M
    if not (lo <= min(w, h) and max(w, h) <= hi):
        return None, "footprint %.2fx%.2f m vs model %.2fx%.2f" % (w, h, ex, ey)

    top_z = float(np.median(sel[:, 2]))
    fix = TopFaceFix(
        center=(float(rcx), float(rcy), top_z),
        yaw=math.radians(angle) % (math.pi / 2),
        footprint=(w, h),
        n_px=int(plateau.sum()),
    )
    return fix, "ok"


def container_pose_from_top(center, yaw, model):
    """Top-face fix -> ContainerPose (bottom-center origin + yaw), the
    same contract container_pose_from_tag honours: origin z = measured
    top minus nominal height, so button_offset lands back on the
    MEASURED lid top."""
    return ContainerPose(
        xyz=(float(center[0]), float(center[1]), float(center[2]) - model.dims[2]),
        yaw=float(yaw),
    )


class BoxTopWatcher:
    """TagWatcher's twin for the depth source: continuous detection on a
    node timer, same FixWindow commit rules, same fix()/status()/reset()
    surface — the mission cannot tell which source produced its pose."""

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
        self.last_debug = None  # the last TopFaceFix (bench diagnostics)
        self.last_reject = None  # why the last non-hit frame was refused
        self._last_stamp = None
        node.create_timer(period_s, self._tick)

    def _tick(self):
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
        fix, why = top_face_from_depth(
            g.depth, g.k, rot_cam, trans_cam, self.table_z, self.model
        )
        if fix is None:
            self.last_reject = why
            return
        self.hits += 1
        self.refined_hits += 1
        self.last_debug = fix
        self.window.add(np.asarray(fix.center), fix.yaw, time.monotonic())

    def reset(self):
        """Purge the window when the arm settles (same rationale as the
        tag path: in-motion sightings carry TF/depth timing skew)."""
        self.window.samples = []
        self.window.last_seen = None

    def fix(self, now=None):
        return self.window.fix(time.monotonic() if now is None else now)

    def to_container_pose(self, got):
        pos, yaw = got
        return container_pose_from_top(pos, float(yaw), self.model)

    def status(self):
        missing = self.grab.missing()
        if missing:
            return "camera streams missing: %s" % ", ".join(missing)
        why = " (last reject: %s)" % self.last_reject if self.last_reject else ""
        return "%d/%d frames found a container top%s" % (
            self.hits,
            self.frames,
            why if self.hits == 0 else "",
        )
