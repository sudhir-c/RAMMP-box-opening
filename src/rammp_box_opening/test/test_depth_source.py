"""The depth pose source, proven against ray-cast synthetic scenes.

The renderer inverts the exact deprojection the detector performs: for
every pixel, the camera ray is intersected with the table plane and,
where the footprint covers it, the lid plane — so a recovered pose is a
round trip through real projective geometry, not an echo of the
detector's own math.
"""

import math

import numpy as np
import pytest

from rammp_box_opening.models.container import ContainerModel
from rammp_box_opening.perception.depth_source import (
    container_pose_from_top,
    top_face_from_depth,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"

# the field-observed scan-pose camera: looking straight down from 0.575
ROT_DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
T_CAM = np.array([0.42, -0.075, 0.575])
K = np.array([[430.0, 0.0, 424.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]])
TABLE_Z = -0.027
H, W = 480, 848


def render_depth(boxes, rot_cam=ROT_DOWN, t_cam=T_CAM, table_z=TABLE_Z):
    """Ray-cast a depth image of the table plus rotated-rect box tops.

    boxes: [(cx, cy, top_z, w, h, yaw)] in base_link.
    """
    us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    rays = np.stack(
        [(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)],
        axis=-1,
    ).reshape(-1, 3)
    d_base = rays @ rot_cam.T  # ray directions in base
    dz = d_base[:, 2]
    depth = np.full(rays.shape[0], np.nan)
    ok = np.abs(dz) > 1e-9

    def lam_for(plane_z):
        lam = np.full_like(dz, np.nan)
        lam[ok] = (plane_z - t_cam[2]) / dz[ok]
        return lam

    lam_t = lam_for(table_z)
    hit_t = ok & (lam_t > 0)
    depth[hit_t] = lam_t[hit_t]
    for cx, cy, top_z, bw, bh, yaw in boxes:
        lam_b = lam_for(top_z)
        p = t_cam + d_base * lam_b[:, None]
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        inside = ok & (lam_b > 0) & (np.abs(lx) < bw / 2) & (np.abs(ly) < bh / 2)
        depth[inside] = lam_b[inside]
    return depth.reshape(H, W).astype(np.float32)


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def test_recovers_the_field_box_pose(model):
    """The box exactly where the 2026-09-01 bench run detected it."""
    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], math.radians(9.0))
    depth = render_depth([truth])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(truth[0], abs=0.004)
    assert fix.center[1] == pytest.approx(truth[1], abs=0.004)
    assert fix.center[2] == pytest.approx(truth[2], abs=0.003)
    got = fix.yaw % (math.pi / 2)
    want = truth[5] % (math.pi / 2)
    err = min(abs(got - want), math.pi / 2 - abs(got - want))
    assert err < math.radians(3.0)
    # the measured footprint is the plateau CORE — the smoothness gate and
    # the core erosion shave a boundary ring by design, so it reads SMALLER
    # than the physical box (real captures read 0.056-0.104 for a 75 mm
    # lid). The gate's lower bound accounts for it; assert the core stays
    # inside the gate's window rather than at nominal size.
    for side in fix.footprint:
        assert 0.075 - 0.035 <= side <= 0.075 + 0.035

    cpose = container_pose_from_top(fix.center, fix.yaw, model)
    assert cpose.xyz[2] == pytest.approx(truth[2] - model.dims[2], abs=0.003)


def test_empty_table_yields_no_fix(model):
    depth = render_depth([])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "container height" in why


def test_wrong_size_plateau_is_rejected(model):
    """A tote-sized surface at box height must not be mistaken for the
    container — footprint sanity is the one defence the band has."""
    depth = render_depth([(0.45, 0.0, 0.080, 0.30, 0.22, 0.0)])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "footprint" in why


def test_truncated_box_at_the_border_is_refused(model):
    """Half a box in view = a biased centroid; refusal beats a guess."""
    # camera footprint at table: ~0.9 x 0.5 m around [0.42, -0.075];
    # park the box on the image's +x edge
    depth = render_depth([(0.42 - 0.51, -0.075, 0.080, 0.075, 0.075, 0.0)])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None


def test_wall_points_cannot_drag_the_centroid(model):
    """A second, lower plateau overlapping the band (stand-in for box
    side-wall returns) must not shift the fix: the plateau refinement
    re-tightens around the top face."""
    truth = (0.46, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    skirt = (0.46 + 0.050, -0.10, 0.035, 0.05, model.dims[1], 0.0)
    depth = render_depth([skirt, truth])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(truth[0], abs=0.005)
    assert fix.center[2] == pytest.approx(truth[2], abs=0.003)


def test_missing_depth_frame_is_honest(model):
    fix, why = top_face_from_depth(None, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and why == "no depth frame"


def test_two_boxes_are_ambiguous_without_an_roi(model):
    """Two container-sized tops: guessing which to press is refused."""
    a = (0.40, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    b = (0.55, 0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([a, b])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "ambiguous" in why


def test_roi_disambiguates_two_boxes(model):
    """A pixel-space roi (the VLM's bbox) picks the intended box."""
    a = (0.40, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    b = (0.55, 0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([a, b])
    # project box a's center into the image to build its roi
    p_cam = ROT_DOWN.T @ (np.array([a[0], a[1], a[2]]) - T_CAM)
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    roi = (u - 90, v - 90, u + 90, v + 90)
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model, roi=roi)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(a[0], abs=0.006)
    assert fix.center[1] == pytest.approx(a[1], abs=0.006)


def test_stereo_speckle_cannot_fake_a_box(model):
    """Locally-wild noise at container height (the blank-table failure of
    capture 20260901-130610) fails the smoothness gate: no false fix."""
    depth = render_depth([])
    rng = np.random.RandomState(7)
    # a large patch of the table speckled into the height band
    patch = depth[100:400, 200:700]
    depth[100:400, 200:700] = patch - rng.uniform(0.04, 0.11, patch.shape).astype(
        np.float32
    ) * (rng.rand(*patch.shape) < 0.5)
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None
