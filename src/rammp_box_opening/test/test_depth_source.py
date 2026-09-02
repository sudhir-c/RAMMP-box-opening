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


def test_camera_still_filter():
    """Only still-camera frames may vote: moving frames carry a
    systematic TF-vs-exposure bias that agreed with itself 10-15 mm off
    the truth and pressed the button's EDGE (field 2026-09-01)."""
    from rammp_box_opening.perception.depth_source import camera_is_still

    eye = np.eye(3)
    a = (eye, np.array([0.42, -0.07, 0.575]))
    assert camera_is_still(a, (eye, a[1] + [0.001, 0.0, 0.0]))
    assert not camera_is_still(a, (eye, a[1] + [0.01, 0.0, 0.0]))  # translating
    c, s = np.cos(0.05), np.sin(0.05)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert not camera_is_still(a, (rz @ eye, a[1]))  # rotating
    assert not camera_is_still(None, a)  # first frame never votes


def _project(pt, rot=ROT_DOWN, t=T_CAM):
    p = rot.T @ (np.asarray(pt, dtype=float) - t)
    return (
        K[0, 0] * p[0] / p[2] + K[0, 2],
        K[1, 1] * p[1] / p[2] + K[1, 2],
        p[2],
    )


def test_button_circle_refine_recovers_the_true_button(model):
    """The plateau centroid finds the LID; the circle finds the BUTTON.
    A dark disk painted 8 mm off the centroid must win the aim."""
    import cv2

    from rammp_box_opening.perception.depth_source import button_circle_refine

    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([truth])
    color = np.full((H, W, 3), 235, dtype=np.uint8)
    button = (truth[0] + 0.008, truth[1] - 0.008, truth[2])
    u, v, z = _project(button)
    r_px = int(K[0, 0] * (model.button_diameter_m / 2) / z)
    cv2.circle(color, (int(u), int(v)), r_px, (60, 60, 60), -1)

    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    got = button_circle_refine(
        color, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m
    )
    assert got is not None
    assert got[0] == pytest.approx(button[0], abs=0.003)
    assert got[1] == pytest.approx(button[1], abs=0.003)


def test_button_circle_declines_honestly(model):
    """No circle in view, or a truncated crop -> None; the centroid
    fallback is always sane."""
    from rammp_box_opening.perception.depth_source import button_circle_refine

    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([truth])
    blank = np.full((H, W, 3), 235, dtype=np.uint8)
    fix, _ = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert (
        button_circle_refine(
            blank, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m
        )
        is None
    )
    assert (
        button_circle_refine(
            None, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m
        )
        is None
    )
    # centre projected at the image edge -> truncated crop -> decline
    edge_center = (0.85, -0.075, 0.080)
    assert (
        button_circle_refine(
            blank, depth, K, ROT_DOWN, T_CAM, edge_center, model.button_diameter_m
        )
        is None
    )


def test_origin_z_is_pinned_to_the_calibrated_table(model):
    """The plateau median wandered -3 mm one run (grip grazed the lid)
    and +8 mm the next (field 2026-09-01/02); the box rests on the
    surveyed table and its moulded height is constant, so the origin z
    comes from table_z whenever one is known."""
    from rammp_box_opening.perception.depth_source import container_pose_from_top

    top = (0.38, -0.17, 0.088)  # measured 8 mm above nominal
    pinned = container_pose_from_top(top, 0.0, model, table_z=-0.027)
    assert pinned.xyz[2] == -0.027
    legacy = container_pose_from_top(top, 0.0, model)
    assert abs(legacy.xyz[2] - (0.088 - model.dims[2])) < 1e-9


def test_watcher_wires_the_table_into_the_pose(model, capsys):
    """The 2026-09-02 regression: a 6-frame plateau median 8 mm high
    became origin z=-0.019 and shifted the press geometry. The wiring
    under test is BoxTopWatcher.to_container_pose passing its calibrated
    table_z through — dropping it must fail THIS test."""
    from rammp_box_opening.perception.depth_source import BoxTopWatcher

    w = BoxTopWatcher.__new__(BoxTopWatcher)  # wiring only, no node
    w.model = model
    w.table_z = TABLE_Z
    top = (0.38, -0.17, TABLE_Z + model.dims[2] + 0.008)
    cp = w.to_container_pose((top, 0.3))
    assert cp.xyz[2] == TABLE_Z
    assert "8.0 mm" in capsys.readouterr().out  # residual stays visible


def test_far_from_nominal_top_is_refused_for_commit(model):
    """With origin z pinned, a top far from table+dims means not-our-box
    or a stale calibration; pressing nominal geometry would stroke air
    (below) or leave real lid unmodelled (above). The gate is tighter
    than travel_m so a committed fix always reaches material."""
    from rammp_box_opening.perception.depth_source import top_residual_reject

    nominal = TABLE_Z + model.dims[2]
    assert top_residual_reject(nominal + 0.008, TABLE_Z, model) is None
    assert top_residual_reject(nominal - 0.008, TABLE_Z, model) is None
    for bad in (nominal + 0.020, nominal - 0.020):
        why = top_residual_reject(bad, TABLE_Z, model)
        assert why and "recalibrat" in why


def test_fix_window_commits_median_then_goes_stale():
    from rammp_box_opening.perception.depth_source import FixWindow

    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    assert fw.fix(now=0.0) is None
    for i, t in enumerate([0.0, 0.2, 0.4]):
        fw.add(np.array([0.5, 0.0, 0.130 + 0.001 * i]), 0.3, t)
    got = fw.fix(now=0.5)
    assert got is not None
    pos, yaw = got
    assert pos[2] == pytest.approx(0.131)
    assert float(yaw) == pytest.approx(0.3)
    assert fw.fix(now=2.0) is None  # nothing seen for > fresh_s


def test_fix_window_rejects_disagreeing_frames():
    from rammp_box_opening.perception.depth_source import FixWindow

    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    fw.add(np.array([0.50, 0.0, 0.13]), 0.0, 0.0)
    fw.add(np.array([0.60, 0.0, 0.13]), 0.0, 0.2)  # 10 cm jump
    fw.add(np.array([0.50, 0.0, 0.13]), 0.0, 0.4)
    assert fw.fix(now=0.5) is None


def test_watcher_reset_zeroes_the_window_counters(model):
    """The re-look line printed scan-pose totals and hid its own reject
    reason (2026-09-02): reset() now zeroes the per-window counters and
    status() always names the last reject."""
    from rammp_box_opening.perception.depth_source import BoxTopWatcher, FixWindow

    w = BoxTopWatcher.__new__(BoxTopWatcher)
    w.grab = type("G", (), {"missing": lambda self: []})()
    w.window = FixWindow(3, 0.015, 2.0, 1.0)
    w.frames, w.hits, w.refined_hits, w.circle_hits = 40, 12, 12, 11
    w.last_reject = "a top touches the image border"
    assert "border" in w.status()  # reported even though hits > 0
    w.reset()
    assert (w.frames, w.hits, w.circle_hits, w.last_reject) == (0, 0, 0, None)


def test_offered_roi_applies_only_from_the_still_epoch(model):
    """A bbox from a frame taken while the camera was still decelerating
    must not gate the parked frames; one from a still frame must (review
    2026-09-02)."""
    from rammp_box_opening.perception.depth_source import BoxTopWatcher

    w = BoxTopWatcher.__new__(BoxTopWatcher)
    w.roi = None
    w._still_since = None
    w._pending_roi = None
    w.offer_roi((1, 2, 3, 4), frame_t=10.0)  # seen while moving
    w._apply_pending_roi(still=False, frame_t=10.2)
    w._apply_pending_roi(still=True, frame_t=10.5)  # still epoch starts at 10.5
    assert w.roi is None  # the moving-frame bbox was dropped
    w.offer_roi((5, 6, 7, 8), frame_t=10.9)  # seen while still
    w._apply_pending_roi(still=True, frame_t=11.0)
    assert w.roi == (5, 6, 7, 8)
    w._apply_pending_roi(still=False, frame_t=12.0)  # moved again
    w.offer_roi((9, 9, 9, 9), frame_t=11.9)
    w._apply_pending_roi(still=True, frame_t=12.3)
    assert w.roi == (5, 6, 7, 8)  # the pre-epoch bbox did not apply
