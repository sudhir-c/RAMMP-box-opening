import math

import numpy as np
import pytest

from rammp_box_opening.models.container import ContainerModel, load_press_demo
from rammp_box_opening.perception.tag_source import (
    FixWindow,
    container_pose_from_tag,
    refine_point,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"

R_FLAT = np.eye(3)  # tag lying flat, +z up, +x along world +x
K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])


def test_container_pose_from_tag_identity_yaw():
    m = ContainerModel.load(CFG)
    cp = container_pose_from_tag([0.5, 0.1, 0.133], R_FLAT, m, (0.0, 0.0, 0.0))
    assert cp.yaw == pytest.approx(0.0)
    # tag on the button: container origin = tag minus button_offset
    assert list(cp.xyz) == pytest.approx([0.5, 0.1, 0.133 - m.button_offset[2]])


def test_container_pose_from_tag_rotates_offsets_by_yaw():
    m = ContainerModel.load(CFG)
    r90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cp = container_pose_from_tag([0.5, 0.0, 0.133], r90, m, (0.02, 0.0, 0.0))
    assert cp.yaw == pytest.approx(math.pi / 2)
    # container-frame +x tag->button offset maps to world +y at yaw 90
    assert cp.xyz[0] == pytest.approx(0.5)
    assert cp.xyz[1] == pytest.approx(0.02)
    assert cp.xyz[2] == pytest.approx(0.133 - m.button_offset[2])


def test_refine_point_replaces_range_with_depth_at_the_tag_pixel():
    tvec = np.array([0.05, 0.0, 0.30])  # RGB solve: 0.30 m out
    # structured depth: the true value ONLY at the tag's projected pixel
    # window, decoy elsewhere — a wrong-pixel lookup (transposed indices,
    # off-by-N) reads 0.60 and fails the assertions (2026-08-24 review)
    depth = np.full((480, 640), 0.60, np.float32)
    u = int(round(400.0 * 0.05 / 0.30 + 320.0))  # 386
    v = 240
    depth[v - 4 : v + 5, u - 4 : u + 5] = 0.35
    p = refine_point(tvec, K, depth)
    assert p is not None
    assert p[2] == pytest.approx(0.35)
    # x re-derived from the same pixel ray at the corrected depth
    assert p[0] == pytest.approx(0.05 * 0.35 / 0.30, rel=1e-3)


def test_refine_point_refuses_bad_depth():
    tvec = np.array([0.0, 0.0, 0.30])
    assert refine_point(tvec, K, None) is None
    assert refine_point(tvec, K, np.full((480, 640), 2.5, np.float32)) is None
    assert refine_point(tvec, K, np.zeros((480, 640), np.float32)) is None
    # tag center projecting outside the image
    far = np.array([1.0, 0.0, 0.2])
    assert refine_point(far, K, np.full((480, 640), 0.3, np.float32)) is None


def test_fix_window_commits_median_then_goes_stale():
    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    assert fw.fix(now=0.0) is None
    for i, t in enumerate([0.0, 0.2, 0.4]):
        fw.add(np.array([0.5, 0.0, 0.130 + 0.001 * i]), R_FLAT, t)
    got = fw.fix(now=0.5)
    assert got is not None
    pos, rot = got
    assert pos[2] == pytest.approx(0.131)
    assert rot.shape == (3, 3)
    assert fw.fix(now=2.0) is None  # nothing seen for > fresh_s


def test_fix_window_rejects_disagreeing_frames():
    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    fw.add(np.array([0.50, 0.0, 0.13]), R_FLAT, 0.0)
    fw.add(np.array([0.60, 0.0, 0.13]), R_FLAT, 0.2)  # 10 cm jump
    fw.add(np.array([0.50, 0.0, 0.13]), R_FLAT, 0.4)
    assert fw.fix(now=0.5) is None


def test_load_press_demo_cfg():
    cfg = load_press_demo(CFG)
    assert cfg.tag_id == 0
    assert cfg.tag_size_m == pytest.approx(0.05)
    assert cfg.hover_m == pytest.approx(0.0254)
    assert cfg.press_speed == pytest.approx(0.25)  # owner decision 2026-08-24
    assert cfg.travel_m > 0
    assert cfg.min_hits >= 2 and cfg.timeout_s > 0


def _cfg_variant(tmp_path, old, new):
    p = tmp_path / "variant.yaml"
    p.write_text(open(CFG).read().replace(old, new))
    return str(p)


def test_loader_refuses_nonpositive_hover(tmp_path):
    # hover_m is the ONLY thing keeping the unguarded hover leg out of
    # contact (check_standoff is skipped for the tag-driven press)
    with pytest.raises(ValueError, match="hover_m"):
        load_press_demo(_cfg_variant(tmp_path, "hover_m: 0.0254", "hover_m: 0.0"))
    with pytest.raises(ValueError, match="hover_m"):
        load_press_demo(_cfg_variant(tmp_path, "hover_m: 0.0254", "hover_m: -0.01"))


def test_loader_validates_staging_against_real_container_geometry(tmp_path):
    # recessed button: dims.z 0.16, button_offset.z 0.11 -> staging must
    # clear dims.z - button_offset.z + 0.04 = 0.09; the default 0.08 fails
    with pytest.raises(ValueError, match="staging_m"):
        load_press_demo(
            _cfg_variant(
                tmp_path,
                "button_offset: [0.0, 0.0, 0.16]",
                "button_offset: [0.0, 0.0, 0.11]",
            )
        )
