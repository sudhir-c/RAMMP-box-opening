import math

import pytest

from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    attitude_quat,
    from_container,
    load_lid_place,
    wrist_flat_quat,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def test_load_and_validate():
    m = ContainerModel.load(CFG)
    assert m.measure_me is False  # owner accepted values 2026-08-25
    assert len(m.dims) == 3 and len(m.lid_dims) == 3
    assert m.button_offset[2] == pytest.approx(m.dims[2])  # button = top
    assert m.touch_nm > 0 and m.hover_standoff > 0


def test_from_container_rotates_by_yaw():
    cpose = ContainerPose(xyz=(1.0, 2.0, 0.0), yaw=math.pi / 2)
    # +x offset in container frame maps to +y in base at yaw 90 deg
    assert from_container(cpose, (0.1, 0.0, 0.05)) == pytest.approx([1.0, 2.1, 0.05])


def test_wrist_flat_quat_is_bearing_steered_xyzw():
    q0 = wrist_flat_quat([1.0, 0.0, 0.5])  # bearing 0 -> home attitude
    assert q0 == pytest.approx([0.5, 0.5, 0.5, 0.5])
    q90 = wrist_flat_quat([0.0, 1.0, 0.5])
    assert q90 != pytest.approx(q0)  # steered, unit
    assert sum(v * v for v in q90) == pytest.approx(1.0)


def test_attitude_quat_top_down_points_tool_down():
    from rammp_curobo.geometry import tool_axis, xyzw_to_wxyz

    q = attitude_quat([180.0, 0.0, 0.0], yaw=0.7)
    ax = tool_axis(xyzw_to_wxyz(q))  # explicit order conversion
    assert ax[2] == pytest.approx(-1.0, abs=1e-6)  # tool z straight down


def test_lid_place_is_a_table_height_pose():
    lid = load_lid_place(CFG)
    assert len(lid.xyz) == 3
    assert lid.xyz[2] == pytest.approx(-0.027)  # the measured table top


def test_load_press_demo_cfg():
    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    assert cfg.detect_source in ("depth", "vlm")
    assert cfg.grip_band[0] < cfg.grip_band[1] < 0.8
    assert 0.0 <= cfg.grip_clear_m <= 0.02
    assert all(abs(v) <= 0.02 for v in cfg.grip_offset_xy)
    assert cfg.lift_m > 0 and 0 < cfg.lift_speed <= 1.0
    assert cfg.press_speed == pytest.approx(0.35)  # owner decision 2026-08-25
    assert cfg.travel_m > 0
    assert cfg.min_hits >= 2 and cfg.timeout_s > 0


def _cfg_variant(tmp_path, old, new):
    p = tmp_path / "variant.yaml"
    p.write_text(open(CFG).read().replace(old, new))
    return str(p)


def test_loader_validates_staging_against_real_container_geometry(tmp_path):
    from rammp_box_opening.models.container import load_press_demo

    # recessed button: dims.z 0.112, button_offset.z 0.05 -> staging
    # must clear 0.112 - 0.05 + 0.10 = 0.162; the shipped 0.12 fails
    with pytest.raises(ValueError, match="staging_m"):
        load_press_demo(
            _cfg_variant(
                tmp_path,
                "button_offset: [0.0, 0.0, 0.112]",
                "button_offset: [0.0, 0.0, 0.05]",
            )
        )


def test_loader_refuses_the_retired_tag_source(tmp_path):
    from rammp_box_opening.models.container import load_press_demo

    with pytest.raises(ValueError, match="detect.source"):
        load_press_demo(_cfg_variant(tmp_path, "source: vlm", "source: tag"))
