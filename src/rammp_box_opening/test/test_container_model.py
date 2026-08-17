import math

import pytest

from rammp_box_opening.models.container import (
    ConfigPoseSource,
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
    assert m.measure_me is True  # placeholders flagged
    assert m.press_depth_window[0] < m.press_depth_window[1]
    assert m.lid_grasp.width_m <= m.aperture_at_0  # graspable
    assert m.body_grasp.width_m <= m.aperture_at_0


def test_width_to_command_endpoints_and_refusal():
    m = ContainerModel.load(CFG)
    assert m.width_to_command(m.aperture_at_0) == pytest.approx(0.0)
    assert m.width_to_command(m.aperture_at_08) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        m.width_to_command(m.aperture_at_0 + 0.01)


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


def test_pose_source_and_lid_place():
    cp = ConfigPoseSource(CFG).container_pose()
    assert len(cp.xyz) == 3
    lid = load_lid_place(CFG)
    assert lid.xyz != cp.xyz  # set-down spot is elsewhere
