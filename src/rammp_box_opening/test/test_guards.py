import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import (
    TorqueGuard,
    check_standoff,
    classify_press,
    in_band,
    min_standoff,
    press_outcome,
    reverse_retrace,
    sanity_violations,
)


def _traj(rows, dt=0.5):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, len(rows[0]) + 1)]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * len(row)
        p.accelerations = [0.0] * len(row)
        p.time_from_start.sec = int(i * dt)
        p.time_from_start.nanosec = int((i * dt % 1) * 1e9)
        t.points.append(p)
    return t


def test_guard_baseline_anchored_at_progress():
    g = TorqueGuard(touch_nm=3.0)
    assert g.on_efforts([9.0, 9.0, 9.0, 9.0]) is False  # not armed: ignored
    g.on_progress(0.0)
    assert g.armed is False  # progress 0 != started
    g.on_progress(0.01)
    assert g.on_efforts([1.0, 1.0, 1.0, 1.0]) is False  # first sample = baseline
    assert g.on_efforts([2.0, 1.0, 1.0, 1.0]) is False  # dev 1.0 < 3.0
    assert g.on_efforts([1.0, 5.0, 1.0, 1.0]) is True  # dev 4.0 > 3.0
    assert g.peak == pytest.approx(4.0)


def test_guard_ignores_none_efforts():
    g = TorqueGuard(touch_nm=3.0)
    g.on_progress(0.5)
    assert g.on_efforts(None) is False


def test_sanity_gate_flags_wandering_joint():
    # joint_1 wanders 1.0 rad out and back on a 0.1 rad net move
    bad = _traj([[0.0, 0.0], [1.0, 0.05], [0.1, 0.1]])
    good = _traj([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
    assert sanity_violations(good, margin_rad=0.35) == []
    v = sanity_violations(bad, margin_rad=0.35)
    assert len(v) == 1 and "joint_1" in v[0]


def test_sanity_gate_wrap_aware():
    # joint crossing the pi boundary: 3.10 -> -3.10 is a 0.08 rad move
    t = _traj([[3.10], [3.14], [-3.10]])
    assert sanity_violations(t, margin_rad=0.35) == []


def test_standoff_floor():
    assert min_standoff() == pytest.approx(0.051)
    check_standoff(hover_z=0.10, contact_z=0.0)  # 0.10 > 0.051: ok
    with pytest.raises(ValueError):
        check_standoff(hover_z=0.04, contact_z=0.0)


def test_press_classification():
    window = (0.004, 0.012)
    assert classify_press(0.008, window) == "pressed"
    assert classify_press(0.001, window) == "rim"
    ok, detail = press_outcome("touch", 0.008, window)
    assert ok
    ok, detail = press_outcome("touch", 0.001, window)
    assert not ok and "rim" in detail
    ok, detail = press_outcome("arrived", None, window)  # bottomed out, no click
    assert not ok


def test_in_band():
    assert in_band(0.6, (0.55, 0.75))
    assert not in_band(0.8, (0.55, 0.75))  # closed on air


def test_reverse_retrace_reverses_executed_portion():
    t = _traj([[0.0], [0.2], [0.4], [0.6]], dt=1.0)  # 3 s total
    r = reverse_retrace(t, progress=0.5)  # stopped ~1.5 s in
    starts = [p.positions[0] for p in r.points]
    assert starts[0] == pytest.approx(0.4)  # from deepest executed
    assert starts[-1] == pytest.approx(0.0)  # back to the start
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in r.points]
    assert times == sorted(times) and times[0] == 0.0  # re-timed, monotonic
