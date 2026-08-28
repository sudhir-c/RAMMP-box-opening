import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.warp import warp_trajectory


def _descent(n=60, dt=0.02, reach=0.6):
    """A uniform single-joint descent: n points, constant step, constant v."""
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    v = reach / (n * dt)
    for i in range(n):
        p = JointTrajectoryPoint()
        p.positions = [reach * i / (n - 1)] + [0.0] * 6
        p.velocities = [v] + [0.0] * 6
        tt = (i + 1) * dt
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int(round((tt - int(tt)) * 1e9))
        t.points.append(p)
    return t


def _times(traj):
    return [
        p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points
    ]


def test_warp_leaves_every_position_untouched():
    """The whole safety argument: the validated PATH is the executed path."""
    src = _descent()
    out, _ = warp_trajectory(src, 0.3, 0.5, 0.15)
    assert len(out.points) == len(src.points)
    for a, b in zip(src.points, out.points):
        assert list(a.positions) == list(b.positions)


def test_warp_never_commands_a_faster_velocity_than_planned():
    """Every scale is <= 1.0, so the executor's limit check cannot newly
    fail on a warped trajectory."""
    src = _descent()
    out, _ = warp_trajectory(src, 0.3, 0.5, 0.15)
    for a, b in zip(src.points, out.points):
        for va, vb in zip(a.velocities, b.velocities):
            assert abs(vb) <= abs(va) + 1e-12


def test_warp_times_stay_strictly_monotonic():
    """A non-positive dt is rejected outright by the executor."""
    out, _ = warp_trajectory(_descent(), 0.3, 0.5, 0.15)
    ts = _times(out)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts[0] > 0.0


def test_warp_is_slower_overall_but_faster_early():
    src = _descent()
    out, arm = warp_trajectory(src, 0.3, 0.5, 0.15)
    src_t, out_t = _times(src), _times(out)
    # the whole thing takes longer than the full-speed plan...
    assert out_t[-1] > src_t[-1]
    # ...but far less than running the WHOLE leg at contact speed, which
    # is what a single speed_scale did
    assert out_t[-1] < src_t[-1] / 0.15
    assert 0.0 < arm < 1.0


def test_warp_arm_fraction_marks_the_slow_zone():
    """rebaseline_after must land where the speed actually changes, so the
    guard re-baselines in the regime the touch happens in."""
    out, arm = warp_trajectory(_descent(), 0.25, 0.5, 0.1)
    ts = _times(out)
    total = ts[-1]
    # the tail after arm_frac should be moving at the slow scale: its
    # per-point dt is 5x the fast zone's (0.5 / 0.1)
    dts = [b - a for a, b in zip(ts, ts[1:])]
    assert dts[-1] == pytest.approx(dts[0] * 5.0, rel=0.05)
    # and the marked time is genuinely inside the trajectory
    assert 0.0 < arm * total < total


def test_warp_declines_when_it_would_do_nothing():
    src = _descent()
    for bad in ((0.3, 0.15, 0.15), (0.3, 0.1, 0.5), (0.0, 0.5, 0.15), (1.0, 0.5, 0.15)):
        out, arm = warp_trajectory(src, *bad)
        assert out is src and arm is None


def test_warp_rejects_a_scale_above_full_speed():
    with pytest.raises(ValueError, match="warp scales"):
        warp_trajectory(_descent(), 0.3, 1.5, 0.15)
