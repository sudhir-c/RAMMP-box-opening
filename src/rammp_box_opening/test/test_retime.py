"""The human-motion re-timer: positions untouched, limits by construction,
flow through shallow corners, stop at reversals, long ease-in."""
import math

import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.constants import JOINT_VMAX
from rammp_box_opening.runtime.retime import (
    RetimeParams,
    check_like_executor,
    concat_paths,
    retime_group,
    speed_profile,
    velocity_caps,
)


def _traj(points, dt=0.02):
    msg = JointTrajectory()
    msg.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k, q in enumerate(points):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        t = (k + 1) * dt
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        msg.points.append(p)
    return msg


def _line(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _arrays(msg):
    q = np.asarray([p.positions for p in msg.points])
    v = np.asarray([p.velocities for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    return q, v, t


def test_positions_untouched_and_gates_hold():
    src = _traj(_line([0] * 7, [0.5, -0.3, 0.2, 0.0, 0.4, -0.2, 0.1], 60))
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    q, v, t = _arrays(out)
    assert np.allclose(q, np.asarray([p.positions for p in src.points]))
    assert check_like_executor(out, JOINT_VMAX) == []
    assert np.all(np.diff(t) > 0)
    assert np.allclose(v[0], 0) and np.allclose(v[-1], 0)  # rest at both ends
    assert np.abs(v).max(axis=0).max() <= 0.9 * max(JOINT_VMAX) + 1e-9
    assert info["stops"] == 0 and info["junction_speeds"] == []


def test_arrival_is_the_long_half():
    """Asymmetric ease: decel < accel, so the slow-down into rest takes
    longer than the ease out of it — the precision signature."""
    src = _traj(_line([0] * 7, [0.8, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0], 80))
    out, _ = retime_group([src], [1.0], JOINT_VMAX)
    q, v, t = _arrays(out)
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = ds / np.diff(t)
    k = int(s.argmax())
    assert t[-1] - t[k] > 1.3 * t[k]  # tail longer than head


def test_shallow_corner_flows_and_reversal_stops():
    a = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 30)
    b = _line([0.4, 0, 0, 0, 0, 0, 0], [0.4, 0.4, 0, 0, 0, 0, 0], 30)  # 90 deg corner
    out, info = retime_group([_traj(a), _traj(b)], [0.75, 0.75], JOINT_VMAX)
    assert 50 <= info["n_points"] <= 59  # junction de-duplicated, neighbours thinned
    assert info["stops"] == 0
    assert 0.15 < info["junction_speeds"][0] < 1.0  # slowed, not stopped
    assert check_like_executor(out, JOINT_VMAX) == []
    back = _line([0.4, 0, 0, 0, 0, 0, 0], [0.0] * 7, 30)  # full reversal
    out2, info2 = retime_group([_traj(a), _traj(back)], [0.75, 0.75], JOINT_VMAX)
    assert info2["stops"] == 1 and info2["junction_speeds"][0] == 0.0
    assert check_like_executor(out2, JOINT_VMAX) == []


def test_each_leg_keeps_its_own_cruise_fraction():
    a = _line([0] * 7, [0.5, 0, 0, 0, 0, 0, 0], 40)
    b = _line([0.5, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0], 40)  # straight on
    out, _ = retime_group([_traj(a), _traj(b)], [0.75, 0.35], JOINT_VMAX)
    q, v, t = _arrays(out)
    s = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(t)
    fast = s[10:30].max()
    slow = s[50:70].max()
    assert slow < 0.6 * fast  # the carry-class leg cruises slower


def test_per_joint_cap_binds_on_the_busiest_joint():
    q = _line([0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 50)  # only joint_7 moves
    cap, _ = velocity_caps(np.asarray(q), JOINT_VMAX, 0.9)
    assert np.allclose(cap[1:-1], 0.9 * JOINT_VMAX[6])


def test_duplicates_and_short_paths_do_not_break():
    q = [[0.1] * 7] * 5 + _line([0.1] * 7, [0.3] * 7, 3) + [[0.3] * 7] * 4
    out, info = retime_group([_traj(q)], [0.75], JOINT_VMAX)
    assert info["n_points"] == 3  # the holds at both ends collapsed
    assert check_like_executor(out, JOINT_VMAX) == []
    single = _traj([[0.2] * 7])
    same, info1 = retime_group([single], [0.75], JOINT_VMAX)
    assert same is single and info1["n_points"] == 1


def test_speed_profile_respects_caps_everywhere():
    q = np.asarray(_line([0] * 7, [0.9, 0.9, 0, 0, 0, 0, 0], 70))
    v, _ = speed_profile(q, np.full(len(q), 1.0), JOINT_VMAX, RetimeParams())
    cap, _ = velocity_caps(q, JOINT_VMAX, 0.9)
    assert np.all(v <= cap + 1e-9) and v[0] == 0.0 and v[-1] == 0.0


def test_time_scale_is_a_uniform_dilation_after_profiling():
    src = _traj(_line([0] * 7, [0.6, 0.2, 0, 0, 0, 0, 0], 50))
    fast, i1 = retime_group([src], [0.75], JOINT_VMAX, RetimeParams())
    slow, i2 = retime_group([src], [0.75], JOINT_VMAX, RetimeParams(time_scale=0.25))
    assert i2["duration_s"] == pytest.approx(4 * (i1["duration_s"] - 0.02) + 0.02, rel=1e-6)
    assert check_like_executor(slow, JOINT_VMAX) == []


def test_two_point_path_is_a_triangle_not_a_crawl():
    """An interval between two rest samples is timed accelerate-then-
    decelerate, not at the 0.02 rad/s floor (13 s for 0.26 rad)."""
    src = _traj([[0.0] * 7, [0.1] * 7])
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    assert info["duration_s"] < 1.0
    assert check_like_executor(out, JOINT_VMAX) == []


def test_reversal_stops_only_at_its_apex():
    a = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 30)
    back = _line([0.4, 0, 0, 0, 0, 0, 0], [0.0] * 7, 30)
    out, info = retime_group([_traj(a), _traj(back)], [0.75, 0.75], JOINT_VMAX)
    q, v, t = _arrays(out)
    s = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(t)
    assert info["stops"] == 1
    assert info["duration_s"] < 3.0  # no multi-second crawl around the apex
    assert (s < 0.05).sum() <= 2  # only the intervals touching the apex are slow


def test_float32_near_duplicates_are_merged():
    """cuRobo emits float32 positions: one-ulp near-duplicates (2e-7 rad)
    must not become nanosecond intervals and 1e5 rad/s^2 on the wire."""
    q = _line([0] * 7, [0.5, 0, 0, 0, 0, 0, 0], 40)
    q.insert(20, q[20] + np.array([0, 0, 0, 0, 0, 2.4e-7, 0]))
    out, info = retime_group([_traj(q)], [0.75], JOINT_VMAX)
    assert info["n_points"] == 40
    _, _, t = _arrays(out)
    assert np.diff(t).min() > 1e-4
    acc = np.asarray([p.accelerations for p in out.points])
    assert np.abs(acc).max() < 30.0


def test_kink_between_chained_legs_is_bounded_by_the_robot_accel():
    """Two legs meeting at a kink inside one sample interval: the cap must
    come from the LOCAL geometry and amax, not from a wide window."""
    from rammp_box_opening.runtime.retime import corner_caps

    # dense samples near the junction, as a planner's deceleration tail has
    a = _line([0] * 7, [0.3, 0, 0, 0, 0, 0, 0], 20) + _line([0.3, 0, 0, 0, 0, 0, 0], [0.302, 0, 0, 0, 0, 0, 0], 6)[1:]
    b = _line([0.302, 0, 0, 0, 0, 0, 0], [0.302, 0.002, 0, 0, 0, 0, 0], 6)[1:] + _line([0.302, 0.002, 0, 0, 0, 0, 0], [0.302, 0.3, 0, 0, 0, 0, 0], 20)[1:]
    q = np.asarray(a + b)
    from rammp_box_opening.runtime.retime import RetimeParams
    p = RetimeParams()
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cap, stop = corner_caps(q, ds, p)
    k = 24  # the kink sample
    L_local = ds[k - 1] + ds[k]
    assert cap[k] <= math.sqrt(p.amax * L_local / (2 * math.sin(math.pi / 4))) + 1e-9
    assert not stop.any()


def test_long_rest_to_rest_interval_respects_the_joint_cap():
    """A 2-point path longer than the triangle can cover under the cap
    becomes a trapezoid at the cap, never a peak above 0.9 x the limit."""
    src = _traj([[0.0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # 1 rad on joint_7
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    _, v, t = _arrays(out)
    # velocities on the wire are zero at both samples; check the implied
    # average speed against the cap instead
    assert 1.0 / (t[-1] - t[0]) < 0.9 * JOINT_VMAX[6]
    assert check_like_executor(out, JOINT_VMAX) == []
    # the crawl is gone but the cap held: duration bounded below by ds/cap
    assert info["duration_s"] > 1.0 / (0.9 * JOINT_VMAX[6])


def test_executed_acceleration_at_a_crowded_junction_stays_under_amax():
    """The real defect: two legs meeting at a kink with the planner's
    crowded end samples. The JTC's quintic must not be asked for more
    than the robot's 25 rad/s^2 anywhere — measured on the spline, not
    on the cap formula (review 2026-09-03)."""
    from jtc_model import peak_accel

    def crowded_tail(a, b, n_far=20, n_near=12, tail=0.02):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        far = _line(a, b - d * tail, n_far)
        near = _line(b - d * tail, b, n_near)[1:]  # 12 samples over the last 2 cm
        return far + near

    def crowded_head(a, b, n_near=12, n_far=20, head=0.02):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        near = _line(a, a + d * head, n_near)
        far = _line(a + d * head, b, n_far)[1:]
        return near + far

    P0 = [0.0] * 7
    P1 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    P2 = [0.4 + 0.4 * math.cos(math.radians(60)), 0.4 * math.sin(math.radians(60)), 0, 0, 0, 0, 0]
    legs = [_traj(crowded_tail(P0, P1)), _traj(crowded_head(P1, P2))]
    out, info = retime_group(legs, [0.75, 0.75], JOINT_VMAX)
    assert check_like_executor(out, JOINT_VMAX) == []
    assert info["junction_speeds"][0] > 0.15  # it flows
    peak, _ = peak_accel(out)
    print("junction %.2f rad/s, executed peak %.1f rad/s^2" % (info["junction_speeds"][0], peak))
    assert peak <= 25.0, peak


def test_reverse_tail_starts_at_the_live_stop_and_never_goes_deeper():
    """The walk back begins at the sample NEAREST the stop, not at the
    deepest planned one — stepping to that would drive into the contact."""
    from rammp_box_opening.runtime.retime import reverse_tail

    down = _line([0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 50)
    src = _traj(down)
    live = np.asarray(down[30])  # the guard stopped here, short of the plan
    path = reverse_tail(src, progress=0.62, live=live, arc_rad=0.09)
    assert np.allclose(path[0], live)  # the executor's start gate
    depth = [float(q[6]) for q in path]
    assert all(b <= a + 1e-9 for a, b in zip(depth, depth[1:]))  # only back out
    assert depth[0] - depth[-1] >= 0.09 - 1e-6  # the arc was covered
    # a trip at the very start has nothing to reverse
    assert reverse_tail(src, progress=0.0, live=np.asarray(down[0]), arc_rad=0.09) is None


def test_reverse_tail_covers_the_cancel_overshoot():
    """Cancel latency carries the arm past the last fully elapsed point;
    the retrace must include it."""
    from rammp_box_opening.runtime.retime import reverse_tail

    down = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 20)
    path = reverse_tail(_traj(down), progress=0.5, live=np.asarray(down[10]), arc_rad=0.05)
    assert path is not None and len(path) >= 3
