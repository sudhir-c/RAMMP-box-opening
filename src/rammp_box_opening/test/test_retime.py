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
    assert info["n_points"] == 59  # the shared junction sample de-duplicated
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
