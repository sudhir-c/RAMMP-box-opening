import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import (
    Kind,
    Leg,
    VerifyCtx,
    can_merge,
    merge_groups,
    merge_trajectories,
)


def _traj(rows, dt=1.0):
    t = JointTrajectory()
    t.joint_names = ["joint_1"]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(row)]
        p.velocities = [0.0]
        p.accelerations = [0.0]
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


def leg(name, chain=0, speed=0.25, guard=None, verify=None, kind=Kind.MOTION):
    return Leg(
        name=name,
        kind=kind,
        traj=_traj([0.0, 0.1]),
        speed=speed,
        guard=guard,
        world="full",
        chain=chain,
        target=("joints", [0.1]),
        goal_joints=[0.1],
        verify=verify,
    )


def test_same_chain_same_speed_merges():
    groups = merge_groups([leg("a"), leg("b")])
    assert [len(g) for g in groups] == [2]


def test_chain_break_splits():
    groups = merge_groups([leg("a", chain=0), leg("b", chain=1)])
    assert [len(g) for g in groups] == [1, 1]


def test_speed_change_splits():
    groups = merge_groups([leg("a", speed=0.25), leg("b", speed=0.15)])
    assert [len(g) for g in groups] == [1, 1]


def test_guarded_leg_always_alone():
    g = GuardSpec(touch_nm=3.0, trip="press")
    groups = merge_groups([leg("a"), leg("b", guard=g), leg("c")])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_verify_closes_group():
    def v(ctx):
        return True, ""

    groups = merge_groups([leg("a", verify=v), leg("b"), leg("c")])
    assert [len(g) for g in groups] == [1, 2]


def test_gripper_never_merges():
    gl = Leg(
        name="close",
        kind=Kind.GRIPPER,
        traj=None,
        speed=0.0,
        guard=None,
        world="full",
        chain=0,
        target=None,
        goal_joints=None,
        gripper_cmd=0.8,
    )
    groups = merge_groups([leg("a"), gl, leg("b")])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_can_merge_is_symmetric_gate():
    assert can_merge(leg("a"), leg("b"))
    assert not can_merge(leg("a", chain=0), leg("b", chain=1))


def test_merge_trajectories_offsets_time():
    merged = merge_trajectories([_traj([0.0, 0.1]), _traj([0.1, 0.2])])
    times = [
        p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in merged.points
    ]
    assert times == sorted(times)
    assert times[-1] == pytest.approx(2.0)  # 1 s + 1 s, offset applied
    assert merged.points[-1].positions[0] == pytest.approx(0.2)


def test_verify_ctx_defaults():
    ctx = VerifyCtx(outcome="arrived")
    assert ctx.depth_m is None and ctx.gripper_pos is None
