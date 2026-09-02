from trajectory_msgs.msg import JointTrajectoryPoint

from conftest import Q0, Q1, Q2, FakeClient, leg, runner, traj

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind


def test_dry_run_executes_nothing(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run([leg("a")], execute=False)
    assert [r.outcome for r in res] == ["skipped"]
    assert c.executed == [] and c.worlds_pushed == []


def test_merged_group_is_one_execution(tmp_path):
    c = FakeClient()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1  # merged: one goal


def test_unguarded_leg_requires_full_world(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("a", world="interaction_button")], execute=True, assume_yes=True
    )
    assert res[0].outcome == "refused" and c.executed == []


def test_gripper_gate_reads_planner_param(tmp_path):
    c = FakeClient()
    c.exec_enabled = False
    gl = leg("close", kind=Kind.GRIPPER, cmd=0.8)
    res = runner(c, tmp_path).run([gl], execute=True, assume_yes=True)
    assert res[0].outcome == "refused"
    assert "execute" in res[0].detail


def test_guarded_leg_refused_without_efforts(tmp_path):
    c = FakeClient()
    c.efforts = False
    g = GuardSpec(
        touch_nm=3.0, trip="press", depth_window=(0.004, 0.012), target_z=0.09
    )
    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b")], execute=True, assume_yes=True
    )
    assert res[0].outcome == "refused" and c.executed == []


def test_start_drift_triggers_replan(tmp_path):
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6  # 0.06 > 0.04 threshold
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1)], execute=True, assume_yes=True)
    assert res[0].ok
    # re-planned from live: the executed trajectory starts at the live joints
    assert c.exec_starts and c.exec_starts[0][0] == 0.06


def test_no_motion_retry_once(tmp_path):
    c = FakeClient()
    c.exec_script = [
        (
            "failed",
            {
                "message": "goal aborted: arm never left the start",
                "progress": 0.0,
                "torque_peak": None,
            },
        ),
        ("arrived", {"message": "ok", "progress": 1.0, "torque_peak": None}),
    ]
    res = runner(c, tmp_path).run([leg("a")], execute=True, assume_yes=True)
    assert res[0].ok and len(c.executed) == 2


def test_other_failure_stops_without_retry(tmp_path):
    c = FakeClient()
    c.exec_script = [
        (
            "failed",
            {"message": "controller rejected", "progress": 0.4, "torque_peak": None},
        )
    ]
    res = runner(c, tmp_path).run(
        [leg("a"), leg("b", Q1, Q2)], execute=True, assume_yes=True
    )
    assert res[0].outcome == "failed"
    assert len(res) == 1 and len(c.executed) == 1  # stopped, b never ran


def test_contact_invalidates_downstream(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.5, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("after", Q1, Q2, chain=0),
    ]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert res[0].outcome == "touch" and res[0].ok  # setdown: trip = success
    assert res[1].ok
    assert len(c.executed) == 2  # never merged with contact


def test_press_verify_uses_depth(tmp_path):
    from rammp_box_opening.runtime.guards import press_outcome

    c = FakeClient()
    c.tool_z = 0.082  # 8 mm below target_z 0.09
    g = GuardSpec(
        touch_nm=3.0,
        trip="press",
        depth_window=(0.004, 0.012),
        target_z=0.09,
        needs_depth=True,  # Descend-style press: its verify reads depth_m
    )
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.6, "torque_peak": 4.2})
    ]

    def v(ctx):
        return press_outcome(ctx.outcome, ctx.depth_m, (0.004, 0.012))

    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b", verify=v)],
        execute=True,
        assume_yes=True,
    )
    assert res[0].ok and "pressed" in res[0].detail


def test_jsonl_log_written(tmp_path):
    import json

    c = FakeClient()
    runner(c, tmp_path).run([leg("a")], execute=True, assume_yes=True)
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    row = json.loads(logs[0].read_text().splitlines()[0])
    assert row["leg"] == "a" and row["outcome"] == "arrived"


def test_transit_gate_accepts_bench_world_pre_detection(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    assert r._refusal(leg("scan", world="bench")) is None
    assert "full or bench" in r._refusal(leg("weird", world="interaction_button"))


def test_replanned_trajectories_pass_the_sanity_gate(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    wandering = traj(Q0, Q1)
    mid = JointTrajectoryPoint()
    mid.positions = [1.5] + [0.05] * 6  # joint_1 wanders way out and back
    mid.time_from_start.sec = 1
    wandering.points.insert(1, mid)
    wandering.points[-1].time_from_start.sec = 2

    class R:
        success = True
        message = "ok"
        trajectory = wandering

    c.plans = [R]
    group, chain = r._replan_group([leg("a")], next_chain=5)
    assert group is None  # wandering replan refused, same gate as pre-built


def test_fast_retreat_allowed_in_interaction_world(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    ok_leg = leg("retreat", speed=0.35, world="interaction_button")
    assert r._refusal(ok_leg) is None  # ascends its own corridor
    other = leg("wander", speed=0.35, world="interaction_button")
    assert r._refusal(other) is not None  # only retreats get the pass


def test_no_replan_when_contact_left_arm_on_plan(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.97, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=0, speed=0.15, world="interaction_x"),
    ]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True, assume_yes=True)
    assert all(x.ok for x in res)
    # live == planned start (FakeClient tracks to traj end): NO replan —
    # the pre-planned retreat executed as built (no pause at the bottom)
    assert c.exec_starts[1] == Q1


def test_tf_depth_lookup_skipped_unless_the_verify_needs_it(tmp_path):
    """PressFixed judges by progress + torque, never by depth.

    The lookup blocks for its whole timeout on this TF tree (no
    tool_frame exists), so a guard that does not consume depth must not
    trigger it — 0.5 s per press trip, measured 2026-08-28."""
    c = FakeClient()
    calls = []
    inner = c.tool_xyz
    c.tool_xyz = lambda *a, **k: (calls.append(1), inner(*a, **k))[1]

    g = GuardSpec(touch_nm=3.0, trip="press", target_z=0.09)  # needs_depth False
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b")],
        execute=True,
        assume_yes=True,
    )
    assert res[0].outcome == "touch"
    assert calls == [], "no tool_frame lookup when the verify ignores depth"


def test_lift_may_run_faster_than_contact_in_an_interaction_world(tmp_path):
    """Lift ascends out of the corridor it just descended, exactly like
    retreat — both are exempt from the transit-speed world gate."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("lift", speed=0.35, world="interaction_button")],
        execute=True,
        assume_yes=True,
    )
    assert res[0].ok and res[0].outcome != "refused"


def test_unguarded_fast_leg_in_an_interaction_world_is_still_refused(tmp_path):
    """The exemption is name-scoped — it must not open the gate widely."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("place:lid:transit", speed=0.35, world="interaction_place")],
        execute=True,
        assume_yes=True,
    )
    assert res[0].outcome == "refused"


def test_merged_group_takes_its_guard_from_the_guarded_member(tmp_path):
    """can_merge forbids this today; the lookup is what keeps a future
    relaxation from running a guarded stroke with the lead's (absent)
    guard, the lead's speed and the lead's verify."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    group = [leg("fast", Q0, Q1, chain=0), leg("descend", Q1, Q2, chain=0, guard=g)]
    r = runner(c, tmp_path)
    res = r._run_motion(group)
    # setdown semantics come from the guarded member: a trip is SUCCESS
    assert res.outcome == "touch" and res.ok


def test_merged_group_refuses_a_guard_that_is_not_last(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    group = [leg("descend", Q0, Q1, chain=0, guard=g), leg("after", Q1, Q2, chain=0)]
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="not last"):
        runner(c, tmp_path)._run_motion(group)


def test_replans_push_the_legs_own_world(tmp_path):
    """Worlds are pushed at plan time and by the replan path per leg —
    execution never consults the collision world, so the runner no longer
    pushes per group (one tracker lives in the client, audit 2026-09-02).
    A replanned leg must reach the planner with ITS world (content-hashed
    path), not whatever was loaded."""
    c = FakeClient()
    a = leg("a", Q0, Q1, world="full", chain=0)
    a.world_path = "/w/full-aaaa.yaml"
    b = leg("b", Q1, Q2, world="full", chain=1)
    b.world_path = "/w/full-bbbb.yaml"  # same name, new contents
    c.live = [0.06] + [0.0] * 6  # drift: a replans, then b chains clean
    runner(c, tmp_path).run([a, b], execute=True, assume_yes=True)
    assert c.worlds_pushed == ["/w/full-aaaa.yaml"]


def test_lazy_leg_is_planned_once_from_live_at_execution(tmp_path):
    """A lazy leg (traj None) after a touch is planned from live joints
    exactly when it executes, in its own world."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})
    ]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.goal_joints = None
    lazy.world_path = "/w/interaction_x-1.yaml"
    legs = [leg("down", guard=g, world="interaction_x", chain=0), lazy]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert [r.outcome for r in res] == ["touch", "arrived"]
    assert c.worlds_pushed == ["/w/interaction_x-1.yaml"]
    assert len(c.executed) == 2


class _AsyncGripClient(FakeClient):
    """FakeClient that records send/join ordering against motion."""

    def __init__(self):
        super().__init__()
        self.events = []

    def gripper_send(self, position):
        self.events.append("send")
        return ("handle", position)

    def gripper_join(self, handle):
        self.events.append("join")
        return True, float(handle[1]), False

    def gripper_cmd(self, position):
        # the real client's blocking path is send + join; mirror it so the
        # ordering assertions mean the same thing on both paths
        if position is None:
            return super().gripper_cmd(position)
        return self.gripper_join(self.gripper_send(position))

    def execute(self, traj, speed, guard=None, while_running=None):
        self.events.append("execute")
        return super().execute(traj, speed, guard=guard, while_running=while_running)


def test_deferred_gripper_close_overlaps_the_next_transit(tmp_path):
    """The owner's own example: fingers shut WHILE the arm moves."""
    c = _AsyncGripClient()
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    legs = [close, leg("approach", Q0, Q1, chain=0)]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True, assume_yes=True)
    assert all(r_.ok for r_ in res)
    # sent, THEN the transit ran; the join is lazy — it outlives the run
    # to overlap whatever planning follows, and finish() collects it
    assert c.events == ["send", "execute"]
    assert r.finish().ok
    assert c.events == ["send", "execute", "join"]


def test_deferred_gripper_is_joined_before_any_guarded_leg(tmp_path):
    """A press descends with the fingers closed — the overlap must not
    let a guarded leg start while they are still moving."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="press", target_z=0.09)
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    legs = [close, leg("press:down", Q0, Q1, chain=0, guard=g, world="interaction_b")]
    runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert c.events.index("join") < c.events.index("execute")


def test_a_release_is_never_deferred(tmp_path):
    """defer_join is opt-in; an un-flagged gripper leg still blocks."""
    c = _AsyncGripClient()
    legs = [leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0), leg("retreat", Q0, Q1)]
    runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert c.events[:2] == ["send", "join"], "release settles before the arm moves"


def test_touch_forces_replan_even_under_the_drift_gate(tmp_path):
    """After a guard trip the predicted start is wrong BY DESIGN, and a
    sub-gate joint delta is still a multi-mm Cartesian shove into the
    thing just touched: the pre-planned retreat re-pressed the button at
    full speed, guardless (field 2026-09-02). Post-touch, the next
    motion replans from live unconditionally."""
    class TouchStopsShort(FakeClient):
        # a real trip halts the arm shy of the endpoint; the plain fake
        # teleports to it, which would hide exactly the hazard under test
        def execute(self, traj, speed, guard=None, while_running=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "touch":
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = TouchStopsShort()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})
    ]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=1),
    ]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert res[0].outcome == "touch" and res[1].ok
    # the trip left the arm 0.01 rad from the predicted end — well under
    # the 0.04 free-air drift gate, which must NOT matter after a touch
    assert c.exec_starts[1][0] == 0.11  # replanned from live, not Q1


def test_arrived_leg_still_uses_the_drift_gate(tmp_path):
    """Free-air chaining keeps the owner's fewer-pauses rule: no touch,
    sub-gate drift, the pre-planned leg runs as planned. (The drift must
    appear AFTER leg a runs — the fake teleports live to each executed
    endpoint, so a pre-set offset only ever tests leg a's own gate.)"""

    class DriftsAfterArrive(FakeClient):
        def execute(self, traj, speed, guard=None, while_running=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "arrived" and len(self.executed) == 1:
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = DriftsAfterArrive()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert all(r.ok for r in res)
    # 0.01 under the 0.04 gate, no touch: pre-planned start kept — an
    # always-replan mutation would execute from live (0.11) instead
    assert c.exec_starts[1][0] == 0.1


def test_replanned_warped_leg_keeps_its_execution_profile(tmp_path):
    """_apply_warp bakes fast-then-slow into the trajectory and sets
    speed=1.0 as a do-not-dilate sentinel. A replan swaps in a fresh
    UNWARPED trajectory — inheriting the sentinel would descend into
    contact at full speed (review 2026-09-02). The profile is re-warped,
    or the leg honestly downgrades to the slow contact speed."""
    from rammp_box_opening.runtime.runner import _restore_execution_profile

    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0, rebaseline_after=0.7)

    # a trajectory long enough to warp: profile re-applied, sentinel kept
    many = [[0.0 + 0.01 * i] * 7 for i in range(40)]
    t = traj(many[0], many[-1])
    t.points = []
    from trajectory_msgs.msg import JointTrajectoryPoint

    for i, row in enumerate(many):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = i
        t.points.append(pt)
    lg = leg("down", guard=g, world="interaction_x")
    lg.speed = 1.0
    lg.warp = (0.5, 0.35, 0.3)
    lg.traj = t
    _restore_execution_profile(lg)
    assert lg.speed == 1.0  # profile baked in again
    assert lg.guard.rebaseline_after is not None
    end = lg.traj.points[-1].time_from_start
    assert end.sec + end.nanosec * 1e-9 > 39.0  # slower than the raw plan

    # a degenerate trajectory that cannot be warped: honest downgrade
    lg2 = leg("down2", guard=g, world="interaction_x")
    lg2.speed = 1.0
    lg2.warp = (0.35, 0.35, 0.3)  # fast==slow -> warp declines
    _restore_execution_profile(lg2)
    assert lg2.speed == 0.35  # the slow contact speed, never the sentinel
    assert lg2.guard.rebaseline_after is None

    # the retime hook follows the trajectory actually flown
    seen = {}
    lg3 = leg("press", guard=g, world="interaction_x")
    lg3.retime = lambda traj: seen.__setitem__("traj", traj)
    _restore_execution_profile(lg3)
    assert seen["traj"] is lg3.traj


def test_guard_observes_but_cannot_trip_before_arm_after():
    """The fast warp segment's dynamics tripped the gentle set-down
    threshold 64 ms into the descent and the lid was released 110 mm up
    (field 2026-09-02). Before arm_after the guard watches but never
    trips; after it, the same deviation trips."""
    from rammp_box_opening.runtime.guards import TorqueGuard

    g = TorqueGuard(4.0, arm_after=0.5)
    g.on_progress(0.05)
    assert g.on_efforts([0.0] * 4) is False  # baseline
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is False  # fast-zone jolt held
    assert g.peak == 9.0  # still observed
    g.on_progress(0.6)
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is True  # same dev now trips


def test_pending_gripper_outlives_the_run_and_joins_before_a_guarded_leg(tmp_path):
    """grip:open is dispatched on arrival at the hop (last leg of the press
    phase) and must overlap the NEXT phase's planning — so a run() ends
    without joining it, and the following run joins it before its guarded
    descent (audit 2026-09-02)."""
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0)
    open_leg.defer_join = True
    r.run([leg("retreat", Q0, Q1), open_leg], execute=True, assume_yes=True)
    assert c.events == ["execute", "send"]  # NOT joined at the end of run()
    g = GuardSpec(touch_nm=3.0, trip="obstruction", target_z=0.0)
    r.run([leg("grip:down", Q1, Q2, guard=g, world="interaction_b")], execute=True, assume_yes=True)
    assert c.events == ["execute", "send", "join", "execute"]


def test_start_gripper_dispatches_now_and_joins_lazily(tmp_path):
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    assert r.start_gripper("press:close", 0.8, execute=True)
    assert c.events == ["send"]
    g = GuardSpec(touch_nm=3.0, trip="press", target_z=0.09)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("press:down", Q0, Q1, guard=g, world="interaction_b")], execute=True, assume_yes=True)
    assert c.events == ["send", "join", "execute"]
    assert r.finish() is None  # nothing left pending


def test_release_overlaps_the_replan_but_never_the_motion(tmp_path):
    """place:lid:open is dispatched at once; the retreat's post-touch replan
    proceeds while the fingers open; the join lands before the retreat
    EXECUTES (a release completes before the arm moves away)."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    release = leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0)
    release.defer_join = True
    release.join_before_motion = True

    class Spy(_AsyncGripClient):
        def plan_to_pose(self, *a, **k):
            self.events.append("plan")
            return super().plan_to_pose(*a, **k)

    c = Spy()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.target = ("pose", [0.5, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0])
    legs = [
        leg("down", guard=g, world="interaction_x", chain=0),
        release,
        lazy,
    ]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert all(r.ok for r in res)
    assert c.events == ["execute", "send", "plan", "join", "execute"]


def test_lookahead_runs_during_the_last_unguarded_motion(tmp_path):
    """The next phase is planned while this run's last unguarded motion
    flies, from that motion's predicted end joints; a guarded stroke never
    hosts it (audit 2026-09-02)."""
    c = FakeClient()
    r = runner(c, tmp_path)
    seen = []
    res = r.run(
        [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)],
        execute=True,
        assume_yes=True,
        lookahead=lambda q: seen.append(list(q)) or ["next-legs"],
    )
    assert all(x.ok for x in res)
    assert seen == [Q2]  # hosted on b, from b's predicted end
    assert r.lookahead_result == ["next-legs"]

    # a failing lookahead is not a failed leg: the caller builds afterwards
    def boom(q):
        raise RuntimeError("planner said no")

    res = r.run([leg("c", Q2, Q1, chain=0)], execute=True, assume_yes=True, lookahead=boom)
    assert res[0].ok and r.lookahead_result is None

    # guarded strokes never host it
    g = GuardSpec(touch_nm=3.0, trip="press", target_z=0.09)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    seen.clear()
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b")], execute=True, assume_yes=True, lookahead=lambda q: seen.append(q))
    assert seen == [] and r.lookahead_result is None
