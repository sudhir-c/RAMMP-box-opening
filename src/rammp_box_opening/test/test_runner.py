from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind, Leg
from rammp_box_opening.runtime.runner import Runner


def _traj(start, end, dt=1.0):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for i, row in enumerate([start, end]):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


Q0 = [0.0] * 7
Q1 = [0.1] * 7
Q2 = [0.2] * 7


class FakeClient:
    def __init__(self):
        self.live = list(Q0)
        self.efforts = True
        self.exec_enabled = True
        self.executed = []  # (n_points, speed)
        self.exec_starts = []  # first waypoint of each executed trajectory
        self.worlds_pushed = []
        self.plans = []  # scripted plan_to_* responses (FIFO), else auto
        self.exec_script = []  # scripted execute outcomes (FIFO)
        self.tool_z = 0.08

    def joints(self):
        return list(self.live)

    def wrist_efforts(self):
        return [0.0] * 4 if self.efforts else None

    def efforts_present(self):
        return self.efforts

    def tool_xyz(self, timeout_s=1.5):
        return [0.45, 0.0, self.tool_z]

    def _plan(self, end, start):
        class R:
            success = True
            message = "ok"

        R.trajectory = _traj(start if start else self.live, end)
        return R

    def plan_to_pose(self, xyz, quat_xyzw, start_joints):
        if self.plans:
            return self.plans.pop(0)
        return self._plan(Q1, start_joints)

    def plan_to_joints(self, q7, start_joints):
        if self.plans:
            return self.plans.pop(0)
        return self._plan(list(q7), start_joints)

    def execute(self, traj, speed, guard=None):
        self.executed.append((len(traj.points), speed))
        self.exec_starts.append(list(traj.points[0].positions))
        if self.exec_script:
            outcome, info = self.exec_script.pop(0)
        else:
            outcome, info = (
                "arrived",
                {
                    "message": "ok",
                    "progress": 1.0,
                    "torque_peak": 0.0,
                },
            )
        if outcome != "failed":
            self.live = list(traj.points[-1].positions)
        return outcome, info

    def set_world(self, path_or_name):
        self.worlds_pushed.append(str(path_or_name))
        return True, "ok"

    def planner_execute_enabled(self):
        return self.exec_enabled

    def gripper_cmd(self, position):
        return True, float(position if position is not None else 0.0), False


class FakeStore:
    def push_name(self, kind, **kw):
        tag = kw.get("tag", "")
        name = kind + (("_" + tag) if tag else "")
        return name, name + ".yaml"


def leg(
    name,
    start=Q0,
    end=Q1,
    chain=0,
    speed=0.25,
    world="full",
    guard=None,
    kind=Kind.MOTION,
    verify=None,
    invalidates=False,
    cmd=None,
):
    return Leg(
        name=name,
        kind=kind,
        traj=_traj(start, end) if kind is Kind.MOTION else None,
        speed=speed,
        guard=guard,
        world=world,
        chain=chain,
        target=("joints", end) if kind is Kind.MOTION else None,
        goal_joints=end if kind is Kind.MOTION else None,
        invalidates_downstream=invalidates,
        verify=verify,
        gripper_cmd=cmd,
    )


def runner(client, tmp_path):
    return Runner(client, FakeStore(), log_dir=tmp_path)


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
        leg("descend", guard=g, world="interaction_x", invalidates=True, chain=0),
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
        touch_nm=3.0, trip="press", depth_window=(0.004, 0.012), target_z=0.09
    )
    c.exec_script = [
        ("touch", {"message": "contact", "progress": 0.6, "torque_peak": 4.2})
    ]

    def v(ctx):
        return press_outcome(ctx.outcome, ctx.depth_m, (0.004, 0.012))

    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b", verify=v, invalidates=True)],
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
    wandering = _traj(Q0, Q1)
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
    group, chain = r._replan_group([leg("a", invalidates=True)], next_chain=5)
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
        leg("descend", guard=g, world="interaction_x", invalidates=True, chain=0),
        leg("retreat", Q1, Q2, chain=0, speed=0.15, world="interaction_x"),
    ]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True, assume_yes=True)
    assert all(x.ok for x in res)
    # live == planned start (FakeClient tracks to traj end): NO replan —
    # the pre-planned retreat executed as built (no pause at the bottom)
    assert c.exec_starts[1] == Q1
