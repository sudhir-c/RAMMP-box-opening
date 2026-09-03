"""Shared test doubles and fixtures.

FakeClient stands in for runtime.client.PlannerClient: plans are
two-point straight lines from the given start, executions teleport the
live joints to the goal unless scripted, worlds are recorded. FakeStore
stands in for worlds.WorldStore: records every push and returns the
world's name as its path. `leg` and `runner` build Runner inputs; the
`ctx` fixture is a mission context around the shipped container config
with a detected pose on the bench.
"""

import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.legs import Kind, Leg
from rammp_box_opening.runtime.runner import Runner

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"

Q0 = [0.0] * 7
Q1 = [0.1] * 7
Q2 = [0.2] * 7


def traj(start, end, dt=1.0):
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
        self.approach_offsets = []  # per plan_to_pose call
        self.joint_starts = []  # per plan_to_joints call

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

        R.trajectory = traj(start if start else self.live, end)
        return R

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        self.approach_offsets.append(float(approach_offset_m))
        if self.plans:
            return self.plans.pop(0)
        return self._plan(Q1, start_joints)

    def plan_to_joints(self, q7, start_joints):
        self.joint_starts.append(list(start_joints))
        if self.plans:
            return self.plans.pop(0)
        return self._plan(list(q7), start_joints)

    def execute(self, traj_, speed, guard=None, while_running=None):
        self.executed.append((len(traj_.points), speed))
        self.exec_starts.append(list(traj_.points[0].positions))
        info_extra = {}
        if while_running is not None and guard is None:
            try:
                info_extra["while_running"] = while_running()
            except Exception as exc:
                info_extra["while_running_error"] = str(exc)
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
            self.live = list(traj_.points[-1].positions)
        info = dict(info)
        info.update(info_extra)
        return outcome, info

    def set_world(self, path_or_name):
        self.worlds_pushed.append(str(path_or_name))
        return True, "ok"

    def planner_execute_enabled(self):
        return self.exec_enabled

    def gripper_send(self, position):
        return ("handle", float(position))

    def gripper_join(self, handle):
        return True, float(handle[1]), False

    def gripper_cmd(self, position):
        return True, float(position if position is not None else 0.0), False


class FakeStore:
    def __init__(self):
        self.pushes = []  # (kind, kwargs) — world-shape assertions

    def push_name(self, kind, **kw):
        self.pushes.append((kind, kw))
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
    cmd=None,
):
    return Leg(
        name=name,
        kind=kind,
        traj=traj(start, end) if kind is Kind.MOTION else None,
        speed=speed,
        guard=guard,
        world=world,
        chain=chain,
        target=("joints", end) if kind is Kind.MOTION else None,
        goal_joints=end if kind is Kind.MOTION else None,
        verify=verify,
        gripper_cmd=cmd,
    )


def runner(client, tmp_path):
    r = Runner(client, FakeStore(), log_dir=tmp_path)
    r.no_motion_retry_delay_s = 0.0  # production waits 3 s; tests must not
    return r


@pytest.fixture
def ctx():
    """A mission context: the shipped container, detected on the bench."""
    return Ctx(
        model=ContainerModel.load(CFG),
        cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
        client=FakeClient(),
        worlds=FakeStore(),
        config_path=CFG,
    )
