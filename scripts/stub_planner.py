"""Stand-in /rammp_curobo planner for stub-isolated e2e harnesses.

Serves the action/service surface our client uses, minus the physics:
plans are straight-line joint interpolations, execution publishes
interpolated /joint_states and honors cancel. Run ONLY on an isolated
ROS_DOMAIN_ID (the harnesses enforce this) — it serves the real names.

STUB_PLAN_S env shortens the fake plan duration (default 6 s) so flow
harnesses finish quickly while abort harnesses keep long strokes.
STUB_TRIP_EXEC_N=<n> spikes the published wrist efforts partway through
exec goal #n, so harnesses can exercise the torque-guard trip + cancel
path (the press's primary designed outcome).
"""

import os
import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_curobo_interfaces.action import (
    ExecuteTrajectory,
    PlanToJoints,
    PlanToPose,
)
from rammp_curobo_interfaces.srv import SetWorld

START = [0.4, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]  # off-home: real move
NAMES = ["joint_%d" % i for i in range(1, 8)]
PLAN_S = float(os.environ.get("STUB_PLAN_S", "6.0"))
TRIP_EXEC_N = int(os.environ.get("STUB_TRIP_EXEC_N", "0"))


def interp_traj(q0, q1, n=40, dur=PLAN_S):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for i in range(n):
        a = i / (n - 1)
        p = JointTrajectoryPoint()
        p.positions = [x + a * (y - x) for x, y in zip(q0, q1)]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        tt = a * dur
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int((tt - int(tt)) * 1e9)
        t.points.append(p)
    return t


class StubPlanner(Node):
    """The /rammp_curobo surface our client uses, minus the physics."""

    def __init__(self):
        super().__init__("rammp_curobo")
        self.declare_parameter("execute", True)
        cb = ReentrantCallbackGroup()
        self.q = list(START)
        self.plan_goals = 0
        self.exec_goals = 0
        self.spike = False  # STUB_TRIP_EXEC_N: efforts spike mid-goal
        # RELIABLE depth 10, like the real joint_state_broadcaster (the
        # client subscribes RELIABLE; sensor-data QoS would never connect)
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(0.05, self._tick, callback_group=cb)
        ActionServer(
            self,
            PlanToJoints,
            "/rammp_curobo/plan_to_joints",
            execute_callback=self._plan_joints,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=cb,
        )
        ActionServer(
            self,
            PlanToPose,
            "/rammp_curobo/plan_to_pose",
            execute_callback=self._plan_pose,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=cb,
        )
        ActionServer(
            self,
            ExecuteTrajectory,
            "/rammp_curobo/execute_trajectory",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=self._cancel,
            callback_group=cb,
        )
        ActionServer(
            self,
            GripperCommand,
            "/robotiq_gripper_controller/gripper_cmd",
            execute_callback=self._gripper,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            callback_group=cb,
        )
        self.create_service(
            SetWorld, "/rammp_curobo/set_world", self._set_world, callback_group=cb
        )
        print("STUB READY", flush=True)

    def _tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = NAMES + ["robotiq_85_left_knuckle_joint"]
        m.position = list(self.q) + [0.0]
        m.velocity = [0.0] * 8
        m.effort = [9.0 if self.spike else 0.0] * 8
        self.pub.publish(m)

    def _set_world(self, req, resp):
        print("SET_WORLD %s" % req.world, flush=True)
        resp.success = True
        resp.message = "stub"
        return resp

    def _plan_joints(self, gh):
        self.plan_goals += 1
        print("PLAN GOAL #%d (joints)" % self.plan_goals, flush=True)
        res = PlanToJoints.Result()
        res.success = True
        res.message = "stub plan"
        res.trajectory = interp_traj(
            list(gh.request.start_joints) or self.q, list(gh.request.target_joints)
        )
        res.planning_time = 0.01
        res.goal_mismatch_rad = 0.0
        gh.succeed()
        return res

    def _plan_pose(self, gh):
        self.plan_goals += 1
        print("PLAN GOAL #%d (pose)" % self.plan_goals, flush=True)
        q0 = list(gh.request.start_joints) or self.q
        q1 = list(q0)
        q1[0] += 0.3
        res = PlanToPose.Result()
        res.success = True
        res.message = "stub plan"
        res.trajectory = interp_traj(q0, q1)
        res.planning_time = 0.01
        gh.succeed()
        return res

    def _gripper(self, gh):
        pos = float(gh.request.command.position)
        print("GRIPPER GOAL pos=%.2f" % pos, flush=True)
        res = GripperCommand.Result()
        res.position = pos
        res.effort = 0.0
        res.stalled = False
        res.reached_goal = True
        gh.succeed()
        return res

    def _cancel(self, _goal):
        print("CANCEL RECEIVED", flush=True)
        return CancelResponse.ACCEPT

    def _execute(self, gh):
        self.exec_goals += 1
        traj = gh.request.trajectory
        speed = gh.request.speed_scale or 1.0
        pts = traj.points
        base = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
        dur = base / speed
        times = [
            (p.time_from_start.sec + p.time_from_start.nanosec * 1e-9) / speed
            for p in pts
        ]
        print(
            "EXEC GOAL ACCEPTED #%d: %d points over %.1f s"
            % (self.exec_goals, len(pts), dur),
            flush=True,
        )
        t0 = time.monotonic()
        k = 0
        goal_n = self.exec_goals
        while time.monotonic() - t0 < dur:
            el = time.monotonic() - t0
            while k < len(times) - 1 and times[k] < el:
                k += 1
            self.q = list(pts[k].positions)
            if goal_n == TRIP_EXEC_N and el / dur > 0.4 and not self.spike:
                print("EFFORT SPIKE injected (goal #%d)" % goal_n, flush=True)
                self.spike = True
            fb = ExecuteTrajectory.Feedback()
            fb.progress = float(el / dur)
            gh.publish_feedback(fb)
            if gh.is_cancel_requested:
                self.spike = False
                print("STOPPED at %.1f of %.1f s" % (el, dur), flush=True)
                gh.canceled()
                res = ExecuteTrajectory.Result()
                res.success = False
                res.message = "cancelled — controller stops and holds"
                return res
            time.sleep(0.05)
        self.spike = False
        self.q = list(pts[-1].positions)
        print("RAN TO COMPLETION", flush=True)
        gh.succeed()
        res = ExecuteTrajectory.Result()
        res.success = True
        res.message = "arrived"
        return res


def main():
    rclpy.init()
    ex = MultiThreadedExecutor()
    ex.add_node(StubPlanner())
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
