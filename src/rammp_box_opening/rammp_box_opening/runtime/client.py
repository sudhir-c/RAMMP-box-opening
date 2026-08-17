"""The single ROS surface: planner action clients + gripper + TF + params.

Every pattern here is lifted from proven RAMMP-CuRobo clients:
spin_until_done/cancel-on-Ctrl+C from tour_demo.py, the guarded execute
loop from the recovered palm_demo.py — with the spec §6 hardening: the
guard is ARMED by ExecuteTrajectory feedback progress > 0 (never at
goal-accept) and efforts come from the /joint_states stream.
"""

import sys
import time

import rclpy
from control_msgs.action import GripperCommand
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from rammp_curobo_interfaces.action import (
    ExecuteTrajectory,
    PlanToJoints,
    PlanToPose,
)
from rammp_curobo_interfaces.srv import SetWorld

from rammp_box_opening.constants import GRIPPER_ACTION, JOINTS, NODE_NAMESPACE

_GRIPPER_JOINT_HINTS = ("robotiq", "knuckle", "finger")


def spin_until_done(node, future, timeout_s):
    """Spin `node` until `future` resolves; None on timeout."""
    t0 = time.monotonic()
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


class PlannerClient:
    def __init__(self, node):
        self.node = node
        self._q = None
        self._eff = None
        self._gripper_pos = None
        node.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self._plan_pose = ActionClient(node, PlanToPose, NODE_NAMESPACE + "/plan_to_pose")
        self._plan_joints = ActionClient(
            node, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints"
        )
        self._execute = ActionClient(
            node, ExecuteTrajectory, NODE_NAMESPACE + "/execute_trajectory"
        )
        self._gripper = ActionClient(node, GripperCommand, GRIPPER_ACTION)
        self._set_world = node.create_client(SetWorld, NODE_NAMESPACE + "/set_world")
        self._params = node.create_client(
            GetParameters, NODE_NAMESPACE + "/get_parameters"
        )
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)

    # -- state streams -----------------------------------------------------
    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [float(msg.position[idx[n]]) for n in JOINTS]
            eff = (
                [float(msg.effort[idx[n]]) for n in JOINTS]
                if len(msg.effort) == len(msg.name)
                else None
            )
        except (KeyError, IndexError):
            return
        self._q = q
        self._eff = eff
        for name in msg.name:
            if any(h in name for h in _GRIPPER_JOINT_HINTS):
                self._gripper_pos = float(msg.position[idx[name]])
                break

    def joints(self):
        t0 = time.monotonic()
        while self._q is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > 10:
                sys.exit(
                    "no /joint_states — start the arm bringup (RAMMP-Kinova "
                    "workspace) and the planner first:\n"
                    "  ros2 launch rammp_curobo_ros planner.launch.py "
                    "config:=gen3_real.yaml"
                )
        return list(self._q)

    def wrist_efforts(self):
        return None if self._eff is None else list(self._eff[3:])

    def efforts_present(self):
        self.joints()  # ensure at least one message arrived
        return self._eff is not None

    def tool_xyz(self, timeout_s=1.5):
        """Live base_link -> tool_frame translation via TF, or None."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                tf = self._tf.lookup_transform("base_link", "tool_frame", Time())
                tr = tf.transform.translation
                return [tr.x, tr.y, tr.z]
            except Exception:
                rclpy.spin_once(self.node, timeout_sec=0.1)
        return None

    # -- planning ----------------------------------------------------------
    def _call(self, client, goal, timeout_s=120.0):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner node not running")
        send = spin_until_done(self.node, client.send_goal_async(goal), 10.0)
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(self.node, send.get_result_async(), timeout_s)
        return None if wrapped is None else wrapped.result

    def plan_to_pose(self, xyz, quat_xyzw, start_joints):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = (
            float(v) for v in xyz
        )
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = (float(v) for v in quat_xyzw)
        g.start_joints = [float(v) for v in start_joints] if start_joints else []
        return self._call(self._plan_pose, g)

    def plan_to_joints(self, q7, start_joints):
        g = PlanToJoints.Goal(target_joints=[float(v) for v in q7])
        g.start_joints = [float(v) for v in start_joints] if start_joints else []
        return self._call(self._plan_joints, g)

    # -- execution ---------------------------------------------------------
    def execute(self, traj, speed, guard=None):
        """Run one trajectory; outcome 'arrived' | 'touch' | 'failed'.

        With a guard: feedback progress arms it, /joint_states efforts
        feed it; a trip cancels the goal (controller stops and holds)."""
        info = {"message": "", "progress": 0.0, "torque_peak": None}
        goal = ExecuteTrajectory.Goal(trajectory=traj, speed_scale=float(speed))
        if not self._execute.wait_for_server(timeout_sec=5.0):
            info["message"] = "execute_trajectory server not available"
            return "failed", info

        def _fb(msg):
            info["progress"] = float(msg.feedback.progress)
            if guard is not None:
                guard.on_progress(info["progress"])

        send = spin_until_done(
            self.node, self._execute.send_goal_async(goal, feedback_callback=_fb), 10.0
        )
        if send is None or not send.accepted:
            info["message"] = "goal not accepted"
            return "failed", info
        result_future = send.get_result_async()
        contact = False
        t0 = time.monotonic()
        try:
            while not result_future.done():
                rclpy.spin_once(self.node, timeout_sec=0.05)
                if guard is not None and guard.on_efforts(self.wrist_efforts()):
                    contact = True
                    spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                    spin_until_done(self.node, result_future, 10.0)
                    break
                if time.monotonic() - t0 > 240:
                    spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                    info["message"] = "execution watchdog timeout (240 s)"
                    return "failed", info
        except KeyboardInterrupt:
            spin_until_done(self.node, send.cancel_goal_async(), 3.0)
            print("\nCtrl+C — goal cancelled, arm holds")
            raise
        if guard is not None:
            info["torque_peak"] = guard.peak
        if contact:
            info["message"] = "torque guard trip"
            return "touch", info
        wrapped = result_future.result()
        if wrapped is not None and wrapped.result.success:
            info["message"] = wrapped.result.message
            return "arrived", info
        info["message"] = wrapped.result.message if wrapped is not None else "no result"
        return "failed", info

    # -- services / gripper ------------------------------------------------
    def set_world(self, path_or_name):
        if not self._set_world.wait_for_service(timeout_sec=5.0):
            return False, "set_world service unavailable"
        req = SetWorld.Request(world=str(path_or_name))
        resp = spin_until_done(self.node, self._set_world.call_async(req), 10.0)
        if resp is None:
            return False, "set_world timed out"
        return resp.success, resp.message

    def planner_execute_enabled(self):
        """Read the planner's LIVE execute parameter; False if unreachable
        (fail closed — the gripper action itself is not server-gated)."""
        if not self._params.wait_for_service(timeout_sec=3.0):
            return False
        req = GetParameters.Request(names=["execute"])
        resp = spin_until_done(self.node, self._params.call_async(req), 5.0)
        if resp is None or not resp.values:
            return False
        return bool(resp.values[0].bool_value)

    def gripper_cmd(self, position):
        """Command the gripper (0.0 open … 0.8 closed); position None =
        query only (live joint-state position, no motion)."""
        if position is None:
            return self._gripper_pos is not None, self._gripper_pos or 0.0, False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 100.0
        if not self._gripper.wait_for_server(timeout_sec=2.0):
            return False, 0.0, False
        send = spin_until_done(self.node, self._gripper.send_goal_async(goal), 5.0)
        if send is None or not send.accepted:
            return False, 0.0, False
        wrapped = spin_until_done(self.node, send.get_result_async(), 10.0)
        if wrapped is None:
            return False, 0.0, False
        return True, float(wrapped.result.position), bool(wrapped.result.stalled)
