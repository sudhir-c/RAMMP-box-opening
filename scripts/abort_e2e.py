#!/usr/bin/env python3
"""Does Ctrl+C in OUR CLIs actually cancel the in-flight goal? Stub-proven.

    python3 scripts/abort_e2e.py            # isolates itself on ROS_DOMAIN_ID=77

RAMMP-CuRobo's abort_checks.py proves the PLANNER's Ctrl+C stops the
controller. This proves OUR side of the same chain: a rammp_box_opening
CLI (`home_arm --execute`) driving a STUB planner node — SIGINT mid-stroke
must deliver an ExecuteTrajectory CANCEL to the server before the process
exits. The failure this catches is silent and severe: with rclpy's
default SIGINT handler the context dies before the cancel can be sent —
the CLI stops WATCHING the arm while the trajectory runs to its end
(verified on this stack by abort_checks; field lesson 7).

No arm, no GPU, no real planner. The harness refuses to run on a graph
with a real controller_manager or planner (the stub serves the real
/rammp_curobo names — on a shared graph, discovery binding is a coin
flip), and audits goal counts (CLI-sent == stub-received) per lesson 6.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOMAIN = os.environ.get("ABORT_E2E_DOMAIN", "77")

CHAIN = (
    "export ROS_DOMAIN_ID=%s; export ROS_LOCALHOST_ONLY=1; "
    "source /opt/ros/humble/setup.zsh; "
    "source ~/RAMMP-CuRobo/install/setup.zsh; "
    "source %s/install/setup.zsh; " % (DOMAIN, REPO)
)

STUB = '''
import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from rammp_curobo_interfaces.action import ExecuteTrajectory, PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
START = [0.4, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]   # off-home: real move
NAMES = ["joint_%d" % i for i in range(1, 8)]
PLAN_S = 6.0


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
        # RELIABLE depth 10, like the real joint_state_broadcaster (the
        # client subscribes RELIABLE; sensor-data QoS would never connect)
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(0.05, self._tick, callback_group=cb)
        ActionServer(
            self, PlanToJoints, "/rammp_curobo/plan_to_joints",
            execute_callback=self._plan_joints,
            goal_callback=lambda _g: GoalResponse.ACCEPT, callback_group=cb,
        )
        ActionServer(
            self, PlanToPose, "/rammp_curobo/plan_to_pose",
            execute_callback=self._plan_pose,
            goal_callback=lambda _g: GoalResponse.ACCEPT, callback_group=cb,
        )
        ActionServer(
            self, ExecuteTrajectory, "/rammp_curobo/execute_trajectory",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=self._cancel, callback_group=cb,
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
        m.effort = [0.0] * 8
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
        res.trajectory = interp_traj(list(gh.request.start_joints) or self.q,
                                     list(gh.request.target_joints))
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
        while time.monotonic() - t0 < dur:
            el = time.monotonic() - t0
            while k < len(times) - 1 and times[k] < el:
                k += 1
            self.q = list(pts[k].positions)
            fb = ExecuteTrajectory.Feedback()
            fb.progress = float(el / dur)
            gh.publish_feedback(fb)
            if gh.is_cancel_requested:
                print("STOPPED at %.1f of %.1f s" % (el, dur), flush=True)
                gh.canceled()
                res = ExecuteTrajectory.Result()
                res.success = False
                res.message = "cancelled — controller stops and holds"
                return res
            time.sleep(0.05)
        self.q = list(pts[-1].positions)
        print("RAN TO COMPLETION", flush=True)
        gh.succeed()
        res = ExecuteTrajectory.Result()
        res.success = True
        res.message = "arrived"
        return res


rclpy.init()
ex = MultiThreadedExecutor()
ex.add_node(StubPlanner())
try:
    ex.spin()
except KeyboardInterrupt:
    pass
'''


def sh(cmd, **kw):
    return subprocess.run(
        ["zsh", "-c", CHAIN + cmd], capture_output=True, text=True, **kw
    )


def kill(proc):
    if proc is None:
        return
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def wait_for(path, needle, timeout, proc=None, what="", also_dump=()):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if needle in Path(path).read_text():
            return True
        if proc is not None and proc.poll() is not None:
            dumps = "".join(
                "\n--- %s ---\n%s" % (p.name, Path(p).read_text()[-2000:])
                for p in (path, *also_dump)
            )
            sys.exit("%s died before %r:%s" % (what, needle, dumps))
        time.sleep(0.2)
    return False


def main():
    tmp = Path(tempfile.mkdtemp(prefix="abort_e2e_"))
    print("workdir %s (domain %s)" % (tmp, DOMAIN))

    sh("ros2 daemon stop", timeout=30)  # a daemon bound to another domain lies
    probe = sh("timeout 20 ros2 node list", timeout=30)
    nodes = probe.stdout
    if "/controller_manager" in nodes or "/rammp_curobo" in nodes:
        sys.exit(
            "REAL arm stack or planner visible on ROS_DOMAIN_ID=%s:\n%s\n"
            "This harness serves fake /rammp_curobo names — refusing "
            "(discovery binding on a shared graph is a coin flip)." % (DOMAIN, nodes)
        )

    stub_py = tmp / "stub.py"
    stub_py.write_text(STUB)
    stub_log = tmp / "stub.log"
    cli_log = tmp / "cli.log"

    # measured-config copy: --execute refuses while measure_me is true
    cfg = tmp / "oxo_measured.yaml"
    src_cfg = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"
    cfg.write_text(src_cfg.read_text().replace("measure_me: true", "measure_me: false"))

    stub = cli = None
    try:
        stub = subprocess.Popen(
            ["zsh", "-c", CHAIN + "exec python3 %s" % stub_py],
            stdout=open(stub_log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if not wait_for(stub_log, "STUB READY", 30, stub, "stub"):
            sys.exit("stub never became ready:\n" + stub_log.read_text()[-2000:])

        cli = subprocess.Popen(
            [
                "zsh",
                "-c",
                CHAIN
                + "exec ros2 run rammp_box_opening home_arm --execute --container %s"
                % cfg,
            ],
            stdin=subprocess.PIPE,
            stdout=open(cli_log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        cli.stdin.write("yes\n")
        cli.stdin.flush()

        if not wait_for(
            stub_log, "EXEC GOAL ACCEPTED", 60, cli, "CLI", also_dump=(cli_log,)
        ):
            sys.exit(
                "no execution goal reached the stub:\n--- cli ---\n%s\n--- stub ---\n%s"
                % (cli_log.read_text()[-2000:], stub_log.read_text()[-2000:])
            )
        time.sleep(2.0)  # mid-stroke (stub stroke is ~24 s at speed 0.25)

        print("SIGINT to the CLI, mid-stroke...")
        os.killpg(os.getpgid(cli.pid), signal.SIGINT)
        hung = False
        try:
            cli.wait(timeout=15)
        except subprocess.TimeoutExpired:
            hung = True
            print("CLI still running 15 s after SIGINT")
        time.sleep(1.0)  # let the stub's goal loop notice and log
    finally:
        kill(cli)
        kill(stub)

    said = stub_log.read_text()
    cli_said = cli_log.read_text()
    print("\n--- stub saw ---\n%s" % said.strip())
    print("\n--- cli tail ---\n%s" % cli_said.strip()[-1200:])

    cancelled = "CANCEL RECEIVED" in said and "STOPPED at" in said
    completed = "RAN TO COMPLETION" in said
    sent = cli_said.count("Type 'yes'")  # one prompt == one run attempt
    execs = said.count("EXEC GOAL ACCEPTED")
    audit_ok = execs == 1 and sent == 1
    # a cancel that escapes as the context dies is LUCK, not ownership:
    # the CLI must survive its own abort path and say what it confirmed
    crashed = "Traceback" in cli_said or "RCLError" in cli_said
    confirmed = "cancel delivered" in cli_said
    double_spoke = "unconfirmed" in cli_said  # backstop fired on the owned path

    print()
    if (
        cancelled
        and confirmed
        and not (completed or hung or crashed or double_spoke)
        and audit_ok
    ):
        print(
            "PASS — Ctrl+C delivered the cancel on a live context; stub "
            "stopped mid-stroke; CLI confirmed and exited cleanly"
        )
        sys.exit(0)
    for cond, msg in [
        (completed, "trajectory RAN TO COMPLETION — the cancel never arrived"),
        (not cancelled, "no cancel reached the stub"),
        (
            crashed,
            "CLI crashed in its abort path (context died before the "
            "cancel round-trip — the delivery was a race, not owned)",
        ),
        (not confirmed, "CLI never confirmed the cancel round-trip"),
        (
            double_spoke,
            "owned path AND backstop both spoke — contradictory " "abort report",
        ),
        (hung, "CLI did not exit after SIGINT"),
        (not audit_ok, "goal-count audit failed (%d exec goals seen)" % execs),
    ]:
        if cond:
            print("FAIL — " + msg)
    sys.exit(1)


if __name__ == "__main__":
    main()
