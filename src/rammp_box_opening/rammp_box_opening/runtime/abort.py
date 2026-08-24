"""SIGINT ownership for motion CLIs: cancel-then-exit (spec §1, lesson 7).

rclpy's default SIGINT handler shuts the context down out from under an
in-flight ExecuteTrajectory goal. On this stack the cancel request
usually still escapes onto the wire, but the confirmation spin dies on
the invalidated context — a traceback instead of an answer to "did the
arm stop?", and delivery itself is a version/timing race, not a
guarantee (RAMMP-CuRobo's abort_checks documented the planner-side
variant where the cancel never arrived at all).

Ownership pattern (from sweep_demo.py / planner_node.py):
`rclpy.init(signal_handler_options=SignalHandlerOptions.NO)`, install
this handler, and let PlannerClient.execute deliver the cancel on a
LIVE context, await the server's result, report honestly, then exit.

Semantics:
- Ctrl+C with NO arm goal in flight (prompt, planning, gripper wait):
  plain KeyboardInterrupt, immediately — the CLI dies, nothing to stop.
  (A gripper close in flight finishes server-side, <= 1 s; the arm
  trajectory goals are the hazard this module owns.)
- Ctrl+C mid-trajectory: sets the flag only; the execute loop cancels,
  waits for the server to confirm the stop, prints the truth, raises.
- Second Ctrl+C: KeyboardInterrupt regardless — the escalation path for
  a wedged cancel (the first request is already on the wire).
- SIGTERM is NOT owned: `kill` stays a hard stop with no cancel. Use the
  physical e-stop for hardware emergencies; Ctrl+C is the drilled path.
"""

import signal


class AbortFlag:
    """Shared between the SIGINT handler and PlannerClient.execute."""

    def __init__(self):
        self.requested = False
        self.goal_in_flight = False


def sigint_handler(flag):
    def _handler(_signum, _frame):
        first = not flag.requested
        flag.requested = True
        if not flag.goal_in_flight or not first:
            raise KeyboardInterrupt

    return _handler


def install_sigint(flag):
    signal.signal(signal.SIGINT, sigint_handler(flag))
