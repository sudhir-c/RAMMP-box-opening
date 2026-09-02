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
with a real controller_manager or planner (e2e_common), and audits goal
counts (CLI-sent == stub-received) per lesson 6.
"""

import os
import signal
import subprocess
import sys
import time

from e2e_common import REPO, Shell, kill, measured_config, wait_for, workdir

# pin the stub stroke: the mid-stroke SIGINT premise (~24 s at 0.25) must
# not bend to an inherited STUB_PLAN_S from the environment
SH = Shell("export STUB_PLAN_S=6.0; unset STUB_TRIP_EXEC_N STUB_GRIP_POS; ")


def main():
    tmp = workdir("abort_e2e_")
    SH.refuse_real_stack()

    stub_log = tmp / "stub.log"
    cli_log = tmp / "cli.log"
    cfg = measured_config(tmp)  # --execute refuses while measure_me is true

    stub = cli = None
    try:
        stub = SH.spawn("exec python3 %s" % (REPO / "scripts/stub_planner.py"), stub_log)
        if not wait_for(stub_log, "STUB READY", 30, stub, "stub"):
            sys.exit("stub never became ready:\n" + stub_log.read_text()[-2000:])

        cli = SH.spawn(
            "exec ros2 run rammp_box_opening home_arm --execute --container %s" % cfg,
            cli_log,
            stdin=subprocess.PIPE,
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
            "owned path AND backstop both spoke — contradictory abort report",
        ),
        (hung, "CLI did not exit after SIGINT"),
        (not audit_ok, "goal-count audit failed (%d exec goals seen)" % execs),
    ]:
        if cond:
            print("FAIL — " + msg)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        SH.daemon_reset()
