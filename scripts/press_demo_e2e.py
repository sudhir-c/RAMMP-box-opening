#!/usr/bin/env python3
"""press_demo end-to-end against stubs: real detection, fake physics.

    python3 scripts/press_demo_e2e.py           # isolates on ROS_DOMAIN_ID=77

Two scenarios, both with a stub planner (scripts/stub_planner.py) and a
synthetic D405 (scripts/stub_d405.py) publishing REAL rendered ArUco
frames + the mount-consistent static TF, so the CLI's whole perception
path (detect -> PnP -> TF -> depth refinement -> container pose) runs
for real:

  tag:     full flow — scan, fix, staging, close, hover, press, retreat,
           home. Must exit 0 with 6 exec goals, 1 gripper goal, no
           cancels, and a recovered container origin within 5 mm of the
           geometry the synthetic camera encoded.
  no-tag:  tagless frames — scan, detect timeout, home, exit 2, exactly
           2 exec goals.

Goal counts are audited (lesson 6); the harness refuses to run beside a
real controller_manager or planner.
"""

import os
import re
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
    "export STUB_PLAN_S=1.2; "
    "source /opt/ros/humble/setup.zsh; "
    "source ~/RAMMP-CuRobo/install/setup.zsh; "
    "source %s/install/setup.zsh; " % (DOMAIN, REPO)
)

TAG_XYZ = (0.45, 0.02, 0.133)  # what the synthetic camera encodes


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


def wait_for(path, needle, timeout, proc=None, what=""):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if needle in Path(path).read_text():
            return True
        if proc is not None and proc.poll() is not None:
            sys.exit(
                "%s died before %r:\n%s"
                % (what, needle, Path(path).read_text()[-2500:])
            )
        time.sleep(0.2)
    return False


def spawn(cmd, log):
    return subprocess.Popen(
        ["zsh", "-c", CHAIN + cmd],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_scenario(tmp, cfg, marker):
    name = "tag" if marker else "no-tag"
    stub_log = tmp / ("stub_%s.log" % name)
    cam_log = tmp / ("cam_%s.log" % name)
    cli_log = tmp / ("cli_%s.log" % name)
    stub = cam = cli = None
    try:
        stub = spawn("exec python3 %s" % (REPO / "scripts/stub_planner.py"), stub_log)
        cam = spawn(
            "exec python3 %s%s"
            % (REPO / "scripts/stub_d405.py", "" if marker else " --no-marker"),
            cam_log,
        )
        if not wait_for(stub_log, "STUB READY", 30, stub, "stub planner"):
            sys.exit("stub planner never ready")
        if not wait_for(cam_log, "STUB D405 READY", 30, cam, "stub d405"):
            sys.exit("stub d405 never ready")

        cli = spawn(
            "exec ros2 run rammp_box_opening press_demo --execute --container %s" % cfg,
            cli_log,
        )
        deadline = 180
        t0 = time.monotonic()
        while cli.poll() is None and time.monotonic() - t0 < deadline:
            time.sleep(0.5)
        hung = cli.poll() is None
        code = cli.returncode
    finally:
        kill(cli)
        kill(cam)
        kill(stub)

    said = stub_log.read_text()
    cli_said = cli_log.read_text()
    print("\n===== scenario %s =====" % name)
    print("--- cli tail ---\n%s" % cli_said.strip()[-1500:])
    print(
        "--- stub counts: exec=%d gripper=%d complete=%d cancel=%d"
        % (
            said.count("EXEC GOAL ACCEPTED"),
            said.count("GRIPPER GOAL"),
            said.count("RAN TO COMPLETION"),
            said.count("CANCEL RECEIVED"),
        )
    )

    fails = []
    if hung:
        fails.append("CLI hung past %d s" % deadline)
    if "Traceback" in cli_said:
        fails.append("CLI traceback")
    execs = said.count("EXEC GOAL ACCEPTED")
    if marker:
        if code != 0:
            fails.append("exit %s != 0" % code)
        if execs != 6:
            fails.append("exec goals %d != 6" % execs)
        if said.count("GRIPPER GOAL") != 1:
            fails.append("gripper goals != 1")
        if said.count("CANCEL RECEIVED") != 0:
            fails.append("unexpected cancel")
        if "depth-refined" not in cli_said:
            fails.append("depth refinement never engaged")
        m = re.search(r"container origin \[([-\d.]+), ([-\d.]+), ([-\d.]+)\]", cli_said)
        if not m:
            fails.append("no container-origin line")
        else:
            got = [float(v) for v in m.groups()]
            want = [TAG_XYZ[0], TAG_XYZ[1], TAG_XYZ[2] - 0.16]  # minus button_offset z
            err = max(abs(a - b) for a, b in zip(got, want))
            print("--- recovered origin %s vs true %s (err %.4f m)" % (got, want, err))
            if err > 0.005:
                fails.append("origin error %.4f m > 5 mm" % err)
    else:
        if code != 2:
            fails.append("exit %s != 2" % code)
        if execs != 2:
            fails.append("exec goals %d != 2 (scan + home)" % execs)
        if "NO TAG" not in cli_said:
            fails.append("no NO TAG line")
    return fails


def main():
    tmp = Path(tempfile.mkdtemp(prefix="press_demo_e2e_"))
    print("workdir %s (domain %s)" % (tmp, DOMAIN))

    sh("ros2 daemon stop", timeout=30)
    probe = sh("timeout 20 ros2 node list", timeout=30)
    if "/controller_manager" in probe.stdout or "/rammp_curobo" in probe.stdout:
        sys.exit(
            "REAL arm stack or planner visible on ROS_DOMAIN_ID=%s:\n%s\nrefusing."
            % (DOMAIN, probe.stdout)
        )

    cfg = tmp / "oxo_measured.yaml"
    src_cfg = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"
    cfg.write_text(src_cfg.read_text().replace("measure_me: true", "measure_me: false"))

    all_fails = []
    for marker in (True, False):
        all_fails += [
            "%s: %s" % ("tag" if marker else "no-tag", f)
            for f in run_scenario(tmp, cfg, marker)
        ]

    print()
    if not all_fails:
        print("PASS — full tag flow and no-tag exit both behave; goal audit clean")
        sys.exit(0)
    for f in all_fails:
        print("FAIL — " + f)
    sys.exit(1)


if __name__ == "__main__":
    main()
