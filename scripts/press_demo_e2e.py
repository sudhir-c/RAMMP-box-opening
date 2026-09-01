#!/usr/bin/env python3
"""press_demo end-to-end against stubs: real detection, fake physics.

    python3 scripts/press_demo_e2e.py           # isolates on ROS_DOMAIN_ID=77

Three scenarios, each with a stub planner (scripts/stub_planner.py) and
a synthetic D405 (scripts/stub_d405.py) publishing REAL rendered ArUco
frames + mount-consistent TF, so the CLI's whole perception path
(detect -> PnP -> TF -> depth refinement -> container pose) runs for
real. Depth is spatially structured (tag range only at the tag's
pixels) and the tag carries a 30 deg yaw plus a nonzero tag->button
offset, so wrong-pixel depth sampling and wrong rotation composition
both move the recovered origin and FAIL the 5 mm / 3 deg checks.

  tag:    full flow — exit 0, 8 exec goals, 4 gripper goals, one cancel
          (the guarded set-down trips by design), origin within 5 mm and
          yaw within 3 deg of the
          geometry the synthetic camera encoded.
  trip:   efforts spike late in the press (STUB_TRIP_EXEC_N=2) — the guard
          cancels the stroke, the CLI reports pressed-via-trip, retreat
          and home replan from the stop, exit 0. Exactly one cancel.
  no-tag: tagless frames — scan, a detect wait that provably lasts
          timeout_s, home, exit 2, exactly 2 exec goals.

Goal counts are audited (lesson 6); the harness refuses to run beside a
real controller_manager or planner.
"""

import math
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
    "export STUB_PLAN_S=1.2; export STUB_GRIP_POS=0.45; "
    "source /opt/ros/humble/setup.zsh; "
    "source ~/RAMMP-CuRobo/install/setup.zsh; "
    "source %s/install/setup.zsh; " % (DOMAIN, REPO)
)

TAG_XYZ = (0.42, 0.0, 0.133)  # on the scan camera's axis: servo is a no-op
TAG_YAW_DEG = 30.0
TAG_OFFSET = (0.01, 0.0, 0.0)  # container-frame tag->button, in the cfg
TAG_SIZE_M = 0.05  # harness-internal: cfg and synthetic camera BOTH get
#                    this, whatever edge the real printed tag measures
DETECT_TIMEOUT_S = 10.0  # oxo_pop.yaml detect.timeout_s


def button_z(cfg_text):
    """button_offset z from the yaml itself — never a stale constant."""
    m = re.search(r"button_offset: \[[^,]+, [^,]+, ([\d.]+)\]", cfg_text)
    return float(m.group(1))


def expected_origin(btn_z):
    yaw = math.radians(TAG_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, _ = TAG_OFFSET
    return [
        TAG_XYZ[0] + c * ox - s * oy,
        TAG_XYZ[1] + s * ox + c * oy,
        TAG_XYZ[2] - btn_z,
    ]


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


def spawn(cmd, log, extra_env=""):
    return subprocess.Popen(
        ["zsh", "-c", CHAIN + extra_env + cmd],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_scenario(tmp, cfg, mode):
    stub_log = tmp / ("stub_%s.log" % mode)
    cam_log = tmp / ("cam_%s.log" % mode)
    cli_log = tmp / ("cli_%s.log" % mode)
    cam_args = " --size %g --tag-x %g --tag-y %g --tag-z %g" % (
        (TAG_SIZE_M,) + TAG_XYZ
    ) + (" --no-marker" if mode == "no-tag" else (" --tag-yaw-deg %g" % TAG_YAW_DEG))
    # the guarded set-down (place:lid:down) must always trip; the trip
    # scenario also trips the press stroke
    stub_env = (
        "export STUB_TRIP_EXEC_N=2,7; "
        if mode == "trip"
        else "export STUB_TRIP_EXEC_N=7; "
    )
    stub = cam = cli = None
    try:
        stub = spawn(
            "exec python3 %s" % (REPO / "scripts/stub_planner.py"), stub_log, stub_env
        )
        cam = spawn(
            "exec python3 %s%s" % (REPO / "scripts/stub_d405.py", cam_args), cam_log
        )
        if not wait_for(stub_log, "STUB READY", 30, stub, "stub planner"):
            sys.exit("stub planner never ready")
        if not wait_for(cam_log, "STUB D405 READY", 30, cam, "stub d405"):
            sys.exit("stub d405 never ready")

        t_cli = time.monotonic()
        cli = spawn(
            "exec ros2 run rammp_box_opening press_demo --execute --container %s" % cfg,
            cli_log,
        )
        deadline = 180
        while cli.poll() is None and time.monotonic() - t_cli < deadline:
            time.sleep(0.5)
        hung = cli.poll() is None
        code = cli.returncode
        elapsed = time.monotonic() - t_cli
    finally:
        kill(cli)
        kill(cam)
        kill(stub)

    said = stub_log.read_text()
    cli_said = cli_log.read_text()
    execs = said.count("EXEC GOAL ACCEPTED")
    cancels = said.count("CANCEL RECEIVED")
    print("\n===== scenario %s =====" % mode)
    print("--- cli tail ---\n%s" % cli_said.strip()[-1500:])
    print(
        "--- stub: exec=%d gripper=%d complete=%d cancel=%d  elapsed=%.0fs"
        % (
            execs,
            said.count("GRIPPER GOAL"),
            said.count("RAN TO COMPLETION"),
            cancels,
            elapsed,
        )
    )

    fails = []
    if hung:
        fails.append("CLI hung past %d s" % deadline)
    if "Traceback" in cli_said:
        fails.append("CLI traceback")

    if mode == "no-tag":
        if code != 2:
            fails.append("exit %s != 2" % code)
        if execs != 2:
            fails.append("exec goals %d != 2 (scan + home)" % execs)
        if "NO TAG" not in cli_said:
            fails.append("no NO TAG line")
        # the detect wait must actually last the configured window:
        # scan + wait (10 s) + home; legs are ~1.6 s each at 0.75 — a
        # shortened wait would finish well under timeout_s + leg time
        if elapsed < DETECT_TIMEOUT_S + 2.0:
            fails.append(
                "run took %.0f s — detect wait shorter than timeout_s?" % elapsed
            )
        return fails

    # tag and trip scenarios share the flow assertions
    if code != 0:
        fails.append("exit %s != 0" % code)
    if execs != 8:
        # scan, approach, press, retreat, grip:down, lift, place transit,
        # place:down, place-retreat+home (merged)
        fails.append("exec goals %d != 8" % execs)
    if said.count("GRIPPER GOAL") != 4:
        fails.append("gripper goals %d != 4" % said.count("GRIPPER GOAL"))
    if "LID PULLED" not in cli_said:
        fails.append("no LID PULLED line")
    if "DONE — box open" not in cli_said:
        fails.append("no final DONE line")
    if "depth-refined" not in cli_said:
        fails.append("depth refinement never engaged")
    if "CENTERED" not in cli_said:
        fails.append("servo never reported CENTERED")
    m = re.search(
        r"PRESS target origin \[([-\d.]+), ([-\d.]+), ([-\d.]+)\] yaw ([-\d.]+) deg",
        cli_said,
    )
    if not m:
        fails.append("no PRESS-target line")
    else:
        got = [float(v) for v in m.groups()[:3]]
        yaw = float(m.group(4))
        want = expected_origin(button_z(Path(cfg).read_text()))
        err = max(abs(a - b) for a, b in zip(got, want))
        print(
            "--- recovered origin %s yaw %.1f vs true %s yaw %.1f (err %.4f m)"
            % (got, yaw, [round(v, 4) for v in want], TAG_YAW_DEG, err)
        )
        if err > 0.005:
            fails.append("origin error %.4f m > 5 mm" % err)
        if abs(yaw - TAG_YAW_DEG) > 3.0:
            fails.append("yaw error %.1f deg > 3" % abs(yaw - TAG_YAW_DEG))

    if mode == "tag":
        if cancels != 1:  # the guarded set-down trips (by design)
            fails.append("cancels %d != 1 (set-down trip)" % cancels)
        if "full travel" not in cli_said:
            fails.append("press did not report full-travel outcome")
    if mode == "trip":
        if cancels != 2:  # press trip + set-down trip
            fails.append("cancels %d != 2 (press + set-down)" % cancels)
        if "EFFORT SPIKE" not in said:
            fails.append("stub never injected the spike")
        if "guard stopped the stroke" not in cli_said:
            fails.append("CLI did not report pressed-via-trip")
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
    cfg.write_text(
        re.sub(
            r"size_m: [\d.]+",
            "size_m: %g" % TAG_SIZE_M,  # stay in sync with the synthetic
            src_cfg.read_text()
            .replace("measure_me: true", "measure_me: false")
            # the synthetic camera renders TAGS; its flat depth plane
            # would (rightly) never pass the depth source's footprint
            # gate, so the harness pins the tag path explicitly
            .replace("source: vlm", "source: tag")
            .replace(
                "offset_xyz: [0.0, 0.0, 0.0]",
                "offset_xyz: [%g, %g, %g]" % TAG_OFFSET,
            ),
            count=1,
        )
    )

    all_fails = []
    for mode in ("tag", "trip", "no-tag"):
        all_fails += ["%s: %s" % (mode, f) for f in run_scenario(tmp, cfg, mode)]

    print()
    if not all_fails:
        print(
            "PASS — tag flow (yaw+offset recovered), guard-trip press, and "
            "no-tag exit all behave; goal audits clean"
        )
        sys.exit(0)
    for f in all_fails:
        print("FAIL — " + f)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        # the probe above rebinds the ros2 CLI daemon to the isolated
        # domain; leaving it there makes `ros2 node list` in normal shells
        # come up empty (field lesson 8) — put it back down on the way out
        sh("ros2 daemon stop", timeout=30)
