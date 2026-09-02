#!/usr/bin/env python3
"""press_demo end-to-end against stubs: real detection, fake physics.

    python3 scripts/press_demo_e2e.py           # isolates on ROS_DOMAIN_ID=77

Three scenarios, each with a stub planner (scripts/stub_planner.py) and a
synthetic D405 + OWL stub (scripts/stub_d405.py) publishing a RAY-CAST
depth scene — a box-shaped plateau at table + dims.z with the
container's footprint, a button disc in colour, a mount-consistent TF
and the owl node's bbox topic — so the CLI's whole SHIPPED perception
path (owl rung -> depth plateau -> TF lift -> button circle -> container
pose) runs for real under the shipped ladder (detect.source: vlm; only
the cloud rung is dropped, so nothing leaves the machine). The box sits
off the camera axis at a 30 deg yaw, so wrong deprojection or rotation
composition moves the recovered origin and FAILS the 5 mm / 3 deg
checks (yaw mod 90: the box is square and the depth path says so).

  box:    full flow — exit 0, 8 exec goals, 4 gripper goals, one cancel
          (the guarded set-down trips by design), origin within 5 mm and
          yaw within 3 deg of the geometry the synthetic camera encoded,
          origin z pinned to the calibrated table.
  trip:   efforts spike late in the press (STUB_TRIP_EXEC_N=2) — the guard
          cancels the stroke, the CLI reports pressed-via-trip, retreat
          and home replan from the stop, exit 0. Exactly two cancels.
  no-box: an empty table — scan, the owl rung is consulted (the stub
          heartbeats), a detect wait that provably lasts timeout_s, home,
          exit 2, exactly 2 exec goals.

Goal counts are audited (lesson 6); the harness refuses to run beside a
real controller_manager or planner.
"""

import re
import sys
import time

from e2e_common import REPO, Shell, kill, measured_config, wait_for, workdir

sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

from rammp_box_opening.worlds import WorldStore  # noqa: E402

SH = Shell("export STUB_PLAN_S=1.2; export STUB_GRIP_POS=0.45; ")

BOX_XY = (0.46, -0.05)  # off the scan camera's axis: deprojection errors show
BOX_YAW_DEG = 30.0  # the depth path reports yaw mod 90 (square box)
DETECT_TIMEOUT_S = 10.0  # oxo_pop.yaml detect.timeout_s
BENCH_YAML = REPO / "src/rammp_box_opening/config/world_bench.yaml"


def yaw_err_deg(got, want):
    """Yaw error for a square box: the depth path reports yaw mod 90."""
    d = abs((got - want) % 90.0)
    return min(d, 90.0 - d)


def run_scenario(tmp, cfg, table_z, mode):
    stub_log = tmp / ("stub_%s.log" % mode)
    cam_log = tmp / ("cam_%s.log" % mode)
    cli_log = tmp / ("cli_%s.log" % mode)
    # the stub renders the container the CLI is configured for, on the
    # table the CLI's bench world says it stands on
    cam_args = " --container %s --table-z %g" % (cfg, table_z)
    if mode == "no-box":
        cam_args += " --no-box"
    else:
        cam_args += " --box-x %g --box-y %g --box-yaw-deg %g" % (BOX_XY + (BOX_YAW_DEG,))
    # the guarded set-down (place:lid:down) must always trip; the trip
    # scenario also trips the press stroke
    stub_env = (
        "export STUB_TRIP_EXEC_N=2,7; "
        if mode == "trip"
        else "export STUB_TRIP_EXEC_N=7; "
    )
    stub = cam = cli = None
    try:
        stub = SH.spawn(
            "exec python3 %s" % (REPO / "scripts/stub_planner.py"), stub_log, stub_env
        )
        cam = SH.spawn(
            "exec python3 %s%s" % (REPO / "scripts/stub_d405.py", cam_args), cam_log
        )
        if not wait_for(stub_log, "STUB READY", 30, stub, "stub planner"):
            sys.exit("stub planner never ready")
        if not wait_for(cam_log, "STUB D405 READY", 30, cam, "stub d405"):
            sys.exit("stub d405 never ready")

        t_cli = time.monotonic()
        cli = SH.spawn(
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

    if mode == "no-box":
        if code != 2:
            fails.append("exit %s != 2" % code)
        if execs != 2:
            fails.append("exec goals %d != 2 (scan + home)" % execs)
        if "NO BOX" not in cli_said:
            fails.append("no NO BOX line")
        # depth found nothing in its first beat, so the ladder ran and the
        # owl rung was consulted — the stub answers with heartbeats. (Its
        # verdict is not asserted: the node stamps its messages as
        # float32 seconds, which quantizes wall time to 128 s and makes
        # the rung's freshness classification a coin flip.)
        if "VLM owl:" not in cli_said:
            fails.append("the owl rung was never consulted")
        # the detect wait must actually last the configured window:
        # scan + wait (10 s) + home; legs are ~1.6 s each at 0.75 — a
        # shortened wait would finish well under timeout_s + leg time
        if elapsed < DETECT_TIMEOUT_S + 2.0:
            fails.append(
                "run took %.0f s — detect wait shorter than timeout_s?" % elapsed
            )
        return fails

    # box and trip scenarios share the flow assertions
    if code != 0:
        fails.append("exit %s != 0" % code)
    if execs != 8:
        # scan, press, retreat, grip:down, lift, place transit, place:down,
        # place-retreat+home (merged)
        fails.append("exec goals %d != 8" % execs)
    if said.count("GRIPPER GOAL") != 4:
        fails.append("gripper goals %d != 4" % said.count("GRIPPER GOAL"))
    if "LID PULLED" not in cli_said:
        fails.append("no LID PULLED line")
    if "DONE — box open" not in cli_said:
        fails.append("no final DONE line")
    for needle, what in (
        ("[depth] measured top", "the depth watcher never reported the top residual"),
        ("origin z pinned to the table", "origin z was not pinned to the table"),
        ("[press_demo] BOX at", "no BOX line"),
        ("found a container top", "the depth status never reported a container top"),
    ):
        if needle not in cli_said:
            fails.append(what)
    m = re.search(
        r"(\d+)/(\d+) frames found a container top, (\d+) button-circle", cli_said
    )
    if m and int(m.group(3)) == 0:
        fails.append("the button circle never refined a sighting")
    m = re.search(
        r"PRESS target origin \[([-\d.]+), ([-\d.]+), ([-\d.]+)\] yaw ([-\d.]+) deg",
        cli_said,
    )
    if not m:
        fails.append("no PRESS-target line")
    else:
        got = [float(v) for v in m.groups()[:3]]
        yaw = float(m.group(4))
        want = [BOX_XY[0], BOX_XY[1], table_z]
        err = max(abs(a - b) for a, b in zip(got, want))
        yerr = yaw_err_deg(yaw, BOX_YAW_DEG)
        print(
            "--- recovered origin %s yaw %.1f vs true %s yaw %.1f (err %.4f m, %.1f deg)"
            % (got, yaw, [round(v, 4) for v in want], BOX_YAW_DEG, err, yerr)
        )
        if err > 0.005:
            fails.append("origin error %.4f m > 5 mm" % err)
        if yerr > 3.0:
            fails.append("yaw error %.1f deg > 3" % yerr)

    if mode == "box":
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
    tmp = workdir("press_demo_e2e_")
    SH.refuse_real_stack()

    # the shipped ladder minus its cloud rung: the harness runs offline
    # and must never make a network call
    cfg = measured_config(
        tmp,
        edit=lambda text: re.sub(
            r"backends: \[[^\]]*\]", "backends: [owl]", text, count=1
        ),
    )
    # the table the CLI bands container candidates above — and pins the
    # container origin to — is the bench world's; the stub renders it
    table_z = WorldStore(str(BENCH_YAML), out_dir=tmp).table_top_z

    all_fails = []
    for mode in ("box", "trip", "no-box"):
        all_fails += [
            "%s: %s" % (mode, f) for f in run_scenario(tmp, cfg, table_z, mode)
        ]

    print()
    if not all_fails:
        print(
            "PASS — depth flow (origin+yaw recovered, z pinned), guard-trip "
            "press, and no-box exit all behave; goal audits clean"
        )
        sys.exit(0)
    for f in all_fails:
        print("FAIL — " + f)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        SH.daemon_reset()
