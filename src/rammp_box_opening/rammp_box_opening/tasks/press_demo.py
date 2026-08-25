"""press_demo — the tag-driven autonomous press (owner design 2026-08-24).

    ros2 launch rammp_box_opening press_demo.launch.py execute:=true
    ros2 run rammp_box_opening press_demo --execute

Flow, states logged one line each: SCAN (tool-down look pose, bench-only
world) -> DETECT (continuous wrist-camera watcher; no fresh stable fix
within the timeout -> home, exit 2) -> APPROACH staging (full world with
the tag-derived container cuboid) -> PRESS (hover at 1 inch, one
fixed-travel guarded stroke) -> RETREAT -> HOME.

--execute alone arms it: NO typed confirmation (owner decision
2026-08-24 — autonomous once started, Ctrl+C stops everything via the
stub-proven cancel path). The run is attended, e-stop in hand, and the
planner's execute parameter plus the measure_me worksheet gate still
apply.
"""

import math
import sys
import time

import rclpy

from rammp_box_opening.constants import HOME, TRANSIT_SPEED
from rammp_box_opening.models.container import (
    ContainerModel,
    load_press_demo,
)
from rammp_box_opening.perception.tag_source import (
    TagWatcher,
    container_pose_from_tag,
    servo_step,
)
from rammp_box_opening.primitives.core import (
    Ctx,
    Home,
    PlanState,
    PressFixed,
    Retreat,
    _full_world,
    _plan_motion,
)
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


def _state(joints):
    return PlanState(joints=list(joints), chain=0, contact_broke_chain=False)


def build_scan_leg(ctx, cfg, start_joints):
    """Tool-down look pose over the bench. The bench world carries the
    unseen-container keep-out band: a container is somewhere, its pose
    unknown, so pre-detection transits stay above container height."""
    world = ctx.worlds.push_name("bench", model=ctx.model)
    bearing = math.atan2(cfg.scan_xyz[1], cfg.scan_xyz[0])
    quat = attitude_quat([180.0, 0.0, 0.0], bearing)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        "scan",
        ("pose", list(cfg.scan_xyz), quat),
        world,
        TRANSIT_SPEED,
    )
    return leg


def build_home_leg(ctx, start_joints):
    """No-tag exit: back to HOME above the unseen-container band."""
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        "home",
        ("joints", list(HOME)),
        world,
        TRANSIT_SPEED,
    )
    return leg


def build_approach_leg(ctx, cfg):
    """Staging directly above the tag-derived button, full world."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    # bearing-steered attitude, same as PressFixed: press is yaw-invariant
    # and the bearing family is the reach-map-certified one (2026-08-25)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(button[1], button[0]))
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    leg, _ = _plan_motion(
        ctx,
        _state(ctx.client.joints()),
        "approach:staging",
        ("pose", staging, quat),
        _full_world(ctx),
        TRANSIT_SPEED,
    )
    return leg


def build_press_legs(ctx, cfg):
    """Hover + fixed stroke + retreat + home, planned from live joints
    (run AFTER the approach so the close-range re-fix can update cpose)."""
    st = _state(ctx.client.joints())
    press_legs, st = PressFixed(cfg).plan(ctx, st)
    # from the press bottom back up to staging height
    retreat_legs, st = Retreat(cfg.staging_m + cfg.travel_m).plan(ctx, st)
    home_legs, st = Home().plan(ctx, st)
    return [*press_legs, *retreat_legs, *home_legs]


def build_demo_legs(ctx, cfg):
    """The one-shot composition (offline tests); main runs it in phases."""
    return [build_approach_leg(ctx, cfg), *build_press_legs(ctx, cfg)]


def build_servo_leg(ctx, cfg, disp, i):
    """One lateral centering translation at the current height."""
    tool = ctx.client.tool_xyz()
    if tool is None:
        return None
    target = [tool[0] + disp[0], tool[1] + disp[1], tool[2]]
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(target[1], target[0]))
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(ctx.client.joints()),
        "servo:%d" % i,
        ("pose", target, quat),
        world,
        TRANSIT_SPEED,
    )
    return leg


def center_on_tag(node, watcher, ctx, cfg, runner, execute):
    """Owner design 2026-08-25: translate at scan height until the tag is
    at the image center, then press from what the CENTERED camera sees.
    Robust to camera-mount error — the loop converges even with an
    imperfect direction, and a centered tag is under the optical axis no
    matter what the mount calibration believes."""
    for i in range(cfg.servo_max_iters + 1):
        got = wait_for_fix(node, watcher, cfg)
        if got is None:
            return None
        p_cam, rot_cam, _t = watcher.last_debug
        disp, px = servo_step(
            p_cam,
            rot_cam,
            watcher.grab.k,
            cfg.servo_tol_px,
            cfg.servo_min_step_m,
            cfg.servo_max_step_m,
        )
        if disp is None:
            print("[press_demo] CENTERED — tag %.0f px off the optical axis" % px)
            return got
        if i == cfg.servo_max_iters:
            print(
                "[press_demo] centering unconverged (%.0f px after %d moves) "
                "— pressing on the freshest fix" % (px, i)
            )
            return got
        print(
            "[press_demo] SERVO %d: tag %.0f px off — shifting [%.3f, %.3f]"
            % (i + 1, px, disp[0], disp[1])
        )
        leg = build_servo_leg(ctx, cfg, disp, i + 1)
        if leg is None:
            print("[press_demo] no tool TF for the servo move — stopping")
            return None
        res = runner.run([leg], execute=execute, assume_yes=True)
        if any(not r.ok for r in res):
            return None
    return None


def _spin_detect(node):
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except RuntimeError as e:
        # rclpy teardown artifact: our SIGINT handler raising inside a
        # subscription take surfaces as RuntimeError, not KeyboardInterrupt
        raise KeyboardInterrupt from e


def wait_for_fix(node, watcher, cfg, timeout_s=None):
    """Spin (the watcher ticks on its timer) until a fresh stable fix.

    The window is purged first: sightings gathered while the arm was
    still moving carry TF/depth timing skew — only parked-camera frames
    may commit the fix the press will trust."""
    watcher.reset()
    limit = cfg.timeout_s if timeout_s is None else timeout_s
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        _spin_detect(node)
        got = watcher.fix()
        if got is not None:
            return got
    return None


def detect_only_report(node, watcher, ctx, cfg, runner, execute):
    """Mount-calibration observation: hold at the scan pose reporting every
    fresh fix with its raw camera-frame ingredients, then park home.

    Place the container at a TAPE-MEASURED spot first; the printed base
    pose vs truth solves the wrist-mount error."""
    watcher.reset()
    print("[press_demo] DETECT-ONLY: reporting fixes for 15 s")
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < 15.0:
        _spin_detect(node)
        got = watcher.fix()
        if got is None:
            continue
        pos, rot = got
        key = tuple(round(float(v), 4) for v in pos)
        if key == last:
            continue
        last = key
        yaw = math.degrees(math.atan2(rot[1][0], rot[0][0]))
        p_cam, rot_cam, t_cam = watcher.last_debug
        print(
            "fix: base [%.3f, %.3f, %.3f] yaw %.1f | p_cam [%.3f, %.3f, %.3f]"
            " | cam_t [%.3f, %.3f, %.3f]"
            % (
                pos[0],
                pos[1],
                pos[2],
                yaw,
                p_cam[0],
                p_cam[1],
                p_cam[2],
                t_cam[0],
                t_cam[1],
                t_cam[2],
            )
        )
        print(
            "     cam_R rows [%.3f %.3f %.3f] [%.3f %.3f %.3f] [%.3f %.3f %.3f]"
            % tuple(float(v) for row in rot_cam for v in row)
        )
    print("[press_demo] detect-only done (%s) — homing" % watcher.status())
    runner.run(
        [build_home_leg(ctx, ctx.client.joints())], execute=execute, assume_yes=True
    )


def main():
    ap = cli_common.make_parser(__doc__)
    ap.add_argument(
        "--detect-only",
        action="store_true",
        help="scan, report fixes + camera-frame diagnostics for 15 s, home, "
        "exit — the wrist-mount calibration observation (no press)",
    )
    args = ap.parse_args()
    cfg_path = args.container or cli_common.default_container_yaml()
    bench = args.bench_world or cli_common.default_bench_yaml()
    model = ContainerModel.load(cfg_path)
    cfg = load_press_demo(cfg_path)
    cli_common.refuse_unmeasured(model, args.execute)
    node, client = cli_common.init_runtime()
    worlds = WorldStore(bench)
    runner = Runner(client, worlds)
    watcher = TagWatcher(node, cfg)  # camera on from here to exit
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=worlds,
        config_path=cfg_path,
    )

    try:
        print("[press_demo] SCAN: tool-down look pose %s" % (list(cfg.scan_xyz),))
        res = runner.run(
            [build_scan_leg(ctx, cfg, client.joints())],
            execute=args.execute,
            assume_yes=True,  # --execute alone arms the run (owner 2026-08-24)
        )
        if any(not r.ok for r in res):
            sys.exit(1)

        if args.detect_only:
            detect_only_report(node, watcher, ctx, cfg, runner, args.execute)
            sys.exit(0)

        print(
            "[press_demo] DETECT: waiting %.0f s for a stable fix (tag id %d)"
            % (cfg.timeout_s, cfg.tag_id)
        )
        got = center_on_tag(node, watcher, ctx, cfg, runner, args.execute)
        if got is None:
            print("[press_demo] NO TAG — %s — returning home" % watcher.status())
            runner.run(
                [build_home_leg(ctx, client.joints())],
                execute=args.execute,
                assume_yes=True,
            )
            sys.exit(2)

        pos, rot = got
        ctx.cpose = container_pose_from_tag(pos, rot, model, cfg.tag_offset)
        print(
            "[press_demo] TAG at [%.3f, %.3f, %.3f] (%s) -> container origin "
            "[%.3f, %.3f, %.3f] yaw %.1f deg"
            % (
                pos[0],
                pos[1],
                pos[2],
                watcher.status(),
                ctx.cpose.xyz[0],
                ctx.cpose.xyz[1],
                ctx.cpose.xyz[2],
                math.degrees(ctx.cpose.yaw),
            )
        )

        res = runner.run(
            [build_approach_leg(ctx, cfg)], execute=args.execute, assume_yes=True
        )
        if any(not r.ok for r in res):
            sys.exit(1)

        # close-range re-fix: from staging (~20 cm range) any residual
        # mount error shrinks proportionally. Opportunistic — the closed
        # gripper may occlude the tag; the centered fix then stands.
        got2 = wait_for_fix(node, watcher, cfg, timeout_s=2.5)
        if got2 is not None:
            cp2 = container_pose_from_tag(got2[0], got2[1], model, cfg.tag_offset)
            d = math.dist(cp2.xyz, ctx.cpose.xyz)
            if d > 0.05:
                print(
                    "[press_demo] close-range re-fix is %.3f m from the "
                    "centered fix — inconsistent, aborting to home" % d
                )
                runner.run(
                    [build_home_leg(ctx, client.joints())],
                    execute=args.execute,
                    assume_yes=True,
                )
                sys.exit(3)
            if d > 0.005:
                print(
                    "[press_demo] close-range re-fix shifts the target "
                    "%.1f mm — using it" % (d * 1000)
                )
            ctx.cpose = cp2
        else:
            print(
                "[press_demo] no close-range re-fix (gripper may occlude) — "
                "keeping the centered fix"
            )

        res = runner.run(
            build_press_legs(ctx, cfg), execute=args.execute, assume_yes=True
        )
        bad = [r for r in res if not r.ok]
        if bad:
            sys.exit(1)
        press = [r for r in res if r.leg_name.startswith("press")]
        print(
            "[press_demo] DONE — %s"
            % (press[-1].detail if press else "no press leg ran (dry-run)")
        )
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed


if __name__ == "__main__":
    main()
