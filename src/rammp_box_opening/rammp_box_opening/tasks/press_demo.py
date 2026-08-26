"""press_demo — the tag-driven autonomous press (owner design 2026-08-24).

    ros2 launch rammp_box_opening press_demo.launch.py execute:=true
    ros2 run rammp_box_opening press_demo --execute

Flow, states logged one line each: SCAN (tool-down look pose, bench-only
world) -> DETECT + center-the-tag servo -> CLOSE gripper + APPROACH
staging (one phase; full world with the tag-derived container cuboid) ->
close-range re-fix -> PRESS (ONE guarded stroke from staging to travel_m
below the tag plane; a trip near expected contact depth or full travel =
pressed, an early trip = honest strike failure) -> RETREAT -> HOME.

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


from rammp_box_opening.constants import GRIPPER_CMD_CLOSED, HOME, TRANSIT_SPEED
from rammp_box_opening.models.container import (
    ContainerModel,
    load_press_demo,
)
from rammp_box_opening.perception.tag_source import (
    TagWatcher,
    container_pose_from_tag,
    servo_step,
    to_camera,
)
from rammp_box_opening.primitives.core import (
    Ctx,
    Home,
    PlanState,
    PressFixed,
    Retreat,
    _full_world,
    _gripper_leg,
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


def build_close_and_approach(ctx, cfg):
    """Gripper close bundled with the staging approach (owner 2026-08-25:
    fewer pauses — the fingers close before/while the arm transits, not
    as a separate stop at the press site)."""
    close = _gripper_leg(
        ctx,
        _state(ctx.client.joints()),
        "press:close",
        GRIPPER_CMD_CLOSED,
        _full_world(ctx),
    )
    return [close, build_approach_leg(ctx, cfg)]


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
    # retreat at TRANSIT speed (owner: everything fast EXCEPT the press
    # stroke) — it still merges with home into one continuous motion
    retreat_legs, st = Retreat(
        cfg.staging_m + cfg.travel_m, speed=TRANSIT_SPEED
    ).plan(ctx, st)
    home_legs, st = Home().plan(ctx, st)
    return [*press_legs, *retreat_legs, *home_legs]


def build_demo_legs(ctx, cfg):
    """The one-shot composition (offline tests); main runs it in phases."""
    return [*build_close_and_approach(ctx, cfg), *build_press_legs(ctx, cfg)]


def build_servo_leg(ctx, cfg, disp, i):
    """One lateral centering translation at the current height.

    Anchored on the last COMMANDED pose (arrival-enforced by the runner),
    not live TF: this bringup's TF tree has no `tool_frame`, and the
    commanded pose is the more deterministic anchor anyway — each step
    re-observes, so small arrival error self-corrects (field 2026-08-25)."""
    if ctx.last_pose is None:
        return None
    tool = ctx.last_pose[0]
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


def center_on_tag(node, watcher, ctx, cfg, runner, execute, wait=None):
    """Owner design 2026-08-25: translate at scan height until the tag is
    at the image center, then press from what the CENTERED camera sees.

    Returns (fix, why): why is "ok" (centered, or unconverged-but-honest),
    "no_tag" (benign detect timeout -> caller homes, exit 2), or
    "servo_failed" (plan/exec/TF failure -> caller does NOT command more
    motion; the arm holds, like every other leg failure)."""
    wait = wait_for_fix if wait is None else wait
    prev_px = None
    for i in range(cfg.servo_max_iters + 1):
        got = wait(node, watcher, cfg)
        if got is None:
            return None, "no_tag"
        if watcher.last_debug is None:
            return None, "servo_failed"
        # judge centering on the MEDIAN fix the press will use, not the
        # last single sighting (2026-08-25 review)
        _p_last, rot_cam, trans_cam = watcher.last_debug
        p_med = to_camera(got[0], rot_cam, trans_cam)
        disp, px = servo_step(
            p_med,
            rot_cam,
            watcher.grab.k,
            cfg.servo_tol_px,
            cfg.servo_min_step_m,
            cfg.servo_max_step_m,
        )
        if disp is None:
            print("[press_demo] CENTERED — tag %.0f px off the optical axis" % px)
            return got, "ok"
        if prev_px is not None and px > prev_px + 20.0:
            # a correct servo shrinks the error every step; growth means
            # the camera frame is wrong (e.g. a flipped mount mirrors the
            # correction) — stop instead of walking away (field 2026-08-25)
            print(
                "[press_demo] servo DIVERGING (%.0f -> %.0f px) — camera "
                "frame is wrong, stopping" % (prev_px, px)
            )
            return None, "servo_failed"
        prev_px = px
        if i == cfg.servo_max_iters:
            print(
                "[press_demo] centering unconverged (%.0f px after %d moves) "
                "— pressing on the freshest fix" % (px, i)
            )
            return got, "ok"
        print(
            "[press_demo] SERVO %d: tag %.0f px off — shifting [%.3f, %.3f]"
            % (i + 1, px, disp[0], disp[1])
        )
        try:
            leg = build_servo_leg(ctx, cfg, disp, i + 1)
        except RuntimeError as e:
            print("[press_demo] servo plan refused: %s" % e)
            return None, "servo_failed"
        if leg is None:
            print("[press_demo] no commanded pose to anchor the servo move")
            return None, "servo_failed"
        res = runner.run([leg], execute=execute, assume_yes=True)
        if any(not r.ok for r in res):
            return None, "servo_failed"
    return None, "servo_failed"


def try_home(ctx, runner, execute, why):
    """Recovery home that cannot crash the exit path: a refused home plan
    leaves the arm holding with an honest line (2026-08-25 review)."""
    print("[press_demo] %s — returning home" % why)
    try:
        leg = build_home_leg(ctx, ctx.client.joints())
    except RuntimeError as e:
        print("[press_demo] home plan refused (%s) — arm holds" % e)
        return
    runner.run([leg], execute=execute, assume_yes=True)


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
        got, why = center_on_tag(node, watcher, ctx, cfg, runner, args.execute)
        if got is None:
            if why == "no_tag":
                try_home(
                    ctx,
                    runner,
                    args.execute,
                    "NO TAG — %s" % watcher.status(),
                )
                sys.exit(2)
            print("[press_demo] SERVO failed — arm holds (no blind homing)")
            sys.exit(1)

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
            build_close_and_approach(ctx, cfg), execute=args.execute, assume_yes=True
        )
        if any(not r.ok for r in res):
            sys.exit(1)

        # close-range re-fix: from staging (~20 cm range) any residual
        # mount error shrinks proportionally. Opportunistic — the closed
        # gripper may occlude the tag; the centered fix then stands.
        got2 = wait_for_fix(node, watcher, cfg, timeout_s=1.5)
        if got2 is not None:
            cp2 = container_pose_from_tag(got2[0], got2[1], model, cfg.tag_offset)
            d = math.dist(cp2.xyz, ctx.cpose.xyz)
            if d > 0.05:
                try_home(
                    ctx,
                    runner,
                    args.execute,
                    "close-range re-fix is %.3f m from the centered fix — "
                    "inconsistent" % d,
                )
                sys.exit(3)
            shift_xy = math.hypot(
                cp2.xyz[0] - ctx.cpose.xyz[0], cp2.xyz[1] - ctx.cpose.xyz[1]
            )
            ctx.cpose = cp2
            if shift_xy > 0.02:
                # a sub-2cm shift presses as a slightly diagonal stroke
                # (bounded, trivial over a 13 cm descent); beyond that,
                # re-approach above the NEW xy so the descent stays
                # overhead (2026-08-25 review + owner: fewer pauses)
                print(
                    "[press_demo] re-fix shifts the target %.1f mm laterally "
                    "— re-approaching overhead" % (shift_xy * 1000)
                )
                res = runner.run(
                    [build_approach_leg(ctx, cfg)],
                    execute=args.execute,
                    assume_yes=True,
                )
                if any(not r.ok for r in res):
                    sys.exit(1)
            elif d > 0.005:
                print(
                    "[press_demo] close-range re-fix shifts the target "
                    "%.1f mm — using it" % (d * 1000)
                )
        else:
            print(
                "[press_demo] no close-range re-fix (gripper may occlude) — "
                "keeping the centered fix"
            )

        print(
            "[press_demo] PRESS target origin [%.3f, %.3f, %.3f] yaw %.1f deg"
            % (
                ctx.cpose.xyz[0],
                ctx.cpose.xyz[1],
                ctx.cpose.xyz[2],
                math.degrees(ctx.cpose.yaw),
            )
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
