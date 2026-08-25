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


def build_demo_legs(ctx, cfg):
    """Post-detection sequence: staging approach -> PressFixed -> retreat
    to staging height -> HOME. All poses tool-down at the tag's yaw."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    # bearing-steered attitude, same as PressFixed: press is yaw-invariant
    # and the bearing family is the reach-map-certified one (2026-08-25)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(button[1], button[0]))
    st = _state(ctx.client.joints())
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    approach, st = _plan_motion(
        ctx,
        st,
        "approach:staging",
        ("pose", staging, quat),
        _full_world(ctx),
        TRANSIT_SPEED,
    )
    press_legs, st = PressFixed(cfg).plan(ctx, st)
    # from the press bottom back up to staging height
    retreat_legs, st = Retreat(cfg.staging_m + cfg.travel_m).plan(ctx, st)
    home_legs, st = Home().plan(ctx, st)
    return [approach, *press_legs, *retreat_legs, *home_legs]


def wait_for_fix(node, watcher, cfg):
    """Spin (the watcher ticks on its timer) until a fresh stable fix.

    The window is purged first: sightings gathered while the arm was
    still moving carry TF/depth timing skew — only parked-camera frames
    may commit the fix the press will trust."""
    watcher.reset()
    t0 = time.monotonic()
    while time.monotonic() - t0 < cfg.timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        got = watcher.fix()
        if got is not None:
            return got
    return None


def main():
    ap = cli_common.make_parser(__doc__)
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

        print(
            "[press_demo] DETECT: waiting %.0f s for a stable fix (tag id %d)"
            % (cfg.timeout_s, cfg.tag_id)
        )
        got = wait_for_fix(node, watcher, cfg)
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

        legs = build_demo_legs(ctx, cfg)
        res = runner.run(legs, execute=args.execute, assume_yes=True)
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
