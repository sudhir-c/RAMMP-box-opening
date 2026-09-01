"""press_demo — the camera-driven autonomous press (owner design 2026-08-24).

The container pose comes from a configurable source (detect.source):
depth (default — the lid plateau found by geometry, no print to wear
out) or tag (the original ArUco path).

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
from dataclasses import replace

import rclpy


from rammp_box_opening.constants import (
    GRIPPER_CMD_CLOSED,
    GRIPPER_CMD_OPEN,
    HOME,
    HOME_START_TOL_RAD,
    TRANSIT_SPEED,
)
from rammp_curobo.geometry import ang_diff
from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    load_lid_place,
    load_press_demo,
)
from rammp_box_opening.perception.depth_source import BoxTopWatcher
from rammp_box_opening.perception.vlm_source import resolve_roi
from rammp_box_opening.perception.tag_source import (
    TagWatcher,
    container_pose_from_tag,
    servo_step,
    to_camera,
)
from rammp_box_opening.primitives.core import (
    Ctx,
    Home,
    Lift,
    Place,
    PlanState,
    PressFixed,
    Retreat,
    _full_world,
    _gripper_leg,
    _interaction_world,
    _plan_motion,
    band_verify,
)
from rammp_box_opening.runtime.guards import (
    GuardSpec,
    time_fraction_at_path_fraction,
)
from rammp_box_opening.runtime.warp import warp_trajectory
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


def fix_to_cpose(watcher, got, model, cfg):
    """Fix -> ContainerPose, whichever source produced it. The depth
    watcher owns its conversion; the tag path keeps the existing free
    function (TagWatcher predates the model being available to it)."""
    if hasattr(watcher, "to_container_pose"):
        return watcher.to_container_pose(got)
    return container_pose_from_tag(got[0], got[1], model, cfg.tag_offset)


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
        # the fingers shut into free air while the arm transits to staging,
        # instead of the arm standing still for a full action round trip.
        # The Runner joins before the guarded press, which needs them shut.
        defer_join=True,
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


def build_press_legs(ctx, cfg, include_home=True):
    """Fixed stroke + retreat to staging, planned from live joints (run
    AFTER the approach so the close-range re-fix can update cpose).
    include_home=False when the open-box tail continues from staging."""
    st = _state(ctx.client.joints())
    press_legs, st = PressFixed(cfg).plan(ctx, st)
    # retreat at TRANSIT speed (owner: everything fast EXCEPT the press
    # stroke) — with home appended they merge into one continuous motion.
    #
    # How FAR up depends on what comes next. The open-box tail re-descends
    # immediately, so it stops at grip_hop_m and saves a 253 mm round trip
    # for a 2 mm reposition. press-only continues to HOME, which is planned
    # in the FULL world where the finger spheres need the taller staging
    # clearance (staging_m 0.12 is the measured boundary, not a guess).
    up_to = cfg.staging_m if include_home else cfg.grip_hop_m
    retreat_legs, st = Retreat(up_to + cfg.travel_m, speed=TRANSIT_SPEED).plan(ctx, st)
    legs = [*press_legs, *retreat_legs]
    if include_home:
        home_legs, st = Home().plan(ctx, st)
        legs += home_legs
    return legs


def _apply_warp(leg, cfg, slow_speed):
    """Run a guarded descent fast through free air and slow into contact.

    Positions are untouched; only the timing changes, and every scale is
    <= 1.0 so no executed velocity exceeds the plan's. The guard stays
    armed the whole way and re-baselines at the speed change, so contact
    is judged against a same-regime reference (spec §6 intent preserved).
    Returns the leg, warped in place, or unchanged when warping is off.
    """
    if not cfg.warp_fast_speed or cfg.warp_fast_speed <= slow_speed:
        return leg
    warped, arm_frac = warp_trajectory(
        leg.traj, cfg.warp_slow_frac, cfg.warp_fast_speed, slow_speed
    )
    if arm_frac is None:
        return leg
    leg.traj = warped
    leg.speed = 1.0  # the profile is baked in; do not dilate it again
    leg.warp = (cfg.warp_fast_speed, slow_speed, cfg.warp_slow_frac)
    leg.guard = replace(leg.guard, rebaseline_after=arm_frac)
    return leg


def merged_press_ok(ctx, cfg):
    """May the approach and the press become ONE motion?

    Only when the arm is already essentially above the button. The merged
    solve is planned in the REDUCED world (the container's top has to be
    absent, or there is no way to plan to a point inside it), so a long
    LATERAL run through that world would travel where the container is
    invisible. A near-vertical descent does not: it stays inside the
    column above the button, which is free by construction.

    Returns (ok, lateral_m). last_pose is the last COMMANDED tool pose —
    this TF tree has no tool_frame to ask (field 2026-08-25).
    """
    if not cfg.merge_press or ctx.last_pose is None:
        return False, None
    button = from_container(ctx.cpose, ctx.model.button_offset)
    here = ctx.last_pose[0]
    lateral = math.hypot(here[0] - button[0], here[1] - button[1])
    return lateral <= cfg.merge_press_max_lateral_m, lateral


def build_merged_press_legs(ctx, cfg, include_home=False):
    """Close the fingers, then ONE continuous descent to the button.

    Replaces [transit to staging] STOP [guarded press]. There is no seam
    and no splice: a single cuRobo solve is velocity-continuous by
    construction, and the time warp gives it the fast-then-slow profile
    that the two-leg version got from two different speed scales.

    What this gives up is the close-range re-fix, which used the staging
    STOP to re-measure the tag from ~20 cm. The press therefore leans
    entirely on the fix taken at scan height (owner: no recalibrating mid
    flight). If presses start landing off-centre, that is the reason.
    """
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(button[1], button[0]))
    world = _interaction_world(
        ctx, button, button[2], cfg.travel_m, "button", ring=False
    )
    ctx.last_world = world
    st = _state(ctx.client.joints())
    close = _gripper_leg(
        ctx, st, "press:close", GRIPPER_CMD_CLOSED, world, defer_join=True
    )
    guard = GuardSpec(
        touch_nm=m.touch_nm,
        trip="press",
        depth_window=(0.0, cfg.travel_m),
        target_z=button[2],
    )
    # contact is expected once the tool has covered all but the last
    # travel_m of the descent; converted to a TIME fraction below, after
    # the warp, because progress is elapsed/duration
    target = [button[0], button[1], button[2] - cfg.travel_m]
    expect = {"frac": 1.0}

    def verify(v):
        expected = expect["frac"]
        if v.outcome == "touch":
            if v.progress is not None and v.progress < expected - 0.15:
                return False, (
                    "guard tripped EARLY at %.0f%% of the stroke (contact "
                    "expected ~%.0f%%) — struck something above the button"
                    % (v.progress * 100, expected * 100)
                )
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "guard stopped the stroke%s — pressed" % peak
        if v.outcome == "arrived":
            return True, "full travel %.1f mm, no trip — pressed" % (
                cfg.travel_m * 1000
            )
        return False, "press %s" % v.outcome

    # the final 60 mm are constrained VERTICAL: a diagonal descent
    # touches the button before its lateral convergence finishes (10 mm
    # off-centre at 20 mm height from a 199 mm start — the edge presses
    # of 2026-09-01); the constrained plan measures 0.0-0.3 mm there
    press, st = _plan_motion(
        ctx,
        st,
        "press:down",
        ("pose", target, quat, 0.06),
        world,
        cfg.press_speed,
        guard=guard,
        invalidates=True,
        verify=verify,
    )
    _apply_warp(press, cfg, cfg.press_speed)
    # after the warp, because warping changes the time base
    here = ctx.last_pose[0] if ctx.last_pose else None
    total = abs((here[2] - target[2])) if here else cfg.staging_m + cfg.travel_m
    dist_frac = max(0.0, (total - cfg.travel_m)) / total if total > 0 else 0.9
    expect["frac"] = time_fraction_at_path_fraction(press.traj, dist_frac)
    # retreat height mirrors build_press_legs: the open-box tail re-descends
    # at once, so the hop suffices; press-only continues to HOME, planned in
    # the FULL world where the finger spheres need the staging clearance.
    up_to = cfg.staging_m if include_home else cfg.grip_hop_m
    retreat_legs, st = Retreat(up_to + cfg.travel_m, speed=TRANSIT_SPEED).plan(ctx, st)
    legs = [close, press, *retreat_legs]
    if include_home:
        home_legs, st = Home().plan(ctx, st)
        legs += home_legs
    return legs


def build_grip_legs(ctx, cfg):
    """Step 2: open the fingers, descend to just ABOVE the tag plane —
    around the now-popped button, grabbing it at its BASE — close on it
    (band-verified: 0.8 means closed on air), and slowly pull the lid.

    The tag plane IS the lid top surface (the tag sits on the flush
    button at scan time). Press depth is below it: the press compresses
    the sprung button, but open fingertips sent there hit solid lid and
    trip the guard (field 2026-08-26) — grip_clear_m keeps them above."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(button[1], button[0]))
    world = _interaction_world(
        ctx, button, button[2], cfg.travel_m, "button", ring=False
    )
    ctx.last_world = world
    st = _state(ctx.client.joints())
    open_leg = _gripper_leg(ctx, st, "grip:open", GRIPPER_CMD_OPEN, world)
    target = [
        button[0] + cfg.grip_offset_xy[0],
        button[1] + cfg.grip_offset_xy[1],
        button[2] + cfg.grip_clear_m,
    ]
    # obstruction semantics: a trip on the way down = the open fingers
    # STRUCK the knob/rim instead of straddling it — honest failure
    guard = GuardSpec(touch_nm=m.touch_nm, trip="obstruction", target_z=button[2])
    down, st = _plan_motion(
        ctx,
        st,
        "grip:down",
        # vertical final 40 mm: the fingers must straddle the knob from
        # straight above, not arrive on an arc (field 2026-09-01)
        ("pose", target, quat, 0.04),
        world,
        cfg.grip_speed,
        guard=guard,
        invalidates=True,
    )
    _apply_warp(down, cfg, cfg.grip_speed)
    close = _gripper_leg(
        ctx,
        st,
        "grip:close",
        GRIPPER_CMD_CLOSED,
        world,
        verify=band_verify(cfg.grip_band),
    )
    lift_legs, st = Lift(cfg.lift_m, band=cfg.grip_band, speed=cfg.lift_speed).plan(
        ctx, st
    )
    return [open_leg, down, close, *lift_legs]


DROP_X_M = (0.28, 0.65)  # set-down zone: inside the tool-down reach band
DROP_Y_M = (-0.42, 0.42)


def lid_place_min_clear(m):
    """Smallest planar container-origin-to-lid_place distance that leaves
    the place hover IK-solvable: both footprint half-diagonals plus
    gripper-body room. Field 2026-08-26: IK_FAIL at 0.073 m separation,
    clean plan at 0.162 m."""
    return (
        math.hypot(m.dims[0], m.dims[1]) + math.hypot(m.lid_dims[0], m.lid_dims[1])
    ) / 2 + 0.05


def resolve_lid_drop(m, cpose, lid_xyz):
    """Pick the actual drop spot: the configured lid_place when the
    DETECTED box clears it, else slid directly away from the box to the
    required clearance (the box lands wherever it lands — a fixed spot
    cannot assume the table around it is free; field 2026-08-26).
    Returns (xyz, shifted), or (None, False) when nothing in-zone clears."""
    need = lid_place_min_clear(m)
    bx, by = cpose.xyz[0], cpose.xyz[1]
    dx, dy = lid_xyz[0] - bx, lid_xyz[1] - by
    d = math.hypot(dx, dy)
    if d >= need:
        return list(lid_xyz), False
    dirs = [(dx / d, dy / d)] if d > 1e-6 else []
    dirs += [(0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)]
    for ux, uy in dirs:
        x = min(max(bx + ux * need, DROP_X_M[0]), DROP_X_M[1])
        y = min(max(by + uy * need, DROP_Y_M[0]), DROP_Y_M[1])
        if math.hypot(x - bx, y - by) >= need - 1e-9:
            return [x, y, lid_xyz[2]], True
    return None, False


def build_place_legs(ctx, cfg):
    """Carry the lid to the configured side spot, guarded set-down,
    release, retreat, home — the placed lid joins the collision world."""
    m = ctx.model
    lid = ctx.lid_drop or load_lid_place(ctx.config_path)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(lid.xyz[1], lid.xyz[0]))
    target = [lid.xyz[0], lid.xyz[1], lid.xyz[2] + m.lid_dims[2]]
    st = _state(ctx.client.joints())
    legs, st = Place(
        target, quat, open_after=True, name="place:lid", speed=cfg.setdown_speed
    ).plan(ctx, st)
    for lg in legs:
        if lg.name == "place:lid:down":
            _apply_warp(lg, cfg, cfg.setdown_speed)
    ctx.lid_at = lid  # worlds carry the placed lid from here on
    retreat_legs, st = Retreat(
        m.hover_standoff + m.lid_dims[2], speed=TRANSIT_SPEED
    ).plan(ctx, st)
    try:
        home_legs, st = Home().plan(ctx, st)
    except RuntimeError as e:
        # The retreat plans in the REDUCED world, so its end config can
        # be collision-free there yet read as inside the real (padded)
        # container in the full world — a rare family draw did exactly
        # that live (2026-09-01) and killed the mission at plan time
        # while the arm held the lid. The transit end is full-world
        # valid BY CONSTRUCTION and sits ~5 mm from the retreat end:
        # re-plan home from there, in its own group so it cannot merge
        # onto the retreat with a mismatched junction.
        print(
            "[press_demo] home from the retreat end refused (%s) — "
            "re-planning from the transit end" % e
        )
        home_state = PlanState(
            joints=list(legs[0].goal_joints),
            chain=st.chain + 1,
            contact_broke_chain=False,
        )
        home_legs, st = Home().plan(ctx, home_state)
    return [*legs, *retreat_legs, *home_legs]


def build_demo_legs(ctx, cfg):
    """The one-shot composition (offline tests); main runs it in phases."""
    return [
        *build_close_and_approach(ctx, cfg),
        *build_press_legs(ctx, cfg, include_home=False),
        *build_grip_legs(ctx, cfg),
        *build_place_legs(ctx, cfg),
    ]


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

    Whether the window is purged first is the SOURCE's policy
    (PURGE_ON_WAIT). The tag path purges: PnP orientation is fragile in
    motion. The depth path keeps its in-flight samples — geometry is
    lifted with frame-stamp TF, the freshness window (1 s) means only
    the scan's DECELERATION tail can support a commit anyway, and the
    3-agreeing gate still stands — so a fix is often ready the moment
    the arm parks instead of half a second later (owner: detect during
    the flip, 2026-09-01)."""
    if getattr(watcher, "PURGE_ON_WAIT", True):
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
        if isinstance(watcher, BoxTopWatcher):
            f = watcher.last_debug
            print(
                "fix: top [%.3f, %.3f, %.3f] yaw %.1f | footprint "
                "%.3fx%.3f m | %d px"
                % (
                    pos[0],
                    pos[1],
                    pos[2],
                    math.degrees(float(rot)),
                    f.footprint[0],
                    f.footprint[1],
                    f.n_px,
                )
            )
            continue
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
        "--press-only",
        action="store_true",
        help="stop after the press (the step-1 demo): press, retreat, home",
    )
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
    # NOTE: no in-process OWL preload here. The persistent owl_detector
    # node owns the model; the CLI's fallback copy loads lazily inside
    # owl_box_roi only when the node's topic does not answer. A boot-time
    # preload put TWO OWLv2 copies on the GPU beside cuRobo (field
    # 2026-09-01) for the price of zero — the node path never used it.
    impls = None
    if cfg.detect_source == "vlm" and "owl" in cfg.vlm_backends:
        from rammp_box_opening.perception.owl_source import make_topic_rung
        from rammp_box_opening.perception.vlm_source import fetch_box_roi

        # listener starts NOW, not at detect time: the node sees the box
        # mid-scan and its bbox gates the depth watcher while the arm is
        # still flipping over — by arrival the fix is usually already
        # committed (owner: detect during the flip, 2026-09-01)
        rung = make_topic_rung(node, cfg, watcher_holder := {})
        impls = {"owl": rung, "claude": fetch_box_roi}
    if cfg.detect_source in ("depth", "vlm"):
        # the box found by geometry: lid plateau above the measured table.
        # No print to wear out — the press knuckles destroyed two tag
        # prints in a week (field 2026-09-01).
        watcher = BoxTopWatcher(node, cfg, model, worlds.table_top_z)
        if impls is not None:
            watcher_holder["watcher"] = watcher
    else:
        watcher = TagWatcher(node, cfg)
    # camera on from here to exit
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=worlds,
        config_path=cfg_path,
    )

    try:
        print("[press_demo] SCAN: tool-down look pose %s" % (list(cfg.scan_xyz),))
        live = client.joints()
        worst = max(abs(ang_diff(a, b)) for a, b in zip(live, HOME))
        if worst > HOME_START_TOL_RAD:
            # every legitimate run starts near HOME; a distant start means
            # the previous run ended badly. Planning anything from wreckage
            # produced a half-inverted swing in the field (2026-09-01) —
            # refuse BEFORE any motion and name the recovery.
            print(
                "[press_demo] arm starts %.2f rad from HOME (tol %.1f) — "
                "refusing to plan from a failure pose. Recover first:\n"
                "    python3 ~/RAMMP-CuRobo/scripts/go_home.py --execute"
                % (worst, HOME_START_TOL_RAD)
            )
            sys.exit(3)
        res = runner.run(
            [build_scan_leg(ctx, cfg, live)],
            execute=args.execute,
            assume_yes=True,  # --execute alone arms the run (owner 2026-08-24)
        )
        if any(not r.ok for r in res):
            sys.exit(1)

        if args.detect_only:
            detect_only_report(node, watcher, ctx, cfg, runner, args.execute)
            sys.exit(0)

        noun = "BOX" if cfg.detect_source in ("depth", "vlm") else "TAG"
        if cfg.detect_source in ("depth", "vlm"):
            # DEPTH FIRST, INSTANTLY. One box-sized plateau on the bench
            # is unambiguous geometry — with in-flight samples the fix is
            # often committed the moment the arm parks, and blocking on a
            # semantic model first cost 13 s in the field (2026-09-01:
            # OWL score dipped under threshold -> 5 s rung timeout ->
            # 3.3 s Claude network call -> commit). The ladder now runs
            # ONLY when depth cannot answer alone (ambiguity, or nothing
            # found in the first beat) — semantics on demand.
            t_detect = time.monotonic()
            got = wait_for_fix(node, watcher, cfg, timeout_s=2.0)
            if got is None and cfg.detect_source == "vlm":
                while watcher.grab.color is None:
                    _spin_detect(node)
                roi, lines = resolve_roi(watcher.grab.color, cfg, impls=impls)
                for ln in lines:
                    print("[press_demo] VLM %s" % ln)
                watcher.roi = roi  # None = ungated, honest refusals stand
                remaining = cfg.timeout_s - (time.monotonic() - t_detect)
                if remaining > 0:
                    got = wait_for_fix(node, watcher, cfg, timeout_s=remaining)
            elif got is None:
                remaining = cfg.timeout_s - (time.monotonic() - t_detect)
                if remaining > 0:
                    got = wait_for_fix(node, watcher, cfg, timeout_s=remaining)
            why = "ok" if got is not None else "no_tag"
        else:
            got, why = center_on_tag(node, watcher, ctx, cfg, runner, args.execute)
        if got is None:
            if why == "no_tag":
                try_home(
                    ctx,
                    runner,
                    args.execute,
                    "NO %s — %s" % (noun, watcher.status()),
                )
                sys.exit(2)
            print("[press_demo] SERVO failed — arm holds (no blind homing)")
            sys.exit(1)

        pos, _rot = got
        ctx.cpose = fix_to_cpose(watcher, got, model, cfg)
        print(
            "[press_demo] %s at [%.3f, %.3f, %.3f] (%s) -> container origin "
            "[%.3f, %.3f, %.3f] yaw %.1f deg"
            % (
                noun,
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

        if not args.press_only:
            # the box lands wherever it lands: resolve the drop spot NOW,
            # before any container-directed motion — configured lid_place
            # when clear, slid away from the box when crowded, refusal
            # only when nothing in the set-down zone clears
            lid = load_lid_place(ctx.config_path)
            drop, shifted = resolve_lid_drop(model, ctx.cpose, lid.xyz)
            if drop is None:
                try_home(
                    ctx,
                    runner,
                    args.execute,
                    "no lid drop spot clears the box at [%.2f, %.2f] "
                    "(need %.0f mm, inside x %s y %s) — move the box"
                    % (
                        ctx.cpose.xyz[0],
                        ctx.cpose.xyz[1],
                        lid_place_min_clear(model) * 1000,
                        list(DROP_X_M),
                        list(DROP_Y_M),
                    ),
                )
                sys.exit(4)
            if shifted:
                print(
                    "[press_demo] drop spot [%.2f, %.2f] is only %.0f mm "
                    "from the box — sliding it to [%.2f, %.2f]"
                    % (
                        lid.xyz[0],
                        lid.xyz[1],
                        math.hypot(
                            lid.xyz[0] - ctx.cpose.xyz[0],
                            lid.xyz[1] - ctx.cpose.xyz[1],
                        )
                        * 1000,
                        drop[0],
                        drop[1],
                    )
                )
            ctx.lid_drop = ContainerPose(xyz=tuple(drop), yaw=lid.yaw)

        # ONE continuous motion when the arm is already above the button:
        # close the fingers, then descend straight to contact with no stop
        # at staging and no close-range re-fix (owner: constant motion, no
        # recalibrating mid flight). Falls back to the two-leg path — with
        # its stop and its re-fix — whenever the guard-rail says the run
        # through the reduced world would be too lateral.
        merged, lateral = merged_press_ok(ctx, cfg)
        declined_why = None
        merged_legs = None
        if merged:
            # a plan failure here must FALL BACK, not crash: the staged
            # path still exists and still works (field 2026-09-01)
            try:
                merged_legs = build_merged_press_legs(
                    ctx, cfg, include_home=args.press_only
                )
            except RuntimeError as e:
                declined_why = "plan failed: %s" % e
                merged = False
        elif cfg.merge_press:
            declined_why = (
                "no commanded pose yet"
                if lateral is None
                else "%.0f mm off-axis > %.0f mm limit"
                % (lateral * 1000, cfg.merge_press_max_lateral_m * 1000)
            )
        if merged:
            print(
                "[press_demo] MERGED PRESS — one motion to the button "
                "(%.0f mm off-axis, limit %.0f mm; no staging stop, no re-fix)"
                % (lateral * 1000, cfg.merge_press_max_lateral_m * 1000)
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
                merged_legs,
                execute=args.execute,
                assume_yes=True,
            )
            bad = [r for r in res if not r.ok]
            if bad:
                sys.exit(1)
            press = [r for r in res if r.leg_name.startswith("press:down")]
            print(
                "[press_demo] PRESSED — %s"
                % (press[-1].detail if press else "no press leg ran (dry-run)")
            )
        else:
            if declined_why is not None:
                print(
                    "[press_demo] merged press declined (%s) — using the "
                    "staged approach" % declined_why
                )
            res = runner.run(
                build_close_and_approach(ctx, cfg),
                execute=args.execute,
                assume_yes=True,
            )
            if any(not r.ok for r in res):
                sys.exit(1)

            # close-range re-fix: from staging (~20 cm range) any residual
            # mount error shrinks proportionally. Opportunistic — the closed
            # gripper may occlude the tag; the centered fix then stands.
            if hasattr(watcher, "roi"):
                watcher.roi = None  # the scan-pose bbox is stale from here
            got2 = wait_for_fix(node, watcher, cfg, timeout_s=1.5)
            if got2 is not None:
                cp2 = fix_to_cpose(watcher, got2, model, cfg)
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
                build_press_legs(ctx, cfg, include_home=args.press_only),
                execute=args.execute,
                assume_yes=True,
            )
            bad = [r for r in res if not r.ok]
            if bad:
                sys.exit(1)
            press = [r for r in res if r.leg_name.startswith("press")]
            print(
                "[press_demo] PRESSED — %s"
                % (press[-1].detail if press else "no press leg ran (dry-run)")
            )
        if args.press_only:
            sys.exit(0)

        # The press can scoot the box (any rim contact shoves it — both
        # 2026-09-01 air-grabs descended onto the PRE-press position). So
        # the re-look before the grip is MANDATORY, not a peek: purge the
        # window and wait briefly for a fix taken from the hop pose — the
        # camera sits ~0.18 m over the lid there and the top fits the
        # view. No fresh fix within the budget = the scan fix stands.
        if hasattr(watcher, "roi"):
            watcher.roi = None  # scan-pose bbox is stale here
        watcher.reset()  # only hop-pose sightings may re-aim the grip
        got3 = wait_for_fix(node, watcher, cfg, timeout_s=1.2)
        if got3 is None:
            print(
                "[press_demo] pre-grip re-look found nothing (%s) — gripping "
                "the scan fix" % watcher.status()
            )
        if got3 is not None:
            cp3 = fix_to_cpose(watcher, got3, model, cfg)
            d3 = math.hypot(
                cp3.xyz[0] - ctx.cpose.xyz[0], cp3.xyz[1] - ctx.cpose.xyz[1]
            )
            if d3 < 0.08:
                if d3 > 0.003:
                    print(
                        "[press_demo] pre-grip re-fix: box moved %.1f mm — "
                        "gripping where it is NOW" % (d3 * 1000)
                    )
                ctx.cpose = cp3  # z re-measured too — grip height anchors
                # to the CURRENT lid, not the scan-time estimate
        print("[press_demo] GRIP: open, descend to press depth, close, pull")
        res = runner.run(
            build_grip_legs(ctx, cfg), execute=args.execute, assume_yes=True
        )
        if any(not r.ok for r in res):
            sys.exit(1)  # grip failed (band miss / strike) — arm holds
        grip = [r for r in res if r.leg_name == "grip:close"]
        print("[press_demo] LID PULLED — %s" % (grip[-1].detail if grip else "dry-run"))

        print("[press_demo] PLACE: carrying the lid to the side spot")
        res = runner.run(
            build_place_legs(ctx, cfg), execute=args.execute, assume_yes=True
        )
        if any(not r.ok for r in res):
            sys.exit(1)
        print("[press_demo] DONE — box open, lid placed, arm home")
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed


if __name__ == "__main__":
    main()
