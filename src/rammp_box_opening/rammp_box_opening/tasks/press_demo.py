"""press_demo — the camera-driven autonomous press-and-open (owner design
2026-08-24).

    ros2 launch rammp_box_opening press_demo.launch.py execute:=true
    ros2 run rammp_box_opening press_demo --execute

Flow, states logged one line each:

  SCAN    tool-down look pose over the bench (bench world with the
          unseen-container keep-out band); no flight when the arm is
          already parked tool-down there (open_box.park_tool_down).
  DETECT  the lid-top plateau above the measured table (depth_source),
          gated by the persistent OWL node's bbox (detect.source: vlm;
          the ladder in vlm.backends is walked only when depth alone has
          not answered in its first beat); origin z pinned to the
          calibrated table. No stable fix within detect.timeout_s ->
          home, exit 2. press:close is dispatched the moment the fix
          commits, overlapping the press plan.
  PRESS   ONE merged guarded stroke from the scan pose to travel_m below
          the lid plane, fast through free air and at press speed into
          contact (a trip near the expected contact or full travel =
          pressed; an early trip = honest strike failure). A scan pose
          too far off-axis falls back to a staged approach + stroke.
          The post-press retreat to the hop is LAZY (planned from live
          after the touch) and grip:open goes out on arrival there.
  GRIP    guarded descent to grip_clear_m above the lid around the
          popped knob (a trip = struck it), band-verified close, gentle
          lift to the carry height.
  PLACE   carry to the lid drop (configured lid_place, slid clear of the
          detected box when it has to be), guarded set-down at
          setdown_touch_nm (the trip IS the success), release, lazy
          retreat to the carry hover, home — or PARK when park_tool_down.

Each next phase is planned as a Runner lookahead while the previous
phase's last unguarded motion flies. --press-only stops after the press
(retreat to staging, home).

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
    FINGERTIP_FRAMES,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    PARK,
    REST_TOL_RAD,
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
from rammp_box_opening.primitives.core import (
    press_push,
    tcp_z,
    SETDOWN_OVERDRIVE_M,
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
    press_stroke,
)
from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.warp import warp_trajectory
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


def _state(joints):
    return PlanState(joints=list(joints), chain=0)


def rest_joints(cfg):
    """Where a run starts and ends: factory HOME, or PARK (tool-down at
    the scan pose) when open_box.park_tool_down is set."""
    return list(PARK if cfg.park_tool_down else HOME)


def rest_distance(live, joints):
    """Worst per-joint distance (wrap-aware) from a rest pose."""
    return max(abs(ang_diff(a, b)) for a, b in zip(live, joints))


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


def build_home_leg(ctx, start_joints, joints=None):
    """No-tag exit: back to HOME above the unseen-container band."""
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        "home",
        ("joints", list(HOME if joints is None else joints)),
        world,
        TRANSIT_SPEED,
    )
    return leg


def build_close_and_approach(ctx, cfg):
    """The staging approach. The gripper close is no longer a leg here:
    main() dispatches it the moment the fix commits (runner.start_gripper)
    so the fingers shut while the press is being PLANNED, and the Runner
    joins it before the guarded stroke as ever."""
    return [build_approach_leg(ctx, cfg)]


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
    # the TOUCH only: the push, retreat and what follows are built from the
    # contact the touch measures (build_push_legs)
    press_legs, st = PressFixed(cfg).plan(ctx, st)
    return list(press_legs)


def build_push_legs(ctx, cfg, contact_xyz, include_home=True):
    """After the touch: the bounded push from the measured contact, then
    the retreat, then home (press-only) or grip:open on arrival at the hop.

    Retreat at TRANSIT speed (owner: everything fast EXCEPT the press
    stroke). How FAR up depends on what comes next: the open-box tail
    re-descends immediately, so it stops at grip_hop_m; press-only
    continues to HOME, planned in the FULL world where the finger spheres
    need the taller staging clearance (staging_m 0.12 is the measured
    boundary, not a guess). The retreat is LAZY: the push may stop on a
    trip, so its start is unknown until then — the Runner plans it (and
    home) from live, once."""
    st = _state(ctx.client.joints())
    world = ctx.last_world or _full_world(ctx)
    push, st = press_push(ctx, st, cfg, contact_xyz, world)
    up_to = cfg.staging_m if include_home else cfg.grip_hop_m
    retreat_legs, st = Retreat(
        up_to + cfg.button_travel_m, speed=TRANSIT_SPEED, lazy=True
    ).plan(ctx, st)
    legs = [push, *retreat_legs]
    if include_home:
        home_legs, st = Home(rest_joints(cfg)).plan(ctx, st)
        legs += home_legs
    else:
        legs.append(_grip_open_after_retreat(ctx, st))
    return legs


def _grip_open_after_retreat(ctx, st):
    """grip:open dispatched on arrival at the hop, overlapping the grip
    phase's planning; the Runner joins it before the guarded grip:down.
    Never at the press bottom: the pads sit in the button recess there and
    the knob pops 15 mm — at the hop they are 35 mm above it."""
    world = ctx.last_world or _full_world(ctx)
    return _gripper_leg(ctx, st, "grip:open", GRIPPER_CMD_OPEN, world, defer_join=True)


# time fraction the warp's speed ramp needs to settle after the rebaseline
# before a guard may judge efforts against the new baseline
WARP_SETTLE_FRAC = 0.05
# a guard is never armed later than this fraction of its stroke
ARM_AFTER_CAP = 0.95


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
    # A warped guard OBSERVES the whole stroke but may only TRIP once the
    # slow-zone rebaseline has settled: in the fast segment the arm's own
    # dynamics swing the wrist torques by several Nm, and the 3 Nm touch
    # threshold tripped there at 42 % of the stroke — "struck something
    # above the button" with nothing struck (bench 2026-09-03). The
    # runner keeps this rule on every replan (_restore_execution_profile).
    # capped: a degenerate warp (no real slow zone) must still leave the
    # guard able to trip at the very end, never disable it outright
    leg.guard = replace(
        leg.guard,
        rebaseline_after=arm_frac,
        arm_after=min(
            max(leg.guard.arm_after or 0.0, arm_frac + WARP_SETTLE_FRAC), ARM_AFTER_CAP
        ),
    )
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


def build_merged_press_legs(ctx, cfg, include_home=False):  # include_home: kept for callers; the tail is build_push_legs'
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
    st = _state(ctx.client.joints())
    # the stroke STARTS at the last commanded pose (the scan pose) — read
    # it before planning overwrites last_pose with the press target, or
    # the contact expectation silently falls back to 0.9 (review 2026-09-02)
    here = ctx.last_pose[0] if ctx.last_pose else None
    # the one guarded stroke (core.press_stroke) with the merged caller's
    # own contact expectation: once the tool has covered all but the last
    # travel_m of the descent, re-timed AFTER the warp because warping
    # changes the time base
    press, st = press_stroke(ctx, st, cfg, "press:down", 0.06, 1.0)
    _apply_warp(press, cfg, cfg.press_speed)
    target_z = press.target[1][2]
    total = abs((here[2] - target_z)) if here else cfg.staging_m + cfg.travel_m
    press.contact_path_frac = (
        max(0.0, (total - cfg.travel_m)) / total if total > 0 else 0.9
    )
    press.retime(press.traj)
    # the TOUCH only: the push and everything after it are built from the
    # contact this stroke measures (build_push_legs)
    return [press]


def build_grip_legs(ctx, cfg, start_joints=None):
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
    # start_joints: the post-press retreat's end, when built as a lookahead
    # while that retreat flies (audit 2026-09-02)
    st = _state(ctx.client.joints() if start_joints is None else start_joints)
    # the fingers were opened on arrival at the hop (press phase); the
    # Runner joins that before the guarded descent below
    target = [
        button[0] + cfg.grip_offset_xy[0],
        button[1] + cfg.grip_offset_xy[1],
        tcp_z(button[2] + cfg.grip_clear_m),
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
    return [down, close, *lift_legs]


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


def build_place_legs(ctx, cfg, start_joints=None):
    """Carry the lid to the configured side spot, guarded set-down,
    release, retreat, home — the placed lid joins the collision world.
    start_joints: the lift's predicted end, when built as a lookahead
    while the lift flies (audit 2026-09-02)."""
    m = ctx.model
    # this builder OWNS lid_at: a discarded earlier build (a lookahead
    # thrown away by a no-motion retry) must not leave the transit and
    # set-down planning around a lid that is still in the gripper
    ctx.lid_at = None
    lid = ctx.lid_drop or load_lid_place(ctx.config_path)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(lid.xyz[1], lid.xyz[0]))
    # the fingers grip the knob grip_clear_m ABOVE the lid plane, so the
    # lid touches down when the TOOL is that much above lid-top height —
    # without this the stroke over-travels by grip_clear_m past contact
    # and crunches the lid into the table (field 2026-09-02, felt as
    # "pushes too hard" the moment grip_clear_m grew to 5 mm)
    target = [
        lid.xyz[0],
        lid.xyz[1],
        # the same fingertip correction as the grip: the lid hangs from
        # where the fingers took it, so both ends must shift together
        tcp_z(lid.xyz[2] + m.lid_dims[2] + cfg.grip_clear_m),
    ]
    st = _state(ctx.client.joints() if start_joints is None else start_joints)
    hover = Place.hover_for(ctx, target)
    legs, st = Place(
        target,
        quat,
        open_after=True,
        name="place:lid",
        speed=cfg.setdown_speed,
        touch_nm=cfg.setdown_touch_nm,
    ).plan(ctx, st)
    for lg in legs:
        if lg.name == "place:lid:down":
            _apply_warp(lg, cfg, cfg.setdown_speed)
            # arm no earlier than the slow zone: between arm_after and the
            # rebaseline the guard would judge slow-zone efforts against a
            # fast-regime baseline
            if lg.guard.rebaseline_after is not None:
                lg.guard = replace(
                    lg.guard,
                    arm_after=max(lg.guard.arm_after or 0.0, lg.guard.rebaseline_after),
                )
    ctx.lid_at = lid  # worlds carry the placed lid from here on
    # Retreat to the CARRY height, not 0.11 m: the transit hover sits
    # 37 mm higher (carry floor), and from the lower retreat end the arm's
    # spheres sit 18-21 mm from the padded container — inside its 20 mm
    # padding — so home was refused 12/12 draws at two of three bench
    # geometries (2026-09-02); from the hover it is valid 18/18. Lazy:
    # the set-down stops on a touch, so both legs plan from live, once,
    # as one group.
    down_z = target[2] - SETDOWN_OVERDRIVE_M
    retreat_legs, st = Retreat(hover[2] - down_z, speed=TRANSIT_SPEED, lazy=True).plan(
        ctx, st
    )
    home_legs, st = Home(rest_joints(cfg)).plan(ctx, st)
    return [*legs, *retreat_legs, *home_legs]




def predicted_contact(ctx):
    """Where the fingertip TF WOULD read if the touch met the button exactly
    at the camera's estimate — the offline stand-in for a measured contact
    (the tip-link origin sits TIP_TO_TOOL_M + TCP_OFFSET_M above the pad
    face)."""
    button = from_container(ctx.cpose, ctx.model.button_offset)
    return [button[0], button[1], button[2] + TCP_OFFSET_M + TIP_TO_TOOL_M]


def build_demo_legs(ctx, cfg):
    """The one-shot composition (offline tests); main runs it in phases,
    and its push starts from the contact the touch actually measured."""
    return [
        *build_close_and_approach(ctx, cfg),
        *build_press_legs(ctx, cfg, include_home=False),
        *build_push_legs(ctx, cfg, predicted_contact(ctx), include_home=False),
        *build_grip_legs(ctx, cfg),
        *build_place_legs(ctx, cfg),
    ]


def run_push(ctx, cfg, runner, args, touch):
    """The push stage from the touch's measured contact, then the retreat
    and what follows; the grip phase is planned while the retreat flies.
    Exits honestly when the arm could not measure its contact."""
    if not args.execute:
        # dry-run: the touch never ran, so there is no contact to push from
        return []
    contact = touch[-1].contact_xyz if touch else None
    if contact is None:
        try_home(
            ctx,
            runner,
            args.execute,
            "the touch tripped but no fingertip TF was readable — cannot "
            "bound the push (are %s in the tree?)" % (FINGERTIP_FRAMES[0],),
        )
        sys.exit(1)
    legs = build_push_legs(ctx, cfg, contact, include_home=args.press_only)
    res = runner.run(
        legs,
        execute=args.execute,
        assume_yes=True,
        lookahead=None
        if args.press_only
        else (lambda q: build_grip_legs(ctx, cfg, start_joints=q)),
    )
    if any(not r.ok for r in res):
        sys.exit(1)
    return res


def report_press_depth(runner, touch, press, ctx, model):
    """How deep the press actually went, measured by the arm.

    The touch's fingertip TF is the button's true top height in the frame
    the arm is commanded in — no camera, no dims.z, no TCP constant. It is
    printed against what the camera predicted, with the push's depth, and
    logged, so 'it pressed too low' is a number."""
    if not touch or touch[-1].contact_xyz is None:
        return
    tip_z = float(touch[-1].contact_xyz[2])
    surface = tip_z - TIP_TO_TOOL_M - TCP_OFFSET_M  # the pad face, base z
    predicted = from_container(ctx.cpose, model.button_offset)[2]
    pushed = None
    if press:
        end = press[-1].contact_xyz
        pushed = (tip_z - float(end[2])) * 1000 if end is not None else None
    runner.note(
        "press_depth",
        surface_z=round(surface, 4),
        predicted_button_z=round(predicted, 4),
        surface_vs_predicted_mm=round((surface - predicted) * 1000, 1),
        pushed_mm=None if pushed is None else round(pushed, 1),
        push_outcome=press[-1].outcome if press else None,
    )
    print(
        "[press_demo] button surface at z %.4f by touch — the camera predicted "
        "%.4f (%+.1f mm); push %s"
        % (
            surface,
            predicted,
            (surface - predicted) * 1000,
            "met a stop after %.1f mm" % pushed
            if pushed is not None
            else ("ran its full bound" if press else "did not run"),
        )
    )


def _spin_detect(node):
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except RuntimeError as e:
        # rclpy teardown artifact: our SIGINT handler raising inside a
        # subscription take surfaces as RuntimeError, not KeyboardInterrupt
        raise KeyboardInterrupt from e


def wait_for_fix(node, watcher, cfg, timeout_s=None):
    """Spin (the watcher ticks on its timer) until a fresh stable fix.

    The window is never purged here: the watcher keeps its in-flight
    samples — geometry is lifted with frame-stamp TF, the still-camera
    filter drops moving frames, the freshness window (1 s) means only
    the scan's parked tail can support a commit anyway, and the
    3-agreeing gate still stands — so a fix is often ready the moment
    the arm parks instead of half a second later (owner: detect during
    the flip, 2026-09-01)."""
    limit = cfg.timeout_s if timeout_s is None else timeout_s
    t0 = time.monotonic()
    watcher.active = True  # the detector only works inside a detect window
    try:
        while time.monotonic() - t0 < limit:
            _spin_detect(node)
            got = watcher.fix()
            if got is not None:
                return got
        return None
    finally:
        watcher.active = False


def try_home(ctx, runner, execute, why):
    """Recovery home that cannot crash the exit path: a refused home plan
    leaves the arm holding with an honest line (2026-08-25 review). Goes
    to the mission's rest pose (ctx.press_cfg, when main set it)."""
    print("[press_demo] %s — returning home" % why)
    cfg = getattr(ctx, "press_cfg", None)
    try:
        leg = build_home_leg(
            ctx, ctx.client.joints(), None if cfg is None else rest_joints(cfg)
        )
    except RuntimeError as e:
        print("[press_demo] home plan refused (%s) — arm holds" % e)
        return
    runner.run([leg], execute=execute, assume_yes=True)


def detect_only_report(node, watcher, ctx, cfg, runner, execute):
    """Mount-calibration observation: hold at the scan pose reporting every
    fresh fix (top-face centre, yaw, footprint), then park home.

    Place the container at a TAPE-MEASURED spot first; the printed base
    pose vs truth solves the wrist-mount error."""
    watcher.reset()
    watcher.active = True
    print("[press_demo] DETECT-ONLY: reporting fixes for 15 s")
    t0 = time.monotonic()
    last = None
    try:
        while time.monotonic() - t0 < 15.0:
            _spin_detect(node)
            got = watcher.fix()
            if got is None:
                continue
            pos, yaw = got
            key = tuple(round(float(v), 4) for v in pos)
            if key == last:
                continue
            last = key
            f = watcher.last_debug
            print(
                "fix: top [%.3f, %.3f, %.3f] yaw %.1f | footprint %.3fx%.3f m | %d px"
                % (
                    pos[0],
                    pos[1],
                    pos[2],
                    math.degrees(float(yaw)),
                    f.footprint[0],
                    f.footprint[1],
                    f.n_px,
                )
            )
    finally:
        watcher.active = False
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
    cli_common.apply_speed_scale(runner, args)
    # The persistent owl_detector node owns the OWL model; there is no
    # in-process copy (a second OWLv2 beside cuRobo on one GPU, field
    # 2026-09-01). The rung also owns the node's ENABLE gate: inference
    # runs only inside the mission's detect windows, because at 100 % GPU
    # duty it doubled every cuRobo solve (measured 2026-09-02).
    impls = {}
    owl = None
    watcher_holder = {}
    if cfg.detect_source == "vlm":
        from rammp_box_opening.perception.vlm_source import fetch_box_roi

        impls["claude"] = fetch_box_roi  # bounded at call time, below
        if "owl" in cfg.vlm_backends:
            from rammp_box_opening.perception.owl_source import make_topic_rung

            # listener starts NOW, not at detect time: the node sees the
            # box as the arm settles and its bbox gates the depth watcher,
            # so the fix is usually ready within a few still frames
            owl = make_topic_rung(node, cfg, watcher_holder)
            impls["owl"] = owl
    # the box found by geometry: lid plateau above the measured table.
    # No print to wear out — the press knuckles destroyed two tag prints
    # in a week (field 2026-09-01).
    watcher = BoxTopWatcher(node, cfg, model, worlds.table_top_z)
    watcher_holder["watcher"] = watcher
    # camera on from here to exit
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=worlds,
        config_path=cfg_path,
    )

    ctx.press_cfg = cfg  # recovery homes go to the mission's rest pose
    try:
        print("[press_demo] SCAN: tool-down look pose %s" % (list(cfg.scan_xyz),))
        live = client.joints()
        rests = [HOME] + ([PARK] if cfg.park_tool_down else [])
        worst = min(rest_distance(live, r) for r in rests)
        if worst > HOME_START_TOL_RAD:
            # every legitimate run starts near a rest pose; a distant start
            # means the previous run ended badly. Planning anything from
            # wreckage produced a half-inverted swing in the field
            # (2026-09-01) — refuse BEFORE any motion and name the recovery.
            print(
                "[press_demo] arm starts %.2f rad from any rest pose (tol %.1f) "
                "— refusing to plan from a failure pose. Recover first:\n"
                "    ros2 run rammp_box_opening home_arm --execute"
                % (worst, HOME_START_TOL_RAD)
            )
            sys.exit(3)
        if owl is not None:
            owl.enable()  # the node infers only while a detect window is open
        if cfg.park_tool_down and rest_distance(live, PARK) <= REST_TOL_RAD:
            # already parked tool-down at the scan pose: no scan flight.
            # The merged press needs the last COMMANDED pose to judge its
            # lateral rail — seed it with the pose PARK was planned to.
            bearing = math.atan2(cfg.scan_xyz[1], cfg.scan_xyz[0])
            ctx.last_pose = (list(cfg.scan_xyz), list(attitude_quat([180.0, 0.0, 0.0], bearing)))
            print("[press_demo] parked at the scan pose — no scan flight")
        else:
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

        # DEPTH FIRST, INSTANTLY. One box-sized plateau on the bench is
        # unambiguous geometry — with in-flight samples the fix is often
        # committed the moment the arm parks, and blocking on a semantic
        # model first cost 13 s in the field (2026-09-01: OWL score
        # dipped under threshold -> 5 s rung timeout -> 3.3 s Claude
        # network call -> commit). The ladder runs ONLY when depth
        # cannot answer alone (ambiguity, or nothing found in the first
        # beat) — semantics on demand.
        t_detect = time.monotonic()
        got = wait_for_fix(node, watcher, cfg, timeout_s=2.0)
        if got is None and cfg.detect_source == "vlm":
            while watcher.grab.color is None:
                _spin_detect(node)
            # the cloud rung is bounded to the detect time LEFT: on the
            # no-internet target an unbounded call stalled far past the
            # budget (review 2026-09-02)
            bound = dict(impls)
            if "claude" in bound:
                # budget read when the rung is CALLED (the owl rung may have
                # waited 2 s first), never frozen early
                bound["claude"] = lambda img, c: fetch_box_roi(
                    img,
                    c,
                    budget_s=cfg.timeout_s - (time.monotonic() - t_detect),
                )
            roi, lines = resolve_roi(watcher.grab.color, cfg, impls=bound)
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
        if owl is not None:
            owl.disable()  # detect window closed: give the GPU back
        if got is None:
            # benign detect timeout: park home, exit 2
            try_home(ctx, runner, args.execute, "NO BOX — %s" % watcher.status())
            sys.exit(2)

        pos, _yaw = got
        ctx.cpose = watcher.to_container_pose(got)
        print(
            "[press_demo] BOX at [%.3f, %.3f, %.3f] (%s) -> container origin "
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
        runner.note(
            "fix",
            top_xyz=[round(float(v), 4) for v in pos],
            origin_xyz=[round(float(v), 4) for v in ctx.cpose.xyz],
            yaw_deg=round(math.degrees(ctx.cpose.yaw), 2),
            top_residual_mm=round((pos[2] - (watcher.table_z + model.dims[2])) * 1000, 1),
            roi=list(watcher.roi) if watcher.roi is not None else None,
            status=watcher.status(),
            detect_s=round(time.monotonic() - t_detect, 2),
        )

        # the fingers shut NOW, while the press is planned — the join lands
        # before the guarded stroke, which needs them closed
        if not runner.start_gripper("press:close", GRIPPER_CMD_CLOSED, args.execute):
            # no closed fingers = no press: a 7 Nm stroke with an open
            # aperture strikes the lid and can still read "pressed"
            try_home(
                ctx,
                runner,
                args.execute,
                "press:close refused or failed — the fingers are not closed",
            )
            sys.exit(1)

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
        runner.note("press_plan", merged=bool(merged), lateral_mm=None if lateral is None else round(lateral * 1000, 1))
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
            res = runner.run(merged_legs, execute=args.execute, assume_yes=True)
            bad = [r for r in res if not r.ok]
            if bad:
                sys.exit(1)
            touch = [r for r in res if r.leg_name.startswith("press:down")]
            res = run_push(ctx, cfg, runner, args, touch)
            press = [r for r in res if r.leg_name.startswith("press:push")]
            print(
                "[press_demo] PRESSED — %s"
                % (press[-1].detail if press else "no push ran (dry-run)")
            )
            report_press_depth(runner, touch, press, ctx, model)
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
            # gripper may occlude the lid; the scan fix then stands.
            watcher.roi = None  # the scan-pose bbox is stale from here
            got2 = wait_for_fix(node, watcher, cfg, timeout_s=1.5)
            if got2 is not None:
                cp2 = watcher.to_container_pose(got2)
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
            touch = [r for r in res if r.leg_name.startswith("press")]
            res = run_push(ctx, cfg, runner, args, touch)
            press = [r for r in res if r.leg_name.startswith("press:push")]
            print(
                "[press_demo] PRESSED — %s"
                % (press[-1].detail if press else "no push ran (dry-run)")
            )
            report_press_depth(runner, touch, press, ctx, model)
        if args.press_only:
            runner.finish()
            sys.exit(0)

        # No pre-grip re-look: at the hop the lid does not fit in the depth
        # frame (its far edge projects past the last row), so the border
        # gate refused every attempt — 1.2 s per run for nothing (audit
        # 2026-09-02). A press that scoots the box is caught by the guarded
        # grip:down (an obstruction trip) and the band verify (closed on
        # air), both honest failures.
        print("[press_demo] GRIP: descend to press depth, close, pull")
        grip_legs = runner.lookahead_result or build_grip_legs(ctx, cfg)
        res = runner.run(
            grip_legs,
            execute=args.execute,
            assume_yes=True,
            # the place phase is planned while the lift flies
            lookahead=lambda q: build_place_legs(ctx, cfg, start_joints=q),
        )
        if any(not r.ok for r in res):
            sys.exit(1)  # grip failed (band miss / strike) — arm holds
        grip = [r for r in res if r.leg_name == "grip:close"]
        print("[press_demo] LID PULLED — %s" % (grip[-1].detail if grip else "dry-run"))

        print("[press_demo] PLACE: carrying the lid to the side spot")
        place_legs = runner.lookahead_result or build_place_legs(ctx, cfg)
        res = runner.run(place_legs, execute=args.execute, assume_yes=True)
        if any(not r.ok for r in res):
            sys.exit(1)
        runner.finish()
        print("[press_demo] DONE — box open, lid placed, arm home")
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed


if __name__ == "__main__":
    main()
