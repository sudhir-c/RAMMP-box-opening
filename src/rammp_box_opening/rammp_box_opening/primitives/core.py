"""The primitives (spec §5): plan/execute split, guarded descents shared.

Each primitive chain-plans from `state.joints` (tour_demo chaining). A
contact leg invalidates downstream pre-plans: it increments the chain, so
nothing merges across it, and what follows it is LAZY — planned by the
Runner from the live arm once the guard has stopped it.

Ctx.last_pose / last_world track the most recent commanded tool pose and
the world it was planned against, so pose-relative primitives (Lift,
Retreat) need no pose argument of their own.
"""

import math
import time
from dataclasses import dataclass

from rammp_box_opening.constants import (
    CONTACT_SPEED,
    GRIPPER_CMD_OPEN,
    HOME,
    TRANSIT_SPEED,
)
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.models.container import wrist_flat_quat
from rammp_box_opening.runtime.guards import (
    GuardSpec,
    in_band,
    time_fraction_at_path_fraction,
)
from rammp_box_opening.runtime.legs import Kind, Leg

# a set-down's success IS the guard trip — overdrive the commanded depth
# past nominal surface contact so the table is always felt (err-TALL world
# modeling can otherwise leave an exact-height target arriving untouched)
SETDOWN_OVERDRIVE_M = 0.005

# clearance between the CARRIED lid's underside and the container top
# during the place transit — the planner cannot model a held object
CARRY_CLEAR_M = 0.04

# extra container xy half-extent in FULL worlds planned after a contact
# leg: a press can scoot the box off its detected pose (field 2026-09-01,
# ~2 cm at 8.1 Nm) and a transit must not thread the needle beside a
# cuboid the box may no longer be inside
CONTACT_SHIFT_PAD_M = 0.03


@dataclass
class Ctx:
    model: object
    cpose: object
    client: object
    worlds: object
    lid_at: object = None  # set after Place(lid): later worlds carry the lid
    lid_drop: object = None  # runtime-resolved drop spot (adapts to the box)
    config_path: str = None
    last_pose: tuple = None  # (xyz, quat_xyzw) of the last commanded pose
    last_world: tuple = None  # (name, path) of the last interaction world
    contact_pad: float = 0.0  # container xy padding once contact has happened


@dataclass
class PlanState:
    joints: list  # predicted joints the next leg plans from; None = lazy
    chain: int  # bumped by every contact leg: nothing merges across it


def hover_above(xyz, standoff):
    return [xyz[0], xyz[1], xyz[2] + standoff]


def band_verify(band):
    """Grip-band check with an honest fallback when no gripper state exists."""

    def verify(ctx):
        if ctx.gripper_pos is None:
            return True, "band unchecked — no gripper state"
        ok = in_band(ctx.gripper_pos, band)
        return ok, "grip %.3f vs band %s" % (ctx.gripper_pos, list(band))

    return verify


def _full_world(ctx, tag=""):
    tag = tag or ("lid" if ctx.lid_at is not None else "")
    return ctx.worlds.push_name(
        "full",
        model=ctx.model,
        cpose=ctx.cpose,
        lid_at=ctx.lid_at,
        tag=tag,
        container_pad_xy=ctx.contact_pad,
    )


def _interaction_world(ctx, target_xyz, contact_z, depth_max, tag, ring=True):
    return ctx.worlds.push_name(
        "interaction",
        model=ctx.model,
        cpose=ctx.cpose,
        target_xyz=target_xyz,
        contact_z=contact_z,
        depth_max=depth_max,
        lid_at=ctx.lid_at,
        tag=tag,
        ring=ring,
    )


def _plan_motion(
    ctx,
    state,
    name,
    target,
    world,
    speed,
    guard=None,
    invalidates=False,
    verify=None,
    lazy=False,
):
    world_name, world_path = world
    kind, *rest = target
    if lazy or state.joints is None:
        # LAZY: this leg follows an expected touch, so its start is unknown
        # until the guard stops the arm. Pre-planning it was pure waste —
        # the post-touch replan discarded it every run (audit 2026-09-02).
        # The Runner plans it from live joints, once, in the leg's world.
        # Everything chained after it is lazy too (its end is unknown).
        if kind == "pose":
            ctx.last_pose = (list(rest[0]), list(rest[1]))
        leg = Leg(
            name=name,
            kind=Kind.MOTION,
            traj=None,
            speed=speed,
            guard=guard,
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=target,
            goal_joints=None,
            verify=verify,
        )
        next_chain = state.chain + 1 if invalidates else state.chain
        return leg, PlanState(joints=None, chain=next_chain)
    # Worlds are a PLAN-time concern (spec §6): the planner must hold this
    # leg's world BEFORE the plan is requested — SetWorld only at execution
    # time means every trajectory was actually planned against the previous
    # world (2026-08-24 review, critical). The client deduplicates pushes of
    # the world it already holds (one tracker, review 2026-09-02).
    ok, msg = ctx.client.set_world(world_path)
    if not ok:
        raise RuntimeError("set_world before planning %s failed: %s" % (name, msg))
    t_plan = time.monotonic()

    if kind == "pose":
        offset = float(rest[2]) if len(rest) > 2 else 0.0
        plan = ctx.client.plan_to_pose(
            rest[0], rest[1], state.joints, approach_offset_m=offset
        )
        ctx.last_pose = (list(rest[0]), list(rest[1]))
    else:
        plan = ctx.client.plan_to_joints(rest[0], state.joints)
    plan_s = time.monotonic() - t_plan
    if plan is None or not plan.success:
        raise RuntimeError(
            "planning failed for %s: %s"
            % (name, getattr(plan, "message", "no response"))
        )
    end = list(plan.trajectory.points[-1].positions)
    leg = Leg(
        name=name,
        kind=Kind.MOTION,
        traj=plan.trajectory,
        speed=speed,
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=target,
        goal_joints=end,
        verify=verify,
        plan_s=plan_s,
        plan_server_s=getattr(plan, "planning_time", None),
    )
    if guard is not None:
        # from here on the box may not be exactly where it was detected —
        # every later FULL world allows for a contact-shifted container
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain)


def _gripper_leg(
    ctx, state, name, cmd, world, verify=None, defer_join=False, join_before_motion=False
):
    world_name, world_path = world
    return Leg(
        name=name,
        kind=Kind.GRIPPER,
        traj=None,
        speed=0.0,
        guard=None,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=None,
        goal_joints=None,
        gripper_cmd=cmd,
        verify=verify,
        defer_join=defer_join,
        join_before_motion=join_before_motion,
    )


class Lift:
    """Planned ascent by dz; re-checks the grip band afterward (slip)."""

    def __init__(self, dz, band=None, name="lift", speed=CONTACT_SPEED):
        self.dz = float(dz)
        self.band = band
        self.name = name
        self.speed = float(speed)

    def plan(self, ctx, state):
        if ctx.last_pose is not None:
            xyz, quat = ctx.last_pose
        else:  # isolated CLI use: straight up from wherever the tool is
            xyz = ctx.client.tool_xyz() or [0.45, 0.0, 0.2]
            quat = wrist_flat_quat(xyz)
        target = [xyz[0], xyz[1], xyz[2] + self.dz]
        world = ctx.last_world or _full_world(ctx)
        verify = band_verify(self.band) if self.band is not None else None
        leg, state = _plan_motion(
            ctx,
            state,
            self.name,
            ("pose", target, list(quat)),
            world,
            self.speed,
            verify=verify,
        )
        return [leg], state


class Place:
    """Transit above the pose (+ lid-height margin — the planner cannot
    model a held object), guarded descent where a trip = set-down, then
    open the gripper (spec §5, §6)."""

    def __init__(
        self, target_xyz, quat, open_after=True, name="place", speed=None, touch_nm=None
    ):
        self.target_xyz = list(target_xyz)
        self.quat = list(quat)
        self.open_after = open_after
        self.name = name
        # descent speed; None keeps the conservative contact default for
        # callers that predate the config knob (tests, isolated CLI use)
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        # set-down trip threshold; None keeps the model's press threshold.
        # A lid touching a table loads the wrist far less than a press
        # pops a seal — at 7.0 the guard stayed blind through a 10 mm
        # crunch at the slid drop spot (field 2026-09-02).
        self.touch_nm = touch_nm

    @staticmethod
    def hover_for(ctx, target_xyz):
        """The carry pose above a set-down target: hover standoff plus a
        lid height, raised to the carry floor — the carried lid hangs a
        lid-height below the fingertips and the planner cannot see it, so
        the carry must clear the container body even directly overhead."""
        m = ctx.model
        hover = hover_above(list(target_xyz), m.hover_standoff + m.lid_dims[2])
        carry_floor = ctx.cpose.xyz[2] + m.dims[2] + m.lid_dims[2] + CARRY_CLEAR_M
        hover[2] = max(hover[2], carry_floor)
        return hover

    def plan(self, ctx, state):
        m = ctx.model
        hover = self.hover_for(ctx, self.target_xyz)
        full = _full_world(ctx)
        transit, state = _plan_motion(
            ctx,
            state,
            self.name + ":transit",
            ("pose", hover, self.quat),
            full,
            TRANSIT_SPEED,
        )
        world = _interaction_world(
            ctx,
            self.target_xyz,
            self.target_xyz[2],
            SETDOWN_OVERDRIVE_M,
            self.name,
            # ring walls collide with the gripper body at hover heights
            # (empirical 2026-08-25; bit the lid set-down's start state
            # live 2026-08-26 — INVALID_START_STATE at the hover)
            ring=False,
        )
        ctx.last_world = world
        guard = GuardSpec(
            touch_nm=m.touch_nm if self.touch_nm is None else float(self.touch_nm),
            trip="setdown",
            target_z=self.target_xyz[2],
            # contact is only possible at the stroke's very end — the
            # fast segment's dynamics must not trip the gentler set-down
            # threshold (lid released 110 mm up, field 2026-09-02)
            arm_after=0.5,
        )

        def verify(v):
            if v.outcome == "touch":
                if v.progress is not None and v.progress < 0.5:
                    return False, (
                        "guard tripped at %.0f%% of the descent — struck "
                        "something on the way down, set-down NOT confirmed"
                        % (v.progress * 100)
                    )
                peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
                return True, "surface felt%s — set down" % peak
            if v.outcome == "arrived":
                return False, (
                    "full stroke with no trip — never felt the surface, "
                    "set-down NOT confirmed"
                )
            return False, "set-down %s" % v.outcome
        # a set-down SUCCEEDS only on the touch: target exactly at surface
        # height can 'arrive' without ever feeling the table — command a
        # hair below so the guard verdict is deterministic
        down_xyz = [
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2] - SETDOWN_OVERDRIVE_M,
        ]
        descend, state = _plan_motion(
            ctx,
            state,
            self.name + ":down",
            # vertical final 50 mm: every free plan bows a little (the
            # "small arch", field 2026-09-01); a set-down comes straight
            # down onto its spot
            ("pose", down_xyz, self.quat, 0.05),
            world,
            self.speed,
            guard=guard,
            invalidates=True,
            verify=verify,
        )
        legs = [transit, descend]
        if self.open_after:
            legs.append(
                _gripper_leg(
                    ctx,
                    state,
                    self.name + ":open",
                    GRIPPER_CMD_OPEN,
                    world,
                    # dispatched at once; the retreat's post-touch replan
                    # runs while the fingers open and the join lands right
                    # before that retreat executes — a release still
                    # completes before the arm moves away (audit 2026-09-02)
                    defer_join=True,
                    join_before_motion=True,
                )
            )
        return legs, state


class Retreat:
    """Vertical disengage by dz from the last commanded pose. Planned
    against the interaction world (a full-world plan would start inside
    the container cuboid after contact). Lazy after a touch: the Runner
    plans it from live once the guard has stopped the arm, and a failed
    post-touch replan stops the mission with the arm holding — there is
    no plan-free fallback."""

    def __init__(self, dz, name="retreat", speed=CONTACT_SPEED, lazy=False):
        self.dz = float(dz)
        self.name = name
        self.speed = float(speed)
        self.lazy = lazy  # follows an expected touch: planned at execution

    def plan(self, ctx, state):
        if ctx.last_pose is None:
            raise RuntimeError("retreat needs a preceding pose-directed leg")
        xyz, quat = ctx.last_pose
        target = [xyz[0], xyz[1], xyz[2] + self.dz]
        world = ctx.last_world or _full_world(ctx)
        leg, state = _plan_motion(
            ctx,
            state,
            self.name,
            ("pose", target, list(quat)),
            world,
            self.speed,
            lazy=self.lazy,
        )
        return [leg], state


def press_stroke(ctx, state, cfg, name, approach_offset_m, contact_path_frac):
    """ONE guarded stroke to travel_m below the button: the press the
    mission runs, whether it starts at staging (PressFixed) or merged
    with the descent from the scan pose (press_demo). Returns (leg, state).

    The guard arms in free air during the descent. A trip counts as
    "pressed" only near where contact is EXPECTED — a trip well above the
    button means the stroke struck something else, and reports as the
    failure it is. Full travel with no trip also counts as pressed.

    `contact_path_frac` is where along the stroke's PATH contact is
    expected — but v.progress is a TIME fraction, and the two differ
    because cuRobo's profile is not constant-speed (live trips landed at
    0.826/0.835 against a 0.739 floor: 0.087 of margin, less than any
    velocity-profile change would move it). The conversion needs the
    trajectory, so it lives in leg.retime: called here on the planned
    one, by the Runner on every replan, and by a caller that re-times the
    stroke (a warp changes the time base) after setting
    leg.contact_path_frac to its own expectation.

    approach_offset_m constrains the final stretch VERTICAL: a diagonal
    descent touches the button before its lateral convergence finishes
    (10 mm off-centre at 20 mm height from a 199 mm start — the edge
    presses of 2026-09-01); the constrained plan measures 0.0-0.3 mm
    there, and contact happens travel_m above the goal, well inside it.
    """
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = attitude_quat(m.press_attitude_rpy_deg, math.atan2(button[1], button[0]))
    # ring=False: the aperture walls collide with the gripper body at
    # these heights (live IK_FAIL, margin probe 2026-08-25); the descent
    # comes from directly overhead
    world = _interaction_world(ctx, button, button[2], cfg.travel_m, "button", ring=False)
    ctx.last_world = world
    guard = GuardSpec(
        touch_nm=m.touch_nm,
        trip="press",
        depth_window=(0.0, cfg.travel_m),
        target_z=button[2],
    )
    expect = {"frac": 1.0}  # TIME fraction; set by retime below

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

    target = [button[0], button[1], button[2] - cfg.travel_m]
    press, state = _plan_motion(
        ctx,
        state,
        name,
        ("pose", target, quat, approach_offset_m),
        world,
        cfg.press_speed,
        guard=guard,
        invalidates=True,
        verify=verify,
    )

    def retime(traj):
        # a replan swaps the trajectory — the expected-contact fraction
        # must follow the one actually flown (review 2026-09-02)
        expect["frac"] = time_fraction_at_path_fraction(traj, press.contact_path_frac)

    press.contact_path_frac = float(contact_path_frac)
    press.retime = retime
    retime(press.traj)
    return press, state


class PressFixed:
    """Owner-simplified press v2 (2026-08-25): ONE guarded stroke from the
    staging pose straight to travel_m below the lid plane — the hover
    waypoint is gone (owner: fewer pauses; the gripper closes during the
    approach instead). Contact is expected after staging/(staging+travel)
    of the stroke's path (see press_stroke)."""

    def __init__(self, cfg, name="press"):
        self.cfg = cfg
        self.name = name

    def plan(self, ctx, state):
        cfg = self.cfg
        press, state = press_stroke(
            ctx,
            state,
            cfg,
            self.name + ":down",
            0.06,
            cfg.staging_m / (cfg.staging_m + cfg.travel_m),
        )
        return [press], state


class Home:
    """Return to the rest joints (factory HOME, or PARK when the mission
    rests tool-down) via plan_to_joints (spec §5). Lazy when it follows a
    lazy retreat (its start is unknown until then)."""

    def __init__(self, joints=None):
        self.joints = list(HOME if joints is None else joints)

    def plan(self, ctx, state):
        world = _full_world(ctx)
        leg, state = _plan_motion(
            ctx, state, "home", ("joints", list(self.joints)), world, TRANSIT_SPEED
        )
        return [leg], state
