"""The primitives (spec §5): plan/execute split, guarded descents shared.

Each primitive chain-plans from `state.joints` (tour_demo chaining). A
contact leg invalidates downstream pre-plans: it increments the chain and
the next primitive previews from the descent plan's end joints (nominal
contact depth); the Runner re-plans stale legs from the live arm at
execution time.

Ctx.last_pose / last_world track the most recent commanded tool pose and
the world it was planned against, so pose-relative primitives (Lift,
Retreat) need no pose argument of their own.
"""

from dataclasses import dataclass

from rammp_box_opening.constants import (
    CONTACT_SPEED,
    GRIPPER_CMD_CLOSED,
    GRIPPER_CMD_OPEN,
    HOME,
    TRANSIT_SPEED,
)
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.models.container import wrist_flat_quat
from rammp_box_opening.runtime.guards import (
    GuardSpec,
    check_standoff,
    in_band,
    press_outcome,
)
from rammp_box_opening.runtime.legs import Kind, Leg


@dataclass
class Ctx:
    model: object
    cpose: object
    client: object
    worlds: object
    lid_at: object = None  # set after Place(lid): later worlds carry the lid
    config_path: str = None
    last_pose: tuple = None  # (xyz, quat_xyzw) of the last commanded pose
    last_world: tuple = None  # (name, path) of the last interaction world


@dataclass
class PlanState:
    joints: list
    chain: int
    contact_broke_chain: bool


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
        "full", model=ctx.model, cpose=ctx.cpose, lid_at=ctx.lid_at, tag=tag
    )


def _interaction_world(ctx, target_xyz, contact_z, depth_max, tag):
    return ctx.worlds.push_name(
        "interaction",
        model=ctx.model,
        cpose=ctx.cpose,
        target_xyz=target_xyz,
        contact_z=contact_z,
        depth_max=depth_max,
        lid_at=ctx.lid_at,
        tag=tag,
    )


def _plan_motion(
    ctx, state, name, target, world, speed, guard=None, invalidates=False, verify=None
):
    world_name, world_path = world
    kind, *rest = target
    if kind == "pose":
        plan = ctx.client.plan_to_pose(rest[0], rest[1], state.joints)
        ctx.last_pose = (list(rest[0]), list(rest[1]))
    else:
        plan = ctx.client.plan_to_joints(rest[0], state.joints)
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
        invalidates_downstream=invalidates,
        verify=verify,
    )
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain, contact_broke_chain=invalidates)


def _gripper_leg(ctx, state, name, cmd, world, verify=None):
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
    )


class Approach:
    """Transit to a hover/staging pose — never to contact depth (spec §5)."""

    def __init__(self, target_xyz, quat, name="approach"):
        self.target_xyz = list(target_xyz)
        self.quat = list(quat)
        self.name = name

    def plan(self, ctx, state):
        world = _full_world(ctx)
        leg, state = _plan_motion(
            ctx,
            state,
            self.name,
            ("pose", self.target_xyz, self.quat),
            world,
            TRANSIT_SPEED,
        )
        return [leg], state


class Press:
    """Close the gripper, then a guarded descent onto the button (spec §5).

    Trip inside depth_window = pressed; before it = rim; untripped at
    depth_window.max = no click. All three classified by the leg verify."""

    def plan(self, ctx, state):
        m = ctx.model
        button = from_container(ctx.cpose, m.button_offset)
        quat = attitude_quat(m.press_attitude_rpy_deg, ctx.cpose.yaw)
        window = m.press_depth_window
        check_standoff(button[2] + m.hover_standoff, button[2])
        world = _interaction_world(ctx, button, button[2], window[1], "button")
        ctx.last_world = world
        close = _gripper_leg(ctx, state, "press:close", GRIPPER_CMD_CLOSED, world)
        target = [button[0], button[1], button[2] - window[1]]
        guard = GuardSpec(
            touch_nm=m.touch_nm,
            trip="press",
            depth_window=window,
            target_z=button[2],
        )

        def verify(v):
            return press_outcome(v.outcome, v.depth_m, window)

        descend, state = _plan_motion(
            ctx,
            state,
            "press:down",
            ("pose", target, quat),
            world,
            CONTACT_SPEED,
            guard=guard,
            invalidates=True,
            verify=verify,
        )
        return [close, descend], state


class Grasp:
    """Guarded descent (a trip = mispositioned strike = FAILURE), then a
    graded close verified against the expected grip band (spec §5)."""

    def __init__(self, spec, name="grasp"):
        self.spec = spec
        self.name = name

    def plan(self, ctx, state):
        m = ctx.model
        point = from_container(ctx.cpose, self.spec.offset)
        quat = attitude_quat(self.spec.attitude_rpy_deg, ctx.cpose.yaw)
        check_standoff(point[2] + m.hover_standoff, point[2])
        world = _interaction_world(ctx, point, point[2], 0.0, self.name)
        ctx.last_world = world
        guard = GuardSpec(touch_nm=m.touch_nm, trip="obstruction", target_z=point[2])
        descend, state = _plan_motion(
            ctx,
            state,
            self.name + ":down",
            ("pose", point, quat),
            world,
            CONTACT_SPEED,
            guard=guard,
        )
        close = _gripper_leg(
            ctx,
            state,
            self.name + ":close",
            m.width_to_command(self.spec.width_m),
            world,
            verify=band_verify(self.spec.expect_band),
        )
        return [descend, close], state


class Lift:
    """Planned ascent by dz; re-checks the grip band afterward (slip)."""

    def __init__(self, dz, band=None, name="lift"):
        self.dz = float(dz)
        self.band = band
        self.name = name

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
            CONTACT_SPEED,
            verify=verify,
        )
        return [leg], state


class Place:
    """Transit above the pose (+ lid-height margin — the planner cannot
    model a held object), guarded descent where a trip = set-down, then
    open the gripper (spec §5, §6)."""

    def __init__(self, target_xyz, quat, open_after=True, name="place"):
        self.target_xyz = list(target_xyz)
        self.quat = list(quat)
        self.open_after = open_after
        self.name = name

    def plan(self, ctx, state):
        m = ctx.model
        hover = hover_above(self.target_xyz, m.hover_standoff + m.lid_dims[2])
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
            ctx, self.target_xyz, self.target_xyz[2], 0.0, self.name
        )
        ctx.last_world = world
        guard = GuardSpec(
            touch_nm=m.touch_nm, trip="setdown", target_z=self.target_xyz[2]
        )
        descend, state = _plan_motion(
            ctx,
            state,
            self.name + ":down",
            ("pose", self.target_xyz, self.quat),
            world,
            CONTACT_SPEED,
            guard=guard,
            invalidates=True,
        )
        legs = [transit, descend]
        if self.open_after:
            legs.append(
                _gripper_leg(ctx, state, self.name + ":open", GRIPPER_CMD_OPEN, world)
            )
        return legs, state


class Retreat:
    """Vertical disengage by dz from the last commanded pose. Planned
    against the interaction world (a full-world plan would start inside
    the container cuboid after contact); slow and short. The Runner's
    reverse-retrace covers the plan-fails case (spec §6)."""

    def __init__(self, dz, name="retreat"):
        self.dz = float(dz)
        self.name = name

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
            CONTACT_SPEED,
        )
        return [leg], state


class PressFixed:
    """Owner-simplified press (2026-08-24): close, hover at cfg.hover_m,
    then ONE fixed-travel guarded stroke at cfg.press_speed.

    The guard is a STOP, not a classifier: a torque trip OR reaching the
    commanded depth both count as pressed — the report says which ended
    the stroke. check_standoff is deliberately not applied: its 0.051 m
    floor priced in +/-2 cm hand-measured z, while the tag pose's z is
    depth-refined to mm; the guard still arms in free air above the
    button. The hover waypoint is only plannable in the interaction
    world (in the full world it sits inside the err-tall container
    cuboid's padding), which is why staging (full world, cfg.staging_m)
    must precede this primitive."""

    def __init__(self, cfg, name="press"):
        self.cfg = cfg
        self.name = name

    def plan(self, ctx, state):
        m = ctx.model
        cfg = self.cfg
        button = from_container(ctx.cpose, m.button_offset)
        quat = attitude_quat(m.press_attitude_rpy_deg, ctx.cpose.yaw)
        world = _interaction_world(ctx, button, button[2], cfg.travel_m, "button")
        ctx.last_world = world
        close = _gripper_leg(
            ctx, state, self.name + ":close", GRIPPER_CMD_CLOSED, world
        )
        hover = [button[0], button[1], button[2] + cfg.hover_m]
        hover_leg, state = _plan_motion(
            ctx,
            state,
            self.name + ":hover",
            ("pose", hover, quat),
            world,
            CONTACT_SPEED,
        )
        guard = GuardSpec(
            touch_nm=m.touch_nm,
            trip="press",
            depth_window=(0.0, cfg.travel_m),
            target_z=button[2],
        )

        def verify(v):
            if v.outcome == "touch":
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
            self.name + ":down",
            ("pose", target, quat),
            world,
            cfg.press_speed,
            guard=guard,
            invalidates=True,
            verify=verify,
        )
        return [close, hover_leg, press], state


class Home:
    """Return to HOME joints via plan_to_joints (spec §5)."""

    def plan(self, ctx, state):
        world = _full_world(ctx)
        leg, state = _plan_motion(
            ctx, state, "home", ("joints", list(HOME)), world, TRANSIT_SPEED
        )
        return [leg], state
