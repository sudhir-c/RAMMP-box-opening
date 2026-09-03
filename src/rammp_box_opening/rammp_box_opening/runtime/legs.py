"""Legs (the unit of planning/execution) and the merge rules (spec §5, §6).

MOTION legs merge into one trajectory only when dynamically valid: same
planning chain (B planned from A's predicted endpoint), same speed, no
guard on either, and a verify closes its group. Guarded legs always
execute alone. `world` is a checked precondition, never a merge key.
"""

from dataclasses import dataclass, field
from enum import Enum

from trajectory_msgs.msg import JointTrajectory


class Kind(Enum):
    MOTION = "motion"
    GRIPPER = "gripper"


@dataclass
class VerifyCtx:
    outcome: str
    depth_m: float = None
    gripper_pos: float = None
    progress: float = None
    torque_peak: float = None


@dataclass
class Leg:
    name: str
    kind: Kind
    # JointTrajectory, or None for a LAZY motion leg: one that follows an
    # expected touch, whose start is unknown until the guard stops the arm
    # — the Runner plans it from live joints exactly once, at execution
    traj: object
    speed: float
    guard: object  # GuardSpec | None
    world: str  # world name this leg was PLANNED against
    chain: int
    target: tuple  # ("pose", xyz, quat_xyzw) | ("joints", q7) | None
    goal_joints: list  # predicted end joints (MOTION), None for GRIPPER
    verify: object = None  # Callable[[VerifyCtx], tuple[bool, str]] | None
    gripper_cmd: float = None
    # GRIPPER legs only: start the command and carry on, joining before
    # anything that needs the fingers to have arrived. Set it where the
    # overlap is SAFE — a close during a transit — never on a release,
    # which must complete before the arm moves away from what it dropped.
    defer_join: bool = False
    # GRIPPER legs only, with defer_join: a RELEASE may be dispatched now
    # and the next motion's replan may proceed, but the arm must not move
    # away from what it dropped before the fingers have settled — the
    # Runner joins it right before that motion executes
    join_before_motion: bool = False
    world_path: str = None  # generated world YAML to push (SetWorld wants a path)
    # Planning cost, for the preview table. plan_s is the client's round
    # trip; plan_server_s is what the planner reports it spent solving.
    # The gap between them is action/transport overhead — worth watching:
    # off-bench the solve measures ~0.21 s while live runs showed ~1.0 s
    # per plan, and only these two numbers side by side say which half.
    plan_s: float = field(default=None, compare=False)
    plan_server_s: float = field(default=None, compare=False)
    # (fast_scale, slow_scale, slow_path_fraction) when a guarded descent
    # was time-warped; display only — leg.speed is 1.0 once it is baked in
    warp: tuple = field(default=None, compare=False)


def can_merge(a, b):
    return (
        a.kind is Kind.MOTION
        and b.kind is Kind.MOTION
        and a.chain == b.chain
        # speeds may differ: the re-timer builds ONE profile for the group,
        # each leg's speed becoming its cruise fraction (retime.py)
        # a planned lead never merges with a lazy tail (or vice versa):
        # merge_trajectories cannot chain a trajectory that does not exist
        and (a.traj is None) == (b.traj is None)
        and a.guard is None
        and b.guard is None
        and a.verify is None  # a verify CLOSES its merge group
        # ...and a verify cannot be ABSORBED into one either: only the
        # execution's owning member is verified, so appending a
        # verify-carrying leg as a non-lead member silently discarded its
        # check (review 2026-08-28 — latent, never yet triggered because
        # every verify-carrying MOTION leg today also carries a guard).
        and b.verify is None
    )


def merge_groups(legs):
    groups = []
    for leg in legs:
        if groups and can_merge(groups[-1][-1], leg):
            groups[-1].append(leg)
        else:
            groups.append([leg])
    return groups


def merge_trajectories(trajs):
    """Chained per-segment trajectories -> ONE continuous JointTrajectory.

    tour_demo.py pattern: zero controller goal transitions is the
    no-motion-fault mitigation. Callers guarantee chaining validity
    (merge_groups)."""
    merged = JointTrajectory()
    merged.joint_names = list(trajs[0].joint_names)
    offset = 0.0
    for traj in trajs:
        for pt in traj.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 + offset
            q = type(pt)()
            q.positions = list(pt.positions)
            q.velocities = list(pt.velocities)
            q.accelerations = list(pt.accelerations)
            q.time_from_start.sec = int(t)
            q.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            merged.points.append(q)
        last = traj.points[-1].time_from_start
        offset += last.sec + last.nanosec * 1e-9
    return merged
