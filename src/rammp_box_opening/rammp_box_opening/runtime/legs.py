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
    traj: object  # JointTrajectory | None
    speed: float
    guard: object  # GuardSpec | None
    world: str  # world name this leg was PLANNED against
    chain: int
    target: tuple  # ("pose", xyz, quat_xyzw) | ("joints", q7) | None
    goal_joints: list  # predicted end joints (MOTION), None for GRIPPER
    invalidates_downstream: bool = False
    verify: object = None  # Callable[[VerifyCtx], tuple[bool, str]] | None
    gripper_cmd: float = None
    world_path: str = None  # generated world YAML to push (SetWorld wants a path)
    stale: bool = field(default=False, compare=False)  # set by the Runner


def can_merge(a, b):
    return (
        a.kind is Kind.MOTION
        and b.kind is Kind.MOTION
        and a.chain == b.chain
        and a.speed == b.speed
        and a.guard is None
        and b.guard is None
        and a.verify is None  # a verify CLOSES its merge group
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
