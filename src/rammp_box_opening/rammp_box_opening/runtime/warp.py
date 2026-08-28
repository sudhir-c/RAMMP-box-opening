"""Non-uniform time warping of a planned trajectory.

Every leg today executes under ONE speed scale (retime.py's exact time
dilation), so a guarded descent crawls at contact speed for its whole
length — `grip:down` spent 4.16 s covering 118 mm when only the last
centimetres are near anything. A person moves briskly and *decelerates
into* contact; that is most of what "not robotic" means here.

This module retimes a trajectory so it runs fast through the free-air
portion and slows into the final stretch. Three properties make it safe:

  - POSITIONS ARE UNTOUCHED. The path the planner validated is the path
    executed; only the timestamps and the velocities that annotate them
    change. The sanity gate and the collision plan still hold.
  - IT ONLY EVER SLOWS relative to the plan. Every scale is <= 1.0, so no
    executed velocity exceeds one cuRobo already limit-checked, and the
    executor's own velocity and continuity checks pass by construction
    (a longer dt only widens the allowed step).
  - THE TRANSITION IS RAMPED. A step change in scale would put a velocity
    discontinuity mid-trajectory — the exact jerk this is meant to remove.

The caller pairs the returned `arm_frac` with GuardSpec.rebaseline_after
so the torque guard re-captures its baseline once the arm is in the slow
regime: the guard stays armed the whole way (coverage is not reduced),
but its reference is taken in the regime where contact will happen,
instead of carrying a fast-motion baseline into a slow touch.
"""

import copy
import math

from trajectory_msgs.msg import JointTrajectory

RAMP_POINTS = 8  # over how many points the scale eases from fast to slow


def _cumulative_path(points):
    cum, total = [0.0], 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt(sum((x - y) ** 2 for x, y in zip(a.positions, b.positions)))
        cum.append(total)
    return cum, total


def warp_trajectory(traj, slow_frac, fast_scale, slow_scale):
    """Retime `traj` fast-then-slow. Returns (new_traj, arm_frac).

    slow_frac: the final fraction of the PATH run at slow_scale.
    arm_frac:  the TIME fraction of the result at which the slow zone
               begins — feed it to GuardSpec.rebaseline_after.
    """
    pts = list(traj.points)
    fast_scale, slow_scale = float(fast_scale), float(slow_scale)
    if len(pts) < 2 or not 0.0 < slow_frac < 1.0 or fast_scale <= slow_scale:
        return traj, None
    for s in (fast_scale, slow_scale):
        if not 0.0 < s <= 1.0:
            raise ValueError("warp scales must be in (0, 1]; got %r" % s)

    cum, total = _cumulative_path(pts)
    if total <= 0.0:
        return traj, None
    slow_starts = (1.0 - float(slow_frac)) * total

    # per-point scale, eased across RAMP_POINTS before the slow zone
    ramp_from = max(
        0, next(i for i, c in enumerate(cum) if c >= slow_starts) - RAMP_POINTS
    )
    scales = []
    for i, c in enumerate(cum):
        if c >= slow_starts:
            scales.append(slow_scale)
        elif i >= ramp_from:
            # cosine ease: no corner in the velocity profile
            u = (i - ramp_from) / float(max(1, RAMP_POINTS))
            e = 0.5 - 0.5 * math.cos(math.pi * min(1.0, u))
            scales.append(fast_scale + (slow_scale - fast_scale) * e)
        else:
            scales.append(fast_scale)

    def stamp(p):
        return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9

    out = JointTrajectory()
    out.joint_names = list(traj.joint_names)
    t_acc, arm_time = 0.0, None
    prev_t = 0.0
    for i, p in enumerate(pts):
        dt = stamp(p) - prev_t
        prev_t = stamp(p)
        t_acc += dt / scales[i] if dt > 0 else 0.0
        q = copy.deepcopy(p)
        if p.velocities:
            q.velocities = [float(v) * scales[i] for v in p.velocities]
        if p.accelerations:
            q.accelerations = [
                float(a) * scales[i] * scales[i] for a in p.accelerations
            ]
        q.time_from_start.sec = int(t_acc)
        q.time_from_start.nanosec = int(round((t_acc - int(t_acc)) * 1e9))
        out.points.append(q)
        if arm_time is None and cum[i] >= slow_starts:
            arm_time = t_acc

    total_t = t_acc
    arm_frac = None if (arm_time is None or total_t <= 0) else arm_time / total_t
    return out, arm_frac
