"""One speed profile for every unguarded motion — the human-motion pass.

What read as "robotic" (measured 2026-09-03): every leg started and ended
at rest (the planner pins terminal velocity), so a run made ~8 dead stops
where a person makes 3; and every leg was a symmetric bell uniformly slowed
to 0.75 — sluggish through the middle, still abrupt at the end.

This module re-times a joint path. POSITIONS ARE NEVER TOUCHED: the path
the planner validated is the path flown (the contract warp.py established
for guarded descents); only the timestamps change, and the velocities and
accelerations annotating them are recomputed from those timestamps.

The profile, along the path's joint-space arc length:
  - ease out of rest at `accel`, cruise at `cruise` x the per-joint velocity
    cap, ease into rest at `decel` — decel < accel, so the arrival is the
    long half (the precision signature of human reaching);
  - corners (direction changes along the path, including the junction
    between two chained legs) are taken at the speed their turning
    acceleration allows — a shallow corner flows, a sharp one slows, a
    reversal (> reversal_deg) is a genuine stop;
  - per-joint velocity never exceeds vmax_margin x the joint limit, so the
    executor's own gates (URDF velocity limits, per-interval continuity at
    3 x vmax x dt) hold by construction — and refuse the goal if not.

Guarded strokes are NOT re-timed here: their fast-then-slow warp and the
guard's rebaseline/arm fractions are tuned to contact and stay in warp.py.
"""

import math
from dataclasses import dataclass

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


@dataclass(frozen=True)
class RetimeParams:
    accel: float = 6.0  # rad/s^2 along the path: easing OUT of rest / a corner
    decel: float = 3.0  # rad/s^2 easing INTO rest / a corner (longer: precision)
    corner_accel: float = 4.0  # turning acceleration allowed through a corner
    reversal_deg: float = 150.0  # a direction change beyond this is a stop
    corner_window: float = 0.03  # rad of path on each side over which a turn is measured
    vmax_margin: float = 0.9  # per-joint velocity cap, fraction of the limit
    v_floor: float = 0.02  # rad/s: never divide by a smaller average speed
    smooth_pts: int = 4  # the profile's tops are rounded over this many samples
    first_dt: float = 0.02  # point 0 sits one step in the future (planner convention)
    # uniform time dilation applied AFTER profiling (1.0 = as profiled): the
    # operator's whole-run slow mode (--speed-scale), not a per-leg speed
    time_scale: float = 1.0


def _dedup(positions, tol=1e-9):
    """Drop consecutive duplicate samples (the planner's terminal hold, and
    the shared junction point when legs are chained)."""
    keep = [0]
    for k in range(1, len(positions)):
        if np.max(np.abs(positions[k] - positions[keep[-1]])) > tol:
            keep.append(k)
    return np.asarray(keep)


def concat_paths(trajs):
    """Chained trajectories -> (positions (N,dof), segment index per sample).
    Junction duplicates are dropped; the shared sample belongs to the
    EARLIER segment (its speed is the min of both, see cruise_per_sample)."""
    parts, seg = [], []
    for i, tr in enumerate(trajs):
        q = np.asarray([[float(v) for v in pt.positions] for pt in tr.points])
        parts.append(q)
        seg.extend([i] * len(q))
    q = np.vstack(parts)
    keep = _dedup(q)
    return q[keep], np.asarray(seg)[keep]


def velocity_caps(positions, vmax, margin):
    """Per-sample path-speed cap so that every joint stays under margin x
    its limit: on interval i, v <= vmax_j * ds_i / |dq_ij| for all j."""
    dq = np.diff(positions, axis=0)
    ds = np.linalg.norm(dq, axis=1)
    vmax = np.asarray(vmax, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint = np.where(np.abs(dq) > 0, vmax[None, :] * ds[:, None] / np.abs(dq), np.inf)
    cap_int = margin * per_joint.min(axis=1)
    cap = np.empty(len(positions))
    cap[0], cap[-1] = cap_int[0], cap_int[-1]
    cap[1:-1] = np.minimum(cap_int[:-1], cap_int[1:])
    return cap, ds


def corner_caps(positions, ds, params):
    """Per-sample speed cap from the direction change at that sample: the
    turning acceleration 2 v sin(theta/2) / (L / v) must stay under
    corner_accel over the corner's length L; a reversal is a stop.

    The turn is measured over a fixed PATH window on each side, not the
    two adjacent samples: a planner's samples crowd together near a leg's
    end (it decelerates to rest), so adjacent-sample directions there are
    noise and their length is microscopic — every junction read as a
    hairpin (found on the real retreat+home pair, 2026-09-03)."""
    n = len(positions)
    cap = np.full(n, np.inf)
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    w = params.corner_window
    rev = math.radians(params.reversal_deg)
    for k in range(1, n - 1):
        i = int(np.searchsorted(cum, cum[k] - w, side="right")) - 1
        i = max(0, min(i, k - 1))
        j = int(np.searchsorted(cum, cum[k] + w, side="left"))
        j = min(n - 1, max(j, k + 1))
        d_in = positions[k] - positions[i]
        d_out = positions[j] - positions[k]
        n_in, n_out = np.linalg.norm(d_in), np.linalg.norm(d_out)
        if n_in < 1e-9 or n_out < 1e-9:
            continue
        c = float(np.clip(np.dot(d_in, d_out) / (n_in * n_out), -1.0, 1.0))
        theta = math.acos(c)
        if theta >= rev:
            cap[k] = 0.0
            continue
        s = math.sin(theta / 2.0)
        if s < 1e-6:
            continue
        L = cum[j] - cum[i]
        cap[k] = math.sqrt(params.corner_accel * L / (2.0 * s))
    return cap


def cruise_per_sample(seg, speeds):
    """Segment speed (the leg's cruise fraction) per sample; the sample a
    junction shares takes the lower of the two."""
    speeds = np.asarray(speeds, dtype=float)
    cr = speeds[seg].astype(float)
    for k in range(1, len(seg)):
        if seg[k] != seg[k - 1]:
            cr[k - 1] = min(cr[k - 1], cr[k])
    return cr


def speed_profile(positions, cruise, vmax, params, v_start=0.0, v_end=0.0):
    """Path speed at every sample (rad/s of joint-space arc length)."""
    cap, ds = velocity_caps(positions, vmax, params.vmax_margin)
    target = np.minimum(np.asarray(cruise) * cap, corner_caps(positions, ds, params))
    n = len(positions)
    v = np.minimum(target, cap)
    v[0] = min(v_start, cap[0])
    v[-1] = min(v_end, cap[-1])
    for k in range(1, n):  # ease out: bounded acceleration along the path
        v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2.0 * params.accel * ds[k - 1]))
    for k in range(n - 2, -1, -1):  # ease in: bounded (gentler) deceleration
        v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2.0 * params.decel * ds[k]))
    if params.smooth_pts > 0 and n > 2 * params.smooth_pts + 2:
        # round the tops: the lower envelope of the profile and its moving
        # average keeps every cap (it only ever lowers) and pins the ends
        w = 2 * params.smooth_pts + 1
        avg = np.convolve(v, np.ones(w) / w, mode="same")
        inner = slice(params.smooth_pts, n - params.smooth_pts)
        v[inner] = np.minimum(v[inner], avg[inner])
        v[0], v[-1] = min(v_start, cap[0]), min(v_end, cap[-1])
    return v, ds


def times_from_profile(v, ds, params):
    dt = ds / np.maximum(0.5 * (v[:-1] + v[1:]), params.v_floor)
    dt = dt / max(float(params.time_scale), 1e-3)
    t = np.empty(len(v))
    t[0] = params.first_dt
    t[1:] = params.first_dt + np.cumsum(dt)
    return t


def to_message(joint_names, positions, t):
    """Positions as given; velocities/accelerations by central differences on
    the new time base; rest at both ends."""
    n = len(positions)
    vel = np.zeros_like(positions)
    if n > 2:
        vel[1:-1] = (positions[2:] - positions[:-2]) / (t[2:] - t[:-2])[:, None]
    acc = np.zeros_like(positions)
    if n > 2:
        acc[1:-1] = (vel[2:] - vel[:-2]) / (t[2:] - t[:-2])[:, None]
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in positions[k]]
        p.velocities = [float(x) for x in vel[k]]
        p.accelerations = [float(x) for x in acc[k]]
        p.time_from_start.sec = int(t[k])
        p.time_from_start.nanosec = int(round((t[k] - int(t[k])) * 1e9))
        msg.points.append(p)
    return msg


def retime_group(trajs, speeds, vmax, params=RetimeParams()):
    """Chained unguarded legs -> ONE re-timed JointTrajectory + a report.

    speeds: each leg's cruise fraction (its Leg.speed). Returns
    (trajectory, info) with info = {duration_s, junction_speeds (rad/s at
    each chained junction), stops (junctions taken at rest), n_points}.
    """
    positions, seg = concat_paths(trajs)
    if len(positions) < 2:
        return trajs[0], {"duration_s": 0.0, "junction_speeds": [], "stops": 0, "n_points": len(positions)}
    cruise = cruise_per_sample(seg, speeds)
    v, ds = speed_profile(positions, cruise, vmax, params)
    t = times_from_profile(v, ds, params)
    junctions = [k for k in range(1, len(seg)) if seg[k] != seg[k - 1]]
    jspeeds = [float(v[k - 1]) for k in junctions]
    info = {
        "duration_s": float(t[-1]),
        "junction_speeds": jspeeds,
        "stops": int(sum(1 for s in jspeeds if s < params.v_floor * 2)),
        "n_points": int(len(positions)),
    }
    return to_message(trajs[0].joint_names, positions, t), info


def check_like_executor(msg, vmax, continuity_slack=3.0):
    """The executor's own gates, mirrored: velocity limits, monotonic time,
    per-interval continuity. Returns a list of problems (empty = go)."""
    pos = np.asarray([[float(x) for x in p.positions] for p in msg.points])
    vel = np.asarray([[float(x) for x in p.velocities] for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    vmax = np.asarray(vmax, dtype=float)
    problems = []
    if (np.abs(vel).max(axis=0) > vmax * 1.01).any():
        problems.append("velocity exceeds limit")
    dts = np.diff(t, prepend=0.0)
    if (dts <= 0).any():
        problems.append("non-monotonic time_from_start")
    else:
        steps = np.abs(np.diff(pos, axis=0))
        if (steps > continuity_slack * vmax[None, :] * dts[1:, None]).any():
            problems.append("discontinuity")
    return problems
