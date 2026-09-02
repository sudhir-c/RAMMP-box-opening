"""Contact guard, trajectory sanity gate, and guarded-descent bookkeeping.

TorqueGuard is the palm-demo pattern with the spec §6 hardening: the
baseline is anchored at the first ExecuteTrajectory feedback with
progress > 0 (never at goal-accept), and guarded runs REFUSE to start
without effort fields (enforced by the Runner, which owns the streams).
"""

import copy
import math
from dataclasses import dataclass

from trajectory_msgs.msg import JointTrajectory

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import (
    BASELINE_TRAVEL_M,
    POSE_UNCERTAINTY_M,
    TIP_BIAS_M,
)


class TorqueGuard:
    def __init__(self, touch_nm, rebaseline_after=None, arm_after=None):
        self.touch_nm = float(touch_nm)
        self.armed = False
        self._baseline = None
        self.peak = 0.0
        # Progress fraction before which deviations are OBSERVED but never
        # trip: a set-down's contact physically cannot happen in the first
        # half of a stroke that ends 5 mm below the surface, yet the fast
        # warp segment's motion dynamics tripped a 4.0 Nm threshold 64 ms
        # in and the lid was released 110 mm up (field 2026-09-02).
        self.arm_after = None if arm_after is None else float(arm_after)
        self._progress = 0.0
        # Time fraction at which a warped descent changes speed. The guard
        # stays ARMED throughout — coverage is not reduced — but its
        # reference is re-taken once the arm is in the slow regime, so the
        # dynamic-torque shift from decelerating is not mistaken for
        # contact, and the touch is judged against a same-regime baseline.
        self.rebaseline_after = (
            None if rebaseline_after is None else float(rebaseline_after)
        )
        self._rebaselined = False

    def on_progress(self, progress):
        self._progress = float(progress)
        if progress > 0.0:
            self.armed = True
        if (
            self.rebaseline_after is not None
            and not self._rebaselined
            and progress >= self.rebaseline_after
        ):
            self._rebaselined = True
            self._baseline = None  # re-captured on the next efforts reading

    def on_efforts(self, wrist_efforts):
        if not self.armed or wrist_efforts is None:
            return False
        if self._baseline is None:
            self._baseline = [float(v) for v in wrist_efforts]
            return False
        dev = max(abs(a - b) for a, b in zip(wrist_efforts, self._baseline))
        self.peak = max(self.peak, dev)
        if self.arm_after is not None and self._progress < self.arm_after:
            return False
        return dev > self.touch_nm


@dataclass(frozen=True)
class GuardSpec:
    touch_nm: float
    trip: str  # "press" | "obstruction" | "setdown"
    depth_window: tuple = None  # "press" only, m below nominal contact z
    target_z: float = None  # nominal contact z (base_link) for depth calc
    # Whether this leg's verify actually CONSUMES the measured depth.
    # Only Descend's press_outcome does. Reading it costs a tool_frame TF
    # lookup, and this TF tree has no tool_frame at all — the lookup spins
    # its full timeout and returns None (field 2026-08-25). PressFixed
    # judges by progress + torque instead, so it leaves this False and the
    # Runner skips the lookup entirely.
    needs_depth: bool = False
    # time fraction at which a warped descent enters its slow zone;
    # the Runner hands it to TorqueGuard so the baseline is re-taken
    # in the regime the touch actually happens in
    rebaseline_after: float = None
    # progress fraction before which the guard observes but cannot trip
    # (set-down: contact is only possible at the stroke's very end)
    arm_after: float = None


def sanity_violations(traj, margin_rad):
    """Per-joint excursion beyond |start->end| + margin: planner wandered.

    Wrap-aware: reported positions wrap to (-pi, pi], so the series is
    unwrapped by accumulating ang_diff deltas before measuring excursion
    (joint_3 sits AT +pi at home — spec §3)."""
    out = []
    for j, name in enumerate(traj.joint_names):
        pos = [p.positions[j] for p in traj.points]
        unwrapped = [pos[0]]
        for prev, cur in zip(pos, pos[1:]):
            unwrapped.append(unwrapped[-1] + ang_diff(cur, prev))
        allowed = abs(unwrapped[-1] - unwrapped[0]) + margin_rad
        excursion = max(unwrapped) - min(unwrapped)
        if excursion > allowed:
            out.append(
                "%s excursion %.3f rad > |Δ| + margin %.3f" % (name, excursion, allowed)
            )
    return out


def min_standoff():
    """Below this the guard baseline could be captured already in contact."""
    return POSE_UNCERTAINTY_M + TIP_BIAS_M + BASELINE_TRAVEL_M


def check_standoff(hover_z, contact_z):
    gap = hover_z - contact_z
    if gap < min_standoff():
        raise ValueError(
            "hover standoff %.3f m < required %.3f m — the guard baseline "
            "could be captured in contact (spec §6)" % (gap, min_standoff())
        )


def classify_press(depth_m, window):
    lo, hi = window
    return "pressed" if lo <= depth_m <= hi else "rim"


def press_outcome(outcome, depth_m, window):
    if outcome == "touch":
        if depth_m is None:
            return False, "trip depth unknown (no tool z)"
        verdict = classify_press(depth_m, window)
        if verdict == "pressed":
            return True, "button pressed at depth %.4f m" % depth_m
        return False, "rim/edge contact at depth %.4f m (before window)" % depth_m
    if outcome == "arrived":
        return False, "reached depth_window.max untripped — no click detected"
    return False, "descent %s" % outcome


def in_band(pos, band):
    lo, hi = band
    return lo <= float(pos) <= hi


def reverse_retrace(traj, progress):
    """Plan-free retreat: the executed portion of a descent, reversed.

    Used when post-contact planning fails (the start may read as
    in-collision, spec §6). Revalidated by the server's own gates."""
    end = traj.points[-1].time_from_start
    total = end.sec + end.nanosec * 1e-9
    cut = total * float(progress)
    done = [
        p
        for p in traj.points
        if p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 <= cut
    ]
    # Round UP one waypoint: cancel latency means the arm traveled beyond
    # the last fully-elapsed point; the retrace must cover that stretch.
    if len(done) < len(traj.points):
        done.append(traj.points[len(done)])
    out = JointTrajectory()
    out.joint_names = list(traj.joint_names)
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in done]
    t_deep = times[-1]
    # One interpolation step of lead-in. Mirroring times about t_deep puts
    # the FIRST point at exactly t=0.0, and the executor rejects a goal
    # whose diff(times, prepend=0) contains a non-positive dt — so the
    # naive retrace was refused on contact with the arm. Offsetting by one
    # step matches how the planner stamps point k at (k+1)*dt.
    step = (times[-1] - times[0]) / max(1, len(times) - 1) if len(times) > 1 else 0.05
    for p, t in zip(reversed(done), reversed(times)):
        q = copy.deepcopy(p)
        # velocities/accelerations zeroed deliberately: a retrace starts
        # from a standstill after a guard cancel, so mirroring the source
        # cruise velocity would command full speed at the first point.
        q.velocities = [0.0] * len(p.positions)
        q.accelerations = [0.0] * len(p.positions)
        t_new = t_deep - t + step
        q.time_from_start.sec = int(t_new)
        q.time_from_start.nanosec = int(round((t_new - int(t_new)) * 1e9))
        out.points.append(q)
    return out


def time_fraction_at_path_fraction(traj, path_frac):
    """Time fraction at which `traj` has covered `path_frac` of its own
    joint-space path length.

    Execution feedback reports `progress` as elapsed/duration — a TIME
    fraction (executor.py, and the action's own comment). Callers that know
    where along the PATH contact is expected must convert, because the two
    only coincide for a constant-speed profile and cuRobo's is not one.
    Trajectory points are uniformly spaced in time, so a point's time
    fraction is just its index over the count.
    """
    pts = list(traj.points)
    if len(pts) < 2:
        return float(path_frac)
    cum, total = [0.0], 0.0
    for a, b in zip(pts, pts[1:]):
        total += math.sqrt(sum((x - y) ** 2 for x, y in zip(a.positions, b.positions)))
        cum.append(total)
    if total <= 0.0:
        return float(path_frac)

    def stamp(p):
        return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9

    duration = stamp(pts[-1])
    if duration <= 0.0:
        return float(path_frac)
    target = float(path_frac) * total
    for i, c in enumerate(cum):
        if c >= target:
            return stamp(pts[i]) / duration
    return 1.0
