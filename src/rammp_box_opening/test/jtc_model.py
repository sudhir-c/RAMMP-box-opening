"""What the ros2_controllers JTC (humble, 'splines') commands from a
message carrying positions, velocities and accelerations: a quintic
Hermite per interval. Used to assert the EXECUTED acceleration of a
re-timed trajectory, which the executor's gates never look at."""
import numpy as np


def quintic_coeffs(p0, v0, a0, p1, v1, a1, h):
    c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * h - (3 * a0 - a1) * h**2) / (2 * h**3)
    c4 = (-30 * (p1 - p0) + (14 * v1 + 16 * v0) * h + (3 * a0 - 2 * a1) * h**2) / (2 * h**4)
    c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * h + (a1 - a0) * h**2) / (2 * h**5)
    return p0, v0, a0 / 2.0, c3, c4, c5


def peak_accel(msg, nsub=24):
    """Peak |joint acceleration| the spline commands, sampled finely inside
    every interval whatever its length, plus the worst interval index."""
    q = np.asarray([p.positions for p in msg.points])
    v = np.asarray([p.velocities for p in msg.points])
    a = np.asarray([p.accelerations for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    worst, where = 0.0, -1
    for k in range(len(q) - 1):
        h = t[k + 1] - t[k]
        c = quintic_coeffs(q[k], v[k], a[k], q[k + 1], v[k + 1], a[k + 1], h)
        s = np.linspace(0, h, nsub)[:, None]
        aa = 2 * c[2] + 6 * c[3] * s + 12 * c[4] * s**2 + 20 * c[5] * s**3
        m = float(np.abs(aa).max())
        if m > worst:
            worst, where = m, k
    return worst, where
