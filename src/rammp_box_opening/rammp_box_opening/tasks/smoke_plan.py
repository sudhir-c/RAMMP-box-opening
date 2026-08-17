"""Phase-0 smoke: plan a pose with explicit start_joints, print it.

Proves build, discovery, and the planner contract end to end — no arm
needed (start_joints = HOME). There is deliberately NO execution path in
this tool.
"""

import rclpy

from rammp_box_opening.constants import HOME
from rammp_box_opening.models.container import wrist_flat_quat
from rammp_box_opening.runtime.client import PlannerClient

# Benign frontal pose, well above any bench. NOT closer/lower: at
# r=0.45, z=0.35 the wrist-flat goal has no collision-free IK (verified
# against the live planner — the flange sits 12 cm behind the fingertip).
TARGET = [0.55, 0.0, 0.40]


def main():
    rclpy.init()
    node = rclpy.create_node("rammp_box_opening_smoke")
    client = PlannerClient(node)
    plan = client.plan_to_pose(TARGET, wrist_flat_quat(TARGET), HOME)
    if plan is None or not plan.success:
        raise SystemExit("SMOKE FAILED: %s" % getattr(plan, "message", "no response"))
    last = plan.trajectory.points[-1].time_from_start
    print(
        "SMOKE OK: %d points, %.2f s at full speed (planning %.2f s)"
        % (
            len(plan.trajectory.points),
            last.sec + last.nanosec * 1e-9,
            plan.planning_time,
        )
    )
    for j, name in enumerate(plan.trajectory.joint_names):
        pos = [p.positions[j] for p in plan.trajectory.points]
        print(
            "  %-9s %8.3f -> %8.3f  (excursion %.3f)"
            % (name, pos[0], pos[-1], max(pos) - min(pos))
        )
