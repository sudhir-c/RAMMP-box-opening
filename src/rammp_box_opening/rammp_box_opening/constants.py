"""Shared constants: arm facts and runner defaults (spec §3, §6)."""

# Gen3 home joints in controller order — FK-verified in RAMMP-CuRobo's
# tour_demo.py; joint_3 sits AT +pi (every comparison uses ang_diff).
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

# tool_frame attitude at HOME: tool z level along world +x, wrist flat.
WRIST_FLAT_XYZW = [0.5, 0.5, 0.5, 0.5]

NODE_NAMESPACE = "/rammp_curobo"
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"

TRANSIT_SPEED = 1.0  # full rated speed (owner 2026-08-26); the press
#                      stroke alone stays at press_demo.speed
CONTACT_SPEED = 0.15
DRIFT_REPLAN_RAD = 0.04  # < server start gate (0.05); catches arrival-tol drift
SANITY_MARGIN_RAD = 0.35  # per-joint excursion allowance beyond |start->end|

POSE_UNCERTAINTY_M = 0.02  # calibration floor (spec §3)
TIP_BIAS_M = 0.021  # 2F-85 pad face beyond tool_frame (disabled in gen3.yaml)
BASELINE_TRAVEL_M = 0.01  # descent distance budget while the guard baselines

GRIPPER_CMD_CLOSED = 0.8  # GripperCommand position at full close
GRIPPER_CMD_OPEN = 0.0  # ~85 mm aperture
