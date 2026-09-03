"""Shared constants: arm facts and runner defaults (spec §3, §6)."""

# Gen3 home joints in controller order — FK-verified in RAMMP-CuRobo's
# tour_demo.py; joint_3 sits AT +pi (every comparison uses ang_diff).
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

# tool_frame attitude at HOME: tool z level along world +x, wrist flat.
WRIST_FLAT_XYZW = [0.5, 0.5, 0.5, 0.5]

NODE_NAMESPACE = "/rammp_curobo"
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"

TRANSIT_SPEED = 0.75  # was 1.0: full-speed arrivals read rough and settle
# imprecisely at the bench (owner 2026-09-02, "slow it down a bit"); the press
#                      stroke alone stays at press_demo.speed
CONTACT_SPEED = 0.15
DRIFT_REPLAN_RAD = 0.04  # < server start gate (0.05); catches arrival-tol drift
SANITY_MARGIN_RAD = 0.35  # per-joint excursion allowance beyond |start->end|
# A mission must START near HOME: every legit run begins there, so a
# distant start means the last run ended badly. Refuse before any motion
# instead of planning something dramatic from wreckage (field
# 2026-09-01: a plan from a failure-held pose swung the arm half upside
# down). This is the ONLY start-shape guard: a per-leg joint-sweep cap
# was tried and measured wrong the same day — wrap-aware endpoint sweep
# saturates at pi, and six live HOME->scan plans legitimately swept
# 2.4-2.9 rad on wrist/elbow joints for the tool-down reorientation.
HOME_START_TOL_RAD = 1.2

POSE_UNCERTAINTY_M = 0.02  # calibration floor (spec §3)
TIP_BIAS_M = 0.021  # 2F-85 pad face beyond tool_frame (disabled in gen3.yaml)
BASELINE_TRAVEL_M = 0.01  # descent distance budget while the guard baselines

GRIPPER_CMD_CLOSED = 0.8  # GripperCommand position at full close
GRIPPER_CMD_OPEN = 0.0  # ~85 mm aperture

# Tool-down rest pose at the scan pose [0.42, 0, 0.45] (open_box.park_tool_down).
# HOME is the wrist-flat factory pose and every mission pose is tool-down —
# a different IK family — so each run paid a 2.4-2.9 rad wrist/elbow flip
# twice (scan flight 3.7-4.7 s, final home 3.5-5.5 s). Parked here, the
# scan leg vanishes and the mission ends with a same-family move. Planned
# from HOME with the real planner 2026-09-02: FK lands on the scan pose to
# the mm, valid in the bench world, and PARK -> HOME plans clean.
PARK = [0.608, 0.8905, 1.1224, -1.1386, -2.1213, 2.1767, 2.2738]
REST_TOL_RAD = 0.05  # "already there": the server's own start gate

# Joint velocity limits the planner plans against (cuRobo gen3_real.yaml, the
# URDF's): the re-timer caps every joint below these and the executor refuses
# a goal above them — two independent gates on the same numbers.
JOINT_VMAX = [1.396, 1.396, 1.396, 1.396, 1.222, 1.222, 1.222]
