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

# Reflex recoil after a press trip: how much of the descent's own path to
# reverse, and at what cruise fraction. Measured with the real planner and
# FK at two bench placements (2026-09-03): 0.085 rad of joint arc lifts
# the tool 20.7-21.0 mm, 0.05 rad lifts 13.3-13.9 mm — the mapping barely
# moves with the box, so a joint-arc budget is a fair stand-in for the
# Cartesian lift this TF tree cannot measure. 0.09 rad ~ 22 mm: enough to
# unload the contact while the considered retreat is planned.
# A reflex, not a considered move: brisk, short, no planning.
RECOIL_ARC_RAD = 0.09
RECOIL_SPEED = 0.5

# The planner is commanded in tool_frame, but the FINGERTIPS reach past it.
# Measured two independent ways (2026-09-03), agreeing to 0.7 mm:
#   - cuRobo's own model: the *_inner_finger_pad link sits 10.3 mm beyond
#     tool_frame (sphere index API on robot_gen3_2f85.yaml);
#   - the arm's own contact event: replaying run-20260903-123629's press,
#     tool_frame was at z 0.0941 when the guard tripped on a button top the
#     depth had measured at 0.0831 -> 11.0 mm.
# Nothing accounted for it, so every fingertip-referenced target was that
# much too deep: grip:down, commanded to button + 5 mm, put the pads at
# button - 6 mm — INSIDE the lid (its logged peak torque was 1.5-1.7 Nm
# where free air reads ~0), so the fingers could not close on the knob.
# Note the real robot's description does not even define tool_frame once a
# gripper is attached; it is cuRobo's own frame. Command tool_frame this
# much HIGHER than where the fingertips should land.
TCP_OFFSET_M = 0.011
