import inspect

from rammp_box_opening.runtime.client import PlannerClient


def test_protocol_surface():
    required = {
        "joints": [],
        "wrist_efforts": [],
        "efforts_present": [],
        "tool_xyz": ["timeout_s"],
        "plan_to_pose": ["xyz", "quat_xyzw", "start_joints"],
        "plan_to_joints": ["q7", "start_joints"],
        "execute": ["traj", "speed", "guard"],
        "set_world": ["path_or_name"],
        "planner_execute_enabled": [],
        "gripper_cmd": ["position"],
    }
    for name, params in required.items():
        fn = getattr(PlannerClient, name)
        sig = list(inspect.signature(fn).parameters)[1:]  # drop self
        for p in params:
            assert p in sig, "%s missing param %s" % (name, p)
