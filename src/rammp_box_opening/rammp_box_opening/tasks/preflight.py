"""Every-session preflight (spec §4): checks + idempotent world push.

FAIL (nonzero exit): /joint_states missing or without effort fields;
planner actions unreachable; world push rejected.
Report-only: planner execute param value, controller states.
"""

import time

import rclpy

from rammp_box_opening.models.container import ConfigPoseSource, ContainerModel
from rammp_box_opening.runtime.client import PlannerClient
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


def _controllers(node):
    try:
        from controller_manager_msgs.srv import ListControllers

        from rammp_box_opening.runtime.client import spin_until_done

        cli = node.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        if not cli.wait_for_service(timeout_sec=3.0):
            return None
        resp = spin_until_done(node, cli.call_async(ListControllers.Request()), 5.0)
        if resp is None:
            return None
        return [(c.name, c.state) for c in resp.controller]
    except Exception:
        return None


def main():
    ap = cli_common.make_parser(__doc__)
    args = ap.parse_args()
    cfg = args.container or cli_common.default_container_yaml()
    bench = args.bench_world or cli_common.default_bench_yaml()

    rclpy.init()
    node = rclpy.create_node("rammp_box_opening_preflight")
    client = PlannerClient(node)
    failures = 0

    # 1. /joint_states fresh, with efforts (guarded primitives require them)
    t0 = time.monotonic()
    try:
        client.joints()
        fresh = time.monotonic() - t0
        efforts = client.efforts_present()
        print(
            "PASS  /joint_states fresh (%.1f s) — efforts %s"
            % (fresh, "present" if efforts else "MISSING")
        )
        if not efforts:
            print("FAIL  effort fields absent — guarded primitives will refuse")
            failures += 1
    except SystemExit as exc:
        print("FAIL  %s" % exc)
        raise SystemExit(1) from exc

    # 2. planner actions reachable
    if client.planner_reachable(timeout_s=5.0):
        print("PASS  planner actions reachable")
    else:
        print("FAIL  planner actions unreachable — launch planner.launch.py")
        failures += 1

    # 3. planner execute param (report-only — dry-run planning is fine)
    print("INFO  planner execute param: %s" % client.planner_execute_enabled())

    # 4. controllers (report-only; absent on planner-only sessions)
    ctrls = _controllers(node)
    if ctrls is None:
        print("INFO  controller_manager not responding (no arm bringup?)")
    else:
        for name, state in ctrls:
            print("INFO  controller %s: %s" % (name, state))

    # 5. idempotent full-world push (SetWorld is write-only: preflight
    #    ESTABLISHES the world rather than querying it)
    model = ContainerModel.load(cfg)
    cpose = ConfigPoseSource(cfg).container_pose()
    name, path = WorldStore(bench).push_name("full", model=model, cpose=cpose)
    ok, msg = client.set_world(str(path))
    print("%s  set_world(%s): %s" % ("PASS" if ok else "FAIL", name, msg))
    if not ok:
        failures += 1

    raise SystemExit(1 if failures else 0)
