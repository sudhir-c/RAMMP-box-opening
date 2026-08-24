"""Shared CLI plumbing: args, node/client/runner wiring, safety refusals.

Dry-run is the default for every CLI. --execute additionally requires the
typed 'yes' inside Runner.run, the planner's own execute:=true for MOTION
legs, and a measured container config (measure_me: false).
"""

import argparse
import math
import sys
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions

from rammp_box_opening.models.container import (
    ConfigPoseSource,
    ContainerModel,
    ContainerPose,
)
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.abort import AbortFlag, install_sigint
from rammp_box_opening.runtime.client import PlannerClient
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.worlds import WorldStore


def _share_path(*parts):
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("rammp_box_opening"), *parts)


def default_container_yaml():
    return str(_share_path("config", "containers", "oxo_pop.yaml"))


def default_bench_yaml():
    return str(_share_path("config", "world_bench.yaml"))


def make_parser(desc):
    ap = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="after previewing, offer to execute (planner must be launched "
        "with execute:=true for arm motion; NOTE: gripper closes go over "
        "the direct gripper action — the runner refuses them while the "
        "planner is dry-run, planner dry-run alone does not prevent them)",
    )
    ap.add_argument(
        "--container",
        default=None,
        help="container config YAML (default: installed oxo_pop.yaml)",
    )
    ap.add_argument(
        "--bench-world",
        default=None,
        help="bench world YAML (default: installed world_bench.yaml)",
    )
    ap.add_argument(
        "--lid-place",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="override the lid set-down spot (base_link, m)",
    )
    return ap


def lid_place_of(args):
    if args.lid_place is None:
        return None
    return ContainerPose(xyz=tuple(args.lid_place), yaw=0.0)


def build_ctx(args):
    cfg = args.container or default_container_yaml()
    bench = args.bench_world or default_bench_yaml()
    model = ContainerModel.load(cfg)
    if args.execute and model.measure_me:
        sys.exit(
            "container config still carries measure_me: true — run the "
            "Phase-1 measurement worksheet (docs/HARDWARE_BRINGUP.md) and "
            "flip it before any hardware execution (dry-run is fine)."
        )
    # own SIGINT: an in-flight stroke must get its cancel delivered on a
    # live context before we exit (runtime/abort.py; proven by
    # scripts/abort_e2e.py — rclpy's default handler makes it a race)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    abort = AbortFlag()
    install_sigint(abort)
    node = rclpy.create_node("rammp_box_opening")
    client = PlannerClient(node, abort=abort)
    ctx = Ctx(
        model=model,
        cpose=ConfigPoseSource(cfg).container_pose(),
        client=client,
        worlds=WorldStore(bench),
        config_path=cfg,
    )
    runner = Runner(client, ctx.worlds)
    return ctx, runner


def run_task(args, build_legs, **kwargs):
    ctx, runner = build_ctx(args)
    legs = build_legs(ctx, **{k: v for k, v in kwargs.items() if v is not None})
    try:
        results = runner.run(legs, execute=args.execute)
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    bad = [r for r in results if not r.ok]
    if bad:
        sys.exit(1)


def yaw_deg_to_rad(deg):
    return math.radians(float(deg))
