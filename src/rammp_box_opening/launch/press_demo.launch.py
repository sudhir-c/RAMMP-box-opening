"""Planner + wrist-camera driver for the tag-driven press demo.

    ros2 launch rammp_box_opening press_demo.launch.py                # dry-run
    ros2 launch rammp_box_opening press_demo.launch.py execute:=true  # it moves

Then, in its own shell (the CLI is human-run; --execute alone arms it):

    ros2 run rammp_box_opening press_demo --execute

Does NOT start the arm bringup — start ros2_kortex first, in its own
terminal (see docs/HARDWARE_BRINGUP.md):

    ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 \\
        dof:=7 gripper:=robotiq_2f_85 launch_rviz:=false

Stray handling follows sweep_demo.launch.py: any pre-existing planner
node is stale by definition and gets killed before ours starts.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ARGS = [
    ("execute", "false", "allow motion (default: dry-run, nothing moves)"),
    ("config", "gen3_real.yaml", "planner YAML (world comes from it)"),
    ("camera", "true", "start the D405 driver (false: already running)"),
]


def _sweep_strays():
    """Kill leftover planner nodes (sweep_demo.launch.py pattern): this
    launch starts the planner itself, so a pre-existing one is stale —
    and on a shared graph the CLI would bind to whichever discovery
    finds first. Only installed rammp_curobo_ros binaries are touched."""
    import signal as _signal
    import subprocess as _sp
    import time as _time

    def strays():
        found = []
        out = _sp.run(
            ["pgrep", "-af", "rammp_curobo_ros"], capture_output=True, text=True
        ).stdout
        for ln in out.splitlines():
            pid, _, cmd = ln.partition(" ")
            parts = cmd.split()
            if len(parts) >= 2 and "/lib/rammp_curobo_ros/" in parts[1]:
                found.append((int(pid), parts[1].rsplit("/", 1)[-1]))
        # a stale camera driver keeps the D405 claimed: the new driver gets
        # "Device or resource busy", the device drops off the bus, and both
        # die (field 2026-08-26: two overlapping launches killed the camera)
        out = _sp.run(
            ["pgrep", "-af", "realsense2_camera_node"], capture_output=True, text=True
        ).stdout
        for ln in out.splitlines():
            pid, _, _cmd = ln.partition(" ")
            found.append((int(pid), "realsense2_camera_node"))
        return found

    found = strays()
    for pid, name in found:
        print("[press_demo.launch] killing stray %s (pid %d)" % (name, pid))
        try:
            os.kill(pid, _signal.SIGINT)
        except ProcessLookupError:
            pass
    if found:
        _time.sleep(1.5)
        for pid, _name in strays():
            try:
                os.kill(pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass


def _nodes(context, *_args, **_kwargs):
    _sweep_strays()

    def val(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return val(name).strip().lower() in ("1", "true", "yes", "on")

    planner = Node(
        package="rammp_curobo_ros",
        executable="planner_node",
        name="rammp_curobo",
        output="screen",
        emulate_tty=True,
        # an in-flight cuRobo solve runs ~5 s and cannot be interrupted;
        # give launch more than its 5 s default before SIGINT -> SIGTERM
        sigterm_timeout="12",
        parameters=[{"config": val("config"), "execute": flag("execute")}],
    )
    nodes = [planner]
    if flag("camera"):
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        FindPackageShare("realsense2_camera"),
                        "/launch/rs_launch.py",
                    ]
                ),
                launch_arguments={
                    "camera_namespace": "d405",
                    "camera_name": "d405",
                    # the watcher's depth refinement reads depth at the
                    # tag's COLOR pixel — alignment is required
                    "align_depth.enable": "true",
                }.items(),
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [OpaqueFunction(function=_nodes)]
    )
