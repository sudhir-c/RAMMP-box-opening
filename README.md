# RAMMP-box-opening

Open and pick up an OXO POP container with the Kinova Gen3 7-DoF +
Robotiq 2F-85, as parameterized, composable primitives (approach, press,
grasp, lift, place) over the RAMMP-CuRobo planning service. This repo is
a **client**: interfaces come from the `~/RAMMP-CuRobo/install` overlay —
no copied code, no submodule, no modifications to RAMMP-CuRobo.

Design spec (authoritative): `docs/superpowers/specs/2026-08-14-box-opening-design.md`
Implementation plan: `docs/superpowers/plans/2026-08-17-box-opening-phase-0-1.md`
Hardware sessions: `docs/HARDWARE_BRINGUP.md` (Phase 1)

## Build (zsh)

```zsh
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source ~/RAMMP-CuRobo/install/setup.zsh
cd ~/RAMMP-box-opening
colcon build --symlink-install
source install/setup.zsh
python3 -m pytest src/rammp_box_opening/test -q
```

## Runtime sourcing chain (execution shells)

humble → RAMMP-Kinova ws → RAMMP-CuRobo → this ws, with
`export ROS_LOCALHOST_ONLY=1` in EVERY shell, explicitly.

## Safety stance

Dry-run is the default everywhere: every CLI plans and previews without
motion unless given `--execute` AND a typed `yes`, and motion CLIs are
run by a human with a hand on the physical e-stop. No autonomous motion
from agent sessions, ever. The planner's `execute` parameter is a third,
server-side gate for arm motion; the runner additionally refuses gripper
commands while the planner is dry-run (the direct gripper action is not
server-gated).
