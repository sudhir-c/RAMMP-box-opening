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

Ctrl+C during motion is OWNED, not inherited: motion CLIs install their
own SIGINT handler (`runtime/abort.py`) so an in-flight stroke gets its
cancel delivered on a live context and confirmed by the server before
the process exits — rclpy's default handler makes that a race that ends
in a traceback instead of an answer. Proven by `scripts/abort_e2e.py`
(stub planner, isolated `ROS_DOMAIN_ID=77`, goal-count audit; refuses to
run beside a real stack). Re-run it after any change to the client,
runner, or CLI wiring — it is the software half of the attended abort
drill in `docs/HARDWARE_BRINGUP.md`.

## Where the container may sit — measured tool-down reach band

`scripts/reach_probe.py` (offline, in-process planner, nothing can move)
swept tool-down poses — the press-attitude family every contact
primitive uses — across the bench at the two mission heights for a
container on the table: contact (button top, z=0.09) and hover (+0.08).
Result (2026-08-24, placeholder bench geometry, 5 cm pitch, 459 plans;
raw data `docs/reach_map.json`):

```
     x 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75
y -0.45  #   #   #   #   #   #   #   #   .   .   .   .
y -0.35  #   #   #   #   #   #   #   #   #   o   .   .
y -0.25  #   #   #   #   #   #   #   #   #   #   .   .
y -0.15  #   #   #   #   #   #   #   #   #   #   #   .
y -0.05  #   #   #   #   #   #   #   #   #   #   #   .
y +0.00  o   #   #   #   #   #   #   #   #   #   #   .
y +0.05  #   #   #   #   #   #   #   #   #   #   #   .
y +0.15  #   #   #   #   #   #   #   #   #   #   #   .
y +0.25  #   #   #   #   #   #   #   #   #   #   .   .
y +0.35  #   #   #   #   #   #   #   #   #   o   .   .
y +0.45  #   #   #   #   #   #   #   #   .   .   .   .
(# = hover AND contact plan, o = one height only, . = neither;
 every-other row shown — full 19-row map in docs/reach_map.json)
```

Read it as: **place the container with its button between x 0.25 and a
radial edge of ~0.70 m, anywhere in y ±0.45** — 188/228 grid points are
usable at both heights. The inner edge is real but tight: (0.20, 0.00)
fails at contact height (near-base fold-in), so keep 0.25 m as the
practical minimum. This is the complement of the wrist-flat lore: the
wrist-flat TRANSIT family fails inside ~0.5 m radius, while TOOL-DOWN
work covers 0.25–0.70 m — the two families overlap only in an annulus,
which is why contact primitives never use wrist-flat attitudes. The
configured `bench_pose` (0.45, 0, yaw 0) and `lid_place` (0.45, −0.25)
hover/contact poses all plan from HOME.

Caveats (also in the JSON meta): bare-bench world — the container's own
collision model shrinks this only locally; every plan starts at HOME;
probe yaw is the point bearing (joint_7 absorbs tool-down yaw). Re-run
after the Phase-1 worksheet reconciles the real bench geometry:

```zsh
python3 scripts/reach_probe.py            # ~2 min on the Orin; --quick to smoke
```
