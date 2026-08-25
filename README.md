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
motion unless given `--execute`, and motion CLIs are run by a human
with a hand on the physical e-stop. The Phase-1 primitive CLIs
additionally require a typed `yes`; `press_demo` deliberately does not
(owner decision 2026-08-24: `--execute` alone arms it, autonomous once
started, Ctrl+C stops everything). No autonomous motion from agent
sessions, ever. The planner's `execute` parameter is a third,
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
container on the MEASURED table (top −0.027 m, reconciled 2026-08-24
from RAMMP-CuRobo's Orbbec measurement): contact (button top, z=0.133)
and staging (+0.08, z=0.213). Result (2026-08-24, 5 cm pitch, 459
plans; raw data `docs/reach_map.json`):

```
     x 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75
y -0.45  #   #   #   #   #   #   #   #   .   .   .   .
y -0.35  #   #   #   #   #   #   #   #   #   .   .   .
y -0.25  #   #   #   #   #   #   #   #   #   #   .   .
y -0.15  #   #   #   #   #   #   #   #   #   #   o   .
y -0.05  #   #   #   #   #   #   #   #   #   #   #   .
y +0.00  #   #   #   #   #   #   #   #   #   #   #   .
y +0.05  #   o   #   #   #   #   #   #   #   #   #   .
y +0.15  #   #   #   #   #   #   #   #   #   #   o   .
y +0.25  #   #   #   #   #   #   #   #   #   #   .   .
y +0.35  #   #   #   o   #   #   #   #   #   .   .   .
y +0.45  #   #   #   #   #   #   #   #   .   .   .   .
(# = staging AND contact plan, o = one height only, . = neither;
 every-other row shown — full 19-row map in docs/reach_map.json)
```

Read it as: **place the container with its button between x 0.20 and a
radial edge of ~0.70 m, anywhere in y ±0.45** — 180/228 grid points are
usable at both heights, with a few isolated single-point holes (planner
stochasticity; nudge an inch if one bites). This is the complement of
the wrist-flat lore: the wrist-flat TRANSIT family fails inside ~0.5 m
radius, while TOOL-DOWN work covers the whole band — which is why every
mission pose uses the tool-down family. The configured `bench_pose`
(0.45, 0) and `lid_place` (0.45, −0.25) poses all plan from HOME.

For the press demo, the practical zone is tighter than the reach band:
the wrist camera at the scan pose sees roughly ±0.30 m in x and
±0.19 m in y around (0.45, 0) at tag height — the tag must be VISIBLE
before it can be reached (a container outside the view exits honestly
via the no-tag path).

Caveats (also in the JSON meta): bare-bench world — the container's own
collision model shrinks this only locally; every plan starts at HOME;
probe yaw is the point bearing (joint_7 absorbs tool-down yaw). Re-run
whenever bench geometry changes:

```zsh
python3 scripts/reach_probe.py            # ~2 min on the Orin; --quick to smoke
```

## The press demo (current milestone, owner design 2026-08-24)

One launch, one command, autonomous once started:

```zsh
ros2 launch rammp_box_opening press_demo.launch.py execute:=true   # planner + D405
ros2 run rammp_box_opening press_demo --execute                    # in its own shell
```

Flow: SCAN (tool-down look pose over the bench) → DETECT (continuous
wrist-camera ArUco watcher; the camera runs from CLI start to exit) →
no fresh stable fix within the timeout → home, exit 2 — otherwise
CLOSE gripper + APPROACH staging above the tag (full world with the
tag-derived container cuboid) → close-range re-fix → ONE guarded press
stroke straight from staging (`press_demo.travel_m` below the tag plane
at `press_demo.speed`; a torque trip near the expected contact depth or
full travel both count as pressed — an EARLY trip reports as a strike
failure) → retreat → HOME.

The tag (DICT_4X4_50 id 0, `docs/tag0_50mm.png`) sits ON the button
top; its depth-refined pose IS the press target — `tag.offset_xyz`
supports off-button placement. All knobs live in
`config/containers/oxo_pop.yaml`; measure `tag.size_m` (printed edge),
`press_demo.travel_m` (caliper), and the scan pose per
`docs/HARDWARE_BRINGUP.md`, then flip `measure_me: false`.

Proven off-bench by `scripts/press_demo_e2e.py` (stub planner +
synthetic D405 publishing real rendered ArUco frames on an isolated
domain): the actual detection→PnP→TF→depth→pose pipeline recovers the
container origin to ~1 mm, the full leg chain runs, and the no-tag
path homes and exits 2.
