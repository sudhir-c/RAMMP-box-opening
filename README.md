# RAMMP-box-opening

Open an OXO POP container with the Kinova Gen3 7-DoF + Robotiq 2F-85:
find the box with the wrist camera, press its lid button, pull the lid
and set it aside — one autonomous mission (`press_demo`) composed of
guarded legs over the RAMMP-CuRobo planning service. This repo is a
**client**: interfaces come from the `~/RAMMP-CuRobo/install` overlay —
no copied code, no submodule, no modifications to RAMMP-CuRobo.

Design spec (authoritative): `docs/superpowers/specs/2026-08-14-box-opening-design.md`
Implementation plan: `docs/superpowers/plans/2026-08-17-box-opening-phase-0-1.md`
Hardware sessions: `docs/HARDWARE_BRINGUP.md`

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

Python files are COPIED into `build/` at build time (the egg-link points
there, not at `src/`), so rebuild before any `ros2 run` or e2e after
editing code. YAMLs are symlinked and update in place.

## Runtime sourcing chain (execution shells)

humble → RAMMP-Kinova ws → RAMMP-CuRobo → this ws, with
`export ROS_LOCALHOST_ONLY=1` in EVERY shell, explicitly.

## Safety stance

Dry-run is the default everywhere: every CLI plans and previews without
motion unless given `--execute`, and motion CLIs are run by a human
with a hand on the physical e-stop. `home_arm` additionally requires a
typed `yes`; `press_demo` deliberately does not (owner decision
2026-08-24: `--execute` alone arms it, autonomous once started, Ctrl+C
stops everything). No autonomous motion from agent sessions, ever. The
planner's `execute` parameter is a third, server-side gate for arm
motion; the runner additionally refuses gripper commands while the
planner is dry-run (the direct gripper action is not server-gated).

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
swept tool-down poses — the press-attitude family every contact leg
uses — across the bench at two heights for a container on the MEASURED
table (top −0.027 m, reconciled 2026-08-24 from RAMMP-CuRobo's Orbbec
measurement): contact (button top, z=0.133 at the time of the sweep) and
staging (+0.08, z=0.213). Result (2026-08-24, 5 cm pitch, 459 plans;
raw data `docs/reach_map.json`):

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
mission pose uses the tool-down family. The configured scan pose
(0.42, 0, 0.45) and the `lid_place` (0.45, −0.25) carry hover both plan
from HOME — and with `open_box.park_tool_down` on (the default), the
scan pose is also where the arm RESTS, so a run that starts there skips
the scan flight and the mission ends there instead of flipping the wrist
back to the factory HOME.

For the mission, the practical zone is tighter than the reach band: the
wrist camera at the scan pose sees roughly 0.6 × 0.4 m of bench around
(0.42, 0) at lid height — the box must be in VIEW before it can be
reached (a container outside it exits honestly via the no-box path),
and the merged press wants it within `open_box.merge_press_max_lateral_m`
(0.20 m) of the scan axis; further out, the staged approach runs instead.

Caveats (also in the JSON meta): bare-bench world — the container's own
collision model shrinks this only locally; every PROBE plan starts at
HOME (mission legs chain from the previous leg's predicted end, and a
mission may start from PARK); probe yaw is the point bearing (joint_7
absorbs tool-down yaw). Re-run whenever bench geometry changes:

```zsh
python3 scripts/reach_probe.py            # ~2 min on the Orin; --quick to smoke
```

## The mission (owner design 2026-08-24; press proven live 2026-08-25)

One launch, one command, autonomous once started:

```zsh
ros2 launch rammp_box_opening press_demo.launch.py execute:=true   # planner, relay, OWL node, D405
ros2 run rammp_box_opening press_demo --execute                    # in its own shell
```

Flow (`tasks/press_demo.py`, one log line per state):

1. **SCAN** — tool-down look pose over the bench, planned in the bench
   world whose unseen-container band blocks the whole placement zone to
   container height. No flight when the arm already rests tool-down
   there (`open_box.park_tool_down`).
2. **DETECT** — the lid-top plateau above the measured table
   (`perception/depth_source.py`: band, smoothness, footprint and border
   gates, button-circle refine, origin z pinned to the calibrated table),
   gated by the persistent OWL node's bbox (`owl_detector`,
   `perception/owl_source.py`). Depth answers first; the semantic ladder
   (`vlm.backends`: local OWL, then the cloud) is walked only when depth
   alone has not answered in its first beat, and every rung declining
   means plain depth. No stable fix within `detect.timeout_s` → home,
   exit 2. The fingers close (`press:close`) the moment the fix commits,
   overlapping the press plan.
3. **PRESS** — ONE merged guarded stroke from the scan pose to
   `press_demo.travel_m` below the lid plane: fast through free air,
   `press_demo.speed` into contact, the guard re-baselined at the speed
   change. A torque trip near the expected contact or full travel both
   count as pressed; an EARLY trip reports as a strike failure. A scan
   pose more than `merge_press_max_lateral_m` off-axis falls back to a
   staged approach + stroke. The post-press retreat to the hop is LAZY
   (planned from live after the touch) and `grip:open` goes out on
   arrival there.
4. **GRIP** — guarded descent to `grip_clear_m` above the lid around the
   popped knob (a trip = struck it), band-verified close
   (`open_box.grip_band`; 0.8 = closed on air), gentle lift.
5. **PLACE** — carry to the lid drop (configured `lid_place`, slid clear
   of the detected box when it has to be), guarded set-down at
   `setdown_touch_nm` (4.0 Nm — the trip IS the success), release,
   lazy retreat to the carry height, home (or PARK).

Each next phase is planned as a Runner lookahead while the previous
phase's last unguarded motion flies. `--press-only` stops after the
press (retreat to staging, home); `--detect-only` reports fixes for
15 s and homes (the wrist-mount calibration observation). All knobs live
in `config/containers/oxo_pop.yaml`.

Proven off-bench by `scripts/press_demo_e2e.py` (stub planner +
synthetic D405/OWL on an isolated domain, shipped config minus the cloud
rung): the actual depth pipeline recovers the box origin to ~1 mm and
its yaw to a fraction of a degree, the full leg chain runs with the
guard-trip/cancel/replan path exercised, and an empty table homes and
exits 2. Rebuild, then run it and `scripts/abort_e2e.py` after any change.
