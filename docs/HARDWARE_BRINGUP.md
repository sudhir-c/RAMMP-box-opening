# Hardware bringup and attended sessions

Every motion in this document is run BY A HUMAN with a hand on the
physical e-stop. Agent sessions prepare, analyze logs, and never execute.
Spec: `docs/superpowers/specs/2026-08-14-box-opening-design.md` (§6, §8).

## 1. Session preconditions

- Exactly ONE arm stack at a time. Check before starting:
  `pgrep -fa 'kortex|planner_node'` — note this Jetson has carried a
  root-owned planner_node from `/opt/rammp_curobo` (same `/rammp_curobo`
  names!); confirm with the owner which instance owns the session before
  launching another.
- Bringup order (human, separate shells, each with
  `export ROS_LOCALHOST_ONLY=1` typed explicitly — non-interactive shells
  skip `~/.zshrc`):
  1. Arm driver (T1), from a shell sourcing humble → `~/RAMMP-Kinova/ros2_ws`:

     ```zsh
     ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 launch_rviz:=false gripper:=robotiq_2f_85
     ```

     The `gripper:=robotiq_2f_85` argument is REQUIRED: its default is
     empty, and an empty gripper argument silently spawns no gripper
     controller — the arm comes up, `/joint_states` flows, and every
     gripper command then fails with no controller to take it.
  2. Planner, joint-state relay, OWL detector and wrist camera (T2):
     `ros2 launch rammp_box_opening press_demo.launch.py execute:=true`
     from a task shell (below). It kills stray planners, camera drivers
     and OWL nodes first. (Planner alone, without the mission's relay
     and camera: `ros2 launch rammp_curobo_ros planner.launch.py
     config:=gen3_real.yaml execute:=true`.)
  3. Task shells: full chain humble → RAMMP-Kinova ws → RAMMP-CuRobo →
     `~/RAMMP-box-opening/install/setup.zsh`.
- The planner's `execute` parameter gates ARM motion server-side. It does
  NOT gate the gripper: gripper closes go over the direct
  `/robotiq_gripper_controller/gripper_cmd` action. The runner refuses
  gripper legs while the planner is dry-run, but treat any `--execute`
  run as capable of moving the gripper.

## 2. Measurement worksheet (once, before the first `--execute`)

All measurements are from base_link: +x forward, +z up, tape measure,
metres. Record into the two config files (`colcon build
--symlink-install` symlinks the YAMLs, so they update in place; python
is copied at build, so rebuild after code edits).

### 2a. Bench world — `src/rammp_box_opening/config/world_bench.yaml`

1. Table top height relative to base_link and extents. This number is
   load-bearing twice: the planner's table cuboid, AND the depth
   detector's search band (container candidates are looked for at
   table + `dims.z`) and the pinned container origin z. A wrong table
   height fails detection honestly ("top N mm from nominal … table_z
   needs recalibrating") rather than pressing on a guess.
2. Any wall/shelf within reach: add as cuboids, same schema.
3. Reconcile ONCE with `~/RAMMP-CuRobo`'s `world_real_bench.yaml` (theirs
   also carries placeholder geometry — measuring supersedes both; after
   this, OUR generated worlds are what the planner uses at runtime).

### 2b. Container model — `src/rammp_box_opening/config/containers/oxo_pop.yaml`

With the container at its bench spot, lid on:

1. `dims`: outer x, y, z of the body (lid ON). `dims.z` sets the
   detector's expected lid height above the table and the container
   cuboid's height.
2. `lid_dims`: the lid alone (it is carried and placed as a cuboid).
3. `button_offset`: bottom-center origin → button TOP center (the flush
   button top IS the container top on this box).
4. `button_diameter_m`: caliper the round button — it sizes the circle
   the detector aims the press at.
5. `press.touch_nm`: the press guard's trip threshold. Bracketed at the
   bench (6.1 stopped short of the seal, 8.1 scooted the box); tune only
   with `torque_peak` from the run log in front of you.
6. `press_demo.travel_m`: press the button by hand with a caliper — the
   travel at which the lid releases, plus ~2 mm margin. The stroke is
   position-controlled with the torque guard as a stop; too-large travel
   means the guard is your only brake.
7. `open_container.lid_place`: a clear spot for the lid, inside the reach
   band (README "Where the container may sit"). The mission slides the
   drop away from the DETECTED box when the configured spot is too close,
   and refuses (exit 4) when nothing in the set-down zone clears.
8. `open_box.grip_band`: close the fingers on the POPPED knob by hand and
   record the position feedback the runner prints (0.387 at the bench);
   the band must hold it and exclude 0.8 (closed on air). Also
   `grip_clear_m` (fingertip stop above the lid plane) and
   `grip_offset_xy` (a mm-scale trim) — read their comments.
9. Flip `measure_me: false` — the CLIs refuse `--execute` until then;
   that refusal is the worksheet's completion gate.
10. Scan pose (needs the camera AT the pose, so this comes after the
    flip; the scan leg is a transit into free air): run §4 steps 2–3,
    then `press_demo --execute --detect-only` with the container placed
    at a tape-measured spot. It reports every fix for 15 s and homes;
    the printed top-face centre vs the tape solves the wrist-mount error.
    If it reports no fix, check `ros2 topic hz
    /d405/d405/aligned_depth_to_color/image_raw` (driver up, aligned
    depth on?), then the status line's last reject reason, then
    `scan.xyz` — the camera sees ~0.6 × 0.4 m of bench around it at lid
    height. The default [0.42, 0, 0.45] plans from HOME. A failed detect
    always parks the arm home.

## 3. Every-session preflight

1. `ros2 run rammp_box_opening preflight` — must PASS:
   `/joint_states` fresh WITH effort fields (guarded legs refuse
   without them) and planner actions reachable. The `execute` param and
   controller states print as INFO — `robotiq_gripper_controller` must be
   listed and active, or T1 was launched without the gripper argument.
   No world is pushed here: every leg pushes the world it is planned
   against at plan time.
2. **Abort drill** (every session, no exceptions): start
   `ros2 run rammp_box_opening home_arm --execute` (type `yes`), then
   Ctrl+C mid-motion. PASS = the arm stops and holds immediately AND the
   CLI prints `cancel delivered; controller stops and holds` (that line
   is the server's confirmation, not a hope). A session does not proceed
   past a failed drill. The software half is stub-proven off-bench by
   `python3 scripts/abort_e2e.py` (isolated domain, no arm) — run it
   after any client/runner/CLI change, BEFORE burning bench time on the
   live drill.

## 4. The mission (owner design 2026-08-24; press proven live 2026-08-25)

Attended, e-stop in hand. `--execute` alone arms it — NO typed
confirmation (owner decision: autonomous once started, Ctrl+C stops
everything; the abort drill above is the proof it does).

1. Off-bench, after any code change: rebuild, then
   `python3 scripts/abort_e2e.py && python3 scripts/press_demo_e2e.py`
   — both must PASS before bench time.
2. Bringup: arm (T1 above), then
   `ros2 launch rammp_box_opening press_demo.launch.py execute:=true`
   (planner + joint-state relay + OWL detector + D405 driver with aligned
   depth; kills strays). The OWL model loads once here, overlapping the
   planner's GPU init.
3. Preflight + abort drill (§3). Container in the reach band AND the
   camera's view zone (README), lid on, button flush, nothing else
   box-sized on the bench (two container-sized tops in view is refused
   as ambiguous unless the OWL bbox picks one).
4. Dry-run: `ros2 run rammp_box_opening press_demo` — read the leg
   preview and the BOX line (container origin must match reality to
   ~1 cm; if not, stop and check the mount / `table_z`).
5. `ros2 run rammp_box_opening press_demo --execute`. Expected, one log
   line per state: SCAN → BOX at … (fix committed; `press:close`
   dispatched) → MERGED PRESS, one stroke, "guard stopped the stroke" or
   "full travel" both = PRESSED (the lid should visibly release) →
   retreat to the hop, `grip:open` on arrival → GRIP: guarded descent,
   band-verified close, LID PULLED → PLACE: carry, guarded set-down
   ("surface felt … set down" at 4.0 Nm), release, retreat to the carry
   height, home → DONE, exit 0. No box → the arm parks home and it exits 2.
   Pre-detection legs (scan, no-box home) plan above an unseen-container
   keep-out band covering the whole placement zone to container height —
   a failed detection never sweeps low through the container it could
   not see. Keep OTHER tall objects out of the band.
   `--press-only` stops after the press (retreat to staging, home).
6. Repeat from different container positions in the band. Exit
   criterion: repeatable pressed-and-opened runs, verified in
   `~/.ros/rammp_box_opening/runs/run-*.jsonl`.
7. Optional: `open_box.park_tool_down: true` rests the arm tool-down at
   the scan pose between runs (saves the wrist flip twice per run; the
   arm then hovers over the bench where the box is placed). `home_arm`
   still returns to HOME.

## 5. Retired: the Phase-1 primitive ladder

The per-primitive CLIs (approach / press / grasp / lift / place /
retreat / open_container / pickup_container) and the hand-measured
`bench_pose` they planned from were retired with the Phase-0/1 tier once
the camera-driven mission ran live. `home_arm` is the only isolated
mover left (§3). Exercise the phases through `press_demo` itself:
`--detect-only` for perception, `--press-only` for the press, the full
run for grip and place.

## 6. When something trips

- The runner stops at the first unexpected outcome with the arm holding
  and prints leg name, outcome, torque peak, progress. The same row is
  appended to `~/.ros/rammp_box_opening/runs/run-*.jsonl` (the metrics
  source — do not delete).
- `press:down` "guard tripped EARLY at N% of the stroke" → the stroke
  struck something above the button: the box is not where the fix said
  (scooted by a previous press, or a wrong `table_z`/mount), or something
  is in the corridor. Stop, look, re-place; re-check the BOX line.
- `grip:down` trips (obstruction) → the open fingers struck the knob or
  rim instead of straddling it — the press scooted the box, or
  `grip_offset_xy` is off. `grip:close` "grip X vs band" fails → closed on
  air (0.8: the knob was not popped, or the fix was off) or on something
  too wide.
- `place:lid:down` "full stroke with no trip — never felt the surface" →
  the drop spot is higher than modelled or `setdown_touch_nm` is too
  high; "tripped at N% … set-down NOT confirmed" → struck something on
  the way down. The lid is still held; clear the area, `home_arm`.
- No-motion fault lore: the transient kortex fault fires at controller
  goal TRANSITIONS. Merged runs have zero transitions; the runner retries
  once from standstill only on the server's "never left the start"
  signature. Repeated faults on isolated goals → check the arm, not the
  code.
- Guard refuses to run: `/joint_states` has no effort fields — the arm
  bringup is wrong, not the guard.
- press_demo NO BOX with the container plainly in view: read the status
  line — "camera streams missing" means the driver (aligned depth
  included) isn't up; "0/N frames … (last reject: …)" names the gate that
  refused: "no points at container height" → `table_z` or `dims.z`;
  "footprint" → `dims` xy; "touches the image border" → move the box
  toward the scan axis; "ambiguous" → a second box-sized top is in view
  and the OWL node did not pick one; "camera moving" only → the arm never
  parked still.
- A failed post-touch replan (retreat or home refused from the contact
  pose) stops the mission with the arm holding — there is no plan-free
  fallback. Clear the area and run `home_arm --execute` at low speed; if
  that is refused too, jog clear by hand first.
- press_demo refuses to start "N rad from any rest pose" → the previous
  run ended badly. Recover with `home_arm --execute` before anything
  else; never plan a mission from wreckage.
