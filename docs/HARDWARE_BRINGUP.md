# Hardware bringup and Phase-1 attended sessions

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
  1. Arm driver: ros2_kortex bringup from `~/RAMMP-Kinova/ros2_ws`
     (its runbook owns this step; Gen3 at 192.168.1.10).
  2. Planner: `ros2 launch rammp_curobo_ros planner.launch.py
     config:=gen3_real.yaml execute:=true` from a shell sourcing
     humble → `~/RAMMP-CuRobo/install/setup.zsh`.
  3. Task shells: full chain humble → RAMMP-Kinova ws → RAMMP-CuRobo →
     `~/RAMMP-box-opening/install/setup.zsh`.
- The planner's `execute` parameter gates ARM motion server-side. It does
  NOT gate the gripper: gripper closes go over the direct
  `/robotiq_gripper_controller/gripper_cmd` action. The runner refuses
  gripper legs while the planner is dry-run, but treat any `--execute`
  run as capable of moving the gripper.

## 2. Measurement worksheet (once, before the first `--execute`)

All measurements are from base_link: +x forward, +z up, tape measure,
metres. Record into the two config files, then rebuild
(`colcon build --symlink-install` — symlinked YAMLs update in place, a
rebuild is only needed if files were added).

### 2a. Bench world — `src/rammp_box_opening/config/world_bench.yaml`

1. Table top height relative to base_link (err TALL: a table modeled too
   tall costs reachable volume, never safety) and extents.
2. Any wall/shelf within reach: add as cuboids, same schema.
3. Reconcile ONCE with `~/RAMMP-CuRobo`'s `world_real_bench.yaml` (theirs
   also carries placeholder geometry — measuring supersedes both; after
   this, OUR generated worlds are what the planner uses at runtime).

### 2b. Container model — `src/rammp_box_opening/config/containers/oxo_pop.yaml`

With the container at its bench spot:

1. `dims`: outer x, y, z of the body (lid ON).
2. `lid_dims`: the lid alone (it is carried and placed as a cuboid).
3. `button_offset`: bottom-center origin → button TOP center.
4. `lid_grasp.offset` / `body_grasp.offset`: origin → rim grasp point /
   mid-body grasp point. Widths: jaw opening that clears the feature plus
   margin; must be ≤ 0.085 m.
5. `press.depth_window`: press the button by hand with a caliper —
   `min` = travel where the seal audibly/tactilely releases, `max` = full
   bottom-out travel plus ~3 mm cancel-latency budget.
6. `bench_pose`: container bottom-center in base_link + yaw. **Place it
   inside the measured tool-down reach band** (README "Where the
   container may sit"; raw map `docs/reach_map.json`): button x between
   0.25 and ~0.70 m radial, y within ±0.45. After reconciling the bench
   geometry (2a), re-run `python3 scripts/reach_probe.py` (~2 min,
   offline) and re-check before committing the pose.
7. `open_container.lid_place`: a clear spot ≥ container-width away —
   also inside the reach band.
8. `hover_standoff`: keep ≥ 0.06 (must exceed the 0.051 m guard floor:
   2 cm pose uncertainty + 2.1 cm tip bias + 1 cm baseline travel).

### 2c. Press-demo measurements (the current milestone needs ONLY
### 2a items 1–3, 2b items 1–3, and this section)

1. Print `docs/tag0_50mm.png` at 100% / actual size, measure the black
   square with a ruler, record it as `tag.size_m` (PNGs carry no DPI —
   printers rescale silently; pose error scales directly with this).
2. Stick the tag centered ON the button top (`tag.offset_xyz` stays
   [0,0,0]; measure and set it only if the tag must sit off-button).
3. `press_demo.travel_m`: press the button by hand with a caliper —
   the travel at which the lid releases, plus ~2 mm margin. The stroke
   is position-controlled with the torque guard as a stop; too-large
   travel means the guard (touch_nm) is your only brake.
4. Flip `measure_me: false` — the CLIs refuse `--execute` until then.
5. Scan pose (needs the camera AT the pose, so this comes after the
   flip; the scan leg is a transit into free air): run §4 steps 2–3,
   then `press_demo --execute` with the container placed. If DETECT
   reports no tag, check `ros2 topic hz /d405/d405/color/image_raw`
   (driver up?), then adjust `scan.xyz` until the status line reports
   sightings — the camera sees ~±0.30 m x, ±0.19 m y around the pose at
   tag height. The default [0.42, 0, 0.45] plans from HOME (verified
   offline 2026-08-24). A failed detect always parks the arm home.

### 2d. Gripper width→command calibration + grip bands (ladder only —
### the press demo just closes the fingers)

1. `ros2 run rammp_box_opening preflight` first (below) — planner
   `execute` must be true for gripper motion.
2. Close on nothing to 0.8, open to 0.0. Measure the physical aperture at
   command 0.0 → `gripper_map.aperture_at_0` (≈0.085) and confirm 0.8 is
   fully closed → `aperture_at_08` (0.0; if the fingers stop short,
   record the residual aperture).
3. Close on two gauge objects of known width (e.g. the container wall and
   a 20 mm block) and record the position FEEDBACK the runner prints —
   these anchor `expect_band` for `lid_grasp` (feedback when holding the
   rim) and `body_grasp` (holding the body). Fully-closed feedback =
   grasped air; keep each band's upper edge below it.
4. Flip `measure_me: false`. The CLIs refuse `--execute` until this is
   done — that refusal is the worksheet's completion gate.

## 3. Every-session preflight

1. `ros2 run rammp_box_opening preflight` — must PASS:
   `/joint_states` fresh WITH effort fields (guarded primitives refuse
   without them), planner actions reachable, full world pushed
   (`SetWorld` is write-only: preflight ESTABLISHES the world). The
   `execute` param and controller states print as INFO.
2. **Abort drill** (every session, no exceptions): start
   `ros2 run rammp_box_opening home_arm --execute` (type `yes`), then
   Ctrl+C mid-motion. PASS = the arm stops and holds immediately AND the
   CLI prints `cancel delivered; controller stops and holds` (that line
   is the server's confirmation, not a hope). A session does not proceed
   past a failed drill. The software half is stub-proven off-bench by
   `python3 scripts/abort_e2e.py` (isolated domain, no arm) — run it
   after any client/runner/CLI change, BEFORE burning bench time on the
   live drill.

## 4. The press demo (current milestone, owner design 2026-08-24)

Attended, e-stop in hand. `--execute` alone arms it — NO typed
confirmation (owner decision: autonomous once started, Ctrl+C stops
everything; the abort drill above is the proof it does).

1. Off-bench, after any code change:
   `python3 scripts/abort_e2e.py && python3 scripts/press_demo_e2e.py`
   — both must PASS before bench time.
2. Bringup: arm (ros2_kortex), then
   `ros2 launch rammp_box_opening press_demo.launch.py execute:=true`
   (planner + D405 driver with aligned depth; kills stray planners).
3. Preflight + abort drill (§3). Container in the reach band AND the
   camera's view zone (README map), tag up.
4. Dry-run: `ros2 run rammp_box_opening press_demo` — read the leg
   preview and the TAG line (container origin must match reality to
   ~1 cm; if not, stop and check `tag.size_m` / the mount).
5. `ros2 run rammp_box_opening press_demo --execute`. Expected: scan,
   fix (+servo), close+staging, one press stroke ("guard stopped" or "full
   travel" both = pressed — the lid should visibly release), retreat,
   home, exit 0. No tag → the arm parks home and it exits 2.
   Pre-detection legs (scan, no-tag home) plan above an
   unseen-container keep-out band covering the whole placement zone to
   container height — a failed detection never sweeps low through the
   container it could not see. Keep OTHER tall objects out of the band.
6. Repeat from different container positions in the band. Exit
   criterion: repeatable pressed-and-released runs, verified in
   `~/.ros/rammp_box_opening/runs/run-*.jsonl`.

## 5. The attended ladder (Phase-1 choreography — SUPERSEDED as the
## milestone by §4; kept for the grasp/lift/place future)

Rules: dry-run every rung FIRST (no `--execute`) and read the leg
preview; first `--execute` of a new rung at default speeds (0.25 transit
/ 0.15 contact) or slower; one rung at a time; after any unexpected stop,
read the report and the run log before re-trying.

| Rung | Command | Proves |
| --- | --- | --- |
| 1 | `approach --execute` | transit planning + worlds sane, hover is where the button is |
| 2 | `press --execute` | guarded descent, baseline anchoring, click classification on the real container |
| 3 | `grasp --grasp lid --execute` | obstruction-trip semantics, graded close, expect_band correctness |
| 4 | `lift --execute` | ascent + slip re-check band |
| 5 | `place --execute` | set-down trip, release, lid world cuboid |
| 6 | `open_container --execute` | the whole choreography, merged legs, one clean end-to-end opening |

Phase-1 exit: ONE clean `open_container` run — every leg `ok` in the run
log, press classified `pressed`, grasp band held through lift.

## 6. When something trips

- The runner stops at the first unexpected outcome with the arm holding
  and prints leg name, outcome, torque peak, progress. The same row is
  appended to `~/.ros/rammp_box_opening/runs/run-*.jsonl` (the Phase-3
  metrics source — do not delete).
- `press` reports `rim/edge contact before window` → the button offset or
  bench_pose xy is off; re-measure before touching thresholds.
- `no click detected` (bottomed out untripped) → depth_window.max too
  small or touch_nm too high. Tune in the CONTAINER CONFIG, never code.
- No-motion fault lore: the transient kortex fault fires at controller
  goal TRANSITIONS. Merged runs have zero transitions; the runner retries
  once from standstill only on the server's "never left the start"
  signature. Repeated faults on isolated goals → check the arm, not the
  code.
- Guard refuses to run: `/joint_states` has no effort fields — the arm
  bringup is wrong, not the guard.
- press_demo NO TAG with the container plainly in view: check the CLI's
  status line — "camera streams missing" means the driver (aligned
  depth included) isn't up; "0/N frames" with frames flowing means tag
  id/size/lighting. "(RGB z only)" in the TAG line means depth
  refinement failed — trust the run less, check alignment.
- press_demo guard trip DURING hover or staging (before the stroke):
  something unexpected in the corridor — stop, look, re-place.
- Post-contact plan failure triggers the plan-free reverse-retrace of the
  executed descent. If THAT is refused by the server, the arm holds:
  clear the area, home manually at low speed.
