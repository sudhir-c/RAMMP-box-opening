# Hardware bringup and Phase-1 attended sessions

Every motion in this document is run BY A HUMAN with a hand on the
physical e-stop. Agent sessions prepare, analyze logs, and never execute.
Spec: `docs/superpowers/specs/2026-08-14-box-opening-design.md` (§6, §8).

## 1. Session preconditions

- Exactly ONE arm stack at a time. Check before starting:
  `pgrep -fa 'kortex|planner_node'` — note this Jetson has carried a
  root-owned planner_node from `/opt/rammp_curobo` (same `/rammp_curobo`
  names!); confirm with Chris which instance owns the session before
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

### 2c. Gripper width→command calibration + grip bands

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

## 4. The attended ladder (Phase 1, spec §8)

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

## 5. When something trips

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
- Post-contact plan failure triggers the plan-free reverse-retrace of the
  executed descent. If THAT is refused by the server, the arm holds:
  clear the area, home manually at low speed.
