# Box opening with the RAMMP Kinova Gen3 — design

Date: 2026-08-14. Status: approved by Chris in four interactive sections
(decision log in §9), then revised after a three-reviewer adversarial
self-review (consistency, fact-check against sources, design attack);
this is rev 2 with those fixes applied.

## 1. Goal and scope

Open an OXO POP container on the bench with the Kinova Gen3 7-DoF +
Robotiq 2F-85, then pick the container up — as parameterized, composable
primitives (approach, press, grasp, lift, place), not one-off scripts.

Two deliverable tasks, in order:

1. `open_container` — press the lid button to release the seal, grasp the
   lid, lift it clear, set it down.
2. `pickup_container` — grasp the container body and lift it, composed
   from the same primitives. Default behavior places the container back
   at its pickup pose (cycle-friendly for Phase-3 metrics); `--hold`
   keeps it lifted.

**Scope is box opening only.** The package is named for it
(`rammp_box_opening`); there is no "general ADL platform" framing. The
primitives are still parameterized and composable — that is how task 2
reuses task 1's pieces — but generalization beyond these two tasks is
explicitly out of scope.

The **first implementation plan covers Phases 0–1 only** (§8). Phases 2
and 3 get their own plans once Phase-1 measurements and the camera
prerequisites exist.

### Non-goals

- No modifications to RAMMP-CuRobo (change requests go to Chris; §10
  lists one candidate).
- No arm driver, bringup, or controller code — execution ownership stays
  with ros2_kortex from `~/RAMMP-Kinova/ros2_ws`, launched by the human.
- No autonomous motion from agent sessions, ever. Motion CLIs are run by
  the human, dry-run by default, with a human on the physical e-stop.

## 2. The planning service we consume (contract summary)

`~/RAMMP-CuRobo` is the planning SERVICE; this repo is a client. With the
planner launched (`ros2 launch rammp_curobo_ros planner.launch.py
config:=gen3_real.yaml [execute:=true]`):

| Interface | Type | Notes |
| --- | --- | --- |
| `/rammp_curobo/plan_to_pose` | `PlanToPose` action | `geometry_msgs/Pose` in base_link, **xyzw** quat, EE = `tool_frame`; optional `start_joints` for chained pre-planning |
| `/rammp_curobo/plan_to_joints` | `PlanToJoints` action | joint-space goal; check `goal_mismatch_rad` before executing anything that assumes exact joints |
| `/rammp_curobo/execute_trajectory` | `ExecuteTrajectory` action | `speed_scale`: 0.0 is the "use server default (0.25)" sentinel; any other value outside (0, 1] is REFUSED, not clamped. Live start-state match (0.05 rad); cancel = controller stop+hold; feedback = progress + joint_states |
| `/rammp_curobo/set_world` | `SetWorld` srv | world YAML path or packaged name; **cuboids only** (v0.7.8 drops other shapes); plan-time concern only; write-only — there is no query, so the runner must track what it last set |
| `/rammp_curobo/open_gripper` / `close_gripper` | `Trigger` srv | full open/close only; refused when the node is dry-run |
| `/robotiq_gripper_controller/gripper_cmd` | `GripperCommand` action | graded closes: position 0.0 (open, ≈85 mm aperture) … 0.8 (closed), used directly — NOT gated by the planner's `execute` param (see §6 gates) |

`tool_frame` sits ≈ fingertip midpoint, but the 2F-85 pad-face center is
a further **0.021 m along tool z** (sim-measured; the correction ships
DISABLED in gen3.yaml until re-measured — see `tip_to_tool` in
`geometry.py`). Poses authored at fingertip level carry that systematic
bias on top of the ~2 cm calibration uncertainty; this design absorbs
both with guarded descents and bench-measured config offsets, never with
open-loop depth.

Client-relevant server behavior (verified in `planner_node.py`):

- Plan results are time-parameterized at FULL speed; slowdown is
  execution-side `speed_scale`.
- One plan and one execution at a time (busy goals abort with a message).
- The `execute` parameter (default false) gates trajectory execution and
  the two Trigger gripper services, and is read live (`ros2 param get`
  works — the runner uses this, §6).
- A no-motion abort (message contains "never left the start") triggers
  the server's own servoing recovery before the abort returns — a client
  retry lands on a live arm.

## 3. Environment invariants (source of truth: `~/RAMMP-CuRobo/CLAUDE.md`)

- Jetson AGX Orin "abra" (192.168.1.11), ROS 2 Humble, zsh (`setup.zsh`
  files only). `export ROS_LOCALHOST_ONLY=1` in EVERY ROS shell,
  explicitly (non-interactive shells skip `~/.zshrc`).
- Arm: Gen3 at 192.168.1.10, ros2_kortex bringup from RAMMP-Kinova ws.
  Exactly one arm stack at a time.
- Joint reports wrap to (−π, π]; joint_3 sits AT +π at home — every angle
  comparison goes through `ang_diff`.
- Quaternion orders: ROS interfaces are xyzw; cuRobo — and the helpers
  `tool_axis` / `spin_about_tool` — take **wxyz**. Convert explicitly via
  `xyzw_to_wxyz`. (Trap: at the wrist-flat `[0.5, 0.5, 0.5, 0.5]` both
  orders coincide, so an order bug passes early bench tests.)
- Tool-tip precision is UNCALIBRATED beyond ~2 cm — approach slow, detect
  contact by torque, never trust open-loop depth for the last cm.
- Home elbow family (joint_3 ≈ π) is the good IK family; wrist-flat
  orientation is `Rz(bearing) ⊗ [0.5, 0.5, 0.5, 0.5]`
  (`yaw_about_world_z`).
- Speed 1.0 = full rated speed; new motions start at 0.15–0.25; slowing
  down is execution-side time dilation only.
- Transient kortex no-motion fault fires at controller goal TRANSITIONS —
  merge chained motions; retry once from a standstill only (§6 retry).
- The collision world must match reality before any cartesian goal near
  surfaces; err tall on tables (§6 worlds).
- pytest needs `-p no:anyio` on this machine.

## 4. Repo layout

```
~/RAMMP-box-opening/                  # git repo = colcon workspace
├── src/
│   └── rammp_box_opening/            # ament_python — the ONLY package here
│       ├── package.xml               # <depend>rammp_curobo_interfaces</depend>
│       ├── setup.py, setup.cfg, resource/
│       ├── rammp_box_opening/
│       │   ├── runtime/              # client.py (node + action clients),
│       │   │                         #   runner.py (legs, merging, retry, gates),
│       │   │                         #   guards.py (torque guard, grip checks,
│       │   │                         #   trajectory sanity gate),
│       │   │                         #   confirm.py (dry-run + typed-confirm gates)
│       │   ├── primitives/           # approach, press, grasp, lift, place,
│       │   │                         #   retreat, home (guarded descent machinery
│       │   │                         #   shared via runtime/guards.py)
│       │   ├── tasks/                # open_container.py, pickup_container.py
│       │   ├── models/               # container model + pose derivation
│       │   └── worlds.py             # world generation (see §6), set_world client
│       ├── config/
│       │   ├── containers/oxo_pop.yaml
│       │   ├── cameras/              # Phase 2 only (D405 + Orbbec yamls)
│       │   └── world_bench.yaml
│       └── test/                     # offline pytest
├── docs/superpowers/specs/           # this document
├── pytest.ini                        # addopts = -p no:anyio
├── .pre-commit-config.yaml           # mirrored from RAMMP-CuRobo
├── .gitignore                        # build/ install/ log/
└── README.md                         # sourcing chain, runbook pointers
```

**Interfaces come from the overlay, not a copy.** `rammp_curobo_interfaces`
is resolved by sourcing `~/RAMMP-CuRobo/install/setup.zsh` before building
and running — no copied code, no submodule, single source of truth. The
Docker image route was considered and rejected for this bench: the image
needs ~25 GB free to build (~15 GB after) and RAMMP-CuRobo's own
`docker/README.md` states this Jetson deliberately does not host it; and
even the Docker path requires the client to build the interfaces package
itself. The client only addresses `/rammp_curobo/*` action names, so a
future Dockerized planner on another host changes nothing here.

Build:

```zsh
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source ~/RAMMP-CuRobo/install/setup.zsh
colcon build --symlink-install
```

Runtime (execution shells) uses the runbook chain plus ours: humble →
RAMMP-Kinova ws → RAMMP-CuRobo → this ws.

Conventions carried from RAMMP-CuRobo: Ruff v0.3.0 defaults via
pre-commit, mdformat/gfm, Apache-2.0, maintainer
`RAMMP <chrisman4247@gmail.com>` (set as repo-local git identity — the
machine's global identity belongs to someone else). Geometry helpers are
imported from the installed `rammp_curobo` pip core — consumed, not
forked: `ang_diff` and `yaw_about_world_z` as `tour_demo.py` does, plus
`tool_axis`, `tip_to_tool`, `xyzw_to_wxyz` (mind the wxyz orders, §3).

CLI surface (`ros2 run rammp_box_opening …`): one entry point per task,
one per primitive (isolated attended bring-up), plus `smoke_plan`
(Phase 0 planner round-trip) and `preflight` (controllers responding,
`/joint_states` fresh with effort fields present, planner reachable,
planner `execute` param read and reported, and an idempotent `set_world`
push of the full world — `SetWorld` is write-only, so preflight
establishes the world rather than querying it).

### Reference implementations (normative sources)

| Pattern | Where |
| --- | --- |
| `merge_trajectories`, chained `start_joints`, no-motion retry | live file `~/RAMMP-CuRobo/rammp_curobo_ros/rammp_curobo_ros/tour_demo.py` |
| standalone client, dry-run CLI, typed confirm | live file `~/RAMMP-CuRobo/examples/plan_and_execute.py` |
| torque guard (contact detection) | recovered: `git -C ~/RAMMP-CuRobo show '0e0ec5b~1:rammp_curobo_ros/rammp_curobo_ros/palm_demo.py'` |
| depth→base_link, robot self-filter (Phase 2) | recovered: `git -C ~/RAMMP-CuRobo show '0e0ec5b~1:rammp_curobo_ros/rammp_curobo_ros/scan_common.py'` |
| D405 wrist mount calibration (Phase 2) | recovered: `git -C ~/RAMMP-CuRobo show '0e0ec5b~1:rammp_curobo_ros/config/camera_d405_wrist.yaml'` |

## 5. Primitive API

Every primitive is a small class with a plan/execute split so a whole
task can be pre-planned and previewed before anything moves:

```python
class Primitive(Protocol):
    def plan(self, ctx: Ctx, start: Joints) -> tuple[list[Leg], Joints]:
        """Chain-plan from `start`; returns legs + predicted end joints
        (fed as the next primitive's start — tour_demo's chaining)."""

@dataclass
class Leg:
    name: str                    # e.g. "press:down"
    kind: Kind                   # MOTION | GRIPPER
    plan: PlanResult | None      # MOTION: planner trajectory (full speed)
    speed: float                 # execution speed_scale for this leg
    guard: TorqueGuard | None    # stop-on-contact (wrist Nm threshold)
    world: WorldSpec             # world this leg was PLANNED against —
                                 #   a checked precondition, not a note (§6)
    chain: ChainId               # planning chain this leg belongs to (§6 merge)
    invalidates_downstream: bool # contact legs: later pre-plans are stale
    verify: Check | None         # runs when the leg completes; a leg with
                                 #   verify CLOSES its merge group (§6)
```

**Guarded descent** is the shared contact mechanism (in
`runtime/guards.py`), used by three primitives with different trip
semantics. It is a short, slow (≤0.15), torque-guarded MOTION leg planned
from a hover pose **directly above** the target (bounded standoff), and
it is the ONLY way this design ever closes the last centimeters — per the
§3 invariant, never open-loop.

Primitives (all poses in base_link, wrist-flat orientation family):

- `approach(pose)` — `plan_to_pose` to a hover/staging pose (never to
  contact depth).
- `press(target, depth_window=(min, max), touch_nm)` — guarded descent
  onto the button with the CLOSED gripper's fingertips. Guard trip with
  descent progress inside `depth_window` = success (button pressed);
  trip BEFORE `min` = failure (hit rim/edge) → retreat; reaching `max`
  untripped = failure → retreat. Window and threshold live in the
  container config.
- `grasp(width_m, expect_band)` — guarded descent from hover to grasp
  height where a trip = FAILURE (fingers struck the lid/rim —
  mispositioned) → retreat; then a graded `GripperCommand` close to
  `width_m` (physical aperture in meters, mapped linearly to the 0.0–0.8
  command with a bench-calibrated map in config); verify position
  feedback inside `expect_band` (fully closed = grasped air = fail).
- `lift(dz)` — planned ascent; re-checks the grip band afterward (slip).
- `place(pose)` — transit to hover above `pose`, guarded descent where a
  trip = SUCCESS (set-down detected) → gripper open → retreat.
- `retreat(dz)` — vertical disengage. After a contact stop this must not
  depend on planning from a possibly-in-collision start (§6 worlds).
- `home()` — return to HOME joints via `plan_to_joints`.

Tasks are thin compositions:

- `open_container` = approach(above_button) → press(button) → retreat →
  approach(above_lid) → grasp(lid) → lift(0.10 m) → place(lid_spot) →
  home.
- `pickup_container` = approach(above_body_grasp) → grasp(body) → lift →
  place(pickup pose by default; `--hold` skips place) → home.

Neither task computes a pose itself. All poses derive from
`ContainerModel` (`config/containers/oxo_pop.yaml`: outer dimensions,
button offset from container origin, lid-rim and body grasp
offsets/widths, expected grip bands, press depth window — measured at
Phase-1 bench time) combined with a `PoseSource`: Phase 1 a hand-measured
container pose in config, Phase 2 a perception-based detection
(method chosen at Phase-2 planning). Same interface,
so task code does not change between phases. The one pose that is not
container-derived — the lid set-down spot — lives in the task section of
the container config (`open_container.lid_place`), overridable by CLI
flag.

**Contact honesty:** a guarded descent ends wherever contact stopped it,
so joint-state prediction breaks there. Legs after a contact leg
(`invalidates_downstream`) are re-planned from the live arm at execution
time; dry-run previews them from the nominal contact depth. Chained
pre-planning applies to every stretch between contact events, not across
them.

**Straightness honesty:** the contract has no cartesian/linear planner —
`plan_to_pose` may return a curved path. Descents and lifts are therefore
(a) short (hover directly above the target, standoff bounded by config),
(b) slow and guarded, and (c) checked by the runner's trajectory sanity
gate (§6) before execution. If bench experience shows curved descents
survive all three, a linear-segment capability becomes a change request
to RAMMP-CuRobo (§10) — not a local workaround.

## 6. Runner, safety gates, error handling

The runner owns the single ROS node, the three planner action clients,
and a `GripperCommand` client. Client plumbing reuses the proven
`spin_until_done` / cancel-on-Ctrl+C patterns.

**Merging.** MOTION legs merge into ONE trajectory (`merge_trajectories`
pattern from the live tour_demo.py) only when dynamically valid: leg B
must belong to the **same planning chain** as leg A — B was planned with
`start_joints` equal to A's predicted endpoint — with the same speed and
no guard on either; a leg carrying `verify` closes its merge group. A leg
re-planned from the live arm starts a NEW chain (a seam between chains
merged naively would be a discontinuity the server rejects). `world` is
bookkeeping for the plan-time invariant below, never a merge key. Guarded
legs always execute alone. Merged runs have zero controller-goal
transitions — the no-motion-fault mitigation.

**Worlds are plan-time, generated per leg batch.** `worlds.py` takes
`config/world_bench.yaml` (the measured bench geometry — the single
source of truth for this project; reconciled once at Phase-1 bench time
with RAMMP-CuRobo's `world_real_bench.yaml`, after which OUR generated
worlds are what the planner uses) and extends it with container-derived
cuboids, writing variants under `~/.ros/rammp_box_opening/worlds/` and
pushing via `set_world` before each planning batch:

- *full* — bench + full container cuboid (err tall). Required for ALL
  transit-speed and unguarded legs — that is a **checked precondition**:
  the runner refuses to plan or execute an unguarded/transit leg whose
  `world` is not *full* (`SetWorld` is write-only, so the runner tracks
  what it last pushed and re-asserts rather than queries).
- *interaction(target)* — generated per contact target, not fixed:
  the reduction height derives from the target (button press, lid rim,
  mid-body grasp — task 2's body grasp needs a lower opening than the
  press does). Where geometry allows, the opening is an **aperture ring
  of cuboids** around the descent corridor rather than wholesale removal,
  so lateral entry stays forbidden while vertical entry is allowed.
  The reduction must leave the planner able to plan FROM the deepest
  commanded contact point (reduction plane ≥ `depth_window.max` +
  calibration margin below the lowest commanded pose) so post-contact
  re-planning never starts in collision; if a post-contact plan fails
  anyway, the fallback retreat is plan-free — reverse-retrace of the
  executed portion of the descent, revalidated by the server's own gates.
- After `place(lid)`, a lid cuboid is added at the place pose for all
  subsequent legs, and lift/place hover heights include the lid's own
  dimensions as margin — the planner cannot model a held object, so
  while the lid is in the gripper, clearances carry it explicitly.

**Trajectory sanity gate** (client-side, before executing any MOTION
leg): per-joint excursion beyond the start→end delta plus a configured
margin is refused — the cheap joint-space proxy for "the planner
wandered". Cartesian-intent legs (descents, lifts) additionally require
the bounded standoff of §5. Hard refusals, like the no-effort refusal
below.

**Torque guard** (palm-demo pattern with three hardening changes): watch
`/joint_states` efforts on joints 4–7; trip when max per-joint deviation
from baseline exceeds `touch_nm` (default 3.0 Nm) → cancel the
ExecuteTrajectory goal → controller stops and holds → leg resolves as
`touch`. Hardening vs the original: (a) the baseline is anchored at the
first ExecuteTrajectory feedback with `progress > 0` — not at
goal-accept, which can land the 0.4 s snapshot inside the acceleration
transient or after server latency; (b) the hover standoff must exceed
pose uncertainty (~2 cm + the 21 mm tip bias) plus the distance traveled
during the baseline window at 0.15 speed — refused otherwise — so the
baseline can never be captured already in contact; (c) the cancel-path
latency (client → server 10 Hz poll → controller) is budgeted inside
`depth_window.max`. If effort fields are absent from `/joint_states`,
guarded primitives REFUSE to run (no silent position-only degradation).

**Gates.** Dry-run is the default everywhere — plan everything, print a
per-leg excursion table (wrap-aware `ang_diff`) and timing summary.
Motion requires `--execute` AND a typed `yes`. For MOTION legs the
planner's `execute:=true` is a third, server-side gate. **GRIPPER legs
have no server-side gate** — the direct `GripperCommand` action bypasses
the planner — so the runner enforces symmetry itself: it reads the
planner's live `execute` parameter and refuses GRIPPER legs while the
planner is dry-run, and gripper-capable CLIs state plainly that planner
dry-run alone does not prevent gripper motion. Speed defaults: 0.25
transit, 0.15 contact. Ctrl+C anywhere cancels the active goal → arm
holds; the runner never auto-continues past a cancel. All motion CLIs are
run by the human.

**Outcomes.** Every leg resolves to `arrived | touch | failed` plus its
verification result. First unexpected outcome stops the task with the arm
holding and an exact report (leg name, outcome, torque peak, progress).
Two narrow recoveries, nothing else:

- *No-motion retry:* execution aborted AND the abort message carries the
  no-motion signature ("never left the start") → retry once from
  standstill. Displacement alone is NOT sufficient — a partial-motion
  abort that stopped early may be physical contact, and re-pushing into
  an obstacle is exactly wrong.
- *Start-drift re-plan:* the server's arrival tolerance (0.08 rad) is
  wider than its start gate (0.05 rad), so a leg can "arrive" yet leave
  the next pre-planned leg refusable. Before executing any pre-planned
  leg the runner compares live joints to the leg start (wrap-aware);
  above 0.04 rad it re-plans that leg from live state to the same
  endpoint (starting a new chain) and continues.

**Run log.** One JSONL line per leg (timestamps, outcome, torque peak,
progress, speeds) appended under `~/.ros/rammp_box_opening/runs/` — the
Phase-3 metrics source and the hardware debugging record.

## 7. Testing

1. **Offline pytest** (`-p no:anyio`; needs sourced interfaces, no
   running nodes): merge validity (chain identity, verify-closes-group,
   time offsets), container-model pose derivation (including wxyz/xyzw
   conversions), torque-guard logic on synthetic `JointState` streams
   (baseline anchoring, early-trip, in-window trip, untripped-at-max),
   trajectory sanity gate, leg sequencing + downstream invalidation +
   start-drift re-plan against a fake planner client, world generation
   (cuboids only, aperture rings, reduction-plane rule, lid cuboid after
   place).
2. **Phase-0 smoke** (planner running, no arm): `smoke_plan` plans a pose
   with explicit `start_joints` and prints the trajectory — proves build,
   discovery, and the contract end to end.
3. **Sim (optional, honest limits):** the MuJoCo bringup exposes the same
   controllers, so transit motions, merged execution, the abort drill,
   and gripper mechanics can be validated in sim. Contact legs cannot be
   (no container exists in the sim world, and adding one would mean
   modifying RAMMP-Kinova).
4. **Hardware, attended** (per `docs/HARDWARE_BRINGUP.md`, human on the
   e-stop): `preflight` + abort drill every session, per-primitive CLIs
   before task CLIs, first runs ≤ 25% speed.

## 8. Roadmap

- **Phase 0 — scaffold + smoke (no arm):** repo + package skeleton,
  overlay build, pytest green, `smoke_plan` round-trip against the
  planner. Proves: workspace consumes the service correctly.
- **Phase 1 — choreographed opening at a known pose (attended):** measure
  the container pose + model by hand into config (dimensions, button
  offset, rim/body grasp offsets, press depth window, grip bands, the
  gripper width→command calibration). Attended ladder, each step stating
  what it proves: abort drill → `approach` → `press` (guarded, real
  container) → `grasp` → `lift` → `place` → full `open_container`. Goal:
  ONE clean end-to-end opening before any perception.
- **Phase 2 — perception:** camera yamls for BOTH cameras. D405 on the
  wrist (recovered mount calibration, re-verified before trust) is the
  close-range manipulation view; the Orbbec Gemini 336L provides a static
  second view for container pose when the wrist camera is occluded or too
  close during interaction — its driver is ALREADY installed system-wide
  (`ros-humble-orbbec-camera 2.8.6`, `gemini_330_series.launch.py`), so
  the remaining cost is configuration and extrinsics.
  Depth→base_link via the recovered `scan_common.py` pipeline. Detection
  method is chosen at Phase-2 planning on task merit — fiducial (ArUco /
  AprilTag) and tag-free / learned detection are all in scope. Detected
  container pose → derived button/lid/grasp poses + container cuboid
  pushed via `set_world`. Planned separately after Phase 1.
- **Phase 3 — robustness + task 2:** randomized placement within reach,
  per-primitive retry with verification, `pickup_container`. Success
  metric: **5 consecutive** open+pickup cycles from random placements
  (Chris may raise N at Phase-3 kickoff), zero human touches except the
  e-stop hand, measured from the run logs. Planned separately.

## 9. Decision log (approved by Chris, 2026-08-14)

1. Architecture: layered library (primitives / runner / tasks) over
   scripts-first and declarative-framework alternatives.
2. Scope: box opening ONLY — package named `rammp_box_opening`, no ADL
   generalization.
3. Interfaces: overlay `~/RAMMP-CuRobo/install` — no copy, no submodule.
   Docker rejected for this bench (it wouldn't remove the client-side
   interfaces need anyway).
4. Phase-2 cameras: both — D405 (will be plugged in, wrist) and Orbbec
   Gemini 336L (static second view).
5. ~~Fiducial: ArUco via existing opencv-contrib instead of an AprilTag
   install.~~ Superseded 2026-08-17 (see 7).
6. Git: repo-local identity `RAMMP <chrisman4247@gmail.com>`; local
   commits only until a remote is chosen (`gh` not installed).
7. (2026-08-17) Disk constraint lifted — more Jetson storage is coming,
   so "no model downloads / zero new installs" no longer gates design
   choices. Phase-2 detection method is picked on task merit at Phase-2
   planning; decision 5's install-driven rationale is superseded.

## 10. Risks and open items

- **OXO container geometry** (dimensions, button offset, rim grasp,
  depth window, grip bands, width→command map) is measured at Phase-1
  bench time — a planned activity, not a design gap.
- **Press verification tuning** (torque threshold, depth window, what a
  "click" looks like in the signature) needs real hardware data; the
  design fixes mechanisms, thresholds live in config.
- **No cartesian-linear planning in the contract.** Descents rely on
  short segments + the sanity gate + the guard. If that proves
  insufficient on hardware, the change request to RAMMP-CuRobo is a
  linear-segment (or path-constrained) planning capability — specified
  as a request to Chris, never patched locally.
- **D405 mount** must be re-verified before Phase 2 trust (bracket may
  have moved; the recovered yaml says re-verify after ANY bracket
  change).
- **Triple OpenCV installs** (`opencv-python` 4.10, `opencv-python-headless`
  4.10, `opencv-contrib-python` 4.11) risk shadowing; today `import cv2`
  resolves to 4.11 with `cv2.aruco` present — if Phase 2 chooses ArUco,
  re-verify before relying on it.
- **Gripper feedback bands** for lid-rim vs body grasps are unknown until
  measured; `expect_band` values live in the container model config.
