# Box opening with the RAMMP Kinova Gen3 — design

Date: 2026-08-14. Status: approved by Chris (all four sections reviewed
interactively; decisions logged at the bottom).

## 1. Goal and scope

Open an OXO POP container on the bench with the Kinova Gen3 7-DoF +
Robotiq 2F-85, then pick the container up — as parameterized, composable
primitives (approach, press, grasp, lift, place), not one-off scripts.

Two deliverable tasks, in order:

1. `open_container` — press the lid button to release the seal, grasp the
   lid, lift it clear, set it down.
2. `pickup_container` — grasp the container body and lift it, composed
   from the same primitives.

**Scope is box opening only.** The package is named for it
(`rammp_box_opening`); there is no "general ADL platform" framing. The
primitives are still parameterized and composable — that is how task 2
reuses task 1's pieces — but generalization beyond these two tasks is
explicitly out of scope.

### Non-goals

- No modifications to RAMMP-CuRobo (change requests go to Chris).
- No arm driver, bringup, or controller code — execution ownership stays
  with ros2_kortex from `~/RAMMP-Kinova/ros2_ws`, launched by the human.
- No autonomous motion from agent sessions, ever. Motion CLIs are run by
  the human, dry-run by default, with a human on the physical e-stop.
- No neural perception, no model downloads (disk is at 93%). Phase 2 is
  fiducial-based; tag-free detection is a possible follow-on, not planned
  here.

## 2. The planning service we consume (contract summary)

`~/RAMMP-CuRobo` is the planning SERVICE; this repo is a client. With the
planner launched (`ros2 launch rammp_curobo_ros planner.launch.py
config:=gen3_real.yaml [execute:=true]`):

| Interface | Type | Notes |
| --- | --- | --- |
| `/rammp_curobo/plan_to_pose` | `PlanToPose` action | `geometry_msgs/Pose` in base_link, **xyzw** quat, EE = `tool_frame` (≈ fingertip midpoint); optional `start_joints` for chained pre-planning |
| `/rammp_curobo/plan_to_joints` | `PlanToJoints` action | joint-space goal; check `goal_mismatch_rad` before executing anything that assumes exact joints |
| `/rammp_curobo/execute_trajectory` | `ExecuteTrajectory` action | `speed_scale` (0,1] — out-of-range is REFUSED, not clamped; live start-state match (0.05 rad); cancel = controller stop+hold; feedback = progress + joint_states |
| `/rammp_curobo/set_world` | `SetWorld` srv | world YAML path or packaged name; **cuboids only** (v0.7.8 drops other shapes); plan-time concern only |
| `/rammp_curobo/open_gripper` / `close_gripper` | `Trigger` srv | full open/close only; refused when the node is dry-run |
| `/robotiq_gripper_controller/gripper_cmd` | `GripperCommand` action | graded closes: position 0.0 (open) … 0.8 (closed), used directly |

Client-relevant server behavior (verified in `planner_node.py`):

- Plan results are time-parameterized at FULL speed; slowdown is
  execution-side `speed_scale` (default 0.25 when the goal sends 0.0).
- One plan and one execution at a time (busy goals abort with a message).
- The `execute` parameter (default false) gates all motion including the
  gripper services, and is read live (`ros2 param set` works).
- A no-motion abort ("never left the start") triggers the server's own
  servoing recovery before the abort returns — a client retry lands on a
  live arm.

## 3. Environment invariants (source of truth: `~/RAMMP-CuRobo/CLAUDE.md`)

- Jetson AGX Orin "abra" (192.168.1.11), ROS 2 Humble, zsh (`setup.zsh`
  files only). `export ROS_LOCALHOST_ONLY=1` in EVERY ROS shell,
  explicitly (non-interactive shells skip `~/.zshrc`).
- Arm: Gen3 at 192.168.1.10, ros2_kortex bringup from RAMMP-Kinova ws.
  Exactly one arm stack at a time.
- Disk: ~3.9 GB free of 57 GB. Phases 0–1 install nothing.
- Joint reports wrap to (−π, π]; joint_3 sits AT +π at home — every angle
  comparison goes through `ang_diff`. Quaternions are ROS xyzw everywhere
  in this stack.
- Tool-tip precision is UNCALIBRATED beyond ~2 cm — approach slow, detect
  contact by torque, never trust open-loop depth for the last cm.
- Home elbow family (joint_3 ≈ π) is the good IK family; wrist-flat
  orientation is `Rz(bearing) ⊗ [0.5, 0.5, 0.5, 0.5]`
  (`yaw_about_world_z`).
- Speed 1.0 = full rated speed; new motions start at 0.15–0.25; slowing
  down is execution-side time dilation only.
- Transient kortex no-motion fault fires at controller goal TRANSITIONS —
  merge multi-segment motions; retry once from a standstill only.
- `world_real_bench.yaml` must match reality before any cartesian goal
  near surfaces; err tall on tables.
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
│       │   │                         #   runner.py (legs, merging, retry),
│       │   │                         #   guards.py (torque guard, grip checks),
│       │   │                         #   confirm.py (dry-run + typed-confirm gates)
│       │   ├── primitives/           # approach, press, grasp, lift, place,
│       │   │                         #   retreat, home
│       │   ├── tasks/                # open_container.py, pickup_container.py
│       │   ├── models/               # container model + pose derivation
│       │   └── worlds.py             # bench world + container cuboid variants,
│       │                             #   set_world client
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
machine's global identity belongs to someone else). Angle/quaternion
helpers (`ang_diff`, `yaw_about_world_z`, `tool_axis`) are imported from
the installed `rammp_curobo` pip core exactly as `tour_demo.py` does —
consumed, not forked.

CLI surface (`ros2 run rammp_box_opening …`): one entry point per task,
one per primitive (isolated attended bring-up), plus `smoke_plan`
(Phase 0 planner round-trip) and `preflight` (controllers active,
`/joint_states` fresh, planner reachable, world set).

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
    kind: Kind                   # MOTION | GRIPPER | CHECK
    plan: PlanResult | None      # MOTION: planner trajectory (full speed)
    speed: float                 # execution speed_scale for this leg
    guard: TorqueGuard | None    # stop-on-contact (wrist Nm threshold)
    world: str                   # world variant this leg was planned against
    invalidates_downstream: bool # contact legs: later pre-plans are stale
    verify: Check | None         # post-leg verification
```

Primitives (all poses in base_link, wrist-flat orientation family):

- `approach(pose)` — `plan_to_pose` to a hover/staging pose.
- `press(target, max_depth, touch_nm)` — slow straight-down guarded
  MOTION leg. Guard trip *within the expected depth window* is success;
  reaching `max_depth` untripped is failure → retreat. Pressing tool is
  the CLOSED gripper's fingertips (palm-demo pattern).
- `grasp(width, expect_band)` — graded `GripperCommand` close; verify
  position feedback lands inside `expect_band` (fully closed = grasped
  air = fail).
- `lift(dz)` — straight up; re-checks the grip band afterward (slip).
- `place(pose)` — move + open.
- `retreat(dz)`, `home()` — disengage vertically; return to HOME joints
  via `plan_to_joints`.

Tasks are thin compositions:

- `open_container` = approach(above_button) → press(button) → retreat →
  approach(lid_grasp) → grasp(lid) → lift(0.10 m) → place(lid_spot) →
  home.
- `pickup_container` = approach(body_grasp) → grasp(body) → lift →
  optional place → home.

Neither task computes a pose itself. All poses derive from
`ContainerModel` (`config/containers/oxo_pop.yaml`: outer dimensions,
button offset from container origin, lid-rim and body grasp
offsets/widths — measured at Phase-1 bench time) combined with a
`PoseSource`: Phase 1 a hand-measured container pose in config, Phase 2 a
fiducial detection. Same interface, so task code does not change between
phases.

**Contact honesty:** a guarded press ends wherever contact stopped it, so
joint-state prediction breaks there. Legs after a contact leg
(`invalidates_downstream`) are re-planned from the live arm at execution
time; dry-run previews them from the nominal press depth. Chained
pre-planning applies to every stretch between contact events, not across
them — the planner's live start-state gate enforces this anyway.

## 6. Runner, safety gates, error handling

The runner owns the single ROS node, the three planner action clients,
and a `GripperCommand` client. Client plumbing reuses the proven
`spin_until_done` / cancel-on-Ctrl+C patterns.

**Merging.** Contiguous MOTION legs with the same speed, same world, and
no guard merge into ONE trajectory (`merge_trajectories` pattern,
recovered verbatim from tour_demo) — zero controller-goal transitions
inside a merged run. Guarded legs always execute alone.

**Worlds are plan-time.** `worlds.py` generates the bench world plus a
container cuboid (err tall) in two named variants: *full* (transit
planning) and *interaction* (container reduced below button/rim height so
press/grasp targets are not inside an obstacle — safe because those legs
run slow and guarded). The runner calls `set_world` before each planning
batch, including post-contact re-planning. Execution is unaffected by
world state.

**Torque guard** (recovered palm-demo pattern, verbatim semantics): watch
`/joint_states` efforts on joints 4–7; baseline is a raw snapshot 0.4 s
after motion start; trip when max per-joint deviation from baseline
exceeds `touch_nm` (default 3.0 Nm) → cancel the ExecuteTrajectory goal →
controller stops and holds → leg resolves as `touch`. If effort is absent
from `/joint_states`, the press primitive REFUSES to run (no silent
position-only degradation). Guarded legs cap at 0.15 speed.

**Gates, layered on the server's:** dry-run is the default everywhere —
plan everything, print a per-leg excursion table (wrap-aware `ang_diff`)
and timing summary. Motion requires `--execute` AND the planner launched
`execute:=true` AND a typed `yes`. Speed defaults: 0.25 transit, 0.15
contact. Ctrl+C anywhere cancels the active goal → arm holds; the runner
never auto-continues past a cancel. All motion CLIs are run by the human.

**Outcomes.** Every leg resolves to `arrived | touch | failed` plus its
verification result. First unexpected outcome stops the task with the arm
holding and an exact report (leg name, outcome, torque peak, progress).
The only automatic retry: execution aborted AND the arm moved < 0.05 rad
from the leg start (the transient no-motion fault) → retry once from
standstill. Partial-motion failures never retry.

**Run log.** One JSONL line per leg (timestamps, outcome, torque peak,
progress, speeds) appended under `~/.ros/rammp_box_opening/runs/` — the
Phase-3 metrics source and the hardware debugging record.

## 7. Testing

1. **Offline pytest** (`-p no:anyio`; needs sourced interfaces, no
   running nodes): merge time-offset math, container-model pose
   derivation, torque-guard logic on synthetic `JointState` streams, leg
   sequencing + downstream invalidation against a fake planner client,
   world-variant YAML generation (cuboids only).
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
  the container pose + model by hand into config. Attended ladder, each
  step stating what it proves: abort drill → `approach` → `press` (guarded,
  real container) → `grasp` → `lift` → `place` → full `open_container`.
  Goal: ONE clean end-to-end opening before any perception.
- **Phase 2 — perception:** camera yamls for BOTH cameras (D405 wrist
  mount from the recovered `camera_d405_wrist.yaml`, re-verified before
  trust; Orbbec Gemini 336L as a second view — its ROS 2 driver is not
  yet installed, a known cost). Depth→base_link via the recovered
  `scan_common.py` pipeline. Fiducial: **ArUco via the installed
  opencv-contrib** (approved deviation from the brief's "AprilTag" — same
  fiducial-first intent, zero new installs). Tag pose → derived
  button/lid/grasp poses + container cuboid pushed via `set_world`.
- **Phase 3 — robustness + task 2:** randomized placement within reach,
  per-primitive retry with verification, `pickup_container`. Success
  metric: N consecutive open+pickup cycles from random placements, zero
  human touches except the e-stop hand, measured from the run logs.

## 9. Decision log (all approved by Chris, 2026-08-14)

1. Architecture: layered library (primitives / runner / tasks) over
   scripts-first and declarative-framework alternatives.
2. Scope: box opening ONLY — package named `rammp_box_opening`, no ADL
   generalization.
3. Interfaces: overlay `~/RAMMP-CuRobo/install` — no copy, no submodule.
   Docker rejected for this bench (disk; and it wouldn't remove the
   client-side interfaces need anyway).
4. Phase-2 cameras: both — D405 (will be plugged in, wrist) and Orbbec
   Gemini 336L.
5. Fiducial: ArUco via existing opencv-contrib instead of an AprilTag
   install.
6. Git: repo-local identity `RAMMP <chrisman4247@gmail.com>`; local
   commits only until a remote is chosen (`gh` not installed).

## 10. Risks and open items

- **OXO container geometry** (dimensions, button offset, rim grasp) is
  measured at Phase-1 bench time — a planned activity, not a design gap.
- **Press verification tuning** (torque threshold, depth window, what a
  "click" looks like in the signature) needs real hardware data; the
  design only fixes the mechanism, thresholds live in config.
- **D405 mount** must be re-verified before Phase 2 trust (bracket may
  have moved; the recovered yaml says re-verify after ANY bracket
  change). The Orbbec needs a driver install — evaluate disk cost then.
- **Dual OpenCV installs** (`opencv-python` 4.10 + `opencv-contrib`
  4.11) risk shadowing; Phase 2 must verify `cv2.aruco` resolves before
  relying on it.
- **Gripper feedback bands** for lid-rim vs body grasps are unknown until
  measured; `expect_band` values live in the container model config.
