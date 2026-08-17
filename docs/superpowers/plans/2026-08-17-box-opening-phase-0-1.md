# Box Opening Phases 0–1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `rammp_box_opening` package through Phase 0 (scaffold + smoke against the planner, no arm) and prepare Phase 1 (attended choreographed opening at a known pose) per the approved design spec.

**Architecture:** A layered client library over the RAMMP-CuRobo planning service: pure-logic modules (container model, guards, leg merging, world generation) that are fully offline-testable, one thin ROS wrapper (`PlannerClient`), a `Runner` that owns gates/merging/retries/logging and is tested against a fake client, and thin primitives/tasks/CLIs on top. Dry-run is the default everywhere; motion CLIs are human-run.

**Tech Stack:** ROS 2 Humble (rclpy, ament_python), `rammp_curobo_interfaces` from the `~/RAMMP-CuRobo/install` overlay, `rammp_curobo` pip core geometry helpers, pytest (`-p no:anyio`), Ruff v0.3.0 via pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-14-box-opening-design.md` (rev 3) — the authoritative design. Read it before implementing. This plan implements §§4–7 and the Phase 0–1 rows of §8.

## Global Constraints

- Interfaces come from the overlay: source `~/RAMMP-CuRobo/install/setup.zsh` — no copied code, no submodule, no modifications to RAMMP-CuRobo (spec §4, decision 3).
- Geometry helpers are CONSUMED from the installed `rammp_curobo` pip core: `from rammp_curobo.geometry import ang_diff, yaw_about_world_z, tool_axis, tip_to_tool, xyzw_to_wxyz, euler_deg_to_quat_xyzw` — never reimplemented (spec §4).
- Quaternions: ROS interfaces are **xyzw**; `tool_axis`/`spin_about_tool` take **wxyz**; convert explicitly via `xyzw_to_wxyz` (spec §3).
- Every angle comparison goes through `ang_diff` (joint_3 sits AT +π at home; reports wrap to (−π, π]) (spec §3).
- No autonomous motion from agent sessions, ever. Motion CLIs are run by the human, dry-run by default, typed `yes` to execute (spec §1, §6). Phase 0's `smoke_plan` and the planner node with `execute:=false` are planning-only — nothing can move.
- `export ROS_LOCALHOST_ONLY=1` in EVERY ROS shell, explicitly. Shells are zsh; source `setup.zsh` files. From non-zsh tooling run commands via `zsh -c '…'`.
- pytest needs `-p no:anyio` on this machine (pytest.ini `addopts`).
- Speeds: 0.25 transit, 0.15 contact; `speed_scale` 0.0 is the server-default sentinel and values outside (0, 1] are REFUSED by the server (spec §2, §6).
- Server tolerances (spec §6): start gate 0.05 rad, arrival 0.08 rad, client drift-replan threshold 0.04 rad.
- Git: repo-local identity `RAMMP <chrisman4247@gmail.com>`, local commits only. Conventions mirrored from RAMMP-CuRobo: Ruff v0.3.0 defaults, mdformat/gfm, Apache-2.0.
- Build/test cycle (run via `zsh -c`):

```zsh
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source ~/RAMMP-CuRobo/install/setup.zsh
cd ~/RAMMP-box-opening
colcon build --symlink-install
source install/setup.zsh
python3 -m pytest src/rammp_box_opening/test -q
```

**Fixed constants used throughout (verified against live sources 2026-08-17):**

- `HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]`, `JOINTS = ["joint_1"…"joint_7"]`, `WRIST_FLAT_XYZW = [0.5, 0.5, 0.5, 0.5]` (tour_demo.py).
- Namespace `/rammp_curobo`; gripper action `/robotiq_gripper_controller/gripper_cmd` (`control_msgs/GripperCommand`, `goal.command.position` 0.0 open ≈85 mm … 0.8 closed, `goal.command.max_effort = 100.0`); planner Trigger services `~/open_gripper`, `~/close_gripper` exist but are NOT used by this package (full open/close only, and planner-gated — we need graded closes).
- Interface result fields: `PlanToPose.Result`: `success, message, trajectory, planning_time`; `PlanToJoints.Result` adds `goal_mismatch_rad`; `ExecuteTrajectory` goal `{trajectory, speed_scale}`, result `{success, message}`, feedback `{joint_states, progress}`; `SetWorld` request `{world}` (YAML path or packaged name), response `{success, message}`.
- No-motion abort signature: result message contains `"never left the start"` (tour_demo).
- Pose uncertainty 0.02 m + tip bias 0.021 m (`tip_offset_m` ships 0.0/disabled in gen3.yaml until re-measured).

---

### Task 1: Package scaffold + tooling

**Files:**

- Create: `src/rammp_box_opening/package.xml`
- Create: `src/rammp_box_opening/setup.py`
- Create: `src/rammp_box_opening/setup.cfg`
- Create: `src/rammp_box_opening/resource/rammp_box_opening` (empty marker file)
- Create: `src/rammp_box_opening/rammp_box_opening/__init__.py`, plus empty `runtime/__init__.py`, `primitives/__init__.py`, `tasks/__init__.py`, `models/__init__.py`
- Create: `src/rammp_box_opening/rammp_box_opening/constants.py`
- Create: `src/rammp_box_opening/test/test_scaffold.py`
- Create: `pytest.ini`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`

**Interfaces:**

- Produces: importable package `rammp_box_opening` and `rammp_box_opening.constants` (every later task imports these). Entry points are added in Task 9 — `setup.py` starts with an empty `console_scripts` list.

- [ ] **Step 1: Write the scaffold files**

`src/rammp_box_opening/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>rammp_box_opening</name>
  <version>0.1.0</version>
  <description>Open and pick up an OXO POP container with the Kinova Gen3:
    composable primitives over the RAMMP-CuRobo planning service. Client
    only — interfaces come from the ~/RAMMP-CuRobo/install overlay.</description>
  <maintainer email="chrisman4247@gmail.com">RAMMP</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>rammp_curobo_interfaces</depend>
  <depend>trajectory_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>control_msgs</depend>
  <depend>std_srvs</depend>
  <depend>builtin_interfaces</depend>
  <depend>tf2_ros</depend>
  <depend>controller_manager_msgs</depend>
  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-yaml</exec_depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/rammp_box_opening/setup.py` (mirrors rammp_curobo_ros; `console_scripts` filled by Task 9):

```python
import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rammp_box_opening"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "config", "containers"),
            glob("config/containers/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RAMMP",
    maintainer_email="chrisman4247@gmail.com",
    description="Box-opening primitives over the RAMMP-CuRobo planning service",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
```

`src/rammp_box_opening/setup.cfg`:

```ini
[develop]
script_dir=$base/lib/rammp_box_opening
[install]
install_scripts=$base/lib/rammp_box_opening
```

`src/rammp_box_opening/rammp_box_opening/constants.py`:

```python
"""Shared constants: arm facts and runner defaults (spec §3, §6)."""

# Gen3 home joints in controller order — FK-verified in RAMMP-CuRobo's
# tour_demo.py; joint_3 sits AT +pi (every comparison uses ang_diff).
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

# tool_frame attitude at HOME: tool z level along world +x, wrist flat.
WRIST_FLAT_XYZW = [0.5, 0.5, 0.5, 0.5]

NODE_NAMESPACE = "/rammp_curobo"
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"

TRANSIT_SPEED = 0.25
CONTACT_SPEED = 0.15
DRIFT_REPLAN_RAD = 0.04     # < server start gate (0.05); > its arrival tol drift
SANITY_MARGIN_RAD = 0.35    # per-joint excursion allowance beyond |start->end|

POSE_UNCERTAINTY_M = 0.02   # calibration floor (spec §3)
TIP_BIAS_M = 0.021          # 2F-85 pad face beyond tool_frame (disabled in gen3.yaml)
BASELINE_TRAVEL_M = 0.01    # descent distance budget while the guard baselines

GRIPPER_CMD_CLOSED = 0.8    # GripperCommand position at full close
GRIPPER_CMD_OPEN = 0.0      # ~85 mm aperture
```

`pytest.ini` (repo root):

```ini
[pytest]
; -p no:anyio: the Jetson's user site carries an anyio pytest plugin built
; for a newer pytest than Ubuntu 22.04 ships — autoloading it crashes
; collection (same workaround as RAMMP-CuRobo).
addopts = -p no:anyio
testpaths = src/rammp_box_opening/test
```

`.pre-commit-config.yaml`: copy `~/RAMMP-CuRobo/.pre-commit-config.yaml` verbatim (pre-commit-hooks v4.5.0, ruff v0.3.0 + ruff-format, mdformat 0.7.17 + mdformat-gfm).

`.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
build/
install/
log/
dist/
.pytest_cache/
.ruff_cache/
```

`README.md`: title, one-paragraph purpose, the sourcing chain from Global Constraints (build chain AND the runtime chain: humble → RAMMP-Kinova ws → RAMMP-CuRobo → this ws), pointer to the spec and (once it exists, Task 11) `docs/HARDWARE_BRINGUP.md`, and the safety stance (dry-run default, human-run motion CLIs).

`src/rammp_box_opening/test/test_scaffold.py`:

```python
from rammp_box_opening.constants import HOME, JOINTS, WRIST_FLAT_XYZW


def test_constants_shape():
    assert len(HOME) == 7
    assert JOINTS[0] == "joint_1" and JOINTS[-1] == "joint_7"
    assert len(WRIST_FLAT_XYZW) == 4
```

- [ ] **Step 2: Build and test**

Run (zsh, per Global Constraints): `colcon build --symlink-install`, then `python3 -m pytest src/rammp_box_opening/test -q`.
Expected: build succeeds; 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: rammp_box_opening package scaffold (Phase 0, Task 1)"
```

---

### Task 2: Container model + pose derivation

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/models/container.py`
- Create: `src/rammp_box_opening/config/containers/oxo_pop.yaml`
- Test: `src/rammp_box_opening/test/test_container_model.py`

**Interfaces:**

- Consumes: `constants.WRIST_FLAT_XYZW`; `rammp_curobo.geometry` helpers.
- Produces (used by primitives/tasks/worlds):
  - `GraspSpec(offset, width_m, expect_band, attitude_rpy_deg)` (frozen dataclass)
  - `ContainerModel.load(path) -> ContainerModel` with fields `dims, lid_dims, button_offset, press_depth_window, touch_nm, press_attitude_rpy_deg, lid_grasp, body_grasp, hover_standoff, aperture_at_0, aperture_at_08, measure_me: bool`
  - `ContainerModel.width_to_command(width_m) -> float` (0.0–0.8, ValueError outside)
  - `ContainerPose(xyz, yaw)`; `ConfigPoseSource(path).container_pose() -> ContainerPose`
  - `from_container(cpose, offset) -> list[3]` (container-frame offset → base_link, yaw-rotated)
  - `wrist_flat_quat(xyz) -> list[4] xyzw` (transit attitude, bearing-steered)
  - `attitude_quat(rpy_deg, yaw) -> list[4] xyzw` (primitive attitude: config rpy, then `yaw_about_world_z` steer)
  - `load_lid_place(path) -> ContainerPose` (the one non-derived pose, `open_container.lid_place`)

- [ ] **Step 1: Write the failing tests**

```python
import math

import pytest

from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    ConfigPoseSource,
    attitude_quat,
    from_container,
    load_lid_place,
    wrist_flat_quat,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def test_load_and_validate():
    m = ContainerModel.load(CFG)
    assert m.measure_me is True                      # placeholders flagged
    assert m.press_depth_window[0] < m.press_depth_window[1]
    assert m.lid_grasp.width_m <= m.aperture_at_0    # graspable
    assert m.body_grasp.width_m <= m.aperture_at_0


def test_width_to_command_endpoints_and_refusal():
    m = ContainerModel.load(CFG)
    assert m.width_to_command(m.aperture_at_0) == pytest.approx(0.0)
    assert m.width_to_command(m.aperture_at_08) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        m.width_to_command(m.aperture_at_0 + 0.01)


def test_from_container_rotates_by_yaw():
    cpose = ContainerPose(xyz=(1.0, 2.0, 0.0), yaw=math.pi / 2)
    # +x offset in container frame maps to +y in base at yaw 90 deg
    assert from_container(cpose, (0.1, 0.0, 0.05)) == pytest.approx([1.0, 2.1, 0.05])


def test_wrist_flat_quat_is_bearing_steered_xyzw():
    q0 = wrist_flat_quat([1.0, 0.0, 0.5])            # bearing 0 -> home attitude
    assert q0 == pytest.approx([0.5, 0.5, 0.5, 0.5])
    q90 = wrist_flat_quat([0.0, 1.0, 0.5])
    assert q90 != pytest.approx(q0)                  # steered, unit
    assert sum(v * v for v in q90) == pytest.approx(1.0)


def test_attitude_quat_top_down_points_tool_down():
    from rammp_curobo.geometry import tool_axis, xyzw_to_wxyz

    q = attitude_quat([180.0, 0.0, 0.0], yaw=0.7)
    ax = tool_axis(xyzw_to_wxyz(q))                  # explicit order conversion
    assert ax[2] == pytest.approx(-1.0, abs=1e-6)    # tool z straight down


def test_pose_source_and_lid_place():
    cp = ConfigPoseSource(CFG).container_pose()
    assert len(cp.xyz) == 3
    lid = load_lid_place(CFG)
    assert lid.xyz != cp.xyz                         # set-down spot is elsewhere
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_container_model.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the config and implementation**

`config/containers/oxo_pop.yaml` — every geometric value is a **placeholder measured at Phase-1 bench time** (spec §10); `measure_me: true` makes primitives refuse hardware runs until the worksheet (Task 11) flips it:

```yaml
# OXO POP container model — PLACEHOLDERS, MEASURE AT PHASE-1 BENCH TIME.
measure_me: true

dims: [0.075, 0.075, 0.16]        # outer x, y, z (m), origin = bottom center
lid_dims: [0.075, 0.075, 0.03]
button_offset: [0.0, 0.0, 0.16]   # origin -> button top center
press:
  depth_window: [0.004, 0.012]    # m below nominal button top (min=real press,
  touch_nm: 3.0                   #  max includes cancel-latency budget, spec §6)
press_attitude_rpy_deg: [180.0, 0.0, 0.0]   # top-down; yaw steered per pose
lid_grasp:
  offset: [0.0, 0.0, 0.15]        # rim grasp point
  width_m: 0.02
  expect_band: [0.55, 0.75]       # gripper position feedback = holding lid
  attitude_rpy_deg: [180.0, 0.0, 0.0]
body_grasp:
  offset: [0.0, 0.0, 0.08]
  width_m: 0.072
  expect_band: [0.10, 0.60]
  attitude_rpy_deg: [180.0, 0.0, 0.0]
hover_standoff: 0.08              # m above target; must clear guards.min_standoff
gripper_map:
  aperture_at_0: 0.085            # bench-calibrated two-point width->command map
  aperture_at_08: 0.0
bench_pose:                       # Phase-1 PoseSource (hand-measured)
  xyz: [0.45, 0.0, -0.07]
  yaw_deg: 0.0
open_container:
  lid_place:                      # the one non-container-derived pose (spec §5)
    xyz: [0.45, -0.25, -0.07]
    yaw_deg: 0.0
```

`models/container.py`:

```python
"""Container model: every primitive target derives from this + a pose source.

Attitudes: transit poses use the wrist-flat family (Rz(bearing) ⊗ q_home,
spec §3); contact primitives use per-primitive attitudes from the container
config (default top-down [180, 0, 0] rpy — wrist-flat tool z is HORIZONTAL,
which cannot press a lid button from above), yaw-steered the same way.
"""

import math
from dataclasses import dataclass

import yaml

from rammp_curobo.geometry import euler_deg_to_quat_xyzw, yaw_about_world_z

from rammp_box_opening.constants import WRIST_FLAT_XYZW


@dataclass(frozen=True)
class GraspSpec:
    offset: tuple
    width_m: float
    expect_band: tuple
    attitude_rpy_deg: tuple


@dataclass(frozen=True)
class ContainerPose:
    xyz: tuple
    yaw: float


@dataclass(frozen=True)
class ContainerModel:
    dims: tuple
    lid_dims: tuple
    button_offset: tuple
    press_depth_window: tuple
    touch_nm: float
    press_attitude_rpy_deg: tuple
    lid_grasp: GraspSpec
    body_grasp: GraspSpec
    hover_standoff: float
    aperture_at_0: float
    aperture_at_08: float
    measure_me: bool

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)

        def grasp(key):
            g = raw[key]
            return GraspSpec(
                offset=tuple(g["offset"]),
                width_m=float(g["width_m"]),
                expect_band=tuple(g["expect_band"]),
                attitude_rpy_deg=tuple(g["attitude_rpy_deg"]),
            )

        model = cls(
            dims=tuple(raw["dims"]),
            lid_dims=tuple(raw["lid_dims"]),
            button_offset=tuple(raw["button_offset"]),
            press_depth_window=tuple(raw["press"]["depth_window"]),
            touch_nm=float(raw["press"]["touch_nm"]),
            press_attitude_rpy_deg=tuple(raw["press_attitude_rpy_deg"]),
            lid_grasp=grasp("lid_grasp"),
            body_grasp=grasp("body_grasp"),
            hover_standoff=float(raw["hover_standoff"]),
            aperture_at_0=float(raw["gripper_map"]["aperture_at_0"]),
            aperture_at_08=float(raw["gripper_map"]["aperture_at_08"]),
            measure_me=bool(raw.get("measure_me", True)),
        )
        lo, hi = model.press_depth_window
        if not lo < hi:
            raise ValueError("press.depth_window must be (min, max) with min < max")
        for g in (model.lid_grasp, model.body_grasp):
            model.width_to_command(g.width_m)  # raises if ungraspable
        return model

    def width_to_command(self, width_m):
        span = self.aperture_at_0 - self.aperture_at_08
        cmd = 0.8 * (self.aperture_at_0 - float(width_m)) / span
        if not 0.0 <= cmd <= 0.8:
            raise ValueError(
                "width %.3f m outside gripper range [%.3f, %.3f]"
                % (width_m, self.aperture_at_08, self.aperture_at_0)
            )
        return cmd


def from_container(cpose, offset):
    """Container-frame offset -> base_link, rotated by the container yaw."""
    c, s = math.cos(cpose.yaw), math.sin(cpose.yaw)
    ox, oy, oz = offset
    return [
        cpose.xyz[0] + c * ox - s * oy,
        cpose.xyz[1] + s * ox + c * oy,
        cpose.xyz[2] + oz,
    ]


def wrist_flat_quat(xyz):
    """Transit attitude: wrist-flat, steered to the point's bearing (xyzw)."""
    bearing = math.atan2(xyz[1], xyz[0])
    return list(yaw_about_world_z(WRIST_FLAT_XYZW, bearing))


def attitude_quat(rpy_deg, yaw):
    """Primitive attitude from config rpy, yaw-steered about world z (xyzw)."""
    return list(yaw_about_world_z(euler_deg_to_quat_xyzw(rpy_deg), yaw))


class ConfigPoseSource:
    """Phase-1 PoseSource: the hand-measured bench pose from the config.

    Phase 2 swaps in a perception-based source with the same
    container_pose() -> ContainerPose interface (spec §5)."""

    def __init__(self, path):
        self._path = path

    def container_pose(self):
        with open(self._path) as f:
            raw = yaml.safe_load(f)["bench_pose"]
        return ContainerPose(
            xyz=tuple(raw["xyz"]), yaw=math.radians(float(raw["yaw_deg"]))
        )


def load_lid_place(path):
    with open(path) as f:
        raw = yaml.safe_load(f)["open_container"]["lid_place"]
    return ContainerPose(
        xyz=tuple(raw["xyz"]), yaw=math.radians(float(raw["yaw_deg"]))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_container_model.py -q`
Expected: PASS (6 tests). Note the config `data_files` glob from Task 1 already installs `config/containers/*.yaml`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: container model, pose derivation, config pose source (Task 2)"
```

---

### Task 3: Guards — torque guard, sanity gate, retrace, classification

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/runtime/guards.py`
- Test: `src/rammp_box_opening/test/test_guards.py`

**Interfaces:**

- Consumes: `rammp_curobo.geometry.ang_diff`; constants.
- Produces (used by Runner and primitives):
  - `TorqueGuard(touch_nm)` — `.on_progress(p)`, `.on_efforts(list4) -> bool tripped`, `.peak: float`, `.armed: bool`
  - `GuardSpec(touch_nm, trip, depth_window=None, target_z=None)` where `trip` ∈ `"press" | "obstruction" | "setdown"`
  - `sanity_violations(traj, margin_rad) -> list[str]`
  - `min_standoff() -> float` and `check_standoff(hover_z, contact_z) -> None | raises ValueError`
  - `classify_press(depth_m, window) -> "pressed" | "rim" ` and `press_outcome(outcome, depth_m, window) -> (ok, detail)`
  - `in_band(pos, band) -> bool`
  - `reverse_retrace(traj, progress) -> JointTrajectory` (plan-free fallback retreat)

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import (
    GuardSpec,
    TorqueGuard,
    check_standoff,
    classify_press,
    in_band,
    min_standoff,
    press_outcome,
    reverse_retrace,
    sanity_violations,
)


def _traj(rows, dt=0.5):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, len(rows[0]) + 1)]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * len(row)
        p.accelerations = [0.0] * len(row)
        p.time_from_start.sec = int(i * dt)
        p.time_from_start.nanosec = int((i * dt % 1) * 1e9)
        t.points.append(p)
    return t


def test_guard_baseline_anchored_at_progress():
    g = TorqueGuard(touch_nm=3.0)
    assert g.on_efforts([9.0, 9.0, 9.0, 9.0]) is False   # not armed: ignored
    g.on_progress(0.0)
    assert g.armed is False                              # progress 0 != started
    g.on_progress(0.01)
    assert g.on_efforts([1.0, 1.0, 1.0, 1.0]) is False   # first sample = baseline
    assert g.on_efforts([2.0, 1.0, 1.0, 1.0]) is False   # dev 1.0 < 3.0
    assert g.on_efforts([1.0, 5.0, 1.0, 1.0]) is True    # dev 4.0 > 3.0
    assert g.peak == pytest.approx(4.0)


def test_guard_ignores_none_efforts():
    g = TorqueGuard(touch_nm=3.0)
    g.on_progress(0.5)
    assert g.on_efforts(None) is False


def test_sanity_gate_flags_wandering_joint():
    # joint_1 wanders 1.0 rad out and back on a 0.1 rad net move
    bad = _traj([[0.0, 0.0], [1.0, 0.05], [0.1, 0.1]])
    good = _traj([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
    assert sanity_violations(good, margin_rad=0.35) == []
    v = sanity_violations(bad, margin_rad=0.35)
    assert len(v) == 1 and "joint_1" in v[0]


def test_sanity_gate_wrap_aware():
    # joint crossing the pi boundary: 3.10 -> -3.10 is a 0.08 rad move
    t = _traj([[3.10], [3.14], [-3.10]])
    assert sanity_violations(t, margin_rad=0.35) == []


def test_standoff_floor():
    assert min_standoff() == pytest.approx(0.051)
    check_standoff(hover_z=0.10, contact_z=0.0)          # 0.10 > 0.051: ok
    with pytest.raises(ValueError):
        check_standoff(hover_z=0.04, contact_z=0.0)


def test_press_classification():
    window = (0.004, 0.012)
    assert classify_press(0.008, window) == "pressed"
    assert classify_press(0.001, window) == "rim"
    ok, detail = press_outcome("touch", 0.008, window)
    assert ok
    ok, detail = press_outcome("touch", 0.001, window)
    assert not ok and "rim" in detail
    ok, detail = press_outcome("arrived", None, window)  # bottomed out, no click
    assert not ok


def test_in_band():
    assert in_band(0.6, (0.55, 0.75))
    assert not in_band(0.8, (0.55, 0.75))                # closed on air


def test_reverse_retrace_reverses_executed_portion():
    t = _traj([[0.0], [0.2], [0.4], [0.6]], dt=1.0)      # 3 s total
    r = reverse_retrace(t, progress=0.5)                 # stopped ~1.5 s in
    starts = [p.positions[0] for p in r.points]
    assert starts[0] == pytest.approx(0.4)               # from deepest executed
    assert starts[-1] == pytest.approx(0.0)              # back to the start
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
             for p in r.points]
    assert times == sorted(times) and times[0] == 0.0    # re-timed, monotonic
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_guards.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation**

```python
"""Contact guard, trajectory sanity gate, and guarded-descent bookkeeping.

TorqueGuard is the palm-demo pattern with the spec §6 hardening: the
baseline is anchored at the first ExecuteTrajectory feedback with
progress > 0 (never at goal-accept), and guarded runs REFUSE to start
without effort fields (enforced by the Runner, which owns the streams).
"""

import copy
from dataclasses import dataclass

from trajectory_msgs.msg import JointTrajectory

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import (
    BASELINE_TRAVEL_M,
    POSE_UNCERTAINTY_M,
    TIP_BIAS_M,
)


class TorqueGuard:
    def __init__(self, touch_nm):
        self.touch_nm = float(touch_nm)
        self.armed = False
        self._baseline = None
        self.peak = 0.0

    def on_progress(self, progress):
        if progress > 0.0:
            self.armed = True

    def on_efforts(self, wrist_efforts):
        if not self.armed or wrist_efforts is None:
            return False
        if self._baseline is None:
            self._baseline = [float(v) for v in wrist_efforts]
            return False
        dev = max(abs(a - b) for a, b in zip(wrist_efforts, self._baseline))
        self.peak = max(self.peak, dev)
        return dev > self.touch_nm


@dataclass(frozen=True)
class GuardSpec:
    touch_nm: float
    trip: str                    # "press" | "obstruction" | "setdown"
    depth_window: tuple = None   # "press" only, m below nominal contact z
    target_z: float = None       # nominal contact z (base_link) for depth calc


def sanity_violations(traj, margin_rad):
    """Per-joint excursion beyond |start->end| + margin: planner wandered."""
    out = []
    for j, name in enumerate(traj.joint_names):
        pos = [p.positions[j] for p in traj.points]
        allowed = abs(ang_diff(pos[-1], pos[0])) + margin_rad
        excursion = max(pos) - min(pos)
        if excursion > allowed:
            out.append(
                "%s excursion %.3f rad > |Δ| + margin %.3f" % (name, excursion, allowed)
            )
    return out


def min_standoff():
    """Below this the guard baseline could be captured already in contact."""
    return POSE_UNCERTAINTY_M + TIP_BIAS_M + BASELINE_TRAVEL_M


def check_standoff(hover_z, contact_z):
    gap = hover_z - contact_z
    if gap < min_standoff():
        raise ValueError(
            "hover standoff %.3f m < required %.3f m — the guard baseline "
            "could be captured in contact (spec §6)" % (gap, min_standoff())
        )


def classify_press(depth_m, window):
    lo, hi = window
    return "pressed" if lo <= depth_m <= hi else "rim"


def press_outcome(outcome, depth_m, window):
    if outcome == "touch":
        if depth_m is None:
            return False, "trip depth unknown (no tool z)"
        verdict = classify_press(depth_m, window)
        if verdict == "pressed":
            return True, "button pressed at depth %.4f m" % depth_m
        return False, "rim/edge contact at depth %.4f m (before window)" % depth_m
    if outcome == "arrived":
        return False, "reached depth_window.max untripped — no click detected"
    return False, "descent %s" % outcome


def in_band(pos, band):
    lo, hi = band
    return lo <= float(pos) <= hi


def reverse_retrace(traj, progress):
    """Plan-free retreat: the executed portion of a descent, reversed.

    Used when post-contact planning fails (the start may read as
    in-collision, spec §6). Revalidated by the server's own gates."""
    end = traj.points[-1].time_from_start
    total = end.sec + end.nanosec * 1e-9
    cut = total * float(progress)
    done = [
        p
        for p in traj.points
        if p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 <= cut
    ]
    if not done:
        done = [traj.points[0]]
    out = JointTrajectory()
    out.joint_names = list(traj.joint_names)
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in done]
    t_deep = times[-1]
    for p, t in zip(reversed(done), reversed(times)):
        q = copy.deepcopy(p)
        q.velocities = [0.0] * len(p.positions)
        q.accelerations = [0.0] * len(p.positions)
        t_new = t_deep - t
        q.time_from_start.sec = int(t_new)
        q.time_from_start.nanosec = int(round((t_new - int(t_new)) * 1e9))
        out.points.append(q)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_guards.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: torque guard, sanity gate, retrace, press classification (Task 3)"
```

---

### Task 4: Legs + merge rules

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/runtime/legs.py`
- Test: `src/rammp_box_opening/test/test_merge.py`

**Interfaces:**

- Consumes: `GuardSpec` from Task 3.
- Produces (Runner and primitives build on these):
  - `Kind` enum: `MOTION`, `GRIPPER`
  - `Leg(name, kind, traj, speed, guard, world, chain, target, goal_joints, invalidates_downstream=False, verify=None, gripper_cmd=None)` — `traj` is a `JointTrajectory` or None (GRIPPER / not yet planned); `target` is `("pose", xyz, quat_xyzw)` or `("joints", q7)` so the Runner can re-plan any leg; `verify` is `Callable[[VerifyCtx], tuple[bool, str]]` or None
  - `VerifyCtx(outcome, depth_m, gripper_pos, progress, torque_peak)` (dataclass, all optional-None)
  - `can_merge(a, b) -> bool`; `merge_groups(legs) -> list[list[Leg]]`
  - `merge_trajectories(trajs) -> JointTrajectory` (tour_demo pattern, takes trajectory list)

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import (
    Kind,
    Leg,
    VerifyCtx,
    can_merge,
    merge_groups,
    merge_trajectories,
)


def _traj(rows, dt=1.0):
    t = JointTrajectory()
    t.joint_names = ["joint_1"]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(row)]
        p.velocities = [0.0]
        p.accelerations = [0.0]
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


def leg(name, chain=0, speed=0.25, guard=None, verify=None, kind=Kind.MOTION):
    return Leg(
        name=name, kind=kind, traj=_traj([0.0, 0.1]), speed=speed, guard=guard,
        world="full", chain=chain, target=("joints", [0.1]), goal_joints=[0.1],
        verify=verify,
    )


def test_same_chain_same_speed_merges():
    groups = merge_groups([leg("a"), leg("b")])
    assert [len(g) for g in groups] == [2]


def test_chain_break_splits():
    groups = merge_groups([leg("a", chain=0), leg("b", chain=1)])
    assert [len(g) for g in groups] == [1, 1]


def test_speed_change_splits():
    groups = merge_groups([leg("a", speed=0.25), leg("b", speed=0.15)])
    assert [len(g) for g in groups] == [1, 1]


def test_guarded_leg_always_alone():
    g = GuardSpec(touch_nm=3.0, trip="press")
    groups = merge_groups([leg("a"), leg("b", guard=g), leg("c")])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_verify_closes_group():
    v = lambda ctx: (True, "")
    groups = merge_groups([leg("a", verify=v), leg("b"), leg("c")])
    assert [len(g) for g in groups] == [1, 2]


def test_gripper_never_merges():
    gl = Leg(name="close", kind=Kind.GRIPPER, traj=None, speed=0.0, guard=None,
             world="full", chain=0, target=None, goal_joints=None, gripper_cmd=0.8)
    groups = merge_groups([leg("a"), gl, leg("b")])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_merge_trajectories_offsets_time():
    merged = merge_trajectories([_traj([0.0, 0.1]), _traj([0.1, 0.2])])
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
             for p in merged.points]
    assert times == sorted(times)
    assert times[-1] == pytest.approx(2.0)   # 1 s + 1 s, offset applied
    assert merged.points[-1].positions[0] == pytest.approx(0.2)


def test_verify_ctx_defaults():
    ctx = VerifyCtx(outcome="arrived")
    assert ctx.depth_m is None and ctx.gripper_pos is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_merge.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation**

```python
"""Legs (the unit of planning/execution) and the merge rules (spec §5, §6).

MOTION legs merge into one trajectory only when dynamically valid: same
planning chain (B planned from A's predicted endpoint), same speed, no
guard on either, and a verify closes its group. Guarded legs always
execute alone. `world` is a checked precondition, never a merge key.
"""

from dataclasses import dataclass, field
from enum import Enum

from trajectory_msgs.msg import JointTrajectory


class Kind(Enum):
    MOTION = "motion"
    GRIPPER = "gripper"


@dataclass
class VerifyCtx:
    outcome: str
    depth_m: float = None
    gripper_pos: float = None
    progress: float = None
    torque_peak: float = None


@dataclass
class Leg:
    name: str
    kind: Kind
    traj: object            # JointTrajectory | None
    speed: float
    guard: object           # GuardSpec | None
    world: str              # world name this leg was PLANNED against
    chain: int
    target: tuple           # ("pose", xyz, quat_xyzw) | ("joints", q7) | None
    goal_joints: list       # predicted end joints (MOTION), None for GRIPPER
    invalidates_downstream: bool = False
    verify: object = None   # Callable[[VerifyCtx], tuple[bool, str]] | None
    gripper_cmd: float = None
    stale: bool = field(default=False, compare=False)  # set by the Runner


def can_merge(a, b):
    return (
        a.kind is Kind.MOTION
        and b.kind is Kind.MOTION
        and a.chain == b.chain
        and a.speed == b.speed
        and a.guard is None
        and b.guard is None
        and a.verify is None            # a verify CLOSES its merge group
    )


def merge_groups(legs):
    groups = []
    for leg in legs:
        if groups and can_merge(groups[-1][-1], leg):
            groups[-1].append(leg)
        else:
            groups.append([leg])
    return groups


def merge_trajectories(trajs):
    """Chained per-segment trajectories -> ONE continuous JointTrajectory.

    tour_demo.py pattern: zero controller goal transitions is the
    no-motion-fault mitigation. Callers guarantee chaining validity
    (merge_groups)."""
    merged = JointTrajectory()
    merged.joint_names = list(trajs[0].joint_names)
    offset = 0.0
    for traj in trajs:
        for pt in traj.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 + offset
            q = type(pt)()
            q.positions = list(pt.positions)
            q.velocities = list(pt.velocities)
            q.accelerations = list(pt.accelerations)
            q.time_from_start.sec = int(t)
            q.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            merged.points.append(q)
        last = traj.points[-1].time_from_start
        offset += last.sec + last.nanosec * 1e-9
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_merge.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: legs and merge rules (Task 4)"
```

---

### Task 5: World generation

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/worlds.py`
- Create: `src/rammp_box_opening/config/world_bench.yaml`
- Test: `src/rammp_box_opening/test/test_worlds.py`

**Interfaces:**

- Consumes: `ContainerModel`, `ContainerPose`, `from_container` (Task 2).
- Produces (Runner + preflight use these):
  - `full_world(bench, model, cpose, lid_at=None) -> dict` — bench + err-tall container cuboid; optional placed-lid cuboid
  - `interaction_world(bench, model, cpose, target_xyz, contact_z, depth_max, lid_at=None) -> dict` — body reduced to the reduction plane + 4-cuboid aperture ring around the descent corridor
  - `reduction_plane_z(contact_z, depth_max) -> float` (≥ margin below the lowest commanded pose)
  - `WorldStore(bench_yaml_path, out_dir=None)` — `.push_name(kind, **kwargs) -> (name, path)` writes the YAML variant under `~/.ros/rammp_box_opening/worlds/` and returns its identity; deterministic names (e.g. `full`, `interaction_button`) so pushes are idempotent
  - Module constants: `ERR_TALL_M = 0.02`, `APERTURE_HALF_M = 0.06`, `RING_THICK_M = 0.05`, `PLANE_MARGIN_M = 0.03`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.worlds import (
    PLANE_MARGIN_M,
    WorldStore,
    full_world,
    interaction_world,
    reduction_plane_z,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"
BENCH = "src/rammp_box_opening/config/world_bench.yaml"


def _model_pose():
    m = ContainerModel.load(CFG)
    return m, ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0)


def _cuboids(world):
    return {o["name"]: o for o in world["obstacles"]}


def test_full_world_is_cuboids_only_and_err_tall():
    m, cp = _model_pose()
    w = full_world(_bench(), m, cp)
    for o in w["obstacles"]:
        assert set(o) == {"name", "position", "dims"}     # cuboids only (§2)
    c = _cuboids(w)["container"]
    assert c["dims"][2] > m.dims[2]                       # err tall
    top = c["position"][2] + c["dims"][2] / 2
    assert top > cp.xyz[2] + m.dims[2]


def _bench():
    import yaml

    with open(BENCH) as f:
        return yaml.safe_load(f)


def test_bench_obstacles_survive_in_all_variants():
    m, cp = _model_pose()
    names_full = set(_cuboids(full_world(_bench(), m, cp)))
    w = interaction_world(_bench(), m, cp, target_xyz=[0.45, 0.0, 0.09],
                          contact_z=0.09, depth_max=0.012)
    names_int = set(_cuboids(w))
    assert {"pedestal", "table"} <= names_full
    assert {"pedestal", "table"} <= names_int


def test_reduction_plane_below_deepest_command():
    z = reduction_plane_z(contact_z=0.09, depth_max=0.012)
    assert z <= 0.09 - 0.012 - PLANE_MARGIN_M + 1e-9


def test_interaction_ring_leaves_corridor_but_blocks_lateral():
    m, cp = _model_pose()
    tx = [0.45, 0.0, 0.09]
    w = interaction_world(_bench(), m, cp, target_xyz=tx, contact_z=0.09,
                          depth_max=0.012)
    cs = _cuboids(w)
    ring = [o for n, o in cs.items() if n.startswith("ring_")]
    assert len(ring) == 4
    for o in ring:                                        # corridor xy stays free
        dx = abs(o["position"][0] - tx[0]) - o["dims"][0] / 2
        dy = abs(o["position"][1] - tx[1]) - o["dims"][1] / 2
        assert max(dx, dy) >= 0.0                         # ring outside corridor
    body = cs["container_body"]
    body_top = body["position"][2] + body["dims"][2] / 2
    assert body_top <= reduction_plane_z(0.09, 0.012) + 1e-9


def test_lid_cuboid_added_after_place():
    m, cp = _model_pose()
    lid_at = ContainerPose(xyz=(0.45, -0.25, -0.07), yaw=0.0)
    w = full_world(_bench(), m, cp, lid_at=lid_at)
    lid = _cuboids(w)["placed_lid"]
    assert lid["position"][:2] == pytest.approx([0.45, -0.25])


def test_store_idempotent_names(tmp_path):
    m, cp = _model_pose()
    store = WorldStore(BENCH, out_dir=tmp_path)
    n1, p1 = store.push_name("full", model=m, cpose=cp)
    n2, p2 = store.push_name("full", model=m, cpose=cp)
    assert n1 == n2 == "full" and p1 == p2
    assert p1.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_worlds.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the config and implementation**

`config/world_bench.yaml` — OUR single source of truth for bench geometry (spec §6); values start as RAMMP-CuRobo's placeholders and are **reconciled once at Phase-1 bench time** with `world_real_bench.yaml`:

```yaml
# Measured bench geometry — PLACEHOLDERS copied from RAMMP-CuRobo's
# world_real_bench.yaml. MEASURE AT PHASE-1 BENCH TIME (tape measure from
# base_link: +x forward, +z up; err TALL on the table — too tall costs
# reachable volume, never safety). After reconciliation this file is the
# single source of truth; generated variants extend it (worlds.py).
base_frame: base_link

obstacles:
  - name: pedestal
    position: [0.0, 0.0, -0.05]
    dims: [0.14, 0.14, 0.04]
  - name: table
    position: [0.15, 0.0, -0.10]
    dims: [1.3, 1.4, 0.06]
```

`worlds.py`:

```python
"""Collision-world generation: bench + container-derived cuboid variants.

Worlds are a plan-time concern (spec §6). SetWorld is write-only, so the
Runner tracks what it last pushed; this module only builds and writes the
variants. Cuboids only — v0.7.8 drops other shapes.
"""

from pathlib import Path

import yaml

ERR_TALL_M = 0.02        # container cuboid extra height (err tall, spec §3)
APERTURE_HALF_M = 0.06   # half-extent of the free descent corridor
RING_THICK_M = 0.05      # aperture ring wall thickness
RING_HEIGHT_M = 0.25     # ring wall height above the reduction plane
PLANE_MARGIN_M = 0.03    # reduction plane below deepest command (≥ calib margin)


def _bench_obstacles(bench):
    return [dict(o) for o in bench["obstacles"]]


def _container_cuboid(model, cpose, name="container", top_z=None):
    dx, dy, dz = model.dims
    top = (cpose.xyz[2] + dz + ERR_TALL_M) if top_z is None else top_z
    height = top - cpose.xyz[2]
    return {
        "name": name,
        "position": [cpose.xyz[0], cpose.xyz[1], cpose.xyz[2] + height / 2],
        "dims": [dx, dy, height],
    }


def _lid_cuboid(model, lid_at):
    lx, ly, lz = model.lid_dims
    return {
        "name": "placed_lid",
        "position": [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + lz / 2],
        "dims": [lx, ly, lz],
    }


def full_world(bench, model, cpose, lid_at=None):
    obstacles = _bench_obstacles(bench)
    obstacles.append(_container_cuboid(model, cpose))
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {"base_frame": bench.get("base_frame", "base_link"),
            "obstacles": obstacles, "objects": [], "targets": []}


def reduction_plane_z(contact_z, depth_max):
    """The interaction world must let the planner plan FROM the deepest
    commanded contact point (spec §6): plane ≥ margin below it."""
    return contact_z - depth_max - PLANE_MARGIN_M


def interaction_world(bench, model, cpose, target_xyz, contact_z, depth_max,
                      lid_at=None):
    obstacles = _bench_obstacles(bench)
    plane = reduction_plane_z(contact_z, depth_max)
    if plane > cpose.xyz[2]:  # body below the plane stays solid
        obstacles.append(
            _container_cuboid(model, cpose, name="container_body", top_z=plane)
        )
    # Aperture ring: lateral entry forbidden, vertical corridor free.
    tx, ty = target_xyz[0], target_xyz[1]
    a, w, h = APERTURE_HALF_M, RING_THICK_M, RING_HEIGHT_M
    zc = plane + h / 2
    span = 2 * (a + w)
    obstacles += [
        {"name": "ring_xp", "position": [tx + a + w / 2, ty, zc],
         "dims": [w, span, h]},
        {"name": "ring_xn", "position": [tx - a - w / 2, ty, zc],
         "dims": [w, span, h]},
        {"name": "ring_yp", "position": [tx, ty + a + w / 2, zc],
         "dims": [span, w, h]},
        {"name": "ring_yn", "position": [tx, ty - a - w / 2, zc],
         "dims": [span, w, h]},
    ]
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {"base_frame": bench.get("base_frame", "base_link"),
            "obstacles": obstacles, "objects": [], "targets": []}


class WorldStore:
    def __init__(self, bench_yaml_path, out_dir=None):
        with open(bench_yaml_path) as f:
            self._bench = yaml.safe_load(f)
        self._dir = Path(
            out_dir
            if out_dir is not None
            else Path.home() / ".ros" / "rammp_box_opening" / "worlds"
        )
        self._dir.mkdir(parents=True, exist_ok=True)

    def push_name(self, kind, model=None, cpose=None, target_xyz=None,
                  contact_z=None, depth_max=None, lid_at=None, tag=""):
        if kind == "full":
            world = full_world(self._bench, model, cpose, lid_at=lid_at)
            name = "full" + (("_" + tag) if tag else "")
        elif kind == "interaction":
            world = interaction_world(self._bench, model, cpose, target_xyz,
                                      contact_z, depth_max, lid_at=lid_at)
            name = "interaction" + (("_" + tag) if tag else "")
        else:
            raise ValueError("unknown world kind %r" % kind)
        path = self._dir / (name + ".yaml")
        path.write_text(yaml.safe_dump(world, sort_keys=False))
        return name, path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_worlds.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: collision-world generation with aperture rings (Task 5)"
```

---

### Task 6: Planner client (the one ROS surface)

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/runtime/client.py`
- Test: `src/rammp_box_opening/test/test_client_protocol.py`

**Interfaces:**

- Consumes: overlay interfaces, `control_msgs`, `rcl_interfaces`, `tf2_ros`, `std ros` msgs; `TorqueGuard` (Task 3); constants.
- Produces — the protocol every Runner test fakes (keep names EXACT):
  - `PlannerClient(node)` with:
    - `joints() -> list[7]` (blocks ≤10 s for /joint_states, exits with the bringup hint otherwise — tour_demo pattern)
    - `wrist_efforts() -> list[4] | None`; `efforts_present() -> bool`
    - `tool_xyz(timeout_s=1.5) -> list[3] | None` (TF `base_link` → `tool_frame`)
    - `plan_to_pose(xyz, quat_xyzw, start_joints) -> result | None`
    - `plan_to_joints(q7, start_joints) -> result | None`
    - `execute(traj, speed, guard=None) -> tuple[str, dict]` — outcome `"arrived" | "touch" | "failed"`; info dict has `message, progress, torque_peak`
    - `set_world(path_or_name) -> tuple[bool, str]`
    - `planner_execute_enabled() -> bool` (GetParameters on `/rammp_curobo/get_parameters`)
    - `gripper_cmd(position) -> tuple[bool, float, bool]` — (accepted+finished, reached position, stalled)
  - The guard hookup: ExecuteTrajectory **feedback** drives `guard.on_progress(fb.progress)`; the `/joint_states` subscription drives `guard.on_efforts(...)`; trip → cancel goal → outcome `"touch"` (palm-demo pattern + spec §6 hardening).

- [ ] **Step 1: Write the failing protocol test**

The client is thin glue over proven patterns; its logic lives in Tasks 3–5. The offline test pins the protocol surface so Runner fakes cannot drift:

```python
import inspect

from rammp_box_opening.runtime.client import PlannerClient


def test_protocol_surface():
    required = {
        "joints": [],
        "wrist_efforts": [],
        "efforts_present": [],
        "tool_xyz": ["timeout_s"],
        "plan_to_pose": ["xyz", "quat_xyzw", "start_joints"],
        "plan_to_joints": ["q7", "start_joints"],
        "execute": ["traj", "speed", "guard"],
        "set_world": ["path_or_name"],
        "planner_execute_enabled": [],
        "gripper_cmd": ["position"],
    }
    for name, params in required.items():
        fn = getattr(PlannerClient, name)
        sig = list(inspect.signature(fn).parameters)[1:]  # drop self
        for p in params:
            assert p in sig, "%s missing param %s" % (name, p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/rammp_box_opening/test/test_client_protocol.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation**

Structure (follow the quoted reference patterns exactly; full file ≈180 lines):

```python
"""The single ROS surface: planner action clients + gripper + TF + params.

Every pattern here is lifted from proven RAMMP-CuRobo clients:
spin_until_done/cancel-on-Ctrl+C from tour_demo.py, the guarded execute
loop from the recovered palm_demo.py (with the spec §6 baseline change:
armed by action FEEDBACK progress > 0, efforts from /joint_states)."""

import sys
import time

import rclpy
from control_msgs.action import GripperCommand
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from rammp_curobo_interfaces.action import (
    ExecuteTrajectory,
    PlanToJoints,
    PlanToPose,
)
from rammp_curobo_interfaces.srv import SetWorld

from rammp_box_opening.constants import GRIPPER_ACTION, JOINTS, NODE_NAMESPACE


def spin_until_done(node, future, timeout_s):
    t0 = time.monotonic()
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


class PlannerClient:
    def __init__(self, node):
        self.node = node
        self._q = None
        self._eff = None
        node.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self._plan_pose = ActionClient(node, PlanToPose,
                                       NODE_NAMESPACE + "/plan_to_pose")
        self._plan_joints = ActionClient(node, PlanToJoints,
                                         NODE_NAMESPACE + "/plan_to_joints")
        self._execute = ActionClient(node, ExecuteTrajectory,
                                     NODE_NAMESPACE + "/execute_trajectory")
        self._gripper = ActionClient(node, GripperCommand, GRIPPER_ACTION)
        self._set_world = node.create_client(SetWorld,
                                             NODE_NAMESPACE + "/set_world")
        self._params = node.create_client(GetParameters,
                                          NODE_NAMESPACE + "/get_parameters")
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)
    ...
```

Fill in, per reference:

- `_js_cb`: map positions AND efforts by name to joint order (palm_demo `_js_cb`, including the `len(msg.effort) == len(msg.name)` presence check).
- `joints()`: spin-wait ≤10 s then `sys.exit` with the bringup hint (tour_demo `joints()`).
- `efforts_present()`: one fresh `joints()` then `self._eff is not None`.
- `tool_xyz()`: `self._tf.lookup_transform("base_link", "tool_frame", Time())` inside try/except returning None; spin briefly first.
- `_call(client, goal, timeout_s)`: tour_demo `_call` verbatim (wait_for_server 5 s → `sys.exit("planner node not running")`, send, result).
- `plan_to_pose` / `plan_to_joints`: build goals exactly as tour_demo `plan_pose_from`/`plan_home_from` (xyzw assignment to `g.target.orientation.*`; `g.start_joints = [float(v) for v in start] if start else []`).
- `execute(traj, speed, guard=None)`: palm_demo `run_traj` shape — send goal, then loop `spin_once(0.05)` until result future done; when `guard`: feed `guard.on_progress` from a feedback callback registered via `send_goal_async(goal, feedback_callback=...)` (feedback `.feedback.progress`), feed `guard.on_efforts(self.wrist_efforts())` each loop; on trip cancel + await result → `("touch", info)`. KeyboardInterrupt → cancel, re-raise (tour_demo). 240 s watchdog → cancel → `("failed", ...)`. Result success → `("arrived", ...)`; else `("failed", {"message": result.message, ...})`. Record final `progress` from the last feedback and `torque_peak` from `guard.peak`.
- `set_world(path_or_name)`: `wait_for_service(5.0)` else `(False, "set_world unavailable")`; call, return `(resp.success, resp.message)`.
- `planner_execute_enabled()`: GetParameters request `names=["execute"]`, return `resp.values[0].bool_value`; on timeout return False (fail-closed).
- `gripper_cmd(position)`: build `GripperCommand.Goal()`, `goal.command.position = float(position)`, `goal.command.max_effort = 100.0`; send/await like planner_node `_gripper_cmd`; return `(True, wrapped.result.position, wrapped.result.stalled)` or `(False, 0.0, False)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test -q` (full suite — imports prove the overlay wiring).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: PlannerClient — the single ROS surface (Task 6)"
```

---

### Task 7: Runner — gates, merging, retries, log

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/runtime/runner.py`
- Create: `src/rammp_box_opening/rammp_box_opening/runtime/confirm.py`
- Test: `src/rammp_box_opening/test/test_runner.py` (with `FakeClient`)

**Interfaces:**

- Consumes: legs/merge (Task 4), guards (Task 3), the PlannerClient protocol (Task 6), `WorldStore` (Task 5).
- Produces:
  - `LegResult(leg_name, outcome, ok, detail, torque_peak, progress, t_wall)` (dataclass; `outcome` ∈ `arrived|touch|failed|refused|skipped`)
  - `Runner(client, world_store, log_dir=None, margin_rad=SANITY_MARGIN_RAD)`
    - `.preview(legs) -> str` — per-leg table: name, kind, speed, world, chain, per-joint excursion (wrap-aware), time at speed
    - `.run(legs, execute: bool, assume_yes=False) -> list[LegResult]`
  - `confirm.typed_yes(prompt) -> bool` (EOFError-safe, exact `"yes"` — plan_and_execute pattern)

**Runner.run semantics (each is a test below):**

1. Dry-run (`execute=False`): print preview, return all `skipped`. Nothing touches the client's execute/gripper paths.
2. Sanity gate: any MOTION leg whose `traj` fails `sanity_violations` → `refused`, stop.
3. World precondition: an unguarded MOTION leg whose `world` doesn't start with `"full"` → `refused` (checked BEFORE anything executes). Before each group whose world differs from the last pushed, `client.set_world(...)` re-asserts (Runner tracks `_last_world`; SetWorld is write-only).
4. Gripper gate: GRIPPER legs require `client.planner_execute_enabled()` → else `refused` with the symmetry message (planner dry-run does not gate the direct gripper action — the Runner enforces it).
5. Guarded legs require `client.efforts_present()` → else `refused` (no silent position-only degradation).
6. Merge groups via `merge_groups`; merged group → one `client.execute(merge_trajectories([...]), speed)`.
7. Start-drift: before a pre-planned MOTION group, compare `client.joints()` to the group's first leg start (wrap-aware `ang_diff`); > `DRIFT_REPLAN_RAD` → re-plan that leg from live to the same `target` (new chain id `max(chains)+1`), continue.
8. Downstream invalidation: after a leg with `invalidates_downstream` resolves, all later legs are marked `stale`; a stale leg is re-planned from live before executing (dry-run previews keep nominal plans).
9. No-motion retry: `("failed", msg)` with `"never left the start"` in msg → retry ONCE from standstill. Any other failure → stop, arm holds, exact report.
10. Verify: leg with `verify` gets `VerifyCtx(outcome, depth_m=client.tool-z-derived depth if guard.trip=="press", gripper_pos=..., progress, torque_peak)`; `ok=False` → stop.
11. Guard trip depth for press legs: `depth_m = guard.target_z - client.tool_xyz()[2]` at trip.
12. Every leg appends one JSONL line to `log_dir/run-<ts>.jsonl` (timestamps, outcome, torque peak, progress, speed).
13. First unexpected outcome stops the task; the Runner NEVER auto-continues past a cancel.

- [ ] **Step 1: Write the failing tests**

`FakeClient` scripts outcomes; keep it minimal and reusable:

```python
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind, Leg
from rammp_box_opening.runtime.runner import LegResult, Runner


def _traj(start, end, dt=1.0):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for i, row in enumerate([start, end]):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


Q0 = [0.0] * 7
Q1 = [0.1] * 7
Q2 = [0.2] * 7


class FakeClient:
    def __init__(self):
        self.live = list(Q0)
        self.efforts = True
        self.exec_enabled = True
        self.executed = []          # (leg-ish label, n_points, speed)
        self.worlds_pushed = []
        self.plans = []             # scripted plan_to_* responses (FIFO), else auto
        self.exec_script = []       # scripted execute outcomes (FIFO)
        self.tool_z = 0.08

    def joints(self):
        return list(self.live)

    def wrist_efforts(self):
        return [0.0] * 4 if self.efforts else None

    def efforts_present(self):
        return self.efforts

    def tool_xyz(self, timeout_s=1.5):
        return [0.45, 0.0, self.tool_z]

    def _plan(self, end, start):
        class R:
            success = True
            message = "ok"
        R.trajectory = _traj(start if start else self.live, end)
        return R

    def plan_to_pose(self, xyz, quat_xyzw, start_joints):
        if self.plans:
            return self.plans.pop(0)
        return self._plan(Q1, start_joints)

    def plan_to_joints(self, q7, start_joints):
        if self.plans:
            return self.plans.pop(0)
        return self._plan(list(q7), start_joints)

    def execute(self, traj, speed, guard=None):
        self.executed.append((len(traj.points), speed))
        if self.exec_script:
            outcome, info = self.exec_script.pop(0)
        else:
            outcome, info = "arrived", {"message": "ok", "progress": 1.0,
                                        "torque_peak": 0.0}
        if outcome != "failed":
            self.live = list(traj.points[-1].positions)
        return outcome, info

    def set_world(self, path_or_name):
        self.worlds_pushed.append(str(path_or_name))
        return True, "ok"

    def planner_execute_enabled(self):
        return self.exec_enabled

    def gripper_cmd(self, position):
        return True, float(position), False


class FakeStore:
    def push_name(self, kind, **kw):
        tag = kw.get("tag", "")
        name = kind + (("_" + tag) if tag else "")
        return name, name + ".yaml"


def leg(name, start=Q0, end=Q1, chain=0, speed=0.25, world="full",
        guard=None, kind=Kind.MOTION, verify=None, invalidates=False,
        cmd=None):
    return Leg(name=name, kind=kind,
               traj=_traj(start, end) if kind is Kind.MOTION else None,
               speed=speed, guard=guard, world=world, chain=chain,
               target=("joints", end) if kind is Kind.MOTION else None,
               goal_joints=end if kind is Kind.MOTION else None,
               invalidates_downstream=invalidates, verify=verify,
               gripper_cmd=cmd)


def runner(client, tmp_path):
    return Runner(client, FakeStore(), log_dir=tmp_path)


def test_dry_run_executes_nothing(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run([leg("a")], execute=False)
    assert [r.outcome for r in res] == ["skipped"]
    assert c.executed == [] and c.worlds_pushed == []


def test_merged_group_is_one_execution(tmp_path):
    c = FakeClient()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1                    # merged: one goal


def test_unguarded_leg_requires_full_world(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("a", world="interaction_button")], execute=True, assume_yes=True)
    assert res[0].outcome == "refused" and c.executed == []


def test_gripper_gate_reads_planner_param(tmp_path):
    c = FakeClient()
    c.exec_enabled = False
    gl = leg("close", kind=Kind.GRIPPER, cmd=0.8)
    res = runner(c, tmp_path).run([gl], execute=True, assume_yes=True)
    assert res[0].outcome == "refused"
    assert "execute" in res[0].detail


def test_guarded_leg_refused_without_efforts(tmp_path):
    c = FakeClient()
    c.efforts = False
    g = GuardSpec(touch_nm=3.0, trip="press", depth_window=(0.004, 0.012),
                  target_z=0.09)
    res = runner(c, tmp_path).run([leg("press", guard=g, world="interaction_b")],
                                  execute=True, assume_yes=True)
    assert res[0].outcome == "refused" and c.executed == []


def test_start_drift_triggers_replan(tmp_path):
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6                     # 0.06 > 0.04 threshold
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1)], execute=True, assume_yes=True)
    assert res[0].ok
    # re-planned from live: executed trajectory starts at the live joints
    assert c.executed and c.executed[0][0] == 2


def test_no_motion_retry_once(tmp_path):
    c = FakeClient()
    c.exec_script = [
        ("failed", {"message": "goal aborted: arm never left the start",
                    "progress": 0.0, "torque_peak": None}),
        ("arrived", {"message": "ok", "progress": 1.0, "torque_peak": None}),
    ]
    res = runner(c, tmp_path).run([leg("a")], execute=True, assume_yes=True)
    assert res[0].ok and len(c.executed) == 2


def test_other_failure_stops_without_retry(tmp_path):
    c = FakeClient()
    c.exec_script = [("failed", {"message": "controller rejected",
                                 "progress": 0.4, "torque_peak": None})]
    res = runner(c, tmp_path).run([leg("a"), leg("b", Q1, Q2)],
                                  execute=True, assume_yes=True)
    assert res[0].outcome == "failed"
    assert len(res) == 1 and len(c.executed) == 1   # stopped, b never ran


def test_contact_invalidates_downstream(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown", target_z=0.0)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.5,
                                "torque_peak": 4.0})]
    legs = [leg("descend", guard=g, world="interaction_x", invalidates=True,
                chain=0),
            leg("after", Q1, Q2, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True, assume_yes=True)
    assert res[0].outcome == "touch" and res[0].ok    # setdown: trip = success
    assert res[1].ok
    assert len(c.executed) == 2                       # never merged with contact


def test_press_verify_uses_depth(tmp_path):
    from rammp_box_opening.runtime.guards import press_outcome

    c = FakeClient()
    c.tool_z = 0.082                                  # 8 mm below target_z 0.09
    g = GuardSpec(touch_nm=3.0, trip="press", depth_window=(0.004, 0.012),
                  target_z=0.09)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.6,
                                "torque_peak": 4.2})]
    v = lambda ctx: press_outcome(ctx.outcome, ctx.depth_m, (0.004, 0.012))
    res = runner(c, tmp_path).run(
        [leg("press", guard=g, world="interaction_b", verify=v,
             invalidates=True)],
        execute=True, assume_yes=True)
    assert res[0].ok and "pressed" in res[0].detail


def test_jsonl_log_written(tmp_path):
    import json

    c = FakeClient()
    runner(c, tmp_path).run([leg("a")], execute=True, assume_yes=True)
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    row = json.loads(logs[0].read_text().splitlines()[0])
    assert row["leg"] == "a" and row["outcome"] == "arrived"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_runner.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation**

`runtime/confirm.py`:

```python
"""Typed-confirm gate (plan_and_execute.py pattern): exact 'yes' or abort."""


def typed_yes(prompt):
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    return answer.strip() == "yes"
```

`runtime/runner.py` — implement the 13 semantics above. Skeleton with the load-bearing logic:

```python
"""The Runner: gates, merging, retries, run log (spec §6)."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import DRIFT_REPLAN_RAD, SANITY_MARGIN_RAD
from rammp_box_opening.runtime import confirm
from rammp_box_opening.runtime.guards import TorqueGuard, sanity_violations
from rammp_box_opening.runtime.legs import (
    Kind,
    VerifyCtx,
    merge_groups,
    merge_trajectories,
)

NO_MOTION_SIGNATURE = "never left the start"


@dataclass
class LegResult:
    leg_name: str
    outcome: str          # arrived | touch | failed | refused | skipped
    ok: bool
    detail: str = ""
    torque_peak: float = None
    progress: float = None
    t_wall: float = None


class Runner:
    def __init__(self, client, world_store, log_dir=None,
                 margin_rad=SANITY_MARGIN_RAD):
        self.client = client
        self.worlds = world_store
        self.margin_rad = margin_rad
        self._log_dir = Path(
            log_dir
            if log_dir is not None
            else Path.home() / ".ros" / "rammp_box_opening" / "runs"
        )
        self._log_path = None
        self._last_world = None

    # -- preview -----------------------------------------------------------
    def preview(self, legs):
        rows = ["%-18s %-7s %5s %-18s %5s %8s"
                % ("leg", "kind", "speed", "world", "chain", "time_s")]
        for leg in legs:
            secs = "-"
            if leg.kind is Kind.MOTION and leg.traj is not None:
                last = leg.traj.points[-1].time_from_start
                secs = "%.2f" % ((last.sec + last.nanosec * 1e-9) / leg.speed)
            rows.append("%-18s %-7s %5.2f %-18s %5d %8s"
                        % (leg.name, leg.kind.value, leg.speed, leg.world,
                           leg.chain, secs))
        return "\n".join(rows)

    # -- gates (pre-flight, before anything executes) ----------------------
    def _refusal(self, leg):
        if leg.kind is Kind.MOTION:
            if leg.guard is None and not leg.world.startswith("full"):
                return ("unguarded MOTION leg planned against %r — full world "
                        "required (spec §6)" % leg.world)
            if leg.traj is not None:
                bad = sanity_violations(leg.traj, self.margin_rad)
                if bad:
                    return "trajectory sanity gate: " + "; ".join(bad)
            if leg.guard is not None and not self.client.efforts_present():
                return ("no effort fields in /joint_states — guarded legs "
                        "refuse to run (spec §6)")
        if leg.kind is Kind.GRIPPER and not self.client.planner_execute_enabled():
            return ("planner execute param is false — refusing GRIPPER leg "
                    "(planner dry-run does NOT gate the direct gripper "
                    "action; the runner enforces symmetry)")
        return None

    # -- run ---------------------------------------------------------------
    def run(self, legs, execute, assume_yes=False):
        print(self.preview(legs))
        if not execute:
            print("dry-run complete — nothing moved (add --execute)")
            return [LegResult(leg.name, "skipped", True) for leg in legs]
        if not assume_yes and not confirm.typed_yes(
            "Type 'yes' to execute (human on the physical e-stop): "
        ):
            print("aborted — nothing moved")
            return [LegResult(leg.name, "skipped", True) for leg in legs]

        for leg in legs:                      # hard refusals BEFORE any motion
            why = self._refusal(leg)
            if why:
                res = LegResult(leg.name, "refused", False, why)
                self._log(res, leg)
                return [res]

        results = []
        stale = False
        next_chain = max((leg.chain for leg in legs), default=0) + 1
        for group in merge_groups(legs):
            leg = group[0]
            # world re-assertion (SetWorld is write-only: track, re-push)
            if leg.world != self._last_world:
                ok, msg = self.client.set_world(leg.world)
                if not ok:
                    res = LegResult(leg.name, "refused", False,
                                    "set_world failed: " + msg)
                    self._log(res, leg)
                    results.append(res)
                    return results
                self._last_world = leg.world

            if leg.kind is Kind.GRIPPER:
                res = self._run_gripper(leg)
            else:
                if stale or self._drifted(group):
                    group, next_chain = self._replan_group(group, next_chain)
                    if group is None:
                        res = LegResult(leg.name, "failed", False,
                                        "re-plan from live state failed")
                        self._log(res, leg)
                        results.append(res)
                        return results
                    leg = group[0]
                res = self._run_motion(group)
                if any(g.invalidates_downstream for g in group):
                    stale = True
            for g in group:
                self._log(res, g)
            results.append(res)
            if not res.ok:
                print("STOP: leg %s -> %s (%s) — arm holds"
                      % (res.leg_name, res.outcome, res.detail))
                return results
        return results

    def _drifted(self, group):
        start = group[0].traj.points[0].positions
        live = self.client.joints()
        return max(abs(ang_diff(a, b)) for a, b in zip(live, start)) \
            > DRIFT_REPLAN_RAD

    def _replan_group(self, group, next_chain):
        """Re-plan each leg of the group from live state, same targets."""
        live = self.client.joints()
        out = []
        for leg in group:
            kind, *rest = leg.target
            if kind == "pose":
                plan = self.client.plan_to_pose(rest[0], rest[1], live)
            else:
                plan = self.client.plan_to_joints(rest[0], live)
            if plan is None or not plan.success:
                return None, next_chain
            leg.traj = plan.trajectory
            leg.chain = next_chain
            leg.stale = False
            live = list(plan.trajectory.points[-1].positions)
            out.append(leg)
        return out, next_chain + 1

    def _run_motion(self, group):
        leg = group[0]
        traj = (merge_trajectories([g.traj for g in group])
                if len(group) > 1 else leg.traj)
        guard = TorqueGuard(leg.guard.touch_nm) if leg.guard else None
        t0 = time.monotonic()
        outcome, info = self.client.execute(traj, leg.speed, guard=guard)
        if (outcome == "failed"
                and NO_MOTION_SIGNATURE in info.get("message", "")):
            print("  no-motion fault at start — one retry from standstill")
            time.sleep(3.0)
            outcome, info = self.client.execute(traj, leg.speed, guard=guard)
        depth = None
        if outcome == "touch" and leg.guard and leg.guard.trip == "press":
            tool = self.client.tool_xyz()
            depth = (leg.guard.target_z - tool[2]) if tool else None
        ok = self._leg_ok(leg, outcome)
        detail = info.get("message", "")
        if leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(
                outcome=outcome, depth_m=depth,
                progress=info.get("progress"),
                torque_peak=info.get("torque_peak")))
        return LegResult(group[-1].name if len(group) > 1 else leg.name,
                         outcome, ok, detail,
                         torque_peak=info.get("torque_peak"),
                         progress=info.get("progress"),
                         t_wall=time.monotonic() - t0)

    @staticmethod
    def _leg_ok(leg, outcome):
        if leg.guard is None:
            return outcome == "arrived"
        if leg.guard.trip == "setdown":
            return outcome == "touch"
        if leg.guard.trip == "obstruction":
            return outcome == "arrived"      # trip = struck the lid/rim
        return outcome == "touch"            # press: verify refines via depth
    
    def _run_gripper(self, leg):
        t0 = time.monotonic()
        ok, pos, stalled = self.client.gripper_cmd(leg.gripper_cmd)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s" % (pos, " (stalled)" if stalled else "")
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail,
                         t_wall=time.monotonic() - t0)

    def _log(self, res, leg):
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if self._log_path is None:
            self._log_path = self._log_dir / time.strftime("run-%Y%m%d-%H%M%S.jsonl")
        row = {"t": time.time(), "leg": leg.name, "kind": leg.kind.value,
               "world": leg.world, "speed": leg.speed, **asdict(res)}
        row.pop("leg_name", None)
        row["outcome"] = res.outcome
        with open(self._log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_runner.py -q`
Expected: PASS (11 tests). Fix the implementation, not the tests, until the semantics hold.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: runner with gates, merging, retries, run log (Task 7)"
```

---

### Task 8: Primitives

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/primitives/core.py` (Ctx, guarded-descent builder, all seven primitives — they share machinery and are small)
- Test: `src/rammp_box_opening/test/test_primitives.py`

**Interfaces:**

- Consumes: model (Task 2), guards (Task 3), legs (Task 4), worlds (Task 5), client protocol (Task 6 — plan-time only), constants.
- Produces (tasks compose these):
  - `Ctx(model, cpose, client, worlds, lid_at=None)` (dataclass; `lid_at` set after place for world generation)
  - `PlanState(joints, chain, contact_broke_chain)` — threaded through planning
  - Primitives, each `plan(ctx, state) -> tuple[list[Leg], PlanState]`:
    - `Approach(target_xyz, quat, name)` — hover transit, world full, TRANSIT_SPEED
    - `Press()` — close-gripper GRIPPER leg + guarded descent (trip="press", window+touch_nm from model, target = button top, world interaction tag "button", invalidates_downstream, verify=press_outcome)
    - `Grasp(spec, name)` — guarded descent (trip="obstruction") + GRIPPER close to `model.width_to_command(spec.width_m)` with band verify
    - `Lift(dz)` — planned ascent, verify grip band still held
    - `Place(target_xyz, quat, open_after=True)` — hover transit + guarded descent (trip="setdown") + gripper open
    - `Retreat(dz)` — ascend to hover height (planned; the Runner's reverse-retrace covers the plan-fails case)
    - `Home()` — `plan_to_joints(HOME)`
  - `hover_above(xyz, standoff) -> xyz` helper; descent legs check `check_standoff` at build time.
  - Chain rules: legs planned with `start_joints` = predecessor's predicted end share a chain; a contact leg increments the chain for what follows and previews from the descent plan's end joints (nominal contact depth, spec §5).

- [ ] **Step 1: Write the failing tests**

Reuse `FakeClient` from Task 7's test module (import it):

```python
from test_runner import FakeClient, FakeStore

from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    attitude_quat,
    from_container,
)
from rammp_box_opening.primitives.core import (
    Approach,
    Ctx,
    Grasp,
    Home,
    Lift,
    Place,
    PlanState,
    Press,
    hover_above,
)
from rammp_box_opening.runtime.legs import Kind

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def ctx():
    m = ContainerModel.load(CFG)
    return Ctx(model=m, cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
               client=FakeClient(), worlds=FakeStore())


def state():
    return PlanState(joints=[0.0] * 7, chain=0, contact_broke_chain=False)


def test_press_emits_close_then_guarded_descent():
    legs, st = Press().plan(ctx(), state())
    assert [leg.kind for leg in legs] == [Kind.GRIPPER, Kind.MOTION]
    close, descend = legs
    assert close.gripper_cmd == 0.8
    assert descend.guard is not None and descend.guard.trip == "press"
    assert descend.world.startswith("interaction")
    assert descend.invalidates_downstream
    assert descend.speed == 0.15
    assert st.chain > 0                       # contact broke the chain


def test_press_descent_verify_classifies():
    legs, _ = Press().plan(ctx(), state())
    descend = legs[1]
    from rammp_box_opening.runtime.legs import VerifyCtx

    ok, detail = descend.verify(VerifyCtx(outcome="touch", depth_m=0.008))
    assert ok
    ok, _ = descend.verify(VerifyCtx(outcome="arrived"))
    assert not ok                             # bottomed out untripped


def test_grasp_trip_is_failure_and_band_checked():
    c = ctx()
    legs, _ = Grasp(c.model.lid_grasp, "grasp:lid").plan(c, state())
    descend = [leg for leg in legs if leg.kind is Kind.MOTION][0]
    close = [leg for leg in legs if leg.kind is Kind.GRIPPER][0]
    assert descend.guard.trip == "obstruction"
    from rammp_box_opening.runtime.legs import VerifyCtx

    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.6))
    assert ok                                  # inside expect_band
    ok, detail = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.8))
    assert not ok                              # closed on air


def test_approach_targets_hover_not_contact():
    c = ctx()
    button = from_container(c.cpose, c.model.button_offset)
    hov = hover_above(button, c.model.hover_standoff)
    legs, _ = Approach(hov, attitude_quat(c.model.press_attitude_rpy_deg, 0.0),
                       "approach:button").plan(c, state())
    assert len(legs) == 1 and legs[0].world.startswith("full")
    assert legs[0].speed == 0.25
    assert legs[0].target[1][2] > button[2]    # hover, never contact depth


def test_chaining_start_joints_flow():
    c = ctx()
    st = state()
    legs_a, st = Approach([0.45, 0.0, 0.1], [0.5, 0.5, 0.5, 0.5],
                          "approach:a").plan(c, st)
    legs_b, st = Home().plan(c, st)
    # same chain (no contact between them) and b planned from a's end
    assert legs_a[0].chain == legs_b[0].chain
    assert legs_b[0].traj.points[0].positions == legs_a[0].goal_joints


def test_place_sequence_and_lid_world():
    c = ctx()
    legs, _ = Place([0.45, -0.25, -0.07 + c.model.lid_dims[2]],
                    [0.5, 0.5, 0.5, 0.5]).plan(c, state())
    kinds = [leg.kind for leg in legs]
    assert kinds == [Kind.MOTION, Kind.MOTION, Kind.GRIPPER]
    transit, descend, open_ = legs
    assert descend.guard.trip == "setdown"
    assert open_.gripper_cmd == 0.0


def test_lift_reverifies_band():
    c = ctx()
    legs, _ = Lift(0.10, band=c.model.lid_grasp.expect_band).plan(c, state())
    assert legs[0].verify is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_primitives.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Write the implementation**

`primitives/core.py` essentials (full file follows this structure; every primitive plans via `ctx.client.plan_to_pose/plan_to_joints` with `start_joints=state.joints` and returns updated `PlanState`):

```python
"""The primitives (spec §5): plan/execute split, guarded descents shared.

Each primitive chain-plans from `state.joints` (tour_demo chaining). A
contact leg invalidates downstream pre-plans: it increments the chain and
the next primitive previews from the descent plan's end joints (nominal
contact depth)."""

import math
from dataclasses import dataclass, replace

from rammp_box_opening.constants import (
    CONTACT_SPEED,
    GRIPPER_CMD_CLOSED,
    GRIPPER_CMD_OPEN,
    HOME,
    TRANSIT_SPEED,
)
from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.runtime.guards import (
    GuardSpec,
    check_standoff,
    in_band,
    press_outcome,
)
from rammp_box_opening.runtime.legs import Kind, Leg, VerifyCtx


@dataclass
class Ctx:
    model: object
    cpose: object
    client: object
    worlds: object
    lid_at: object = None


@dataclass
class PlanState:
    joints: list
    chain: int
    contact_broke_chain: bool


def hover_above(xyz, standoff):
    return [xyz[0], xyz[1], xyz[2] + standoff]


def _plan_motion(ctx, state, name, target, world, speed, guard=None,
                 invalidates=False, verify=None):
    kind, *rest = target
    if kind == "pose":
        plan = ctx.client.plan_to_pose(rest[0], rest[1], state.joints)
    else:
        plan = ctx.client.plan_to_joints(rest[0], state.joints)
    if plan is None or not plan.success:
        raise RuntimeError("planning failed for %s: %s"
                           % (name, getattr(plan, "message", "no response")))
    end = list(plan.trajectory.points[-1].positions)
    leg = Leg(name=name, kind=Kind.MOTION, traj=plan.trajectory, speed=speed,
              guard=guard, world=world, chain=state.chain, target=target,
              goal_joints=end, invalidates_downstream=invalidates,
              verify=verify)
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain,
                          contact_broke_chain=invalidates)
```

Then the primitives (all `plan(self, ctx, state)`):

- `Approach(target_xyz, quat, name)`: full world via `ctx.worlds.push_name("full", model=ctx.model, cpose=ctx.cpose, lid_at=ctx.lid_at)`; one `_plan_motion(..., ("pose", xyz, quat), world_name, TRANSIT_SPEED)`.
- `Press()`: button top = `from_container(cpose, model.button_offset)`; quat = `attitude_quat(model.press_attitude_rpy_deg, cpose.yaw)`; `check_standoff(button_z + hover_standoff, button_z)`; GRIPPER close leg (`gripper_cmd=GRIPPER_CMD_CLOSED`, chain=state.chain); descent target z = `button_z - depth_window[1]`; interaction world `push_name("interaction", ..., target_xyz=button, contact_z=button_z, depth_max=window[1], tag="button")`; `GuardSpec(touch_nm=model.touch_nm, trip="press", depth_window=window, target_z=button_z)`; `verify=lambda v: press_outcome(v.outcome, v.depth_m, window)`; CONTACT_SPEED; invalidates.
- `Grasp(spec, name)`: grasp point = `from_container(cpose, spec.offset)`; descent from hover to grasp z with `GuardSpec(trip="obstruction", touch_nm=model.touch_nm, target_z=grasp_z)`, interaction world tag from name, invalidates; then GRIPPER close to `model.width_to_command(spec.width_m)` with `verify=lambda v: (in_band(v.gripper_pos, spec.expect_band), "grip %.3f vs band %s" % (v.gripper_pos, spec.expect_band))`.
- `Lift(dz, band=None)`: `_plan_motion` to `("pose", [x, y, z + dz], same quat)` — planned from state (post-contact the Runner re-plans from live); verify = band re-check when `band` given (slip detection): the Runner passes `gripper_pos` for MOTION legs too by reading `client.gripper_cmd`-reported position? No — keep honest: `Lift`'s verify closure captures `ctx.client` and reads the LIVE gripper feedback is not in the protocol; instead the band re-check is a GRIPPER no-op leg: emit `Leg(kind=GRIPPER, gripper_cmd=None, verify=band check)` — the Runner's `_run_gripper` treats `gripper_cmd=None` as "query only": calls `client.gripper_cmd` with None → FakeClient/real client return current position without motion. Add that convention to Task 6/7 while implementing (real client sends NO goal when position is None; returns last known feedback via a `GripperCommand` send with unchanged position is NOT acceptable — instead subscribe once to the gripper controller state topic; if unavailable, verify returns (True, "band unchecked — no gripper state") and logs it).
- `Place(target_xyz, quat, open_after=True)`: transit to hover (hover includes `model.lid_dims[2]` margin — held lid carried explicitly, spec §6), guarded descent `trip="setdown"` (invalidates), then GRIPPER open (`GRIPPER_CMD_OPEN`).
- `Retreat(dz)`: `_plan_motion` straight up by dz from state joints' pose — planned; the plan-free reverse-retrace lives in the Runner as the fallback when post-contact planning fails.
- `Home()`: `_plan_motion(..., ("joints", HOME), full world, TRANSIT_SPEED)`.

Where a primitive needs the current tool pose to compute an ascent target at plan time and the state is post-contact, plan from nominal (descent end) — the Runner re-plans stale legs from live at execution time (Task 7 semantics 8).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test/test_primitives.py -q`
Expected: PASS (7 tests). Then the full suite: `python3 -m pytest src/rammp_box_opening/test -q`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: primitives with shared guarded descent (Task 8)"
```

---

### Task 9: Tasks + CLI surface

**Files:**

- Create: `src/rammp_box_opening/rammp_box_opening/tasks/open_container.py`
- Create: `src/rammp_box_opening/rammp_box_opening/tasks/pickup_container.py`
- Create: `src/rammp_box_opening/rammp_box_opening/tasks/cli_common.py`
- Create: `src/rammp_box_opening/rammp_box_opening/tasks/primitive_clis.py`
- Create: `src/rammp_box_opening/rammp_box_opening/tasks/smoke_plan.py`
- Create: `src/rammp_box_opening/rammp_box_opening/tasks/preflight.py`
- Modify: `src/rammp_box_opening/setup.py` (fill `console_scripts`)
- Test: `src/rammp_box_opening/test/test_tasks.py`

**Interfaces:**

- Consumes: everything above.
- Produces:
  - `open_container.build_legs(ctx) -> list[Leg]` — approach(above_button) → press → retreat → approach(above_lid) → grasp(lid) → lift(0.10) → place(lid_place) → home (spec §5); after Place the ctx gains `lid_at` so later worlds carry the lid cuboid
  - `pickup_container.build_legs(ctx, hold=False) -> list[Leg]` — approach(above_body) → grasp(body) → lift(0.10) → \[place(pickup pose) unless hold\] → home
  - `cli_common.make_parser(desc)` — `--execute`, `--speed` (transit override), `--container` (config path, default the installed oxo_pop.yaml via `ament_index_python` share lookup), `--lid-place x y z` override; `cli_common.build_ctx(args, node) -> (ctx, runner)`; refuses hardware execution while `model.measure_me` is true (dry-run always allowed)
  - `primitive_clis`: one `main_<name>()` per primitive (approach/press/grasp/lift/place/retreat/home) — each builds JUST its primitive's legs (attended bring-up isolation, spec §4)
  - `smoke_plan.main()` — `plan_to_pose([0.45, 0.0, 0.35], wrist_flat_quat(...), start_joints=HOME)`, print the plan_and_execute-style excursion table. NO execution path at all
  - `preflight.main()` — checks and reports: `/joint_states` fresh WITH effort fields; planner actions reachable; planner `execute` param value; controllers responding (`/controller_manager/list_controllers` service, 3 s timeout — report-only); idempotent full-world push via `set_world`
  - `console_scripts`: `open_container`, `pickup_container`, `approach`, `press`, `grasp`, `lift`, `place`, `retreat`, `home_arm`, `smoke_plan`, `preflight` (all `rammp_box_opening.tasks.…:main…`)

- [ ] **Step 1: Write the failing tests**

```python
from test_runner import FakeClient, FakeStore

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.legs import Kind
from rammp_box_opening.tasks import open_container, pickup_container

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def ctx():
    return Ctx(model=ContainerModel.load(CFG),
               cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
               client=FakeClient(), worlds=FakeStore())


def names(legs):
    return [leg.name for leg in legs]


def test_open_container_sequence():
    legs = open_container.build_legs(ctx())
    seq = names(legs)
    assert seq[0].startswith("approach:button")
    assert any(n.startswith("press") for n in seq)
    assert any(n.startswith("retreat") for n in seq)
    assert any(n.startswith("grasp:lid") for n in seq)
    assert any(n.startswith("lift") for n in seq)
    assert any(n.startswith("place:lid") for n in seq)
    assert seq[-1].startswith("home")
    # ordering: press before grasp, grasp before lift, lift before place
    assert seq.index(next(n for n in seq if n.startswith("press"))) < \
        seq.index(next(n for n in seq if n.startswith("grasp:lid")))


def test_open_container_worlds_carry_lid_after_place():
    legs = open_container.build_legs(ctx())
    place_i = next(i for i, leg in enumerate(legs)
                   if leg.name.startswith("place:lid") and leg.kind is Kind.MOTION)
    after = [leg for leg in legs[place_i + 1:] if leg.kind is Kind.MOTION]
    assert after, "home leg expected after place"
    assert all("lid" in leg.world for leg in after), \
        "post-place worlds must include the placed-lid cuboid"


def test_pickup_places_back_by_default_and_holds_on_request():
    default = names(pickup_container.build_legs(ctx()))
    assert any(n.startswith("place:container") for n in default)
    held = names(pickup_container.build_legs(ctx(), hold=True))
    assert not any(n.startswith("place:") for n in held)


def test_entry_points_registered():
    import configparser
    from pathlib import Path

    setup = Path("src/rammp_box_opening/setup.py").read_text()
    for ep in ["open_container", "pickup_container", "smoke_plan", "preflight",
               "press", "grasp", "home_arm"]:
        assert ep + " = " in setup
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest src/rammp_box_opening/test/test_tasks.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

`tasks/open_container.py` core:

```python
"""open_container: press the button, lift the lid clear, set it down."""

from rammp_box_opening.models.container import (
    attitude_quat,
    from_container,
    load_lid_place,
    wrist_flat_quat,
)
from rammp_box_opening.primitives.core import (
    Approach,
    Ctx,
    Grasp,
    Home,
    Lift,
    Place,
    PlanState,
    Press,
    Retreat,
    hover_above,
)

LIFT_DZ = 0.10


def build_legs(ctx, lid_place=None):
    m, cp = ctx.model, ctx.cpose
    lid_at = lid_place if lid_place is not None else load_lid_place(ctx.config_path)
    button = from_container(cp, m.button_offset)
    lid_grasp_pt = from_container(cp, m.lid_grasp.offset)
    press_quat = attitude_quat(m.press_attitude_rpy_deg, cp.yaw)
    grasp_quat = attitude_quat(m.lid_grasp.attitude_rpy_deg, cp.yaw)

    state = PlanState(joints=list(ctx.client.joints()), chain=0,
                      contact_broke_chain=False)
    legs = []
    for prim in [
        Approach(hover_above(button, m.hover_standoff), press_quat,
                 "approach:button"),
        Press(),
        Retreat(m.hover_standoff),
        Approach(hover_above(lid_grasp_pt, m.hover_standoff), grasp_quat,
                 "approach:lid"),
        Grasp(m.lid_grasp, "grasp:lid"),
        Lift(LIFT_DZ, band=m.lid_grasp.expect_band),
    ]:
        new, state = prim.plan(ctx, state)
        legs += new
    # the lid is now in the gripper: place it, then all later worlds carry it
    place_xyz = [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + m.lid_dims[2]]
    new, state = Place(place_xyz, grasp_quat, name="place:lid").plan(ctx, state)
    legs += new
    ctx.lid_at = lid_at            # worlds generated after this include the lid
    new, state = Home().plan(ctx, state)
    legs += new
    return legs


def main():
    from rammp_box_opening.tasks import cli_common

    args = cli_common.make_parser(__doc__).parse_args()
    cli_common.run_task(args, build_legs)
```

(`Ctx` gains a `config_path` field — add it in Task 8's dataclass while implementing, defaulted `None`; `cli_common.build_ctx` sets it.)

`tasks/pickup_container.py`: same shape; `build_legs(ctx, hold=False)` = Approach(above body grasp) → `Grasp(m.body_grasp, "grasp:body")` → `Lift(LIFT_DZ, band=m.body_grasp.expect_band)` → (`Place(pickup pose, name="place:container")` unless `hold`) → Home. `main()` adds `--hold`.

`tasks/cli_common.py`: argparse (`--execute`, `--speed`, `--container`, `--lid-place`, nargs=3), `rclpy.init` + node `rammp_box_opening`, `PlannerClient`, `WorldStore` (bench yaml via ament share), `ConfigPoseSource`, `Runner`; `run_task(args, build_legs, **kw)` = build ctx → legs → `runner.run(legs, execute=args.execute)`; **refusal**: if `args.execute and model.measure_me` → exit "container config still carries measure_me: true — run the Phase-1 measurement worksheet first (dry-run is fine)".

`tasks/primitive_clis.py`: `main_approach`, `main_press`, `main_grasp` (`--grasp lid|body`), `main_lift`, `main_place`, `main_retreat`, `main_home` — each builds its primitive's legs from the model/pose exactly as the task does and hands them to `run_task`-style plumbing. The press CLI states in `--help` that the planner's dry-run does NOT gate the gripper close (Runner enforces it — spec §6).

`tasks/smoke_plan.py`: standalone Phase-0 round-trip:

```python
"""Phase-0 smoke: plan a pose with explicit start_joints, print it.

Proves build, discovery, and the planner contract end to end. There is
deliberately NO execution path in this tool."""

import rclpy

from rammp_box_opening.constants import HOME
from rammp_box_opening.models.container import wrist_flat_quat
from rammp_box_opening.runtime.client import PlannerClient

TARGET = [0.45, 0.0, 0.35]      # benign frontal pose, well above any bench


def main():
    rclpy.init()
    node = rclpy.create_node("rammp_box_opening_smoke")
    client = PlannerClient(node)
    plan = client.plan_to_pose(TARGET, wrist_flat_quat(TARGET), HOME)
    if plan is None or not plan.success:
        raise SystemExit("SMOKE FAILED: %s"
                         % getattr(plan, "message", "no response"))
    last = plan.trajectory.points[-1].time_from_start
    print("SMOKE OK: %d points, %.2f s at full speed (planning %.2f s)"
          % (len(plan.trajectory.points),
             last.sec + last.nanosec * 1e-9, plan.planning_time))
    for j, name in enumerate(plan.trajectory.joint_names):
        pos = [p.positions[j] for p in plan.trajectory.points]
        print("  %-9s %8.3f -> %8.3f  (excursion %.3f)"
              % (name, pos[0], pos[-1], max(pos) - min(pos)))
```

`tasks/preflight.py`: run the five checks from the Interfaces block, print PASS/FAIL per line, exit nonzero on any FAIL (effort fields and planner reachability are FAILs; controllers/execute-param lines are report-only), then push the full world and report `set_world`'s response. World push uses the measured bench + container model + bench_pose — idempotent by name.

`setup.py` `console_scripts`:

```python
        "console_scripts": [
            "open_container = rammp_box_opening.tasks.open_container:main",
            "pickup_container = rammp_box_opening.tasks.pickup_container:main",
            "approach = rammp_box_opening.tasks.primitive_clis:main_approach",
            "press = rammp_box_opening.tasks.primitive_clis:main_press",
            "grasp = rammp_box_opening.tasks.primitive_clis:main_grasp",
            "lift = rammp_box_opening.tasks.primitive_clis:main_lift",
            "place = rammp_box_opening.tasks.primitive_clis:main_place",
            "retreat = rammp_box_opening.tasks.primitive_clis:main_retreat",
            "home_arm = rammp_box_opening.tasks.primitive_clis:main_home",
            "smoke_plan = rammp_box_opening.tasks.smoke_plan:main",
            "preflight = rammp_box_opening.tasks.preflight:main",
        ],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest src/rammp_box_opening/test -q` (full suite).
Expected: PASS, all tasks' tests included.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: tasks, per-primitive CLIs, smoke_plan, preflight (Task 9)"
```

---

### Task 10: Phase-0 exit — build + suite + live smoke

**Files:** none new (verification task).

- [ ] **Step 1: Clean build + full suite**

```zsh
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source ~/RAMMP-CuRobo/install/setup.zsh
cd ~/RAMMP-box-opening
rm -rf build install log
colcon build --symlink-install
source install/setup.zsh
python3 -m pytest src/rammp_box_opening/test -q
```

Expected: build clean, whole suite green.

- [ ] **Step 2: Pre-commit clean**

Run: `pre-commit run --all-files` (if pre-commit is installed; otherwise `ruff check . && ruff format --check .`).
Expected: no diffs.

- [ ] **Step 3: Live smoke against the planner (planning-only)**

The planner with `execute:=false` (the default) is planning-only — nothing can move, and `smoke_plan` has no execution path at all. If the planner is not already running, launch it in a background shell:

```zsh
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.zsh
source ~/RAMMP-CuRobo/install/setup.zsh
ros2 launch rammp_curobo_ros planner.launch.py config:=gen3_real.yaml
```

Wait for "rammp_curobo ready — execute=False", then in a second shell (full chain + this ws):

```zsh
ros2 run rammp_box_opening smoke_plan
```

Expected: `SMOKE OK: … points` and a sane excursion table. If the planner cannot start (cuRobo/GPU issue), STOP and report — do not work around.

- [ ] **Step 4: Commit any fixes + tag the milestone in the commit message**

```bash
git add -A && git commit -m "feat: Phase 0 complete — build, suite, planner smoke green" --allow-empty
```

---

### Task 11: Phase-1 prep — hardware runbook + measurement worksheet

**Files:**

- Create: `docs/HARDWARE_BRINGUP.md`

**Interfaces:**

- Consumes: everything; this document is what the humans at the bench follow.
- Produces: the attended-session procedure Task 12 executes.

- [ ] **Step 1: Write `docs/HARDWARE_BRINGUP.md`** with these sections (full prose, no placeholders):

1. **Session preconditions** — exactly one arm stack; bringup order (human): ros2_kortex from `~/RAMMP-Kinova/ros2_ws`, then `planner.launch.py config:=gen3_real.yaml execute:=true`; every shell exports `ROS_LOCALHOST_ONLY=1`; runtime sourcing chain humble → RAMMP-Kinova → RAMMP-CuRobo → this ws; human on the physical e-stop for every motion.
2. **Measurement worksheet** — tape-measure procedure filling `config/world_bench.yaml` (reconcile once with RAMMP-CuRobo's `world_real_bench.yaml`, err tall on the table) and `config/containers/oxo_pop.yaml`: outer dims, button offset, lid/body grasp offsets + widths, press depth window, hover standoff sanity (> `min_standoff()` = 0.051 m), bench_pose (base_link, +x forward), lid_place. Gripper width→command calibration: close on gauge objects of known width at two points, record `aperture_at_0`/`aperture_at_08` and the observed feedback positions → `expect_band`s. Last step: flip `measure_me: false`.
3. **Every-session preflight** — `ros2 run rammp_box_opening preflight`; then the **abort drill**: start a `home_arm --execute` motion and Ctrl+C mid-motion; confirm the arm stops and holds. No session proceeds past a failed drill.
4. **The attended ladder** (Task 12) — per-primitive CLIs before task CLIs, first runs ≤ 25% speed, each rung's purpose and pass criterion (from spec §8 Phase 1), dry-run each rung before its `--execute` run.
5. **When something trips** — read the leg report + `~/.ros/rammp_box_opening/runs/*.jsonl`; no-motion fault lore (goal transitions; merged runs avoid them); torque/window tuning goes in the container config, never in code.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "docs: hardware bringup runbook + measurement worksheet (Task 11)"
```

---

### Task 12: Phase-1 attended ladder (HUMAN AT THE BENCH — agent prepares, human runs)

**Files:** config value updates from the worksheet; no code (fixes discovered at the bench become their own reviewed commits).

This task is executed WITH Chris per `docs/HARDWARE_BRINGUP.md`. The agent's role is preparation and log analysis between rungs; every `--execute` is typed by the human.

- [ ] **Step 1:** Measurement worksheet completed; `world_bench.yaml` + `oxo_pop.yaml` filled; `measure_me: false`; commit the measured configs.
- [ ] **Step 2:** `preflight` PASS + abort drill PASS.
- [ ] **Step 3:** Ladder, in order, each dry-run first (what each proves, spec §8): `approach` (transit + worlds sane) → `press` (guarded descent + click classification on the real container) → `grasp` (obstruction trip semantics + band) → `lift` (slip re-check) → `place` (set-down trip + release) → full `open_container`.
- [ ] **Step 4:** Exit criterion: **ONE clean end-to-end `open_container`** run, verified from the run log (every leg `ok`, press classified `pressed`, grasp band held through lift). Commit the run log reference + any config tuning.
- [ ] **Step 5:** Update the project memory / spec §10 with anything the bench falsified.

---

## Self-Review (completed at plan time)

- **Spec coverage:** §4 layout → Tasks 1, 9 (CLI surface incl. preflight/smoke_plan); §5 primitives + contact/straightness honesty → Tasks 3, 4, 8; §6 runner/gates/worlds/guard/outcomes/log → Tasks 5, 6, 7; §7 testing rows 1–2 → every task's tests + Task 10 (row 3 sim is optional — not planned; row 4 → Tasks 11–12); §8 Phase 0 → Tasks 1–10, Phase 1 → Tasks 11–12. Phases 2–3: separate plans per spec.
- **Known deviations, on the record:** (1) contact-primitive attitudes come from container config (default top-down) rather than the literal wrist-flat formula — wrist-flat tool z is horizontal and cannot press downward; wrist-flat remains the transit family. Flag to Chris at review. (2) Lift's grip re-check needs gripper state feedback; the plan specifies a query-only gripper leg with an honest "unchecked" fallback (Task 8).
- **Type consistency check:** `PlannerClient` protocol (Task 6) matches `FakeClient` (Task 7 tests) method-for-method; `GuardSpec` fields consistent across Tasks 3/7/8; `Leg` fields consistent across 4/7/8/9; `push_name` signature consistent across 5/7(FakeStore)/8.
