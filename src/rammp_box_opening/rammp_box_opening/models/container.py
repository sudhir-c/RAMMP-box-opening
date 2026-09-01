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
    return ContainerPose(xyz=tuple(raw["xyz"]), yaw=math.radians(float(raw["yaw_deg"])))


@dataclass(frozen=True)
class PressDemoCfg:
    """Knobs of the tag-driven press flow (owner design 2026-08-24)."""

    tag_id: int
    tag_size_m: float
    tag_offset: tuple  # tag center -> button top center, container frame
    travel_m: float  # press depth below the tag plane
    press_speed: float
    staging_m: float  # full-world approach height above the tag
    scan_xyz: tuple
    timeout_s: float
    min_hits: int
    tol_m: float
    window_s: float
    fresh_s: float
    servo_tol_px: float
    servo_max_iters: int
    servo_min_step_m: float
    servo_max_step_m: float
    grip_band: tuple  # gripper feedback band = holding the popped button
    lift_m: float
    lift_speed: float
    grip_speed: float  # descent onto the popped button (ramp at the bench)
    grip_hop_m: float  # retreat height above the plane between press and grip
    setdown_speed: float  # guarded set-down; the trip IS the success
    merge_press: bool  # one continuous motion instead of approach+stop+press
    merge_press_max_lateral_m: float  # xy limit for allowing the merge
    warp_fast_speed: float  # free-air scale of a warped descent
    warp_slow_frac: float  # final path fraction at contact speed
    grip_clear_m: float  # fingertip stop height above the tag/lid plane
    grip_offset_xy: tuple  # base-frame grasp trim (bench-measured bias)
    detect_period_s: float  # detector tick; min_hits * this = commit floor
    detect_source: str  # 'vlm' | 'depth' | 'tag'
    vlm_model: str
    vlm_target: str  # what to ask Claude to box — plain English
    vlm_timeout_s: float
    vlm_pad_px: int


def load_press_demo(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    tag, pd = raw["tag"], raw["press_demo"]
    cfg = PressDemoCfg(
        tag_id=int(tag["id"]),
        tag_size_m=float(tag["size_m"]),
        tag_offset=tuple(tag.get("offset_xyz", (0.0, 0.0, 0.0))),
        travel_m=float(pd["travel_m"]),
        press_speed=float(pd["speed"]),
        staging_m=float(pd["staging_m"]),
        scan_xyz=tuple(raw["scan"]["xyz"]),
        timeout_s=float(raw["detect"]["timeout_s"]),
        min_hits=int(raw["detect"]["min_hits"]),
        tol_m=float(raw["detect"]["tol_m"]),
        window_s=float(raw["detect"]["window_s"]),
        fresh_s=float(raw["detect"]["fresh_s"]),
        servo_tol_px=float(raw["servo"]["tol_px"]),
        servo_max_iters=int(raw["servo"]["max_iters"]),
        servo_min_step_m=float(raw["servo"]["min_step_m"]),
        servo_max_step_m=float(raw["servo"]["max_step_m"]),
        grip_band=tuple(raw["open_box"]["grip_band"]),
        lift_m=float(raw["open_box"]["lift_m"]),
        lift_speed=float(raw["open_box"]["lift_speed"]),
        grip_speed=float(raw["open_box"].get("grip_speed", 0.15)),
        grip_hop_m=float(raw["open_box"].get("grip_hop_m", 0.12)),
        setdown_speed=float(raw["open_box"].get("setdown_speed", 0.15)),
        merge_press=bool(raw["open_box"].get("merge_press", False)),
        merge_press_max_lateral_m=float(
            raw["open_box"].get("merge_press_max_lateral_m", 0.05)
        ),
        warp_fast_speed=float(raw["open_box"].get("warp_fast_speed", 0.0)),
        warp_slow_frac=float(raw["open_box"].get("warp_slow_frac", 0.3)),
        grip_clear_m=float(raw["open_box"]["grip_clear_m"]),
        grip_offset_xy=tuple(raw["open_box"]["grip_offset_xy"]),
        detect_period_s=float(raw["detect"].get("period_s", 0.15)),
        detect_source=str(raw["detect"].get("source", "tag")),
        vlm_model=str(raw.get("vlm", {}).get("model", "claude-opus-5")),
        vlm_target=str(raw.get("vlm", {}).get("target", "the food-storage container")),
        vlm_timeout_s=float(raw.get("vlm", {}).get("timeout_s", 20.0)),
        vlm_pad_px=int(raw.get("vlm", {}).get("pad_px", 20)),
    )
    if not raw["open_box"]["grip_band"][0] < raw["open_box"]["grip_band"][1] < 0.8:
        raise ValueError(
            "open_box.grip_band must be (lo, hi) with hi < 0.8 — 0.8 is the "
            "closed-on-air feedback and can never mean 'holding the button'"
        )
    if not 0.0 <= cfg.grip_clear_m <= 0.02:
        raise ValueError(
            "open_box.grip_clear_m must be in [0, 0.02] m — 0 puts the "
            "fingertips ON the lid plane, more than 2 cm closes above the knob"
        )
    if any(abs(v) > 0.02 for v in cfg.grip_offset_xy):
        raise ValueError(
            "open_box.grip_offset_xy is a mm-scale bias trim, not an offset "
            "— |each| must be <= 0.02 m (re-measure the tag/mount instead)"
        )
    if cfg.servo_tol_px <= 0 or cfg.servo_max_iters < 0:
        raise ValueError("servo.tol_px must be positive, max_iters >= 0")
    if not 0 < cfg.servo_min_step_m < cfg.servo_max_step_m:
        raise ValueError("servo step bounds must satisfy 0 < min < max")
    if cfg.travel_m <= 0:
        raise ValueError("press_demo.travel_m must be positive")
    if not 0.0 < cfg.press_speed <= 1.0:
        raise ValueError("press_demo.speed outside (0, 1]")
    # Contact-leg speeds are bounded well below transit: the torque guard
    # only arms after the first execution feedback, so the descent travels
    # speed * executor_poll before a baseline exists (BASELINE_TRAVEL_M is
    # the 10 mm budget). 0.5 keeps that inside budget even if the poll
    # regresses to 0.1 s. Raise the poll rate, not this bound.
    for field in ("grip_speed", "setdown_speed", "lift_speed"):
        v = getattr(cfg, field)
        if not 0.0 < v <= 0.5:
            raise ValueError(
                "open_box.%s must be in (0, 0.5] — contact/carry legs are "
                "guard-limited, not transit legs" % field
            )
    if cfg.warp_fast_speed and not 0.0 < cfg.warp_fast_speed <= 0.6:
        raise ValueError(
            "open_box.warp_fast_speed must be in (0, 0.6] — the free-air part "
            "of a GUARDED descent, not a transit leg"
        )
    if not 0.0 < cfg.merge_press_max_lateral_m <= 0.35:
        raise ValueError(
            "open_box.merge_press_max_lateral_m must be in (0, 0.35] — past "
            "the camera's own scan footprint the tag could not have been "
            "seen, and the merged solve plans in the reduced world"
        )
    if not 0.0 < cfg.warp_slow_frac < 1.0:
        raise ValueError("open_box.warp_slow_frac must be in (0, 1)")
    if cfg.detect_source not in ("tag", "depth", "vlm"):
        raise ValueError("detect.source must be 'tag', 'depth' or 'vlm'")
    if not 0.0 < cfg.vlm_timeout_s <= 60.0:
        raise ValueError("vlm.timeout_s must be in (0, 60] s")
    if not 0.0 < cfg.detect_period_s <= 0.5:
        raise ValueError("detect.period_s must be in (0, 0.5] s")
    # staging is anchored at the BUTTON TOP; the full-world obstacle tops
    # out at container top + err-tall (2 cm) + collision padding (2 cm),
    # and the finger collision spheres reach ~6 cm below tool_frame.
    # 0.10 above the cuboid gap is the MEASURED plans/fails boundary
    # (margin probe vs the real planner, 2026-08-25).
    clearance = float(raw["dims"][2]) - float(raw["button_offset"][2]) + 0.10
    if cfg.staging_m < clearance:
        raise ValueError(
            "press_demo.staging_m %.3f cannot clear the container cuboid: "
            "dims.z - button_offset.z + 0.10 (err-tall + padding + gripper "
            "spheres, measured 2026-08-25) = %.3f m needed" % (cfg.staging_m, clearance)
        )
    return cfg
