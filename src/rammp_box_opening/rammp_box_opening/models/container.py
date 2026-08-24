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
    hover_m: float  # pre-press waypoint above the tag plane
    travel_m: float  # press depth below the tag plane
    press_speed: float
    staging_m: float  # full-world approach height above the tag
    scan_xyz: tuple
    timeout_s: float
    min_hits: int
    tol_m: float
    window_s: float
    fresh_s: float


def load_press_demo(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    tag, pd = raw["tag"], raw["press_demo"]
    cfg = PressDemoCfg(
        tag_id=int(tag["id"]),
        tag_size_m=float(tag["size_m"]),
        tag_offset=tuple(tag.get("offset_xyz", (0.0, 0.0, 0.0))),
        hover_m=float(pd["hover_m"]),
        travel_m=float(pd["travel_m"]),
        press_speed=float(pd["speed"]),
        staging_m=float(pd["staging_m"]),
        scan_xyz=tuple(raw["scan"]["xyz"]),
        timeout_s=float(raw["detect"]["timeout_s"]),
        min_hits=int(raw["detect"]["min_hits"]),
        tol_m=float(raw["detect"]["tol_m"]),
        window_s=float(raw["detect"]["window_s"]),
        fresh_s=float(raw["detect"]["fresh_s"]),
    )
    if cfg.travel_m <= 0:
        raise ValueError("press_demo.travel_m must be positive")
    if not 0.0 < cfg.press_speed <= 1.0:
        raise ValueError("press_demo.speed outside (0, 1]")
    if cfg.hover_m < 0.01:
        raise ValueError(
            "press_demo.hover_m must be >= 0.01 m — it is the only thing "
            "keeping the UNGUARDED hover leg out of contact (check_standoff "
            "is deliberately skipped for the tag-driven press; the guard "
            "arms on the press stroke, not the hover)"
        )
    if cfg.staging_m <= cfg.hover_m:
        raise ValueError("press_demo.staging_m must exceed hover_m")
    # staging is anchored at the BUTTON TOP; the full-world obstacle tops
    # out at container top + err-tall (2 cm) + collision padding (2 cm):
    # validate against the real geometry, not hover_m (2026-08-24 review)
    clearance = float(raw["dims"][2]) - float(raw["button_offset"][2]) + 0.04
    if cfg.staging_m < clearance:
        raise ValueError(
            "press_demo.staging_m %.3f cannot clear the container cuboid: "
            "dims.z - button_offset.z + err_tall + padding = %.3f m needed"
            % (cfg.staging_m, clearance)
        )
    return cfg
