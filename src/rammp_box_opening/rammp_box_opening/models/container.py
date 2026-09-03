"""Container model: every mission target derives from this + a detected pose.

Attitudes: transit poses use the wrist-flat family (Rz(bearing) ⊗ q_home,
spec §3); contact legs use the press attitude from the container config
(default top-down [180, 0, 0] rpy — wrist-flat tool z is HORIZONTAL, which
cannot press a lid button from above), steered to the target's bearing.
"""

import math
from dataclasses import dataclass

import yaml

from rammp_curobo.geometry import euler_deg_to_quat_xyzw, yaw_about_world_z

from rammp_box_opening.constants import WRIST_FLAT_XYZW


@dataclass(frozen=True)
class ContainerPose:
    xyz: tuple
    yaw: float


@dataclass(frozen=True)
class ContainerModel:
    dims: tuple
    lid_dims: tuple
    button_offset: tuple
    button_diameter_m: float
    touch_nm: float
    press_attitude_rpy_deg: tuple
    hover_standoff: float  # carry/set-down hover above a target (Place)
    measure_me: bool

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            dims=tuple(raw["dims"]),
            lid_dims=tuple(raw["lid_dims"]),
            button_offset=tuple(raw["button_offset"]),
            button_diameter_m=float(raw.get("button_diameter_m", 0.036)),
            touch_nm=float(raw["press"]["touch_nm"]),
            press_attitude_rpy_deg=tuple(raw["press_attitude_rpy_deg"]),
            hover_standoff=float(raw["hover_standoff"]),
            measure_me=bool(raw.get("measure_me", True)),
        )


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


def load_lid_place(path):
    with open(path) as f:
        raw = yaml.safe_load(f)["open_container"]["lid_place"]
    return ContainerPose(xyz=tuple(raw["xyz"]), yaw=math.radians(float(raw["yaw_deg"])))


@dataclass(frozen=True)
class PressDemoCfg:
    """Knobs of the camera-driven press flow (owner design 2026-08-24)."""

    travel_m: float  # press depth below the lid plane
    press_speed: float
    staging_m: float  # full-world approach height above the button
    scan_xyz: tuple
    timeout_s: float
    min_hits: int
    tol_m: float
    window_s: float
    fresh_s: float
    grip_band: tuple  # gripper feedback band = holding the popped button
    lift_m: float
    lift_speed: float
    grip_speed: float  # descent onto the popped button (ramp at the bench)
    grip_hop_m: float  # retreat height above the plane between press and grip
    setdown_speed: float  # guarded set-down; the trip IS the success
    setdown_touch_nm: float  # set-down trip threshold (gentler than the press)
    park_tool_down: bool  # rest at the scan pose between runs, not factory HOME
    press_offset_xy: tuple  # base-frame trim of the press target (m)
    merge_press: bool  # one continuous motion instead of approach+stop+press
    merge_press_max_lateral_m: float  # xy limit for allowing the merge
    warp_fast_speed: float  # free-air scale of a warped descent
    warp_slow_frac: float  # final path fraction at contact speed
    grip_clear_m: float  # fingertip stop height above the lid plane
    grip_offset_xy: tuple  # base-frame grasp trim (bench-measured bias)
    detect_period_s: float  # detector tick; min_hits * this = commit floor
    detect_source: str  # 'vlm' | 'depth'
    vlm_backends: tuple  # ladder order, e.g. ('owl', 'claude')
    vlm_model: str
    owl_model: str
    owl_queries: tuple
    owl_min_score: float
    vlm_target: str  # what to ask Claude to box — plain English
    vlm_timeout_s: float
    vlm_pad_px: int


def load_press_demo(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    pd = raw["press_demo"]
    cfg = PressDemoCfg(
        travel_m=float(pd["travel_m"]),
        press_speed=float(pd["speed"]),
        staging_m=float(pd["staging_m"]),
        scan_xyz=tuple(raw["scan"]["xyz"]),
        timeout_s=float(raw["detect"]["timeout_s"]),
        min_hits=int(raw["detect"]["min_hits"]),
        tol_m=float(raw["detect"]["tol_m"]),
        window_s=float(raw["detect"]["window_s"]),
        fresh_s=float(raw["detect"]["fresh_s"]),
        grip_band=tuple(raw["open_box"]["grip_band"]),
        lift_m=float(raw["open_box"]["lift_m"]),
        lift_speed=float(raw["open_box"]["lift_speed"]),
        grip_speed=float(raw["open_box"].get("grip_speed", 0.15)),
        grip_hop_m=float(raw["open_box"].get("grip_hop_m", 0.12)),
        setdown_speed=float(raw["open_box"].get("setdown_speed", 0.15)),
        setdown_touch_nm=float(raw["open_box"].get("setdown_touch_nm", 4.0)),
        park_tool_down=bool(raw["open_box"].get("park_tool_down", False)),
        press_offset_xy=tuple(
            float(v) for v in raw["open_box"].get("press_offset_xy", [0.0, 0.0])
        ),
        merge_press=bool(raw["open_box"].get("merge_press", False)),
        merge_press_max_lateral_m=float(
            raw["open_box"].get("merge_press_max_lateral_m", 0.05)
        ),
        warp_fast_speed=float(raw["open_box"].get("warp_fast_speed", 0.0)),
        warp_slow_frac=float(raw["open_box"].get("warp_slow_frac", 0.3)),
        grip_clear_m=float(raw["open_box"]["grip_clear_m"]),
        grip_offset_xy=tuple(raw["open_box"]["grip_offset_xy"]),
        detect_period_s=float(raw["detect"].get("period_s", 0.15)),
        detect_source=str(raw["detect"].get("source", "depth")),
        vlm_backends=tuple(raw.get("vlm", {}).get("backends", ("claude",))),
        vlm_model=str(raw.get("vlm", {}).get("model", "claude-opus-5")),
        owl_model=str(
            raw.get("vlm", {}).get("owl_model", "google/owlv2-base-patch16-ensemble")
        ),
        owl_queries=tuple(
            raw.get("vlm", {}).get("owl_queries", ("a small white square box",))
        ),
        owl_min_score=float(raw.get("vlm", {}).get("owl_min_score", 0.18)),
        vlm_target=str(raw.get("vlm", {}).get("target", "the food-storage container")),
        vlm_timeout_s=float(raw.get("vlm", {}).get("timeout_s", 20.0)),
        vlm_pad_px=int(raw.get("vlm", {}).get("pad_px", 20)),
    )
    if not raw["open_box"]["grip_band"][0] < raw["open_box"]["grip_band"][1] < 0.8:
        raise ValueError(
            "open_box.grip_band must be (lo, hi) with hi < 0.8 — 0.8 is the "
            "closed-on-air feedback and can never mean 'holding the button'"
        )
    if len(cfg.press_offset_xy) != 2 or any(abs(v) > 0.02 for v in cfg.press_offset_xy):
        raise ValueError(
            "open_box.press_offset_xy must be two values within +/-0.02 m — a "
            "bigger trim means the fix or the mount is wrong, not the pads"
        )
    if not 0.0 <= cfg.grip_clear_m <= 0.02:
        raise ValueError(
            "open_box.grip_clear_m must be in [0, 0.02] m — 0 puts the "
            "fingertips ON the lid plane, more than 2 cm closes above the knob"
        )
    if any(abs(v) > 0.02 for v in cfg.grip_offset_xy):
        raise ValueError(
            "open_box.grip_offset_xy is a mm-scale bias trim, not an offset "
            "— |each| must be <= 0.02 m (re-measure the mount instead)"
        )
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
            "the camera's own scan footprint the box could not have been "
            "seen, and the merged solve plans in the reduced world"
        )
    if not 0.0 < cfg.warp_slow_frac < 1.0:
        raise ValueError("open_box.warp_slow_frac must be in (0, 1)")
    if cfg.detect_source not in ("depth", "vlm"):
        raise ValueError("detect.source must be 'depth' or 'vlm'")
    bad = set(cfg.vlm_backends) - {"owl", "claude"}
    if bad or (cfg.detect_source == "vlm" and not cfg.vlm_backends):
        raise ValueError("vlm.backends must be a non-empty subset of ['owl', 'claude']")
    if not 0.0 < cfg.owl_min_score < 1.0:
        raise ValueError("vlm.owl_min_score must be in (0, 1)")
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
