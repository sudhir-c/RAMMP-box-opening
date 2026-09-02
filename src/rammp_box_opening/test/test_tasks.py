import math
from dataclasses import replace
from pathlib import Path

from test_runner import FakeClient, FakeStore

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.legs import Kind

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def ctx():
    return Ctx(
        model=ContainerModel.load(CFG),
        cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
        client=FakeClient(),
        worlds=FakeStore(),
        config_path=CFG,
    )


def names(legs):
    return [leg.name for leg in legs]


def test_entry_points_registered():
    setup = Path("src/rammp_box_opening/setup.py").read_text()
    for ep in ["press_demo", "home_arm", "preflight", "owl_detector", "joint_state_relay"]:
        assert ep + " = " in setup


def test_home_arm_plans_home_in_the_bench_world():
    """The isolated recovery home knows no container pose: it plans above
    the unseen-container band, like the mission's own recovery home."""
    from rammp_box_opening.constants import HOME, TRANSIT_SPEED
    from rammp_box_opening.tasks import home_arm

    c = ctx()
    c.cpose = None
    legs = home_arm.build_legs(c)
    assert names(legs) == ["home"]
    assert legs[0].world == "bench" and legs[0].speed == TRANSIT_SPEED
    assert legs[0].target == ("joints", list(HOME))


def _demo_cfg():
    from rammp_box_opening.models.container import load_press_demo

    return load_press_demo(CFG)


def test_press_demo_legs_compose_close_staging_press_retreat_home():
    import pytest

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_demo_legs(c, cfg)
    seq = names(legs)
    # press:close is no longer a leg: main() dispatches it at fix commit
    assert seq[:4] == ["approach:staging", "press:down", "retreat", "grip:open"]
    button = from_container(c.cpose, c.model.button_offset)
    staging = legs[0]
    assert staging.world.startswith("full") and staging.speed == TRANSIT_SPEED
    assert staging.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    retreat = legs[2]
    assert retreat.traj is None  # lazy: planned from live after the touch
    assert legs[3].defer_join  # the fingers open on arrival at the hop
    assert retreat.world.startswith("interaction")
    assert retreat.speed == pytest.approx(TRANSIT_SPEED)  # fast up
    # the OPEN-BOX composition re-descends straight away, so the retreat
    # stops at the hop height instead of climbing back to staging — a
    # 253 mm round trip for a 2 mm reposition (speed pass 2026-08-28)
    assert retreat.target[1][2] == pytest.approx(button[2] + cfg.grip_hop_m)
    assert cfg.grip_hop_m < cfg.staging_m


def test_press_demo_scan_and_no_tag_home_use_bench_world():
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    scan = press_demo.build_scan_leg(c, cfg, [0.0] * 7)
    assert scan.name == "scan" and scan.world == "bench"
    from rammp_box_opening.constants import TRANSIT_SPEED

    assert scan.kind is Kind.MOTION and scan.speed == TRANSIT_SPEED
    assert scan.target[0] == "pose" and list(scan.target[1]) == list(cfg.scan_xyz)
    home = press_demo.build_home_leg(c, [0.0] * 7)
    assert home.name == "home" and home.world == "bench"
    assert home.target[0] == "joints"


def test_open_box_grip_and_place_legs():
    import pytest

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_grip_legs(c, cfg)
    assert names(legs) == ["grip:down", "grip:close", "lift"]
    down, close, lift = legs
    assert close.gripper_cmd == 0.8
    button = from_container(c.cpose, c.model.button_offset)
    # ABOVE the tag/lid plane (press depth = into the lid — field
    # 2026-08-26), with the bench-measured lateral trim applied
    assert down.target[1][2] == pytest.approx(button[2] + cfg.grip_clear_m)
    assert down.target[1][0] == pytest.approx(button[0] + cfg.grip_offset_xy[0])
    assert down.target[1][1] == pytest.approx(button[1] + cfg.grip_offset_xy[1])
    assert down.guard is not None and down.guard.trip == "obstruction"
    assert down.invalidates_downstream
    # closed-on-air (0.8) fails the band; holding the knob passes
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.8))
    assert not ok
    held = (cfg.grip_band[0] + cfg.grip_band[1]) / 2
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=held))
    assert ok
    # bench 2026-08-26: the real knob reads 0.387 — the band must hold it
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.387))
    assert ok
    assert lift.speed == pytest.approx(cfg.lift_speed)  # "slowly lift"
    assert lift.target[1][2] == pytest.approx(button[2] + cfg.grip_clear_m + cfg.lift_m)

    place_legs = press_demo.build_place_legs(c, cfg)
    assert names(place_legs) == [
        "place:lid:transit",
        "place:lid:down",
        "place:lid:open",
        "retreat",
        "home",
    ]
    from rammp_box_opening.primitives.core import CARRY_CLEAR_M

    # carry clears the container even directly overhead (held lid unmodeled)
    carry_floor = c.cpose.xyz[2] + c.model.dims[2] + c.model.lid_dims[2] + CARRY_CLEAR_M
    assert place_legs[0].target[1][2] >= carry_floor - 1e-9
    assert place_legs[1].guard.trip == "setdown"
    assert place_legs[2].gripper_cmd == 0.0
    # a release: dispatched at once, joined before the retreat MOVES
    assert place_legs[2].defer_join and place_legs[2].join_before_motion
    assert c.lid_at is not None  # the placed lid joins later worlds
    assert "lid" in place_legs[4].world  # home plans around the placed lid
    assert place_legs[3].speed == TRANSIT_SPEED
    # both lazy, one chain: planned from live as one group after the touch
    assert place_legs[3].traj is None and place_legs[4].traj is None
    assert place_legs[3].chain == place_legs[4].chain
    # the retreat climbs to the CARRY height, where home is plannable
    assert place_legs[3].target[1][2] == pytest.approx(place_legs[0].target[1][2])


def test_press_demo_full_composition():
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    legs = press_demo.build_demo_legs(c, _demo_cfg())
    seq = names(legs)
    assert seq[:4] == ["approach:staging", "press:down", "retreat", "grip:open"]
    assert seq[4:7] == ["grip:down", "grip:close", "lift"]
    assert seq[7:] == [
        "place:lid:transit",
        "place:lid:down",
        "place:lid:open",
        "retreat",
        "home",
    ]


def test_lid_place_clearance_gate_threshold():
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    need = press_demo.lid_place_min_clear(c.model)
    # both footprint half-diagonals + gripper-body room; the field runs
    # bracket it: IK_FAIL at 0.073 m separation, clean plan at 0.162 m
    assert 0.073 < need < 0.162
    import math

    assert need == pytest.approx(
        (math.hypot(*c.model.dims[:2]) + math.hypot(*c.model.lid_dims[:2])) / 2 + 0.05
    )


def test_resolve_lid_drop_adapts_to_the_box():
    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    m = c.model
    need = press_demo.lid_place_min_clear(m)
    lid = [0.45, -0.25, -0.027]
    far = ContainerPose(xyz=(0.45, 0.1, -0.02), yaw=0.0)
    xyz, shifted = press_demo.resolve_lid_drop(m, far, lid)
    assert xyz == lid and not shifted
    # field 2026-08-26: box 55 mm from the spot — slide, don't refuse
    close = ContainerPose(xyz=(0.396, -0.24, -0.02), yaw=0.0)
    xyz, shifted = press_demo.resolve_lid_drop(m, close, lid)
    assert shifted
    assert math.hypot(xyz[0] - 0.396, xyz[1] + 0.24) >= need - 1e-9
    assert press_demo.DROP_X_M[0] <= xyz[0] <= press_demo.DROP_X_M[1]
    assert press_demo.DROP_Y_M[0] <= xyz[1] <= press_demo.DROP_Y_M[1]
    assert xyz[2] == lid[2]  # table height never changes
    # box ON the spot: the direction degenerates, a fallback still clears
    on_top = ContainerPose(xyz=(0.45, -0.25, -0.02), yaw=0.0)
    xyz, shifted = press_demo.resolve_lid_drop(m, on_top, lid)
    assert shifted and math.hypot(xyz[0] - 0.45, xyz[1] + 0.25) >= need - 1e-9


def test_place_legs_use_the_resolved_drop_spot():
    import pytest

    from rammp_box_opening.models.container import ContainerPose
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    c.lid_drop = ContainerPose(xyz=(0.30, 0.30, -0.027), yaw=0.0)
    legs = press_demo.build_place_legs(c, _demo_cfg())
    transit = legs[0]
    assert transit.target[1][0] == pytest.approx(0.30)
    assert transit.target[1][1] == pytest.approx(0.30)
    assert c.lid_at is c.lid_drop  # the placed-lid world follows the shift


def test_contact_leg_speeds_come_from_config():
    """grip:down, lift and the set-down each read their own config knob.

    They were three hardcoded 0.15s; the set-down and lift now run at
    0.35 while grip:down stays conservative until the bench ramp."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    grip = press_demo.build_grip_legs(c, cfg)
    down = next(x for x in grip if x.name == "grip:down")
    lift = next(x for x in grip if x.name == "lift")
    # grip:down is time-warped, so its contact scale lives in leg.warp and
    # leg.speed is 1.0 (the profile is already baked into the timing)
    assert down.warp[1] == pytest.approx(cfg.grip_speed)
    assert lift.speed == pytest.approx(cfg.lift_speed)

    c.lid_drop = None
    place = press_demo.build_place_legs(c, cfg)
    setdown = next(x for x in place if x.name == "place:lid:down")
    assert setdown.speed == pytest.approx(cfg.setdown_speed)
    # the set-down keeps its guard: speed rose, the trip=success did not move
    assert setdown.guard is not None and setdown.guard.trip == "setdown"


def test_shipped_config_speeds_are_guard_safe():
    """The shipped values are what actually runs at the bench."""
    import pytest

    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    assert cfg.lift_speed == pytest.approx(0.35)
    assert cfg.setdown_speed == pytest.approx(0.35)
    assert cfg.grip_speed == pytest.approx(0.15)  # ramp at the bench first
    assert cfg.detect_period_s == pytest.approx(0.05)
    # every contact/carry speed stays inside the guard-limited band
    for v in (cfg.grip_speed, cfg.setdown_speed, cfg.lift_speed):
        assert 0.0 < v <= 0.5


def test_guarded_descents_are_time_warped_and_rebaseline_the_guard():
    """grip:down and the set-down run fast through free air and slow into
    contact, and the guard re-baselines where the speed changes."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    grip = press_demo.build_grip_legs(c, cfg)
    down = next(x for x in grip if x.name == "grip:down")
    assert down.warp == (cfg.warp_fast_speed, cfg.grip_speed, cfg.warp_slow_frac)
    # the profile is baked into the timing, so it must NOT be dilated again
    assert down.speed == pytest.approx(1.0)
    # ...and the guard is told where the regime changes
    assert down.guard is not None and down.guard.rebaseline_after is not None
    assert 0.0 < down.guard.rebaseline_after <= 1.0
    assert down.guard.trip == "obstruction"  # semantics unchanged

    # the set-down goes through the same _apply_warp; the fake planner
    # returns a zero-length descent for it, so exercise the hook directly
    # on a trajectory that actually moves
    from rammp_box_opening.runtime.guards import GuardSpec

    moving = next(x for x in grip if x.name == "grip:down")
    probe = replace(moving, warp=None, speed=cfg.setdown_speed)
    probe.guard = GuardSpec(touch_nm=6.0, trip="setdown", target_z=0.0)
    press_demo._apply_warp(probe, cfg, cfg.setdown_speed)
    assert probe.warp == (cfg.warp_fast_speed, cfg.setdown_speed, cfg.warp_slow_frac)
    assert probe.guard.trip == "setdown" and probe.guard.rebaseline_after is not None


def test_warping_is_off_when_the_config_disables_it():
    from dataclasses import replace as _replace

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _replace(_demo_cfg(), warp_fast_speed=0.0)
    down = next(x for x in press_demo.build_grip_legs(c, cfg) if x.name == "grip:down")
    assert down.warp is None
    assert down.speed == cfg.grip_speed  # plain single-scale behaviour
    assert down.guard.rebaseline_after is None


def test_merged_press_is_one_continuous_motion_with_no_staging_stop():
    """[transit to staging] STOP [press] becomes close + ONE descent."""
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    # the servo leaves the arm above the tag: directly over the button
    c.last_pose = ([button[0], button[1], button[2] + 0.35], [0.0, 1.0, 0.0, 0.0])
    ok, lateral = press_demo.merged_press_ok(c, cfg)
    assert ok and lateral == pytest.approx(0.0, abs=1e-9)

    legs = press_demo.build_merged_press_legs(c, cfg)
    assert names(legs) == ["press:down", "retreat", "grip:open"]
    assert "approach:staging" not in names(legs)  # the stop is gone
    press, retreat, open_leg = legs
    assert retreat.traj is None  # lazy: planned from live after the touch
    assert open_leg.defer_join and open_leg.gripper_cmd == 0.0  # opens at the hop
    assert press.guard is not None and press.guard.trip == "press"
    assert press.target[1][2] == pytest.approx(button[2] - cfg.travel_m)
    # continuous by construction: ONE solve, then warped fast-into-slow
    assert press.warp is not None and press.guard.rebaseline_after is not None
    assert retreat.target[1][2] == pytest.approx(button[2] + cfg.grip_hop_m)


def test_merged_press_is_declined_when_the_arm_is_off_axis():
    """The merged solve plans in the REDUCED world, so a long lateral run
    through it — where the container is invisible — must not happen."""
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    c.last_pose = (
        [button[0] + 0.30, button[1], button[2] + 0.35],
        [0.0, 1.0, 0.0, 0.0],
    )
    ok, lateral = press_demo.merged_press_ok(c, cfg)
    assert not ok and lateral > cfg.merge_press_max_lateral_m

    c.last_pose = None  # nothing commanded yet
    assert press_demo.merged_press_ok(c, cfg) == (False, None)


def test_merged_press_off_by_config_uses_the_staged_path():
    from dataclasses import replace as _replace

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _replace(_demo_cfg(), merge_press=False)
    button = from_container(c.cpose, c.model.button_offset)
    c.last_pose = ([button[0], button[1], button[2] + 0.35], [0.0, 1.0, 0.0, 0.0])
    assert press_demo.merged_press_ok(c, cfg) == (False, None)


def test_merged_press_press_only_keeps_full_retreat_and_home():
    """press-only may merge too — but its retreat must climb back to
    staging height (home is planned in the FULL world) and home follows."""
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    c.last_pose = ([button[0], button[1], button[2] + 0.35], [0.0, 1.0, 0.0, 0.0])
    legs = press_demo.build_merged_press_legs(c, cfg, include_home=True)
    assert names(legs) == ["press:down", "retreat", "home"]
    retreat = legs[1]
    assert retreat.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    assert legs[2].world.startswith("full")
    assert legs[2].traj is None  # lazy, chained after the lazy retreat


def test_merged_press_accepts_a_realistic_off_axis_box():
    """Field 2026-09-01: a real box sat 152 mm from the scan axis and the
    old 50 mm rail declined the merge — the pause the owner asked to
    remove. 0.20 admits it; the descent converges over the button."""
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    assert cfg.merge_press_max_lateral_m >= 0.20
    button = from_container(c.cpose, c.model.button_offset)
    # scan pose 152 mm off the button, like the field run
    c.last_pose = (
        [button[0] - 0.038, button[1] + 0.147, 0.45],
        [0.0, 1.0, 0.0, 0.0],
    )
    ok, lateral = press_demo.merged_press_ok(c, cfg)
    assert ok and 0.14 < lateral < 0.16


def test_merged_press_constrains_the_final_approach_vertical():
    """A diagonal descent touches the button before lateral convergence
    finishes (edge presses, 2026-09-01) — the merged press target carries
    a 60 mm vertical-final constraint, and replans preserve it."""
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    button = from_container(c.cpose, c.model.button_offset)
    c.last_pose = ([button[0], button[1], button[2] + 0.35], [0.0, 1.0, 0.0, 0.0])
    legs = press_demo.build_merged_press_legs(c, cfg)
    press = next(x for x in legs if x.name == "press:down")
    assert len(press.target) == 4
    assert press.target[3] == pytest.approx(0.06)


def test_every_descent_carries_a_vertical_final_constraint():
    """The staged press arched into the button edge exactly like the
    merged press did before it got the grasp-approach constraint (field
    2026-09-01) — and a bowed grip or set-down misses the same way. Every
    contact-bound descent now ends vertical; the constraint rides in the
    target tuple so drift replans preserve it too."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_demo_legs(c, cfg)
    want = {"press:down": 0.06, "grip:down": 0.04, "place:lid:down": 0.05}
    for name, off in want.items():
        leg = next(x for x in legs if x.name == name)
        assert len(leg.target) == 4, name
        assert leg.target[3] == pytest.approx(off), name


def test_post_touch_legs_are_lazy_and_home_needs_no_fallback():
    """Legs after an expected touch used to be pre-planned and then thrown
    away by the post-touch replan every run; home was even planned twice
    (retreat-end refused, transit-end fallback) and then failed live
    anyway. Now they are LAZY — no plan call at build time — and the
    Runner plans them once, from live, as one group (audit 2026-09-02)."""
    from rammp_box_opening.tasks import press_demo

    class CountingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.joint_plans = 0

        def plan_to_joints(self, q7, start_joints):
            self.joint_plans += 1
            return super().plan_to_joints(q7, start_joints)

    c = ctx()
    c.client = CountingClient()
    legs = press_demo.build_place_legs(c, _demo_cfg())
    assert names(legs)[-2:] == ["retreat", "home"]
    assert c.client.joint_plans == 0  # home is not planned at build time
    assert all(x.traj is None for x in legs[-2:])
    # build-time plans: transit + set-down only
    assert len(c.client.approach_offsets) == 2


def test_place_accounts_for_grip_height_and_gentle_touch():
    """The fingers hold the knob grip_clear_m above the lid plane, so lid
    contact happens with the TOOL that much above lid-top height — the
    uncompensated target over-travelled by grip_clear_m and crunched the
    lid into the table without tripping the press-strength guard (field
    2026-09-02). The set-down also gets its own gentler threshold."""
    import pytest

    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_place_legs(c, cfg)
    down = next(x for x in legs if x.name == "place:lid:down")
    from rammp_box_opening.primitives.core import SETDOWN_OVERDRIVE_M
    from rammp_box_opening.tasks.press_demo import load_lid_place

    lid = load_lid_place(CFG)
    want = lid.xyz[2] + c.model.lid_dims[2] + cfg.grip_clear_m - SETDOWN_OVERDRIVE_M
    assert down.target[1][2] == pytest.approx(want)
    assert down.guard.touch_nm == pytest.approx(cfg.setdown_touch_nm)
    assert cfg.setdown_touch_nm < c.model.touch_nm  # gentler than the press


def test_setdown_verify_rejects_early_trips_and_no_touch():
    """A trip in the first half of the stroke is a strike, not a set-down
    (the lid was dropped from 110 mm when a fast-segment trip counted as
    touch, field 2026-09-02); arriving without ever feeling the surface
    stays a failure. Only a late trip confirms the set-down."""
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_place_legs(c, cfg)
    down = next(x for x in legs if x.name == "place:lid:down")
    assert down.guard.arm_after is not None and down.guard.arm_after >= 0.5

    def v(outcome, progress):
        return down.verify(
            VerifyCtx(outcome=outcome, progress=progress, torque_peak=4.2)
        )

    ok, why = v("touch", 0.06)
    assert not ok and "NOT confirmed" in why
    ok, why = v("touch", 0.9)
    assert ok
    ok, why = v("arrived", 1.0)
    assert not ok and "never felt the surface" in why


def test_park_tool_down_rests_at_the_scan_pose():
    """open_box.park_tool_down: the mission ends at PARK (tool-down at the
    scan pose) instead of the factory HOME, saving the 2.4-2.9 rad wrist
    flip twice per run; off by default because the arm then rests over
    the bench (audit 2026-09-02)."""
    import math

    from rammp_box_opening.constants import HOME, PARK, REST_TOL_RAD
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    assert cfg.park_tool_down is False
    assert press_demo.rest_joints(cfg) == list(HOME)
    on = replace(cfg, park_tool_down=True)
    assert press_demo.rest_joints(on) == list(PARK)
    legs = press_demo.build_place_legs(c, on)
    assert legs[-1].name == "home" and legs[-1].target == ("joints", list(PARK))
    # "already parked" is judged against the server's own start gate
    assert press_demo.rest_distance(PARK, PARK) == 0.0
    nudged = list(PARK)
    nudged[3] += REST_TOL_RAD * 2
    assert press_demo.rest_distance(nudged, PARK) > REST_TOL_RAD
    assert press_demo.rest_distance([math.pi] + PARK[1:], PARK) > 1.0
