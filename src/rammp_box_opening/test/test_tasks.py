import math
from dataclasses import replace
from pathlib import Path

from test_runner import FakeClient, FakeStore

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.legs import Kind
from rammp_box_opening.tasks import open_container, pickup_container

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


def test_open_container_sequence():
    legs = open_container.build_legs(ctx())
    seq = names(legs)
    assert seq[0].startswith("approach:button")
    for prefix in [
        "press",
        "retreat",
        "approach:lid",
        "grasp:lid",
        "lift",
        "place:lid",
        "home",
    ]:
        assert any(n.startswith(prefix) for n in seq), prefix
    # ordering: press before grasp, grasp before lift, lift before place
    idx = {
        p: min(i for i, n in enumerate(seq) if n.startswith(p))
        for p in ["press", "grasp:lid", "lift", "place:lid"]
    }
    assert idx["press"] < idx["grasp:lid"] < idx["lift"] < idx["place:lid"]
    assert seq[-1] == "home"


def test_open_container_worlds_carry_lid_after_place():
    legs = open_container.build_legs(ctx())
    place_i = max(i for i, leg in enumerate(legs) if leg.name.startswith("place:lid"))
    after = [leg for leg in legs[place_i + 1 :] if leg.kind is Kind.MOTION]
    assert after, "home leg expected after place"
    assert all(
        "lid" in leg.world for leg in after
    ), "post-place worlds must include the placed-lid cuboid"


def test_pickup_places_back_by_default_and_holds_on_request():
    default = names(pickup_container.build_legs(ctx()))
    assert any(n.startswith("place:container") for n in default)
    held = names(pickup_container.build_legs(ctx(), hold=True))
    assert not any(n.startswith("place:") for n in held)
    assert held[-1] == "home"


def test_entry_points_registered():
    setup = Path("src/rammp_box_opening/setup.py").read_text()
    for ep in [
        "open_container",
        "pickup_container",
        "smoke_plan",
        "preflight",
        "press",
        "grasp",
        "home_arm",
    ]:
        assert ep + " = " in setup


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
    assert seq[:4] == ["press:close", "approach:staging", "press:down", "retreat"]
    button = from_container(c.cpose, c.model.button_offset)
    staging = legs[1]
    assert staging.world.startswith("full") and staging.speed == TRANSIT_SPEED
    assert staging.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    retreat = legs[3]
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


class _FakeWatcher:
    def __init__(self):
        import numpy as np

        self.k = None
        self.grab = type("G", (), {})()
        self.grab.k = np.array(
            [[430.0, 0.0, 424.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]]
        )
        self.last_debug = None


class _FakeRunner:
    def __init__(self, ok=True):
        self.ok = ok
        self.ran = []

    def run(self, legs, execute, assume_yes=False):
        self.ran.append([leg.name for leg in legs])

        class R:
            pass

        out = []
        for leg in legs:
            r = R()
            r.ok = self.ok
            r.leg_name = leg.name
            out.append(r)
        return out


def _servo_fixture(p_cams):
    """Scripted wait: each call yields a fix whose camera-frame position is
    the next entry (rot/trans fixed, tool-down at yaw 0)."""
    import numpy as np

    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    t = np.array([0.42, -0.075, 0.575])
    watcher = _FakeWatcher()
    seq = list(p_cams)

    def wait(_node, w, _cfg, timeout_s=None):
        if not seq:
            return None
        p = seq.pop(0)
        if p is None:
            return None
        p = np.asarray(p, float)
        watcher.last_debug = (p, rot, t)
        return rot @ p + t, np.eye(3)

    return watcher, wait


def test_center_on_tag_converges_after_one_move():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.tasks.press_demo import center_on_tag

    c = ctx()
    c.last_pose = ([0.42, 0.0, 0.45], [0.0, 1.0, 0.0, 0.0])
    # ships with servo.max_iters 0 (detect once, press); these tests
    # cover the loop mechanics, so they pin their own budget
    cfg = replace(load_press_demo(CFG), servo_max_iters=4)
    watcher, wait = _servo_fixture([[0.10, 0.0, 0.4], [0.001, 0.0, 0.4]])
    runner = _FakeRunner()
    got, why = center_on_tag(None, watcher, c, cfg, runner, True, wait=wait)
    assert why == "ok" and got is not None
    assert runner.ran == [["servo:1"]]


def test_center_on_tag_unconverged_presses_on_freshest_fix():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.tasks.press_demo import center_on_tag

    c = ctx()
    c.last_pose = ([0.42, 0.0, 0.45], [0.0, 1.0, 0.0, 0.0])
    # ships with servo.max_iters 0 (detect once, press); these tests
    # cover the loop mechanics, so they pin their own budget
    cfg = replace(load_press_demo(CFG), servo_max_iters=4)
    watcher, wait = _servo_fixture([[0.10, 0.0, 0.4]] * (cfg.servo_max_iters + 1))
    runner = _FakeRunner()
    got, why = center_on_tag(None, watcher, c, cfg, runner, True, wait=wait)
    assert why == "ok" and got is not None
    assert len(runner.ran) == cfg.servo_max_iters


def test_center_on_tag_failure_semantics():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.tasks.press_demo import center_on_tag

    c = ctx()
    c.last_pose = ([0.42, 0.0, 0.45], [0.0, 1.0, 0.0, 0.0])
    # ships with servo.max_iters 0 (detect once, press); these tests
    # cover the loop mechanics, so they pin their own budget
    cfg = replace(load_press_demo(CFG), servo_max_iters=4)
    # servo exec failure: holds, never homes
    watcher, wait = _servo_fixture([[0.10, 0.0, 0.4], [0.10, 0.0, 0.4]])
    runner = _FakeRunner(ok=False)
    got, why = center_on_tag(None, watcher, c, cfg, runner, True, wait=wait)
    assert got is None and why == "servo_failed"
    # benign timeout: no_tag
    watcher, wait = _servo_fixture([None])
    got, why = center_on_tag(None, watcher, c, cfg, _FakeRunner(), True, wait=wait)
    assert got is None and why == "no_tag"


def test_center_on_tag_divergence_stops():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.tasks.press_demo import center_on_tag

    c = ctx()
    c.last_pose = ([0.42, 0.0, 0.45], [0.0, 1.0, 0.0, 0.0])
    # ships with servo.max_iters 0 (detect once, press); these tests
    # cover the loop mechanics, so they pin their own budget
    cfg = replace(load_press_demo(CFG), servo_max_iters=4)
    # a mirrored camera frame: the error GROWS after the first move
    watcher, wait = _servo_fixture([[0.10, 0.0, 0.4], [0.22, 0.0, 0.4]])
    runner = _FakeRunner()
    got, why = center_on_tag(None, watcher, c, cfg, runner, True, wait=wait)
    assert got is None and why == "servo_failed"
    assert len(runner.ran) == 1  # stopped after one move, no walking away


def test_open_box_grip_and_place_legs():
    import pytest

    from rammp_box_opening.constants import TRANSIT_SPEED
    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.runtime.legs import VerifyCtx
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_grip_legs(c, cfg)
    assert names(legs) == ["grip:open", "grip:down", "grip:close", "lift"]
    open_leg, down, close, lift = legs
    assert open_leg.gripper_cmd == 0.0 and close.gripper_cmd == 0.8
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
    assert c.lid_at is not None  # the placed lid joins later worlds
    assert "lid" in place_legs[4].world  # home plans around the placed lid
    assert place_legs[3].speed == TRANSIT_SPEED


def test_press_demo_full_composition():
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    legs = press_demo.build_demo_legs(c, _demo_cfg())
    seq = names(legs)
    assert seq[:4] == ["press:close", "approach:staging", "press:down", "retreat"]
    assert seq[4:8] == ["grip:open", "grip:down", "grip:close", "lift"]
    assert seq[8:] == [
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
    assert cfg.servo_max_iters == 0  # detect once, press on that fix
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
    assert names(legs) == ["press:close", "press:down", "retreat"]
    assert "approach:staging" not in names(legs)  # the stop is gone
    close, press, retreat = legs
    assert close.defer_join  # fingers shut while the arm is already moving
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
