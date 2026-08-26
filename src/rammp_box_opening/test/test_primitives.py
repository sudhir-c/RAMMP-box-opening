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
from rammp_box_opening.runtime.legs import Kind, VerifyCtx

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"


def ctx():
    m = ContainerModel.load(CFG)
    return Ctx(
        model=m,
        cpose=ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0),
        client=FakeClient(),
        worlds=FakeStore(),
    )


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
    assert st.chain > 0  # contact broke the chain


def test_press_descent_verify_classifies():
    legs, _ = Press().plan(ctx(), state())
    descend = legs[1]
    ok, detail = descend.verify(VerifyCtx(outcome="touch", depth_m=0.008))
    assert ok
    ok, _ = descend.verify(VerifyCtx(outcome="arrived"))
    assert not ok  # bottomed out untripped


def test_grasp_trip_is_failure_and_band_checked():
    c = ctx()
    legs, _ = Grasp(c.model.lid_grasp, "grasp:lid").plan(c, state())
    descend = [leg for leg in legs if leg.kind is Kind.MOTION][0]
    close = [leg for leg in legs if leg.kind is Kind.GRIPPER][0]
    assert descend.guard.trip == "obstruction"
    ok, _ = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.6))
    assert ok  # inside expect_band
    ok, detail = close.verify(VerifyCtx(outcome="arrived", gripper_pos=0.8))
    assert not ok  # closed on air


def test_approach_targets_hover_not_contact():
    c = ctx()
    button = from_container(c.cpose, c.model.button_offset)
    hov = hover_above(button, c.model.hover_standoff)
    legs, _ = Approach(
        hov, attitude_quat(c.model.press_attitude_rpy_deg, 0.0), "approach:button"
    ).plan(c, state())
    assert len(legs) == 1 and legs[0].world.startswith("full")
    from rammp_box_opening.constants import TRANSIT_SPEED

    assert legs[0].speed == TRANSIT_SPEED
    assert legs[0].target[1][2] > button[2]  # hover, never contact depth


def test_chaining_start_joints_flow():
    c = ctx()
    st = state()
    legs_a, st = Approach([0.45, 0.0, 0.1], [0.5, 0.5, 0.5, 0.5], "approach:a").plan(
        c, st
    )
    legs_b, st = Home().plan(c, st)
    # same chain (no contact between them) and b planned from a's end
    assert legs_a[0].chain == legs_b[0].chain
    assert list(legs_b[0].traj.points[0].positions) == list(legs_a[0].goal_joints)


def test_place_sequence_and_release():
    import pytest

    from rammp_box_opening.primitives.core import (
        CARRY_CLEAR_M,
        SETDOWN_OVERDRIVE_M,
    )

    c = ctx()
    target_z = -0.07 + c.model.lid_dims[2]
    legs, _ = Place([0.45, -0.25, target_z], [0.5, 0.5, 0.5, 0.5]).plan(c, state())
    kinds = [leg.kind for leg in legs]
    assert kinds == [Kind.MOTION, Kind.MOTION, Kind.GRIPPER]
    transit, descend, open_ = legs
    assert descend.guard.trip == "setdown"
    assert descend.invalidates_downstream
    assert open_.gripper_cmd == 0.0
    # field 2026-08-26: the hover start state sat inside the aperture-ring
    # walls (INVALID_START_STATE) — the set-down world must be ring-free
    interaction = [kw for k, kw in c.worlds.pushes if k == "interaction"]
    assert interaction and interaction[-1]["ring"] is False
    # the carried lid hangs below the fingertips, invisible to the
    # planner: the transit hover must clear the container top by a
    # lid-height plus margin (field 2026-08-26: lid clipped the box line)
    carry_floor = c.cpose.xyz[2] + c.model.dims[2] + c.model.lid_dims[2] + CARRY_CLEAR_M
    assert transit.target[1][2] == pytest.approx(carry_floor)
    # success is the TOUCH: the stroke overdrives past nominal surface
    # contact so an exact-height 'arrived' can't slip through untripped
    assert descend.target[1][2] == pytest.approx(target_z - SETDOWN_OVERDRIVE_M)
    assert descend.guard.target_z == pytest.approx(target_z)


def test_lift_reverifies_band():
    c = ctx()
    legs, _ = Lift(0.10, band=c.model.lid_grasp.expect_band).plan(c, state())
    assert len(legs) == 1 and legs[0].verify is not None
    ok, _ = legs[0].verify(VerifyCtx(outcome="arrived", gripper_pos=0.6))
    assert ok
    ok, _ = legs[0].verify(VerifyCtx(outcome="arrived", gripper_pos=0.79))
    assert not ok  # slipped to fully closed
    ok, detail = legs[0].verify(VerifyCtx(outcome="arrived", gripper_pos=None))
    assert ok and "unchecked" in detail  # honest fallback, logged


def test_press_fixed_single_stroke_from_staging():
    import pytest

    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PressFixed

    c = ctx()
    cfg = load_press_demo(CFG)
    legs, st = PressFixed(cfg).plan(c, state())
    assert len(legs) == 1  # v2: no hover, no close (close rides the approach)
    press = legs[0]
    button = from_container(c.cpose, c.model.button_offset)
    assert press.kind is Kind.MOTION and press.world.startswith("interaction")
    assert press.guard is not None and press.guard.trip == "press"
    assert press.speed == pytest.approx(cfg.press_speed)
    assert press.target[1][2] == pytest.approx(button[2] - cfg.travel_m)
    assert press.invalidates_downstream and st.chain > 0


def test_press_fixed_verify_expected_depth_semantics():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PressFixed

    cfg = load_press_demo(CFG)
    legs, _ = PressFixed(cfg).plan(ctx(), state())
    press = legs[0]
    expected = cfg.staging_m / (cfg.staging_m + cfg.travel_m)
    # trip near the expected contact depth = pressed
    ok, detail = press.verify(VerifyCtx(outcome="touch", progress=expected))
    assert ok and "pressed" in detail
    # trip far ABOVE the button = struck something else, honest failure
    ok, detail = press.verify(VerifyCtx(outcome="touch", progress=0.3))
    assert not ok and "EARLY" in detail
    # full travel with no trip = pressed
    ok, detail = press.verify(VerifyCtx(outcome="arrived"))
    assert ok and "no trip" in detail
    ok, _ = press.verify(VerifyCtx(outcome="failed"))
    assert not ok


def test_worlds_are_pushed_at_plan_time():
    # spec §6: the planner must hold the leg's world BEFORE the plan is
    # requested — execution-time pushes alone mean every trajectory was
    # planned against the previous world (2026-08-24 review, critical)
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PressFixed

    c = ctx()
    Approach([0.45, 0.0, 0.3], attitude_quat([180.0, 0.0, 0.0], 0.0)).plan(c, state())
    assert c.client.worlds_pushed == ["full.yaml"]
    PressFixed(load_press_demo(CFG)).plan(c, state())
    assert c.client.worlds_pushed == ["full.yaml", "interaction_button.yaml"]
