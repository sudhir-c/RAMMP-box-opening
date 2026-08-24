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
    assert legs[0].speed == 0.25
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
    c = ctx()
    legs, _ = Place(
        [0.45, -0.25, -0.07 + c.model.lid_dims[2]], [0.5, 0.5, 0.5, 0.5]
    ).plan(c, state())
    kinds = [leg.kind for leg in legs]
    assert kinds == [Kind.MOTION, Kind.MOTION, Kind.GRIPPER]
    transit, descend, open_ = legs
    assert descend.guard.trip == "setdown"
    assert descend.invalidates_downstream
    assert open_.gripper_cmd == 0.0


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


def test_press_fixed_legs_speeds_and_targets():
    import pytest

    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PressFixed

    c = ctx()
    cfg = load_press_demo(CFG)
    legs, st = PressFixed(cfg).plan(c, state())
    close, hover, press = legs
    assert close.kind is Kind.GRIPPER and close.gripper_cmd == 0.8
    button = from_container(c.cpose, c.model.button_offset)
    assert hover.kind is Kind.MOTION and hover.speed == 0.15
    assert hover.world.startswith("interaction")
    assert hover.target[1][2] == pytest.approx(button[2] + cfg.hover_m)
    assert press.guard is not None and press.guard.trip == "press"
    assert press.speed == pytest.approx(cfg.press_speed)
    assert press.target[1][2] == pytest.approx(button[2] - cfg.travel_m)
    assert press.invalidates_downstream and st.chain > 0


def test_press_fixed_verify_trip_or_full_travel_both_pass():
    from rammp_box_opening.models.container import load_press_demo
    from rammp_box_opening.primitives.core import PressFixed

    legs, _ = PressFixed(load_press_demo(CFG)).plan(ctx(), state())
    press = legs[2]
    ok, detail = press.verify(VerifyCtx(outcome="touch", torque_peak=4.2))
    assert ok and "guard" in detail
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
