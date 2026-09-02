from test_runner import FakeClient, FakeStore

from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    from_container,
)
from rammp_box_opening.primitives.core import (
    Ctx,
    Home,
    Lift,
    Place,
    PlanState,
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
    return PlanState(joints=[0.0] * 7, chain=0)


def test_chaining_start_joints_flow():
    c = ctx()
    st = state()
    legs_a, st = Lift(0.10).plan(c, st)
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
    assert open_.chain == descend.chain + 1  # contact breaks the chain
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
    legs, _ = Lift(0.10, band=(0.55, 0.75)).plan(c, state())
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
    assert st.chain == press.chain + 1  # contact breaks the chain


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
    Home().plan(c, state())
    assert c.client.worlds_pushed == ["full.yaml"]
    PressFixed(load_press_demo(CFG)).plan(c, state())
    assert c.client.worlds_pushed == ["full.yaml", "interaction_button.yaml"]


def test_contact_sets_the_container_pad_for_later_full_worlds():
    """Any guarded plan marks the mission contact-tainted: every later
    FULL world allows for a scooted container."""
    from rammp_box_opening.primitives.core import (
        CONTACT_SHIFT_PAD_M,
        PressFixed,
        _full_world,
    )

    c, st = ctx(), state()
    assert c.contact_pad == 0.0
    _full_world(c)
    assert c.worlds.pushes[-1][1].get("container_pad_xy", 0.0) == 0.0
    from rammp_box_opening.models.container import load_press_demo

    cfg = load_press_demo(CFG)
    PressFixed(cfg).plan(c, st)
    assert c.contact_pad == CONTACT_SHIFT_PAD_M
    _full_world(c)
    assert c.worlds.pushes[-1][1]["container_pad_xy"] == CONTACT_SHIFT_PAD_M
