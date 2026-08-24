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


def test_press_demo_legs_compose_staging_press_retreat_home():
    import pytest

    from rammp_box_opening.models.container import from_container
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    legs = press_demo.build_demo_legs(c, cfg)
    seq = names(legs)
    assert seq == [
        "approach:staging",
        "press:close",
        "press:hover",
        "press:down",
        "retreat",
        "home",
    ]
    button = from_container(c.cpose, c.model.button_offset)
    staging = legs[0]
    assert staging.world.startswith("full") and staging.speed == 0.25
    assert staging.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    retreat = legs[4]
    assert retreat.world.startswith("interaction") and retreat.speed == 0.15
    # retreat returns to staging height from the press bottom
    assert retreat.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    assert legs[5].world.startswith("full")


def test_press_demo_scan_and_no_tag_home_use_bench_world():
    from rammp_box_opening.tasks import press_demo

    c = ctx()
    cfg = _demo_cfg()
    scan = press_demo.build_scan_leg(c, cfg, [0.0] * 7)
    assert scan.name == "scan" and scan.world == "bench"
    assert scan.kind is Kind.MOTION and scan.speed == 0.25
    assert scan.target[0] == "pose" and list(scan.target[1]) == list(cfg.scan_xyz)
    home = press_demo.build_home_leg(c, [0.0] * 7)
    assert home.name == "home" and home.world == "bench"
    assert home.target[0] == "joints"
