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
    assert seq == [
        "press:close",
        "approach:staging",
        "press:down",
        "retreat",
        "home",
    ]
    button = from_container(c.cpose, c.model.button_offset)
    staging = legs[1]
    assert staging.world.startswith("full") and staging.speed == TRANSIT_SPEED
    assert staging.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    retreat = legs[3]
    assert retreat.world.startswith("interaction")
    assert retreat.speed == pytest.approx(TRANSIT_SPEED)  # fast up
    # retreat returns to staging height from the press bottom
    assert retreat.target[1][2] == pytest.approx(button[2] + cfg.staging_m)
    assert legs[4].world.startswith("full")


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
    cfg = load_press_demo(CFG)
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
    cfg = load_press_demo(CFG)
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
    cfg = load_press_demo(CFG)
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
    cfg = load_press_demo(CFG)
    # a mirrored camera frame: the error GROWS after the first move
    watcher, wait = _servo_fixture([[0.10, 0.0, 0.4], [0.22, 0.0, 0.4]])
    runner = _FakeRunner()
    got, why = center_on_tag(None, watcher, c, cfg, runner, True, wait=wait)
    assert got is None and why == "servo_failed"
    assert len(runner.ran) == 1  # stopped after one move, no walking away
