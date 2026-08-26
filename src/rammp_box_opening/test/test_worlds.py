import pytest
import yaml

from rammp_box_opening.models.container import ContainerModel, ContainerPose
from rammp_box_opening.worlds import (
    PLANE_MARGIN_M,
    WorldStore,
    full_world,
    interaction_world,
    reduction_plane_z,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"
BENCH = "src/rammp_box_opening/config/world_bench.yaml"


def _bench():
    with open(BENCH) as f:
        return yaml.safe_load(f)


def _model_pose():
    m = ContainerModel.load(CFG)
    return m, ContainerPose(xyz=(0.45, 0.0, -0.07), yaw=0.0)


def _cuboids(world):
    return {o["name"]: o for o in world["obstacles"]}


def test_full_world_is_cuboids_only_and_err_tall():
    m, cp = _model_pose()
    w = full_world(_bench(), m, cp)
    for o in w["obstacles"]:
        assert set(o) == {"name", "position", "dims"}  # cuboids only (§2)
    c = _cuboids(w)["container"]
    assert c["dims"][2] > m.dims[2]  # err tall
    top = c["position"][2] + c["dims"][2] / 2
    assert top > cp.xyz[2] + m.dims[2]


def test_bench_obstacles_survive_in_all_variants():
    m, cp = _model_pose()
    names_full = set(_cuboids(full_world(_bench(), m, cp)))
    w = interaction_world(
        _bench(), m, cp, target_xyz=[0.45, 0.0, 0.09], contact_z=0.09, depth_max=0.012
    )
    names_int = set(_cuboids(w))
    assert {"pedestal", "table"} <= names_full
    assert {"pedestal", "table"} <= names_int


def test_reduction_plane_below_deepest_command():
    z = reduction_plane_z(contact_z=0.09, depth_max=0.012)
    assert z <= 0.09 - 0.012 - PLANE_MARGIN_M + 1e-9


def test_interaction_ring_leaves_corridor_but_blocks_lateral():
    m, cp = _model_pose()
    tx = [0.45, 0.0, 0.09]
    w = interaction_world(
        _bench(), m, cp, target_xyz=tx, contact_z=0.09, depth_max=0.012
    )
    cs = _cuboids(w)
    ring = [o for n, o in cs.items() if n.startswith("ring_")]
    assert len(ring) == 4
    for o in ring:  # corridor xy stays free
        dx = abs(o["position"][0] - tx[0]) - o["dims"][0] / 2
        dy = abs(o["position"][1] - tx[1]) - o["dims"][1] / 2
        assert max(dx, dy) >= 0.0  # ring outside corridor
    body = cs["container_body"]
    body_top = body["position"][2] + body["dims"][2] / 2
    assert body_top <= reduction_plane_z(0.09, 0.012) + 1e-9


def test_lid_cuboid_added_after_place():
    m, cp = _model_pose()
    lid_at = ContainerPose(xyz=(0.45, -0.25, -0.07), yaw=0.0)
    w = full_world(_bench(), m, cp, lid_at=lid_at)
    lid = _cuboids(w)["placed_lid"]
    assert lid["position"][:2] == pytest.approx([0.45, -0.25])


def test_store_idempotent_names(tmp_path):
    m, cp = _model_pose()
    store = WorldStore(BENCH, out_dir=tmp_path)
    n1, p1 = store.push_name("full", model=m, cpose=cp)
    n2, p2 = store.push_name("full", model=m, cpose=cp)
    assert n1 == n2 == "full" and p1 == p2
    assert p1.exists()


def test_bench_world_is_bench_only(tmp_path):
    from rammp_box_opening.worlds import bench_world

    w = bench_world(_bench())
    names = set(_cuboids(w))
    assert {"pedestal", "table"} <= names
    assert not any(n.startswith(("container", "ring", "placed_lid")) for n in names)
    store = WorldStore(BENCH, out_dir=tmp_path)
    n, p = store.push_name("bench")
    assert n == "bench" and p.exists()


def test_bench_world_keepout_band_when_container_unseen():
    from rammp_box_opening.worlds import ERR_TALL_M, bench_world

    m, _cp = _model_pose()
    w = bench_world(_bench(), unseen_model=m)
    band = _cuboids(w)["unseen_container_band"]
    top = band["position"][2] + band["dims"][2] / 2
    table = _cuboids(w)["table"]
    table_top = table["position"][2] + table["dims"][2] / 2
    assert top == pytest.approx(table_top + m.dims[2] + ERR_TALL_M)
    assert band["dims"][0] >= 0.5 and band["dims"][1] >= 0.9  # covers the band


def test_interaction_world_ring_optional():
    m, cp = _model_pose()
    w = interaction_world(
        _bench(),
        m,
        cp,
        target_xyz=[0.45, 0.0, 0.09],
        contact_z=0.09,
        depth_max=0.012,
        ring=False,
    )
    assert not any(n.startswith("ring_") for n in _cuboids(w))


def test_interaction_world_caps_bench_at_the_reduction_plane():
    """A set-down onto the bench must be PLANNABLE to its overdrive depth
    (field 2026-08-26: IK_FAIL 25 mm over the solid table); a
    button-height contact leaves the bench untouched."""
    m, cp = _model_pose()
    bench = _bench()
    table = next(o for o in bench["obstacles"] if o["name"] == "table")
    table_top = table["position"][2] + table["dims"][2] / 2
    contact = table_top + 0.03  # lid-height above the table
    w = interaction_world(bench, m, cp, [0.54, -0.27, contact], contact, 0.005)
    plane = reduction_plane_z(contact, 0.005)
    assert plane < table_top
    for o in w["obstacles"]:
        if o["name"].startswith(("container", "ring", "placed_lid")):
            continue
        assert o["position"][2] + o["dims"][2] / 2 <= plane + 1e-9
    # press-like: contact at button height leaves the bench untouched
    w = interaction_world(bench, m, cp, [0.45, 0.0, 0.09], 0.09, 0.015)
    tops = {o["name"]: o["position"][2] + o["dims"][2] / 2 for o in w["obstacles"]}
    assert tops["table"] == pytest.approx(table_top)
