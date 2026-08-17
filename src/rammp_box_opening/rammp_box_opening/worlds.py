"""Collision-world generation: bench + container-derived cuboid variants.

Worlds are a plan-time concern (spec §6). SetWorld is write-only, so the
Runner tracks what it last pushed; this module only builds and writes the
variants. Cuboids only — v0.7.8 drops other shapes.
"""

from pathlib import Path

import yaml

ERR_TALL_M = 0.02  # container cuboid extra height (err tall, spec §3)
APERTURE_HALF_M = 0.06  # half-extent of the free descent corridor
RING_THICK_M = 0.05  # aperture ring wall thickness
RING_HEIGHT_M = 0.25  # ring wall height above the reduction plane
PLANE_MARGIN_M = 0.03  # reduction plane below deepest command (≥ calib margin)


def _bench_obstacles(bench):
    return [dict(o) for o in bench["obstacles"]]


def _container_cuboid(model, cpose, name="container", top_z=None):
    dx, dy, dz = model.dims
    top = (cpose.xyz[2] + dz + ERR_TALL_M) if top_z is None else top_z
    height = top - cpose.xyz[2]
    return {
        "name": name,
        "position": [cpose.xyz[0], cpose.xyz[1], cpose.xyz[2] + height / 2],
        "dims": [dx, dy, height],
    }


def _lid_cuboid(model, lid_at):
    lx, ly, lz = model.lid_dims
    return {
        "name": "placed_lid",
        "position": [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + lz / 2],
        "dims": [lx, ly, lz],
    }


def full_world(bench, model, cpose, lid_at=None):
    obstacles = _bench_obstacles(bench)
    obstacles.append(_container_cuboid(model, cpose))
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
    }


def reduction_plane_z(contact_z, depth_max):
    """The interaction world must let the planner plan FROM the deepest
    commanded contact point (spec §6): plane ≥ margin below it."""
    return contact_z - depth_max - PLANE_MARGIN_M


def interaction_world(bench, model, cpose, target_xyz, contact_z, depth_max,
                      lid_at=None):
    obstacles = _bench_obstacles(bench)
    plane = reduction_plane_z(contact_z, depth_max)
    if plane > cpose.xyz[2]:  # body below the plane stays solid
        obstacles.append(
            _container_cuboid(model, cpose, name="container_body", top_z=plane)
        )
    # Aperture ring: lateral entry forbidden, vertical corridor free.
    tx, ty = target_xyz[0], target_xyz[1]
    a, w, h = APERTURE_HALF_M, RING_THICK_M, RING_HEIGHT_M
    zc = plane + h / 2
    span = 2 * (a + w)
    obstacles += [
        {"name": "ring_xp", "position": [tx + a + w / 2, ty, zc], "dims": [w, span, h]},
        {"name": "ring_xn", "position": [tx - a - w / 2, ty, zc], "dims": [w, span, h]},
        {"name": "ring_yp", "position": [tx, ty + a + w / 2, zc], "dims": [span, w, h]},
        {"name": "ring_yn", "position": [tx, ty - a - w / 2, zc], "dims": [span, w, h]},
    ]
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
    }


class WorldStore:
    def __init__(self, bench_yaml_path, out_dir=None):
        with open(bench_yaml_path) as f:
            self._bench = yaml.safe_load(f)
        self._dir = Path(
            out_dir
            if out_dir is not None
            else Path.home() / ".ros" / "rammp_box_opening" / "worlds"
        )
        self._dir.mkdir(parents=True, exist_ok=True)

    def push_name(self, kind, model=None, cpose=None, target_xyz=None,
                  contact_z=None, depth_max=None, lid_at=None, tag=""):
        if kind == "full":
            world = full_world(self._bench, model, cpose, lid_at=lid_at)
            name = "full" + (("_" + tag) if tag else "")
        elif kind == "interaction":
            world = interaction_world(
                self._bench, model, cpose, target_xyz, contact_z, depth_max,
                lid_at=lid_at,
            )
            name = "interaction" + (("_" + tag) if tag else "")
        else:
            raise ValueError("unknown world kind %r" % kind)
        path = self._dir / (name + ".yaml")
        path.write_text(yaml.safe_dump(world, sort_keys=False))
        return name, path
