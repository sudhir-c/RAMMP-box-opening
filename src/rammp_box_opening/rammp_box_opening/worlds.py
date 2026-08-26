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
# Reduction plane below the deepest command. EMPIRICAL 2026-08-25 (margin
# probe vs the real planner): the finger collision spheres reach ~4 cm
# below tool_frame and padding adds 2 cm — at 0.03 the hover AND press
# goals were in collision (IK_FAIL); 0.08 plans.
PLANE_MARGIN_M = 0.08


def _bench_obstacles(bench):
    return [dict(o) for o in bench["obstacles"]]


def _cap_top(obs, top_z):
    """Copy of a cuboid with its top lowered to top_z; None if nothing of
    it remains. No-op when the top is already at or below top_z."""
    x, y, z = obs["position"]
    dx, dy, dz = obs["dims"]
    top, bottom = z + dz / 2, z - dz / 2
    if top <= top_z:
        return dict(obs)
    if bottom >= top_z:
        return None
    capped = dict(obs)
    capped["position"] = [x, y, bottom + (top_z - bottom) / 2]
    capped["dims"] = [dx, dy, top_z - bottom]
    return capped


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


PLACEMENT_BAND_X = (0.15, 0.75)  # where a container may sit (reach map)
PLACEMENT_BAND_Y = (-0.50, 0.50)


def _table_top_z(bench):
    t = next(o for o in bench["obstacles"] if o["name"] == "table")
    return t["position"][2] + t["dims"][2] / 2


def bench_world(bench, unseen_model=None):
    """The pre-detection world (press_demo scan / no-tag legs).

    No container POSE is known yet, but one is PRESENT somewhere: with a
    model given, the whole placement band is blocked to container height
    so pre-detection transits stay above anything that could be standing
    there — a failed detection must not mean a blind sweep through the
    container (2026-08-24 review). Scan/home poses live well above the
    band."""
    obstacles = _bench_obstacles(bench)
    if unseen_model is not None:
        top = _table_top_z(bench)
        h = unseen_model.dims[2] + ERR_TALL_M
        x0, x1 = PLACEMENT_BAND_X
        y0, y1 = PLACEMENT_BAND_Y
        obstacles.append(
            {
                "name": "unseen_container_band",
                "position": [(x0 + x1) / 2, (y0 + y1) / 2, top + h / 2],
                "dims": [x1 - x0, y1 - y0, h],
            }
        )
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
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


def interaction_world(
    bench, model, cpose, target_xyz, contact_z, depth_max, lid_at=None, ring=True
):
    plane = reduction_plane_z(contact_z, depth_max)
    # bench obstacles cap at the reduction plane too: a set-down ONTO the
    # bench must be plannable to its commanded overdrive depth, same
    # spec §6 rule as the container body (field 2026-08-26: IK_FAIL at a
    # set-down goal 25 mm over the solid table). For button-height
    # contacts the plane sits below every bench top — a no-op. Only
    # guarded/slow legs live in this world; the transit gate stands.
    obstacles = [
        c for c in (_cap_top(o, plane) for o in _bench_obstacles(bench)) if c
    ]
    if plane > cpose.xyz[2]:  # body below the plane stays solid
        obstacles.append(
            _container_cuboid(model, cpose, name="container_body", top_z=plane)
        )
    if ring:
        # Aperture ring: lateral entry forbidden, vertical corridor free.
        # Sized for Phase-1's deep grasp descents; at press-demo hover
        # heights its walls collide with the gripper body (empirical
        # 2026-08-25) — PressFixed passes ring=False.
        tx, ty = target_xyz[0], target_xyz[1]
        a, w, h = APERTURE_HALF_M, RING_THICK_M, RING_HEIGHT_M
        zc = plane + h / 2
        span = 2 * (a + w)
        obstacles += [
            {
                "name": "ring_xp",
                "position": [tx + a + w / 2, ty, zc],
                "dims": [w, span, h],
            },
            {
                "name": "ring_xn",
                "position": [tx - a - w / 2, ty, zc],
                "dims": [w, span, h],
            },
            {
                "name": "ring_yp",
                "position": [tx, ty + a + w / 2, zc],
                "dims": [span, w, h],
            },
            {
                "name": "ring_yn",
                "position": [tx, ty - a - w / 2, zc],
                "dims": [span, w, h],
            },
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

    def push_name(
        self,
        kind,
        model=None,
        cpose=None,
        target_xyz=None,
        contact_z=None,
        depth_max=None,
        lid_at=None,
        tag="",
        ring=True,
    ):
        if kind == "bench":
            world = bench_world(self._bench, unseen_model=model)
            name = "bench" + (("_" + tag) if tag else "")
        elif kind == "full":
            world = full_world(self._bench, model, cpose, lid_at=lid_at)
            name = "full" + (("_" + tag) if tag else "")
        elif kind == "interaction":
            world = interaction_world(
                self._bench,
                model,
                cpose,
                target_xyz,
                contact_z,
                depth_max,
                lid_at=lid_at,
                ring=ring,
            )
            name = "interaction" + (("_" + tag) if tag else "")
        else:
            raise ValueError("unknown world kind %r" % kind)
        path = self._dir / (name + ".yaml")
        path.write_text(yaml.safe_dump(world, sort_keys=False))
        return name, path
