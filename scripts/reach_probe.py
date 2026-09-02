#!/usr/bin/env python3
"""Offline tool-down reach probe: where on the bench can we press?

    zsh -c 'source /opt/ros/humble/setup.zsh &&
            source ~/RAMMP-CuRobo/install/setup.zsh &&
            cd ~/RAMMP-box-opening && source install/setup.zsh &&
            python3 scripts/reach_probe.py'          # add --quick to smoke

Loads the planning core IN PROCESS (CuRoboPlanner.from_config, no ROS, no
arm, nothing can move) and sweeps a bench grid of TOOL-DOWN poses — the
press-attitude family every contact primitive uses (attitude_quat
[180, 0, 0], yaw-steered) — at the two heights the mission commands over a
container on the table: hover (button top + hover_standoff) and contact
(button top). tool_frame is the FINGERTIP midpoint, 0.120 m beyond the
wrist flange: flat-wrist poses fold the arm into itself inside ~0.5 m
radius, and the TOOL-DOWN band is different and previously unmeasured —
this map is what defines where the container may sit (field lesson 1).

Writes docs/reach_map.json (deterministic path, meant to be committed)
and prints per-height ASCII maps. Also probes the two configured mission
poses: the scan pose and the lid_place carry hover, at their own yaw.

Caveats recorded in the JSON meta: bare-bench world (the container adds
obstacles that can only shrink the map), start = HOME for every plan
(mission plans chain from prior legs), probe yaw = point bearing
(joint_7 redundancy makes tool-down yaw second-order; mission poses are
additionally probed at their configured yaw).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

from rammp_box_opening.constants import HOME  # noqa: E402
from rammp_box_opening.models.container import (  # noqa: E402
    ContainerModel,
    attitude_quat,
    load_lid_place,
    load_press_demo,
)
from rammp_box_opening.worlds import WorldStore  # noqa: E402

CONTAINER_YAML = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"
BENCH_YAML = REPO / "src/rammp_box_opening/config/world_bench.yaml"
OUT_JSON = REPO / "docs/reach_map.json"


def probe_grid(planner, xs, ys, z, label):
    rows = []
    for y in ys:
        for x in xs:
            quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(y, x))
            t0 = time.monotonic()
            res = planner.plan_to_pose([x, y, z], quat, start=list(HOME))
            rows.append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "level": label,
                    "success": bool(res.success),
                    "time_s": round(time.monotonic() - t0, 2),
                    "msg": "" if res.success else str(getattr(res, "message", ""))[:80],
                }
            )
        done = sum(1 for r in rows if r["success"])
        print(
            "[%s] y=%+.2f done — %d/%d reachable so far" % (label, y, done, len(rows)),
            flush=True,
        )
    return rows


def ascii_map(rows, xs, ys):
    ok = {(r["x"], r["y"]) for r in rows if r["success"]}
    lines = ["      x " + " ".join("%4.2f" % x for x in xs)]
    for y in ys:
        cells = ["  # " if (round(x, 3), round(y, 3)) in ok else "  . " for x in xs]
        lines.append("y %+.2f " % y + "".join(cells))
    return "\n".join(lines)


def probe_mission_poses(planner, model, scan_xyz, lid):
    """The configured (not detected) poses the mission commands, at their
    own yaw: the scan pose and the lid set-down's carry hover."""
    out = []
    for name, xyz, yaw in [
        ("scan", list(scan_xyz), math.atan2(scan_xyz[1], scan_xyz[0])),
        (
            "lid_place:hover",
            [
                lid.xyz[0],
                lid.xyz[1],
                lid.xyz[2] + model.lid_dims[2] + model.hover_standoff,
            ],
            lid.yaw,
        ),
    ]:
        quat = attitude_quat(list(model.press_attitude_rpy_deg), yaw)
        res = planner.plan_to_pose(list(xyz), quat, start=list(HOME))
        out.append(
            {
                "name": name,
                "xyz": [round(v, 3) for v in xyz],
                "yaw": round(yaw, 3),
                "success": bool(res.success),
                "msg": "" if res.success else str(getattr(res, "message", ""))[:80],
            }
        )
        print(
            "[mission] %-16s %s  %s"
            % (name, "OK  " if res.success else "FAIL", out[-1]["msg"]),
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pitch", type=float, default=0.05, help="grid pitch (m)")
    ap.add_argument(
        "--quick", action="store_true", help="3x3 smoke grid (API/GPU check only)"
    )
    args = ap.parse_args()

    from rammp_curobo import CuRoboPlanner

    model = ContainerModel.load(str(CONTAINER_YAML))
    scan_xyz = load_press_demo(str(CONTAINER_YAML)).scan_xyz
    lid = load_lid_place(str(CONTAINER_YAML))
    top = WorldStore(str(BENCH_YAML)).table_top_z
    z_contact = top + model.button_offset[2]
    z_hover = z_contact + model.hover_standoff

    if args.quick:
        xs = [0.35, 0.45, 0.55]
        ys = [-0.15, 0.0, 0.15]
    else:
        n_x = int(round((0.75 - 0.20) / args.pitch))
        n_y = int(round(0.90 / args.pitch))
        xs = [round(0.20 + i * args.pitch, 3) for i in range(n_x + 1)]
        ys = [round(-0.45 + i * args.pitch, 3) for i in range(n_y + 1)]

    print(
        "reach probe: %d x %d grid, z_contact=%.3f z_hover=%.3f (table top %.3f)"
        % (len(xs), len(ys), z_contact, z_hover, top),
        flush=True,
    )
    print("loading planner (gen3_real.yaml, ~20 s GPU init)...", flush=True)
    planner = CuRoboPlanner.from_config("gen3_real.yaml")

    t0 = time.monotonic()
    levels = {}
    for label, z in [("contact", z_contact), ("hover", z_hover)]:
        levels[label] = probe_grid(planner, xs, ys, z, label)
    mission = probe_mission_poses(planner, model, scan_xyz, lid)

    for label in levels:
        n_ok = sum(1 for r in levels[label] if r["success"])
        print(
            "\n=== %s (z=%.3f): %d/%d reachable ==="
            % (label, levels[label][0]["z"], n_ok, len(levels[label]))
        )
        print(ascii_map(levels[label], xs, ys))

    both = {
        (r["x"], r["y"])
        for r in levels["contact"]
        if r["success"]
        and any(
            s["success"] and (s["x"], s["y"]) == (r["x"], r["y"])
            for s in levels["hover"]
        )
    }
    print(
        "\nusable (hover AND contact): %d/%d grid points"
        % (len(both), len(xs) * len(ys))
    )
    if both:
        bx = [p[0] for p in both]
        by = [p[1] for p in both]
        print(
            "hull: x [%.2f, %.2f], y [%.2f, %.2f] (holes possible — see map)"
            % (min(bx), max(bx), min(by), max(by))
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "meta": {
                    "date": time.strftime("%Y-%m-%d"),
                    "config": "gen3_real.yaml (bare bench — no container obstacle)",
                    "attitude": "tool-down attitude_quat([180,0,0], bearing)",
                    "start": "HOME (mission plans chain from prior legs)",
                    "z_contact": round(z_contact, 3),
                    "z_hover": round(z_hover, 3),
                    "table_top_z": round(top, 3),
                    "pitch": args.pitch,
                    "quick": args.quick,
                },
                "grid": levels["contact"] + levels["hover"],
                "mission_poses": mission,
            },
            f,
            indent=1,
        )
    print("\nwrote %s in %.0f s" % (OUT_JSON, time.monotonic() - t0))


if __name__ == "__main__":
    main()
