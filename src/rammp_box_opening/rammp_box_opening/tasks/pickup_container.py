"""pickup_container: grasp the body, lift; place back by default.

approach(above_body_grasp) -> grasp(body) -> lift -> place(pickup pose,
cycle-friendly for Phase-3 metrics; --hold skips place) -> home (spec §5).
"""

from rammp_box_opening.models.container import attitude_quat, from_container
from rammp_box_opening.primitives.core import (
    Approach,
    Grasp,
    Home,
    Lift,
    Place,
    PlanState,
    hover_above,
)

LIFT_DZ = 0.10


def build_legs(ctx, hold=False):
    m, cp = ctx.model, ctx.cpose
    grasp_pt = from_container(cp, m.body_grasp.offset)
    quat = attitude_quat(m.body_grasp.attitude_rpy_deg, cp.yaw)

    state = PlanState(
        joints=list(ctx.client.joints()), chain=0, contact_broke_chain=False
    )
    legs = []
    prims = [
        Approach(hover_above(grasp_pt, m.hover_standoff), quat, "approach:body"),
        Grasp(m.body_grasp, "grasp:body"),
        Lift(LIFT_DZ, band=m.body_grasp.expect_band),
    ]
    if not hold:
        prims.append(Place(grasp_pt, quat, name="place:container"))
    prims.append(Home())
    for prim in prims:
        new, state = prim.plan(ctx, state)
        legs += new
    return legs


def main():
    from rammp_box_opening.tasks import cli_common

    parser = cli_common.make_parser(__doc__)
    parser.add_argument(
        "--hold", action="store_true",
        help="keep the container lifted instead of placing it back",
    )
    args = parser.parse_args()
    cli_common.run_task(args, build_legs, hold=args.hold)
