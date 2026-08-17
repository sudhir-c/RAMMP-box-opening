"""open_container: press the button, lift the lid clear, set it down.

approach(above_button) -> press -> retreat -> approach(above_lid) ->
grasp(lid) -> lift -> place(lid_spot) -> home (spec §5). Poses derive
from the container model + pose source; the lid set-down spot is the one
config-owned pose (open_container.lid_place), overridable by CLI flag.
"""

from rammp_box_opening.models.container import (
    attitude_quat,
    from_container,
    load_lid_place,
)
from rammp_box_opening.primitives.core import (
    Approach,
    Grasp,
    Home,
    Lift,
    Place,
    PlanState,
    Press,
    Retreat,
    hover_above,
)

LIFT_DZ = 0.10


def build_legs(ctx, lid_place=None):
    m, cp = ctx.model, ctx.cpose
    lid_at = lid_place if lid_place is not None else load_lid_place(ctx.config_path)
    button = from_container(cp, m.button_offset)
    lid_grasp_pt = from_container(cp, m.lid_grasp.offset)
    press_quat = attitude_quat(m.press_attitude_rpy_deg, cp.yaw)
    grasp_quat = attitude_quat(m.lid_grasp.attitude_rpy_deg, cp.yaw)

    state = PlanState(
        joints=list(ctx.client.joints()), chain=0, contact_broke_chain=False
    )
    legs = []
    for prim in [
        Approach(hover_above(button, m.hover_standoff), press_quat,
                 "approach:button"),
        Press(),
        Retreat(m.hover_standoff),
        Approach(hover_above(lid_grasp_pt, m.hover_standoff), grasp_quat,
                 "approach:lid"),
        Grasp(m.lid_grasp, "grasp:lid"),
        Lift(LIFT_DZ, band=m.lid_grasp.expect_band),
    ]:
        new, state = prim.plan(ctx, state)
        legs += new
    place_xyz = [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + m.lid_dims[2]]
    new, state = Place(place_xyz, grasp_quat, name="place:lid").plan(ctx, state)
    legs += new
    ctx.lid_at = lid_at  # worlds generated from here on carry the placed lid
    new, state = Home().plan(ctx, state)
    legs += new
    return legs


def main():
    from rammp_box_opening.tasks import cli_common

    parser = cli_common.make_parser(__doc__)
    args = parser.parse_args()
    cli_common.run_task(args, build_legs, lid_place=cli_common.lid_place_of(args))
