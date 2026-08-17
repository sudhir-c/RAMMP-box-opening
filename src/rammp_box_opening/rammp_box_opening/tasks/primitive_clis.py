"""One CLI per primitive — isolated attended bring-up (spec §4, §8).

Each builds JUST its primitive's legs from the container model + measured
bench pose, previews them, and (only with --execute + typed 'yes' + the
planner's own gates) runs them. The Phase-1 ladder climbs these in order
before the task CLIs.
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
from rammp_box_opening.tasks import cli_common


def _start_state(ctx):
    return PlanState(
        joints=list(ctx.client.joints()), chain=0, contact_broke_chain=False
    )


def _run(args, prims_of):
    ctx, runner = cli_common.build_ctx(args)
    state = _start_state(ctx)
    legs = []
    for prim in prims_of(ctx):
        new, state = prim.plan(ctx, state)
        legs += new
    results = runner.run(legs, execute=args.execute)
    if any(not r.ok for r in results):
        raise SystemExit(1)


def main_approach():
    args = cli_common.make_parser(
        "approach: transit to the hover pose above the button"
    ).parse_args()

    def prims(ctx):
        m, cp = ctx.model, ctx.cpose
        button = from_container(cp, m.button_offset)
        return [
            Approach(
                hover_above(button, m.hover_standoff),
                attitude_quat(m.press_attitude_rpy_deg, cp.yaw),
                "approach:button",
            )
        ]

    _run(args, prims)


def main_press():
    args = cli_common.make_parser(
        "press: approach the button hover, then the guarded press descent"
    ).parse_args()

    def prims(ctx):
        m, cp = ctx.model, ctx.cpose
        button = from_container(cp, m.button_offset)
        return [
            Approach(
                hover_above(button, m.hover_standoff),
                attitude_quat(m.press_attitude_rpy_deg, cp.yaw),
                "approach:button",
            ),
            Press(),
            Retreat(m.hover_standoff),
        ]

    _run(args, prims)


def main_grasp():
    ap = cli_common.make_parser(
        "grasp: approach + guarded descent + graded close, lid or body"
    )
    ap.add_argument("--grasp", choices=["lid", "body"], default="lid")
    args = ap.parse_args()

    def prims(ctx):
        m, cp = ctx.model, ctx.cpose
        spec = m.lid_grasp if args.grasp == "lid" else m.body_grasp
        point = from_container(cp, spec.offset)
        quat = attitude_quat(spec.attitude_rpy_deg, cp.yaw)
        return [
            Approach(hover_above(point, m.hover_standoff), quat,
                     "approach:" + args.grasp),
            Grasp(spec, "grasp:" + args.grasp),
        ]

    _run(args, prims)


def main_lift():
    ap = cli_common.make_parser("lift: planned ascent from the current pose")
    ap.add_argument("--dz", type=float, default=0.10)
    args = ap.parse_args()
    _run(args, lambda ctx: [Lift(args.dz)])


def main_place():
    args = cli_common.make_parser(
        "place: transit above the lid spot, guarded set-down, release"
    ).parse_args()

    def prims(ctx):
        m, cp = ctx.model, ctx.cpose
        lid_at = cli_common.lid_place_of(args) or load_lid_place(ctx.config_path)
        quat = attitude_quat(m.lid_grasp.attitude_rpy_deg, cp.yaw)
        xyz = [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + m.lid_dims[2]]
        return [Place(xyz, quat, name="place:lid")]

    _run(args, prims)


def main_retreat():
    ap = cli_common.make_parser("retreat: vertical disengage from the last pose")
    ap.add_argument("--dz", type=float, default=0.08)
    args = ap.parse_args()

    def prims(ctx):
        m, cp = ctx.model, ctx.cpose
        # isolated use: seed last_pose from the live tool position
        xyz = ctx.client.tool_xyz()
        if xyz is None:
            raise SystemExit("no TF for tool_frame — is the arm bringup running?")
        ctx.last_pose = (
            xyz, attitude_quat(m.press_attitude_rpy_deg, cp.yaw)
        )
        return [Retreat(args.dz)]

    _run(args, prims)


def main_home():
    args = cli_common.make_parser("home: return to HOME joints").parse_args()
    _run(args, lambda ctx: [Home()])
