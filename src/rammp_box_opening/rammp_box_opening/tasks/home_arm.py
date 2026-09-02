"""home_arm: return to HOME — the recovery move and the abort-drill mover.

    ros2 run rammp_box_opening home_arm --execute      # then type 'yes'

Plans in the pre-detection BENCH world: no container pose is known to
an isolated home (a failed run may have left the box anywhere), so the
whole placement band is blocked to container height and the move stays
above anything that could be standing there — the same world the
mission's own recovery home uses. Keeps the typed-'yes' gate; the
attended abort drill Ctrl+Cs this mid-motion (docs/HARDWARE_BRINGUP.md).
"""

from rammp_box_opening.constants import HOME, TRANSIT_SPEED
from rammp_box_opening.primitives.core import PlanState, _plan_motion
from rammp_box_opening.tasks import cli_common


def build_legs(ctx):
    world = ctx.worlds.push_name("bench", model=ctx.model)
    state = PlanState(joints=list(ctx.client.joints()), chain=0, contact_broke_chain=False)
    leg, _ = _plan_motion(
        ctx, state, "home", ("joints", list(HOME)), world, TRANSIT_SPEED
    )
    return [leg]


def main():
    args = cli_common.make_parser(__doc__).parse_args()
    cli_common.run_task(args, build_legs)
