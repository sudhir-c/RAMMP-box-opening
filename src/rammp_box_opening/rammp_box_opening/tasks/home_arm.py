"""home_arm: return to HOME — the recovery move and the abort-drill mover.

    ros2 run rammp_box_opening home_arm --execute      # then type 'yes'

Plans in the pre-detection BENCH world first: no container pose is
known to an isolated home (a failed run may have left the box anywhere),
so the whole placement band is blocked to container height and the move
stays above anything that could be standing there. A recovery home
usually STARTS inside that band, though — the arm is holding at a press,
a grip or a set-down — and then the band makes the start itself invalid.
So a refused start falls back to the bare table world with a printed
caution: the move then ignores where the box may be, and the operator
clears the bench before typing 'yes' (review 2026-09-02). Keeps the
typed-'yes' gate; the attended abort drill Ctrl+Cs this mid-motion.
"""

from rammp_box_opening.constants import HOME, TRANSIT_SPEED
from rammp_box_opening.primitives.core import PlanState, _plan_motion
from rammp_box_opening.tasks import cli_common


def build_legs(ctx):
    state = PlanState(joints=list(ctx.client.joints()), chain=0)
    guarded = ctx.worlds.push_name("bench", model=ctx.model)
    try:
        leg, _ = _plan_motion(
            ctx, state, "home", ("joints", list(HOME)), guarded, TRANSIT_SPEED
        )
        return [leg]
    except RuntimeError as e:
        print("[home_arm] home refused in the guarded bench world (%s)" % e)
    print(
        "[home_arm] CAUTION: planning in the bare table world — the move "
        "ignores where the box may be. Clear the bench before typing 'yes'."
    )
    bare = ctx.worlds.push_name("bench", model=None, tag="bare")
    leg, _ = _plan_motion(
        ctx, state, "home", ("joints", list(HOME)), bare, TRANSIT_SPEED
    )
    return [leg]


def main():
    args = cli_common.make_parser(__doc__).parse_args()
    cli_common.run_task(args, build_legs)
