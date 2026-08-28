"""The Runner: gates, merging, retries, run log (spec §6).

Dry-run is the default; motion needs execute=True AND a typed 'yes'
(assume_yes exists for the tests and for callers that already gated).
First unexpected outcome stops the task with the arm holding; the runner
never auto-continues past a cancel.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import (
    CONTACT_SPEED,
    DRIFT_REPLAN_RAD,
    SANITY_MARGIN_RAD,
)
from rammp_box_opening.runtime import confirm
from rammp_box_opening.runtime.guards import TorqueGuard, sanity_violations
from rammp_box_opening.runtime.legs import (
    Kind,
    VerifyCtx,
    merge_groups,
    merge_trajectories,
)

NO_MOTION_SIGNATURE = "never left the start"


@dataclass
class LegResult:
    leg_name: str
    outcome: str  # arrived | touch | failed | refused | skipped
    ok: bool
    detail: str = ""
    torque_peak: float = None
    progress: float = None
    t_wall: float = None


class Runner:
    def __init__(self, client, world_store, log_dir=None, margin_rad=SANITY_MARGIN_RAD):
        self.client = client
        self.worlds = world_store
        self.margin_rad = margin_rad
        self._log_dir = Path(
            log_dir
            if log_dir is not None
            else Path.home() / ".ros" / "rammp_box_opening" / "runs"
        )
        self._log_path = None
        self._last_world = None

    # -- preview -----------------------------------------------------------
    def preview(self, legs):
        rows = [
            "%-18s %-7s %5s %-18s %5s %8s %7s %7s"
            % (
                "leg",
                "kind",
                "speed",
                "world",
                "chain",
                "time_s",
                "plan_s",
                "solve_s",
            )
        ]
        plan_total = solve_total = 0.0
        for leg in legs:
            secs = "-"
            if leg.kind is Kind.MOTION and leg.traj is not None:
                last = leg.traj.points[-1].time_from_start
                secs = "%.2f" % ((last.sec + last.nanosec * 1e-9) / leg.speed)
            plan_total += leg.plan_s or 0.0
            solve_total += leg.plan_server_s or 0.0
            rows.append(
                "%-18s %-7s %5.2f %-18s %5d %8s %7s %7s"
                % (
                    leg.name,
                    leg.kind.value,
                    leg.speed,
                    leg.world,
                    leg.chain,
                    secs,
                    "-" if leg.plan_s is None else "%.2f" % leg.plan_s,
                    "-" if leg.plan_server_s is None else "%.2f" % leg.plan_server_s,
                )
            )
        if plan_total > 0.0:
            # round trip vs. what the planner says it spent solving: the
            # difference is transport/queueing, and it is the half we have
            # not yet accounted for (see Leg.plan_s)
            rows.append(
                "%-18s %-7s %5s %-18s %5s %8s %7.2f %7.2f"
                % ("(planning total)", "", "", "", "", "", plan_total, solve_total)
            )
        return "\n".join(rows)

    # -- gates (checked BEFORE anything executes) ---------------------------
    def _refusal(self, leg):
        if leg.kind is Kind.MOTION:
            # Transit-speed legs require the fullest world KNOWN: "full"
            # (bench + container), or "bench" in the pre-detection epoch —
            # a container cannot be modeled before one is seen (press_demo
            # scan / no-tag home). Slow (contact-speed) unguarded legs are
            # the one exception: a post-contact retreat must plan against
            # the interaction world its descent used — in the full world
            # its start would read as inside the container.
            if (
                leg.guard is None
                and leg.speed > CONTACT_SPEED
                and not leg.world.startswith(("full", "bench"))
                # retreat and lift both ascend OUT of the corridor just
                # descended, planned collision-free in that same
                # interaction world — faster-than-contact is fine for them
                # (owner: fast up, 2026-08-26; lift added 2026-08-28).
                # Retreat already runs at TRANSIT_SPEED here, so a 0.35
                # lift is the milder of the two.
                and not (
                    leg.name.startswith(("retreat", "lift"))
                    and leg.world.startswith("interaction")
                )
            ):
                return (
                    "transit-speed MOTION leg planned against %r — full or "
                    "bench world required (spec §6)" % leg.world
                )
            if leg.traj is not None:
                bad = sanity_violations(leg.traj, self.margin_rad)
                if bad:
                    return "trajectory sanity gate: " + "; ".join(bad)
            if leg.guard is not None and not self.client.efforts_present():
                return (
                    "no effort fields in /joint_states — guarded legs "
                    "refuse to run (spec §6)"
                )
        if leg.kind is Kind.GRIPPER and not self.client.planner_execute_enabled():
            return (
                "planner execute param is false — refusing GRIPPER leg "
                "(planner dry-run does NOT gate the direct gripper action; "
                "the runner enforces symmetry)"
            )
        return None

    # -- run ----------------------------------------------------------------
    def run(self, legs, execute, assume_yes=False):
        print(self.preview(legs))
        if not execute:
            print("dry-run complete — nothing moved (add --execute)")
            return [LegResult(leg.name, "skipped", True) for leg in legs]
        if not assume_yes and not confirm.typed_yes(
            "Type 'yes' to execute (human on the physical e-stop): "
        ):
            print("aborted — nothing moved")
            return [LegResult(leg.name, "skipped", True) for leg in legs]

        for leg in legs:
            why = self._refusal(leg)
            if why:
                print("REFUSED: %s — %s" % (leg.name, why))
                res = LegResult(leg.name, "refused", False, why)
                self._log(res, leg)
                return [res]

        results = []
        next_chain = max((leg.chain for leg in legs), default=0) + 1
        for group in merge_groups(legs):
            lead = group[0]
            if lead.world != self._last_world:
                ok, msg = self.client.set_world(lead.world_path or lead.world)
                if not ok:
                    res = LegResult(
                        lead.name, "refused", False, "set_world failed: " + msg
                    )
                    self._log(res, lead)
                    results.append(res)
                    return results
                self._last_world = lead.world

            if lead.kind is Kind.GRIPPER:
                res = self._run_gripper(lead)
            else:
                # replan only on MEASURED drift (live vs planned start,
                # 0.04 rad): a guard trip near full travel leaves the arm
                # within tolerance and the pre-planned leg runs at once —
                # forcing replans after every contact cost ~2 s pressed
                # into the button (owner: fewer pauses, 2026-08-26). The
                # server's own start gate (0.05 rad) still backstops.
                if self._drifted(group):
                    group, next_chain = self._replan_group(group, next_chain)
                    if group is None:
                        res = LegResult(
                            lead.name, "failed", False, "re-plan from live state failed"
                        )
                        self._log(res, lead)
                        results.append(res)
                        return results
                    lead = group[0]
                res = self._run_motion(group)
            for g in group:
                self._log(res, g)
            results.append(res)
            if not res.ok:
                print(
                    "STOP: leg %s -> %s (%s) — arm holds"
                    % (res.leg_name, res.outcome, res.detail)
                )
                return results
        return results

    # -- helpers -------------------------------------------------------------
    def _drifted(self, group):
        start = group[0].traj.points[0].positions
        live = self.client.joints()
        return max(abs(ang_diff(a, b)) for a, b in zip(live, start)) > DRIFT_REPLAN_RAD

    def _replan_group(self, group, next_chain):
        """Re-plan each leg of the group from live state, same targets."""
        live = self.client.joints()
        for leg in group:
            kind, *rest = leg.target
            if kind == "pose":
                plan = self.client.plan_to_pose(rest[0], rest[1], live)
            else:
                plan = self.client.plan_to_joints(rest[0], live)
            if plan is None or not plan.success:
                return None, next_chain
            # replanned trajectories pass the SAME sanity gate as pre-built
            # ones — a wandering replan executed ungated defeats spec §6
            bad = sanity_violations(plan.trajectory, self.margin_rad)
            if bad:
                print("REFUSED replan of %s — %s" % (leg.name, "; ".join(bad)))
                return None, next_chain
            leg.traj = plan.trajectory
            leg.chain = next_chain
            leg.stale = False
            leg.goal_joints = list(plan.trajectory.points[-1].positions)
            live = leg.goal_joints
        return group, next_chain + 1

    def _run_motion(self, group):
        lead = group[0]
        traj = (
            merge_trajectories([g.traj for g in group]) if len(group) > 1 else lead.traj
        )
        guard = TorqueGuard(lead.guard.touch_nm) if lead.guard else None
        t0 = time.monotonic()
        outcome, info = self.client.execute(traj, lead.speed, guard=guard)
        if outcome == "failed" and NO_MOTION_SIGNATURE in info.get("message", ""):
            print("  no-motion fault at start — one retry from standstill")
            time.sleep(3.0)
            outcome, info = self.client.execute(traj, lead.speed, guard=guard)
        depth = None
        if outcome == "touch" and lead.guard and lead.guard.needs_depth:
            # gated on needs_depth: the lookup blocks for its full timeout
            # on a TF tree with no tool_frame, and PressFixed — the press
            # the mission actually runs — never reads the result (0.5 s
            # per press trip, measured 2026-08-28)
            tool = self.client.tool_xyz()
            depth = (lead.guard.target_z - tool[2]) if tool else None
        ok = self._leg_ok(lead, outcome)
        detail = info.get("message", "")
        if lead.verify is not None:
            g_ok, g_pos, _ = self.client.gripper_cmd(None)  # query only
            ok, detail = lead.verify(
                VerifyCtx(
                    outcome=outcome,
                    depth_m=depth,
                    gripper_pos=g_pos if g_ok else None,
                    progress=info.get("progress"),
                    torque_peak=info.get("torque_peak"),
                )
            )
        return LegResult(
            group[-1].name if len(group) > 1 else lead.name,
            outcome,
            ok,
            detail,
            torque_peak=info.get("torque_peak"),
            progress=info.get("progress"),
            t_wall=time.monotonic() - t0,
        )

    @staticmethod
    def _leg_ok(leg, outcome):
        if leg.guard is None:
            return outcome == "arrived"
        if leg.guard.trip == "setdown":
            return outcome == "touch"
        if leg.guard.trip == "obstruction":
            return outcome == "arrived"  # a trip means we struck the lid/rim
        return outcome == "touch"  # press: verify refines via depth

    def _run_gripper(self, leg):
        t0 = time.monotonic()
        ok, pos, stalled = self.client.gripper_cmd(leg.gripper_cmd)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s" % (pos, " (stalled)" if stalled else "")
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail, t_wall=time.monotonic() - t0)

    def _log(self, res, leg):
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if self._log_path is None:
            self._log_path = self._log_dir / time.strftime("run-%Y%m%d-%H%M%S.jsonl")
        row = {
            "t": time.time(),
            "leg": leg.name,
            "kind": leg.kind.value,
            "world": leg.world,
            "speed": leg.speed,
            **asdict(res),
        }
        row.pop("leg_name", None)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
