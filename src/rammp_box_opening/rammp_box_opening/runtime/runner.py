"""The Runner: gates, merging, retries, run log (spec §6).

Dry-run is the default; motion needs execute=True AND a typed 'yes'
(assume_yes exists for the tests and for callers that already gated).
First unexpected outcome stops the task with the arm holding; the runner
never auto-continues past a cancel.
"""

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import (
    JOINT_VMAX,
    RECOIL_ARC_RAD,
    RECOIL_SPEED,
    CONTACT_SPEED,
    DRIFT_REPLAN_RAD,
    SANITY_MARGIN_RAD,
)
from rammp_box_opening.runtime import confirm
from rammp_box_opening.runtime.guards import TorqueGuard, sanity_violations
from rammp_box_opening.runtime.retime import (
    RetimeParams,
    positions_to_traj,
    retime_group,
    reverse_tail,
)
from rammp_box_opening.runtime.warp import warp_trajectory
from rammp_box_opening.runtime.legs import (
    Kind,
    Leg,
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
    # where the FINGERTIPS were when a guard tripped (TF, base frame): the
    # arm measuring the surface it touched, independent of the camera and
    # of every model constant
    contact_xyz: list = None
    lookahead: object = field(default=None, compare=False, repr=False)



def _restore_execution_profile(leg):
    """A replan swaps in a fresh planner-native trajectory; anything the
    original had baked in must be re-applied or honestly downgraded.

    Warped legs carry speed=1.0 as a do-not-dilate sentinel — executing
    an UNWARPED replacement at that sentinel is a full-speed descent
    into contact (review 2026-09-02). Re-warp the fresh trajectory; when
    it cannot be warped, run the WHOLE leg at the slow contact speed —
    never faster than intended. A leg with a retime hook (the press
    expects contact at a time fraction of the trajectory it will
    actually fly) gets it called on the final trajectory."""
    w = getattr(leg, "warp", None)
    if w is not None:
        fast, slow_speed, slow_frac = w
        warped, arm_frac = warp_trajectory(leg.traj, slow_frac, fast, slow_speed)
        if arm_frac is None:
            leg.speed = float(slow_speed)
            leg.guard = replace(leg.guard, rebaseline_after=None)
        else:
            leg.traj = warped
            # arm_after rides with the rebaseline: the fresh trajectory has
            # its own time base, and a guard armed before its NEW slow zone
            # judges slow-zone efforts against the fast baseline — the lid-
            # released-mid-air trip (review 2026-09-02)
            arm_after = leg.guard.arm_after
            if arm_after is not None:
                arm_after = max(arm_after, arm_frac)
            leg.guard = replace(leg.guard, rebaseline_after=arm_frac, arm_after=arm_after)
    retime = getattr(leg, "retime", None)
    if retime is not None:
        retime(leg.traj)


class Runner:
    def __init__(self, client, world_store, log_dir=None, margin_rad=SANITY_MARGIN_RAD):
        self.client = client
        self.worlds = world_store
        self.margin_rad = margin_rad
        # after a no-motion fault the server already runs its own ~4.5 s
        # servoing recovery; this is the client-side breather before the
        # single retry (tests set it to 0 — it was 35 % of the suite)
        self.no_motion_retry_delay_s = 3.0
        self._log_dir = Path(
            log_dir
            if log_dir is not None
            else Path.home() / ".ros" / "rammp_box_opening" / "runs"
        )
        self._log_path = None
        # an overlapped gripper command that outlives a run(): sent at fix
        # commit or on arrival at the hop, joined lazily before anything
        # that needs the fingers settled (audit 2026-09-02)
        self._pending = None  # (leg, handle, t0)
        # the human-motion profile for unguarded groups (retime.py); the
        # mission may replace it with the container's motion: block
        self.retime = RetimeParams()
        self.last_retime = None  # report of the most recent re-timed group
        # whole-run slow mode (--speed-scale): dilates every motion, guarded
        # strokes included — a first attempt at a new placement runs at a
        # fraction of speed without touching any per-leg tuning
        self.time_scale = 1.0

    # -- preview -----------------------------------------------------------
    def preview(self, legs):
        rows = [
            "%-18s %-7s %9s %-18s %5s %8s %7s %7s"
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
        # the time an UNGUARDED leg flies is its re-timed profile's, not
        # the planner's duration over its speed: preview what will run
        flown = {}
        for group in merge_groups(legs):
            if group[0].kind is Kind.MOTION and group[0].guard is None and all(
                g.traj is not None for g in group
            ):
                try:
                    _, info = retime_group(
                        [g.traj for g in group],
                        [g.speed for g in group],
                        JOINT_VMAX,
                        replace(self.retime, time_scale=self.time_scale),
                    )
                    per = info["duration_s"] / len(group)
                    for g in group:
                        flown[id(g)] = per if len(group) > 1 else info["duration_s"]
                except Exception:
                    pass
        for leg in legs:
            secs = "-"
            if id(leg) in flown:
                secs = "%.2f" % flown[id(leg)]
            elif leg.kind is Kind.MOTION and leg.traj is not None:
                last = leg.traj.points[-1].time_from_start
                secs = "%.2f" % ((last.sec + last.nanosec * 1e-9) / (leg.speed * self.time_scale))
            speed_txt = "%5.2f" % leg.speed
            if leg.warp:
                # the profile is baked into the timing; showing 1.00 would
                # read as "transit speed" when it is fast-then-contact
                speed_txt = "%.2f>%.2f" % (leg.warp[0], leg.warp[1])
            plan_total += leg.plan_s or 0.0
            solve_total += leg.plan_server_s or 0.0
            rows.append(
                "%-18s %-7s %9s %-18s %5d %8s %7s %7s"
                % (
                    leg.name,
                    leg.kind.value,
                    speed_txt,
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
                "%-18s %-7s %9s %-18s %5s %8s %7.2f %7.2f"
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
                    # a recoil reverses the corridor just descended — the
                    # same case as retreat/lift, by construction
                    leg.name.startswith(("retreat", "lift", "recoil"))
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
    def run(self, legs, execute, assume_yes=False, lookahead=None):
        """Execute legs in merge groups.

        `lookahead(end_joints)` — builds the NEXT phase's legs while this
        run's last unguarded motion flies (planned from that motion's
        predicted end joints). Its return value lands in
        self.lookahead_result; a failure leaves None and the caller builds
        after the run as before. Never hosted on a guarded stroke."""
        self.lookahead_result = None
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
        after_touch = False  # the previous motion stopped ON something
        next_chain = max((leg.chain for leg in legs), default=0) + 1
        groups = merge_groups(legs)
        host = None  # the last unguarded MOTION group hosts the lookahead
        if lookahead is not None:
            for g in reversed(groups):
                if g[0].kind is Kind.MOTION and all(x.guard is None for x in g):
                    host = g
                    break
        for gi, group in enumerate(groups):
            lead = group[0]
            # Worlds are pushed at PLAN time (core._plan_motion) and by the
            # replan path per leg; execution itself never consults the
            # collision world, so nothing is pushed here.

            # A deferred gripper command must be settled before anything
            # that depends on the fingers having arrived: another gripper
            # leg, or any guarded motion (the press descends with them
            # closed). Everything else — a plain transit, a replan — is
            # exactly what we want it to overlap with.
            if self._pending is not None and (
                lead.kind is Kind.GRIPPER or any(g.guard is not None for g in group)
            ):
                res = self._join_pending()
                results.append(res)
                if not res.ok:
                    print(
                        "STOP: leg %s -> %s (%s) — arm holds"
                        % (res.leg_name, res.outcome, res.detail)
                    )
                    return results

            if lead.kind is Kind.GRIPPER:
                if lead.defer_join and lead.verify is None:
                    handle = self.client.gripper_send(lead.gripper_cmd)
                    if handle is not None:
                        self._pending = (lead, handle, time.monotonic())
                        continue
                    # send failed — fall through to the blocking path so
                    # the failure is reported the same way as ever
                res = self._run_gripper(lead)
            else:
                # Free-air drift replans on MEASURED drift only (0.04 rad;
                # owner: fewer pauses, 2026-08-26). But after a TOUCH the
                # predicted start is wrong BY DESIGN — the guard stopped
                # the arm early — and a sub-gate joint delta is still a
                # multi-mm Cartesian shove INTO the thing just touched:
                # the pre-planned retreat re-pressed the button at full
                # speed, guardless, and stalled the arm (field
                # 2026-09-02). Post-touch, replan from live always; the
                # arm stop-and-holds at contact for the ~0.6 s it costs.
                if after_touch or self._drifted(group):
                    # a later leg's plan failing from the fresh chain is
                    # usually an unlucky cuRobo family draw (the
                    # 2026-09-01 INVALID_START home was one) — fresh
                    # draws are cheap, so try thrice before giving up;
                    # deterministic refusals just fail three times
                    for _attempt in range(3):
                        regroup, next_chain = self._replan_group(
                            group, next_chain
                        )
                        if regroup is not None:
                            break
                        print(
                            "[runner] replan of %s failed (attempt %d/3)"
                            % (group[0].name, _attempt + 1)
                        )
                    group = regroup
                    if group is None:
                        res = LegResult(
                            lead.name, "failed", False, "re-plan from live state failed"
                        )
                        self._log(res, lead)
                        results.append(res)
                        return results
                    lead = group[0]
                # a RELEASE overlaps the replan above, never the motion:
                # the arm must not move away from what it dropped before
                # the fingers have settled
                if self._pending is not None and self._pending[0].join_before_motion:
                    res = self._join_pending()
                    results.append(res)
                    if not res.ok:
                        print(
                            "STOP: leg %s -> %s (%s) — arm holds"
                            % (res.leg_name, res.outcome, res.detail)
                        )
                        return results
                after_touch = False
                hook = None
                if group is host:
                    end = list(group[-1].goal_joints)

                    def hook(_end=end):
                        return lookahead(_end)

                res = self._run_motion(group, while_running=hook)
                if group is host and res.lookahead is not None:
                    self.lookahead_result = res.lookahead
                after_touch = res.outcome == "touch"
            for g in group:
                self._log(res, g)
            results.append(res)
            if not res.ok:
                print(
                    "STOP: leg %s -> %s (%s) — arm holds"
                    % (res.leg_name, res.outcome, res.detail)
                )
                return results
            if (
                after_touch
                and lead.kind is Kind.MOTION
                and lead.guard is not None
                and lead.guard.trip == "press"
            ):
                # a good press leaves the arm pressed on the button while
                # the next leg is planned: recoil along the descent first.
                # A FAILED trip is left holding where it struck — the
                # operator needs to see that.
                nxt = next(
                    (g for g in groups[gi + 1 :] if g[0].kind is Kind.MOTION), None
                )
                handled, next_chain = self._reflex_recoil(
                    lead, res, nxt, next_chain, results
                )
                if handled:
                    after_touch = False  # nxt was planned from the recoil's end
        # a pending gripper command deliberately OUTLIVES the run: the next
        # phase's planning is what it overlaps with (finish() collects it)
        return results

    def start_gripper(self, name, cmd, execute, world="full"):
        """Dispatch a gripper command NOW, outside any leg sequence, to
        overlap the planning that follows (the press close goes out the
        moment the fix commits, audit 2026-09-02). Same gates as a leg;
        joined lazily like any deferred command. Returns False if refused."""
        from rammp_box_opening.runtime.legs import Leg

        leg = Leg(
            name=name,
            kind=Kind.GRIPPER,
            traj=None,
            speed=0.0,
            guard=None,
            world=world,
            chain=0,
            target=None,
            goal_joints=None,
            gripper_cmd=cmd,
            defer_join=True,
        )
        if not execute:
            print("[runner] %s: dry-run, not sent" % name)
            return True
        why = self._refusal(leg)
        if why:
            print("REFUSED: %s — %s" % (name, why))
            return False
        if self._pending is not None:
            res = self._join_pending()
            if not res.ok:
                return False
        handle = self.client.gripper_send(cmd)
        if handle is None:
            res = self._run_gripper(leg)  # send failed: blocking path, honest
            self._log(res, leg)
            return res.ok
        self._pending = (leg, handle, time.monotonic())
        return True

    def finish(self):
        """Collect any pending gripper command (mission end / exit)."""
        if self._pending is None:
            return None
        return self._join_pending()

    def _join_pending(self):
        leg, handle, t0 = self._pending
        self._pending = None
        res = self._join_gripper(leg, handle, t0)
        self._log(res, leg)
        return res

    # -- helpers -------------------------------------------------------------
    def _drifted(self, group):
        if any(g.traj is None for g in group):
            return True  # lazy: planned here, from live, for the first time
        start = group[0].traj.points[0].positions
        live = self.client.joints()
        return max(abs(ang_diff(a, b)) for a, b in zip(live, start)) > DRIFT_REPLAN_RAD

    def _reflex_recoil(self, leg, res, next_group, next_chain, results):
        """Back off a press contact along the path just flown, at once.

        The guard stops the arm ON the button and the considered retreat
        then takes 0.5-1.3 s to plan; a person recoils in about a tenth of
        a second. Reversing the descent's own executed tail needs no
        planner and no new collision check — that path was flown
        milliseconds ago, in this world, and it leads directly away from
        what was touched. The considered leg is planned WHILE the recoil
        flies, from the recoil's own end, so the arm is off the button
        before the planner is even asked.

        Returns (next_group was planned from the recoil's end, next_chain).
        """
        if leg.traj is None or res.progress is None:
            return False, next_chain
        path = reverse_tail(
            leg.traj, res.progress, self.client.joints(), RECOIL_ARC_RAD
        )
        if path is None:
            return False, next_chain
        traj, _info = retime_group(
            [positions_to_traj(leg.traj.joint_names, path)],
            [RECOIL_SPEED],
            JOINT_VMAX,
            replace(self.retime, time_scale=self.time_scale),
        )
        reflex = Leg(
            name="recoil",
            kind=Kind.MOTION,
            traj=traj,
            speed=1.0,  # the profile is baked in
            guard=None,
            world=leg.world,
            world_path=leg.world_path,
            chain=next_chain,
            target=None,
            goal_joints=[float(v) for v in path[-1]],
        )
        why = self._refusal(reflex)
        if why:
            print("REFUSED recoil — %s" % why)
            return False, next_chain
        hook = None
        if next_group is not None:
            def hook(_end=list(reflex.goal_joints)):
                return self._replan_group(next_group, next_chain + 1, start=_end)

        t0 = time.monotonic()
        outcome, info = self.client.execute(traj, 1.0, guard=None, while_running=hook)
        ok = outcome == "arrived"
        rres = LegResult(
            reflex.name,
            outcome,
            ok,
            info.get("message", ""),
            progress=info.get("progress"),
            t_wall=time.monotonic() - t0,
        )
        planned = False
        if ok and next_group is not None:
            regroup, _chain = info.get("while_running") or (None, None)
            planned = regroup is not None
        self._log(rres, reflex)
        if ok:
            results.append(rres)
            print(
                "  recoil — off the contact in %.2f s%s"
                % (
                    rres.t_wall or 0.0,
                    " (next leg planned during it)" if planned else "",
                )
            )
        else:
            # not fatal: the arm is somewhere along a path it just flew and
            # the next leg replans from live, exactly as without a recoil
            print(
                "  recoil did not complete (%s) — the next leg replans from live"
                % (rres.detail or outcome)
            )
        return (ok and planned), (next_chain + 2 if planned else next_chain + 1)

    def _replan_group(self, group, next_chain, start=None):
        """Re-plan each leg of the group from live state (or from `start`:
        a recoil plans the next leg from its own predicted end while it is
        still flying), same targets."""
        live = self.client.joints() if start is None else list(start)
        for leg in group:
            # Push THIS leg's world before re-planning it. Without this the
            # replan used whatever world happened to be loaded — for the
            # merged [retreat, home] group that meant re-planning `home` at
            # transit speed against an interaction world, whose obstacles
            # are deliberately capped (review 2026-08-28).
            ok, msg = self.client.set_world(leg.world_path or leg.world)
            if not ok:
                print("REFUSED replan of %s — set_world failed: %s" % (leg.name, msg))
                return None, next_chain
            kind, *rest = leg.target
            if kind == "pose":
                off = float(rest[2]) if len(rest) > 2 else 0.0
                plan = self.client.plan_to_pose(
                    rest[0], rest[1], live, approach_offset_m=off
                )
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
            leg.goal_joints = list(plan.trajectory.points[-1].positions)
            _restore_execution_profile(leg)
            # the pre-execution gates ran against the ORIGINAL trajectory;
            # a replan produces a new one and must clear them again
            why = self._refusal(leg)
            if why:
                print("REFUSED replan of %s — %s" % (leg.name, why))
                return None, next_chain
            live = leg.goal_joints
        return group, next_chain + 1

    def _run_motion(self, group, while_running=None):
        # The member that OWNS this execution's contact semantics. Today
        # can_merge forbids guarded legs in a group, so this is group[0];
        # the lookup exists so that relaxing can_merge cannot silently run
        # a guarded stroke with the lead's guard (None), the lead's speed
        # and the lead's verify — which would be an unguarded press with no
        # error and no log line (review 2026-08-28).
        guarded = [g for g in group if g.guard is not None]
        if len(guarded) > 1:
            raise RuntimeError(
                "merged group has %d guarded legs (%s) — one execution can "
                "carry at most one contact guard"
                % (len(guarded), ", ".join(g.name for g in guarded))
            )
        if guarded and guarded[0] is not group[-1]:
            raise RuntimeError(
                "guarded leg %r is not last in its merged group — a trip "
                "must not strand queued motion behind it" % guarded[0].name
            )
        lead = guarded[0] if guarded else group[0]
        if lead.guard is None:
            # unguarded: ONE profile over the whole group — ease out, cruise
            # at each leg's speed fraction, flow through the junctions, long
            # ease in — baked into the timestamps, flown at the 1.0 sentinel
            params = replace(self.retime, time_scale=self.time_scale)
            traj, self.last_retime = retime_group(
                [g.traj for g in group], [g.speed for g in group], JOINT_VMAX, params
            )
            exec_speed = 1.0
        else:
            traj = (
                merge_trajectories([g.traj for g in group])
                if len(group) > 1
                else group[0].traj
            )
            exec_speed = lead.speed * self.time_scale
        def make_guard():
            return (
                TorqueGuard(
                    lead.guard.touch_nm,
                    rebaseline_after=lead.guard.rebaseline_after,
                    arm_after=lead.guard.arm_after,
                )
                if lead.guard
                else None
            )

        guard = make_guard()
        t0 = time.monotonic()
        outcome, info = self.client.execute(
            traj, exec_speed, guard=guard, while_running=while_running
        )
        if outcome == "failed" and NO_MOTION_SIGNATURE in info.get("message", ""):
            print("  no-motion fault at start — one retry from standstill")
            time.sleep(self.no_motion_retry_delay_s)
            first = info
            # a FRESH guard: the phantom first attempt armed the old one on
            # a standstill baseline and ran its progress to 1.0
            guard = make_guard()
            outcome, info = self.client.execute(traj, exec_speed, guard=guard)
            # the hosted lookahead already planned from this leg's predicted
            # end, which the retry still reaches: keep it (its build-time
            # side effects on ctx are real either way)
            if "while_running" in first:
                info["while_running"] = first["while_running"]
        if guard is not None:
            info.setdefault("torque_peak", guard.peak)
        if info.get("while_running_error"):
            print("  lookahead plan failed (%s) — planning after the leg" % info["while_running_error"])
        contact = self.client.contact_xyz() if outcome == "touch" else None
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
            contact_xyz=contact,
            lookahead=info.get("while_running"),
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

    def _join_gripper(self, leg, handle, t0):
        """Collect an overlapped gripper command. Identical verdicts to
        the blocking path — only the waiting moved."""
        ok, pos, stalled = self.client.gripper_join(handle)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s (overlapped)" % (
            pos,
            " (stalled)" if stalled else "",
        )
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail, t_wall=time.monotonic() - t0)

    def _run_gripper(self, leg):
        t0 = time.monotonic()
        ok, pos, stalled = self.client.gripper_cmd(leg.gripper_cmd)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s" % (pos, " (stalled)" if stalled else "")
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail, t_wall=time.monotonic() - t0)

    def note(self, kind, **fields):
        """A non-leg row in the run log (the detected fix, the press
        lateral): a missed press is diagnosable from disk, not from a
        terminal paste (bench 2026-09-03)."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if self._log_path is None:
            self._log_path = self._log_dir / time.strftime("run-%Y%m%d-%H%M%S.jsonl")
        row = {"t": time.time(), "leg": kind, "kind": "note", **fields}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

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
        row.pop("lookahead", None)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
