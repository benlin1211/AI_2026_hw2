"""Guideline-based deterministic wolf-score deltas.

The intent is not to decide the game alone. These features encode the practical
rules derived from the manual review: line battles, panda push chains, GJ
recalculation, hunter fishing, and claim withdrawal costs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def _clamp(x: float, lo: float = -0.25, hi: float = 0.25) -> float:
    return max(lo, min(hi, float(x)))


def _latest_state(board_states: Dict[str, Any]) -> Dict[str, Any]:
    if not board_states:
        return {}
    day = sorted(board_states.keys(), key=lambda d: int(d) if str(d).isdigit() else 999)[-1]
    return board_states[day]


def _edge_day(e: Dict[str, Any]) -> int:
    try:
        return int(str(e.get("day", 99)))
    except Exception:
        return 99


def _result_days_by_target(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {}
    for r in results:
        t = r.get("target")
        try:
            d = int(str(r.get("day", 99)))
        except Exception:
            d = 99
        if t and (t not in out or d < out[t]):
            out[t] = d
    return out


def score_guidelines(
    players: List[str],
    board_states: Dict[str, Any],
    objective_events: Dict[str, Any],
    interaction_graph: Dict[str, Any],
    objective_pack: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    latest = _latest_state(board_states)
    formation = latest.get("formation", objective_pack.get("formation", "0-0"))
    seer_claimers = set(latest.get("seer_claimers", objective_pack.get("seer_claimers", [])))
    medium_claimers = set(latest.get("medium_claimers", objective_pack.get("medium_claimers", [])))
    hunter_claimers = set(latest.get("hunter_claimers", []))
    pandas = set(latest.get("pandas", []))
    confirmed_whites = set(latest.get("confirmed_whites", []))
    gj_days = set(str(x) for x in latest.get("gj_days", objective_events.get("gj_days", [])))

    out = {p: {"delta": 0.0, "components": defaultdict(float), "reasons": []} for p in players}

    def add(p: str, amount: float, key: str, reason: str):
        if p not in out:
            return
        out[p]["components"][key] += amount
        out[p]["delta"] += amount
        if reason not in out[p]["reasons"]:
            out[p]["reasons"].append(reason)

    # 1. Claim formation priors: contested claimers are suspicious, but not too much
    # because Madman is not Werewolf.
    if len(seer_claimers) >= 2:
        for p in seer_claimers:
            add(p, 0.025, "contested_seer_claim", "contested seer claim: possible wolf or madman fake")
    elif len(seer_claimers) == 1:
        for p in seer_claimers:
            add(p, -0.030, "single_seer_claim", "single seer claim lowers actual-wolf probability")

    if len(medium_claimers) >= 2:
        for p in medium_claimers:
            add(p, 0.025, "contested_medium_claim", "contested medium claim: possible wolf or madman fake")
    elif len(medium_claimers) == 1:
        for p in medium_claimers:
            add(p, -0.035, "single_medium_claim", "single medium claim lowers actual-wolf probability")

    # 2. Withdrawals / village騙 cost. Small positive because it can be town play or madman.
    for w in objective_events.get("claim_withdrawals", []):
        add(w.get("player"), 0.025, "claim_withdrawal", "claim withdrawal consumes village resources")

    # 3. Hard public results.
    hard_results = list(objective_events.get("seer_results", []))
    if not hard_results:
        hard_results = list(objective_pack.get("hard_results", []))
    for r in hard_results:
        t = r.get("target")
        if t not in out:
            continue
        if r.get("result") == "werewolf":
            add(t, 0.080, "hard_black_received", "received hard black result")
        elif r.get("result") == "human":
            add(t, -0.035, "hard_human_received", "received hard human result")

    # Panda floor: mixed hard results are genuinely important but not conclusive.
    for p in pandas:
        add(p, 0.050, "panda", "panda/mixed hard results require elevated wolf risk")

    # 4. Medium results: if someone was medium-black, players who pressured them
    # before that are slightly anti-wolf; defenders are suspicious. If medium-human,
    # pushers are suspicious.
    medium_results = objective_events.get("medium_results", [])
    med_black_targets = {r.get("target") for r in medium_results if r.get("result") == "werewolf"}
    med_human_targets = {r.get("target") for r in medium_results if r.get("result") == "human"}

    for e in interaction_graph.get("edges", []):
        src, dst, typ = e.get("src"), e.get("dst"), e.get("type")
        if src not in out or dst not in out:
            continue
        strength = float(e.get("strength", 0.0))
        if typ in {"pressure", "vote_push", "claim_challenge"}:
            if dst in med_black_targets:
                add(src, -min(0.040, 0.025 * strength), "early_pressure_confirmed_wolf", "pressured a later medium-black target")
            if dst in med_human_targets:
                add(src, min(0.045, 0.030 * strength), "pushed_medium_human", "pushed a later medium-human target")
            if dst in pandas:
                add(src, min(0.040, 0.025 * strength), "panda_push_chain", "pushed a panda target; trace push chain")
        if typ in {"defense", "townread"}:
            if dst in med_black_targets:
                add(src, min(0.045, 0.030 * strength), "defended_medium_black", "defended/townread a later medium-black target")

    # 5. GJ / hunter signals. Hunter claims lower wolf slightly only if not numerous.
    if gj_days:
        for p in hunter_claimers:
            add(p, -0.035, "hunter_claim_after_gj", "hunter claim around GJ lowers wolf probability if plausible")
        # If a player keeps heavy pressure on confirmed/multi-white positions after
        # GJ, mark a small suspicion. This is intentionally weak.
        for e in interaction_graph.get("edges", []):
            src, dst, typ = e.get("src"), e.get("dst"), e.get("type")
            if src in out and dst in confirmed_whites and typ in {"pressure", "vote_push"}:
                add(src, 0.020, "pressure_confirmed_white_after_gj", "pressure on confirmed/multi-white position in GJ-heavy board")

    # 6. Hunter fishing / guard probing. Use raw guard reports by non-claimers as small wolf signal.
    for gr in objective_events.get("guard_reports", []):
        p = gr.get("player")
        if p in out and p not in hunter_claimers and gr.get("confidence", 0) < 0.70:
            txt = str(gr.get("text", "")).lower()
            if "hunter" in txt or "guard" in txt or "gj" in txt:
                add(p, 0.015, "hunter_guard_talk", "guard/GJ talk by non-hunter claimant; possible hunter probing")

    # 7. Formation-specific adjustments.
    if formation == "3-2":
        # In 3-2, gray LW search matters. Ability claimers may be fake, but gray
        # should not be ignored.
        gray = set(latest.get("gray_players", []))
        for p in gray:
            add(p, 0.015, "three_two_gray_lw", "3-2 board leaves high value on gray LW search")
    if formation == "2-2":
        for p in seer_claimers | medium_claimers:
            add(p, 0.010, "two_two_line_battle", "2-2 line battle: claimant fake can be wolf or madman")

    # Final formatting.
    for p in players:
        out[p]["delta"] = round(_clamp(out[p]["delta"]), 6)
        out[p]["components"] = {k: round(float(v), 6) for k, v in out[p]["components"].items()}
        out[p]["reasons"] = out[p]["reasons"][:8]
    return out
