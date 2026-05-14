"""Fast cumulative board-state builder."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set

from src.solver import get_role_counts
from src.formation_policy import classify_formation


def _claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def _day_sort_key(day: Any) -> int:
    try:
        return int(str(day))
    except Exception:
        return 999


def _unique(xs: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in xs if x]))


def _claims_by_day(parsed: Dict[str, Any], day: str) -> Dict[str, List[Any]]:
    return parsed.get("days", {}).get(str(day), {}).get("claims", {})


def _result_day(r: Dict[str, Any]) -> str:
    d = r.get("day")
    return str(d) if d not in (None, "") else "999"


def build_fast_board_states(parsed: Dict[str, Any], objective_events: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    role_counts = get_role_counts(len(players))
    days = [d for d in parsed.get("days", {}).keys() if str(d).isdigit()]
    days = sorted(days, key=_day_sort_key)
    if not days:
        days = ["1"]

    alive: Set[str] = set(players)
    dead: Set[str] = set()
    cumulative_claims: Dict[str, List[Any]] = defaultdict(list)
    seer_results: List[Dict[str, Any]] = []
    medium_results: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []
    night_kills: List[Dict[str, Any]] = []
    sudden_deaths: List[Dict[str, Any]] = []
    hunter_claims: List[Dict[str, Any]] = []
    guard_reports: List[Dict[str, Any]] = []
    gj_days: List[str] = []
    withdrawals: List[Dict[str, Any]] = []

    states: Dict[str, Any] = {}

    seer_by_day = defaultdict(list)
    for r in objective_events.get("seer_results", []):
        seer_by_day[_result_day(r)].append(r)
    medium_by_day = defaultdict(list)
    for r in objective_events.get("medium_results", []):
        medium_by_day[_result_day(r)].append(r)
    hunter_by_day = defaultdict(list)
    for r in objective_events.get("hunter_claims", []):
        hunter_by_day[str(r.get("day", "999"))].append(r)
    guard_by_day = defaultdict(list)
    for r in objective_events.get("guard_reports", []):
        guard_by_day[str(r.get("day", "999"))].append(r)
    withdrawal_by_day = defaultdict(list)
    for r in objective_events.get("claim_withdrawals", []):
        withdrawal_by_day[str(r.get("day", "999"))].append(r)
    formation_hint_by_day = defaultdict(list)
    for r in objective_events.get("formation_hints", []):
        formation_hint_by_day[str(r.get("day", "999"))].append(r)

    cumulative_formation_hints: List[Dict[str, Any]] = []

    for day in days:
        # Claims.
        for p, cs in _claims_by_day(parsed, str(day)).items():
            cumulative_claims[p].extend(cs)

        de = objective_events.get("day_events", {}).get(str(day), {})
        executed.extend(de.get("executed", []))
        night_kills.extend(de.get("night_kills", []))
        sudden_deaths.extend(de.get("sudden_deaths", []))
        if de.get("gj") and str(day) not in gj_days:
            gj_days.append(str(day))

        for x in de.get("executed", []) + de.get("night_kills", []) + de.get("sudden_deaths", []):
            p = x.get("player")
            if p in alive:
                alive.remove(p)
                dead.add(p)

        seer_results.extend(seer_by_day.get(str(day), []))
        medium_results.extend(medium_by_day.get(str(day), []))
        hunter_claims.extend(hunter_by_day.get(str(day), []))
        guard_reports.extend(guard_by_day.get(str(day), []))
        withdrawals.extend(withdrawal_by_day.get(str(day), []))
        cumulative_formation_hints.extend(formation_hint_by_day.get(str(day), []))

        seer_claimers = []
        medium_claimers = []
        not_seer_medium = []
        for p, cs in cumulative_claims.items():
            names = [_claim_name(c) for c in cs]
            if "Seer CO" in names:
                seer_claimers.append(p)
            if "Medium CO" in names:
                medium_claimers.append(p)
            if "Not Seer/Medium" in names or ("Not Seer" in names and "Not Medium" in names):
                not_seer_medium.append(p)
        hunter_claimers = _unique([h.get("player") for h in hunter_claims if h.get("player") in players])

        seer_claimers = _unique(seer_claimers)
        medium_claimers = _unique(medium_claimers)
        not_seer_medium = _unique(not_seer_medium)
        ability_claimers = set(seer_claimers) | set(medium_claimers) | set(hunter_claimers)
        gray_players = [p for p in players if p in alive and p not in ability_claimers]

        # Panda / confirmed-white approximation from public hard seer results.
        target_results = defaultdict(set)
        target_white_sources = defaultdict(set)
        for r in seer_results:
            t = r.get("target")
            res = r.get("result")
            src = r.get("seer")
            if t in players and res in ("human", "werewolf"):
                target_results[t].add(res)
                if res == "human" and src:
                    target_white_sources[t].add(src)
        pandas = [t for t, rs in target_results.items() if "human" in rs and "werewolf" in rs]
        confirmed_whites = [t for t, srcs in target_white_sources.items() if len(srcs) >= 2]

        alive_count = len(alive)
        remaining_ropes = max(0, (alive_count - 1) // 2)
        raw_formation = f"{len(seer_claimers)}-{len(medium_claimers)}"
        # If multiple players explicitly agree on a formation and claim parsing is incomplete,
        # let the policy see that formation.  Claims still remain confidence-based.
        formation = raw_formation
        if cumulative_formation_hints:
            # Prefer the dominant explicit village consensus (e.g. repeated "2-1 confirmed")
            # over raw claim counts, because translated logs often contain false-positive
            # CO snippets or unretracted old claims.
            counts = defaultdict(float)
            latest_by_form = {}
            for h in cumulative_formation_hints:
                f = str(h.get("formation", ""))
                if not f:
                    continue
                counts[f] += float(h.get("confidence", 0.7))
                latest_by_form[f] = h
            if counts:
                hinted = max(counts, key=counts.get)
                if counts[hinted] >= 1.2:  # roughly two weak mentions or one strong cluster
                    formation = hinted
        policy = classify_formation(formation)
        exposed_claim_slots = len(seer_claimers) + len(medium_claimers)
        gray_wolf_slots_upper_bound = max(0, role_counts.get("Werewolf", 0) - max(0, exposed_claim_slots - 2))

        states[str(day)] = {
            "day": str(day),
            "alive_players": sorted(alive),
            "dead_players": sorted(dead),
            "role_counts": role_counts,
            "seer_claimers": seer_claimers,
            "medium_claimers": medium_claimers,
            "hunter_claimers": hunter_claimers,
            "not_seer_medium_claimers": not_seer_medium,
            "ability_claimers": sorted(ability_claimers),
            "gray_players": gray_players,
            "formation": formation,
            "raw_claim_formation": raw_formation,
            "formation_hints": list(cumulative_formation_hints),
            "formation_mode": policy.get("mode"),
            "policy_priority": policy.get("priority", []),
            "seer_results": list(seer_results),
            "medium_results": list(medium_results),
            "executed": list(executed),
            "night_kills": list(night_kills),
            "sudden_deaths": list(sudden_deaths),
            "gj_days": list(gj_days),
            "hunter_claims": list(hunter_claims),
            "guard_reports": list(guard_reports),
            "claim_withdrawals": list(withdrawals),
            "pandas": pandas,
            "confirmed_whites": confirmed_whites,
            "remaining_ropes_estimate": remaining_ropes,
            "gray_wolf_slots_upper_bound": gray_wolf_slots_upper_bound,
            "notes": [
                "Confirmed whites are approximate: multi-human seer results, not absolute role confirmation.",
                "Pandas are based on hard seer-result extraction only.",
            ],
        }
    return states
