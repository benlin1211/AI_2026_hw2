"""Hard public-rule constraints for deterministic Werewolf inference.

This module only encodes facts that are safe across games:
- Gerd is a non-wolf baseline NPC/opening victim in this dataset family.
- A player killed by the werewolves at night is human, therefore cannot be an actual Werewolf.
- Role counts remain fixed by player count.

Do not put soft reads, Seer credibility, or Medium credibility here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from src.solver import get_role_counts

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def _is_gerd(player: str) -> bool:
    return player == "Optimist Gerd" or str(player).strip().lower() == "gerd"


def _uniq(xs: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in xs if x]))


def derive_hard_constraints(
    players: List[str],
    objective_events: Dict[str, Any],
    objective_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    player_set = set(players)
    night_killed = _uniq([
        x.get("player")
        for x in objective_events.get("night_kills", [])
        if isinstance(x, dict) and x.get("player") in player_set
    ])
    executed = _uniq([
        x.get("player")
        for x in objective_events.get("executed", [])
        if isinstance(x, dict) and x.get("player") in player_set
    ])
    sudden = _uniq([
        x.get("player")
        for x in objective_events.get("sudden_deaths", [])
        if isinstance(x, dict) and x.get("player") in player_set
    ])
    gerds = _uniq([p for p in players if _is_gerd(p)])

    impossible_wolves = _uniq(gerds + night_killed)
    forbidden_roles: Dict[str, List[str]] = {p: ["Werewolf"] for p in impossible_wolves}
    forced_roles: Dict[str, str] = {p: "Villager" for p in gerds}

    return {
        "role_counts": (objective_pack or {}).get("role_counts", get_role_counts(len(players))),
        "forced_roles": forced_roles,
        "forbidden_roles": forbidden_roles,
        "impossible_wolves": impossible_wolves,
        "night_killed_humans": night_killed,
        "executed": executed,
        "sudden_deaths": sudden,
        "notes": [
            "Night-killed players are human under the public rules, so they cannot be actual Werewolves.",
            "Executed players are not constrained by alignment unless a trusted Medium result is separately established.",
            "Seer human means non-Werewolf, not necessarily Villager; this module does not force exact non-wolf roles from Seer results.",
        ],
    }


def attach_constraints_to_pack(objective_pack: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(objective_pack or {})
    out["hard_constraints"] = constraints
    out["impossible_wolves"] = list(constraints.get("impossible_wolves", []))
    out["forbidden_roles"] = dict(constraints.get("forbidden_roles", {}))
    out["forced_roles"] = dict(constraints.get("forced_roles", {}))
    return out


def _renormalize(probs: Dict[str, float]) -> Dict[str, float]:
    cleaned = {r: max(0.0, float(probs.get(r, 0.0))) for r in VALID_ROLES}
    s = sum(cleaned.values())
    if s <= 0:
        cleaned = {"Villager": 1.0, "Werewolf": 0.0, "Seer": 0.0, "Medium": 0.0, "Madman": 0.0, "Hunter": 0.0}
    else:
        cleaned = {k: v / s for k, v in cleaned.items()}
    return cleaned


def apply_hard_constraints_to_role_scores(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    constraints: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    forced = dict(constraints.get("forced_roles", {}))
    forbidden = {p: set(rs or []) for p, rs in (constraints.get("forbidden_roles", {}) or {}).items()}
    out: Dict[str, Dict[str, float]] = {}
    for p in players:
        probs = dict(role_scores.get(p, {}))
        if p in forced:
            role = forced[p]
            out[p] = {r: (1.0 if r == role else 0.0) for r in VALID_ROLES}
            continue
        for role in forbidden.get(p, set()):
            probs[role] = 0.0
        out[p] = _renormalize(probs)
    return out


def apply_hard_constraints_to_wolf_scores(
    players: List[str],
    wolf_scores: Dict[str, float],
    constraints: Dict[str, Any],
    cap: float = 0.015,
) -> Dict[str, float]:
    impossible = set(constraints.get("impossible_wolves", []) or [])
    out = {p: max(0.0, min(1.0, float(wolf_scores.get(p, 0.20)))) for p in players}
    for p in impossible:
        if p in out:
            out[p] = min(out[p], cap)
    return out


def violates_constraints(player: str, role: str, constraints: Dict[str, Any]) -> bool:
    forced = constraints.get("forced_roles", {}) or {}
    if player in forced and role != forced[player]:
        return True
    forbidden = constraints.get("forbidden_roles", {}) or {}
    return role in set(forbidden.get(player, []) or [])


def repair_assignment_with_constraints(
    players: List[str],
    assigned_roles: Dict[str, str],
    wolf_scores: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    constraints: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Repair final labels while preserving legal role counts.

    This is intentionally deterministic and local: it only fixes illegal labels and
    then fills missing role slots using role-score + wolf-score ranking.
    """
    role_counts = dict(constraints.get("role_counts") or get_role_counts(len(players)))
    out_roles = {p: assigned_roles.get(p, "Villager") for p in players}
    out_scores = {p: float(wolf_scores.get(p, 0.20)) for p in players}
    forced = dict(constraints.get("forced_roles", {}) or {})
    forbidden = {p: set(rs or []) for p, rs in (constraints.get("forbidden_roles", {}) or {}).items()}
    impossible = set(constraints.get("impossible_wolves", []) or [])

    fixed: Set[str] = set()
    for p, role in forced.items():
        if p in out_roles:
            out_roles[p] = role
            fixed.add(p)

    for p in players:
        if out_roles[p] in forbidden.get(p, set()):
            out_roles[p] = "Villager"

    def score_for_role(p: str, role: str) -> float:
        if p in fixed and out_roles.get(p) != role:
            return -10**9
        if role in forbidden.get(p, set()):
            return -10**9
        try:
            base = float(role_scores.get(p, {}).get(role, 0.0))
        except Exception:
            base = 0.0
        if role == "Werewolf":
            base += float(out_scores.get(p, 0.0))
        return base

    # Remove excess role labels first.
    for role in ["Seer", "Medium", "Madman", "Hunter", "Werewolf"]:
        target_count = int(role_counts.get(role, 0))
        current = [p for p, r in out_roles.items() if r == role and p not in fixed]
        if len(current) > target_count:
            current_sorted = sorted(current, key=lambda p: score_for_role(p, role), reverse=True)
            keep = set(current_sorted[:target_count])
            for p in current:
                if p not in keep:
                    out_roles[p] = "Villager"

    # Fill missing role labels, excluding forbidden assignments.
    for role in ["Seer", "Medium", "Madman", "Hunter", "Werewolf"]:
        target_count = int(role_counts.get(role, 0))
        current = [p for p, r in out_roles.items() if r == role]
        need = target_count - len(current)
        if need <= 0:
            continue
        candidates = [
            p for p in players
            if p not in fixed and out_roles.get(p) == "Villager" and role not in forbidden.get(p, set())
        ]
        candidates.sort(key=lambda p: score_for_role(p, role), reverse=True)
        for p in candidates[:need]:
            out_roles[p] = role

    for p in impossible:
        if p in out_scores:
            out_scores[p] = min(out_scores[p], 0.015)
            if out_roles.get(p) == "Werewolf":
                out_roles[p] = "Villager"

    out_scores = {p: round(max(0.01, min(0.97, float(out_scores[p]))), 6) for p in players}
    return out_roles, out_scores


def hard_constraint_report(players: List[str], assignment: Dict[str, str], constraints: Dict[str, Any]) -> Dict[str, Any]:
    violations = []
    for p in players:
        r = assignment.get(p)
        if violates_constraints(p, r, constraints):
            violations.append({"player": p, "role": r})
    return {"ok": not violations, "violations": violations}
