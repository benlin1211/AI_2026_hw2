"""Final consistency audit.

AP-oriented variant:
- Repair final role labels to legal role counts.
- Do not repair wolf_score back to only 2/3 high suspects.
- Keep an overbooked suspicious band for AP/ranking.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.solver import get_role_counts
from src.hard_constraints import repair_assignment_with_constraints


def audit_and_fix(
    players: List[str],
    assigned_roles: Dict[str, str],
    wolf_scores: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    hard_constraints: Dict[str, Any] | None = None,
) -> tuple[Dict[str, str], Dict[str, float]]:
    role_counts = get_role_counts(len(players))
    out_roles = {p: assigned_roles.get(p, "Villager") for p in players}
    out_scores = {p: float(wolf_scores.get(p, 0.20)) for p in players}

    fixed = set()
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            out_roles[p] = "Villager"
            out_scores[p] = 0.01
            fixed.add(p)

    impossible_wolves = set()
    if hard_constraints:
        out_roles, out_scores = repair_assignment_with_constraints(players, out_roles, out_scores, role_scores, hard_constraints)
        impossible_wolves = set(hard_constraints.get("impossible_wolves", []) or [])

    # Role labels remain legal. This protects role accuracy and avoids invalid
    # submissions. Overbooking is intentionally applied only to wolf_score below.
    def score_for_role(p: str, role: str) -> float:
        try:
            return float(role_scores.get(p, {}).get(role, 0.0))
        except Exception:
            return 0.0

    for role in ["Seer", "Medium", "Madman", "Hunter", "Werewolf"]:
        target_count = int(role_counts.get(role, 0))
        current = [p for p, r in out_roles.items() if r == role and p not in fixed]
        if len(current) > target_count:
            current_sorted = sorted(
                current,
                key=lambda p: score_for_role(p, role) + (out_scores.get(p, 0.0) if role == "Werewolf" else 0.0),
                reverse=True,
            )
            keep = set(current_sorted[:target_count])
            for p in current:
                if p not in keep:
                    out_roles[p] = "Villager"
        elif len(current) < target_count:
            need = target_count - len(current)
            candidates = [p for p in players if p not in fixed and out_roles.get(p) == "Villager"]
            candidates.sort(
                key=lambda p: score_for_role(p, role) + (out_scores.get(p, 0.0) if role == "Werewolf" else 0.0),
                reverse=True,
            )
            for p in candidates[:need]:
                out_roles[p] = role

    # AP-oriented score floor. Preserve top ww_count + buffer candidates, not just
    # the exact legal wolf count.
    ww_count = int(role_counts.get("Werewolf", 3 if len(players) >= 13 else 2))
    buffer = 2 if len(players) >= 13 else 1
    floors = [0.46, 0.42, 0.38, 0.32, 0.27, 0.22]
    ranked = sorted([p for p in players if p not in fixed], key=lambda p: out_scores.get(p, 0.0), reverse=True)
    for i, p in enumerate(ranked[: ww_count + buffer]):
        out_scores[p] = max(out_scores[p], floors[min(i, len(floors) - 1)])

    # Keep a weak tail floor for nearby candidates so AP is less brittle when the
    # true wolf is ranked 4th-7th by broad suspicion.
    for p in ranked[ww_count + buffer : ww_count + buffer + 3]:
        out_scores[p] = max(out_scores[p], 0.16)

    out_scores = {p: round(max(0.01, min(0.97, float(out_scores[p]))), 6) for p in players}
    return out_roles, out_scores
