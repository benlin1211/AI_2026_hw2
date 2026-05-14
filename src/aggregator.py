"""Final role and wolf-score aggregation for fast v3.

AP-oriented variant:
- Keep final role assignment constrained by legal role counts.
- Treat wolf_score as an overbooked suspicion/ranking score.
- Do not let fixed 2/3-wolf world marginals erase broad wolf suspicion.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.solver import get_role_counts

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _safe(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def _safe_role(role_scores: Dict[str, Dict[str, float]], p: str, role: str, default: float = 0.0) -> float:
    try:
        return float(role_scores.get(p, {}).get(role, default))
    except Exception:
        return default


def _hard_constraints(objective_pack: Dict[str, Any], objective_events: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(objective_pack, dict) and objective_pack.get("hard_constraints"):
        return objective_pack.get("hard_constraints") or {}
    if isinstance(objective_events, dict) and objective_events.get("hard_constraints"):
        return objective_events.get("hard_constraints") or {}
    return {}



def aggregate_wolf_scores(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    base_wolf_scores: Dict[str, float],
    wolf_prior: Dict[str, float],
    guideline_scores: Dict[str, Dict[str, Any]],
    world_result: Dict[str, Any],
    objective_pack: Dict[str, Any],
    objective_events: Dict[str, Any],
    assigned_roles: Dict[str, str],
) -> Dict[str, float]:
    """Aggregate continuous Werewolf suspicion scores.

    Important distinction:
    - role assignment should obey the real game counts, so only 2/3 players are
      assigned role == "Werewolf".
    - wolf_score is a ranking score. It is allowed to keep 4/5 players in a
      suspicious range when AP/recall matters.
    """
    role_counts = objective_pack.get("role_counts", get_role_counts(len(players)))
    ww_count = int(role_counts.get("Werewolf", 3 if len(players) >= 13 else 2))
    role_marginals = world_result.get("role_marginals", {}) or {}
    constraints = _hard_constraints(objective_pack, objective_events)
    impossible_wolves = set(constraints.get("impossible_wolves", []) or objective_pack.get("impossible_wolves", []) or [])

    scores: Dict[str, float] = {}
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            scores[p] = 0.01
            continue
        if p in impossible_wolves:
            scores[p] = 0.015
            continue

        world_wolf = _safe(role_marginals.get(p, {}), "Werewolf", None) if p in role_marginals else None
        if world_wolf is None:
            world_wolf = _safe(role_scores.get(p, {}), "Werewolf", 0.18)

        # Unconstrained / broad suspicion signals. These should dominate AP-style
        # wolf_score more than exact legal-world marginals.
        stepwise = float(wolf_prior.get(p, 0.20))
        guide_delta = float((guideline_scores.get(p, {}) or {}).get("delta", 0.0))
        guide_component = _clamp(0.20 + guide_delta, 0.03, 0.90)
        base = float(base_wolf_scores.get(p, 0.20))
        role_wolf = _safe_role(role_scores, p, "Werewolf", 0.0)

        # AP-oriented overbooked suspicion blend.
        # world_wolf is useful but intentionally not dominant: top-K worlds have
        # exact 2/3 wolves and therefore under-score the 4th/5th suspicious slot.
        score = (
            0.18 * world_wolf
            + 0.42 * stepwise
            + 0.18 * base
            + 0.12 * guide_component
            + 0.10 * role_wolf
        )

        # Hard result adjustment remains moderate because parser hard-results can
        # still contain noise and world solver already consumed them.
        hard_results = list(objective_events.get("seer_results", [])) or list(objective_pack.get("hard_results", []))
        black_count = sum(
            1
            for r in hard_results
            if isinstance(r, dict) and r.get("target") == p and r.get("result") == "werewolf"
        )
        white_count = sum(
            1
            for r in hard_results
            if isinstance(r, dict) and r.get("target") == p and r.get("result") == "human"
        )
        score += 0.045 * black_count
        score -= 0.020 * white_count
        if black_count and white_count:
            score = max(0.42, min(0.82, score))

        # Role assignment is deliberately weak. Madman-like or ability-like final
        # labels should not erase Werewolf suspicion, because role assignment is a
        # separate constrained problem.
        assigned = assigned_roles.get(p)
        if assigned == "Werewolf":
            score += 0.025
        elif assigned == "Madman":
            score -= 0.005
        elif assigned in {"Seer", "Medium", "Hunter"}:
            score -= 0.005

        # Prior protection: preserve broad suspicion even if fixed-count worlds
        # or final role labels exclude this player from the exact wolf set.
        if stepwise >= 0.50:
            score = max(score, 0.34)
        if base >= 0.45:
            score = max(score, 0.30)
        if stepwise >= 0.45 and guide_delta > 0:
            score = max(score, 0.32)
        if stepwise >= 0.55 or (stepwise >= 0.48 and base >= 0.35):
            score = max(score, 0.36)

        scores[p] = _clamp(score, 0.03, 0.97)

    # Ranking calibration for AP-style metrics: overbook the suspicious band.
    # For a 3-wolf game, keep top 5 suspicious players separated instead of only
    # protecting top 3. For a 2-wolf game, keep top 3 suspicious players alive.
    ranked = sorted([p for p in players if scores.get(p, 0.0) > 0.011 and p not in impossible_wolves], key=lambda x: scores[x], reverse=True)
    buffer = 2 if len(players) >= 13 else 1
    floors = [0.46, 0.42, 0.38, 0.32, 0.27, 0.22]
    for i, p in enumerate(ranked):
        if i < ww_count + buffer:
            scores[p] = max(scores[p], floors[min(i, len(floors) - 1)])
        elif i < ww_count + buffer + 3:
            scores[p] = max(scores[p], 0.16)
        scores[p] = _clamp(scores[p], 0.03, 0.97)

    return {p: round(float(scores[p]), 6) for p in players}


def assign_roles_from_marginals(
    players: List[str],
    world_result: Dict[str, Any],
    fallback_assignment: Dict[str, str],
) -> Dict[str, str]:
    """Assign final roles from worlds.

    This remains a constrained role assignment. Do not overbook role == Werewolf
    here; overbooking is only for wolf_score.
    """
    worlds = world_result.get("worlds", [])
    if worlds:
        best = worlds[0].get("roles", {})
        out = {p: best.get(p, fallback_assignment.get(p, "Villager")) for p in players}
    else:
        marg = world_result.get("role_marginals", {}) or {}
        out = {}
        for p in players:
            probs = marg.get(p, {})
            out[p] = max(VALID_ROLES, key=lambda r: probs.get(r, 0.0)) if probs else fallback_assignment.get(p, "Villager")
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            out[p] = "Villager"
    return out
