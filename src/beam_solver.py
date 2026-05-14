"""Candidate-pruned top-K world solver.

This is the fast replacement for exhaustive role enumeration. It keeps wolf_score
as a marginal probability across plausible worlds instead of using a single hard
role assignment.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Tuple

from src.solver import get_role_counts
from src.hard_constraints import violates_constraints

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def _clamp(x: float, lo: float = 1e-6, hi: float = 0.999999) -> float:
    return max(lo, min(hi, float(x)))


def _safe_role(role_scores: Dict[str, Dict[str, float]], p: str, role: str, default: float = 0.05) -> float:
    try:
        return float(role_scores.get(p, {}).get(role, default))
    except Exception:
        return default


def _safe_wolf(wolf_scores: Dict[str, float], p: str, default: float = 0.20) -> float:
    try:
        return float(wolf_scores.get(p, default))
    except Exception:
        return default


def _claimers(objective_pack: Dict[str, Any], key: str) -> List[str]:
    return list(dict.fromkeys(objective_pack.get(key, []) or []))


def _constraints_from_pack(objective_pack: Dict[str, Any], objective_events: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if isinstance(objective_pack, dict) and objective_pack.get("hard_constraints"):
        return objective_pack.get("hard_constraints") or {}
    if isinstance(objective_events, dict) and objective_events.get("hard_constraints"):
        return objective_events.get("hard_constraints") or {}
    return {}


def _impossible_wolves(objective_pack: Dict[str, Any], objective_events: Dict[str, Any] | None = None) -> set:
    c = _constraints_from_pack(objective_pack, objective_events)
    return set(c.get("impossible_wolves", []) or objective_pack.get("impossible_wolves", []) or [])


def _top_by(players: List[str], score_fn, k: int) -> List[str]:
    return [p for _, p in sorted([(score_fn(p), p) for p in players], reverse=True)[:k]]


def build_candidate_sets(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    wolf_scores: Dict[str, float],
    wolf_prior: Dict[str, float],
    objective_pack: Dict[str, Any],
    guideline_scores: Dict[str, Any],
    max_role_candidates: int = 6,
    max_wolf_candidates: int = 9,
    max_hunter_candidates: int = 6,
) -> Dict[str, List[str]]:
    impossible = _impossible_wolves(objective_pack)
    non_gerd = [p for p in players if p != "Optimist Gerd" and p.strip().lower() != "gerd"]
    legal_wolf_pool = [p for p in non_gerd if p not in impossible]
    seer_claimers = [p for p in _claimers(objective_pack, "seer_claimers") if p in non_gerd]
    medium_claimers = [p for p in _claimers(objective_pack, "medium_claimers") if p in non_gerd]
    hunter_claimers = [p for p in _claimers(objective_pack, "hunter_claimers") if p in non_gerd]
    not_co = set(_claimers(objective_pack, "not_seer_medium_claimers"))

    def combined_wolf(p: str) -> float:
        guide = float((guideline_scores.get(p, {}) or {}).get("delta", 0.0))
        return 0.45 * _safe_wolf(wolf_scores, p) + 0.35 * float(wolf_prior.get(p, 0.20)) + 0.20 * _safe_role(role_scores, p, "Werewolf") + guide

    seer_candidates = list(seer_claimers)
    if len(seer_candidates) < 1:
        seer_candidates += _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Seer"), max_role_candidates)
    else:
        # Add at most one fallback in case parser missed the true claim.
        extras = [p for p in _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Seer"), max_role_candidates) if p not in seer_candidates and p not in not_co]
        seer_candidates += extras[:1]

    medium_candidates = list(medium_claimers)
    if len(medium_candidates) < 1:
        medium_candidates += _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Medium"), max_role_candidates)
    else:
        extras = [p for p in _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Medium"), max_role_candidates) if p not in medium_candidates and p not in not_co]
        medium_candidates += extras[:1]

    fake_claimers = list(dict.fromkeys(seer_claimers + medium_claimers))
    madman_candidates = list(fake_claimers)
    madman_candidates += _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Madman") + (0.05 if p in fake_claimers else 0.0), max_role_candidates)

    hunter_candidates = list(hunter_claimers)
    hunter_candidates += _top_by(non_gerd, lambda p: _safe_role(role_scores, p, "Hunter"), max_hunter_candidates)

    wolf_candidates = _top_by(legal_wolf_pool, combined_wolf, max_wolf_candidates)
    # Include hard-black targets even if not in top candidates.
    for r in objective_pack.get("hard_results", []):
        if r.get("result") == "werewolf" and r.get("target") in legal_wolf_pool and r.get("target") not in wolf_candidates:
            wolf_candidates.append(r.get("target"))

    def uniq(xs):
        return list(dict.fromkeys([x for x in xs if x in non_gerd]))

    return {
        "Seer": uniq(seer_candidates)[:max_role_candidates + 2],
        "Medium": uniq(medium_candidates)[:max_role_candidates + 2],
        "Madman": uniq(madman_candidates)[:max_role_candidates + len(fake_claimers)],
        "Hunter": uniq(hunter_candidates)[:max_hunter_candidates + len(hunter_claimers)],
        "Werewolf": uniq(wolf_candidates)[:max_wolf_candidates + 3],
    }


def _event_consistency_score(assignment: Dict[str, str], objective_events: Dict[str, Any]) -> float:
    score = 0.0
    for r in objective_events.get("seer_results", []):
        seer = r.get("seer")
        target = r.get("target")
        res = r.get("result")
        conf = float(r.get("confidence", 0.70))
        if seer not in assignment or target not in assignment:
            continue
        if assignment.get(seer) == "Seer":
            if res == "werewolf":
                score += 0.80 * conf if assignment.get(target) == "Werewolf" else -1.20 * conf
            elif res == "human":
                score += 0.45 * conf if assignment.get(target) != "Werewolf" else -1.00 * conf
        elif assignment.get(seer) in {"Werewolf", "Madman"}:
            # Fake result can say anything. Slight bonus for blacking villagers, but weak.
            if res == "werewolf" and assignment.get(target) != "Werewolf":
                score += 0.08 * conf
    for r in objective_events.get("medium_results", []):
        med = r.get("medium")
        target = r.get("target")
        res = r.get("result")
        conf = float(r.get("confidence", 0.65))
        if med not in assignment or target not in assignment:
            continue
        if assignment.get(med) == "Medium":
            if res == "werewolf":
                score += 0.75 * conf if assignment.get(target) == "Werewolf" else -1.10 * conf
            elif res == "human":
                score += 0.40 * conf if assignment.get(target) != "Werewolf" else -1.00 * conf
    return score


def _world_log_score(
    assignment: Dict[str, str],
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    wolf_scores: Dict[str, float],
    wolf_prior: Dict[str, float],
    objective_pack: Dict[str, Any],
    objective_events: Dict[str, Any],
    guideline_scores: Dict[str, Any],
) -> float:
    total = 0.0
    seer_claimers = set(objective_pack.get("seer_claimers", []) or [])
    medium_claimers = set(objective_pack.get("medium_claimers", []) or [])
    fake_claimers = seer_claimers | medium_claimers
    not_co = set(objective_pack.get("not_seer_medium_claimers", []) or [])
    hard_constraints = _constraints_from_pack(objective_pack, objective_events)

    for p in players:
        role = assignment.get(p, "Villager")
        if hard_constraints and violates_constraints(p, role, hard_constraints):
            total -= 50.0
            continue
        
        role = assignment.get(p, "Villager")
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            total += 0.0 if role == "Villager" else -20.0
            continue

        rp = _clamp(_safe_role(role_scores, p, role, 0.05))
        total += math.log(rp)

        if role == "Werewolf":
            wp = 0.55 * _safe_wolf(wolf_scores, p, 0.20) + 0.45 * float(wolf_prior.get(p, 0.20))
            wp += float((guideline_scores.get(p, {}) or {}).get("delta", 0.0))
            total += math.log(_clamp(wp))

        if p in seer_claimers:
            if role == "Seer":
                total += 0.45
            elif role in {"Werewolf", "Madman"}:
                total += 0.15
            else:
                total -= 0.50
        if p in medium_claimers:
            if role == "Medium":
                total += 0.45
            elif role in {"Werewolf", "Madman"}:
                total += 0.12
            else:
                total -= 0.50
        if p in not_co and role in {"Seer", "Medium"}:
            total -= 0.70
        if p in fake_claimers and role in {"Villager", "Hunter"}:
            total -= 0.40

    total += _event_consistency_score(assignment, objective_events)
    return total


def solve_topk_worlds(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    wolf_scores: Dict[str, float],
    wolf_prior: Dict[str, float],
    objective_pack: Dict[str, Any],
    objective_events: Dict[str, Any],
    guideline_scores: Dict[str, Any],
    top_k: int = 256,
    max_role_candidates: int = 6,
    max_wolf_candidates: int = 9,
    max_hunter_candidates: int = 6,
) -> Dict[str, Any]:
    role_counts = get_role_counts(len(players))
    ww_count = int(role_counts.get("Werewolf", 3 if len(players) >= 13 else 2))
    has_madman = role_counts.get("Madman", 0) > 0
    has_hunter = role_counts.get("Hunter", 0) > 0
    gerd_names = {p for p in players if p == "Optimist Gerd" or p.strip().lower() == "gerd"}
    hard_constraints = _constraints_from_pack(objective_pack, objective_events)
    impossible = set(hard_constraints.get("impossible_wolves", []) or objective_pack.get("impossible_wolves", []) or [])

    cand = build_candidate_sets(
        players, role_scores, wolf_scores, wolf_prior, objective_pack, guideline_scores,
        max_role_candidates=max_role_candidates,
        max_wolf_candidates=max_wolf_candidates,
        max_hunter_candidates=max_hunter_candidates,
    )

    worlds: List[Tuple[float, Dict[str, str]]] = []
    seers = cand["Seer"] or [p for p in players if p not in gerd_names]
    mediums = cand["Medium"] or [p for p in players if p not in gerd_names]
    madmen = cand["Madman"] if has_madman else [None]
    hunters = cand["Hunter"] if has_hunter else [None]
    wolf_cands_base = cand["Werewolf"]

    for seer in seers:
        for medium in mediums:
            if medium == seer:
                continue
            for madman in madmen:
                if madman is not None and madman in {seer, medium}:
                    continue
                for hunter in hunters:
                    used = {seer, medium}
                    if madman is not None:
                        used.add(madman)
                    if hunter is not None:
                        if hunter in used:
                            continue
                        used.add(hunter)
                    wolf_cands = [p for p in wolf_cands_base if p not in used and p not in gerd_names and p not in impossible]
                    if len(wolf_cands) < ww_count:
                        # Fallback: append all legal players by wolf prior.
                        extras = [p for p in players if p not in used and p not in gerd_names and p not in impossible and p not in wolf_cands]
                        extras.sort(key=lambda p: float(wolf_prior.get(p, 0.20)), reverse=True)
                        wolf_cands = (wolf_cands + extras)[:max(max_wolf_candidates + 3, ww_count)]
                    for wolves in itertools.combinations(wolf_cands, ww_count):
                        assignment = {p: "Villager" for p in players}
                        for g in gerd_names:
                            assignment[g] = "Villager"
                        assignment[seer] = "Seer"
                        assignment[medium] = "Medium"
                        if madman is not None:
                            assignment[madman] = "Madman"
                        if hunter is not None:
                            assignment[hunter] = "Hunter"
                        for w in wolves:
                            assignment[w] = "Werewolf"
                        score = _world_log_score(assignment, players, role_scores, wolf_scores, wolf_prior, objective_pack, objective_events, guideline_scores)
                        worlds.append((score, assignment))

    if not worlds:
        fallback = {p: "Villager" for p in players}
        for p in _top_by([p for p in players if p not in gerd_names and p not in impossible], lambda x: wolf_prior.get(x, 0.20), ww_count):
            fallback[p] = "Werewolf"
        return {"worlds": [{"logprob": 0.0, "roles": fallback}], "role_marginals": _marginals(players, [(0.0, fallback)]), "candidate_sets": cand}

    worlds.sort(key=lambda x: x[0], reverse=True)
    worlds = worlds[:top_k]
    return {
        "worlds": [{"logprob": round(s, 6), "roles": a} for s, a in worlds],
        "role_marginals": _marginals(players, worlds),
        "candidate_sets": cand,
    }


def _marginals(players: List[str], worlds: List[Tuple[float, Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
    if not worlds:
        return {p: {r: 0.0 for r in VALID_ROLES} for p in players}
    max_log = max(s for s, _ in worlds)
    weights = [math.exp(max(-50.0, s - max_log)) for s, _ in worlds]
    z = sum(weights) or 1.0
    out = {p: {r: 0.0 for r in VALID_ROLES} for p in players}
    for weight, (_, roles) in zip(weights, worlds):
        w = weight / z
        for p in players:
            role = roles.get(p, "Villager")
            out[p][role] = out[p].get(role, 0.0) + w
    return {p: {r: round(float(v), 6) for r, v in probs.items()} for p, probs in out.items()}


def best_assignment_from_worlds(world_result: Dict[str, Any], players: List[str]) -> Dict[str, str]:
    worlds = world_result.get("worlds", [])
    if worlds:
        roles = worlds[0].get("roles", {})
        return {p: roles.get(p, "Villager") for p in players}
    marg = world_result.get("role_marginals", {})
    out = {}
    for p in players:
        probs = marg.get(p, {})
        out[p] = max(VALID_ROLES, key=lambda r: probs.get(r, 0.0)) if probs else "Villager"
    return out
