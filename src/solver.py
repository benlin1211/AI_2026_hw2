from typing import Dict, List, Any, Optional

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def get_role_counts(num_players: int) -> Dict[str, int]:
    return {
        "Werewolf": 3 if num_players >= 13 else 2,
        "Seer": 1,
        "Medium": 1,
        "Madman": 1 if num_players >= 11 else 0,
        "Hunter": 1 if num_players >= 11 else 0,
    }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def safe_score(role_scores: Dict[str, Dict[str, float]], player: str, role: str, default: float = 0.0) -> float:
    try:
        return float(role_scores.get(player, {}).get(role, default))
    except Exception:
        return default


def get_claims(parsed: Dict[str, Any], player: str) -> List[Any]:
    claims = parsed.get("all_claims", {}).get(player, [])
    return claims if isinstance(claims, list) else []


def claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def has_claim(parsed: Dict[str, Any], player: str, claim_keyword: str) -> bool:
    return any(claim_keyword in claim_name(c) for c in get_claims(parsed, player))


def count_claimers(parsed: Dict[str, Any], claim_keyword: str) -> int:
    n = 0
    for _, claims in parsed.get("all_claims", {}).items():
        if any(claim_keyword in claim_name(c) for c in claims):
            n += 1
    return n


def candidate_score_for_role(player: str, role: str, role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> float:
    score = safe_score(role_scores, player, role, default=0.0)

    if has_claim(parsed, player, "Seer CO"):
        if role == "Seer":
            score += 0.28
        elif role == "Madman":
            score += 0.12
        elif role == "Werewolf":
            score += 0.08
        else:
            score -= 0.10

    if has_claim(parsed, player, "Medium CO"):
        if role == "Medium":
            score += 0.28
        elif role == "Werewolf":
            score += 0.08
        elif role == "Madman":
            score += 0.06
        else:
            score -= 0.10

    if has_claim(parsed, player, "Not Seer/Medium"):
        if role in ("Seer", "Medium"):
            score -= 0.35
        elif role == "Villager":
            score += 0.06

    if has_claim(parsed, player, "Not Seer") and role == "Seer":
        score -= 0.25
    if has_claim(parsed, player, "Not Medium") and role == "Medium":
        score -= 0.25

    return score


def assign_best_candidate(role: str, count: int, remaining: set, assigned: Dict[str, str], role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> None:
    if count <= 0 or not remaining:
        return
    candidates = []
    for p in remaining:
        score = candidate_score_for_role(p, role, role_scores, parsed)
        candidates.append((score, p))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, p in candidates[:count]:
        assigned[p] = role
        remaining.discard(p)


def constrained_role_assignment(players: List[str], role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> Dict[str, str]:
    role_counts = get_role_counts(len(players))
    assigned: Dict[str, str] = {}
    remaining = set(players)

    # Ability roles first: this preserves role accuracy better than pushing Werewolf too early.
    for role in ["Seer", "Medium", "Madman", "Hunter", "Werewolf"]:
        assign_best_candidate(role, role_counts.get(role, 0), remaining, assigned, role_scores, parsed)

    for p in remaining:
        assigned[p] = "Villager"
    for p in players:
        assigned.setdefault(p, "Villager")
    return assigned


def _formation_prior(player: str, parsed: Dict[str, Any], objective_pack: Optional[Dict[str, Any]]) -> float:
    if player == "Optimist Gerd" or player.strip().lower() == "gerd":
        return 0.01
    obj = objective_pack or {}
    seer_claimers = obj.get("seer_claimers", []) or []
    medium_claimers = obj.get("medium_claimers", []) or []
    gray = set(obj.get("gray_players", []) or [])
    prior = 0.20
    if player in gray:
        prior += 0.02
    if player in seer_claimers:
        prior += 0.07 if len(seer_claimers) >= 2 else -0.05
    if player in medium_claimers:
        prior += 0.06 if len(medium_claimers) >= 2 else -0.07
    if has_claim(parsed, player, "Not Seer/Medium"):
        prior -= 0.02
    return _clamp(prior, 0.03, 0.85)


def normalize_wolf_scores(
    players: List[str],
    wolf_scores: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    assigned_roles: Dict[str, str],
    parsed: Dict[str, Any],
    stepwise_prior: Optional[Dict[str, float]] = None,
    objective_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    MaKTO-inspired v2 score fusion.

    The role assignment is only a weak signal. This prevents the prior failure mode
    where a true wolf assigned as Medium/Hunter gets compressed to 0.01.
    """
    scores: Dict[str, float] = {}
    stepwise_prior = stepwise_prior or {}
    role_counts = (objective_pack or {}).get("role_counts", get_role_counts(len(players)))
    ww_count = int(role_counts.get("Werewolf", 3 if len(players) >= 13 else 2))
    seer_claimers = count_claimers(parsed, "Seer CO")
    medium_claimers = count_claimers(parsed, "Medium CO")

    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            scores[p] = 0.01
            continue

        try:
            llm_wolf = float(wolf_scores.get(p, 0.20))
        except Exception:
            llm_wolf = 0.20
        prior = float(stepwise_prior.get(p, 0.20))
        role_wolf = safe_score(role_scores, p, "Werewolf", default=0.0)
        formation = _formation_prior(p, parsed, objective_pack)

        # Main blend: wolf agent and stepwise heuristic dominate; role solver is weak.
        score = 0.48 * llm_wolf + 0.27 * prior + 0.13 * formation + 0.12 * role_wolf

        assigned = assigned_roles.get(p)
        if assigned == "Werewolf":
            score += 0.035
        elif assigned == "Madman":
            score -= 0.015
        elif assigned == "Seer":
            # Only lower strongly if claim is explicit and uncontested.
            if has_claim(parsed, p, "Seer CO") and seer_claimers <= 1:
                score -= 0.055
            else:
                score -= 0.005
        elif assigned == "Medium":
            if has_claim(parsed, p, "Medium CO") and medium_claimers <= 1:
                score -= 0.055
            else:
                score -= 0.005
        elif assigned == "Hunter":
            score -= 0.005

        # Hard result adjustment. Soft reads are already in prior.
        hard_results = (objective_pack or {}).get("hard_results", [])
        if not hard_results:
            hard_results = parsed.get("all_divinations", [])
        black_count = 0
        white_count = 0
        for r in hard_results:
            if not isinstance(r, dict) or r.get("target") != p:
                continue
            if r.get("result") == "werewolf":
                black_count += 1
            elif r.get("result") == "human":
                white_count += 1
        score += 0.11 * black_count
        score -= 0.045 * white_count
        if black_count > 0 and white_count > 0:
            score = min(0.82, max(0.42, score))

        scores[p] = _clamp(score, 0.05, 0.97)

    # Mild rank calibration for AP/ranking metrics. Avoid over-forcing villagers into wolves.
    ranked = sorted([p for p in players if scores.get(p, 0) > 0.011], key=lambda x: scores[x], reverse=True)
    for i, p in enumerate(ranked):
        if i < ww_count:
            scores[p] = max(scores[p], 0.32 - 0.02 * i)
        elif i < ww_count + 3:
            scores[p] = max(scores[p], 0.15)
        scores[p] = _clamp(scores[p], 0.05, 0.97)

    return {p: round(float(scores[p]), 6) for p in players}
