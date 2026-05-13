from typing import Dict, List, Any


VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def get_role_counts(num_players: int) -> Dict[str, int]:
    return {
        "Werewolf": 3 if num_players >= 13 else 2,
        "Seer": 1,
        "Medium": 1,
        "Madman": 1 if num_players >= 11 else 0,
        "Hunter": 1 if num_players >= 11 else 0,
    }


def safe_score(
    role_scores: Dict[str, Dict[str, float]],
    player: str,
    role: str,
    default: float = 0.0,
) -> float:
    try:
        return float(role_scores.get(player, {}).get(role, default))
    except Exception:
        return default


def get_claims(parsed: Dict[str, Any], player: str) -> List[Any]:
    claims = parsed.get("all_claims", {}).get(player, [])
    if not isinstance(claims, list):
        return []
    return claims


def claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def has_claim(parsed: Dict[str, Any], player: str, claim_keyword: str) -> bool:
    claims = get_claims(parsed, player)
    return any(claim_keyword in claim_name(c) for c in claims)


def count_claimers(parsed: Dict[str, Any], claim_keyword: str) -> int:
    n = 0
    for _, claims in parsed.get("all_claims", {}).items():
        if any(claim_keyword in claim_name(c) for c in claims):
            n += 1
    return n


def candidate_score_for_role(
    player: str,
    role: str,
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> float:
    """
    Conservative role score.
    Do not over-lock claimers into true Seer/Medium.
    Fake claimers are often Werewolf or Madman.
    """
    score = safe_score(role_scores, player, role, default=0.0)

    if has_claim(parsed, player, "Seer CO"):
        if role == "Seer":
            score += 0.25
        elif role == "Madman":
            score += 0.12
        elif role == "Werewolf":
            score += 0.08
        elif role in ("Villager", "Hunter", "Medium"):
            score -= 0.12

    if has_claim(parsed, player, "Medium CO"):
        if role == "Medium":
            score += 0.22
        elif role == "Werewolf":
            score += 0.08
        elif role == "Madman":
            score += 0.06
        elif role in ("Villager", "Hunter", "Seer"):
            score -= 0.12

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


def assign_best_candidate(
    role: str,
    count: int,
    remaining: set,
    assigned: Dict[str, str],
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> None:
    if count <= 0 or not remaining:
        return

    candidates = []
    for p in remaining:
        score = candidate_score_for_role(
            player=p,
            role=role,
            role_scores=role_scores,
            parsed=parsed,
        )
        candidates.append((score, p))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, p in candidates[:count]:
        assigned[p] = role
        remaining.discard(p)


def constrained_role_assignment(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> Dict[str, str]:
    """
    Stable greedy assignment.

    Priority intentionally keeps role slots simple.
    This tends to be more robust than over-constrained global search when
    role_scores are noisy.
    """
    role_counts = get_role_counts(len(players))

    assigned: Dict[str, str] = {}
    remaining = set(players)

    # 1. Seer first.
    # In multi-claim games, putting Seer before Medium avoids Medium claimers
    # stealing too much solver priority.
    assign_best_candidate(
        role="Seer",
        count=role_counts.get("Seer", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    # 2. Medium.
    assign_best_candidate(
        role="Medium",
        count=role_counts.get("Medium", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    # 3. Werewolves.
    assign_best_candidate(
        role="Werewolf",
        count=role_counts.get("Werewolf", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    # 4. Madman.
    assign_best_candidate(
        role="Madman",
        count=role_counts.get("Madman", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    # 5. Hunter.
    assign_best_candidate(
        role="Hunter",
        count=role_counts.get("Hunter", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    for p in remaining:
        assigned[p] = "Villager"

    for p in players:
        if p not in assigned:
            assigned[p] = "Villager"

    return assigned


def normalize_wolf_scores(
    players: List[str],
    wolf_scores: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    assigned_roles: Dict[str, str],
    parsed: Dict[str, Any],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    seer_claimers = count_claimers(parsed, "Seer CO")
    medium_claimers = count_claimers(parsed, "Medium CO")

    for p in players:
        try:
            base = float(wolf_scores.get(p, 0.20))
        except Exception:
            base = 0.20

        role_wolf = safe_score(role_scores, p, "Werewolf", default=0.0)

        # More trust in wolf_agent ranking.
        score = 0.75 * base + 0.25 * role_wolf

        assigned = assigned_roles.get(p)

        # Smaller assignment adjustment than before.
        # Do not let role solver dominate AP.
        if assigned == "Werewolf":
            score += 0.06

        if assigned == "Seer":
            if seer_claimers <= 1:
                score -= 0.16
            else:
                score -= 0.01

        if assigned == "Medium":
            if medium_claimers <= 1:
                score -= 0.16
            else:
                score -= 0.01

        if assigned == "Madman":
            # Madman is not Werewolf.
            score -= 0.02

        # Claimers in contested formations are often either true role, wolf, or madman.
        # Do not over-lower them.
        if has_claim(parsed, p, "Seer CO") and seer_claimers >= 2:
            score += 0.02

        if has_claim(parsed, p, "Medium CO") and medium_claimers >= 2:
            score += 0.02

        received = [
            r for r in parsed.get("all_divinations", [])
            if isinstance(r, dict) and r.get("target") == p
        ]

        black_count = sum(
            1 for r in received
            if r.get("result") == "werewolf"
        )
        white_count = sum(
            1 for r in received
            if r.get("result") == "human"
        )

        if black_count > 0:
            score += 0.08 * black_count

        if white_count > 0:
            score -= 0.04 * white_count

        soft_received = [
            r for r in parsed.get("all_soft_reads", [])
            if isinstance(r, dict) and r.get("target") == p
        ]

        likely_wolf_reads = sum(
            1 for r in soft_received
            if r.get("read") == "likely_werewolf"
        )
        likely_villager_reads = sum(
            1 for r in soft_received
            if r.get("read") == "likely_villager"
        )

        score += min(0.06, 0.012 * likely_wolf_reads)
        score -= min(0.05, 0.010 * likely_villager_reads)

        if black_count > 0 and white_count > 0:
            score = max(score, 0.42)
            score = min(score, 0.82)

        scores[p] = max(0.01, min(0.99, score))

    # Rank calibration: keep weaker than previous version.
    # Strong rank forcing can hurt if LLM already ranked well.
    ranked = sorted(players, key=lambda x: scores[x], reverse=True)
    denom = max(1, len(players) - 1)

    for i, p in enumerate(ranked):
        rank_prior = 1.0 - i / denom
        scores[p] = 0.93 * scores[p] + 0.07 * rank_prior
        scores[p] = max(0.01, min(0.99, scores[p]))

    return {p: round(float(scores[p]), 6) for p in players}