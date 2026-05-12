from typing import Dict, List, Any


VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def get_role_counts(num_players: int) -> Dict[str, int]:
    """
    Role counts based on Dataset_README.

    - 2 werewolves in villages of up to 12 players.
    - 3 werewolves in villages of 13 or more players.
    - Madman is added in villages with 11 or more players.
    - Hunter is added in villages with 11 or more players.
    - Seer and Medium are assumed to exist.
    """
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


def get_claims(parsed: Dict[str, Any], player: str) -> List[str]:
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
    for p, claims in parsed.get("all_claims", {}).items():
        if any(claim_keyword in claim_name(c) for c in claims):
            n += 1
    return n


def assign_best_candidate(
    role: str,
    count: int,
    remaining: set,
    assigned: Dict[str, str],
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
    claim_boost: float = 0.0,
    claim_keyword: str = "",
) -> None:
    """
    Assign top `count` candidates for a role from remaining players.
    """
    if count <= 0 or not remaining:
        return

    candidates = []
    for p in remaining:
        score = safe_score(role_scores, p, role)

        if claim_keyword and has_claim(parsed, p, claim_keyword):
            score += claim_boost

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
    Greedy constrained assignment.

    Priority:
    1. Medium
    2. Seer
    3. Werewolf
    4. Madman
    5. Hunter
    6. Villager

    This follows the homework's constrained-solver idea:
    summarize role rankings, then make fixed-count role assignments.
    """
    role_counts = get_role_counts(len(players))

    assigned: Dict[str, str] = {}
    remaining = set(players)

    # 1. Medium
    assign_best_candidate(
        role="Medium",
        count=role_counts.get("Medium", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
        claim_boost=1.0,
        claim_keyword="Medium CO",
    )

    # 2. Seer
    assign_best_candidate(
        role="Seer",
        count=role_counts.get("Seer", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
        claim_boost=0.8,
        claim_keyword="Seer CO",
    )

    # 3. Werewolves
    assign_best_candidate(
        role="Werewolf",
        count=role_counts.get("Werewolf", 0),
        remaining=remaining,
        assigned=assigned,
        role_scores=role_scores,
        parsed=parsed,
    )

    # 4. Madman
    # Fake Seer candidates are plausible Madman.
    if role_counts.get("Madman", 0) > 0 and remaining:
        candidates = []
        for p in remaining:
            score = safe_score(role_scores, p, "Madman")
            if has_claim(parsed, p, "Seer CO"):
                score += 0.25
            if has_claim(parsed, p, "Medium CO"):
                score += 0.10
            candidates.append((score, p))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for _, p in candidates[:role_counts["Madman"]]:
            assigned[p] = "Madman"
            remaining.discard(p)

    # 5. Hunter
    if role_counts.get("Hunter", 0) > 0 and remaining:
        assign_best_candidate(
            role="Hunter",
            count=role_counts.get("Hunter", 0),
            remaining=remaining,
            assigned=assigned,
            role_scores=role_scores,
            parsed=parsed,
        )

    # 6. Rest are Villagers
    for p in remaining:
        assigned[p] = "Villager"

    # Safety: include every player exactly once.
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
    """
    Produce continuous wolf_score.

    Do not force 1.0 for assigned werewolves, because AP benefits from ranking.
    Hard divination results are stronger than soft white/black reads.
    """
    scores: Dict[str, float] = {}

    seer_claimers = count_claimers(parsed, "Seer CO")
    medium_claimers = count_claimers(parsed, "Medium CO")

    for p in players:
        try:
            base = float(wolf_scores.get(p, 0.20))
        except Exception:
            base = 0.20

        role_wolf = safe_score(role_scores, p, "Werewolf", default=0.0)

        score = 0.65 * base + 0.35 * role_wolf

        # Role-assignment adjustment.
        if assigned_roles.get(p) == "Werewolf":
            score += 0.12

        assigned = assigned_roles.get(p)

        if assigned == "Seer":
            if seer_claimers <= 1:
                score -= 0.20
            else:
                score -= 0.03

        if assigned == "Medium":
            if medium_claimers <= 1:
                score -= 0.20
            else:
                score -= 0.03

        if assigned_roles.get(p) == "Madman":
            # Madman is wolf-aligned but not a Werewolf.
            # Do not raise too much, because wolf_score asks Werewolf probability.
            score += 0.03

        # Hard divination adjustment.
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
            score += 0.10 * black_count

        if white_count > 0:
            score -= 0.05 * white_count

        # Soft read adjustment.
        # README: white = likely villager, black = likely werewolf.
        # These are weak social reads, not hard results.
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

        score += min(0.08, 0.015 * likely_wolf_reads)
        score -= min(0.06, 0.012 * likely_villager_reads)

        # Panda only comes from mixed hard divination results.
        if black_count > 0 and white_count > 0:
            score = max(score, 0.45)
            score = min(score, 0.85)

        scores[p] = max(0.01, min(0.99, score))

    # Rank calibration: preserve ordering but create separation.
    ranked = sorted(players, key=lambda x: scores[x], reverse=True)

    denom = max(1, len(players) - 1)
    for i, p in enumerate(ranked):
        rank_prior = 1.0 - i / denom
        scores[p] = 0.85 * scores[p] + 0.15 * rank_prior
        scores[p] = max(0.01, min(0.99, scores[p]))

    return {p: round(float(scores[p]), 6) for p in players}