from typing import Dict, List, Any
import itertools
import math

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




"""
新增一個窮舉 / beam 版。你的每局人數大約 10–16，角色數固定，
直接枚舉 Seer、Medium、Madman、Hunter，再從剩下挑狼，成本可接受。
"""
def adjusted_role_score(
    player: str,
    role: str,
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> float:
    score = safe_score(role_scores, player, role, default=0.01)

    if has_claim(parsed, player, "Seer CO"):
        if role == "Seer":
            score += 0.35
        elif role in ("Werewolf", "Madman"):
            score += 0.15
        elif role in ("Medium", "Hunter", "Villager"):
            score -= 0.20

    if has_claim(parsed, player, "Medium CO"):
        if role == "Medium":
            score += 0.35
        elif role in ("Werewolf", "Madman"):
            score += 0.12
        elif role in ("Seer", "Hunter", "Villager"):
            score -= 0.20

    if has_claim(parsed, player, "Not Seer/Medium"):
        if role in ("Seer", "Medium"):
            score -= 0.45
        elif role == "Villager":
            score += 0.08

    if has_claim(parsed, player, "Not Seer") and role == "Seer":
        score -= 0.35

    if has_claim(parsed, player, "Not Medium") and role == "Medium":
        score -= 0.35

    return max(1e-6, min(0.999999, score))


def assignment_log_score(
    assignment: Dict[str, str],
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> float:
    total = 0.0

    for player, role in assignment.items():
        s = adjusted_role_score(player, role, role_scores, parsed)
        total += math.log(s)

    seer_claimers = [
        p for p in assignment.keys()
        if has_claim(parsed, p, "Seer CO")
    ]

    medium_claimers = [
        p for p in assignment.keys()
        if has_claim(parsed, p, "Medium CO")
    ]

    # 如果有人跳占，真占通常應該在跳占者之中。
    # 但不要寫死成硬限制，避免 parser 漏 claim 時整局崩掉。
    if seer_claimers:
        true_seer_in_claimers = any(
            assignment.get(p) == "Seer"
            for p in seer_claimers
        )
        if not true_seer_in_claimers:
            total -= 2.0

    # 如果有人跳靈，真靈通常應該在跳靈者之中。
    if medium_claimers:
        true_medium_in_claimers = any(
            assignment.get(p) == "Medium"
            for p in medium_claimers
        )
        if not true_medium_in_claimers:
            total -= 2.0

    # 假跳占通常是狼或狂，不太會是普通村/獵/靈。
    for p in seer_claimers:
        if assignment.get(p) in ("Villager", "Hunter", "Medium"):
            total -= 0.8

    # 假跳靈通常是狼或狂，不太會是普通村/獵/占。
    for p in medium_claimers:
        if assignment.get(p) in ("Villager", "Hunter", "Seer"):
            total -= 0.8

    return total


def constrained_role_assignment(
    players: List[str],
    role_scores: Dict[str, Dict[str, float]],
    parsed: Dict[str, Any],
) -> Dict[str, str]:
    role_counts = get_role_counts(len(players))

    ww_count = role_counts["Werewolf"]
    has_madman = role_counts.get("Madman", 0) > 0
    has_hunter = role_counts.get("Hunter", 0) > 0

    best_assignment = None
    best_score = -1e18

    players_list = list(players)

    for seer in players_list:
        rem1 = [p for p in players_list if p != seer]

        for medium in rem1:
            rem2 = [p for p in rem1 if p != medium]

            madman_candidates = rem2 if has_madman else [None]
            for madman in madman_candidates:
                rem3 = [p for p in rem2 if p != madman] if madman else rem2

                hunter_candidates = rem3 if has_hunter else [None]
                for hunter in hunter_candidates:
                    rem4 = [p for p in rem3 if p != hunter] if hunter else rem3

                    if len(rem4) < ww_count:
                        continue

                    for wolves in itertools.combinations(rem4, ww_count):
                        assignment = {p: "Villager" for p in players_list}
                        assignment[seer] = "Seer"
                        assignment[medium] = "Medium"

                        if madman:
                            assignment[madman] = "Madman"

                        if hunter:
                            assignment[hunter] = "Hunter"

                        for w in wolves:
                            assignment[w] = "Werewolf"

                        score = assignment_log_score(assignment, role_scores, parsed)

                        if score > best_score:
                            best_score = score
                            best_assignment = assignment

    if best_assignment is None:
        return {p: "Villager" for p in players}

    return best_assignment


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