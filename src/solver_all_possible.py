from typing import Any, Dict, List
import itertools
import math

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def get_role_counts(num_players: int) -> Dict[str, int]:
    return {
        "Werewolf": 3 if num_players >= 13 else 2,
        "Seer": 1,
        "Medium": 1,
        "Madman": 1 if num_players >= 11 else 0,
        "Hunter": 1 if num_players >= 11 else 0,
    }


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
        if isinstance(claims, list) and any(claim_keyword in claim_name(c) for c in claims):
            n += 1
    return n


def adjusted_role_score(player: str, role: str, role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> float:
    # Use probabilities as soft evidence.  Assignment is for Macro-F1 only;
    # wolf_score AP is calibrated separately.
    score = safe_score(role_scores, player, role, default=0.01)

    seer_co = has_claim(parsed, player, "Seer CO")
    medium_co = has_claim(parsed, player, "Medium CO")
    not_both = has_claim(parsed, player, "Not Seer/Medium")
    not_seer = has_claim(parsed, player, "Not Seer")
    not_medium = has_claim(parsed, player, "Not Medium")

    if seer_co:
        if role == "Seer":
            score += 0.32
        elif role == "Madman":
            score += 0.18
        elif role == "Werewolf":
            score += 0.14
        else:
            score -= 0.18

    if medium_co:
        if role == "Medium":
            score += 0.32
        elif role == "Werewolf":
            score += 0.16
        elif role == "Madman":
            score += 0.12
        else:
            score -= 0.18

    if not_both:
        if role in ("Seer", "Medium"):
            score -= 0.50
        elif role == "Villager":
            score += 0.08
        elif role == "Hunter":
            score += 0.03

    if not_seer and role == "Seer":
        score -= 0.35
    if not_medium and role == "Medium":
        score -= 0.35

    if role == "Hunter" and (seer_co or medium_co):
        score -= 0.25

    return max(1e-6, min(0.999999, score))


def assignment_log_score(assignment: Dict[str, str], role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> float:
    total = 0.0
    for player, role in assignment.items():
        total += math.log(adjusted_role_score(player, role, role_scores, parsed))

    seer_claimers = [p for p in assignment if has_claim(parsed, p, "Seer CO")]
    medium_claimers = [p for p in assignment if has_claim(parsed, p, "Medium CO")]

    # If there are claimers, the true role is usually among them, but keep this
    # a penalty rather than a hard constraint because parser can miss claims.
    if seer_claimers and not any(assignment.get(p) == "Seer" for p in seer_claimers):
        total -= 2.5
    if medium_claimers and not any(assignment.get(p) == "Medium" for p in medium_claimers):
        total -= 2.5

    # Fake claimers are usually wolf or madman, not ordinary village power roles.
    for p in seer_claimers:
        if assignment.get(p) in ("Villager", "Hunter", "Medium"):
            total -= 1.0
    for p in medium_claimers:
        if assignment.get(p) in ("Villager", "Hunter", "Seer"):
            total -= 1.0

    # Madman is most often useful as a fake ability claimant in 11+ games.
    madman = next((p for p, r in assignment.items() if r == "Madman"), None)
    if madman and (seer_claimers or medium_claimers) and madman not in set(seer_claimers) | set(medium_claimers):
        total -= 0.25

    return total


def constrained_role_assignment(players: List[str], role_scores: Dict[str, Dict[str, float]], parsed: Dict[str, Any]) -> Dict[str, str]:
    counts = get_role_counts(len(players))
    ww_count = counts["Werewolf"]
    has_madman = counts.get("Madman", 0) > 0
    has_hunter = counts.get("Hunter", 0) > 0

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

    return best_assignment or {p: "Villager" for p in players}


def normalize_wolf_scores(
    players: List[str],
    wolf_scores: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    assigned_roles: Dict[str, str],
    parsed: Dict[str, Any],
) -> Dict[str, float]:
    """
    Continuous actual-Werewolf probability.

    This intentionally keeps wolf_agent ranking dominant because the competition
    score weights Werewolf AP more than role Macro-F1.  The constrained role
    assignment is used only as a small prior.
    """
    scores: Dict[str, float] = {}
    seer_claimers = count_claimers(parsed, "Seer CO")
    medium_claimers = count_claimers(parsed, "Medium CO")
    expected_wolves = get_role_counts(len(players))["Werewolf"]

    for p in players:
        try:
            base = float(wolf_scores.get(p, expected_wolves / max(1, len(players))))
        except Exception:
            base = expected_wolves / max(1, len(players))
        role_wolf = safe_score(role_scores, p, "Werewolf", default=base)

        # AP-protective blend: do not let the role solver dominate.
        score = 0.84 * base + 0.16 * role_wolf

        assigned = assigned_roles.get(p)
        if assigned == "Werewolf":
            score += 0.035
        elif assigned == "Madman":
            score -= 0.035
        elif assigned == "Seer":
            score -= 0.12 if seer_claimers <= 1 else 0.015
        elif assigned == "Medium":
            score -= 0.12 if medium_claimers <= 1 else 0.015
        elif assigned == "Hunter":
            score -= 0.035

        # Contested claimers are often true/fake ability; fake can be wolf or madman.
        if has_claim(parsed, p, "Seer CO") and seer_claimers >= 2:
            score += 0.02
        if has_claim(parsed, p, "Medium CO") and medium_claimers >= 2:
            score += 0.02

        # Hard divination signals, moderated because the source can be fake.
        received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")
        score += 0.075 * black_count
        score -= 0.035 * white_count
        if black_count > 0 and white_count > 0:
            score = max(score, 0.45)
            score = min(score, 0.85)

        # Soft reads are weak. They are helpful mostly for ranking inside gray.
        soft_received = [r for r in parsed.get("all_soft_reads", []) if isinstance(r, dict) and r.get("target") == p]
        likely_wolf_reads = sum(1 for r in soft_received if r.get("read") == "likely_werewolf")
        likely_villager_reads = sum(1 for r in soft_received if r.get("read") == "likely_villager")
        score += min(0.055, 0.010 * likely_wolf_reads)
        score -= min(0.045, 0.008 * likely_villager_reads)

        scores[p] = max(0.01, min(0.99, score))

    # Mild rank-spread calibration. Preserve ordering while preventing ties.
    ranked = sorted(players, key=lambda x: scores[x], reverse=True)
    denom = max(1, len(ranked) - 1)
    for i, p in enumerate(ranked):
        rank_prior = 1.0 - i / denom
        scores[p] = 0.90 * scores[p] + 0.10 * rank_prior
        scores[p] = max(0.01, min(0.99, scores[p]))

    return {p: round(float(scores[p]), 6) for p in players}
