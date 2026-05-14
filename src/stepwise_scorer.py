from collections import defaultdict
from typing import Any, Dict, List


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _unique(xs):
    return list(dict.fromkeys(xs))


def score_interactions(players: List[str], interaction_graph: Dict[str, Any], objective_pack: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Convert interaction edges into stepwise behavioral features.
    This is the inference-time analogue of stepwise feedback heuristics.
    """
    scores: Dict[str, Dict[str, float]] = {
        p: {
            "pressure_received": 0.0,
            "pressure_given": 0.0,
            "early_pressure_given": 0.0,
            "vote_pressure_received": 0.0,
            "vote_pressure_given": 0.0,
            "check_pressure_received": 0.0,
            "claim_challenge_received": 0.0,
            "claim_challenge_given": 0.0,
            "townread_received": 0.0,
            "townread_given": 0.0,
            "defense_received": 0.0,
            "defense_given": 0.0,
            "num_unique_pressure_sources": 0.0,
            "num_unique_defense_sources": 0.0,
        }
        for p in players
    }

    pressure_sources = defaultdict(set)
    defense_sources = defaultdict(set)

    for e in interaction_graph.get("edges", []):
        src = e.get("src")
        dst = e.get("dst")
        etype = e.get("type")
        strength = float(e.get("strength", 0.0))
        timing = e.get("timing")
        if src not in scores or dst not in scores:
            continue

        if etype == "pressure":
            scores[src]["pressure_given"] += strength
            scores[dst]["pressure_received"] += strength
            pressure_sources[dst].add(src)
            if timing == "early":
                scores[src]["early_pressure_given"] += strength
        elif etype == "vote_push":
            scores[src]["vote_pressure_given"] += strength
            scores[dst]["vote_pressure_received"] += strength
            pressure_sources[dst].add(src)
        elif etype == "check_preference":
            scores[dst]["check_pressure_received"] += strength
        elif etype == "claim_challenge":
            scores[src]["claim_challenge_given"] += strength
            scores[dst]["claim_challenge_received"] += strength
            pressure_sources[dst].add(src)
        elif etype == "townread":
            scores[src]["townread_given"] += strength
            scores[dst]["townread_received"] += strength
        elif etype == "defense":
            scores[src]["defense_given"] += strength
            scores[dst]["defense_received"] += strength
            defense_sources[dst].add(src)

    for p in players:
        scores[p]["num_unique_pressure_sources"] = float(len(pressure_sources[p]))
        scores[p]["num_unique_defense_sources"] = float(len(defense_sources[p]))

    return scores


def compute_wolf_feature_prior(players: List[str], stepwise_scores: Dict[str, Dict[str, float]], objective_pack: Dict[str, Any]) -> Dict[str, float]:
    """
    Rule-based wolf prior from objective facts + stepwise interaction signals.
    It should rank, not decide roles.
    """
    seer_claimers = set(objective_pack.get("seer_claimers", []))
    medium_claimers = set(objective_pack.get("medium_claimers", []))
    not_co = set(objective_pack.get("not_seer_medium_claimers", []))
    gray = set(objective_pack.get("gray_players", []))
    hard_results = objective_pack.get("hard_results", [])
    role_counts = objective_pack.get("role_counts", {})
    ww_count = int(role_counts.get("Werewolf", 3 if len(players) >= 13 else 2))

    seer_contested = len(seer_claimers) >= 2
    medium_contested = len(medium_claimers) >= 2

    out: Dict[str, float] = {}
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            out[p] = 0.01
            continue

        f = stepwise_scores.get(p, {})
        score = 0.20

        # Pressure/vote received is wolfy, but capped because villagers can be suspected.
        score += min(0.18, 0.035 * f.get("pressure_received", 0.0))
        score += min(0.16, 0.050 * f.get("vote_pressure_received", 0.0))
        score += min(0.10, 0.030 * f.get("claim_challenge_received", 0.0))
        score += min(0.10, 0.025 * f.get("num_unique_pressure_sources", 0.0))

        # Townreads and early pressure given are weakly anti-wolf.
        score -= min(0.08, 0.025 * f.get("townread_received", 0.0))
        score -= min(0.06, 0.020 * f.get("early_pressure_given", 0.0))

        # Defense can be wolfy if overdone, but weak signal.
        score += min(0.06, 0.018 * f.get("defense_received", 0.0))

        # Claim formation priors.
        if p in seer_claimers:
            score += 0.08 if seer_contested else -0.06
        if p in medium_claimers:
            score += 0.07 if medium_contested else -0.08
        if p in not_co:
            score -= 0.025
        if p in gray:
            score += 0.02

        # Hard result priors.
        black_count = 0
        white_count = 0
        for r in hard_results:
            if r.get("target") != p:
                continue
            if r.get("result") == "werewolf":
                black_count += 1
            elif r.get("result") == "human":
                white_count += 1
        score += 0.18 * black_count
        score -= 0.07 * white_count
        if black_count and white_count:
            score = max(score, 0.42)
            score = min(score, 0.82)

        out[p] = _clamp(score, 0.03, 0.95)

    # Mild role-count-aware rank floor.
    ranked = sorted([p for p in players if out.get(p, 0) > 0.011], key=lambda p: out[p], reverse=True)
    for i, p in enumerate(ranked):
        if i < ww_count:
            out[p] = max(out[p], 0.30 - 0.02 * i)
        elif i < ww_count + 3:
            out[p] = max(out[p], 0.14)
        out[p] = _clamp(out[p], 0.03, 0.95)

    return {p: round(float(out[p]), 6) for p in players}
