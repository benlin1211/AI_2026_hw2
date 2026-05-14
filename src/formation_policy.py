"""Formation policy mapping for fast Werewolf inference."""
from __future__ import annotations

from typing import Any, Dict, List


def classify_formation(formation: str) -> Dict[str, Any]:
    if formation == "1-1":
        return {
            "mode": "confirmed_axis",
            "priority": ["confirmed_medium_or_axis", "gray_wolf_search", "check_claim_survival"],
            "llm_need": "low",
        }
    if formation == "2-1":
        return {
            "mode": "seer_credit_with_medium_axis",
            "priority": ["compare_two_seer_lines", "use_medium_results", "watch_bitten_seer_candidate"],
            "llm_need": "medium",
        }
    if formation == "3-1":
        return {
            "mode": "medium_axis_and_seer_bite",
            "priority": ["medium_axis", "seer_bite_reweight", "panda_push_chain", "madman_vs_wolf_fake"],
            "llm_need": "medium",
        }
    if formation == "2-2":
        return {
            "mode": "line_battle",
            "priority": ["draw_seer_medium_lines", "compare_medium_results", "track_attacks_and_gj"],
            "llm_need": "high",
        }
    if formation == "3-2":
        return {
            "mode": "gray_lw_search_and_role_resolution",
            "priority": ["four_exposed_non_town", "gray_lw_search", "free_check_results", "role_roller_order"],
            "llm_need": "high",
        }
    # Generic fallbacks.
    try:
        s, m = [int(x) for x in str(formation).split("-")[:2]]
    except Exception:
        s, m = 0, 0
    if s + m >= 4:
        mode = "heavy_claim_resolution"
        need = "high"
    elif s >= 2 or m >= 2:
        mode = "contested_ability_resolution"
        need = "medium"
    else:
        mode = "gray_search"
        need = "low"
    return {"mode": mode, "priority": ["objective_events_first", "gray_search"], "llm_need": need}


def analyze_formation_policy(board_states: Dict[str, Any]) -> Dict[str, Any]:
    if not board_states:
        return {"latest_formation": "0-0", "mode": "unknown", "priority": [], "llm_need": "low"}
    latest_day = sorted(board_states.keys(), key=lambda d: int(d) if str(d).isdigit() else 999)[-1]
    latest = board_states[latest_day]
    formation = latest.get("formation", "0-0")
    policy = classify_formation(formation)
    policy.update({
        "latest_day": latest_day,
        "latest_formation": formation,
        "seer_claimers": latest.get("seer_claimers", []),
        "medium_claimers": latest.get("medium_claimers", []),
        "hunter_claimers": latest.get("hunter_claimers", []),
        "gray_players": latest.get("gray_players", []),
        "pandas": latest.get("pandas", []),
        "confirmed_whites": latest.get("confirmed_whites", []),
        "gj_days": latest.get("gj_days", []),
    })
    return policy
