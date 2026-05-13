from typing import Any, Dict, List, Optional

from src.prompts import build_role_prompt, build_wolf_prompt, build_formation_prompt

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def get_claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def player_has_claim(parsed: Dict[str, Any], player: str, claim_name: str) -> bool:
    claims = parsed.get("all_claims", {}).get(player, [])
    return any(get_claim_name(c) == claim_name for c in claims)


def player_has_claim_contains(parsed: Dict[str, Any], player: str, keyword: str) -> bool:
    claims = parsed.get("all_claims", {}).get(player, [])
    return any(keyword in get_claim_name(c) for c in claims)


def normalize_role_probs(role_probs: Dict[str, Any]) -> Dict[str, float]:
    cleaned = {}
    for role in VALID_ROLES:
        try:
            cleaned[role] = clamp(float(role_probs.get(role, 0.0)))
        except Exception:
            cleaned[role] = 0.0
    s = sum(cleaned.values())
    if s <= 0:
        cleaned = {
            "Villager": 0.55,
            "Werewolf": 0.18,
            "Seer": 0.07,
            "Medium": 0.07,
            "Madman": 0.08,
            "Hunter": 0.05,
        }
    else:
        cleaned = {k: v / s for k, v in cleaned.items()}
    return cleaned


def normalize_players_json(raw_players: Any) -> Dict[str, Any]:
    if isinstance(raw_players, dict):
        return raw_players
    if isinstance(raw_players, list):
        out = {}
        for item in raw_players:
            if not isinstance(item, dict):
                continue
            name = item.get("character") or item.get("name") or item.get("player") or item.get("Player")
            if name:
                out[str(name)] = item
        return out
    return {}


def role_counts(num_players: int) -> Dict[str, int]:
    return {
        "Werewolf": 3 if num_players >= 13 else 2,
        "Seer": 1,
        "Medium": 1,
        "Madman": 1 if num_players >= 11 else 0,
        "Hunter": 1 if num_players >= 11 else 0,
    }


def get_claimers(parsed: Dict[str, Any], players: List[str], claim: str) -> List[str]:
    return [p for p in players if player_has_claim(parsed, p, claim)]


def deterministic_formation(players: List[str], parsed: Dict[str, Any], daily_states: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    seers = get_claimers(parsed, players, "Seer CO")
    mediums = get_claimers(parsed, players, "Medium CO")
    ability = set(seers) | set(mediums)
    gray = [p for p in players if p not in ability]
    latest_formation = f"{len(seers)}-{len(mediums)}" if seers or mediums else "0-0"
    if isinstance(daily_states, dict) and daily_states:
        last = daily_states[list(daily_states.keys())[-1]]
        latest_formation = last.get("formation") or last.get("formation_type") or latest_formation
        seers = last.get("seer_claimers", seers) or seers
        mediums = last.get("medium_claimers", mediums) or mediums
        gray = last.get("gray_players", gray) or gray
    ww = role_counts(len(players))["Werewolf"]
    likely_ability_wolf = 1 if (len(seers) + len(mediums)) >= 2 else 0
    return {
        "verified_facts": {
            "latest_formation": latest_formation,
            "seer_claimers": seers,
            "medium_claimers": mediums,
            "gray_players": gray,
            "invalid_or_uncertain_claims": [],
        },
        "latest_formation": latest_formation,
        "seer_claimers": seers,
        "medium_claimers": mediums,
        "gray_players": gray,
        "likely_patterns": [],
        "claimant_alignment_estimates": {},
        "gray_wolf_slots_estimate": max(0, ww - likely_ability_wolf),
        "notes": "deterministic fallback",
    }


def fallback_role_prior(players: List[str], parsed: Dict[str, Any], formation_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
    counts = role_counts(len(players))
    out = {}
    formation = formation_analysis or deterministic_formation(players, parsed)
    facts = formation.get("verified_facts", {}) if isinstance(formation, dict) else {}
    seers = set(facts.get("seer_claimers") or formation.get("seer_claimers", []) or [])
    mediums = set(facts.get("medium_claimers") or formation.get("medium_claimers", []) or [])

    for p in players:
        probs = {
            "Villager": 0.52,
            "Werewolf": 0.16,
            "Seer": 0.06,
            "Medium": 0.06,
            "Madman": 0.10 if counts["Madman"] else 0.0,
            "Hunter": 0.08 if counts["Hunter"] else 0.0,
        }
        if p in seers or player_has_claim(parsed, p, "Seer CO"):
            probs["Seer"] += 0.30
            probs["Madman"] += 0.18
            probs["Werewolf"] += 0.12
            probs["Villager"] *= 0.45
            probs["Hunter"] *= 0.30
            probs["Medium"] *= 0.30
        if p in mediums or player_has_claim(parsed, p, "Medium CO"):
            probs["Medium"] += 0.30
            probs["Werewolf"] += 0.14
            probs["Madman"] += 0.12
            probs["Villager"] *= 0.45
            probs["Hunter"] *= 0.30
            probs["Seer"] *= 0.30
        if player_has_claim(parsed, p, "Not Seer/Medium") or (
            player_has_claim(parsed, p, "Not Seer") and player_has_claim(parsed, p, "Not Medium")
        ):
            probs["Seer"] *= 0.18
            probs["Medium"] *= 0.18
            probs["Villager"] += 0.07
        received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")
        probs["Werewolf"] += 0.10 * black_count
        probs["Werewolf"] -= 0.05 * white_count
        out[p] = normalize_role_probs(probs)
    return out


def fallback_wolf_scores(players: List[str], parsed: Dict[str, Any], formation_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    counts = role_counts(len(players))
    base = counts["Werewolf"] / max(1, len(players))
    formation = formation_analysis or deterministic_formation(players, parsed)
    facts = formation.get("verified_facts", {}) if isinstance(formation, dict) else {}
    seers = set(facts.get("seer_claimers") or formation.get("seer_claimers", []) or [])
    mediums = set(facts.get("medium_claimers") or formation.get("medium_claimers", []) or [])
    contested = len(seers) + len(mediums) >= 2
    out = {}
    for p in players:
        score = base
        if contested and (p in seers or p in mediums):
            score += 0.05
        if player_has_claim(parsed, p, "Not Seer/Medium"):
            score -= 0.01
        received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")
        score += 0.16 * black_count
        score -= 0.07 * white_count
        soft_received = [r for r in parsed.get("all_soft_reads", []) if isinstance(r, dict) and r.get("target") == p]
        likely_wolf_reads = sum(1 for r in soft_received if r.get("read") == "likely_werewolf")
        likely_villager_reads = sum(1 for r in soft_received if r.get("read") == "likely_villager")
        score += min(0.10, 0.018 * likely_wolf_reads)
        score -= min(0.08, 0.014 * likely_villager_reads)
        if black_count > 0 and white_count > 0:
            score = max(score, 0.45)
        out[p] = clamp(score, 0.01, 0.99)
    return out


class FormationAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        daily_states: Dict[str, Any],
        parsed: Optional[Dict[str, Any]] = None,
        evidence_cards: Optional[Dict[str, Any]] = None,
        game_index: str = "unknown",
    ) -> Dict[str, Any]:
        parsed = parsed or {}
        fallback = deterministic_formation(players, parsed, daily_states)
        try:
            prompt = build_formation_prompt(players, daily_states, parsed=parsed, evidence_cards=evidence_cards)
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "formation_agent", prompt)
            data, raw_output = self.llm.generate_json_with_raw(prompt)
            if self.debug_logger:
                self.debug_logger.save_raw_output(game_index, "formation_agent", raw_output)
                self.debug_logger.save_json(game_index, "formation_agent", data)
            if not isinstance(data, dict):
                return fallback
            # Merge deterministic fields if the LLM omitted them.
            facts = data.get("verified_facts") if isinstance(data.get("verified_facts"), dict) else {}
            data.setdefault("verified_facts", facts)
            for k in ["latest_formation", "seer_claimers", "medium_claimers", "gray_players"]:
                if k not in data and k in fallback:
                    data[k] = fallback[k]
            for k, v in fallback.get("verified_facts", {}).items():
                data["verified_facts"].setdefault(k, v)
            data.setdefault("gray_wolf_slots_estimate", fallback.get("gray_wolf_slots_estimate"))
            return data
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "formation_agent", e)
            return fallback


class RoleReasoningAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, Any],
        parsed: Dict[str, Any],
        daily_states: Dict[str, Any] = None,
        formation_analysis: Optional[Dict[str, Any]] = None,
        game_index: str = "unknown",
    ) -> Dict[str, Dict[str, float]]:
        fallback = fallback_role_prior(players, parsed, formation_analysis)
        try:
            prompt = build_role_prompt(players, evidence_cards, parsed, daily_states=daily_states, formation_analysis=formation_analysis)
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "role_agent", prompt)
            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)
                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "role_agent", raw_output)
                    self.debug_logger.save_json(game_index, "role_agent", data)
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "role_agent", e)
                data = {}

            raw_players = normalize_players_json(data.get("players", {}) if isinstance(data, dict) else {})
            out = {}
            for p in players:
                raw = raw_players.get(p, {}) if isinstance(raw_players, dict) else {}
                if not isinstance(raw, dict):
                    raw = {}
                role_probs = {}
                for role in VALID_ROLES:
                    try:
                        role_probs[role] = float(raw.get(role, fallback[p].get(role, 0.0)))
                    except Exception:
                        role_probs[role] = fallback[p].get(role, 0.0)
                # Blend with deterministic prior to prevent malformed prompt outputs
                # or overconfident small-model guesses from dominating.
                blended = {r: 0.72 * role_probs.get(r, 0.0) + 0.28 * fallback[p].get(r, 0.0) for r in VALID_ROLES}
                if player_has_claim(parsed, p, "Seer CO"):
                    blended["Seer"] += 0.08
                    blended["Madman"] += 0.05
                    blended["Werewolf"] += 0.04
                    blended["Villager"] *= 0.70
                    blended["Hunter"] *= 0.45
                if player_has_claim(parsed, p, "Medium CO"):
                    blended["Medium"] += 0.08
                    blended["Madman"] += 0.04
                    blended["Werewolf"] += 0.05
                    blended["Villager"] *= 0.70
                    blended["Hunter"] *= 0.45
                if player_has_claim(parsed, p, "Not Seer/Medium") or (
                    player_has_claim(parsed, p, "Not Seer") and player_has_claim(parsed, p, "Not Medium")
                ):
                    blended["Seer"] *= 0.20
                    blended["Medium"] *= 0.20
                out[p] = normalize_role_probs(blended)
            if self.debug_logger:
                self.debug_logger.save_json(game_index, "role_agent_final_scores", out)
            return out
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "role_agent_outer", e)
                self.debug_logger.save_json(game_index, "role_agent_fallback_scores", fallback)
            return fallback


class WolfReasoningAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, Any],
        parsed: Dict[str, Any],
        daily_states: Dict[str, Any] = None,
        formation_analysis: Optional[Dict[str, Any]] = None,
        game_index: str = "unknown",
    ) -> Dict[str, float]:
        fallback = fallback_wolf_scores(players, parsed, formation_analysis)
        try:
            prompt = build_wolf_prompt(players, evidence_cards, parsed, daily_states=daily_states, formation_analysis=formation_analysis)
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "wolf_agent", prompt)
            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)
                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "wolf_agent", raw_output)
                    self.debug_logger.save_json(game_index, "wolf_agent", data)
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "wolf_agent", e)
                data = {}

            raw_players = normalize_players_json(data.get("players", {}) if isinstance(data, dict) else {})
            # Also accept ranking-only output.
            if isinstance(data, dict) and isinstance(data.get("wolf_ranking"), list):
                n = max(1, len(data["wolf_ranking"]) - 1)
                for i, item in enumerate(data["wolf_ranking"]):
                    if not isinstance(item, dict):
                        continue
                    name = item.get("player") or item.get("character") or item.get("name")
                    if name and name not in raw_players:
                        raw_players[str(name)] = {"wolf_score": 1.0 - i / n}

            out = {}
            for p in players:
                raw = raw_players.get(p, {}) if isinstance(raw_players, dict) else {}
                if not isinstance(raw, dict):
                    raw = {}
                try:
                    llm_score = float(raw.get("wolf_score", fallback[p]))
                except Exception:
                    llm_score = fallback[p]
                score = 0.78 * clamp(llm_score) + 0.22 * fallback[p]

                # Hard result adjustment. These are still moderated because fake
                # Seers can produce fake black/white results.
                received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
                black_count = sum(1 for r in received if r.get("result") == "werewolf")
                white_count = sum(1 for r in received if r.get("result") == "human")
                score += 0.10 * black_count
                score -= 0.045 * white_count
                if black_count > 0 and white_count > 0:
                    score = max(score, 0.45)
                    score = min(score, 0.85)
                out[p] = clamp(score, 0.01, 0.99)

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "wolf_agent_final_scores", out)
            return out
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "wolf_agent_outer", e)
                self.debug_logger.save_json(game_index, "wolf_agent_fallback_scores", fallback)
            return fallback
