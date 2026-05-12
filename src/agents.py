from typing import Dict, List, Any

from src.prompts import build_role_prompt, build_wolf_prompt


VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def get_claim_name(c):
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def player_has_claim(parsed, player, claim_name):
    claims = parsed.get("all_claims", {}).get(player, [])
    return any(get_claim_name(c) == claim_name for c in claims)


def player_has_claim_contains(parsed, player, keyword):
    claims = parsed.get("all_claims", {}).get(player, [])
    return any(keyword in get_claim_name(c) for c in claims)


def fallback_role_prior(players, parsed):
    out = {}

    for p in players:
        role_probs = {
            "Villager": 0.50,
            "Werewolf": 0.18,
            "Seer": 0.07,
            "Medium": 0.07,
            "Madman": 0.10,
            "Hunter": 0.08,
        }

        claims = parsed.get("all_claims", {}).get(p, [])

        if player_has_claim(parsed, p, "Seer CO"):
            role_probs["Seer"] += 0.18
            role_probs["Madman"] += 0.14
            role_probs["Werewolf"] += 0.08
            role_probs["Villager"] -= 0.15

        if player_has_claim(parsed, p, "Medium CO"):
            role_probs["Medium"] += 0.22
            role_probs["Werewolf"] += 0.08
            role_probs["Madman"] += 0.05
            role_probs["Villager"] -= 0.15

        if player_has_claim(parsed, p, "Not Seer/Medium"):
            role_probs["Seer"] *= 0.30
            role_probs["Medium"] *= 0.30
            role_probs["Villager"] += 0.08

        received = [
            r for r in parsed.get("all_divinations", [])
            if isinstance(r, dict) and r.get("target") == p
        ]

        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")

        role_probs["Werewolf"] += 0.08 * black_count
        role_probs["Werewolf"] -= 0.04 * white_count

        out[p] = normalize_role_probs(role_probs)

    return out


def fallback_wolf_scores(players, parsed):
    out = {}

    for p in players:
        score = 0.20

        received = [
            r for r in parsed.get("all_divinations", [])
            if isinstance(r, dict) and r.get("target") == p
        ]

        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")

        score += 0.18 * black_count
        score -= 0.08 * white_count

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

        score += min(0.12, 0.025 * likely_wolf_reads)
        score -= min(0.10, 0.020 * likely_villager_reads)

        if black_count > 0 and white_count > 0:
            score = max(score, 0.45)
            score = min(score, 0.85)

        out[p] = max(0.0, min(1.0, score))

    return out


def normalize_role_probs(role_probs):
    cleaned = {}

    for role in VALID_ROLES:
        try:
            v = float(role_probs.get(role, 0.0))
        except Exception:
            v = 0.0

        cleaned[role] = max(0.0, min(1.0, v))

    s = sum(cleaned.values())

    if s <= 0:
        cleaned = {
            "Villager": 0.70,
            "Werewolf": 0.18,
            "Seer": 0.03,
            "Medium": 0.03,
            "Madman": 0.03,
            "Hunter": 0.03,
        }
    else:
        cleaned = {k: v / s for k, v in cleaned.items()}

    return cleaned


def normalize_players_json(raw_players):
    """
    Accept both formats:

    Format A:
    {
      "Jacob": {"wolf_score": 0.7}
    }

    Format B:
    [
      {"character": "Jacob", "wolf_score": 0.7},
      {"name": "Regina", "wolf_score": 0.2}
    ]
    """
    if isinstance(raw_players, dict):
        return raw_players

    if isinstance(raw_players, list):
        out = {}
        for item in raw_players:
            if not isinstance(item, dict):
                continue

            name = (
                item.get("character")
                or item.get("name")
                or item.get("player")
                or item.get("Player")
            )

            if name:
                out[name] = item

        return out

    return {}


class RoleReasoningAgent:
    """
    Agent 1:
    Estimates role probability for every player.
    """

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, str],
        parsed: Dict[str, Any],
        game_index: str = "unknown",
    ) -> Dict[str, Dict[str, float]]:

        try:
            prompt = build_role_prompt(players, evidence_cards, parsed)

            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "role_agent", prompt)

            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)

                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "role_agent", raw_output)
                    self.debug_logger.save_json(game_index, "role_agent", data)

            except Exception as e:
                print(f"[WARN] RoleReasoningAgent fallback: {e}")

                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "role_agent", e)

                data = {}

            out = {}
            raw_players = normalize_players_json(data.get("players", {}))

            for p in players:
                role_probs = {}
                raw = raw_players.get(p, {})

                if not isinstance(raw, dict):
                    raw = {}

                for role in VALID_ROLES:
                    try:
                        role_probs[role] = float(raw.get(role, 0.0))
                    except Exception:
                        role_probs[role] = 0.0

                claims = parsed.get("all_claims", {}).get(p, [])
                if player_has_claim(parsed, p, "Seer CO"):
                    role_probs["Seer"] += 0.25
                    role_probs["Madman"] += 0.10
                    role_probs["Werewolf"] += 0.06

                if player_has_claim(parsed, p, "Medium CO"):
                    role_probs["Medium"] += 0.30

                if player_has_claim(parsed, p, "Not Seer/Medium") or (
                    player_has_claim(parsed, p, "Not Seer") and player_has_claim(parsed, p, "Not Medium")
                ):
                    role_probs["Seer"] *= 0.3
                    role_probs["Medium"] *= 0.3

                out[p] = normalize_role_probs(role_probs)

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "role_agent_final_scores", out)

            return out

        except Exception as e:
            print(f"[WARN] RoleReasoningAgent outer fallback: {e}")

            if self.debug_logger:
                self.debug_logger.save_error(game_index, "role_agent_outer", e)

            out = fallback_role_prior(players, parsed)

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "role_agent_fallback_scores", out)

            return out


class WolfReasoningAgent:
    """
    Agent 2:
    Estimates wolf_score for every player.
    """

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, str],
        parsed: Dict[str, Any],
        game_index: str = "unknown",
    ) -> Dict[str, float]:

        try:
            prompt = build_wolf_prompt(players, evidence_cards, parsed)

            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "wolf_agent", prompt)

            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)

                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "wolf_agent", raw_output)
                    self.debug_logger.save_json(game_index, "wolf_agent", data)

            except Exception as e:
                print(f"[WARN] WolfReasoningAgent fallback: {e}")

                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "wolf_agent", e)

                data = {}

            out = {}
            raw_players = normalize_players_json(data.get("players", {}))

            for p in players:
                raw = raw_players.get(p, {})

                if not isinstance(raw, dict):
                    raw = {}

                try:
                    score = float(raw.get("wolf_score", 0.2))
                except Exception:
                    score = 0.2

                score = max(0.0, min(1.0, score))

                # Rule-based adjustment from hard divination.
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
                    score += 0.18 * black_count

                if white_count > 0:
                    score -= 0.08 * white_count

                # Rule-based weak adjustment from soft reads.
                # README: white = likely villager, black = likely werewolf.
                # These are not hard divination results.
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

                score += min(0.12, 0.025 * likely_wolf_reads)
                score -= min(0.10, 0.020 * likely_villager_reads)

                # Panda: mixed hard result, keep high but uncertain.
                # Do not create panda from soft white/black reads.
                if black_count > 0 and white_count > 0:
                    score = max(score, 0.45)
                    score = min(score, 0.85)

                out[p] = max(0.0, min(1.0, score))

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "wolf_agent_final_scores", out)

            return out

        except Exception as e:
            print(f"[WARN] WolfReasoningAgent outer fallback: {e}")

            if self.debug_logger:
                self.debug_logger.save_error(game_index, "wolf_agent_outer", e)

            out = fallback_wolf_scores(players, parsed)

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "wolf_agent_fallback_scores", out)

            return out