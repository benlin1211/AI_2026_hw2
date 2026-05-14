import json
from typing import Dict, List, Any, Tuple

from src.prompts import build_formation_prompt, build_event_extraction_prompt, build_state_tracker_prompt

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]
ROLE_ORDER = VALID_ROLES
ROLE_SHORT_KEYS = {
    "V": "Villager",
    "W": "Werewolf",
    "S": "Seer",
    "M": "Medium",
    "C": "Madman",   # C = Crazy/Madman
    "H": "Hunter",
}


# -----------------------------
# Generic helpers
# -----------------------------

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


def _short_json(obj: Any, max_chars: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    if len(s) <= max_chars:
        return s
    # Keep tail: later days often contain more decisive role/vote info.
    return s[-max_chars:]


def _player_code_maps(players: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    code_to_player = {f"P{i}": p for i, p in enumerate(players)}
    player_to_code = {p: c for c, p in code_to_player.items()}
    return code_to_player, player_to_code


def _players_code_text(players: List[str]) -> str:
    code_to_player, _ = _player_code_maps(players)
    return "\n".join(f"{c}={p}" for c, p in code_to_player.items())


def _latest_daily_state(daily_states: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(daily_states, dict) or not daily_states:
        return {}
    # Prefer non-LLM final numeric day if present, otherwise last insertion.
    keys = list(daily_states.keys())
    return daily_states[keys[-1]] if keys else {}


def _compact_daily_states(daily_states: Dict[str, Any], max_days: int = 8) -> Dict[str, Any]:
    if not isinstance(daily_states, dict):
        return {}
    out = {}
    keys = list(daily_states.keys())[-max_days:]
    for d in keys:
        s = daily_states.get(d, {})
        if not isinstance(s, dict):
            continue
        out[str(d)] = {
            "f": s.get("formation"),
            "ft": s.get("formation_type"),
            "sc": s.get("seer_claimers", []),
            "mc": s.get("medium_claimers", []),
            "nsm": s.get("not_seer_medium_claimers", []),
            "gray": s.get("gray_players", []),
            "dead": s.get("dead_players", []),
            "hw": s.get("hard_white_targets", []),
            "hb": s.get("hard_black_targets", []),
            "sus": s.get("top_suspected_players", []),
            "exe": s.get("top_execution_targets", []),
            "seer_t": s.get("top_seer_targets", []),
        }
    return out


def _build_timeline_for_prompt(parsed: Dict[str, Any], max_messages: int = 80) -> List[Dict[str, Any]]:
    keywords = [
        "Seer", "Medium", "not Seer", "not Medium", "claim", "CO",
        "result", "verdict", "divination", "check", "checked",
        "white", "black", "gray", "wolf", "human", "lynch", "execute", "vote",
        "●", "○", "▼", "▽", "占", "霊", "靈", "判定", "結果", "白", "黒", "狼"
    ]
    out = []
    for msg in parsed.get("messages", []):
        text = str(msg.get("text", ""))
        low = text.lower()
        if any(k.lower() in low for k in keywords):
            out.append({
                "d": msg.get("day"),
                "n": msg.get("no"),
                "t": msg.get("time"),
                "s": msg.get("speaker"),
                "x": text[:260],
            })
    return out[-max_messages:]


def _compact_game_payload(
    players: List[str],
    evidence_cards: Dict[str, Any],
    parsed: Dict[str, Any],
    daily_states: Dict[str, Any] = None,
    formation_analysis: Dict[str, Any] = None,
    max_chars: int = 15000,
) -> str:
    global_card = evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {}
    payload = {
        "g": global_card,
        "ds": _compact_daily_states(daily_states or {}),
        "last": _latest_daily_state(daily_states or {}),
        "div": parsed.get("all_divinations", [])[-80:],
        "votes": parsed.get("all_votes", [])[-120:],
        "soft": parsed.get("all_soft_reads", [])[-120:],
        "deaths": parsed.get("deaths", []),
        "tl": _build_timeline_for_prompt(parsed, max_messages=90),
        "llm_state": parsed.get("llm_state", {}),
        "formation": formation_analysis or {},
    }
    return _short_json(payload, max_chars=max_chars)


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    return max(0.0, min(1.0, v))


# -----------------------------
# Fallbacks
# -----------------------------

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
        likely_wolf_reads = sum(1 for r in soft_received if r.get("read") == "likely_werewolf")
        likely_villager_reads = sum(1 for r in soft_received if r.get("read") == "likely_villager")
        score += min(0.12, 0.025 * likely_wolf_reads)
        score -= min(0.10, 0.020 * likely_villager_reads)

        if black_count > 0 and white_count > 0:
            score = max(score, 0.45)
            score = min(score, 0.85)

        out[p] = max(0.0, min(1.0, score))
    return out


# -----------------------------
# Compact prompt builders
# -----------------------------

def build_compact_role_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None) -> str:
    mapping = _players_code_text(players)
    evidence = _compact_game_payload(
        players=players,
        evidence_cards=evidence_cards,
        parsed=parsed,
        daily_states=daily_states,
        formation_analysis=formation_analysis,
        max_chars=15000,
    )
    codes = ",".join(f"P{i}" for i in range(len(players)))
    return f"""
You are a Werewolf role probability estimator.
Return exactly one compact JSON object. No markdown. No explanation.

Player codes:
{mapping}

Role vector order:
[V,W,S,M,C,H]
V=Villager, W=Werewolf, S=Seer, M=Medium, C=Madman, H=Hunter.

Rules:
- Werewolves: 3 if player count >=13, else 2.
- Seer=1, Medium=1.
- Madman and Hunter exist only if player count >=11.
- Madman is not Werewolf; Seer/Medium human result does not prove Villager.
- Soft white/black reads are weak discussion reads, not hard results.
- Hard result only if explicitly result/divination/check/verdict/role report.
- If multiple Seer/Medium claimers exist, at most one of each is true; fakes are often W or C.

Evidence JSON:
{evidence}

Output schema exactly:
{{"p":{{"P0":[0,0,0,0,0,0]}}}}

Constraints:
- Include every code exactly once: {codes}
- Each vector has exactly 6 numbers in [0,1].
- Use codes only. Do not output player names, role names, evidence, comments, or extra keys.
""".strip()


def build_compact_wolf_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None) -> str:
    mapping = _players_code_text(players)
    evidence = _compact_game_payload(
        players=players,
        evidence_cards=evidence_cards,
        parsed=parsed,
        daily_states=daily_states,
        formation_analysis=formation_analysis,
        max_chars=14000,
    )
    codes = ",".join(f"P{i}" for i in range(len(players)))
    return f"""
You are a Werewolf detection agent.
Return exactly one compact JSON object. No markdown. No explanation.

Player codes:
{mapping}

Goal:
Estimate each code's probability of being Werewolf only.
Madman is wolf-aligned but not Werewolf, so do not score Madman as Werewolf.

Use hard evidence first: claims, hard Seer/Medium results, deaths, votes, formation.
Use soft white/black reads only weakly.

Evidence JSON:
{evidence}

Output schema exactly:
{{"w":{{"P0":0.0}}}}

Constraints:
- Include every code exactly once: {codes}
- Values are numbers in [0,1].
- Use codes only. Do not output player names, wolf_score key, evidence arrays, comments, or extra keys.
""".strip()


# -----------------------------
# Compact JSON parsers
# -----------------------------

def _role_probs_from_raw(raw: Any) -> Dict[str, float]:
    # New compact format: [V,W,S,M,C,H]
    if isinstance(raw, list):
        return {
            role: _clamp01(raw[i] if i < len(raw) else 0.0)
            for i, role in enumerate(ROLE_ORDER)
        }

    # Old / alternative formats: {"Villager":..., "W":...}
    if isinstance(raw, dict):
        role_probs = {}
        for role in ROLE_ORDER:
            role_probs[role] = _clamp01(raw.get(role, 0.0))
        for short, role in ROLE_SHORT_KEYS.items():
            if short in raw:
                role_probs[role] = _clamp01(raw.get(short, role_probs.get(role, 0.0)))
        return role_probs

    return {}


def _extract_role_raw_players(data: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    code_to_player, _ = _player_code_maps(players)
    raw = data.get("p", data.get("players", {}))
    if not isinstance(raw, dict):
        return {}

    out = {}
    for k, v in raw.items():
        name = code_to_player.get(str(k), str(k))
        out[name] = v
    return out


def _extract_wolf_raw_players(data: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    code_to_player, _ = _player_code_maps(players)
    raw = data.get("w", data.get("players", {}))
    if not isinstance(raw, dict):
        return {}

    out = {}
    for k, v in raw.items():
        name = code_to_player.get(str(k), str(k))
        out[name] = v
    return out


def _wolf_score_from_raw(raw: Any, default: float = 0.2) -> float:
    # New compact format: number
    if isinstance(raw, (int, float, str)):
        return _clamp01(raw, default=default)

    # Old / alternative formats.
    if isinstance(raw, dict):
        for key in ("wolf_score", "W", "w", "score"):
            if key in raw:
                return _clamp01(raw.get(key), default=default)
    return default


# -----------------------------
# Agents
# -----------------------------

class RoleReasoningAgent:
    """Estimates role probability for every player."""

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, Any],
        parsed: Dict[str, Any],
        daily_states: Dict[str, Any] = None,
        formation_analysis=None,
        game_index: str = "unknown",
    ) -> Dict[str, Dict[str, float]]:
        try:
            prompt = build_compact_role_prompt(
                players,
                evidence_cards,
                parsed,
                daily_states=daily_states,
                formation_analysis=formation_analysis,
            )
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
            raw_players = _extract_role_raw_players(data, players)

            for p in players:
                role_probs = _role_probs_from_raw(raw_players.get(p, {}))

                # If the model omitted a player or output bad data, use fallback prior for that player.
                if not role_probs or sum(role_probs.values()) <= 0:
                    role_probs = fallback_role_prior([p], parsed).get(p, {})

                # Rule-based adjustment from explicit parser facts.
                if player_has_claim(parsed, p, "Seer CO"):
                    role_probs["Seer"] += 0.12
                    role_probs["Madman"] += 0.12
                    role_probs["Werewolf"] += 0.10
                    role_probs["Villager"] *= 0.65
                    role_probs["Hunter"] *= 0.60

                if player_has_claim(parsed, p, "Medium CO"):
                    role_probs["Medium"] += 0.15
                    role_probs["Madman"] += 0.08
                    role_probs["Werewolf"] += 0.10
                    role_probs["Villager"] *= 0.70
                    role_probs["Hunter"] *= 0.65

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
    """Estimates wolf_score for every player."""

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Dict[str, Any],
        parsed: Dict[str, Any],
        daily_states: Dict[str, Any] = None,
        formation_analysis=None,
        game_index: str = "unknown",
    ) -> Dict[str, float]:
        try:
            prompt = build_compact_wolf_prompt(
                players,
                evidence_cards,
                parsed,
                daily_states=daily_states,
                formation_analysis=formation_analysis,
            )
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
            raw_players = _extract_wolf_raw_players(data, players)
            fallback_scores = fallback_wolf_scores(players, parsed)

            for p in players:
                score = _wolf_score_from_raw(raw_players.get(p, fallback_scores.get(p, 0.2)), default=fallback_scores.get(p, 0.2))

                # Rule-based adjustment from hard divination.
                received = [
                    r for r in parsed.get("all_divinations", [])
                    if isinstance(r, dict) and r.get("target") == p
                ]
                black_count = sum(1 for r in received if r.get("result") == "werewolf")
                white_count = sum(1 for r in received if r.get("result") == "human")

                if black_count > 0:
                    score += 0.18 * black_count
                if white_count > 0:
                    score -= 0.08 * white_count

                # Rule-based weak adjustment from soft reads.
                soft_received = [
                    r for r in parsed.get("all_soft_reads", [])
                    if isinstance(r, dict) and r.get("target") == p
                ]
                likely_wolf_reads = sum(1 for r in soft_received if r.get("read") == "likely_werewolf")
                likely_villager_reads = sum(1 for r in soft_received if r.get("read") == "likely_villager")

                score += min(0.12, 0.025 * likely_wolf_reads)
                score -= min(0.10, 0.020 * likely_villager_reads)

                # Panda: mixed hard result, keep high but uncertain.
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


class FormationAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        daily_states: Dict[str, Any],
        game_index: str = "unknown",
    ) -> Dict[str, Any]:
        try:
            prompt = build_formation_prompt(players, daily_states)
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "formation_agent", prompt)

            data, raw_output = self.llm.generate_json_with_raw(prompt)
            if self.debug_logger:
                self.debug_logger.save_raw_output(game_index, "formation_agent", raw_output)
                self.debug_logger.save_json(game_index, "formation_agent", data)
            return data if isinstance(data, dict) else {}

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "formation_agent", e)

            if isinstance(daily_states, dict) and daily_states:
                last_day = list(daily_states.keys())[-1]
                s = daily_states[last_day]
                return {
                    "latest_formation": s.get("formation"),
                    "seer_claimers": s.get("seer_claimers", []),
                    "medium_claimers": s.get("medium_claimers", []),
                    "gray_players": s.get("gray_players", []),
                    "likely_patterns": [],
                    "claimant_alignment_estimates": {},
                    "gray_wolf_slots_estimate": s.get("likely_gray_wolf_slots_upper_bound", None),
                }
            return {}


class EventExtractionAgent:
    """Converts raw day/chunk messages into structured, low-noise events."""

    KEYWORDS = [
        "seer", "medium", "not seer", "not medium", "co", "claim", "result",
        "divination", "white", "black", "wolf", "human", "vote", "execute", "lynch",
        "●", "○", "▼", "▽", "占", "霊", "靈", "判定", "結果", "白", "黒", "狼"
    ]

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def extract(
        self,
        game_index: str,
        day: str,
        players: List[str],
        messages: List[Dict[str, Any]],
        previous_state: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        # Skip chunks with no strategic keywords. This preserves the agent while avoiding RP/food/noise calls.
        joined = "\n".join(str(m.get("text", "")) for m in messages).lower()
        if not any(k.lower() in joined for k in self.KEYWORDS):
            return {"game_id": game_index, "day": str(day), "events": [], "noise_summary": "Skipped non-strategic chunk."}

        prompt = build_event_extraction_prompt(
            game_index=game_index,
            day=day,
            players=players,
            messages=messages,
            previous_state=previous_state,
        )
        agent_name = f"event_extractor_day_{day}"
        if self.debug_logger:
            self.debug_logger.save_prompt(game_index, agent_name, prompt)

        try:
            data, raw_output = self.llm.generate_json_with_raw(prompt)
            if self.debug_logger:
                self.debug_logger.save_raw_output(game_index, agent_name, raw_output)
                self.debug_logger.save_json(game_index, agent_name, data)
            if isinstance(data, dict):
                data.setdefault("game_id", game_index)
                data.setdefault("day", str(day))
                data.setdefault("events", [])
                return data
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, agent_name, e)

        events = []
        for msg in messages:
            text = str(msg.get("text", ""))
            speaker = msg.get("speaker")
            if any(k.lower() in text.lower() for k in self.KEYWORDS):
                events.append({
                    "type": "other_relevant",
                    "speaker": speaker,
                    "target": None,
                    "content": text[:220],
                    "evidence": f"#{msg.get('no')} {text[:120]}",
                    "confidence": 0.35,
                })
        return {"game_id": game_index, "day": str(day), "events": events[:60], "noise_summary": "LLM extraction failed; used keyword fallback."}


class StateTrackerAgent:
    """Merges extracted events into a cumulative public board state."""

    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def update(
        self,
        game_index: str,
        day: str,
        players: List[str],
        previous_state: Dict[str, Any],
        extracted_events: Dict[str, Any],
    ) -> Dict[str, Any]:
        # No need to call LLM for empty extracted chunks.
        if not extracted_events or not extracted_events.get("events"):
            return dict(previous_state or {})

        prompt = build_state_tracker_prompt(
            players=players,
            previous_state=previous_state,
            extracted_events=extracted_events,
            day=day,
        )
        agent_name = f"state_tracker_day_{day}"
        if self.debug_logger:
            self.debug_logger.save_prompt(game_index, agent_name, prompt)

        try:
            data, raw_output = self.llm.generate_json_with_raw(prompt)
            if self.debug_logger:
                self.debug_logger.save_raw_output(game_index, agent_name, raw_output)
                self.debug_logger.save_json(game_index, agent_name, data)
            if isinstance(data, dict):
                return data
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.save_error(game_index, agent_name, e)

        return dict(previous_state or {})
