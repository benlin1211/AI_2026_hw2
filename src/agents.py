import json
from typing import Any, Dict, List, Optional

try:
    from src.prompts import build_formation_prompt
except Exception:
    build_formation_prompt = None

VALID_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]
ROLE_VECTOR_ORDER = VALID_ROLES


def _short_json(obj: Any, max_chars: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _player_maps(players: List[str]):
    code_to_name = {f"P{i}": p for i, p in enumerate(players)}
    name_to_code = {p: c for c, p in code_to_name.items()}
    return code_to_name, name_to_code


def _encode_obj(obj: Any, name_to_code: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        return name_to_code.get(obj, obj)
    if isinstance(obj, list):
        return [_encode_obj(x, name_to_code) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode_obj(v, name_to_code) for k, v in obj.items()}
    return obj


def get_claim_name(c):
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def player_has_claim(parsed: Dict[str, Any], player: str, claim_name: str) -> bool:
    claims = parsed.get("all_claims", {}).get(player, [])
    return any(get_claim_name(c) == claim_name for c in claims)


def fallback_role_prior(players: List[str], parsed: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out = {}
    for p in players:
        role_probs = {
            "Villager": 0.52,
            "Werewolf": 0.18,
            "Seer": 0.07,
            "Medium": 0.07,
            "Madman": 0.08,
            "Hunter": 0.08,
        }
        if player_has_claim(parsed, p, "Seer CO"):
            role_probs["Seer"] += 0.22
            role_probs["Madman"] += 0.12
            role_probs["Werewolf"] += 0.08
            role_probs["Villager"] *= 0.65
        if player_has_claim(parsed, p, "Medium CO"):
            role_probs["Medium"] += 0.25
            role_probs["Madman"] += 0.06
            role_probs["Werewolf"] += 0.08
            role_probs["Villager"] *= 0.65
        if player_has_claim(parsed, p, "Not Seer/Medium"):
            role_probs["Seer"] *= 0.25
            role_probs["Medium"] *= 0.25
            role_probs["Villager"] += 0.08

        received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
        role_probs["Werewolf"] += 0.08 * sum(1 for r in received if r.get("result") == "werewolf")
        role_probs["Werewolf"] -= 0.04 * sum(1 for r in received if r.get("result") == "human")
        out[p] = normalize_role_probs(role_probs)
    return out


def fallback_wolf_scores(players: List[str], parsed: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            out[p] = 0.01
            continue
        score = 0.20
        received = [r for r in parsed.get("all_divinations", []) if isinstance(r, dict) and r.get("target") == p]
        black_count = sum(1 for r in received if r.get("result") == "werewolf")
        white_count = sum(1 for r in received if r.get("result") == "human")
        score += 0.18 * black_count
        score -= 0.08 * white_count
        soft_received = [r for r in parsed.get("all_soft_reads", []) if isinstance(r, dict) and r.get("target") == p]
        score += min(0.12, 0.025 * sum(1 for r in soft_received if r.get("read") == "likely_werewolf"))
        score -= min(0.10, 0.020 * sum(1 for r in soft_received if r.get("read") == "likely_villager"))
        if black_count > 0 and white_count > 0:
            score = min(0.85, max(score, 0.45))
        out[p] = max(0.03, min(0.95, score))
    return out


def normalize_role_probs(role_probs: Dict[str, Any]) -> Dict[str, float]:
    cleaned = {}
    for role in VALID_ROLES:
        try:
            v = float(role_probs.get(role, 0.0))
        except Exception:
            v = 0.0
        cleaned[role] = max(0.0, min(1.0, v))
    s = sum(cleaned.values())
    if s <= 0:
        cleaned = {"Villager": 0.70, "Werewolf": 0.18, "Seer": 0.03, "Medium": 0.03, "Madman": 0.03, "Hunter": 0.03}
    else:
        cleaned = {k: v / s for k, v in cleaned.items()}
    return cleaned


def _parse_role_players(data: Dict[str, Any], players: List[str]) -> Dict[str, Dict[str, float]]:
    code_to_name, _ = _player_maps(players)
    raw_players = data.get("p", data.get("players", {}))
    out = {}
    for p in players:
        code = None
        for c, name in code_to_name.items():
            if name == p:
                code = c
                break
        raw = None
        if isinstance(raw_players, dict):
            raw = raw_players.get(code, raw_players.get(p))
        role_probs = {}
        if isinstance(raw, list):
            for i, role in enumerate(ROLE_VECTOR_ORDER):
                try:
                    role_probs[role] = float(raw[i]) if i < len(raw) else 0.0
                except Exception:
                    role_probs[role] = 0.0
        elif isinstance(raw, dict):
            for role in VALID_ROLES:
                try:
                    role_probs[role] = float(raw.get(role, raw.get(role[0], 0.0)))
                except Exception:
                    role_probs[role] = 0.0
        out[p] = normalize_role_probs(role_probs)
    return out


def _parse_wolf_players(data: Dict[str, Any], players: List[str]) -> Dict[str, float]:
    code_to_name, _ = _player_maps(players)
    raw_players = data.get("w", data.get("players", {}))
    out = {}
    for p in players:
        code = None
        for c, name in code_to_name.items():
            if name == p:
                code = c
                break
        raw = raw_players.get(code, raw_players.get(p, 0.20)) if isinstance(raw_players, dict) else 0.20
        if isinstance(raw, dict):
            raw = raw.get("wolf_score", raw.get("W", 0.20))
        try:
            score = float(raw)
        except Exception:
            score = 0.20
        out[p] = max(0.0, min(1.0, score))
    return out


def build_role_v2_prompt(players: List[str], objective_pack: Dict[str, Any], parsed: Optional[Dict[str, Any]] = None) -> str:
    code_to_name, name_to_code = _player_maps(players)
    encoded_objective = _encode_obj(objective_pack or {}, name_to_code)
    mapping_text = _short_json(code_to_name, 3000)
    objective_text = _short_json(encoded_objective, 18000)

    return f"""
You are RoleAgent for an offline Werewolf log prediction task.
Return exactly one compact JSON object. No markdown. No explanation.

Players mapping:
{mapping_text}

Role vector order:
[V,W,S,M,C,H]
V=Villager, W=Werewolf, S=Seer, M=Medium, C=Madman, H=Hunter.

Objective public facts, Day1+ only:
{objective_text}

Rules:
- There is exactly 1 true Seer and 1 true Medium.
- Werewolf count is in role_counts.
- Madman is human and may fake Seer/Medium; Madman is not Werewolf.
- Seer human result means non-Werewolf, not necessarily Villager.
- Soft white/black reads are weak and are not hard results.
- Use claim_timeline and hard_results more than tone.
- Do not infer Hunter unless there is strong explicit evidence; otherwise keep Hunter moderate/low.
- If several players claim Seer/Medium, at most one is true and the rest are usually Werewolf or Madman.

Output schema:
{{"p":{{"P0":[0,0,0,0,0,0],"P1":[0,0,0,0,0,0]}}}}

Constraints:
- Include every P-code exactly once.
- Each vector has exactly 6 numeric values from 0 to 1.
- Do not use player names as keys.
""".strip()


def build_wolf_v2_prompt(
    players: List[str],
    objective_pack: Dict[str, Any],
    interaction_graph: Dict[str, Any],
    stepwise_scores: Dict[str, Any],
    wolf_prior: Dict[str, float],
) -> str:
    code_to_name, name_to_code = _player_maps(players)
    mapping_text = _short_json(code_to_name, 3000)
    objective_text = _short_json(_encode_obj(objective_pack or {}, name_to_code), 12000)
    interaction_text = _short_json(_encode_obj(interaction_graph or {}, name_to_code), 14000)
    stepwise_text = _short_json(_encode_obj(stepwise_scores or {}, name_to_code), 10000)
    prior_text = _short_json(_encode_obj(wolf_prior or {}, name_to_code), 4000)

    return f"""
You are WolfRankAgent for an offline Werewolf log prediction task.
Return exactly one compact JSON object. No markdown. No explanation.

Players mapping:
{mapping_text}

Goal:
Estimate each player's probability of being an actual Werewolf. This is a ranking task.
Madman is wolf-aligned but is NOT a Werewolf; do not score Madman as Werewolf solely for fake-claim behavior.

Objective public facts:
{objective_text}

Interaction graph:
{interaction_text}

Stepwise behavior scores:
{stepwise_text}

Rule-based wolf prior:
{prior_text}

Guidance:
- Hard black from credible Seer raises Werewolf probability; hard human lowers it but does not prove Villager.
- In contested Seer/Medium formations, fake claimers may be Werewolf or Madman.
- Early independent pressure is more meaningful than late consensus pressure.
- Being pressured by many independent sources raises suspicion, but villagers can be wrongly suspected.
- Defending or townreading a high-risk claimant can be wolfy.
- Being attacked by suspicious players is anti-wolf evidence.
- Do not let role assignment dominate; output continuous scores, not only 0/1.

Output schema:
{{"w":{{"P0":0.0,"P1":0.0}}}}

Constraints:
- Include every P-code exactly once.
- Values must be numeric from 0 to 1.
- Do not output evidence arrays or explanations.
""".strip()


class RoleReasoningAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Optional[Dict[str, Any]] = None,
        parsed: Optional[Dict[str, Any]] = None,
        daily_states: Optional[Dict[str, Any]] = None,
        formation_analysis: Any = None,
        game_index: str = "unknown",
        objective_pack: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        parsed = parsed or {}
        try:
            prompt = build_role_v2_prompt(players, objective_pack or {}, parsed)
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "role_agent_v2", prompt)
            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)
                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "role_agent_v2", raw_output)
                    self.debug_logger.save_json(game_index, "role_agent_v2", data)
                out = _parse_role_players(data if isinstance(data, dict) else {}, players)
            except Exception as e:
                print(f"[WARN] RoleReasoningAgent fallback: {e}")
                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "role_agent_v2", e)
                out = fallback_role_prior(players, parsed)

            # Light rule correction from explicit claims, preserving LLM ranking.
            for p in players:
                probs = dict(out.get(p, {}))
                if player_has_claim(parsed, p, "Seer CO"):
                    probs["Seer"] += 0.10
                    probs["Madman"] += 0.08
                    probs["Werewolf"] += 0.06
                    probs["Villager"] *= 0.75
                    probs["Hunter"] *= 0.70
                if player_has_claim(parsed, p, "Medium CO"):
                    probs["Medium"] += 0.12
                    probs["Madman"] += 0.05
                    probs["Werewolf"] += 0.05
                    probs["Villager"] *= 0.75
                    probs["Hunter"] *= 0.70
                if player_has_claim(parsed, p, "Not Seer/Medium"):
                    probs["Seer"] *= 0.35
                    probs["Medium"] *= 0.35
                out[p] = normalize_role_probs(probs)

            if self.debug_logger:
                self.debug_logger.save_json(game_index, "role_agent_v2_final", out)
            return out
        except Exception as e:
            print(f"[WARN] RoleReasoningAgent outer fallback: {e}")
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "role_agent_v2_outer", e)
            return fallback_role_prior(players, parsed)


class WolfReasoningAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(
        self,
        players: List[str],
        evidence_cards: Optional[Dict[str, Any]] = None,
        parsed: Optional[Dict[str, Any]] = None,
        daily_states: Optional[Dict[str, Any]] = None,
        formation_analysis: Any = None,
        game_index: str = "unknown",
        objective_pack: Optional[Dict[str, Any]] = None,
        interaction_graph: Optional[Dict[str, Any]] = None,
        stepwise_scores: Optional[Dict[str, Any]] = None,
        wolf_prior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        parsed = parsed or {}
        try:
            prompt = build_wolf_v2_prompt(players, objective_pack or {}, interaction_graph or {}, stepwise_scores or {}, wolf_prior or {})
            if self.debug_logger:
                self.debug_logger.save_prompt(game_index, "wolf_agent_v2", prompt)
            try:
                data, raw_output = self.llm.generate_json_with_raw(prompt)
                if self.debug_logger:
                    self.debug_logger.save_raw_output(game_index, "wolf_agent_v2", raw_output)
                    self.debug_logger.save_json(game_index, "wolf_agent_v2", data)
                out = _parse_wolf_players(data if isinstance(data, dict) else {}, players)
            except Exception as e:
                print(f"[WARN] WolfReasoningAgent fallback: {e}")
                if self.debug_logger:
                    self.debug_logger.save_error(game_index, "wolf_agent_v2", e)
                out = fallback_wolf_scores(players, parsed)

            # Blend lightly with prior inside agent so downstream always receives usable ranks.
            if wolf_prior:
                for p in players:
                    out[p] = max(0.0, min(1.0, 0.72 * float(out.get(p, 0.2)) + 0.28 * float(wolf_prior.get(p, 0.2))))
            if self.debug_logger:
                self.debug_logger.save_json(game_index, "wolf_agent_v2_final", out)
            return out
        except Exception as e:
            print(f"[WARN] WolfReasoningAgent outer fallback: {e}")
            if self.debug_logger:
                self.debug_logger.save_error(game_index, "wolf_agent_v2_outer", e)
            return fallback_wolf_scores(players, parsed)


class FormationAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def predict(self, players: List[str], daily_states: Dict[str, Any], game_index: str = "unknown") -> Dict[str, Any]:
        if build_formation_prompt is None:
            return {}
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
                }
            return {}


# Compatibility stubs. V2 does not use slow event/state agents by default.
class EventExtractionAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def extract(self, *args, **kwargs):
        return {"events": [], "noise_summary": "EventExtractionAgent disabled in v2."}


class StateTrackerAgent:
    def __init__(self, llm, debug_logger=None):
        self.llm = llm
        self.debug_logger = debug_logger

    def update(self, game_index, day, players, previous_state, extracted_events):
        return dict(previous_state or {})
