import json
from typing import Any, Dict, List, Optional


ROLE_NAMES = ["Villager", "Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def _players_text(players: List[str]) -> str:
    return "\n".join(f"- {p}" for p in players)


def _short_json(obj: Any, max_chars: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    if len(s) <= max_chars:
        return s
    # Keep head and tail instead of tail only.  CO formation is usually early;
    # vote/result pressure is often late.
    half = max_chars // 2
    return s[:half] + "\n...TRUNCATED_MIDDLE...\n" + s[-half:]


def _claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def _role_counts(num_players: int) -> Dict[str, int]:
    return {
        "Werewolf": 3 if num_players >= 13 else 2,
        "Seer": 1,
        "Medium": 1,
        "Madman": 1 if num_players >= 11 else 0,
        "Hunter": 1 if num_players >= 11 else 0,
        "Villager": max(0, num_players - (3 if num_players >= 13 else 2) - 1 - 1 - (1 if num_players >= 11 else 0) - (1 if num_players >= 11 else 0)),
    }


def _compact_claims(parsed: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p, claims in parsed.get("all_claims", {}).items():
        rows = []
        for c in claims if isinstance(claims, list) else []:
            if isinstance(c, dict):
                rows.append({
                    "claim": c.get("claim"),
                    "day": c.get("day"),
                    "order": c.get("order"),
                    "time": c.get("time"),
                    "text": str(c.get("text", ""))[:240],
                })
            else:
                rows.append({"claim": str(c)})
        if rows:
            out[p] = rows
    return out


def _important_text(text: str) -> bool:
    keys = [
        "Seer", "Medium", "not Seer", "not Medium", "claim", "CO",
        "result", "verdict", "divination", "checked", "human", "werewolf",
        "white", "black", "gray", "wolf", "lynch", "execute", "vote",
        "●", "○", "▼", "▽", "占", "霊", "靈", "判定", "結果", "白", "黒", "狼",
    ]
    low = text.lower()
    return any(k.lower() in low for k in keys)


def _strategic_timeline(parsed: Dict[str, Any], max_early: int = 70, max_late: int = 70) -> List[Dict[str, Any]]:
    events = []
    for msg in parsed.get("messages", []):
        text = str(msg.get("text", ""))
        if _important_text(text):
            events.append({
                "day": msg.get("day"),
                "order": msg.get("no"),
                "time": msg.get("time"),
                "speaker": msg.get("speaker"),
                "text": text[:420],
            })
    if len(events) <= max_early + max_late:
        return events
    return events[:max_early] + [{"note": "middle strategic timeline omitted"}] + events[-max_late:]


def _compact_daily_states(daily_states: Optional[Dict[str, Any]], max_days: int = 10) -> Dict[str, Any]:
    if not isinstance(daily_states, dict):
        return {}
    out = {}
    for day in list(daily_states.keys())[-max_days:]:
        s = daily_states.get(day, {})
        if not isinstance(s, dict):
            continue
        out[day] = {
            "alive_players": s.get("alive_players", []),
            "dead_players": s.get("dead_players", []),
            "role_counts": s.get("role_counts", {}),
            "formation": s.get("formation"),
            "formation_type": s.get("formation_type"),
            "seer_claimers": s.get("seer_claimers", []),
            "medium_claimers": s.get("medium_claimers", []),
            "not_seer_medium_claimers": s.get("not_seer_medium_claimers", []),
            "gray_players": s.get("gray_players", []),
            "hard_white_targets": s.get("hard_white_targets", []),
            "hard_black_targets": s.get("hard_black_targets", []),
            "top_suspected_players": s.get("top_suspected_players", []),
            "top_execution_targets": s.get("top_execution_targets", []),
            "top_seer_targets": s.get("top_seer_targets", []),
            "cumulative_vote_summary": s.get("cumulative_vote_summary", {}),
            "cumulative_soft_read_summary": s.get("cumulative_soft_read_summary", {}),
        }
    return out


def _player_summaries(evidence_cards: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    out = {}
    if not isinstance(evidence_cards, dict):
        return out
    for p in players:
        card = evidence_cards.get(p, {})
        if isinstance(card, dict):
            out[p] = card.get("summary", card)
        else:
            out[p] = str(card)[:800]
    return out


def _compact_context(players, evidence_cards, parsed, daily_states, formation_analysis, max_chars=26000) -> str:
    ctx = {
        "role_counts": _role_counts(len(players)),
        "global": evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {},
        "formation_analysis": formation_analysis or {},
        "daily_states": _compact_daily_states(daily_states),
        "claims_with_quotes": _compact_claims(parsed),
        "hard_divinations": parsed.get("all_divinations", []),
        "votes": parsed.get("all_votes", [])[:260],
        "deaths": parsed.get("deaths", []),
        "strategic_timeline": _strategic_timeline(parsed),
        "players": _player_summaries(evidence_cards, players),
    }
    return _short_json(ctx, max_chars=max_chars)


COMMON_RULES = """
Game rules and scoring-relevant facts:
- There is exactly 1 true Seer and 1 true Medium.
- Werewolf count is 2 for up to 12 players and 3 for 13+ players.
- Madman and Hunter exist only in 11+ player games.
- Madman is human, wolf-aligned, and does not know the Werewolves. Seer/Medium results on Madman are human.
- Hunter is human and usually hides; do not infer Hunter from generic town-like behavior.
- white/black in ordinary discussion are soft reads, not hard divination results.
- Treat white/black as hard results only when the text explicitly says result, verdict, divination, check, announce, role report, 判定, 結果, 占い結果.
- A fake Seer or fake Medium is often Werewolf or Madman, not Villager/Hunter.
- In contested 2-2 formations, usually separate ability claimers from gray players; do not put all wolf probability only on claimers.
- For wolf_score, rank actual Werewolves highest; Madman is not a Werewolf.
""".strip()


def build_formation_prompt(players, daily_states, parsed=None, evidence_cards=None):
    players_text = _players_text(players)
    context = {
        "role_counts": _role_counts(len(players)),
        "daily_states": _compact_daily_states(daily_states),
    }
    if isinstance(parsed, dict):
        context["claims_with_quotes"] = _compact_claims(parsed)
        context["hard_divinations"] = parsed.get("all_divinations", [])
        context["strategic_timeline"] = _strategic_timeline(parsed, max_early=90, max_late=40)
    if isinstance(evidence_cards, dict):
        context["global"] = evidence_cards.get("_global", {})

    return f"""
You are a Werewolf board-formation analyst. Return exactly one valid JSON object and no other text.

Players:
{players_text}

{COMMON_RULES}

Evidence:
{_short_json(context, max_chars=22000)}

Task:
1. Verify self-claims from quoted text.
2. Identify the latest formation, such as 1-1, 2-1, 2-2, 3-1.
3. List Seer claimers, Medium claimers, gray players, and likely hidden Werewolf slots.
4. Estimate common alignment patterns, but do not assign final roles to every player.

Output schema:
{{
  "verified_facts": {{
    "latest_formation": "2-2",
    "seer_claimers": [],
    "medium_claimers": [],
    "gray_players": [],
    "invalid_or_uncertain_claims": []
  }},
  "likely_patterns": [
    {{"pattern": "seer_true_madman_medium_true_wolf", "probability": 0.0, "reason": "short"}}
  ],
  "claimant_alignment_estimates": {{
    "<player>": {{"true_role": 0.0, "wolf_fake": 0.0, "madman_fake": 0.0, "reason": "short"}}
  }},
  "gray_wolf_slots_estimate": 1,
  "notes": "short"
}}
""".strip()


def build_role_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None):
    players_text = _players_text(players)
    context_text = _compact_context(players, evidence_cards, parsed, daily_states, formation_analysis, max_chars=30000)
    return f"""
You are a role probability estimator for a Werewolf game. Return exactly one valid JSON object and no other text.

Players:
{players_text}

Allowed roles: {', '.join(ROLE_NAMES)}

{COMMON_RULES}

Reasoning order:
1. First use formation_analysis and verified claim quotes to separate ability claimers from gray players.
2. Determine true Seer and true Medium candidates among claimers.
3. Determine whether fake claimers are more likely Werewolf or Madman.
4. Assign ordinary gray players among Villager, Werewolf, Hunter, and possible special roles only if evidence supports it.
5. Keep role probabilities calibrated. They do not need to sum to role counts; the solver handles final constraints.

Evidence:
{context_text}

Output schema:
{{
  "verified_facts": {{
    "formation": "string",
    "seer_claimers": [],
    "medium_claimers": [],
    "gray_players": []
  }},
  "players": {{
    "<player>": {{
      "Villager": 0.0,
      "Werewolf": 0.0,
      "Seer": 0.0,
      "Medium": 0.0,
      "Madman": 0.0,
      "Hunter": 0.0,
      "reason": "short evidence-based reason"
    }}
  }}
}}
""".strip()


def build_wolf_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None):
    players_text = _players_text(players)
    context_text = _compact_context(players, evidence_cards, parsed, daily_states, formation_analysis, max_chars=30000)
    ww_count = _role_counts(len(players))["Werewolf"]
    return f"""
You are a Werewolf probability ranking agent. Return exactly one valid JSON object and no other text.

Players:
{players_text}

This game has {ww_count} actual Werewolves.

{COMMON_RULES}

Important objective:
- wolf_score is the probability/ranking that the player is an actual Werewolf, not merely wolf-aligned.
- Madman should not receive a high wolf_score only because they help Werewolves.
- In 2-2 or 3-1 formations, reason about ability wolf slots and gray wolf slots separately.
- Keep a clear ranking: top {ww_count} candidates should be the most likely actual Werewolves.

Reasoning order:
1. Identify formation and gray players from formation_analysis.
2. Estimate which ability claimant slot is likely Werewolf, if any.
3. Estimate which gray players fit the remaining Werewolf slots.
4. Use hard results strongly, soft reads weakly, and vote/pressure patterns moderately.

Evidence:
{context_text}

Output schema:
{{
  "wolf_slots": {{
    "expected_werewolf_count": {ww_count},
    "ability_wolf_slots": 0,
    "gray_wolf_slots": 0,
    "reason": "short"
  }},
  "wolf_ranking": [
    {{"player": "<player>", "wolf_score": 0.0, "reason": "short"}}
  ],
  "players": {{
    "<player>": {{"wolf_score": 0.0, "reason": "short"}}
  }}
}}
""".strip()
