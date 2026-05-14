import json


def _build_timeline_for_prompt(parsed, max_messages=120):
    events = []

    for msg in parsed.get("messages", []):
        text = msg.get("text", "")

        # Keep only strategically relevant messages.
        if re_search_any(text, [
            "Seer", "Medium", "not Seer", "not Medium",
            "claim", "CO", "result", "verdict", "divination",
            "●", "○", "▼", "▽",
            "white", "black", "wolf", "human",
            "lynch", "execute", "vote",
            "占", "霊", "判定", "結果", "白", "黒", "狼"
        ]):
            events.append({
                "day": msg.get("day"),
                "order": msg.get("no"),
                "time": msg.get("time"),
                "speaker": msg.get("speaker"),
                "text": text[:350],
            })

    # Keep early formation events and later key events.
    return events[:max_messages]


def re_search_any(text, keywords):
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _compact_daily_states(daily_states, max_days=10):
    """
    Keep board-state fields that help reasoning.
    Avoid dumping too much text.
    """
    if not isinstance(daily_states, dict):
        return {}

    out = {}
    days = list(daily_states.keys())[-max_days:]

    for day in days:
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
            "ability_claimers": s.get("ability_claimers", []),
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


def _compact_evidence_for_prompt(
    evidence_cards,
    parsed=None,
    daily_states=None,
    max_chars=18000,
):
    compact = {}

    if isinstance(evidence_cards, dict) and "_global" in evidence_cards:
        compact["_global"] = evidence_cards["_global"]

    if daily_states is not None:
        compact["daily_states"] = _compact_daily_states(daily_states)

    if parsed is not None:
        compact["strategic_timeline"] = _build_timeline_for_prompt(parsed)

    players_compact = {}
    if isinstance(evidence_cards, dict):
        for k, v in evidence_cards.items():
            if k == "_global":
                continue
            if isinstance(v, dict):
                players_compact[k] = {
                    "summary": v.get("summary", {}),
                }

    compact["players"] = players_compact
    return _short_json(compact, max_chars=max_chars)

def _short_json(obj, max_chars=12000):
    """
    Convert object to compact JSON string and truncate if needed.
    Prefer keeping the tail because later game states often contain final claims/votes.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)

    if len(s) <= max_chars:
        return s

    return s[-max_chars:]


def _players_text(players):
    return "\n".join(f"- {p}" for p in players)


def build_role_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None):
    players_text = _players_text(players)

    evidence_text = _compact_evidence_for_prompt(
        evidence_cards,
        parsed=parsed,
        daily_states=daily_states,
        max_chars=26000,
    )

    last_daily_state = {}
    if isinstance(daily_states, dict) and daily_states:
        last_day = list(daily_states.keys())[-1]
        last_daily_state = daily_states[last_day]

    parsed_compact = {
        "global": evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {},
        "last_daily_state": last_daily_state,
        "deaths": parsed.get("deaths", []),
        "all_divinations": parsed.get("all_divinations", []),
        "all_votes": parsed.get("all_votes", [])[:200],
        "llm_events": parsed.get("llm_events", [])[-12:],
        "llm_state": parsed.get("llm_state", {}),
    }

    parsed_text = _short_json(parsed_compact, max_chars=8000)
    formation_text = _short_json(formation_analysis or {}, max_chars=6000)

    prompt = f"""
You are a role probability estimator for a Werewolf game.

Return exactly one JSON object.
No markdown. No explanation.

Players:
{players_text}

Allowed roles:
Villager, Werewolf, Seer, Medium, Madman, Hunter

Game setup and role counts:
- The number of Werewolves depends on player count:
  - 2 Werewolves if there are up to 12 players.
  - 3 Werewolves if there are 13 or more players.
- Madman exists only if there are 11 or more players.
- Hunter exists only if there are 11 or more players.
- There is normally 1 Seer and 1 Medium.
- All remaining players are Villagers.

Win conditions:
- Villagers win if all Werewolves are executed.
- Werewolves win if the number of villagers becomes less than or equal to the number of Werewolves.

Glossary:
- white = likely villager in ordinary discussion.
- black = likely werewolf in ordinary discussion.
- gray = unresolved player.
- GS = ranking from white to black among unresolved players.
- CO = claim.
- will = prewritten role reveal or instructions if attacked or killed.
- confirmed town = role-confirmed non-wolf from village perspective.
- panda = one hard white result and one hard black result on the same target.

Important glossary distinction:
- "white" and "black" are usually soft reads in ordinary discussion.
- Soft white/black reads are not confirmed results.
- Treat white/black as hard Seer or Medium results only when the evidence explicitly says result, divination, check, verdict, or role report.
- Do not create panda from soft reads.

Role knowledge and incentives:

Villager:
- Knows only their own role.
- Does not know who the Werewolves, Seer, Medium, Madman, or Hunter are.
- Has no night ability.
- Wants to identify and execute Werewolves.
- Likely behavior:
  - asks questions,
  - compares contradictions,
  - builds GS,
  - reacts with uncertainty because they lack hidden information.

Werewolf:
- Knows their own role.
- Knows the other Werewolves.
- Can communicate privately with other Werewolves at night.
- Each night, the Werewolves choose one human player to attack.
- Does not know who the Madman is.
- Wants to avoid execution, reduce confirmed town, kill or discredit Seer/Medium, and reach parity.
- Likely behavior:
  - may fake Seer or Medium,
  - may protect partners,
  - may soft-bus partners,
  - may push villagers as execution targets,
  - may redirect discussion away from partners,
  - may search for Seer, Medium, or Hunter,
  - may create confusion around claims and results.

Seer:
- Knows their own role.
- Each night chooses one player to divine.
- Learns whether the target is Werewolf or human.
- Does not know the full role of a human result; human can be Villager, Medium, Hunter, Madman, or Seer.
- Does not know the full Werewolf team unless divined.
- Wants to survive and leave accurate divination results.
- Likely behavior:
  - cares about divination target choice,
  - explains results,
  - protects credibility,
  - wants village to use results correctly,
  - may be attacked by Werewolves if believed true.

Medium:
- Knows their own role.
- Learns whether a player who died by execution or sudden death was Werewolf or human.
- Does not learn the exact role of a human result.
- Does not learn the alignment of players killed by Werewolf attack unless rules/log explicitly say so.
- If uncontested, often becomes confirmed town.
- Wants to organize voting, preserve reliable information, and interpret executions.
- Likely behavior:
  - summarizes claims and votes,
  - asks for clear execution decisions,
  - reports execution/sudden-death results,
  - may become a night-kill target if confirmed.

Madman:
- Is human.
- Is aligned with Werewolves.
- Wins if Werewolves win.
- Does not know who the Werewolves are.
- Werewolves do not know who the Madman is.
- Seer result on Madman should be human, not Werewolf.
- Medium result on executed Madman should be human, not Werewolf.
- Wants to confuse the village and help Werewolves indirectly.
- Likely behavior:
  - may fake Seer or Medium,
  - may create fake black results,
  - may defend bad logic,
  - may waste executions,
  - may accidentally attack or black-result a real Werewolf because Madman does not know them.

Hunter:
- Knows their own role.
- Each night chooses one player to protect from Werewolf attack.
- Does not know whether the protection succeeded.
- Does not know who the Werewolves are.
- Wants to protect likely confirmed town, true Seer, or true Medium.
- Usually avoids exposing their role unless necessary.
- Hunter evidence is weak unless protection-related discussion, guard logic, or explicit claim appears.
- Do not overfit Hunter from ordinary town-like behavior.

Board-state guidance:
- daily_states is the cumulative board state after each day.
- Use daily_states to understand formation, claimers, gray players, deaths, hard results, and vote pressure.
- Use player evidence cards to verify player-specific behavior.
- If daily_states and player evidence conflict, prefer explicit quoted claim/result evidence from player cards, but use daily_states for global formation and cumulative state.
- In contested formations, separate ability claimers from gray players before estimating Werewolf probability.

Game evidence:
{evidence_text}

Parsed game facts:
{parsed_text}

Formation analysis:
{formation_text}

Use formation_analysis as a high-level hypothesis, not as hard truth.
If it conflicts with explicit claim/result evidence, prefer explicit evidence.

Parser reliability warning:
- Extracted claims, soft reads, and deaths may contain false positives.
- Treat a role claim as strong only if the quoted text is a first-person self-claim by the speaker.
- If the text says another player "made a Seer claim" or discusses "seer claim order", do not treat the speaker as a Seer claimant.
- If a player has both "Seer CO" and "Not Seer/Medium", inspect the quote text and prefer the explicit self-denial unless there is a later explicit self-claim.
- Treat death events as valid only when the extracted name exactly matches a listed player.
- Soft white/black reads are noisy; use them only as weak evidence.

Reasoning rules:
- Use hard claims, hard divination results, Medium results, deaths, and votes before tone.
- Use soft reads only as weak evidence.
- A human result from Seer or Medium does not prove Villager; it only means not Werewolf.
- Madman is human-aligned by result but wolf-aligned by win condition.
- If a player is hard-black by a credible Seer, Werewolf probability rises.
- If a player is hard-white by a credible Seer, Werewolf probability falls, but Madman/Hunter/Medium/Villager remain possible.
- If a player is a confirmed Medium with no counterclaim, Medium probability should be high.
- If multiple players claim Seer, at most one is true Seer; the others are likely Werewolf or Madman.
- Werewolves may fake claims, but Madman may also fake claims without knowing wolves.
- Hunter should usually have low confidence unless there is explicit evidence.

Output schema:
{{
  "players": {{
    "<exact player name>": {{
      "Villager": 0.0,
      "Werewolf": 0.0,
      "Seer": 0.0,
      "Medium": 0.0,
      "Madman": 0.0,
      "Hunter": 0.0
    }}
  }}
}}

Constraints:
- Include every listed player exactly once.
- Use only exact player names from the Players list.
- Use only the six allowed role keys.
- Each value must be a number from 0 to 1.
""".strip()

    return prompt


def build_wolf_prompt(players, evidence_cards, parsed, daily_states=None, formation_analysis=None):
    players_text = _players_text(players)

    evidence_text = _compact_evidence_for_prompt(evidence_cards, parsed=parsed, daily_states=daily_states, max_chars=22000)

    parsed_compact = {
        "global": evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {},
        "deaths": parsed.get("deaths", []),
        "all_divinations": parsed.get("all_divinations", []),
        "all_votes": parsed.get("all_votes", [])[:200],
        "llm_events": parsed.get("llm_events", [])[-12:],
        "llm_state": parsed.get("llm_state", {}),
    }
    parsed_text = _short_json(parsed_compact, max_chars=8000)
    formation_text = _short_json(formation_analysis or {}, max_chars=6000)

    prompt = f"""
You are a Werewolf detection agent.

Return exactly one JSON object.
No markdown. No explanation.

Players:
{players_text}

Goal:
Estimate each player's continuous probability of being a Werewolf.
This is a ranking task. Do not output only 0 or 1.

Key hidden information:
- Werewolves know the other Werewolves and can coordinate privately.
- Werewolves do not know the Madman.
- Madman is human and wolf-aligned, but does not know who the Werewolves are.
- Seer learns whether one target is Werewolf or human.
- Seer human result does not distinguish Villager, Medium, Hunter, Madman, or Seer.
- Medium learns whether executed or sudden-death players were Werewolf or human.
- Hunter protects one player at night but does not know whether protection succeeded.
- Villagers know only their own role.

Win incentives:
- Werewolves want to avoid execution, reduce confirmed town, kill/discredit Seer and Medium, and reach parity.
- Villagers want to execute all Werewolves.
- Madman wants Werewolves to win, often by confusing the village or faking claims.

Glossary:
- white = likely villager in ordinary discussion.
- black = likely werewolf in ordinary discussion.
- gray = unresolved.
- CO = role claim.
- GS = ranking from white to black.
- confirmed town = role-confirmed non-wolf.
- panda = one hard white result and one hard black result on the same target.
- Soft white/black reads are not confirmed results.
- Treat white/black as hard result only when explicitly tied to result, divination, check, verdict, or role report.

Wolf indicators:
- Hard-black result from a credible Seer.
- Protecting or redirecting pressure away from a likely Werewolf.
- Soft-bussing a partner without real pressure.
- Avoiding commitment near voting deadlines.
- Changing stance after public opinion shifts.
- Pushing easy villagers as execution targets.
- Creating confusion around Seer/Medium claims and results.
- Trying to identify or eliminate Seer, Medium, Hunter, or confirmed town.

Town indicators:
- Early natural suspicion on a likely Werewolf before consensus.
- Consistent reasoning before and after new evidence.
- Being pressured or attacked by likely Werewolves.
- Helping organize claims, votes, and confirmed information.
- Willingness to be checked or resolved when useful.
- Uninformed uncertainty that fits Villager knowledge.

Important:
- A Madman can look wolf-aligned but should not be scored as Werewolf solely for fake-claim behavior.
- Because Madman does not know the Werewolves, accidental pressure on a real Werewolf is possible.
- Do not treat soft "black" as equivalent to hard Seer black.
- Do not treat soft "white" as equivalent to hard Seer white.
- Use soft reads only as weak evidence.

Use daily_states as follows:
1. First identify the latest formation.
2. Separate ability claimers from gray players.
3. If formation is contested, do not score every fake-looking claimant as Werewolf; some fake claimers may be Madman.
4. For gray players, use vote pressure, soft suspicion, stance changes, and links to ability claimers.
5. hard_black_targets should raise wolf_score, but only after considering Seer credibility.
6. hard_white_targets should lower wolf_score, but human result does not prove Villager.

Game evidence:
{evidence_text}

Parsed game facts:
{parsed_text}

Formation analysis:
{formation_text}

Use formation_analysis as a high-level hypothesis, not as hard truth.
If it conflicts with explicit claim/result evidence, prefer explicit evidence.

Evidence reliability:
- Do not blindly trust extracted Seer CO or Medium CO. Verify from the quoted text whether the speaker is self-claiming or merely discussing another player's claim.
- A false parser claim should not make the player look like a fake claimant.
- Death records may include noisy extracted phrases; only exact player-name deaths should matter.
- Soft reads are weaker than votes, deaths, explicit self-claims, and hard role results.
- If many players are extracted as Seer or Medium claimers, suspect parser false positives and rely more on quote text and timeline.

Output schema:
{{
  "players": {{
    "<exact player name>": {{
      "wolf_score": 0.0,
      "main_evidence": ["short evidence"],
      "anti_evidence": ["short anti-evidence"]
    }}
  }}
}}

Constraints:
- Include every listed player exactly once.
- Use only exact player names from the Players list.
- wolf_score must be a number from 0 to 1.
""".strip()

    return prompt


def build_formation_prompt(players, daily_states):
    players_text = _players_text(players)
    daily_text = _short_json(_compact_daily_states(daily_states), max_chars=16000)

    return f"""
You are a Werewolf formation analyst.

Return exactly one JSON object. No markdown.

Players:
{players_text}

Daily board states:
{daily_text}

Task:
1. Identify the latest formation, such as 1-1, 2-1, 2-2, 3-1.
2. Identify Seer claimers and Medium claimers.
3. Estimate whether fake Seer/Medium claimers are more likely Werewolf or Madman.
4. Identify gray players and estimate how many Werewolves are likely hidden among gray players.
5. Do not assign final roles to every player.

Important:
- There is exactly 1 true Seer and 1 true Medium.
- Madman may fake roles but is not Werewolf.
- Werewolves may fake roles and coordinate privately.
- Human result means non-Werewolf, not necessarily Villager.

Output schema:
{{
  "latest_formation": "2-2",
  "seer_claimers": [],
  "medium_claimers": [],
  "gray_players": [],
  "likely_patterns": [
    {{
      "pattern": "seer_true_madman_medium_true_wolf",
      "probability": 0.0,
      "reason": "short reason"
    }}
  ],
  "claimant_alignment_estimates": {{
    "<player>": {{
      "true_role": 0.0,
      "wolf_fake": 0.0,
      "madman_fake": 0.0
    }}
  }},
  "gray_wolf_slots_estimate": 1
}}
""".strip()

def _compact_messages_for_extraction(messages, max_messages=120, max_text=700):
    out = []
    for m in messages[:max_messages]:
        out.append({
            "day": m.get("day"),
            "order": m.get("no"),
            "time": m.get("time"),
            "speaker": m.get("speaker"),
            "text": str(m.get("text", ""))[:max_text],
        })
    return _short_json(out, max_chars=50000)


def build_event_extraction_prompt(game_index, day, players, messages, previous_state=None):
    players_text = _players_text(players)
    messages_text = _compact_messages_for_extraction(messages)
    prev_text = _short_json(previous_state or {}, max_chars=12000)

    return f"""
You are Agent 1, a Werewolf log event extractor.

Return exactly one JSON object. No markdown. No explanation.

Game: {game_index}
Day: {day}

Players:
{players_text}

Previous cumulative state, if any:
{prev_text}

Raw messages for this chunk:
{messages_text}

Task:
Extract only strategically useful events. Do not infer final roles. Do not chat.

Ignore noise:
- roleplay, food, greetings, jokes, weather, thanks, emoji-only messages
- ordinary flavor text with no role, vote, death, attack, suspicion, or result information

Keep these event types:
- claim: Seer CO, Medium CO, Hunter CO, or other explicit first-person role claim
- non_claim: not Seer, not Medium, not Seer/Medium
- divination: hard Seer result only; must explicitly be a result/check/divination/verdict
- medium_result: hard Medium result only; must explicitly be a Medium/霊 result
- vote: vote, execution preference, seer target preference, symbols such as ● ○ ▼ ▽
- execution: player executed by vote
- attack: player killed by Werewolf/night attack
- sudden_death: player died because of no post/no action
- suspicion: speaker suspects target as Werewolf or black-ish
- townread: speaker reads target as villager/white-ish
- formation: 1-1, 2-1, 3-1, 3-2, etc.
- contradiction: explicit retraction, contradiction, or perspective slip
- other_relevant: only if it affects role inference

Strict distinctions:
- white/black in ordinary discussion are soft reads, not hard results.
- A human result means non-Werewolf, not Villager.
- Madman is not a Werewolf; do not add it to wolf_score.
- Only record a role claim when the speaker self-claims. If the speaker says another player claimed, target should be that other player and event should explain it.

Output schema:
{{
  "game_id": "{game_index}",
  "day": "{day}",
  "events": [
    {{
      "type": "claim | non_claim | divination | medium_result | vote | execution | attack | sudden_death | suspicion | townread | formation | contradiction | other_relevant",
      "speaker": "exact player name or system",
      "target": "exact player name or null",
      "content": "short normalized event",
      "evidence": "message number and short quote",
      "confidence": 0.0
    }}
  ],
  "noise_summary": "short description of ignored noise"
}}

Constraints:
- Use exact player names from the list whenever possible.
- Include at most 60 events.
- confidence must be numeric from 0 to 1.
""".strip()


def build_state_tracker_prompt(players, previous_state, extracted_events, day):
    players_text = _players_text(players)
    prev_text = _short_json(previous_state or {}, max_chars=18000)
    events_text = _short_json(extracted_events or {}, max_chars=26000)

    return f"""
You are Agent 2, a Werewolf board-state tracker.

Return exactly one JSON object. No markdown. No explanation.

Players:
{players_text}

Previous cumulative state:
{prev_text}

New extracted events for day/chunk {day}:
{events_text}

Task:
Update the cumulative public board state. Do not assign final hidden roles.

Track:
- alive and dead players
- Seer claimers, Medium claimers, Hunter claimers
- not Seer / not Medium claims
- hard Seer results and hard Medium results
- execution, attack, sudden death history
- vote pressure and target preferences
- major suspicions and townreads
- current formation, such as 1-1, 2-1, 3-1, 3-2
- gray players, excluding current Seer/Medium claimers and dead players
- unresolved contradictions or retractions

Output schema:
{{
  "day": "{day}",
  "alive_players": [],
  "dead_players": [],
  "seer_claimers": [],
  "medium_claimers": [],
  "hunter_claimers": [],
  "not_seer_medium_claimers": [],
  "formation": "0-0",
  "hard_results": [
    {{"source": "player", "target": "player", "result": "human|werewolf", "kind": "seer|medium", "day": "..."}}
  ],
  "deaths": [
    {{"day": "...", "player": "...", "cause": "attack|execution|sudden_death|unknown"}}
  ],
  "votes": [
    {{"day": "...", "voter": "...", "target": "...", "type": "execution|seer_target|other"}}
  ],
  "major_conflicts": [
    {{"a": "...", "b": "...", "summary": "..."}}
  ],
  "gray_players": [],
  "notes": []
}}

Constraints:
- Use exact player names only.
- Preserve prior state unless new events update it.
- If evidence is ambiguous, add a note instead of treating it as hard fact.
""".strip()
