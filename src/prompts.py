import json


def _compact_evidence_for_prompt(evidence_cards, max_chars=18000):
    """
    Keep global card and each player's summary.
    Avoid dumping full by_day unless needed.
    """
    compact = {}

    if isinstance(evidence_cards, dict) and "_global" in evidence_cards:
        compact["_global"] = evidence_cards["_global"]

    players_compact = {}
    if isinstance(evidence_cards, dict):
        for k, v in evidence_cards.items():
            if k == "_global":
                continue
            if isinstance(v, dict):
                players_compact[k] = {
                    "summary": v.get("summary", {}),
                    # Keep by_day small. This helps timing but avoids exploding context.
                    "by_day_keys": list(v.get("by_day", {}).keys()),
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


def build_role_prompt(players, evidence_cards, parsed):
    players_text = _players_text(players)

    evidence_text = _compact_evidence_for_prompt(evidence_cards, max_chars=18000)

    parsed_compact = {
        "global": evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {},
        "deaths": parsed.get("deaths", []),
        "all_divinations": parsed.get("all_divinations", []),
        "all_votes": parsed.get("all_votes", [])[:200],
    }
    parsed_text = _short_json(parsed_compact, max_chars=8000)

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

Game evidence:
{evidence_text}

Parsed game facts:
{parsed_text}

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


def build_wolf_prompt(players, evidence_cards, parsed):
    players_text = _players_text(players)

    evidence_text = _compact_evidence_for_prompt(evidence_cards, max_chars=18000)

    parsed_compact = {
        "global": evidence_cards.get("_global", {}) if isinstance(evidence_cards, dict) else {},
        "deaths": parsed.get("deaths", []),
        "all_divinations": parsed.get("all_divinations", []),
        "all_votes": parsed.get("all_votes", [])[:200],
    }
    parsed_text = _short_json(parsed_compact, max_chars=8000)

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

Game evidence:
{evidence_text}

Parsed game facts:
{parsed_text}

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