import re
from collections import defaultdict, Counter
from typing import Dict, List, Any
from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import get_role_counts

ROLE_KEYWORDS = {
    "Seer": ["Seer", "seer", "占", "占卜", "prophecy", "prophet"],
    "Medium": ["Medium", "medium", "霊", "靈", "驗屍"],
    "Hunter": ["Hunter", "hunter", "guard", "protect"],
}


def get_seer_claimers_from_claims(claims: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    out = []
    for player, cs in claims.items():
        for c in cs:
            name = claim_name(c)
            if name == "Seer CO":
                out.append(player)
    return list(dict.fromkeys(out))


def parse_messages(day_text: str) -> List[Dict[str, Any]]:
    """
    Parse day text into message-level records.

    Expected translated log pattern:
    105.
    Young Girl Liza
    23:45
    [Seer claim]
    ...
    """
    messages = []

    pattern = re.compile(
        r"(?m)^\s*(\d+)\.\s*\n"
        r"([^\n]+?)\s*\n"
        r"(\d{2}:\d{2})\s*\n"
        r"([\s\S]*?)(?=^\s*\d+\.\s*\n|\Z)",
        re.MULTILINE,
    )

    for m in pattern.finditer(day_text):
        no = int(m.group(1))
        speaker = clean_name(m.group(2))
        time = m.group(3)
        text = m.group(4).strip()

        messages.append({
            "no": no,
            "speaker": speaker,
            "time": time,
            "text": text,
        })

    return messages


def extract_claims_from_messages(messages: List[Dict[str, Any]], players: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    claims = defaultdict(list)
    player_set = set(players)

    # Only first-person / explicit bracket-style self claims.
    # Do NOT match generic "Seer claim" because that often means discussing someone else's claim.
    seer_pat = re.compile(
        r"(\[Seer claim\]|\[Seer\s*CO\]|\[Seer\]\s*:|\[(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\]|It seems I[’']?m\s+the\s+\[?Seer\]?|I can see the future|my real profession is\s+(?:a\s+)?\[?Seer\]?|"
        r"\bI[’']?m a Seer\b|\bI am a Seer\b|\bI am the Seer\b|\bI[’']?m the Seer\b|"
        r"\bI claim Seer\b|\bI'?ll claim Seer\b|"
        r"占CO|占いCO|【占】|占いです|占い師CO)",
        re.I,
    )

    medium_pat = re.compile(
        r"(\[Medium claim\]|\[Medium\s*CO\]|\[Medium\]\s*:|\[Medium\s+Claim\]|I[’']?ve awakened to my spiritual abilities|"
        r"\bI[’']?m a Medium\b|\bI am a Medium\b|\bI am the Medium\b|\bI[’']?m the Medium\b|"
        r"\bI claim Medium\b|\bI'?ll claim Medium\b|"
        r"霊CO|霊能CO|霊媒CO|【霊】|霊能者CO)",
        re.I,
    )

    not_both_pat = re.compile(
        r"(not Seer\s*/\s*not Medium|neither a Seer nor a Medium|"
        r"not a Seer or Medium|not Seer and not Medium|"
        r"非占非霊|非占霊|非占.*非霊|非霊.*非占)",
        re.I,
    )

    not_seer_pat = re.compile(
        r"(\bnot Seer\b|\bnot a Seer\b|\bI am not the Seer\b|非占)",
        re.I,
    )

    not_medium_pat = re.compile(
        r"(\bnot Medium\b|\bnot a Medium\b|\bI am not the Medium\b|非霊)",
        re.I,
    )

    # Phrases that indicate discussion of claims, not self-claim.
    discussion_only_pat = re.compile(
        r"(made a Seer claim|made a Medium claim|"
        r"someone.*Seer claim|someone.*Medium claim|"
        r"seer claim order|medium claim order|"
        r"other.*Seer claim|other.*Medium claim|"
        r"if .* Seer claim|if .* Medium claim)",
        re.I,
    )

    for msg in messages:
        speaker = msg["speaker"]
        text = msg["text"]

        if speaker not in player_set:
            continue

        # Self-denial should be extracted first.
        has_not_both = bool(not_both_pat.search(text))
        has_not_seer = bool(not_seer_pat.search(text))
        has_not_medium = bool(not_medium_pat.search(text))

        if has_not_both:
            claims[speaker].append({
                "claim": "Not Seer/Medium",
                "order": msg["no"],
                "time": msg["time"],
                "text": text[:300],
            })
        else:
            if has_not_seer:
                claims[speaker].append({
                    "claim": "Not Seer",
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:300],
                })
            if has_not_medium:
                claims[speaker].append({
                    "claim": "Not Medium",
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:300],
                })

        # If the text is clearly discussing someone else's claim and does not use
        # a strong bracket/first-person self-claim, skip role CO.
        discussion_only = bool(discussion_only_pat.search(text))

        seer_match = bool(seer_pat.search(text))
        medium_match = bool(medium_pat.search(text))

        if seer_match and not (discussion_only and not re.search(r"(\[Seer claim\]|\[Seer\s*CO\]|\[Seer\]\s*:|\[(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\]|It seems I[’']?m\s+the\s+\[?Seer\]?|I can see the future|my real profession is\s+(?:a\s+)?\[?Seer\]?|占CO|占いCO|【占】)", text, re.I)):
            # Avoid contradictory extraction from the same sentence unless there is an explicit bracket claim.
            if not has_not_both or re.search(r"(\[Seer claim\]|\[Seer\s*CO\]|\[Seer\]\s*:|\[(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\]|It seems I[’']?m\s+the\s+\[?Seer\]?|I can see the future|my real profession is\s+(?:a\s+)?\[?Seer\]?|占CO|占いCO|【占】)", text, re.I):
                claims[speaker].append({
                    "claim": "Seer CO",
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:300],
                })

        if medium_match and not (discussion_only and not re.search(r"(\[Medium claim\]|\[Medium\s*CO\]|\[Medium\]\s*:|\[Medium\s+Claim\]|I[’']?ve awakened to my spiritual abilities|霊CO|霊能CO|霊媒CO|【霊】)", text, re.I)):
            if not has_not_both or re.search(r"(\[Medium claim\]|\[Medium\s*CO\]|\[Medium\]\s*:|\[Medium\s+Claim\]|I[’']?ve awakened to my spiritual abilities|霊CO|霊能CO|霊媒CO|【霊】)", text, re.I):
                claims[speaker].append({
                    "claim": "Medium CO",
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:300],
                })

    return dict(claims)


def split_days(text: str) -> Dict[str, str]:
    """
    Split log by day markers such as:
    ===== Day 1 =====
    ===== Day 2 =====
    """
    parts = re.split(r"=+\s*Day\s+(\d+)\s*=+", text)
    days = {}

    if len(parts) <= 1:
        days["0"] = text
        return days

    prefix = parts[0]
    for i in range(1, len(parts), 2):
        day = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        days[day] = content

    if prefix.strip():
        days["prologue"] = prefix

    return days


def extract_deaths(day_text: str, players: List[str]) -> List[str]:
    deaths = []

    for p in players:
        escaped = re.escape(p)

        patterns = [
            rf"\b{escaped}\b was found in a gruesome state",
            rf"The next morning,\s*\b{escaped}\b was found",
            rf"\b{escaped}\b was executed",
            rf"\b{escaped}\b died from sudden death",
            rf"\b{escaped}\b suddenly died",
            rf"\b{escaped}\b died",
        ]

        for pat in patterns:
            if re.search(pat, day_text, re.I):
                deaths.append(p)
                break

    return list(dict.fromkeys(deaths))


def clean_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\[\]【】「」『』:：].*", "", name)
    name = name.strip()
    return name


def canonical_player_name(name: str, players: List[str]) -> str:
    """Map log speaker names like 'Outlaw Dieter' to role CSV names like 'Dieter'."""
    raw = str(name or "").strip()
    if raw in players:
        return raw
    low_map = {str(p).strip().lower(): p for p in players}
    if raw.lower() in low_map:
        return low_map[raw.lower()]
    tokens = [x for x in re.split(r"\s+", raw) if x]
    candidates = []
    if tokens:
        candidates.append(tokens[-1].lower())
        candidates.append(raw.lower())
    for p in players:
        plow = str(p).lower()
        ptoks = [x for x in re.split(r"\s+", str(p)) if x]
        plast = ptoks[-1].lower() if ptoks else plow
        if plast in candidates or plow.endswith(" " + tokens[-1].lower() if tokens else "\0"):
            return p
        # title-only or short-name CSV variants
        if tokens and tokens[-1].lower() == plow:
            return p
    return raw


def extract_claims(day_text: str, players: List[str]) -> Dict[str, List[str]]:
    claims = defaultdict(list)

    for p in players:
        # Find local window around player name.
        for m in re.finditer(re.escape(p), day_text):
            window = day_text[m.start(): m.start() + 400]

            if re.search(r"\bI'?m the Medium\b|\[Medium\]|Medium Claim|I am the Medium", window, re.I):
                claims[p].append("Medium CO")

            if re.search(r"\bI'?m a Seer\b|\bI am the Seer\b|Seer claim|self-proclaimed seer|\[Seer\]", window, re.I):
                claims[p].append("Seer CO")

            if re.search(r"not Seer\s*/\s*not Medium|neither a Seer nor a Medium|not a Seer or Medium", window, re.I):
                claims[p].append("Not Seer/Medium")

    return dict(claims)


def extract_divinations_from_messages(
    messages: List[Dict[str, Any]],
    players: List[str],
    seer_claimers: List[str],
) -> List[Dict[str, Any]]:
    results = []
    player_set = set(players)
    seer_set = set(seer_claimers)

    result_patterns = [
        # English translated logs
        (r"(?:result|verdict|divination|check|checked|announce|announces).*?{target}.*?\b(human|werewolf|white|black)\b", 1),
        (r"{target}.*?\b(is|was)\s+(human|a human|werewolf|a werewolf)\b", 2),

        # Japanese / mixed forms, in case untranslated fragments remain
        (r"{target}.*?(白|黒|黑|人間|狼|人狼)", 1),
    ]

    for msg in messages:
        speaker = msg["speaker"]
        text = msg["text"]

        if speaker not in seer_set:
            continue

        # Avoid extracting ordinary discussion as hard result.
        if not re.search(
            r"(result|verdict|divination|divined|check|checked|announce|announces|判定|結果|占い結果|占結果)",
            text,
            re.I,
        ):
            continue

        for target in players:
            if target == speaker:
                continue

            escaped = re.escape(target)
            for raw_pat, group_idx in result_patterns:
                pat = raw_pat.format(target=escaped)
                m = re.search(pat, text, re.I)
                if not m:
                    continue

                word = m.group(group_idx).lower()
                if word in ("human", "a human", "white", "白", "人間"):
                    result = "human"
                elif word in ("werewolf", "a werewolf", "black", "黒", "黑", "狼", "人狼"):
                    result = "werewolf"
                else:
                    continue

                results.append({
                    "seer": speaker,
                    "target": target,
                    "result": result,
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:250],
                    "evidence_type": "hard_divination",
                })

    seen = set()
    unique = []
    for r in results:
        key = (r["seer"], r["target"], r["result"], r.get("order"))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def extract_soft_reads(day_text: str, players: List[str]) -> List[Dict[str, str]]:
    """
    Message-level soft read extraction.  This replaces the old day-level regex,
    which connected survivor lists / GS tables / quoted blocks to the wrong
    speakers and targets.
    """
    reads: List[Dict[str, str]] = []
    messages = parse_messages(day_text)
    if not messages:
        return []

    read_words = {
        "white-ish": "likely_villager",
        "black-ish": "likely_werewolf",
        "white": "likely_villager",
        "black": "likely_werewolf",
        "gray": "unresolved",
        "grey": "unresolved",
    }

    def aliases(player: str) -> List[str]:
        parts = [x for x in re.split(r"\s+", player) if x]
        out = [player]
        if parts and len(parts[-1]) >= 3:
            out.append(parts[-1])
        low = player.lower()
        extra = {
            "shepherd": ["Sheep"], "baker": ["Baker", "Inn", "Innkeeper"],
            "father": ["Father", "God"], "librarian": ["Librarian", "Book"],
            "wounded soldier": ["Soldier"], "young girl": ["Girl"],
            "young man": ["Young Man", "Blue"], "old man": ["Old Man", "Old"],
            "farmer": ["Farmer"], "merchant": ["Merchant"], "mayor": ["Mayor"],
            "traveler": ["Traveler"], "outlaw": ["Outlaw"], "tailor": ["Tailor"],
        }
        for key, vals in extra.items():
            if key in low:
                out.extend(vals)
        return list(dict.fromkeys(out))

    skip_context = re.compile(
        r"current survivors|survivors are|\bGS\b|gray scale|white\s+.+>.+black|"
        r"confirmed seer result|was found in a gruesome state|formation|current situation|"
        r"\[.*Seer.*\]|\[.*Medium.*\]",
        re.I | re.S,
    )

    for msg in messages:
        speaker = canonical_player_name(msg.get("speaker"), players)
        text = str(msg.get("text", ""))
        if speaker not in players:
            continue
        # Skip summaries, tables, CO/result posts, and very broad posts mentioning many players.
        if skip_context.search(text):
            continue
        mentioned_count = 0
        for p in players:
            if p == speaker:
                continue
            if any(re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", text, re.I) for a in aliases(p)):
                mentioned_count += 1
        if mentioned_count > 5:
            continue

        for target in players:
            if target == speaker:
                continue
            target_pats = [rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])" for a in aliases(target) if len(a) >= 3]
            if not target_pats:
                continue
            target_union = "(?:" + "|".join(target_pats) + ")"
            patterns = [
                rf"{target_union}[^\n]{{0,80}}\b(white-ish|black-ish|white|black|gray|grey)\b",
                rf"\b(white-ish|black-ish|white|black|gray|grey)\b[^\n]{{0,80}}{target_union}",
            ]
            for pat in patterns:
                m = re.search(pat, text, re.I)
                if not m:
                    continue
                word = next((g.lower() for g in m.groups() if g and g.lower() in read_words), None)
                if not word:
                    continue
                reads.append({
                    "speaker": speaker,
                    "target": target,
                    "read": read_words[word],
                    "raw_word": word,
                    "evidence_type": "soft_read",
                    "text": text[max(0, m.start() - 80): min(len(text), m.end() + 80)],
                })
                break

    seen = set(); unique = []
    for r in reads:
        key = (r["speaker"], r["target"], r["read"], r["raw_word"], r["text"][:80])
        if key not in seen:
            seen.add(key); unique.append(r)
    return unique

def extract_votes(day_text: str, players: List[str]) -> List[Dict[str, str]]:
    """
    Extract rough vote / preference markers:
    ● target, ○ target, ▼ target
    """
    votes = []

    for p in players:
        for target in players:
            if p == target:
                continue

            # Preference patterns in the nearby text after speaker.
            pattern = rf"{re.escape(p)}[\s\S]{{0,500}}[●○▼]\s*{re.escape(target)}"
            if re.search(pattern, day_text):
                votes.append({"voter": p, "target": target, "type": "preference"})

    return votes


def extract_votes_from_messages(messages: List[Dict[str, Any]], players: List[str]) -> List[Dict[str, Any]]:
    votes = []
    player_set = set(players)

    for msg in messages:
        speaker = msg["speaker"]
        text = msg["text"]

        if speaker not in player_set:
            continue

        for target in players:
            if target == speaker:
                continue

            # ● = divination first preference, ○ = second preference, ▼ = execution, ▽ = backup execution
            patterns = [
                (rf"●\s*{re.escape(target)}", "seer_first_preference"),
                (rf"○\s*{re.escape(target)}", "seer_second_preference"),
                (rf"▼\s*{re.escape(target)}", "execution_first_preference"),
                (rf"▽\s*{re.escape(target)}", "execution_second_preference"),
            ]

            for pat, vote_type in patterns:
                if re.search(pat, text):
                    votes.append({
                        "voter": speaker,
                        "target": target,
                        "type": vote_type,
                        "order": msg["no"],
                        "time": msg["time"],
                        "text": text[:250],
                    })

    return votes

# 追加 把 evidence card 改成分輪 dict 的 helper： 
def parse_game_log(text: str, players: List[str]) -> Dict[str, Any]:
    days = split_days(text)

    parsed = {
        "days": {},
        "all_claims": defaultdict(list),
        "all_divinations": [],
        "all_soft_reads": [],
        "all_votes": [],
        "deaths": [],
        "messages": [],
    }

    for day, content in days.items():
        messages = parse_messages(content)
        for _m in messages:
            _m["speaker"] = canonical_player_name(_m.get("speaker", ""), players)

        # fallback: if parser fails, keep old behavior
        if messages:
            claims = extract_claims_from_messages(messages, players)
            votes = extract_votes_from_messages(messages, players)

            seer_claimers = get_seer_claimers_from_claims(claims)
            divinations = extract_divinations_from_messages(
                messages=messages,
                players=players,
                seer_claimers=seer_claimers,
            )
            soft_reads = extract_soft_reads(content, players)
        else:
            claims = extract_claims(content, players)
            votes = extract_votes(content, players)

            # No reliable message structure, so avoid extracting hard divination.
            # This prevents false hard results from long raw-text regex.
            divinations = []
            soft_reads = extract_soft_reads(content, players)

        if str(day).isdigit() and int(day) >= 2:
            deaths = extract_deaths(content, players)
        else:
            deaths = []

        parsed["days"][day] = {
            "messages": messages,
            "claims": claims,
            "divinations": divinations,
            "soft_reads": soft_reads,
            "votes": votes,
            "deaths": deaths,
            "summary_hint": content[:1500],
        }

        for msg in messages:
            msg2 = dict(msg)
            msg2["day"] = day
            parsed["messages"].append(msg2)

        for p, cs in claims.items():
            parsed["all_claims"][p].extend(cs)

        parsed["all_divinations"].extend(divinations)
        parsed["all_soft_reads"].extend(soft_reads)
        parsed["all_votes"].extend(votes)
        parsed["deaths"].extend(deaths)

    parsed["all_claims"] = dict(parsed["all_claims"])
    parsed["deaths"] = list(dict.fromkeys(parsed["deaths"]))
    return parsed




def claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def build_global_card(parsed: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    claim_timeline = []

    for player, claims in parsed.get("all_claims", {}).items():
        for c in claims:
            if isinstance(c, dict):
                claim_timeline.append({
                    "player": player,
                    "claim": c.get("claim", ""),
                    "order": c.get("order"),
                    "time": c.get("time"),
                })
            else:
                claim_timeline.append({
                    "player": player,
                    "claim": str(c),
                    "order": None,
                    "time": None,
                })

    claim_timeline.sort(key=lambda x: (x["order"] is None, x["order"] or 10**9))

    seer_claimers = [
        x["player"] for x in claim_timeline
        if x["claim"] == "Seer CO"
    ]
    medium_claimers = [
        x["player"] for x in claim_timeline
        if x["claim"] == "Medium CO"
    ]

    return {
        "num_players": len(players),
        "role_counts": {
            "Werewolf": 3 if len(players) >= 13 else 2,
            "Seer": 1,
            "Medium": 1,
            "Madman": 1 if len(players) >= 11 else 0,
            "Hunter": 1 if len(players) >= 11 else 0,
        },
        "seer_claimers": list(dict.fromkeys(seer_claimers)),
        "medium_claimers": list(dict.fromkeys(medium_claimers)),
        "formation": f"{len(set(seer_claimers))}-{len(set(medium_claimers))}",
        "claim_timeline": claim_timeline,
        "deaths": parsed.get("deaths", []),
    }


def build_evidence_cards(parsed: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    """
    Return structured evidence instead of plain strings.

    Output:
    {
      "_global": {...},
      "Player A": {
        "by_day": {...},
        "summary": {...}
      }
    }
    """
    evidence = {
        "_global": build_global_card(parsed, players)
    }

    for p in players:
        evidence[p] = {
            "by_day": {},
            "summary": {
                "claims_timeline": [],
                "divinations_given": [],
                "divinations_received": [],
                "votes_given": [],
                "votes_received": [],
                "soft_reads_given": [],
                "soft_reads_received": [],
                "death_or_status": [],
            },
        }

    for day, day_obj in parsed.get("days", {}).items():
        for p in players:
            evidence[p]["by_day"][day] = {
                "claims": [],
                "divinations_given": [],
                "divinations_received": [],
                "votes_given": [],
                "votes_received": [],
                "soft_reads_given": [],
                "soft_reads_received": [],
                "deaths": [],
            }

        # claims
        for player, claims in day_obj.get("claims", {}).items():
            if player not in evidence:
                continue
            for c in claims:
                item = c if isinstance(c, dict) else {"claim": str(c)}
                item = dict(item)
                item["day"] = day

                evidence[player]["by_day"][day]["claims"].append(item)
                evidence[player]["summary"]["claims_timeline"].append(item)

        # divinations
        for r in day_obj.get("divinations", []):
            seer = r.get("seer")
            target = r.get("target")
            item = dict(r)
            item["day"] = day

            if seer in evidence:
                evidence[seer]["by_day"][day]["divinations_given"].append(item)
                evidence[seer]["summary"]["divinations_given"].append(item)

            if target in evidence:
                evidence[target]["by_day"][day]["divinations_received"].append(item)
                evidence[target]["summary"]["divinations_received"].append(item)

        # votes
        for v in day_obj.get("votes", []):
            voter = v.get("voter")
            target = v.get("target")
            item = dict(v)
            item["day"] = day

            if voter in evidence:
                evidence[voter]["by_day"][day]["votes_given"].append(item)
                evidence[voter]["summary"]["votes_given"].append(item)

            if target in evidence:
                evidence[target]["by_day"][day]["votes_received"].append(item)
                evidence[target]["summary"]["votes_received"].append(item)

        # soft reads
        for r in day_obj.get("soft_reads", []):
            speaker = r.get("speaker")
            target = r.get("target")
            item = dict(r)
            item["day"] = day

            if speaker in evidence:
                evidence[speaker]["by_day"][day]["soft_reads_given"].append(item)
                evidence[speaker]["summary"]["soft_reads_given"].append(item)

            if target in evidence:
                evidence[target]["by_day"][day]["soft_reads_received"].append(item)
                evidence[target]["summary"]["soft_reads_received"].append(item)

        # deaths
        for d in day_obj.get("deaths", []):
            if d in evidence:
                item = {"day": day, "event": "death"}
                evidence[d]["by_day"][day]["deaths"].append(item)
                evidence[d]["summary"]["death_or_status"].append(item)

    return evidence


def _claim_value(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def _sort_days(days: Dict[str, Any]) -> List[str]:
    def key(d: str):
        s = str(d)
        if s == "prologue":
            return (-1, 0)
        if s.isdigit():
            return (0, int(s))
        return (1, s)

    return sorted(days.keys(), key=key)


def _unique_keep_order(xs: List[str]) -> List[str]:
    return list(dict.fromkeys(xs))


def _summarize_vote_targets(votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Split ●/○ seer preferences and ▼/▽ execution preferences.
    """
    seer_first = Counter()
    seer_second = Counter()
    exe_first = Counter()
    exe_second = Counter()

    for v in votes:
        target = v.get("target")
        vote_type = v.get("type", "")
        if not target:
            continue

        if vote_type == "seer_first_preference":
            seer_first[target] += 1
        elif vote_type == "seer_second_preference":
            seer_second[target] += 1
        elif vote_type == "execution_first_preference":
            exe_first[target] += 1
        elif vote_type == "execution_second_preference":
            exe_second[target] += 1
        elif vote_type == "preference":
            # fallback extractor does not distinguish ●/▼
            seer_first[target] += 1

    return {
        "seer_first": dict(seer_first.most_common()),
        "seer_second": dict(seer_second.most_common()),
        "execution_first": dict(exe_first.most_common()),
        "execution_second": dict(exe_second.most_common()),
        "top_seer_targets": [p for p, _ in seer_first.most_common(5)],
        "top_execution_targets": [p for p, _ in exe_first.most_common(5)],
    }


def _summarize_soft_reads(reads: List[Dict[str, Any]]) -> Dict[str, Any]:
    wolfish = Counter()
    villagery = Counter()

    for r in reads:
        target = r.get("target")
        read = r.get("read")
        if not target:
            continue

        if read == "likely_werewolf":
            wolfish[target] += 1
        elif read == "likely_villager":
            villagery[target] += 1

    return {
        "likely_werewolf_counts": dict(wolfish.most_common()),
        "likely_villager_counts": dict(villagery.most_common()),
        "top_suspected_players": [p for p, _ in wolfish.most_common(5)],
        "top_villagery_players": [p for p, _ in villagery.most_common(5)],
    }


def build_daily_states(parsed: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    """
    Build cumulative board states after each day.

    This is not a final inference result. It is a structured board-state summary
    for downstream agents:
    - formation
    - claimers
    - gray players
    - hard divination results
    - vote pressure
    - soft suspicion concentration
    """
    states: Dict[str, Any] = {}

    alive = set(players)
    dead = set()

    cumulative_claims: Dict[str, List[Any]] = defaultdict(list)
    cumulative_divinations: List[Dict[str, Any]] = []
    cumulative_votes: List[Dict[str, Any]] = []
    cumulative_soft_reads: List[Dict[str, Any]] = []
    cumulative_deaths: List[str] = []

    role_counts = get_role_counts(len(players))

    for day in _sort_days(parsed.get("days", {})):
        day_obj = parsed["days"][day]

        # 1. Accumulate daily events.
        for p, claims in day_obj.get("claims", {}).items():
            cumulative_claims[p].extend(claims)

        cumulative_divinations.extend(day_obj.get("divinations", []))
        cumulative_votes.extend(day_obj.get("votes", []))
        cumulative_soft_reads.extend(day_obj.get("soft_reads", []))

        for d in day_obj.get("deaths", []):
            if d in players:
                dead.add(d)
                alive.discard(d)
                cumulative_deaths.append(d)

        # 2. Current claimers.
        seer_claimers = []
        medium_claimers = []
        not_seer_medium = []

        for p, cs in cumulative_claims.items():
            claim_names = [_claim_value(c) for c in cs]

            if "Seer CO" in claim_names:
                seer_claimers.append(p)
            if "Medium CO" in claim_names:
                medium_claimers.append(p)
            if "Not Seer/Medium" in claim_names or (
                "Not Seer" in claim_names and "Not Medium" in claim_names
            ):
                not_seer_medium.append(p)

        seer_claimers = _unique_keep_order(seer_claimers)
        medium_claimers = _unique_keep_order(medium_claimers)
        not_seer_medium = _unique_keep_order(not_seer_medium)

        ability_claimers = set(seer_claimers) | set(medium_claimers)

        gray_players = [
            p for p in players
            if p in alive and p not in ability_claimers
        ]

        # 3. Hard divination facts.
        hard_white_results = [
            r for r in cumulative_divinations
            if isinstance(r, dict) and r.get("result") == "human"
        ]
        hard_black_results = [
            r for r in cumulative_divinations
            if isinstance(r, dict) and r.get("result") == "werewolf"
        ]

        hard_white_targets = _unique_keep_order([
            r.get("target") for r in hard_white_results if r.get("target") in players
        ])
        hard_black_targets = _unique_keep_order([
            r.get("target") for r in hard_black_results if r.get("target") in players
        ])

        # 4. Vote/read summaries.
        day_vote_summary = _summarize_vote_targets(day_obj.get("votes", []))
        cumulative_vote_summary = _summarize_vote_targets(cumulative_votes)

        day_soft_summary = _summarize_soft_reads(day_obj.get("soft_reads", []))
        cumulative_soft_summary = _summarize_soft_reads(cumulative_soft_reads)

        # 5. Useful derived facts for downstream agents.
        if len(seer_claimers) == 0 and len(medium_claimers) == 0:
            formation_type = "no_claims_or_pre_claim"
        elif len(seer_claimers) >= 2 or len(medium_claimers) >= 2:
            formation_type = "contested_abilities"
        else:
            formation_type = "mostly_uncontested_abilities"

        likely_gray_wolf_slots = max(
            0,
            role_counts["Werewolf"] - min(role_counts["Werewolf"], len(ability_claimers))
        )

        states[day] = {
            "day": day,
            "alive_players": sorted(alive),
            "dead_players": sorted(dead),
            "role_counts": role_counts,

            "seer_claimers": seer_claimers,
            "medium_claimers": medium_claimers,
            "not_seer_medium_claimers": not_seer_medium,
            "formation": f"{len(seer_claimers)}-{len(medium_claimers)}",
            "formation_type": formation_type,

            "ability_claimers": sorted(ability_claimers),
            "gray_players": gray_players,
            "likely_gray_wolf_slots_upper_bound": likely_gray_wolf_slots,

            "hard_white_results": hard_white_results,
            "hard_black_results": hard_black_results,
            "hard_white_targets": hard_white_targets,
            "hard_black_targets": hard_black_targets,

            "day_vote_summary": day_vote_summary,
            "cumulative_vote_summary": cumulative_vote_summary,

            "day_soft_read_summary": day_soft_summary,
            "cumulative_soft_read_summary": cumulative_soft_summary,

            "top_suspected_players": cumulative_soft_summary["top_suspected_players"],
            "top_execution_targets": cumulative_vote_summary["top_execution_targets"],
            "top_seer_targets": cumulative_vote_summary["top_seer_targets"],

            "deaths_so_far": _unique_keep_order(cumulative_deaths),

            "notes": [
                "Hard white means Seer human result, not confirmed Villager.",
                "Soft reads are ordinary discussion reads, not hard role results.",
                "Gray players exclude current Seer/Medium claimers but may include Hunter, Villager, Madman, or Werewolf.",
            ],
        }

    return states
# =========================
# MaKTO-inspired v2 helpers
# =========================

STRATEGIC_KEYWORDS = [
    "Seer", "Medium", "claim", "CO", "counter", "fake", "true", "real",
    "not Seer", "not Medium", "not Seer / not Medium",
    "result", "verdict", "divination", "divined", "checked", "check",
    "white", "black", "gray", "grey", "wolf", "werewolf", "human", "villager",
    "suspicious", "suspect", "doubt", "distrust", "trust", "believe", "defend",
    "vote", "lynch", "execute", "execution", "rope", "line", "formation",
    "●", "○", "▼", "▽",
    "占", "霊", "靈", "判定", "結果", "白", "黒", "黑", "狼", "人狼", "吊", "希望",
]


def _is_game_day(day: Any) -> bool:
    s = str(day)
    return s.isdigit() and int(s) >= 1


def _shorten_text(text: str, n: int = 260) -> str:
    text = str(text or "").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:n]


def _has_any_keyword(text: str, keywords: List[str] = None) -> bool:
    keywords = keywords or STRATEGIC_KEYWORDS
    low = str(text or "").lower()
    return any(str(k).lower() in low for k in keywords)


def build_objective_pack(parsed: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    """
    Build a compact, Day1+ only objective pack.

    This deliberately excludes prologue messages because private/public logs contain
    many RP fake-COs and Gerd jokes before the real game starts.
    """
    role_counts = get_role_counts(len(players))

    claim_timeline: List[Dict[str, Any]] = []
    hard_results: List[Dict[str, Any]] = []
    votes: List[Dict[str, Any]] = []
    deaths: List[Dict[str, Any]] = []
    soft_reads: List[Dict[str, Any]] = []

    for day in _sort_days(parsed.get("days", {})):
        if not _is_game_day(day):
            continue
        day_obj = parsed["days"].get(day, {})

        for player, claims in day_obj.get("claims", {}).items():
            if player not in players:
                continue
            for c in claims:
                if isinstance(c, dict):
                    claim = str(c.get("claim", ""))
                    order = c.get("order")
                    time = c.get("time")
                    text = _shorten_text(c.get("text", ""), 180)
                else:
                    claim = str(c)
                    order = None
                    time = None
                    text = ""
                claim_timeline.append({
                    "day": str(day),
                    "order": order,
                    "time": time,
                    "player": player,
                    "claim": claim,
                    "text": text,
                })

        for r in day_obj.get("divinations", []):
            if not isinstance(r, dict):
                continue
            item = dict(r)
            item["day"] = str(day)
            item["text"] = _shorten_text(item.get("text", ""), 180)
            hard_results.append(item)

        for v in day_obj.get("votes", []):
            if not isinstance(v, dict):
                continue
            item = dict(v)
            item["day"] = str(day)
            item["text"] = _shorten_text(item.get("text", ""), 160)
            votes.append(item)

        for d in day_obj.get("deaths", []):
            if d in players:
                deaths.append({"day": str(day), "player": d, "cause": "death_or_attack"})

        # Keep soft reads compact; these are weak features for interaction, not hard results.
        for r in day_obj.get("soft_reads", [])[:120]:
            if not isinstance(r, dict):
                continue
            item = {
                "day": str(day),
                "speaker": r.get("speaker"),
                "target": r.get("target"),
                "read": r.get("read"),
                "raw_word": r.get("raw_word"),
                "text": _shorten_text(r.get("text", ""), 120),
            }
            if item["speaker"] in players and item["target"] in players:
                soft_reads.append(item)

    claim_timeline.sort(key=lambda x: (int(x.get("day", "999")) if str(x.get("day", "")).isdigit() else 999, x.get("order") or 10**9))

    seer_claimers = _unique_keep_order([x["player"] for x in claim_timeline if x.get("claim") == "Seer CO"])
    medium_claimers = _unique_keep_order([x["player"] for x in claim_timeline if x.get("claim") == "Medium CO"])
    hunter_claimers = _unique_keep_order([x["player"] for x in claim_timeline if "Hunter" in str(x.get("claim", ""))])
    not_seer_medium = _unique_keep_order([
        x["player"] for x in claim_timeline
        if x.get("claim") == "Not Seer/Medium"
        or x.get("claim") == "Not Seer"
        or x.get("claim") == "Not Medium"
    ])

    ability_claimers = set(seer_claimers) | set(medium_claimers) | set(hunter_claimers)
    gray_players = [p for p in players if p not in ability_claimers and p not in {d.get("player") for d in deaths}]

    latest_day = None
    for day in _sort_days(parsed.get("days", {})):
        if _is_game_day(day):
            latest_day = str(day)

    return {
        "num_players": len(players),
        "role_counts": role_counts,
        "latest_day": latest_day,
        "formation": f"{len(seer_claimers)}-{len(medium_claimers)}",
        "seer_claimers": seer_claimers,
        "medium_claimers": medium_claimers,
        "hunter_claimers": hunter_claimers,
        "not_seer_medium_claimers": not_seer_medium,
        "gray_players": gray_players,
        "claim_timeline": claim_timeline[:80],
        "hard_results": hard_results[:120],
        "votes": votes[:180],
        "deaths": deaths,
        "soft_reads": soft_reads[:160],
        "notes": [
            "Objective pack is Day1+ only; prologue RP is excluded.",
            "Soft reads are not hard Seer/Medium results.",
            "Madman is not Werewolf for wolf_score.",
        ],
    }


def build_strategic_snippets(parsed: Dict[str, Any], players: List[str], max_per_day: int = 90, max_total: int = 420) -> List[Dict[str, Any]]:
    """
    Day1+ only snippets for interaction analysis.
    Keeps language actions, not the entire log.
    """
    snippets: List[Dict[str, Any]] = []

    for day in _sort_days(parsed.get("days", {})):
        if not _is_game_day(day):
            continue
        day_obj = parsed["days"].get(day, {})
        kept_for_day = 0

        for msg in day_obj.get("messages", []):
            if kept_for_day >= max_per_day:
                break
            speaker = msg.get("speaker")
            text = str(msg.get("text", ""))
            if speaker not in players:
                continue
            if not _has_any_keyword(text):
                continue

            snippets.append({
                "day": str(day),
                "order": msg.get("no"),
                "time": msg.get("time"),
                "speaker": speaker,
                "text": _shorten_text(text, 360),
            })
            kept_for_day += 1

    # Prefer early formation/claim messages and later high-signal messages.
    snippets.sort(key=lambda x: (int(x.get("day", "999")) if str(x.get("day", "")).isdigit() else 999, x.get("order") or 10**9))
    if len(snippets) <= max_total:
        return snippets

    head = snippets[: max_total // 2]
    tail = snippets[-(max_total - len(head)):]
    return head + tail
