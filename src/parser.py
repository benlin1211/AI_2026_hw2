import re
from collections import defaultdict
from typing import Dict, List, Any


ROLE_KEYWORDS = {
    "Seer": ["Seer", "seer", "占", "占卜", "prophecy", "prophet"],
    "Medium": ["Medium", "medium", "霊", "靈", "驗屍"],
    "Hunter": ["Hunter", "hunter", "guard", "protect"],
}

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

    seer_pat = re.compile(
        r"(Seer claim|\[Seer\]|\bI'?m a Seer\b|\bI am the Seer\b|"
        r"占CO|占いCO|占い師|【占】|占いです)",
        re.I,
    )

    medium_pat = re.compile(
        r"(Medium Claim|\[Medium\]|\bI'?m the Medium\b|\bI am the Medium\b|"
        r"霊CO|霊能CO|霊媒CO|【霊】|霊能者)",
        re.I,
    )

    not_both_pat = re.compile(
        r"(not Seer\s*/\s*not Medium|neither a Seer nor a Medium|"
        r"not a Seer or Medium|非占非霊|非占霊|非占.*非霊|非霊.*非占)",
        re.I,
    )

    not_seer_pat = re.compile(r"(not Seer|非占)", re.I)
    not_medium_pat = re.compile(r"(not Medium|非霊)", re.I)

    for msg in messages:
        speaker = msg["speaker"]
        text = msg["text"]

        if speaker not in player_set:
            continue

        if seer_pat.search(text):
            claims[speaker].append({
                "claim": "Seer CO",
                "order": msg["no"],
                "time": msg["time"],
                "text": text[:300],
            })

        if medium_pat.search(text):
            claims[speaker].append({
                "claim": "Medium CO",
                "order": msg["no"],
                "time": msg["time"],
                "text": text[:300],
            })

        if not_both_pat.search(text):
            claims[speaker].append({
                "claim": "Not Seer/Medium",
                "order": msg["no"],
                "time": msg["time"],
                "text": text[:300],
            })
        else:
            if not_seer_pat.search(text):
                claims[speaker].append({
                    "claim": "Not Seer",
                    "order": msg["no"],
                    "time": msg["time"],
                    "text": text[:300],
                })
            if not_medium_pat.search(text):
                claims[speaker].append({
                    "claim": "Not Medium",
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


def extract_deaths(day_text: str) -> List[str]:
    deaths = []

    patterns = [
        r"(.+?) was found in a gruesome state",
        r"The next morning, (.+?) was found",
        r"(.+?) was executed",
        r"(.+?) died from sudden death",
        r"(.+?) suddenly died",
        r"(.+?) died",
    ]

    for pat in patterns:
        for m in re.finditer(pat, day_text):
            name = clean_name(m.group(1))
            if name:
                deaths.append(name)

    return list(dict.fromkeys(deaths))


def clean_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\[\]【】「」『』:：].*", "", name)
    name = name.strip()
    return name


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


def extract_divinations(day_text: str, players: List[str]) -> List[Dict[str, str]]:
    """
    Extract hard Seer-like results only.
    Conservative by design.

    white/black are treated as hard result only when near:
    - result
    - verdict
    - divination
    - check
    - says
    - is human / is werewolf
    """
    results = []

    result_words = {
        "human": "human",
        "werewolf": "werewolf",
        "white": "human",
        "black": "werewolf",
    }

    result_context = r"(result|verdict|divination|divined|check|checked|says|announces|判定|占|占卜|結果)"

    for seer in players:
        for target in players:
            if seer == target:
                continue

            patterns = [
                # Seer ... result/check/divination ... Target ... human/werewolf/white/black
                rf"{re.escape(seer)}[\s\S]{{0,250}}{result_context}[\s\S]{{0,150}}{re.escape(target)}[\s\S]{{0,80}}\b(human|werewolf|white|black)\b",

                # Seer ... Target ... is human/werewolf
                # Keep white/black here only if "result/check/verdict" also appears nearby.
                rf"{re.escape(seer)}[\s\S]{{0,250}}{re.escape(target)}[\s\S]{{0,50}}\b(is|was)\s+(human|a human|werewolf|a werewolf)\b",

                # Seer ... Target ... white/black result
                rf"{re.escape(seer)}[\s\S]{{0,250}}{re.escape(target)}[\s\S]{{0,50}}\b(white|black)\s+(result|verdict)\b",

                # Target ... human/werewolf by Seer
                rf"{re.escape(target)}[\s\S]{{0,50}}\b(human|werewolf)\b[\s\S]{{0,120}}\bby\b[\s\S]{{0,50}}{re.escape(seer)}",
            ]

            for pat in patterns:
                for m in re.finditer(pat, day_text, re.I):
                    groups = [g for g in m.groups() if g]
                    word = None
                    for g in reversed(groups):
                        g = g.lower().strip()
                        if g in result_words:
                            word = g
                            break
                        if g in ["a human"]:
                            word = "human"
                            break
                        if g in ["a werewolf"]:
                            word = "werewolf"
                            break

                    if word:
                        results.append({
                            "seer": seer,
                            "target": target,
                            "result": result_words[word],
                            "evidence_type": "hard_divination",
                        })

    seen = set()
    unique = []
    for r in results:
        key = (r["seer"], r["target"], r["result"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def extract_soft_reads(day_text: str, players: List[str]) -> List[Dict[str, str]]:
    """
    Extract soft social reads:
    white = likely villager
    black = likely werewolf
    gray = unresolved

    These are not divination results.
    They should affect wolf_score weakly, not role truth directly.
    """
    reads = []

    read_words = {
        "white": "likely_villager",
        "white-ish": "likely_villager",
        "black": "likely_werewolf",
        "black-ish": "likely_werewolf",
        "gray": "unresolved",
        "grey": "unresolved",
    }

    for speaker in players:
        for target in players:
            if speaker == target:
                continue

            patterns = [
                rf"{re.escape(speaker)}[\s\S]{{0,300}}{re.escape(target)}[\s\S]{{0,80}}\b(white-ish|black-ish|white|black|gray|grey)\b",
                rf"{re.escape(speaker)}[\s\S]{{0,300}}\b(white-ish|black-ish|white|black|gray|grey)\b[\s\S]{{0,80}}{re.escape(target)}",
            ]

            for pat in patterns:
                for m in re.finditer(pat, day_text, re.I):
                    word = None
                    for g in m.groups():
                        if g and g.lower() in read_words:
                            word = g.lower()
                            break

                    if word:
                        reads.append({
                            "speaker": speaker,
                            "target": target,
                            "read": read_words[word],
                            "raw_word": word,
                            "evidence_type": "soft_read",
                        })

    seen = set()
    unique = []
    for r in reads:
        key = (r["speaker"], r["target"], r["read"], r["raw_word"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

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

        # fallback: if parser fails, keep old behavior
        if messages:
            claims = extract_claims_from_messages(messages, players)
            votes = extract_votes_from_messages(messages, players)

            # divination / soft read 可以先沿用舊的全文 regex，下一步再改
            divinations = extract_divinations(content, players)
            soft_reads = extract_soft_reads(content, players)
        else:
            claims = extract_claims(content, players)
            votes = extract_votes(content, players)
            divinations = extract_divinations(content, players)
            soft_reads = extract_soft_reads(content, players)

        if str(day).isdigit() and int(day) >= 2:
            deaths = extract_deaths(content)
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