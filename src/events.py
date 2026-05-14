"""Fast deterministic objective event extraction and claim repair.

This version is deliberately conservative about noisy items (hunter/guard/withdrawal)
and more aggressive about high-value board facts (Seer/Medium COs, formation
consensus, and ability-result candidates).  The goal is to stop treating a 2-1
or 3-1 game as 0-0 gray search.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.parser import split_days, parse_messages, canonical_player_name


# --------------------------
# Generic helpers
# --------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _day_int(day: Any) -> Optional[int]:
    try:
        return int(str(day))
    except Exception:
        return None


def _claim_name(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("claim", ""))
    return str(c)


def _contains_any(text: str, pats: List[str]) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


def _player_pattern(player: str) -> str:
    return rf"\[?{re.escape(player)}\]?"


def _message_day(parsed: Dict[str, Any], order: Any) -> Optional[str]:
    if order is None:
        return None
    for msg in parsed.get("messages", []):
        if msg.get("no") == order:
            return str(msg.get("day"))
    return None


def _last_token(player: str) -> str:
    parts = [p for p in re.split(r"\s+", str(player).strip()) if p]
    return parts[-1] if parts else str(player)


def _alias_map(players: List[str]) -> Dict[str, str]:
    """Alias -> canonical player.  Adds common translated role-name aliases."""
    aliases: Dict[str, str] = {}
    for p in players:
        aliases[p.lower()] = p
        last = _last_token(p)
        if len(last) >= 3:
            aliases[last.lower()] = p
        low = p.lower()
        # Common role-title translations in this dataset.
        if "shepherd" in low:
            aliases.setdefault("sheep", p)
            aliases.setdefault("shepherd", p)
        if "baker" in low:
            aliases.setdefault("baker", p)
            aliases.setdefault("inn", p)
            aliases.setdefault("innkeeper", p)
        if "father" in low:
            aliases.setdefault("father", p)
            aliases.setdefault("god", p)
            aliases.setdefault("priest", p)
        if "librarian" in low:
            aliases.setdefault("librarian", p)
            aliases.setdefault("book", p)
            aliases.setdefault("bookkeeper", p)
        if "wounded soldier" in low:
            aliases.setdefault("soldier", p)
            aliases.setdefault("wounded soldier", p)
        if "young girl" in low:
            aliases.setdefault("girl", p)
            aliases.setdefault("young girl", p)
        if "young man" in low:
            aliases.setdefault("young man", p)
            aliases.setdefault("blue", p)
        if "old man" in low:
            aliases.setdefault("old man", p)
            aliases.setdefault("old", p)
        for title in ["farmer", "merchant", "mayor", "traveler", "outlaw", "tailor"]:
            if title in low:
                aliases.setdefault(title, p)
    return aliases


def _alias_regex(alias: str) -> str:
    return rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])"


def _mentioned_players(text: str, players: List[str]) -> List[str]:
    amap = _alias_map(players)
    hits = []
    low_text = text.lower()
    for alias, p in amap.items():
        if len(alias) < 3:
            continue
        if re.search(_alias_regex(alias), low_text, re.I):
            hits.append(p)
    return list(dict.fromkeys(hits))




def _name_pat(alias: str) -> str:
    """Regex for a player alias; spaces and underscores are equivalent in logs."""
    alias = re.sub(r"[\[\]【】]", "", str(alias or "")).strip()
    parts = [p for p in re.split(r"[\s_]+", alias) if p]
    if not parts:
        return r"a^"
    return r"[ _]+".join(re.escape(p) for p in parts)


def _title_aliases_for_player(player: str) -> List[str]:
    low = player.lower()
    titles = []
    if "optimist" in low: titles += ["Optimist"]
    if "young man" in low: titles += ["Young Man"]
    if "sister" in low: titles += ["Sister"]
    if "wounded soldier" in low: titles += ["Wounded Soldier", "Soldier"]
    if "young girl" in low: titles += ["Young Girl", "Girl"]
    if "village girl" in low: titles += ["Village Girl", "Girl"]
    if "librarian" in low: titles += ["Librarian"]
    if "baker" in low: titles += ["Baker", "Innkeeper", "Inn"]
    if "outlaw" in low: titles += ["Outlaw"]
    if "old man" in low: titles += ["Old Man", "Old"]
    if "farmer" in low: titles += ["Farmer"]
    if "father" in low: titles += ["Father", "Priest", "God"]
    if "shepherd" in low: titles += ["Shepherd", "Sheep"]
    if "tailor" in low: titles += ["Tailor"]
    if "traveler" in low: titles += ["Traveler"]
    if "mayor" in low: titles += ["Mayor"]
    return titles


def _player_aliases_for_public_text(player: str) -> List[str]:
    """Aliases for system text: full names, [LAST], [LAST the TITLE], FULL_NAME_AS_UNDERSCORE."""
    aliases = {player, player.upper(), player.replace(" ", "_"), player.replace(" ", "_").upper()}
    last = _last_token(player)
    if last:
        aliases.update({last, last.upper()})
    for title in _title_aliases_for_player(player):
        aliases.add(f"{last} the {title}")
        aliases.add(f"{title} {last}")
        aliases.add(f"{last.upper()}_THE_{title.upper().replace(' ', '_')}")
        aliases.add(f"{title.upper().replace(' ', '_')}_{last.upper()}")
    # Special compact system aliases sometimes used in translated logs.
    aliases.add(last.lower().capitalize())
    return list(dict.fromkeys([a for a in aliases if a]))


def _public_player_regex(player: str) -> str:
    inner = "|".join(_name_pat(a) for a in _player_aliases_for_public_text(player))
    return rf"(?:[\[【]\s*(?:{inner})\s*[\]】]|(?:{inner}))"


# --------------------------
# Public death / GJ events
# --------------------------

def extract_public_death_events(day_text: str, day: str, players: List[str]) -> Dict[str, Any]:
    """Extract public system deaths using both full names and bracket aliases.

    The corpus often switches from full names to formats such as [LIZA],
    [Friedel the Sister], or [FATHER_JIMZON].  The old exact-full-name matcher
    missed those, creating impossible worlds where night-killed humans became wolves.
    """
    events = {"executed": [], "night_kills": [], "sudden_deaths": [], "no_attack": False, "gj": False}

    def add_unique(kind: str, item: Dict[str, Any]) -> None:
        key = (item.get("day"), item.get("player"), item.get("source"))
        existing = {(x.get("day"), x.get("player"), x.get("source")) for x in events[kind]}
        if key not in existing:
            events[kind].append(item)

    for p in players:
        pp = _public_player_regex(p)
        if re.search(rf"{pp}\s+was executed by the villagers", day_text, re.I):
            add_unique("executed", {"day": day, "player": p, "confidence": 0.98, "source": "system_execution"})
        elif re.search(rf"{pp}\s+was executed", day_text, re.I):
            add_unique("executed", {"day": day, "player": p, "confidence": 0.92, "source": "execution_text"})

        if re.search(rf"The next morning,\s*{pp}\s+was found in a gruesome state", day_text, re.I):
            add_unique("night_kills", {"day": day, "player": p, "confidence": 0.98, "source": "system_night_kill"})
        elif re.search(rf"{pp}\s+was found in a gruesome state", day_text, re.I):
            add_unique("night_kills", {"day": day, "player": p, "confidence": 0.93, "source": "night_kill_text"})

        if re.search(rf"{pp}\s+(?:suddenly died|died from sudden death)", day_text, re.I):
            add_unique("sudden_deaths", {"day": day, "player": p, "confidence": 0.92, "source": "sudden_death_text"})

    no_attack_pats = [
        r"There were no attacks", r"there were no attacks", r"no attacks? occurred",
        r"no one was found in a gruesome state", r"no victims?", r"nobody died", r"no one died",
    ]
    if _contains_any(day_text, no_attack_pats):
        events["no_attack"] = True
        events["gj"] = True
    elif not events["night_kills"] and _day_int(day) and _day_int(day) >= 3:
        head = day_text[:2600]
        if re.search(r"\bGJ\b|Good Job|guard success|successful guard|no attack", head, re.I):
            events["no_attack"] = True
            events["gj"] = True
    return events

# --------------------------
# Claim repair
# --------------------------

def _has_not_claim(text: str, role: str) -> bool:
    if role == "seer":
        return bool(re.search(r"not\s+(?:a\s+|the\s+)?Seer|not Seer|can[’']?t\s+see\s+the\s+Seer|非占|\[Non-Seer\]", text, re.I))
    if role == "medium":
        return bool(re.search(r"not\s+(?:a\s+|the\s+)?Medium|not Medium|can[’']?t\s+see\s+(?:the\s+)?Medium|非霊|非靈|not Seer\s*/\s*not Medium", text, re.I))
    return False


def _is_self_seer_claim_text(text: str) -> bool:
    t = str(text or "")
    # Very short self-COs in these translated logs can be just "Seer claim".
    # Accept only when it is effectively the whole message, not when it is a discussion.
    if re.search(r"^\s*(?:it[’']?s begun!?\s*)?(?:\[\s*)?Seer\s+(?:claim|CO)(?:\s*\])?[\.!。]*\s*$", t, re.I):
        return True
    # Reject CO-confirmation posts that mention someone else rather than self.
    if re.search(r"\b(confirm|confirmed|confirming|confirms)\b.*\[?\s*Seer\s+(?:Claim|CO)", t, re.I) and not re.search(r"^\s*\[?\s*Seer\s+(?:Claim|CO)\s*\]?", t, re.I):
        return False
    if re.search(r"\b\w+\s+is\s+confirming\s+the\s+\[?Seer\s+Claim\]?", t, re.I):
        return False
    if _has_not_claim(t, "seer"):
        # Allow rare explicit self-CO in same post only if a strong bracket self marker appears after the not claim.
        if not re.search(r"\[(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\]|\[Seer\s*CO\]", t, re.I):
            return False
    strong = [
        r"\[\s*Seer\s*CO\s*\]",
        r"\[\*?\*?\s*Seer\s+claim\s*\*?\*?\]",
        r"\[\s*(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\s*\]",
        r"\bI[’']?m\s+(?:a\s+|the\s+)?Seer\b",
        r"\bI\s+am\s+(?:a\s+|the\s+)?Seer\b",
        r"\bIt\s+seems\s+I[’']?m\s+the\s+\[?Seer\]?",
        r"\bmy\s+(?:real\s+)?profession\s+is\s+(?:a\s+)?\[?Seer\]?",
        r"\bI\s+can\s+see\s+the\s+future\b",
        r"\bprophecy\s+was\s+that\s+I\s+was\s+the\s+Seer\b",
        r"占CO|占いCO|占い師CO|【占】",
    ]
    if not any(re.search(p, t, re.I | re.M) for p in strong):
        return False
    # Reject pure discussion/confirmation of someone else's claim.
    if re.search(r"confirmed\s+.*Seer claim|I confirmed\s+.*Seer|someone.*Seer|other.*Seer|would .*Seer|maybe .*Seer", t, re.I):
        return bool(re.search(r"\[\s*Seer\s*CO\s*\]|\[(?:Me\s*:\s*)?I[’']?m\s+(?:the\s+)?Seer\]", t, re.I))
    return True


def _is_self_medium_claim_text(text: str, speaker: Optional[str] = None) -> bool:
    t = str(text or "")
    # Reject hypothetical or parenthetical jokes, common in translated logs.
    if re.search(r"\bif\b.{0,120}I[’']?m\s+(?:a\s+|the\s+)?Medium", t, re.I):
        return False
    if re.search(r"\([^)]{0,80}I[’']?m\s+(?:a\s+|the\s+)?medium[^)]{0,80}\)", t, re.I):
        return False
    if re.search(r"\b(confirm|confirmed|confirming|confirms)\b.*\[?\s*Medium\s+(?:Claim|CO)", t, re.I) and not re.search(r"^\s*\[?\s*Medium\s+(?:Claim|CO)\s*\]?", t, re.I):
        return False
    if _has_not_claim(t, "medium"):
        if not re.search(r"\[Medium\s*CO\]|\[Medium\s+Claim\]", t, re.I):
            return False
    strong = [
        r"\[\s*Medium\s*CO\s*\]",
        r"\[\*?\*?\s*Medium\s+claim\s*\*?\*?\]",
        r"(?:^|\n|\.)\s*Medium\s+(?:claim|CO)\b",
        r"\bI[’']?m\s+(?:a\s+|the\s+)?Medium\b",
        r"\bI\s+am\s+(?:a\s+|the\s+)?Medium\b",
        r"\bI[’']?ve\s+awakened\s+to\s+my\s+spiritual\s+abilities\b",
        r"\bability\s+to\s+identify\s+who[’']?s\s+dead\b",
        r"霊CO|霊能CO|霊媒CO|【霊】|靈CO|霊能者CO",
    ]
    if speaker:
        last = re.escape(_last_token(speaker))
        strong.append(rf"\[?\s*{last}\s+is\s+the\s+Medium\s*\]?")
    if not any(re.search(p, t, re.I | re.M) for p in strong):
        return False
    if re.search(r"confirmed\s+.*Medium claim|I confirmed\s+.*Medium|someone.*Medium|other.*Medium|would .*Medium|maybe .*Medium", t, re.I):
        return bool(re.search(r"\[\s*Medium\s*CO\s*\]|\[\s*Medium\s+Claim\s*\]", t, re.I))
    return True


def _add_claim(repaired: Dict[str, List[Dict[str, Any]]], player: str, claim: str, msg: Dict[str, Any], day: str, source: str, confidence: float = 0.80) -> None:
    repaired[player].append({
        "claim": claim,
        "day": str(day),
        "order": msg.get("no"),
        "time": msg.get("time"),
        "text": str(msg.get("text", ""))[:360],
        "source": source,
        "confidence": confidence,
    })


def _extract_mapping_claims(text: str, players: List[str]) -> List[Tuple[str, str]]:
    """Extract compact summaries like 'Peter: Seer, Otto: Medium, Nicholas: Seer'."""
    amap = _alias_map(players)
    out: List[Tuple[str, str]] = []
    # Look around explicit role words; avoid long prose by requiring colon/equals/arrow or bracket-like mapping.
    for alias, player in amap.items():
        if len(alias) < 3:
            continue
        a = re.escape(alias)
        if re.search(rf"{a}\s*(?:[:：=]|->|→)\s*['\"\[]?\s*Seer\b", text, re.I):
            out.append((player, "Seer CO"))
        if re.search(rf"{a}\s*(?:[:：=]|->|→)\s*['\"\[]?\s*Medium\b", text, re.I):
            out.append((player, "Medium CO"))
    return list(dict.fromkeys(out))


def _extract_confirm_claims(text: str, players: List[str]) -> List[Tuple[str, str]]:
    """Extract only compact third-party claim confirmations.

    This deliberately ignores ordinary prose such as "X looks like a seer" or
    "with X confirmed as Medium..." because those phrases produced many fake COs.
    """
    if not re.search(r"Confirmed\s*:|\[Confirmed|CO\s+confirmed|claim\s+confirmed|確認", text, re.I):
        return []
    # Limit the context; long analytical posts with many names are unreliable.
    if len(text) > 520 and not re.search(r"\[Confirmed", text, re.I):
        return []
    amap = _alias_map(players)
    out: List[Tuple[str, str]] = []
    for alias, player in amap.items():
        if len(alias) < 3:
            continue
        a = re.escape(alias)
        possessive = rf"{a}(?:['’]s)?"
        if re.search(rf"(?:Confirmed\s*:|\[Confirmed[^\]]*)[^\n\]]{{0,160}}{possessive}\s+(?:seer\s+claim|Seer\s+CO)", text, re.I):
            out.append((player, "Seer CO"))
        if re.search(rf"(?:Confirmed\s*:|\[Confirmed[^\]]*)[^\n\]]{{0,160}}{possessive}\s+(?:medium\s+claim|Medium\s+CO)", text, re.I):
            out.append((player, "Medium CO"))
    return list(dict.fromkeys(out))

def extract_formation_hints(raw_text: str) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    for day, content in split_days(raw_text).items():
        if not str(day).isdigit():
            continue
        for msg in parse_messages(content):
            text = str(msg.get("text", ""))
            for m in re.finditer(r"(?<!\d)([1-4])\s*[-–—]\s*([0-3])(?!\d)", text):
                # Require a formation-ish context; avoid scores or dates.
                window = text[max(0, m.start() - 80): m.end() + 80]
                if re.search(r"formation|current situation|confirmed|alignment|CO|situation|盤面|陣形|confirmed", window, re.I):
                    hints.append({
                        "day": str(day), "order": msg.get("no"), "time": msg.get("time"),
                        "formation": f"{m.group(1)}-{m.group(2)}", "confidence": 0.72,
                        "text": _norm(window)[:240], "source": "formation_consensus_regex",
                    })
    return hints


def extract_repaired_claims(raw_text: str, players: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    repaired: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    player_set = set(players)
    for day, content in split_days(raw_text).items():
        if not str(day).isdigit():
            continue
        for msg in parse_messages(content):
            speaker = canonical_player_name(msg.get("speaker"), players)
            msg["speaker"] = speaker
            if speaker not in player_set:
                continue
            text = str(msg.get("text", ""))
            if _is_self_not_claim_text(text, "Not Seer/Medium") or (_has_not_claim(text, "seer") and _has_not_claim(text, "medium")):
                _add_claim(repaired, speaker, "Not Seer/Medium", msg, str(day), "self_not_claim_repair", 0.88)
            if _is_self_seer_claim_text(text):
                _add_claim(repaired, speaker, "Seer CO", msg, str(day), "self_claim_repair", 0.92)
            if _is_self_medium_claim_text(text, speaker=speaker):
                _add_claim(repaired, speaker, "Medium CO", msg, str(day), "self_claim_repair", 0.92)

            for p, claim in _extract_mapping_claims(text, players):
                _add_claim(repaired, p, claim, msg, str(day), "mapping_claim_repair", 0.78)
            for p, claim in _extract_confirm_claims(text, players):
                # Treat third-party as repair candidate, not as strong as self claim.
                _add_claim(repaired, p, claim, msg, str(day), "confirm_claim_repair", 0.70)

    # Deduplicate by (player, claim, order, source-ish text).
    deduped: Dict[str, List[Dict[str, Any]]] = {}
    for p, claims in repaired.items():
        seen = set()
        arr = []
        for c in sorted(claims, key=lambda x: (x.get("order") if x.get("order") is not None else 10**9, -float(x.get("confidence", 0)))):
            key = (c.get("claim"), c.get("order"))
            if key in seen:
                continue
            seen.add(key)
            arr.append(c)
        deduped[p] = arr
    return deduped


def _is_self_not_claim_text(text: str, claim: str) -> bool:
    t = str(text or "")
    low = t.lower()
    if ">>" in t and re.search(r"second|third|order|reason|about|position|co-order|claim order", low, re.I):
        return False
    if claim == "Not Seer/Medium":
        return bool(re.search(r"\[?\*?\*?\s*(?:not Seer\s*/\s*not Medium|neither a Seer nor a Medium|not a Seer or Medium|I.?m not a Seer,? and I.?m not a Medium|I.?m not Seer\s*/\s*not Medium|I\s+can[’']?t\s+see\s+the\s+Seer,?\s+and\s+I\s+can[’']?t\s+see\s+(?:the\s+)?Medium)", t, re.I))
    if claim == "Not Seer":
        return bool(re.search(r"\bI.?m not (?:a |the )?Seer\b|\bnot Seer\b|\[Non-Seer\]|非占", t, re.I)) and not re.search(r"reason|only doing|position|order", t, re.I)
    if claim == "Not Medium":
        return bool(re.search(r"\bI.?m not (?:a |the )?Medium\b|\bnot Medium\b|非霊|非靈", t, re.I)) and not re.search(r"reason|position|order", t, re.I)
    return True


def apply_claim_repairs_to_parsed(parsed: Dict[str, Any], repaired_claims: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Clean false-positive COs and add repaired claims to parsed structures."""
    def clean_claim_list(claims: List[Any], p: str) -> List[Any]:
        new_claims = []
        for c in claims:
            name = _claim_name(c)
            txt = c.get("text", "") if isinstance(c, dict) else ""
            if name == "Seer CO" and not _is_self_seer_claim_text(txt):
                # keep if this exact item was produced by a repair source
                if not (isinstance(c, dict) and str(c.get("source", "")).endswith("repair")):
                    continue
            if name == "Medium CO" and not _is_self_medium_claim_text(txt, speaker=p):
                if not (isinstance(c, dict) and str(c.get("source", "")).endswith("repair")):
                    continue
            if name in {"Not Seer/Medium", "Not Seer", "Not Medium"} and not _is_self_not_claim_text(txt, name):
                continue
            new_claims.append(c)
        return new_claims

    parsed.setdefault("all_claims", {})
    for p, claims in list(parsed.get("all_claims", {}).items()):
        parsed["all_claims"][p] = clean_claim_list(claims, p)

    for day_obj in parsed.get("days", {}).values():
        for p, claims in list(day_obj.get("claims", {}).items()):
            day_obj["claims"][p] = clean_claim_list(claims, p)

    for p, claims in repaired_claims.items():
        parsed["all_claims"].setdefault(p, [])
        existing = {(_claim_name(c), c.get("order") if isinstance(c, dict) else None) for c in parsed["all_claims"].get(p, [])}
        for c in claims:
            key = (c.get("claim"), c.get("order"))
            if key not in existing:
                parsed["all_claims"][p].append(c)
            day = str(c.get("day", ""))
            if day in parsed.get("days", {}):
                parsed["days"][day].setdefault("claims", {}).setdefault(p, [])
                day_existing = {(_claim_name(x), x.get("order") if isinstance(x, dict) else None) for x in parsed["days"][day]["claims"].get(p, [])}
                if key not in day_existing:
                    parsed["days"][day]["claims"][p].append(c)

    # Resolve contradictions caused by translated third-party confirmations.
    # If someone explicitly self-denied Not Seer/Medium, keep a later Seer/Medium CO
    # only when that CO itself is a self-claim, not a repair from another speaker.
    def is_self_role_claim(c: Any, p: str, role: str) -> bool:
        if not isinstance(c, dict):
            return False
        src = str(c.get("source", ""))
        txt = str(c.get("text", ""))
        if src == "self_claim_repair":
            return True
        if role == "Seer CO":
            return _is_self_seer_claim_text(txt)
        if role == "Medium CO":
            return _is_self_medium_claim_text(txt, speaker=p)
        return False

    for p, cs in list(parsed.get("all_claims", {}).items()):
        names = [_claim_name(c) for c in cs]
        has_not_both = "Not Seer/Medium" in names or ("Not Seer" in names and "Not Medium" in names)
        filtered = []
        has_seer = "Seer CO" in names
        for c in cs:
            nm = _claim_name(c)
            txt = str(c.get("text", "")) if isinstance(c, dict) else ""
            if has_not_both and nm in {"Seer CO", "Medium CO"} and not is_self_role_claim(c, p, nm):
                continue
            if has_seer and nm == "Medium CO" and not re.search(r"\[\s*Medium\s*CO\s*\]|Medium\s+claim|slide|スライド|霊CO|霊能CO|霊媒CO|【霊】", txt, re.I):
                continue
            filtered.append(c)
        parsed["all_claims"][p] = filtered

    for day_obj in parsed.get("days", {}).values():
        for p, cs in list(day_obj.get("claims", {}).items()):
            global_claims = parsed.get("all_claims", {}).get(p, [])
            allowed = {(_claim_name(c), c.get("order") if isinstance(c, dict) else None) for c in global_claims}
            day_obj["claims"][p] = [c for c in cs if (_claim_name(c), c.get("order") if isinstance(c, dict) else None) in allowed]

    for p, cs in parsed.get("all_claims", {}).items():
        cs.sort(key=lambda c: (c.get("order") if isinstance(c, dict) and c.get("order") is not None else 10**9))
    for day_obj in parsed.get("days", {}).values():
        for p, cs in day_obj.get("claims", {}).items():
            cs.sort(key=lambda c: (c.get("order") if isinstance(c, dict) and c.get("order") is not None else 10**9))
    return parsed


def _claimers(parsed: Dict[str, Any], claim: str) -> List[str]:
    out = []
    for p, claims in parsed.get("all_claims", {}).items():
        if any(_claim_name(c) == claim for c in claims):
            out.append(p)
    return list(dict.fromkeys(out))


# --------------------------
# Result extraction
# --------------------------

def _result_word_to_alignment(word: str) -> Optional[str]:
    w = str(word or "").lower().strip()
    if w in {"human", "a human", "white", "not werewolf", "village", "villager", "town", "白", "人間"}:
        return "human"
    if w in {"werewolf", "a werewolf", "wolf", "black", "黒", "黑", "狼", "人狼"}:
        return "werewolf"
    return None




def _clean_target_fragment(fragment: str) -> str:
    x = str(fragment or "")
    x = re.sub(r"[\[\]【】『』「」()（）]", " ", x)
    x = re.sub(r"\b(?:san|chan|kun|sama|Mr|Ms|the)\b", " ", x, flags=re.I)
    x = re.sub(r"[^A-Za-z_\s-]", " ", x)
    return _norm(x).lower().replace("_", " ")


def _canonical_target_from_fragment(fragment: str, players: List[str]) -> Optional[str]:
    clean = _clean_target_fragment(fragment)
    if not clean:
        return None
    amap = _alias_map(players)
    # Add public-text aliases like "friedel the sister" and "father_jimzon".
    for p in players:
        for a in _player_aliases_for_public_text(p):
            amap.setdefault(_clean_target_fragment(a), p)
    for alias, player in sorted(amap.items(), key=lambda kv: len(kv[0]), reverse=True):
        if len(alias) < 3:
            continue
        a = _clean_target_fragment(alias)
        if not a:
            continue
        if re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", clean, re.I):
            return player
    return None


def _add_result(out: List[Tuple[str, str]], target: Optional[str], word: str) -> None:
    res = _result_word_to_alignment(word)
    if target and res and (target, res) not in out:
        out.append((target, res))

def _extract_result_candidates_from_text(text: str, players: List[str]) -> List[Tuple[str, str]]:
    """Strict ability-result candidate extraction.

    Accept compact result lines and explicit role reports only.  This avoids turning
    ordinary discussion like "if X is black" or "Y looks white" into hard results.
    """
    out: List[Tuple[str, str]] = []
    raw = str(text or "")
    t = _norm(raw)
    if re.search(r"\bGS\b|gray scale|white\s+.+>.+black|current survivors|survivors are", t, re.I):
        return []

    # Explicit fortune-telling style: "Today's guest is Friedel-san ... [Fix White]".
    guest = re.search(r"(?:Today['’]s guest is|guest is|target is)\s+([A-Za-z_\s\-]+?)(?:-san|-chan|-kun|\.|\n|$)", raw, re.I)
    if guest and re.search(r"\[(?:Fix\s+)?(?:White|Human)\]|(?:Fix\s+)?White\b|human result", raw, re.I):
        _add_result(out, _canonical_target_from_fragment(guest.group(1), players), "human")
    if guest and re.search(r"\[(?:Fix\s+)?(?:Black|Werewolf)\]|(?:Fix\s+)?Black\b|werewolf result", raw, re.I):
        _add_result(out, _canonical_target_from_fragment(guest.group(1), players), "werewolf")

    # Scan short lines. Most true result announcements are one-line or bracketed.
    for line in raw.splitlines():
        ln = _norm(line)
        if not ln or len(ln) > 180:
            continue
        if re.search(r"\bif\b|assuming|would|could|maybe|think|thought|seem(?:ed|s)?|kinda|seer preference|●|○|▼|▽|suspect|not\s+like|not\s+.+\b(?:black|white|wolf|human)\b|wolf image|from .* perspective", ln, re.I):
            continue

        # [Human] Friedel / [Black] Valter
        m = re.search(r"^[\[【]\s*(Human|White|Werewolf|Wolf|Black)\s*[\]】]\s*(.+)$", ln, re.I)
        if m:
            _add_result(out, _canonical_target_from_fragment(m.group(2), players), m.group(1))
            continue

        # 【Pamela was human.】 / Friedel was the werewolf.
        m = re.search(r"^[\[【]?\s*(.+?)\s+(?:was|is|came out|turned out)\s+(?:a\s+|the\s+)?(human|white|werewolf|wolf|black)\.?\s*[\]】]?$", ln, re.I)
        if m:
            _add_result(out, _canonical_target_from_fragment(m.group(1), players), m.group(2))
            continue

        # Moritz: "White." / Jimzon: White.
        m = re.search(r"^(.{2,60}?)[：:]\s*[\"“”']?\s*(White|Black|Human|Werewolf|Wolf)\.?[\"“”']?\s*$", ln, re.I)
        if m:
            _add_result(out, _canonical_target_from_fragment(m.group(1), players), m.group(2))
            continue

        # Japanese/mixed compact result forms.
        m = re.search(r"^(.{2,60}?)[：:]?\s*(白|黒|黑|人間|狼|人狼)\s*$", ln, re.I)
        if m:
            _add_result(out, _canonical_target_from_fragment(m.group(1), players), m.group(2))
            continue

    # Fallback for short explicit result posts where target and color appear in same sentence.
    if len(t) <= 520 or re.search(r"\b(result|verdict|judgment|announcement|announce|判定|結果|占い結果|霊判定)\b", t, re.I):
        for sent in re.split(r"(?<=[\.。!?])\s+|\n", raw):
            sent_norm = _norm(sent)
            if len(sent_norm) > 220:
                continue
            if re.search(r"\bif\b|assuming|would|could|maybe|think|thought|seem(?:ed|s)?|kinda|seer preference|●|○|▼|▽|suspect|not\s+like|not\s+.+\b(?:black|white|wolf|human)\b|from .* perspective", sent_norm, re.I):
                continue
            m = re.search(r"(.{2,80}?)\s+(?:was|is|came out|turned out)\s+(?:a\s+|the\s+)?(human|white|werewolf|wolf|black)\b", sent_norm, re.I)
            if m:
                _add_result(out, _canonical_target_from_fragment(m.group(1), players), m.group(2))
    return out

def extract_ability_results(parsed: Dict[str, Any], players: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    seer_claimers = set(_claimers(parsed, "Seer CO"))
    medium_claimers = set(_claimers(parsed, "Medium CO"))
    seer_results: List[Dict[str, Any]] = []
    medium_results: List[Dict[str, Any]] = []

    result_marker = re.compile(r"(result|verdict|judgment|announcement|announce|was white|was black|is white|is black|came out white|came out black|\bWhite\b|\bBlack\b|\bHuman\b|\bWerewolf\b|判定|結果|占い結果|霊判定|白|黒|黑)", re.I)
    for msg in parsed.get("messages", []):
        speaker = msg.get("speaker")
        if speaker not in seer_claimers and speaker not in medium_claimers:
            continue
        text = str(msg.get("text", ""))
        if not result_marker.search(text):
            continue
        candidates = _extract_result_candidates_from_text(text, players)
        if not candidates:
            continue
        for target, res in candidates:
            if target == speaker:
                continue
            item = {
                "day": str(msg.get("day")), "order": msg.get("no"), "time": msg.get("time"),
                "target": target, "result": res,
                "confidence": 0.62, "text": _norm(text)[:280], "source": "ability_result_candidate",
            }
            if re.search(r"\b(result|verdict|judgment|announcement|announce|判定|結果|占い結果|霊判定)\b", text, re.I):
                item["confidence"] = 0.70
            if speaker in seer_claimers:
                x = dict(item); x["seer"] = speaker
                seer_results.append(x)
            if speaker in medium_claimers:
                x = dict(item); x["medium"] = speaker
                medium_results.append(x)

    def dedupe(items: List[Dict[str, Any]], who_key: str) -> List[Dict[str, Any]]:
        seen = set(); out = []
        for r in sorted(items, key=lambda x: (str(x.get("day")), x.get("order") or 10**9, -float(x.get("confidence", 0)))):
            key = (r.get("day"), r.get(who_key), r.get("target"), r.get("result"), r.get("order"))
            if key in seen:
                continue
            seen.add(key); out.append(r)
        return out
    return {"seer_results": dedupe(seer_results, "seer"), "medium_results": dedupe(medium_results, "medium")}



def extract_claim_withdrawals(parsed: Dict[str, Any], players: List[str]) -> List[Dict[str, Any]]:
    # Only role claimants can withdraw a role claim, and role words must be nearby.
    claimers = set(_claimers(parsed, "Seer CO")) | set(_claimers(parsed, "Medium CO"))
    out: List[Dict[str, Any]] = []
    pat = re.compile(r"(withdraw|withdrawal|retract|retraction|cancel|撤回|スライド|slide)", re.I)
    role_near = re.compile(r"(seer|medium|hunter|CO|claim|role|占|霊|靈|狩)", re.I)
    for msg in parsed.get("messages", []):
        speaker = msg.get("speaker")
        if speaker not in claimers:
            continue
        text = str(msg.get("text", ""))
        if pat.search(text) and role_near.search(text):
            # Avoid ordinary opinion retractions.
            if re.search(r"retract my earlier statement|retract that|withdraw my evaluation|change my seer preference|preference", text, re.I):
                continue
            out.append({
                "day": str(msg.get("day")), "order": msg.get("no"), "time": msg.get("time"),
                "player": speaker, "confidence": 0.72, "text": _norm(text)[:260], "source": "role_withdrawal_regex",
            })
    return out


def extract_hunter_events(parsed: Dict[str, Any], players: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    hunter_claims: List[Dict[str, Any]] = []
    guard_reports: List[Dict[str, Any]] = []
    claim_pat = re.compile(r"(\[\s*Hunter\s*CO\s*\]|\bI[’']?m\s+(?:the\s+)?Hunter\b|\bI\s+am\s+(?:the\s+)?Hunter\b|狩人CO|狩CO)", re.I)
    not_hunter_pat = re.compile(r"not\s+Hunter|\[not\s+Hunter\s*co\]|非狩", re.I)
    guard_self_pat = re.compile(r"\bI\s+(?:guarded|protected)\b|護衛(?:先)?", re.I)
    gj_pat = re.compile(r"\bGJ\b|Good Job", re.I)

    for msg in parsed.get("messages", []):
        speaker = msg.get("speaker")
        if speaker not in players:
            continue
        text = str(msg.get("text", ""))
        if not_hunter_pat.search(text):
            continue
        if claim_pat.search(text) and not re.search(r"\bif\s+I[’']?m\s+(?:the\s+)?Hunter|\bif\s+I\s+were\s+(?:the\s+)?Hunter|maybe\s+I\s+was\s+(?:the\s+)?Hunter", text, re.I):
            hunter_claims.append({
                "day": str(msg.get("day")), "order": msg.get("no"), "time": msg.get("time"),
                "player": speaker, "confidence": 0.86, "text": _norm(text)[:260], "source": "hunter_self_claim_regex",
            })
        if guard_self_pat.search(text) or (claim_pat.search(text) and gj_pat.search(text)):
            targets = [p for p in _mentioned_players(text, players) if p != speaker]
            if targets or gj_pat.search(text):
                guard_reports.append({
                    "day": str(msg.get("day")), "order": msg.get("no"), "time": msg.get("time"),
                    "player": speaker, "targets": targets[:3], "confidence": 0.65 if targets else 0.45,
                    "text": _norm(text)[:260], "source": "guard_self_report_regex",
                })
    return {"hunter_claims": hunter_claims, "guard_reports": guard_reports}


# --------------------------
# Main event object
# --------------------------

def extract_objective_events(raw_text: str, parsed: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    days = split_days(raw_text)
    day_events: Dict[str, Dict[str, Any]] = {}
    for day, content in days.items():
        if str(day).isdigit():
            day_events[str(day)] = extract_public_death_events(content, str(day), players)

    # Parser divinations + improved candidate extraction.
    seer_results: List[Dict[str, Any]] = []
    for r in parsed.get("all_divinations", []):
        if isinstance(r, dict) and r.get("seer") in players and r.get("target") in players:
            rr = dict(r)
            rr["day"] = str(rr.get("day", rr.get("day", ""))) if rr.get("day") else str(_message_day(parsed, rr.get("order")) or "")
            rr.setdefault("confidence", 0.75)
            rr.setdefault("source", "parser_hard_divination")
            seer_results.append(rr)

    ability = extract_ability_results(parsed, players)
    # Candidate results are useful but lower-confidence. Keep them all for line scoring.
    seer_results.extend(ability["seer_results"])
    medium_results = ability["medium_results"]

    withdrawals = extract_claim_withdrawals(parsed, players)
    hunter = extract_hunter_events(parsed, players)
    formation_hints = extract_formation_hints(raw_text)

    executed: List[Dict[str, Any]] = []
    night_kills: List[Dict[str, Any]] = []
    sudden_deaths: List[Dict[str, Any]] = []
    gj_days: List[str] = []
    for day, de in day_events.items():
        executed.extend(de.get("executed", []))
        night_kills.extend(de.get("night_kills", []))
        sudden_deaths.extend(de.get("sudden_deaths", []))
        if de.get("gj"):
            gj_days.append(day)

    def _dedupe_results(items: List[Dict[str, Any]], who: str) -> List[Dict[str, Any]]:
        seen = set(); out = []
        for r in items:
            key = (r.get("day"), r.get(who), r.get("target"), r.get("result"), r.get("order"))
            if key in seen:
                continue
            seen.add(key); out.append(r)
        return out

    return {
        "day_events": day_events,
        "executed": executed,
        "night_kills": night_kills,
        "sudden_deaths": sudden_deaths,
        "gj_days": gj_days,
        "seer_results": _dedupe_results(seer_results, "seer"),
        "medium_results": _dedupe_results(medium_results, "medium"),
        "claim_withdrawals": withdrawals,
        "hunter_claims": hunter["hunter_claims"],
        "guard_reports": hunter["guard_reports"],
        "formation_hints": formation_hints,
        "notes": [
            "Events are deterministic and confidence-weighted.",
            "Hunter/guard/withdrawal extraction is conservative to avoid noisy false positives.",
            "Ability results include lower-confidence candidates for line scoring.",
        ],
    }
