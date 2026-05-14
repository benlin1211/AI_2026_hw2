import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

PRESSURE_PAT = re.compile(r"(suspect|suspicious|doubt|distrust|wolf|werewolf|black|black-ish|lynch|execute|vote|吊|黒|黑|狼|人狼|怪しい|疑)", re.I)
TOWNREAD_PAT = re.compile(r"(white|white-ish|human|villager|town|not wolf|村|白|人間|非狼)", re.I)
DEFENSE_PAT = re.compile(r"(trust|believe|defend|protect|looks good|good impression|信用|信じ|守|庇|かば)", re.I)
CLAIM_PAT = re.compile(r"(claim|CO|seer|medium|counter|fake|true|real|credibility|占|霊|靈|対抗|偽|真)", re.I)
VOTE_PAT = re.compile(r"(▼|▽|lynch|execute|vote|吊|処刑)", re.I)
CHECK_PAT = re.compile(r"(●|○|check|divine|seer target|占い希望|占希望|占う)", re.I)


def _is_meta_or_result_text(text: str) -> bool:
    """Do not turn CO confirmations, formation summaries, or result announcements into pressure edges."""
    t = str(text or "")
    if re.search(r"confirmed.*(?:Seer|Medium).*CO|claims? to be the (?:Seer|Medium)|\[(?:Seer|Medium)\s*CO\]|\[(?:Seer|Medium)\s+Claim\]", t, re.I):
        return True
    if re.search(r"(?:current situation|formation|alignment).*[1-4]\s*[-–—]\s*[0-3]|[1-4]\s*[-–—]\s*[0-3].*(?:confirmed|formation|alignment)", t, re.I):
        return True
    if re.search(r"(?:was|is)\s+(?:white|black|human|werewolf)|confirmed seer result|medium result|verdict|判定|結果", t, re.I):
        return True
    if re.search(r"current survivors|survivors are|was found in a gruesome state", t, re.I):
        return True
    return False


def _target_mentions(text: str, players: List[str], speaker: Optional[str]) -> List[str]:
    hits = []
    for p in players:
        if p == speaker:
            continue
        # exact full-name match first
        if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, re.I):
            hits.append(p)
            continue
        # allow last token for translated names: "Joachim", "Liza", etc.
        parts = [x for x in re.split(r"\s+", p) if x]
        if parts:
            last = parts[-1]
            if len(last) >= 4 and re.search(rf"(?<!\w){re.escape(last)}(?!\w)", text, re.I):
                hits.append(p)
    return list(dict.fromkeys(hits))


def _edge_type(text: str) -> str:
    if VOTE_PAT.search(text):
        return "vote_push"
    if CLAIM_PAT.search(text) and PRESSURE_PAT.search(text):
        return "claim_challenge"
    if PRESSURE_PAT.search(text):
        return "pressure"
    if DEFENSE_PAT.search(text):
        return "defense"
    if TOWNREAD_PAT.search(text):
        return "townread"
    if CHECK_PAT.search(text):
        return "check_preference"
    return "other"


def _reason_type(text: str) -> str:
    if CLAIM_PAT.search(text):
        return "claim"
    if VOTE_PAT.search(text):
        return "vote"
    if CHECK_PAT.search(text):
        return "check"
    if re.search(r"(result|verdict|判定|結果|white|black|白|黒|黑)", text, re.I):
        return "result"
    if re.search(r"(tone|feeling|impression|軽い|重い|印象|姿勢)", text, re.I):
        return "tone"
    return "logic"


def _strength(text: str, edge_type: str) -> float:
    s = 0.35
    if edge_type in ("vote_push", "claim_challenge"):
        s += 0.25
    if re.search(r"(strong|definitely|must|sure|clearly|絶対|確実|かなり|very)", text, re.I):
        s += 0.15
    if re.search(r"(maybe|slightly|not sure|kinda|perhaps|少し|微|かも)", text, re.I):
        s -= 0.10
    if "▼" in text or "●" in text:
        s += 0.15
    return max(0.1, min(1.0, s))


def _timing(day: Any, order: Any) -> str:
    try:
        d = int(day)
    except Exception:
        d = 99
    try:
        o = int(order)
    except Exception:
        o = 10**9
    if d == 1 and o <= 140:
        return "early"
    if d <= 2:
        return "middle"
    return "late"


def build_interaction_graph(
    snippets: List[Dict[str, Any]],
    players: List[str],
    objective_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic interaction graph. It is intentionally cheap and Day1+ only.
    """
    edges: List[Dict[str, Any]] = []
    objective_pack = objective_pack or {}

    # 1) Edges from explicit parsed votes are high confidence.
    for v in objective_pack.get("votes", []):
        voter = v.get("voter")
        target = v.get("target")
        if voter in players and target in players and voter != target:
            vtype = str(v.get("type", ""))
            etype = "vote_push" if "execution" in vtype or vtype == "preference" else "check_preference"
            edges.append({
                "day": v.get("day"),
                "order": v.get("order"),
                "src": voter,
                "dst": target,
                "type": etype,
                "strength": 0.9 if etype == "vote_push" else 0.65,
                "timing": _timing(v.get("day"), v.get("order")),
                "reason_type": "vote" if etype == "vote_push" else "check",
                "evidence": str(v.get("text", ""))[:160],
            })

    # 2) Soft-read edges from parser.
    for r in objective_pack.get("soft_reads", []):
        src = r.get("speaker")
        dst = r.get("target")
        if src in players and dst in players and src != dst:
            read = r.get("read")
            etype = "pressure" if read == "likely_werewolf" else "townread"
            edges.append({
                "day": r.get("day"),
                "order": None,
                "src": src,
                "dst": dst,
                "type": etype,
                "strength": 0.22,
                "timing": _timing(r.get("day"), None),
                "reason_type": "tone",
                "evidence": str(r.get("text", ""))[:160],
            })

    # 3) Edges from strategic snippets.
    for snip in snippets:
        speaker = snip.get("speaker")
        text = str(snip.get("text", ""))
        if speaker not in players:
            continue
        if _is_meta_or_result_text(text):
            continue
        targets = _target_mentions(text, players, speaker)
        if not targets:
            continue

        etype = _edge_type(text)
        if etype == "other":
            continue
        for target in targets[:4]:
            edges.append({
                "day": snip.get("day"),
                "order": snip.get("order"),
                "src": speaker,
                "dst": target,
                "type": etype,
                "strength": _strength(text, etype),
                "timing": _timing(snip.get("day"), snip.get("order")),
                "reason_type": _reason_type(text),
                "evidence": text[:180],
            })

    # Deduplicate near-identical edges, keeping strongest.
    best: Dict[tuple, Dict[str, Any]] = {}
    for e in edges:
        key = (e.get("day"), e.get("src"), e.get("dst"), e.get("type"), e.get("reason_type"))
        if key not in best or float(e.get("strength", 0)) > float(best[key].get("strength", 0)):
            best[key] = e
    edges = list(best.values())
    edges.sort(key=lambda x: (int(x.get("day", 999)) if str(x.get("day", "")).isdigit() else 999, x.get("order") or 10**9))

    summary = defaultdict(lambda: defaultdict(float))
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        etype = e.get("type")
        strength = float(e.get("strength", 0.0))
        if src in players:
            summary[src][f"{etype}_given"] += strength
        if dst in players:
            summary[dst][f"{etype}_received"] += strength

    return {
        "edges": edges[:260],
        "summary": {p: dict(summary[p]) for p in players},
        "notes": [
            "Interaction graph is rule-based and Day1+ only.",
            "Edges are weak evidence; objective hard facts should dominate role inference.",
        ],
    }
