"""Source-backed qualification lists from complete, ordered indexed sections."""

import re

LABELS = {
    "required": "Required qualifications",
    "preferred": "Preferred qualifications",
    "optional": "Optional qualifications",
    "unspecified": "Listed qualifications (level not specified)",
}
_BULLET = re.compile(r"^\s*(?:[•●▪∙‣*–-]|\d+[.)])\s+(.+)")
_QUERY_FILLER = set(
    "a all an and are as at be can could description do does essential experience for from have how i in is it job know looking mandatory mention mentioned must need needed of on optional or particular preferred qualification qualifications required requirements role should skill skills some specific technical tell the there these this to want what which with would you your python developer company list".split()
)


def qualification_groups(question):
    text = question.casefold()
    if re.search(r"\bdo i have\b", text):
        return None
    for typo, word in {"wht": "what", "skils": "skills", "requried": "required"}.items():
        text = re.sub(rf"\b{typo}\b", word, text)
    if re.search(r"\b(?:not|never|non[ -])\s*(?:required|mandatory|essential)\b", text):
        return None
    if re.match(r"^\s*(?:is|are|do|does|am|will|would)\b", text) and not re.search(
        r"\b(?:list|show|summari[sz]e)\b", text
    ):
        return None
    if not re.search(
        r"\bqualif(?:ications?|y|ied|ying)\b|\brequirements?\b|\bmust[ -]haves?\b|\bnice[ -]to[ -]haves?\b|"
        r"\b(?:required|preferred|essential|mandatory|necessary|needed) (?:python )?skills\b|"
        r"\b(?:what|which|all) (?:(?:are|the|specific|particular|essential|technical|required|preferred|python) )*skills\b|\bskills (?:are )?(?:required|preferred|needed)\b",
        text,
    ):
        return None
    required = bool(
        re.search(
            r"\brequired\b|\brequirements?\b|\bessential\b|\bmandatory\b|\bneed|\bmust[ -]have",
            text,
        )
    )
    preferred = bool(re.search(r"\bprefer\w*\b|\bnice.to.haves?\b", text))
    if required and not preferred:
        return {"required", "unspecified"}
    if preferred and not required:
        return {"preferred", "unspecified"}
    return set(LABELS)


def has_other_job_question(question):
    """Qualification-only replies need no extra generated qualification summary."""
    return bool(
        re.search(
            r"\b(?:salary|salaries|pay|compensation|bonus|benefits?|perks|location|located|remote|hybrid|office|hours|responsibilities|duties|tasks|relocation|visa|process|team|workplace|department)\b|\bcompany size\b",
            question,
            re.I,
        )
    )


def select_qualifications(question, items):
    groups = qualification_groups(question)
    related = groups or re.search(
        r"\b(?:skills?|qualif\w*|requ\w*|mandatory|preferred|optional|mention(?:ed|s)?|essential)\b",
        question,
        re.I,
    )
    if related and not re.search(r"\ball\b", question, re.I):

        def words(text):
            return set(re.findall(r"[\w]+(?:[-+][\w]+)*", text.casefold()))

        wanted = words(question) - _QUERY_FILLER
        tokens = [words(item["text"]) for item in items]
        specific = {word for word in wanted if 0 < sum(word in value for value in tokens) <= 2}
        matches = [item for item, value in zip(items, tokens, strict=True) if value & specific]
        if matches:
            return matches
    return [item for item in items if groups and item["category"] in groups]


def _heading(line):
    text = re.sub(r"^#+\s*|[:\s]+$", "", line).casefold()
    if re.fullmatch(
        r"(?:required|essential|minimum|mandatory)(?: (?:skills?|qualifications?|requirements?)(?: and qualifications?)?)?|requirements(?: and qualifications)?|must.haves?",
        text,
    ):
        return "required"
    if re.fullmatch(
        r"(?:preferred|desirable)(?: (?:skills?|qualifications?)(?: and qualifications?)?)?|nice.to.haves?",
        text,
    ):
        return "preferred"
    if re.fullmatch(r"optional(?: (?:skills?|qualifications?))?", text):
        return "optional"
    if text in {"skills", "qualifications", "skills and qualifications", "what you bring"}:
        return "unspecified"
    if (
        re.match(
            r"^(?:responsibilities|your tasks|objectives|benefits|perks|about|compensation|salary|how to apply|what we offer)\b",
            text,
        )
        or line.endswith(":")
        or line.startswith("#")
    ):
        return "end"
    return None


def qualification_items(excerpts, groups):
    """Rejoin chunk overlap and keep citation spans across chunks and pages."""
    versions = {}
    for excerpt in sorted(
        excerpts, key=lambda row: (row["document_version_id"], row["page"], row["chunk_index"])
    ):
        version = versions.setdefault(
            excerpt["document_version_id"], {"text": "", "spans": [], "page": None}
        )
        old, value = version["text"], excerpt["text"]
        overlap = 0
        if version["page"] == excerpt["page"]:
            for size in range(min(len(old), len(value)), 0, -1):
                if old.endswith(value[:size]) and (
                    size >= 20 or value[:size] == value.splitlines()[0] == old.splitlines()[-1]
                ):
                    overlap = size
                    break
        start = len(old) - overlap if overlap else len(old) + bool(old)
        version["text"] = old + ("\n" if old and not overlap else "") + value[overlap:]
        version["spans"].append((start, start + len(value), excerpt["id"]))
        version["page"] = excerpt["page"]
    items = []
    seen = set()
    for version_id, version in versions.items():
        category, current = None, None

        def finish(current, category, version, version_id):
            if current is None or category not in groups:
                return
            text = " ".join(current["lines"])
            key = (category, text.casefold())
            if key in seen:
                return
            sources = [
                key
                for start, end, key in version["spans"]
                if start < current["end"] and end > current["start"]
            ]
            items.append(
                {
                    "id": f"{version_id}:qualification:{current['start']}",
                    "category": category,
                    "text": text,
                    "source_ids": list(dict.fromkeys(sources)),
                }
            )
            seen.add(key)

        for line in re.finditer(r"[^\n]+", version["text"]):
            value = line.group().strip()
            bullet = _BULLET.match(value)
            heading = _heading(value) if not bullet else None
            if heading:
                finish(current, category, version, version_id)
                category, current = (None if heading == "end" else heading), None
            elif bullet and category:
                finish(current, category, version, version_id)
                current = {"lines": [bullet[1]], "start": line.start(), "end": line.end()}
            elif current and not re.fullmatch(r"(?:page\s*)?\d+", value, re.I):
                current["lines"].append(value)
                current["end"] = line.end()
        finish(current, category, version, version_id)
    return items


def render_qualifications(items):
    return "\n\n".join(
        f"**{label}**\n"
        + "\n".join(f"- {item['text']}" for item in items if item["category"] == category)
        for category, label in LABELS.items()
        if any(item["category"] == category for item in items)
    )
