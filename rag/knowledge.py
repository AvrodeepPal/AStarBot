"""Knowledge-base schema: the contract between data/*.json, Pinecone and the prompt.

A data file is ``{"meta": {...}, "entries": [...]}`` (a bare list is also
accepted for backwards compatibility). An entry looks like::

    {
      "id": "faq-gate",
      "title": "Is he preparing for GATE?",
      "question": "Is Avrodeep preparing for GATE?",   # faq.json only
      "text": "Yes — GATE 2028 is his main target...",
      "links": ["https://..."],
      "tags": ["gate", "exam"],
      "aliases": ["is he giving gate", "gate plans"],
      "priority": 5,
      "last_verified": "2026-09-21"                     # optional
    }

Two derived strings matter, and they are deliberately different:

``embedding_text``  what gets vectorised: title + question + text + aliases.
    The title disambiguates chunks that are meaningless alone ("Not right now.
    He's committed to his role...") and the aliases only influence retrieval if
    they are actually inside the vector.

``context_block``   what the LLM reads: title + text + links. No tags, no
    aliases, no ids — they are retrieval machinery and would only invite the
    model to quote them.
"""

from typing import Any

# Entry fields copied verbatim into Pinecone metadata (small, all useful at
# answer or debug time). `aliases` is intentionally excluded: it is baked into
# the vector and would otherwise just inflate every payload.
METADATA_FIELDS = ("text", "title", "question", "links", "tags", "priority", "last_verified")

DEFAULT_PRIORITY = 3


def normalize_entry(entry: dict, source: str) -> dict:
    """Validate and normalise one raw JSON entry. Raises ValueError if unusable."""
    entry_id = (entry.get("id") or "").strip()
    text = (entry.get("text") or "").strip()
    if not entry_id:
        raise ValueError("entry is missing 'id'")
    if not text:
        raise ValueError(f"entry '{entry_id}' is missing 'text'")

    return {
        "id": entry_id,
        "title": (entry.get("title") or "").strip(),
        "question": (entry.get("question") or "").strip(),
        "text": text,
        "links": [str(x) for x in (entry.get("links") or [])],
        "tags": [str(x) for x in (entry.get("tags") or [])],
        "aliases": [str(x) for x in (entry.get("aliases") or [])],
        "priority": int(entry.get("priority", DEFAULT_PRIORITY)),
        "last_verified": (entry.get("last_verified") or "").strip(),
        "source": source,
    }


def embedding_text(entry: dict) -> str:
    """Compose the string that is vectorised for this entry.

    Order matters a little: title first so the leading tokens carry the topic,
    then the literal FAQ question, then the body, then aliases as a trailing
    bag of paraphrases users actually type.
    """
    parts: list[str] = []
    if entry.get("title"):
        parts.append(entry["title"])
    if entry.get("question"):
        parts.append(entry["question"])
    parts.append(entry["text"])
    if entry.get("aliases"):
        parts.append(" ".join(entry["aliases"]))
    return "\n".join(parts)


def build_metadata(entry: dict) -> dict[str, Any]:
    """Pinecone metadata payload: only non-empty scalar/list values."""
    meta: dict[str, Any] = {"source": entry["source"]}
    for field in METADATA_FIELDS:
        value = entry.get(field)
        if value or value == 0:
            meta[field] = value
    return meta


def context_block(match: dict) -> str:
    """Render one retrieved match as the text the LLM sees."""
    lines: list[str] = []
    if match.get("title"):
        lines.append(match["title"])
    lines.append(match.get("text", ""))
    links = match.get("links") or []
    if links:
        lines.append("Links: " + " ".join(links))
    return "\n".join(line for line in lines if line)
