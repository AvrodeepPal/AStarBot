"""Schema contract for data/*.json — including the real files on disk."""

import json
from pathlib import Path

import pytest

from rag.knowledge import build_metadata, context_block, embedding_text, normalize_entry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RAW = {
    "id": "faq-gate",
    "title": "Is he preparing for GATE?",
    "question": "Is Avrodeep preparing for GATE?",
    "text": "Yes — GATE 2028 is his main target.",
    "links": ["https://example.com"],
    "tags": ["gate", "exam"],
    "aliases": ["is he giving gate", "gate plans"],
    "priority": 5,
    "last_verified": "2026-09-21",
}


def test_embedding_text_includes_title_question_and_aliases():
    out = embedding_text(normalize_entry(RAW, "faq"))
    assert out.startswith("Is he preparing for GATE?")       # title leads
    assert "Is Avrodeep preparing for GATE?" in out          # faq question
    assert "GATE 2028 is his main target." in out            # body
    assert "is he giving gate" in out and "gate plans" in out  # aliases vectorised


def test_context_block_excludes_retrieval_machinery():
    block = context_block(normalize_entry(RAW, "faq"))
    assert "Is he preparing for GATE?" in block
    assert "Links: https://example.com" in block
    for leaked in ("faq-gate", "gate plans", "tags", "priority"):
        assert leaked not in block


def test_context_block_without_links_or_title():
    assert context_block({"text": "Just body"}) == "Just body"


def test_metadata_keeps_clean_text_and_drops_aliases():
    meta = build_metadata(normalize_entry(RAW, "faq"))
    assert meta["text"] == RAW["text"]          # clean text, no alias soup
    assert "aliases" not in meta
    assert meta["source"] == "faq" and meta["priority"] == 5
    assert meta["last_verified"] == "2026-09-21"


def test_empty_optional_fields_omitted_from_metadata():
    meta = build_metadata(normalize_entry({"id": "x", "text": "t"}, "self"))
    assert "question" not in meta and "last_verified" not in meta
    assert meta["priority"] == 3


@pytest.mark.parametrize("bad", [{"text": "no id"}, {"id": "no-text"}, {"id": "x", "text": "   "}])
def test_invalid_entries_rejected(bad):
    with pytest.raises(ValueError):
        normalize_entry(bad, "self")


# ---- the real knowledge base ----------------------------------------------


def _all_entries():
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload["entries"] if isinstance(payload, dict) else payload
        for e in entries:
            yield path.name, e


@pytest.mark.skipif(not list(DATA_DIR.glob("*.json")), reason="no knowledge base present")
def test_real_data_normalizes_and_ids_are_unique():
    seen = {}
    for filename, raw in _all_entries():
        entry = normalize_entry(raw, filename)  # raises if malformed
        dupe_in = seen.get(entry["id"])
        assert not dupe_in, f"duplicate id {entry['id']} in {filename} and {dupe_in}"
        seen[entry["id"]] = filename
        assert entry["title"], f"{entry['id']} has no title (it leads the embedding text)"
        assert 1 <= entry["priority"] <= 5, f"{entry['id']} priority out of range"
    assert len(seen) > 0
