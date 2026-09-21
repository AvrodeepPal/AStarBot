"""scripts.embed record loading — parsing, validation and ordering guarantees."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts import embed as embed_mod


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "DATA_DIR", tmp_path)
    return tmp_path


def test_reads_meta_entries_shape(data_dir):
    _write(data_dir, "faq.json", {"meta": {"source": "faq"}, "entries": [{"id": "a", "text": "A"}]})
    records, per_file = embed_mod.load_records()
    assert [r["id"] for r in records] == ["a"]
    assert records[0]["source"] == "faq"       # from meta, not the filename
    assert per_file == {"faq.json": 1}


def test_reads_bare_list_for_backwards_compat(data_dir):
    _write(data_dir, "legacy.json", [{"id": "a", "text": "A"}])
    records, _ = embed_mod.load_records()
    assert records[0]["source"] == "legacy"    # falls back to the file stem


def test_globs_every_file(data_dir):
    for name in ("self.json", "faq.json", "projects.json"):
        _write(data_dir, name, {"meta": {}, "entries": [{"id": name, "text": "t"}]})
    records, per_file = embed_mod.load_records()
    assert len(records) == 3 and len(per_file) == 3


def test_malformed_entries_skipped_not_fatal(data_dir):
    entries = [{"id": "ok", "text": "T"}, {"text": "no id"}, {"id": "no-text"}]
    _write(data_dir, "self.json", {"entries": entries})
    records, per_file = embed_mod.load_records()
    assert [r["id"] for r in records] == ["ok"]
    assert per_file["self.json"] == 1


def test_duplicate_ids_across_files_skipped(data_dir):
    _write(data_dir, "a.json", {"entries": [{"id": "dup", "text": "first"}]})
    _write(data_dir, "b.json", {"entries": [{"id": "dup", "text": "second"}]})
    records, _ = embed_mod.load_records()
    assert len(records) == 1 and records[0]["text"] == "first"


def test_no_files_raises(data_dir):
    with pytest.raises(RuntimeError, match="No JSON files"):
        embed_mod.load_records()


def test_all_entries_unusable_raises(data_dir):
    _write(data_dir, "self.json", {"entries": [{"text": "no id"}]})
    with pytest.raises(RuntimeError, match="No usable entries"):
        embed_mod.load_records()


def test_bad_top_level_type_raises(data_dir):
    _write(data_dir, "self.json", "just a string")
    with pytest.raises(RuntimeError, match="expected an object or a list"):
        embed_mod.load_records()


def test_dry_run_never_touches_pinecone(data_dir, monkeypatch, capsys):
    _write(data_dir, "self.json", {"entries": [{"id": "a", "title": "T", "text": "A"}]})
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros((1, 768), dtype="float32")
    monkeypatch.setattr(embed_mod, "load_embedder", lambda: embedder)

    def explode(*a, **k):  # any Pinecone use in dry-run is a bug
        raise AssertionError("Pinecone must not be constructed during a dry run")

    monkeypatch.setattr(embed_mod, "Pinecone", explode)
    assert embed_mod.run_embedding(dry_run=True) == 1
    assert "not written" in capsys.readouterr().out


def test_parse_failure_aborts_before_any_delete(data_dir, monkeypatch):
    """A malformed file must abort before the namespace is cleared."""
    _write(data_dir, "self.json", "not a list or object")
    index = MagicMock()
    pc = MagicMock()
    pc.Index.return_value = index
    monkeypatch.setattr(embed_mod, "Pinecone", lambda api_key: pc)
    monkeypatch.setattr(embed_mod, "load_embedder", lambda: MagicMock())
    with pytest.raises(RuntimeError):
        embed_mod.run_embedding()
    index.delete.assert_not_called()


def test_full_run_clears_namespace_then_upserts(data_dir, monkeypatch):
    _write(data_dir, "self.json", {"entries": [{"id": "a", "title": "T", "text": "A", "priority": 5}]})
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros((1, 768), dtype="float32")
    monkeypatch.setattr(embed_mod, "load_embedder", lambda: embedder)

    calls = []
    index = MagicMock()
    index.describe_index_stats.return_value = {"dimension": 768}
    index.delete.side_effect = lambda **k: calls.append("delete")
    index.upsert.side_effect = lambda **k: calls.append("upsert")
    pc = MagicMock()
    pc.Index.return_value = index
    monkeypatch.setattr(embed_mod, "Pinecone", lambda api_key: pc)

    assert embed_mod.run_embedding() == 1
    assert calls == ["delete", "upsert"]  # stale vectors cleared before the load
    vec = index.upsert.call_args.kwargs["vectors"][0]
    assert vec["id"] == "a" and vec["metadata"]["title"] == "T" and vec["metadata"]["priority"] == 5


def test_dimension_mismatch_refuses_to_write(data_dir, monkeypatch):
    _write(data_dir, "self.json", {"entries": [{"id": "a", "text": "A"}]})
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros((1, 768), dtype="float32")
    monkeypatch.setattr(embed_mod, "load_embedder", lambda: embedder)
    index = MagicMock()
    index.describe_index_stats.return_value = {"dimension": 384}
    pc = MagicMock()
    pc.Index.return_value = index
    monkeypatch.setattr(embed_mod, "Pinecone", lambda api_key: pc)
    with pytest.raises(RuntimeError, match="384-d"):
        embed_mod.run_embedding()
    index.delete.assert_not_called()
