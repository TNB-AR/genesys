"""R1 model-level tests: Ref, normalize_tags, validate_refs (spec 5.2, 5.3).

Tool-level R1 tests (store/recall/search/amend round trips) are added by the
workstreams that implement them.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from papez.engine import config
from papez.models import MemoryNode, Ref
from papez.models.ref import REF_TYPE_PATTERN, ValidationFailure, normalize_tags, validate_refs


def _tags(tags, *, max_tag_length=50, max_tags=20):
    return normalize_tags(tags, max_tag_length=max_tag_length, max_tags=max_tags)


def _refs(refs, *, max_refs=20):
    return validate_refs(refs, max_refs=max_refs)


# -- Tags ---------------------------------------------------------------------


def test_tag_normalization():
    """Every example in spec 5.3."""
    assert _tags([" Manual Payments "]) == ["manual-payments"]
    assert _tags(["Bank A"]) == ["bank-a"]
    with pytest.raises(ValidationFailure) as exc:
        _tags(["q2/2026"])
    assert exc.value.code == "invalid_tag"
    assert exc.value.message == "tag 'q2/2026' contains invalid characters"
    assert exc.value.as_error() == {"error": "tag 'q2/2026' contains invalid characters", "code": "invalid_tag"}


def test_tag_trim_and_lowercase():
    assert _tags(["  URGENT\t"]) == ["urgent"]


def test_tag_space_runs_collapse_to_single_dash():
    assert _tags(["a    b   c"]) == ["a-b-c"]


@pytest.mark.parametrize("tag", ["a-b", "a_b", "a.b", "a:b", "x9", "0", "client:acme.q2_2026-final"])
def test_tag_allowed_characters(tag):
    assert _tags([tag]) == [tag]


@pytest.mark.parametrize("tag", ["q2/2026", "a,b", "café", "a#b", "a+b", "(x)", "a'b", "a\"b", "emoji-☃"])
def test_tag_disallowed_characters_rejected(tag):
    with pytest.raises(ValidationFailure) as exc:
        _tags([tag])
    assert exc.value.code == "invalid_tag"


def test_tag_empty_dropped():
    assert _tags(["", "   ", "ok", "\t"]) == ["ok"]


def test_tag_duplicates_removed_keeping_first_order():
    assert _tags(["b", "A", "a", " B ", "c", "a"]) == ["b", "a", "c"]


def test_tag_length_limit():
    assert _tags(["x" * 50]) == ["x" * 50]
    with pytest.raises(ValidationFailure) as exc:
        _tags(["x" * 51])
    assert exc.value.code == "invalid_tag"
    # limit is the explicit argument, not a module constant
    with pytest.raises(ValidationFailure):
        _tags(["abcd"], max_tag_length=3)


def test_tag_length_measured_after_normalization():
    # 52 raw chars, 50 after trimming
    assert _tags([" " + "x" * 50 + " "]) == ["x" * 50]


def test_tag_count_limit():
    assert len(_tags([f"t{i}" for i in range(20)])) == 20
    with pytest.raises(ValidationFailure) as exc:
        _tags([f"t{i}" for i in range(21)])
    assert exc.value.code == "too_many_tags"


def test_tag_count_measured_after_dedupe_and_drop():
    tags = [f"t{i}" for i in range(20)] + ["T0", " t1 ", "", "  "]
    assert len(_tags(tags)) == 20


def test_tag_none_and_empty_list():
    assert _tags(None) == []
    assert _tags([]) == []


@pytest.mark.parametrize("bad", ["single-string", 5, [1], [None], [["nested"]]])
def test_tag_non_string_input_rejected(bad):
    with pytest.raises(ValidationFailure) as exc:
        _tags(bad if isinstance(bad, list) or isinstance(bad, str) else [bad])
    assert exc.value.code == "invalid_tag"


def test_validation_failure_is_value_error():
    err = ValidationFailure("invalid_tag", "boom")
    assert isinstance(err, ValueError)
    assert str(err) == "boom"
    assert err.code == "invalid_tag"
    assert err.message == "boom"


# -- Refs ---------------------------------------------------------------------


def test_ref_validation():
    """Bad type pattern, empty id, non-http(s) url and 21 refs are all rejected."""
    for bad in (
        {"type": "Document", "id": "1"},          # uppercase
        {"type": "-doc", "id": "1"},              # must start with [a-z0-9]
        {"type": "doc/x", "id": "1"},             # '/' not allowed
        {"type": "a" * 65, "id": "1"},            # too long
        {"type": "", "id": "1"},
        {"type": "doc", "id": ""},                # empty id
        {"type": "doc", "id": "x" * 257},         # id too long
        {"type": "doc", "id": "1", "url": "ftp://x"},
        {"type": "doc", "id": "1", "url": "javascript:alert(1)"},
        {"type": "doc", "id": "1", "url": "https://x/" + "a" * 2048},
        {"type": "doc", "id": "1", "label": "l" * 201},
        {"id": "1"},                              # missing type
        {"type": "doc"},                          # missing id
    ):
        with pytest.raises(ValidationFailure) as exc:
            _refs([bad])
        assert exc.value.code == "invalid_ref", bad

    with pytest.raises(ValidationFailure) as exc:
        _refs([{"type": "doc", "id": str(i)} for i in range(21)])
    assert exc.value.code == "too_many_refs"
    assert len(_refs([{"type": "doc", "id": str(i)} for i in range(20)])) == 20


def test_ref_valid_shapes():
    out = _refs([
        {"type": "document", "id": "D-1", "url": "https://example.com/d1", "label": "Q2 plan.pdf"},
        {"type": "crm.contact", "id": "42"},
        {"type": "meeting", "id": "m1", "url": "http://intranet/m1"},
        Ref(type="ticket", id="T-9"),
    ])
    assert [(r.type, r.id) for r in out] == [("document", "D-1"), ("crm.contact", "42"), ("meeting", "m1"), ("ticket", "T-9")]
    assert out[0].url == "https://example.com/d1"
    assert out[0].label == "Q2 plan.pdf"
    assert out[1].url is None and out[1].label is None


def test_duplicate_refs_removed():
    out = _refs([
        {"type": "doc", "id": "1", "label": "first"},
        {"type": "doc", "id": "2"},
        {"type": "doc", "id": "1", "label": "second"},
        {"type": "meeting", "id": "1"},
    ])
    assert [(r.type, r.id) for r in out] == [("doc", "1"), ("doc", "2"), ("meeting", "1")]
    assert out[0].label == "first"


def test_ref_count_measured_after_dedupe():
    refs = [{"type": "doc", "id": str(i)} for i in range(20)] + [{"type": "doc", "id": "0"}]
    assert len(_refs(refs)) == 20


def test_ref_limit_is_explicit_argument():
    with pytest.raises(ValidationFailure) as exc:
        _refs([{"type": "doc", "id": "1"}, {"type": "doc", "id": "2"}], max_refs=1)
    assert exc.value.code == "too_many_refs"


def test_refs_none_and_empty():
    assert _refs(None) == []
    assert _refs([]) == []


@pytest.mark.parametrize("bad", ["doc:1", {"type": "doc", "id": "1"}, [5], [None], [["doc", "1"]]])
def test_refs_non_object_input_rejected(bad):
    with pytest.raises(ValidationFailure) as exc:
        _refs(bad)
    assert exc.value.code == "invalid_ref"


def test_ref_model_direct():
    assert REF_TYPE_PATTERN == r"^[a-z0-9][a-z0-9_.-]{0,63}$"
    with pytest.raises(ValidationError):
        Ref(type="doc", id="1", url="mailto:a@b")
    assert Ref(type="a" * 64, id="1").type == "a" * 64


# -- Config + node fields -----------------------------------------------------


def test_limit_defaults():
    assert config.MAX_REFS_PER_MEMORY == 20
    assert config.MAX_TAGS_PER_MEMORY == 20
    assert config.MAX_TAG_LENGTH == 50


def test_memory_node_refs_and_tags_default_empty():
    node = MemoryNode(content_summary="x")
    assert node.refs == [] and node.tags == []
    other = MemoryNode(content_summary="y")
    other.tags.append("t")
    assert node.tags == []  # no shared mutable default


def test_memory_node_refs_round_trip():
    node = MemoryNode(content_summary="x", refs=[{"type": "doc", "id": "1", "url": "https://e/1"}], tags=["a"])
    assert isinstance(node.refs[0], Ref)
    dumped = node.model_dump(mode="json")
    assert dumped["refs"] == [{"type": "doc", "id": "1", "url": "https://e/1", "label": None}]
    assert dumped["tags"] == ["a"]
    assert MemoryNode(**dumped).refs == node.refs
