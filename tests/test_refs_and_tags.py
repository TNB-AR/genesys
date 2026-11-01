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


# -- Tool level (WS-L5): store/recall/search/explain/amend round trips ---------------

from conftest import TokenEmbedder, principal  # noqa: E402

from papez.mcp.tools import MCPToolHandler  # noqa: E402
from papez.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider  # noqa: E402

_REFS = [
    {"type": "document", "id": "doc_8812", "label": "Manual Payments Mail Matrix.pdf",
     "url": "https://docs.example.com/d/doc_8812"},
    {"type": "meeting", "id": "mtg_2026_10_14", "label": "Payments process review, Oct 14"},
]


def _handler() -> MCPToolHandler:
    return MCPToolHandler(graph=InMemoryGraphProvider(), embeddings=TokenEmbedder(), cache=InMemoryCacheProvider())


async def test_store_and_recall_returns_refs_and_tags():
    h = _handler()
    with principal("u1"):
        stored = await h.memory_store(
            "Manual payments over 10K ship overnight via FedEx Priority",
            refs=_REFS, tags=["Manual Payments", "shipping", "Bank A"],
        )
        nid = stored["node_id"]
        full = await h.memory_recall("manual payments ship overnight")
        concise = await h.memory_recall("manual payments ship overnight", verbosity="concise")
        search = await h.memory_search("manual payments ship overnight")
        explain = await h.memory_explain(nid)
    hit = next(r for r in full["results"] if r["id"] == nid)
    assert hit["tags"] == ["manual-payments", "shipping", "bank-a"]
    assert hit["refs"] == _REFS
    assert hit["space"] is None
    chit = next(r for r in concise["results"] if r["id"] == nid)
    assert chit["tags"] == ["manual-payments", "shipping", "bank-a"]
    assert chit["ref_count"] == 2 and "refs" not in chit and chit["space"] is None
    shit = next(r for r in search["results"] if r["id"] == nid)
    assert shit["refs"] == _REFS and shit["tags"] == hit["tags"]
    assert explain["refs"] == _REFS and explain["tags"] == hit["tags"] and explain["space"] is None


@pytest.mark.parametrize("bad,code", [
    ({"tags": ["q2/2026"]}, "invalid_tag"),
    ({"tags": [f"t{i}" for i in range(21)]}, "too_many_tags"),
    ({"refs": [{"type": "Bad Type", "id": "x"}]}, "invalid_ref"),
    ({"refs": [{"type": "url", "id": "x", "url": "ftp://x"}]}, "invalid_ref"),
    ({"refs": [{"type": "doc", "id": f"d{i}"} for i in range(21)]}, "too_many_refs"),
])
async def test_store_rejects_bad_refs_and_tags_and_creates_nothing(bad, code):
    h = _handler()
    with principal("u1"):
        result = await h.memory_store("rejected", **bad)
    assert result["code"] == code and "error" in result
    assert not h.graph._user_nodes.get("u1")


async def test_duplicate_refs_stored_once_through_tool():
    h = _handler()
    with principal("u1"):
        res = await h.memory_store("dupes", refs=[{"type": "doc", "id": "1"}, {"type": "doc", "id": "1", "label": "x"}])
        node = await h.graph.get_node(res["node_id"])
    assert [(r.type, r.id) for r in node.refs] == [("doc", "1")]


async def test_amend_copies_refs_and_tags():
    h = _handler()
    with principal("u1"):
        old = await h.memory_store("original fact", refs=_REFS, tags=["a", "b"])
        copied = await h.memory_amend(old["node_id"], "corrected fact")
        replaced = await h.memory_amend(copied["node_id"], "corrected again",
                                        refs=[{"type": "ticket", "id": "T-1"}], tags=["c"])
        cleared = await h.memory_amend(replaced["node_id"], "cleared", refs=[], tags=[])
        n_copied = await h.graph.get_node(copied["node_id"])
        n_replaced = await h.graph.get_node(replaced["node_id"])
        n_cleared = await h.graph.get_node(cleared["node_id"])
        bad = await h.memory_amend(cleared["node_id"], "bad", tags=["no/slash"])
    assert [r.id for r in n_copied.refs] == ["doc_8812", "mtg_2026_10_14"] and n_copied.tags == ["a", "b"]
    assert [r.id for r in n_replaced.refs] == ["T-1"] and n_replaced.tags == ["c"]
    assert n_cleared.refs == [] and n_cleared.tags == []
    assert bad["code"] == "invalid_tag"
    assert copied == {"node_id": copied["node_id"], "supersedes": old["node_id"], "status": "amended"}


async def test_search_filters():
    h = _handler()
    with principal("u1"):
        n1 = (await h.memory_store("alpha note", tags=["x", "y"], refs=[{"type": "doc", "id": "1"}]))["node_id"]
        n2 = (await h.memory_store("alpha note two", tags=["x"], refs=[{"type": "ticket", "id": "1"}]))["node_id"]
        n3 = (await h.memory_store("alpha note three", tags=["z"],
                                   refs=[{"type": "doc", "id": "2"}, {"type": "ticket", "id": "1"}]))["node_id"]

        async def ids(query: str, filters: dict) -> set[str]:
            return {r["id"] for r in (await h.memory_search(query, filters, k=10))["results"]}

        for query in ("", "alpha note"):  # enumeration path and vector path agree
            assert await ids(query, {"tags_any": ["X", "z"]}) == {n1, n2, n3}
            assert await ids(query, {"tags_any": ["y"]}) == {n1}
            assert await ids(query, {"tags_all": ["x", "y"]}) == {n1}
            assert await ids(query, {"tags_all": ["x"]}) == {n1, n2}
            assert await ids(query, {"ref_type": "doc"}) == {n1, n3}
            assert await ids(query, {"ref_id": "1"}) == {n1, n2, n3}
            # same-ref rule: n3 has a doc ref and a ref with id 1, but not doc/1
            assert await ids(query, {"ref_type": "doc", "ref_id": "1"}) == {n1}
            assert await ids(query, {"ref_type": "ticket", "ref_id": "1"}) == {n2, n3}
        bad = await h.memory_search("", {"tags_any": ["a/b"]})
    assert bad["code"] == "invalid_tag"


async def test_search_vector_path_overfetches_before_filtering():
    h = _handler()
    with principal("u1"):
        for i in range(30):
            await h.memory_store(f"common phrase entry {i}")
        tagged = (await h.memory_store("common phrase entry tagged", tags=["needle"]))["node_id"]
        res = await h.memory_search("common phrase entry", {"tags_any": ["needle"]}, k=5)
    assert [r["id"] for r in res["results"]] == [tagged]


async def test_recall_tags_filter_returns_k():
    h = _handler()
    with principal("u1"):
        tagged: set[str] = set()
        for i in range(50):
            tags = ["x"] if i % 4 == 0 and len(tagged) < 12 else ["other"]
            res = await h.memory_store(f"quarterly planning item number {i}", tags=tags)
            if tags == ["x"]:
                tagged.add(res["node_id"])
        assert len(tagged) == 12
        rec = await h.memory_recall("quarterly planning item", k=10, tags_any=["x"])
        concise = await h.memory_recall("quarterly planning item", k=10, tags_any=["X"], verbosity="concise")
        by_type = await h.memory_recall("quarterly planning item", k=10, ref_type="doc")
    assert rec["count"] == 10
    assert {r["id"] for r in rec["results"]} <= tagged
    assert all(r["tags"] == ["x"] for r in rec["results"])
    assert concise["count"] == 10 and {r["id"] for r in concise["results"]} <= tagged
    assert by_type["count"] == 0


async def test_recall_tag_filter_applies_to_pinned_core_injection():
    h = _handler()
    with principal("u1"):
        pinned = (await h.memory_store("an unrelated pinned preference", tags=["pref"]))["node_id"]
        await h.pin_memory(pinned)
        hit = (await h.memory_store("target memory about rockets", tags=["x"]))["node_id"]
        filtered = await h.memory_recall("rockets", tags_any=["x"])
        unfiltered = await h.memory_recall("rockets")
    assert {r["id"] for r in filtered["results"]} == {hit}
    assert pinned in {r["id"] for r in unfiltered["results"]}


async def test_not_found_errors_carry_code():
    h = _handler()
    missing = "00000000-0000-4000-8000-000000000000"
    with principal("u1"):
        for tool in ("memory_explain", "pin_memory", "unpin_memory", "delete_memory"):
            assert await getattr(h, tool)(missing) == {"error": "Node not found", "code": "not_found",
                                                      "node_id": missing}
        assert (await h.memory_amend(missing, "x"))["code"] == "not_found"
        assert (await h.promote_to_org(missing, "org"))["code"] == "not_found"
        assert (await h.memory_traverse(missing))["code"] == "not_found"


async def test_event_payload_carries_identity():
    from papez.storage.memory import InMemoryEventBusProvider

    bus = InMemoryEventBusProvider()
    seen: list[dict] = []

    async def capture(payload: dict) -> None:
        seen.append(payload)

    await bus.subscribe("memory.created", capture)
    h = MCPToolHandler(graph=InMemoryGraphProvider(), embeddings=None, cache=InMemoryCacheProvider(), event_bus=bus)
    with principal("u1", orgs=["org_1"]):
        res = await h.memory_store("payload check")
    import asyncio

    await asyncio.sleep(0)
    assert seen and seen[0]["node_id"] == res["node_id"]
    assert seen[0]["user_id"] == "u1" and seen[0]["org_ids"] == ["org_1"] and seen[0]["space_id"] is None
