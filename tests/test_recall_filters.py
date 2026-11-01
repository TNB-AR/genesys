"""memory_recall: category, exclude_superseded and verbosity="light"."""
from __future__ import annotations

from typing import Any

from conftest import SpaceWorld, principal

from papez import server as srv


async def _store(w: SpaceWorld, content: str, **kwargs: Any) -> str:
    out = await w.alice.call("memory_store", content, space=w.a, **kwargs)
    assert "error" not in out, out
    return out["node_id"]


async def test_category_keeps_k_results_when_enough_match(space_world: SpaceWorld):
    w = space_world
    # Better matches in the wrong category outrank every note.
    for i in range(10):
        await _store(w, f"quarterly roadmap review {i}", category="doc")
    notes = {await _store(w, f"quarterly roadmap review and other unrelated words here {i}", category="note")
             for i in range(5)}
    out = await w.alice.call("memory_recall", "quarterly roadmap review", k=5, space=w.a, category="note")
    assert {r["id"] for r in out["results"]} == notes
    assert out["count"] == 5
    assert all(r["category"] == "note" for r in out["results"])
    unfiltered = await w.alice.call("memory_recall", "quarterly roadmap review", k=5, space=w.a)
    assert not {r["id"] for r in unfiltered["results"]} & notes


async def test_exclude_superseded_keeps_k_results(space_world: SpaceWorld):
    w = space_world
    old = [await _store(w, f"quarterly roadmap review {i}") for i in range(5)]
    new = [(await w.alice.call("memory_amend", nid, f"quarterly roadmap review {i} corrected"))["node_id"]
           for i, nid in enumerate(old)]
    out = await w.alice.call("memory_recall", "quarterly roadmap review", k=5, space=w.a, exclude_superseded=True)
    ids = {r["id"] for r in out["results"]}
    assert ids == set(new)
    assert not any("superseded_by" in r for r in out["results"])
    kept = await w.alice.call("memory_recall", "quarterly roadmap review", k=10, space=w.a)
    assert ids & {r["id"] for r in kept["results"]} and {r["id"] for r in kept["results"]} & set(old)


async def test_filters_apply_to_the_core_injection_list(space_world: SpaceWorld):
    w = space_world
    wrong = await _store(w, "pinned guidance on holidays", category="doc")
    old = await _store(w, "pinned guidance on travel", category="note")
    new = (await w.alice.call("memory_amend", old, "pinned guidance on travel, revised"))["node_id"]
    for nid in (wrong, old, new):
        assert (await w.alice.call("pin_memory", nid))["status"] == "pinned"

    plain = await w.alice.call("memory_recall", "zzzzz", k=50, space=w.a, min_results=0)
    assert {wrong, old, new} <= {r["id"] for r in plain["results"]}

    by_category = await w.alice.call("memory_recall", "zzzzz", k=50, space=w.a, min_results=0, category="note")
    assert {r["id"] for r in by_category["results"]} == {old, new}

    current = await w.alice.call("memory_recall", "zzzzz", k=50, space=w.a, min_results=0, exclude_superseded=True)
    assert {wrong, new} <= {r["id"] for r in current["results"]}
    assert old not in {r["id"] for r in current["results"]}


async def test_filters_apply_to_the_floor_backfill(space_world: SpaceWorld):
    w = space_world
    note = await _store(w, "aardvark", category="note")
    doc = await _store(w, "zebra", category="doc")
    # No keyword match and no similar vector: only the backfill can return these.
    roles = await w.alice.roles()
    with principal("alice", orgs=["org_r"], roles=roles):
        plain = await w.handler.memory_recall("zzzzz", k=50, min_results=50, space=w.a)
        assert {note, doc} <= {r["id"] for r in plain["results"] if r.get("low_confidence")}
        out = await w.handler.memory_recall("zzzzz", k=50, min_results=50, space=w.a, category="note")
    assert [r["id"] for r in out["results"]] == [note]


async def test_light_skips_chain_queries_and_returns_light_hits(space_world: SpaceWorld, monkeypatch: Any):
    w = space_world
    calls: list[str] = []
    batch = w.graph.get_causal_chains_batch
    single = w.graph.get_causal_chain

    async def spy_batch(*args: Any, **kwargs: Any) -> Any:
        calls.append("batch")
        return await batch(*args, **kwargs)

    async def spy_single(*args: Any, **kwargs: Any) -> Any:
        calls.append("single")
        return await single(*args, **kwargs)

    monkeypatch.setattr(w.graph, "get_causal_chains_batch", spy_batch)
    monkeypatch.setattr(w.graph, "get_causal_chain", spy_single)

    light = await w.alice.call("memory_recall", "Bank A pilot budget", space=w.a, verbosity="light")
    assert calls == []
    hit = next(r for r in light["results"] if r["id"] == w.ids["a_ctx"])
    assert hit["content"].startswith("Bank A")
    assert hit["causal_basis"] == []
    assert "causal_chain" not in hit
    assert {"category", "on_behalf_of", "created_by", "refs", "tags", "space"} <= set(hit)

    full = await w.alice.call("memory_recall", "Bank A pilot budget", space=w.a)
    assert calls
    assert next(r for r in full["results"] if r["id"] == w.ids["a_ctx"])["causal_basis"]


async def test_recall_arguments_are_checked(space_world: SpaceWorld):
    w = space_world
    assert (await w.alice.call("memory_recall", "x", category=7))["code"] == "invalid_arguments"
    assert (await w.alice.call("memory_recall", "x", exclude_superseded="yes"))["code"] == "invalid_arguments"


def test_new_recall_parameters_are_registered_for_mcp():
    _, _, optional = srv._TOOL_DISPATCH["memory_recall"]
    assert optional["category"] is None and optional["exclude_superseded"] is False
    props = next(d for d in srv._TOOL_DEFS if d["name"] == "memory_recall")["inputSchema"]["properties"]
    assert {"category", "exclude_superseded"} <= set(props)
    assert "light" in props["verbosity"]["enum"]
