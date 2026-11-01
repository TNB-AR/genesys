"""space_memories, memory_detail, space_tags and the show_service_names setting."""
from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import WORLD_ORG, SpaceWorld, principal

from papez import server as srv
from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, PrincipalType, SpaceRole
from papez.models.node import MemoryNode
from papez.models.space import Space, SpaceMember
from papez.storage.memory import space_bucket_key
from papez.storage.spaces import InMemorySpaceRegistry

BASE = datetime(2026, 1, 1, tzinfo=UTC)
HIT_KEYS = {
    "id", "summary", "status", "score", "activation", "is_core", "tags", "ref_count", "space",
    "content", "refs", "on_behalf_of", "category", "created_at", "created_by",
}


async def _seed(world: SpaceWorld, count: int, *, tags: list[str] | None = None, category: str | None = None,
                start: int = 0) -> list[str]:
    """Store ``count`` memories in A, an hour apart, oldest first."""
    ids = []
    for i in range(start, start + count):
        out = await world.alice.call(
            "memory_store", f"Unique seeded memo {i}", space=world.a, tags=tags, category=category,
            created_at=(BASE + timedelta(hours=i)).isoformat(),
        )
        assert "error" not in out, out
        ids.append(out["node_id"])
    return ids


# -- space_memories ------------------------------------------------------------


async def test_hit_shape(space_world: SpaceWorld):
    w = space_world
    out = await w.alice.call("space_memories", w.a)
    assert out["next_cursor"] is None
    assert {i["id"] for i in out["items"]} == w.a_ids
    for item in out["items"]:
        assert set(item) == HIT_KEYS
        assert item["score"] is None
        assert item["space"]["slug"] == "space-a"
        assert item["created_by"] == "alice"
        assert item["ref_count"] == len(item["refs"]) == 1
        assert item["content"].startswith("Bank A")


async def test_keyset_pages_newest_first_without_gaps_or_repeats(space_world: SpaceWorld):
    w = space_world
    seeded = await _seed(w, 7)
    seen: list[dict[str, Any]] = []
    cursor = None
    pages = 0
    while True:
        out = await w.alice.call("space_memories", w.a, limit=3, cursor=cursor)
        assert len(out["items"]) <= 3
        seen.extend(out["items"])
        pages += 1
        cursor = out["next_cursor"]
        if cursor is None:
            break
    assert pages == 3  # 9 memories (7 + the 2 seeded by the world)
    ids = [i["id"] for i in seen]
    assert len(ids) == len(set(ids)) == 9
    keys = [(i["created_at"], i["id"]) for i in seen]
    assert keys == sorted(keys, reverse=True)
    # The world's two seeds are newest (now), then these in reverse.
    assert ids[2:] == list(reversed(seeded))


async def test_same_timestamp_breaks_ties_on_id(space_world: SpaceWorld):
    w = space_world
    at = BASE.isoformat()
    ids = [(await w.alice.call("memory_store", f"Tied memo {i}", space=w.a, created_at=at))["node_id"] for i in range(5)]
    got: list[str] = []
    cursor = None
    while True:
        out = await w.alice.call("space_memories", w.a, limit=2, cursor=cursor)
        got.extend(i["id"] for i in out["items"] if i["id"] in ids)
        cursor = out["next_cursor"]
        if cursor is None:
            break
    assert got == sorted(ids, reverse=True)


async def test_exact_page_has_no_next_cursor(space_world: SpaceWorld):
    w = space_world
    assert (await w.alice.call("space_memories", w.a, limit=2))["next_cursor"] is None
    assert (await w.alice.call("space_memories", w.a, limit=1))["next_cursor"] is not None


async def test_tag_filter(space_world: SpaceWorld):
    w = space_world
    tagged = await _seed(w, 3, tags=["Weekly Sync"])
    await _seed(w, 2, tags=["other"], start=10)
    out = await w.alice.call("space_memories", w.a, tags_any=["weekly sync", "absent"])
    assert {i["id"] for i in out["items"]} == set(tagged)
    bad = await w.alice.call("space_memories", w.a, tags_any=["not/allowed"])
    assert bad["code"] == "invalid_tag"


async def test_category_filter_pages_through_storage(space_world: SpaceWorld):
    w = space_world
    notes = await _seed(w, 5, category="note")
    await _seed(w, 250, category="doc", start=100)  # newer than every note, and more than one batch
    out = await w.alice.call("space_memories", w.a, category="note", limit=3)
    assert [i["id"] for i in out["items"]] == list(reversed(notes))[:3]
    assert all(i["category"] == "note" for i in out["items"])
    assert out["next_cursor"] is not None
    rest = await w.alice.call("space_memories", w.a, category="note", limit=3, cursor=out["next_cursor"])
    assert [i["id"] for i in rest["items"]] == list(reversed(notes))[3:]
    assert rest["next_cursor"] is None


async def test_superseded_excluded_unless_asked(space_world: SpaceWorld):
    w = space_world
    old = (await w.alice.call("memory_store", "Kickoff is Monday", space=w.a))["node_id"]
    new = (await w.alice.call("memory_amend", old, "Kickoff is Tuesday"))["node_id"]
    plain = await w.alice.call("space_memories", w.a, limit=100)
    assert new in {i["id"] for i in plain["items"]}
    assert old not in {i["id"] for i in plain["items"]}
    everything = await w.alice.call("space_memories", w.a, limit=100, include_superseded=True)
    by_id = {i["id"]: i for i in everything["items"]}
    assert by_id[old]["superseded_by"] == new
    assert "superseded_by" not in by_id[new]
    assert "superseded_by" not in next(i for i in plain["items"] if i["id"] == new)


async def test_scan_cap_returns_a_cursor_at_the_last_row_read(space_world: SpaceWorld):
    w = space_world
    roles = await w.alice.roles()
    note_at = BASE - timedelta(days=30)
    with principal("alice", orgs=[WORLD_ORG], roles=roles):
        # The match sits behind more rows than one call may read.
        note = MemoryNode(content_summary="old note", category="note", space_id=w.a, created_at=note_at,
                          original_user_id="alice")
        await w.graph.create_node(note)
        for i in range(2100):
            await w.graph.create_node(MemoryNode(
                content_summary=f"row {i}", category="doc", space_id=w.a, created_at=BASE + timedelta(minutes=i),
                original_user_id="alice",
            ))
    first = await w.alice.call("space_memories", w.a, category="note")
    assert first["items"] == []
    assert first["next_cursor"] is not None
    second = await w.alice.call("space_memories", w.a, category="note", cursor=first["next_cursor"])
    assert [i["id"] for i in second["items"]] == [str(note.id)]
    assert second["next_cursor"] is None


async def test_scan_cap_keeps_matches_found_before_it(space_world: SpaceWorld):
    w = space_world
    roles = await w.alice.roles()
    with principal("alice", orgs=[WORLD_ORG], roles=roles):
        newest = MemoryNode(content_summary="newest note", category="note", space_id=w.a,
                            created_at=BASE + timedelta(days=30), original_user_id="alice")
        await w.graph.create_node(newest)
        for i in range(2100):
            await w.graph.create_node(MemoryNode(
                content_summary=f"row {i}", category="doc", space_id=w.a, created_at=BASE + timedelta(minutes=i),
                original_user_id="alice",
            ))
    out = await w.alice.call("space_memories", w.a, category="note", limit=5)
    assert [i["id"] for i in out["items"]] == [str(newest.id)]
    assert out["next_cursor"] is not None


def _b64(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


@pytest.mark.parametrize("kwargs", [
    {"sort": "oldest"}, {"sort": None},
    {"limit": 0}, {"limit": 101}, {"limit": "20"}, {"limit": True}, {"limit": None},
    {"cursor": ""}, {"cursor": "not base64!"}, {"cursor": 7}, {"cursor": _b64("no-separator")},
    {"cursor": _b64("not-a-date|2b1f0e2e-0000-4000-8000-000000000000")},
    {"cursor": _b64("2026-01-01T00:00:00+00:00|not-a-uuid")},
    {"cursor": "x" * 300},
    {"category": 7},
])
async def test_bad_arguments(space_world: SpaceWorld, kwargs: dict[str, Any]):
    w = space_world
    out = await w.alice.call("space_memories", w.a, **kwargs)
    assert out["code"] == "invalid_arguments"
    assert set(out) == {"error", "code"}


async def test_non_member_gets_space_not_found_identical_to_missing(space_world: SpaceWorld):
    w = space_world
    for call in ("space_memories", "space_tags"):
        hidden = await w.bob.call(call, w.a)
        missing = await w.bob.call(call, "spc_does_not_exist")
        empty = await w.bob.call(call, "")
        assert hidden["code"] == "space_not_found"
        assert json.dumps(hidden) == json.dumps(missing) == json.dumps(empty)


async def test_reader_and_archived_space_can_read(space_world: SpaceWorld):
    w = space_world
    await w.registry.add_member(SpaceMember(
        space_id=w.a, principal_id="rita", principal_type=PrincipalType.USER, role=SpaceRole.READER, added_by="alice",
    ))
    rita = w.actor("rita")
    assert (await rita.call("space_memories", w.a))["items"]
    assert (await rita.call("space_tags", w.a))["tags"]
    await w.alice.call("space_archive", w.a)
    assert (await rita.call("space_memories", w.a))["items"]
    assert (await rita.call("space_tags", w.a))["tags"]
    assert "error" not in await rita.call("memory_detail", w.ids["a_q1"])


async def test_audit_events(space_world: SpaceWorld):
    w = space_world
    w.audit.events.clear()
    await w.alice.call("space_memories", w.a, limit=1)
    [event] = w.audit.events
    assert event["action"] == "memory.list"
    assert event["detail"] == {"results": 1}
    assert len(event["node_ids"]) == 1
    w.audit.events.clear()
    await w.alice.call("space_tags", w.a)
    [event] = w.audit.events
    assert event["action"] == "space.tags"
    assert event["detail"] == {"results": 2}
    w.audit.events.clear()
    await w.alice.call("memory_detail", w.ids["a_q1"])
    [event] = w.audit.events
    assert event["action"] == "memory.explain"
    assert event["node_ids"][0] == w.ids["a_q1"]


# -- memory_detail -------------------------------------------------------------


async def test_history_chain_newest_first(space_world: SpaceWorld):
    w = space_world
    v1 = (await w.alice.call("memory_store", "Budget is 10", space=w.a, created_at=BASE.isoformat()))["node_id"]
    v2 = (await w.alice.call("memory_amend", v1, "Budget is 20"))["node_id"]
    v3 = (await w.alice.call("memory_amend", v2, "Budget is 30"))["node_id"]
    newest = await w.alice.call("memory_detail", v3)
    assert newest["memory"]["id"] == v3
    assert newest["memory"]["score"] is None
    assert "superseded_by" not in newest["memory"]
    assert [h["id"] for h in newest["history"]] == [v2, v1]
    assert newest["history"][1]["content"] == "Budget is 10"
    assert {"created_at", "created_by", "on_behalf_of"} <= set(newest["history"][0])

    middle = await w.alice.call("memory_detail", v2)
    assert middle["memory"]["superseded_by"] == v3
    assert [h["id"] for h in middle["history"]] == [v1]

    assert (await w.alice.call("memory_detail", v1))["history"] == []


async def test_history_cap_and_cycles(space_world: SpaceWorld):
    w = space_world
    current = (await w.alice.call("memory_store", "Version 0", space=w.a))["node_id"]
    first = current
    for i in range(1, 55):
        current = (await w.alice.call("memory_amend", current, f"Version {i}"))["node_id"]
    out = await w.alice.call("memory_detail", current)
    assert len(out["history"]) == 50
    assert first not in {h["id"] for h in out["history"]}
    # A corrupt cycle must terminate.
    with principal("alice", orgs=[WORLD_ORG], roles=await w.alice.roles()):
        await w.graph.create_edge(MemoryEdge(source_id=first, target_id=current, type=EdgeType.SUPERSEDES))
    assert (await w.alice.call("memory_detail", current))["history"]


async def test_related_either_direction_deduped_and_capped(space_world: SpaceWorld):
    w = space_world
    roles = await w.alice.roles()
    a, ctx = w.ids["a_q1"], w.ids["a_ctx"]
    out = await w.alice.call("memory_detail", a)
    assert out["related"] == [{"id": ctx, "type": "caused_by", "preview": "Bank A pilot budget approved for the payments team"}]
    assert (await w.alice.call("memory_detail", ctx))["related"][0]["id"] == a

    # A second edge to the same memory stays one row; a supersedes edge is history, not related.
    with principal("alice", orgs=[WORLD_ORG], roles=roles):
        await w.graph.create_edge(MemoryEdge(source_id=a, target_id=ctx, type=EdgeType.SUPPORTS))
    out = await w.alice.call("memory_detail", a)
    assert [r["id"] for r in out["related"]] == [ctx]

    fresh = (await w.alice.call("memory_store", "Hub memo", space=w.a))["node_id"]
    new = (await w.alice.call("memory_amend", fresh, "Hub memo two"))["node_id"]
    assert fresh not in {r["id"] for r in (await w.alice.call("memory_detail", new))["related"]}
    with principal("alice", orgs=[WORLD_ORG], roles=roles):
        for i in range(25):
            leaf = MemoryNode(content_summary=f"leaf {i}", space_id=w.a, original_user_id="alice")
            await w.graph.create_node(leaf)
            await w.graph.create_edge(MemoryEdge(source_id=new, target_id=leaf.id, type=EdgeType.RELATED_TO))
    assert len((await w.alice.call("memory_detail", new))["related"]) == 20


async def test_include_limits_the_sections(space_world: SpaceWorld):
    w = space_world
    old = (await w.alice.call("memory_store", "Plan A", space=w.a))["node_id"]
    new = (await w.alice.call("memory_amend", old, "Plan B"))["node_id"]
    assert (await w.alice.call("memory_detail", new, include=("related",)))["history"] == []
    assert (await w.alice.call("memory_detail", w.ids["a_q1"], include=("history",)))["related"] == []


async def test_not_found_is_identical_for_missing_malformed_and_invisible(space_world: SpaceWorld):
    w = space_world
    invisible = w.ids["b_q2"]
    personal = w.ids["p_notes"]
    missing = "00000000-0000-0000-0000-000000000000"
    malformed = "not-a-uuid"
    bodies = {}
    for name, node_id, actor in (
        ("invisible_space", invisible, w.alice),
        ("invisible_personal", personal, w.bob),
        ("missing", missing, w.alice),
        ("malformed", malformed, w.alice),
    ):
        out = await actor.call("memory_detail", node_id)
        assert set(out) == {"error", "code", "node_id"}
        assert out["node_id"] == node_id
        bodies[name] = json.dumps({**out, "node_id": "X"})
    assert len(set(bodies.values())) == 1
    for bad in (None, 7, ""):
        assert (await w.alice.call("memory_detail", bad))["code"] == "not_found"


async def test_detail_never_links_outside_the_scope(space_world: SpaceWorld):
    w = space_world
    # Corrupt edges out of the space: storage hides them, so detail can't link to them.
    bucket = space_bucket_key(w.a)
    for label in ("b_q2", "p_notes"):
        edge = MemoryEdge(source_id=w.ids["a_ctx"], target_id=w.ids[label], type=EdgeType.RELATED_TO)
        w.graph._user_edges[bucket].append(edge)
        w.graph._index_edge(bucket, edge)
    out = await w.svc.call("memory_detail", w.ids["a_ctx"])
    assert not {r["id"] for r in out["related"]} & (w.b_ids | w.personal_ids)
    assert out["related"] == [{
        "id": w.ids["a_q1"], "type": "caused_by", "preview": "Bank A pilot starts in Q1",
    }]


async def test_personal_memory_detail_for_its_owner(space_world: SpaceWorld):
    w = space_world
    out = await w.alice.call("memory_detail", w.ids["p_more"])
    assert out["memory"]["id"] == w.ids["p_more"]
    assert out["memory"]["space"] is None
    assert [r["id"] for r in out["related"]] == [w.ids["p_notes"]]
    w.audit.events.clear()
    await w.alice.call("memory_detail", w.ids["p_more"])
    assert w.audit.events == []


# -- space_tags ----------------------------------------------------------------


async def test_tag_counts_and_categories(space_world: SpaceWorld):
    w = space_world
    await _seed(w, 3, tags=["beta"], category="note")
    await _seed(w, 3, tags=["alpha"], category="doc", start=10)
    await _seed(w, 1, tags=["alpha", "gamma"], category="note", start=20)
    old = (await w.alice.call("memory_store", "Retired tag memo", space=w.a, tags=["stale"], category="stale"))["node_id"]
    await w.alice.call("memory_amend", old, "Memo without the stale tag", tags=[])
    out = await w.alice.call("space_tags", w.a)
    counts = {t["tag"]: t["count"] for t in out["tags"]}
    assert counts == {"alpha": 4, "bank-a": 2, "beta": 3, "gamma": 1, "pilot": 2}
    assert [t["tag"] for t in out["tags"]] == ["alpha", "beta", "bank-a", "pilot", "gamma"]
    # The amendment keeps the category, the superseded row doesn't count twice.
    assert out["categories"] == [
        {"category": "note", "count": 4}, {"category": "doc", "count": 3}, {"category": "stale", "count": 1},
    ]
    limited = await w.alice.call("space_tags", w.a, limit=2)
    assert [t["tag"] for t in limited["tags"]] == ["alpha", "beta"]


@pytest.mark.parametrize("limit", [0, 101, "12", True])
async def test_tags_bad_limit(space_world: SpaceWorld, limit: Any):
    w = space_world
    assert (await w.alice.call("space_tags", w.a, limit=limit))["code"] == "invalid_arguments"


# -- storage -------------------------------------------------------------------


async def test_storage_list_respects_scope_before_and_limit(space_world: SpaceWorld):
    w = space_world
    seeded = await _seed(w, 3)
    roles = await w.alice.roles()
    with principal("alice", orgs=[WORLD_ORG], roles=roles):
        rows = await w.graph.list_space_nodes(w.a, limit=100)
        assert await w.graph.list_space_nodes(w.a, limit=0) == []
        assert await w.graph.list_space_nodes(w.b) == []
        assert await w.graph.space_tag_counts(w.b) == []
        pivot = next(n for n in rows if str(n.id) == seeded[1])
        older = await w.graph.list_space_nodes(w.a, before=(pivot.created_at, str(pivot.id)))
        assert [str(n.id) for n in older] == [seeded[0]]


# -- show_service_names --------------------------------------------------------


async def test_show_service_names_on_create_update_and_defaults(space_world: SpaceWorld):
    w = space_world
    default = await w.alice.call("space_create", "Defaulted")
    assert default["space"]["show_service_names"] is True
    off = await w.alice.call("space_create", "Quiet", show_service_names=False)
    assert off["space"]["show_service_names"] is False
    sid = off["space"]["id"]
    assert (await w.alice.call("space_get", sid))["space"]["show_service_names"] is False
    on = await w.alice.call("space_update", sid, show_service_names=True)
    assert on["space"]["show_service_names"] is True
    untouched = await w.alice.call("space_update", sid, name="Quieter")
    assert untouched["space"]["show_service_names"] is True
    assert (await w.alice.call("space_update", sid, show_service_names="no"))["code"] == "invalid_arguments"
    assert (await w.alice.call("space_create", "Bad", show_service_names="no"))["code"] == "invalid_arguments"


def test_show_service_names_is_registered_for_mcp():
    for name in ("space_create", "space_update"):
        _, _, optional = srv._TOOL_DISPATCH[name]
        assert "show_service_names" in optional
        definition = next(d for d in srv._TOOL_DEFS if d["name"] == name)
        assert "show_service_names" in definition["inputSchema"]["properties"]


async def test_old_sidecar_loads_with_the_default_and_new_value_persists(tmp_path: Path):
    path = tmp_path / "state.json.spaces.json"
    space = Space(org_id="org_r", name="Old", slug="old", created_by="alice")
    dump = space.model_dump(mode="json")
    del dump["show_service_names"]
    path.write_text(json.dumps({"version": 1, "spaces": {space.id: dump}, "members": []}))
    registry = InMemorySpaceRegistry(path)
    await registry.initialize()
    loaded = await registry.get_space(space.id)
    assert loaded is not None and loaded.show_service_names is True

    await registry.update_space(space.id, {"show_service_names": False})
    again = InMemorySpaceRegistry(path)
    await again.initialize()
    reloaded = await again.get_space(space.id)
    assert reloaded is not None and reloaded.show_service_names is False
