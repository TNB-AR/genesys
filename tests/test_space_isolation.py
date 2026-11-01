"""Spec 13.2, tool half: Mind Space isolation and scoped access through MCPToolHandler.

Fixture (``space_world`` in conftest): org ``org_r``; users alice and bob; service
svc_a; space A (alice owner, svc_a writer) and space B (bob owner), seeded with
near-identical content, shared tags, refs and entity names, plus alice's
personal memory. The storage half of the "every read path" matrix lives in
tests/test_space_isolation_storage.py (WS-L2); here every read path is driven
both through storage (with the actor's membership scope) and through the tools.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import WORLD_ORG, SpaceWorld, build_space_world, principal

from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, PrincipalType, SpaceRole
from papez.models.space import Space, SpaceMember


def _walk_strings(obj: Any) -> set[str]:
    """Every string anywhere in a (possibly pydantic) result."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    out: set[str] = set()
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out |= _walk_strings(k) | _walk_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            out |= _walk_strings(v)
    return out


async def _add(world: SpaceWorld, space_id: str, uid: str, role: SpaceRole,
               ptype: PrincipalType = PrincipalType.USER) -> None:
    await world.registry.add_member(SpaceMember(
        space_id=space_id, principal_id=uid, principal_type=ptype, role=role, added_by="test",
    ))


async def _values(coro: Any) -> list[Any]:
    return list((await coro).values())


def _dump(result: Any) -> str:
    return json.dumps(result, sort_keys=True)


# -- every read path ------------------------------------------------------------

# (name, coroutine factory(world, actor, own_id)) -- own_id is a node the actor can see.
STORAGE_READS = [
    ("get_node", lambda w, a, own, other: a.storage("get_node", other)),
    ("get_nodes_by_status", lambda w, a, own, other: a.storage("get_nodes_by_status", MemoryStatus.ACTIVE, 1000)),
    ("get_edges", lambda w, a, own, other: a.storage("get_edges", own, "both")),
    ("get_edges_other", lambda w, a, own, other: a.storage("get_edges", other, "both")),
    ("get_all_edges", lambda w, a, own, other: a.storage("get_all_edges")),
    ("get_all_edges_ids", lambda w, a, own, other: a.storage("get_all_edges", list(w.ids.values()))),
    ("traverse", lambda w, a, own, other: a.storage("traverse", own, 10)),
    ("traverse_other", lambda w, a, own, other: a.storage("traverse", other, 10)),
    ("get_connecting_edges", lambda w, a, own, other: a.storage("get_connecting_edges", list(w.ids.values()))),
    ("get_causal_chain_up", lambda w, a, own, other: a.storage("get_causal_chain", own, "upstream")),
    ("get_causal_chain_down", lambda w, a, own, other: a.storage("get_causal_chain", own, "downstream")),
    # keys echo the requested ids, so only the chains themselves are checked
    ("get_causal_chains_batch", lambda w, a, own, other: _values(a.storage(
        "get_causal_chains_batch", list(w.ids.values()), "upstream"))),
    ("vector_search", None),  # needs an embedding; built in the test
    ("keyword_search", lambda w, a, own, other: a.storage("keyword_search", "bank pilot payments", k=100)),
    ("find_nodes_by_ref", lambda w, a, own, other: a.storage("find_nodes_by_ref", "document", "doc_bank_a")),
    ("get_orphans", lambda w, a, own, other: a.storage("get_orphans")),
]

TOOL_READS = [
    ("memory_recall", lambda w, a, own, other: a.call("memory_recall", "Bank A pilot starts", k=50)),
    ("memory_recall_concise", lambda w, a, own, other: a.call(
        "memory_recall", "Bank A pilot", k=50, verbosity="concise")),
    ("memory_recall_tags", lambda w, a, own, other: a.call("memory_recall", "pilot", k=50, tags_any=["bank-a"])),
    ("memory_search", lambda w, a, own, other: a.call("memory_search", "Bank A pilot starts", k=50)),
    ("memory_search_enumerate", lambda w, a, own, other: a.call("memory_search", "", k=100)),
    ("memory_search_ref", lambda w, a, own, other: a.call(
        "memory_search", "", {"ref_type": "document", "ref_id": "doc_bank_a"}, k=100)),
    ("memory_traverse", lambda w, a, own, other: a.call("memory_traverse", own, depth=10)),
    ("memory_traverse_other", lambda w, a, own, other: a.call("memory_traverse", other, depth=10)),
    ("memory_explain", lambda w, a, own, other: a.call("memory_explain", own)),
    ("memory_explain_other", lambda w, a, own, other: a.call("memory_explain", other)),
    ("list_core_memories", lambda w, a, own, other: a.call("list_core_memories")),
    ("space_list", lambda w, a, own, other: a.call("space_list", include_archived=True)),
]


async def _pin_everything(world: SpaceWorld) -> None:
    """Make one node per scope CORE so core injection / core listing paths have data."""
    for label, actor in (("a_q1", world.alice), ("b_q2", world.bob), ("p_notes", world.alice)):
        res = await actor.call("pin_memory", world.ids[label])
        assert res["status"] == "pinned"


@pytest.mark.parametrize("perspective", ["alice", "bob"])
@pytest.mark.parametrize("name,factory", STORAGE_READS + TOOL_READS, ids=[n for n, _ in STORAGE_READS + TOOL_READS])
async def test_every_read_path_respects_scope(space_world: SpaceWorld, perspective: str, name: str, factory: Any):
    w = space_world
    await _pin_everything(w)
    if perspective == "alice":
        actor, own, other = w.alice, w.ids["a_q1"], w.ids["b_q2"]
        forbidden = w.b_ids
    else:
        actor, own, other = w.bob, w.ids["b_q2"], w.ids["a_q1"]
        forbidden = w.a_ids | w.personal_ids
    if name == "vector_search":
        emb = await w.handler.embeddings.embed("Bank A pilot starts in Q2")
        result = await actor.storage("vector_search", emb, k=100)
    else:
        result = await factory(w, actor, own, other)
    if name.endswith("_other") and name.startswith("memory_"):
        # an invisible id is not_found; the only id in the body is the caller's own input
        assert result["code"] == "not_found"
        result = {k: v for k, v in result.items() if k != "node_id"}
    leaked = _walk_strings(result) & forbidden
    assert not leaked, f"{name} as {perspective} leaked {leaked}"


@pytest.mark.parametrize("perspective", ["alice", "bob"])
async def test_stats_and_degree_count_only_scope(space_world: SpaceWorld, perspective: str):
    w = space_world
    actor, other = (w.alice, w.ids["b_ctx"]) if perspective == "alice" else (w.bob, w.ids["a_ctx"])
    assert await actor.storage("get_degree", other) == 0
    assert await actor.storage("get_causal_weight", other) == 0
    assert await actor.storage("is_orphan", other) is True
    personal = await actor.call("memory_stats")
    expected_personal = 2 if perspective == "alice" else 0
    assert personal["nodes"] == expected_personal
    own_space = w.a if perspective == "alice" else w.b
    in_space = await actor.call("memory_stats", space=own_space)
    assert in_space["nodes"] == 2
    other_space = w.b if perspective == "alice" else w.a
    assert (await actor.call("memory_stats", space=other_space))["code"] == "space_not_found"


async def test_vector_and_keyword_search_no_leak(space_world: SpaceWorld):
    w = space_world
    # B's exact content is the best match, yet alice only gets A + personal.
    rec = await w.alice.call("memory_recall", "Bank A pilot starts in Q2", k=20)
    ids = {r["id"] for r in rec["results"]}
    assert ids and not ids & w.b_ids
    assert ids <= w.a_ids | w.personal_ids
    srch = await w.alice.call("memory_search", "Bank A pilot starts in Q2", k=20)
    assert {r["id"] for r in srch["results"]} <= w.a_ids | w.personal_ids
    # Each result is labelled with its space (or null for personal).
    for r in rec["results"]:
        if r["id"] in w.a_ids:
            assert r["space"] == {"id": w.a, "slug": "space-a", "name": "Space A"}
        else:
            assert r["space"] is None


async def test_traverse_cannot_cross(space_world: SpaceWorld):
    w = space_world
    # A cross-bucket edge can't exist (storage refuses), and shared entity
    # names don't bridge spaces: traversal from A stays in A.
    tr = await w.alice.call("memory_traverse", w.ids["a_q1"], depth=10)
    seen = {n["id"] for n in tr["nodes"]}
    assert w.ids["a_ctx"] in seen
    assert not seen & (w.b_ids | w.personal_ids)
    assert not _walk_strings(tr["edges"]) & w.b_ids


async def test_get_node_by_id_from_other_space(space_world: SpaceWorld):
    w = space_world
    assert await w.alice.storage("get_node", w.ids["b_q2"]) is None
    hidden = await w.alice.call("memory_explain", w.ids["b_q2"])
    assert hidden["code"] == "not_found"
    missing_id = "00000000-0000-4000-8000-000000000000"
    missing = await w.alice.call("memory_explain", missing_id)
    assert {k: v for k, v in hidden.items() if k != "node_id"} == {k: v for k, v in missing.items() if k != "node_id"}
    # traverse of an invisible start is not_found too (not an empty success)
    tr = await w.alice.call("memory_traverse", w.ids["b_q2"])
    assert tr == {"error": "Node not found", "code": "not_found", "node_id": w.ids["b_q2"]}


@pytest.mark.parametrize("tool,args", [
    ("memory_recall", {"query": "pilot"}),
    ("memory_search", {"query": "pilot"}),
    ("memory_stats", {}),
    ("list_core_memories", {}),
    ("memory_store", {"content": "x"}),
    ("space_get", {}),
    ("space_member_list", {}),
    ("space_update", {"name": "n"}),
])
async def test_space_not_found_is_identical(space_world: SpaceWorld, tool: str, args: dict[str, Any]):
    w = space_world
    bodies = set()
    for ref in (w.b, "space-b", "spc_doesnotexist", "no-such-slug"):
        result = await w.alice.call(tool, space=ref, **args)
        bodies.add(_dump(result))
    assert len(bodies) == 1
    assert json.loads(bodies.pop()) == {"error": "Space not found", "code": "space_not_found"}


async def test_space_not_found_for_traverse_and_spaces_list(space_world: SpaceWorld):
    w = space_world
    a = await w.alice.call("memory_traverse", w.ids["a_q1"], space=w.b)
    b = await w.alice.call("memory_traverse", w.ids["a_q1"], space="nope")
    c = await w.alice.call("memory_recall", "pilot", spaces=[w.a, w.b])
    assert _dump(a) == _dump(b) == _dump(c)


async def test_space_and_spaces_together_invalid(space_world: SpaceWorld):
    res = await space_world.alice.call("memory_recall", "pilot", space=space_world.a, spaces=[space_world.a])
    assert res["code"] == "invalid_arguments"


async def test_cross_space_edge_refused_in_tool(space_world: SpaceWorld):
    w = space_world
    before = sum(len(b) for b in w.graph._user_nodes.values())
    res = await w.alice.call("memory_store", "links to B", space=w.a,
                             related=[{"id": w.ids["b_q2"], "type": "supports"}])
    assert res["code"] == "cross_space_edge"
    res = await w.alice.call("memory_store", "links to B legacy", space=w.a, related_to=[w.ids["b_q2"]])
    assert res["code"] == "cross_space_edge"
    assert sum(len(b) for b in w.graph._user_nodes.values()) == before


async def test_cross_space_edge_refused_in_storage(space_world: SpaceWorld):
    w = space_world
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        with pytest.raises(ValueError, match="cross_space_edge"):
            await w.graph.create_edge(MemoryEdge(
                source_id=w.ids["a_q1"], target_id=w.ids["b_q2"], type=EdgeType.RELATED_TO,
            ))


async def test_space_to_personal_edge_refused(space_world: SpaceWorld):
    w = space_world
    before = sum(len(b) for b in w.graph._user_nodes.values())
    into_space = await w.alice.call("memory_store", "space to personal", space=w.a,
                                    related=[{"id": w.ids["p_notes"], "type": "supports"}])
    assert into_space["code"] == "cross_space_edge"
    personal = await w.alice.call("memory_store", "personal to space",
                                  related=[{"id": w.ids["a_q1"], "type": "supports"}])
    assert personal["code"] == "cross_space_edge"
    assert sum(len(b) for b in w.graph._user_nodes.values()) == before
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        with pytest.raises(ValueError, match="cross_space_edge"):
            await w.graph.create_edge(MemoryEdge(
                source_id=w.ids["a_q1"], target_id=w.ids["p_notes"], type=EdgeType.RELATED_TO,
            ))


async def test_personal_write_keeps_legacy_skip_for_invisible_targets(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store", "personal links to bob's space", related_to=[w.ids["b_q2"]])
    assert res["status"] == "stored"  # warn-and-skip, as tests/test_phase1_security.py pins
    edges = await w.alice.storage("get_edges", res["node_id"], "both")
    assert w.ids["b_q2"] not in _walk_strings(edges)


async def test_autolink_stays_in_space(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store", "Bank A pilot starts in Q1 or Q2", space=w.a)
    new_id = res["node_id"]
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner", w.b: "owner"}):
        # even with a scope covering everything, the new node's edges stay in A
        edges = await w.graph.get_edges(new_id, "both")
    auto = [e for e in edges if e.created_by == "auto_link"]
    assert auto, "positive control: near-identical A content must auto-link"
    for e in edges:
        assert {str(e.source_id), str(e.target_id)} <= w.a_ids | {new_id}
    # and a personal store never links into a space
    pres = await w.alice.call("memory_store", "Bank A pilot starts in Q1 or Q2 personally")
    pedges = await w.alice.storage("get_edges", pres["node_id"], "both")
    assert pedges
    for e in pedges:
        assert {str(e.source_id), str(e.target_id)} <= w.personal_ids | {pres["node_id"]}


async def test_possible_conflicts_stay_in_space(space_world: SpaceWorld):
    w = space_world
    await w.alice.call("memory_store", "Payments budget is 500 dollars", space=w.a)
    await w.bob.call("memory_store", "Payments budget is 500 dollars", space=w.b)
    personal = await w.alice.call("memory_store", "Payments budget is 500 dollars")
    res = await w.alice.call("memory_store", "Payments budget is 900 dollars", space=w.a)
    conflicts = res.get("possible_conflicts", [])
    assert conflicts, "positive control: the A twin must be flagged"
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        for c in conflicts:
            node = await w.graph.get_node(c["id"])
            assert node is not None and node.space_id == w.a
    assert personal["node_id"] not in {c["id"] for c in conflicts}


# -- services ---------------------------------------------------------------------


async def test_service_never_sees_personal(space_world: SpaceWorld):
    w = space_world
    forbidden = w.personal_ids | w.b_ids
    rec = await w.svc.call("memory_recall", "Bank A pilot notes", k=50, include_personal=True)
    srch = await w.svc.call("memory_search", "Bank A pilot notes", k=50, include_personal=True)
    enum = await w.svc.call("memory_search", "", k=100, include_personal=True)
    tr = await w.svc.call("memory_traverse", w.ids["a_q1"], depth=10, include_personal=True)
    personal_tr = await w.svc.call("memory_traverse", w.ids["p_notes"], include_personal=True)
    for result in (rec, srch, enum, tr):
        assert not _walk_strings(result) & forbidden
    assert {r["id"] for r in rec["results"]} <= w.a_ids
    assert rec["results"], "positive control: the service sees its space"
    assert personal_tr["code"] == "not_found"
    stats = await w.svc.call("memory_stats")
    assert stats["nodes"] == 0


async def test_service_write_requires_space(space_world: SpaceWorld):
    res = await space_world.svc.call("memory_store", "service note")
    assert res == {"error": "A service principal must name a `space` to write to", "code": "space_required"}
    ok = await space_world.svc.call("memory_store", "service note", space=space_world.a)
    assert ok["status"] == "stored"


async def test_service_cannot_write_other_space(space_world: SpaceWorld):
    w = space_world
    res = await w.svc.call("memory_store", "into B", space=w.b)
    missing = await w.svc.call("memory_store", "into nowhere", space="spc_nope")
    assert res == missing == {"error": "Space not found", "code": "space_not_found"}


async def test_service_cannot_create_spaces(space_world: SpaceWorld):
    res = await space_world.svc.call("space_create", "Svc space")
    assert res["code"] == "insufficient_role"


# -- roles, removal, archive -----------------------------------------------------------


async def test_reader_cannot_write(space_world: SpaceWorld):
    w = space_world
    await _add(w, w.a, "carol", SpaceRole.READER)
    carol = w.actor("carol")
    target = w.ids["a_q1"]
    assert (await carol.call("memory_recall", "pilot", space=w.a))["results"]
    for result in (
        await carol.call("memory_store", "reader write", space=w.a),
        await carol.call("memory_amend", target, "reader amend"),
        await carol.call("pin_memory", target),
        await carol.call("unpin_memory", target),
        await carol.call("delete_memory", target),
        await carol.call("promote_to_space", w.ids["a_q1"], w.a),
    ):
        assert result["code"] in ("insufficient_role", "not_found"), result
    assert (await carol.call("memory_store", "reader write", space=w.a))["code"] == "insufficient_role"
    assert (await carol.call("pin_memory", target))["code"] == "insufficient_role"
    assert (await carol.call("delete_memory", target))["code"] == "insufficient_role"
    assert (await carol.call("memory_amend", target, "x"))["code"] == "insufficient_role"
    assert await w.alice.storage("get_node", target) is not None


async def test_writer_can_write_and_author_is_recorded(space_world: SpaceWorld):
    w = space_world
    res = await w.svc.call("memory_store", "svc writes", space=w.a)
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        node = await w.graph.get_node(res["node_id"])
    assert node.space_id == w.a and node.org_id is None and node.visibility.value == "private"
    assert node.original_user_id == "svc_a"
    # a second writer (not the author) may amend, pin and delete space nodes
    amended = await w.alice.call("memory_amend", res["node_id"], "svc writes, corrected")
    assert amended["status"] == "amended"
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        new = await w.graph.get_node(amended["node_id"])
    assert new.space_id == w.a
    assert (await w.svc.call("pin_memory", amended["node_id"]))["status"] == "pinned"
    assert (await w.svc.call("delete_memory", amended["node_id"]))["status"] == "deleted"


async def test_visibility_org_with_space_invalid(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store", "x", visibility="org", org_id=WORLD_ORG, space=w.a)
    assert res["code"] == "invalid_arguments"


async def test_member_removal_immediate(space_world: SpaceWorld, monkeypatch):
    """Through the stdio transport's per-call refresh (server.call_tool)."""
    import papez.server as srv

    w = space_world
    await _add(w, w.a, "dave", SpaceRole.WRITER)
    monkeypatch.setattr(srv.tools, "spaces", w.registry)
    method, required, optional = srv._TOOL_DISPATCH["memory_recall"]
    monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_recall", (w.handler.memory_recall, required, optional))

    stale_roles = {w.a: "writer"}  # what the transport loaded at session start
    with principal("dave", orgs=[WORLD_ORG], roles=stale_roles, client={"transport": "stdio", "client_name": None}):
        first = json.loads((await srv.call_tool("memory_recall", {"query": "pilot", "space": w.a}))[0].text)
        assert {r["id"] for r in first["results"]} & w.a_ids

        removed = await w.alice.call("space_member_remove", w.a, "dave")
        assert removed == {"removed": True}

        explicit = json.loads((await srv.call_tool("memory_recall", {"query": "pilot", "space": w.a}))[0].text)
        assert explicit == {"error": "Space not found", "code": "space_not_found"}
        default = json.loads((await srv.call_tool("memory_recall", {"query": "Bank A pilot"}))[0].text)
        assert not {r["id"] for r in default["results"]} & w.a_ids


async def test_archived_space(space_world: SpaceWorld):
    w = space_world
    await w.alice.call("pin_memory", w.ids["a_ctx"])
    archived = await w.alice.call("space_archive", w.a)
    assert archived["space"]["archived"] is True
    default = await w.alice.call("memory_recall", "Bank A pilot starts", k=50)
    assert not {r["id"] for r in default["results"]} & w.a_ids
    explicit = await w.alice.call("memory_recall", "Bank A pilot starts", k=50, space=w.a)
    assert {r["id"] for r in explicit["results"]} & w.a_ids
    assert (await w.alice.call("memory_search", "", space="space-a", k=50))["results"]
    assert (await w.alice.call("space_get", w.a))["space"]["archived"] is True
    assert (await w.alice.call("space_member_list", w.a))["members"]
    target = w.ids["a_q1"]
    writes = [
        await w.alice.call("memory_store", "archived write", space=w.a),
        await w.svc.call("memory_store", "archived write", space=w.a),
        await w.alice.call("memory_amend", target, "archived amend"),
        await w.alice.call("pin_memory", target),
        await w.alice.call("unpin_memory", w.ids["a_ctx"]),
        await w.alice.call("delete_memory", target),
        await w.alice.call("promote_to_space", w.ids["p_notes"], w.a),
        await w.alice.call("space_update", w.a, name="New"),
        await w.alice.call("space_member_add", w.a, "carol", "user", "reader"),
        await w.alice.call("space_member_remove", w.a, "svc_a"),
        await w.alice.call("space_archive", w.a),
    ]
    for result in writes:
        assert result["code"] == "space_archived", result
    assert (await w.alice.call("space_unarchive", w.a))["space"]["archived"] is False
    assert (await w.alice.call("memory_store", "after unarchive", space=w.a))["status"] == "stored"


async def test_erase_space(space_world: SpaceWorld):
    w = space_world
    a_nodes = set(w.graph._user_nodes.get(f"space::{w.a}", {}))
    a_edges = len(w.graph._user_edges.get(f"space::{w.a}", []))
    b_nodes_before = set(w.graph._user_nodes.get(f"space::{w.b}", {}))
    assert a_nodes == w.a_ids and a_edges >= 1

    mismatch = await w.alice.call("space_erase", w.a, confirm_slug="wrong")
    assert mismatch["code"] == "confirmation_mismatch"
    dry = await w.alice.call("space_erase", w.a, confirm_slug="space-a", dry_run=True)
    assert dry == {"erased": {"space_id": w.a, "nodes": 2, "edges": a_edges, "members": 2}, "dry_run": True}
    assert set(w.graph._user_nodes.get(f"space::{w.a}", {})) == a_nodes

    assert (await w.bob.call("space_erase", w.a, confirm_slug="space-a"))["code"] == "space_not_found"
    assert (await w.svc.call("space_erase", w.a, confirm_slug="space-a"))["code"] == "insufficient_role"

    real = await w.alice.call("space_erase", w.a, confirm_slug="space-a")
    assert real["erased"]["nodes"] == 2 and real["dry_run"] is False
    assert not w.graph._user_nodes.get(f"space::{w.a}")
    for nid in a_nodes:
        assert w.graph._locate(nid) is None
    assert set(w.graph._user_nodes.get(f"space::{w.b}", {})) == b_nodes_before
    assert (await w.alice.call("space_get", w.a))["code"] == "space_not_found"
    assert (await w.alice.call("space_list"))["spaces"] == []
    # the id and the slug can never be reused
    with pytest.raises(ValueError, match="invalid_arguments"):
        await w.registry.create_space(Space(id=w.a, org_id=WORLD_ORG, name="Again", slug="again", created_by="alice"))
    again = await w.alice.call("space_create", "Space A", slug="space-a")
    assert again["code"] == "slug_taken"


async def test_erase_user_keeps_space_memories(space_world: SpaceWorld):
    w = space_world
    await _add(w, w.a, "bob", SpaceRole.OWNER)  # so alice is not the last owner
    res = await w.alice.call("erase_user", "alice", keep_promoted_nodes=False)
    assert res["status"] == "erased"
    assert res["spaces_still_owned"] == []
    for nid in w.personal_ids:
        assert w.graph._locate(nid) is None
    for nid in w.a_ids:
        located = w.graph._locate(nid)
        assert located is not None
        assert located[1].original_user_id == "erased_user"
    assert await w.registry.get_member(w.a, "alice") is None
    assert res["space_nodes_anonymized"] == 2


async def test_erase_user_last_owner_keeps_membership(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("erase_user", "alice")
    assert res["spaces_still_owned"] == [w.a]
    assert await w.registry.get_member(w.a, "alice") is not None


async def test_last_owner_protected(space_world: SpaceWorld):
    w = space_world
    assert (await w.alice.call("space_member_remove", w.a, "alice"))["code"] == "last_owner"
    assert (await w.alice.call("space_member_add", w.a, "alice", "user", "writer"))["code"] == "last_owner"
    # a service owner doesn't count, and can't be added as owner anyway
    assert (await w.alice.call("space_member_add", w.a, "svc_a", "service", "owner"))["code"] == "invalid_arguments"
    await _add(w, w.a, "bob", SpaceRole.OWNER)
    assert (await w.alice.call("space_member_add", w.a, "alice", "user", "writer"))["created"] is False


async def test_promote_to_space_copies(space_world: SpaceWorld):
    w = space_world
    original_id = w.ids["p_notes"]
    res = await w.alice.call("promote_to_space", original_id, w.a)
    assert res["space"] == {"id": w.a, "slug": "space-a", "name": "Space A"}
    new_id = res["node_id"]
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}, include_personal=True):
        copy = await w.graph.get_node(new_id)
        orig = await w.graph.get_node(original_id)
        copy_edges = await w.graph.get_edges(new_id, "both")
    assert copy.space_id == w.a and copy.promoted_from == original_id
    assert [r.id for r in copy.refs] == ["doc_bank_a"] and copy.tags == ["bank-a", "pilot"]
    assert copy.created_at == orig.created_at and copy.content_full == orig.content_full
    assert copy.original_user_id == "alice" and copy.org_id is None
    assert all(e.created_by != "user_explicit" for e in copy_edges)
    assert not {str(e.source_id) for e in copy_edges} & w.personal_ids
    assert not {str(e.target_id) for e in copy_edges} & w.personal_ids
    assert orig is not None  # kept by default

    moved = await w.alice.call("promote_to_space", w.ids["p_more"], "space-a", keep_original=False)
    assert moved["node_id"]
    assert w.graph._locate(w.ids["p_more"]) is None
    # bob can't promote alice's memory, nor into a space he isn't in
    assert (await w.bob.call("promote_to_space", original_id, w.b))["code"] == "not_found"
    assert (await w.alice.call("promote_to_space", original_id, w.b))["code"] == "space_not_found"
    # a space node can't be promoted again
    assert (await w.alice.call("promote_to_space", new_id, w.a))["code"] == "not_found"


async def test_promote_to_org_refuses_space_nodes(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("promote_to_org", w.ids["a_q1"], WORLD_ORG)
    assert res["code"] == "invalid_arguments"


async def test_world_builder_is_reusable():
    # build_space_world is importable for other test modules (e.g. WS-L6/L7).
    world = await build_space_world()
    assert world.a != world.b and len(world.ids) == 6


async def test_explicit_space_excludes_personal_unless_asked(space_world: SpaceWorld):
    w = space_world
    only_a = await w.alice.call("memory_recall", "Bank A pilot", k=50, space=w.a)
    assert {r["id"] for r in only_a["results"]} == w.a_ids
    with_personal = await w.alice.call("memory_recall", "Bank A pilot", k=50, space=w.a, include_personal=True)
    assert {r["id"] for r in with_personal["results"]} == w.a_ids | w.personal_ids
    no_personal = await w.alice.call("memory_recall", "Bank A pilot", k=50, include_personal=False)
    assert {r["id"] for r in no_personal["results"]} == w.a_ids
    default = await w.alice.call("memory_recall", "Bank A pilot", k=50)
    assert {r["id"] for r in default["results"]} == w.a_ids | w.personal_ids
    searched = await w.alice.call("memory_search", "", space="space-a", k=50)
    assert {r["id"] for r in searched["results"]} == w.a_ids
