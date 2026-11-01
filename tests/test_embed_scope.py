"""With current_embed bound, tools and storage stay on the token's one space.

The issuer here is deliberately a member of space B as well, and alice has
personal memory, so a leak through the issuer's identity would show up.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from conftest import WORLD_ORG, SpaceWorld, build_space_world, principal

from papez.context import current_embed
from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, PrincipalType, SpaceRole
from papez.models.node import MemoryNode
from papez.models.space import SpaceMember

VIEWER = {"id": "u_123", "email": "pat@example.com", "name": "Pat Lee"}
STAMP = {**VIEWER, "via": "embed"}
TOKEN_ID = "emb_test"
ISSUERS = [("alice", "user", "owner"), ("svc_a", "service", "writer")]


@contextmanager
def embed(world: SpaceWorld, issuer: str, ptype: str, role: str, *, extra_roles: dict[str, str] | None = None) -> Iterator[None]:
    """Bind an embed for space A; ``extra_roles`` proves other memberships don't widen it."""
    roles = {world.a: role, **(extra_roles or {})}
    with principal(issuer, orgs=[WORLD_ORG], roles=roles, principal_type=ptype):
        token = current_embed.set({
            "token_id": TOKEN_ID, "space_id": world.a, "viewer": dict(VIEWER), "permissions": ["read", "write_notes"],
        })
        try:
            yield
        finally:
            current_embed.reset(token)


@pytest.fixture()
async def world() -> SpaceWorld:
    w = await build_space_world()
    for uid, ptype in (("alice", PrincipalType.USER), ("svc_a", PrincipalType.SERVICE)):
        await w.registry.add_member(SpaceMember(
            space_id=w.b, principal_id=uid, principal_type=ptype, role=SpaceRole.READER, added_by="bob",
        ))
    for label, actor in (("a_q1", w.alice), ("b_q2", w.bob), ("p_notes", w.alice)):
        assert (await actor.call("pin_memory", w.ids[label]))["status"] == "pinned"
    w.audit.events.clear()
    return w


def _walk(obj: Any) -> set[str]:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    out: set[str] = set()
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            out |= _walk(key) | _walk(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            out |= _walk(value)
    return out


def _forbidden(w: SpaceWorld) -> set[str]:
    return w.b_ids | w.personal_ids


# -- storage ------------------------------------------------------------------


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_every_storage_read_stays_on_the_space(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    every = list(w.ids.values())
    other_scope: dict[str, Any] = {"org_ids": [WORLD_ORG], "space_ids": [w.b]}
    embedding = await w.handler.embeddings.embed("Bank A pilot starts in Q2")
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        results: dict[str, Any] = {
            "get_node": [await w.graph.get_node(i) for i in every],
            "by_status": [
                await w.graph.get_nodes_by_status(s, 1000) for s in (MemoryStatus.ACTIVE, MemoryStatus.CORE)
            ],
            "get_edges": [await w.graph.get_edges(i, "both") for i in every],
            "all_edges": [await w.graph.get_all_edges(), await w.graph.get_all_edges(every)],
            "traverse": [await w.graph.traverse(i, 10, **other_scope) for i in every],
            "connecting": await w.graph.get_connecting_edges(every, **other_scope),
            "chain": [await w.graph.get_causal_chain(i, "upstream", **other_scope) for i in every],
            "chains": list((await w.graph.get_causal_chains_batch(every, "downstream", **other_scope)).values()),
            "vector": await w.graph.vector_search(embedding, k=100, **other_scope),
            "keyword": await w.graph.keyword_search("bank pilot", k=100, **other_scope),
            "ref": await w.graph.find_nodes_by_ref("document", "doc_bank_a", **other_scope),
            "orphans": await w.graph.get_orphans(),
            "list_b": await w.graph.list_space_nodes(w.b),
            "tags_b": await w.graph.space_tag_counts(w.b),
        }
        stats = await w.graph.get_stats()
        degrees = {i: await w.graph.get_degree(i) for i in every}
        allowed = await w.graph.list_space_nodes(w.a, limit=100)
        unknown_degree = await w.graph.get_degree("00000000-0000-0000-0000-000000000000")
    strings = _walk({k: v for k, v in results.items() if k not in ("vector",)})
    strings |= _walk([n for n, _ in results["vector"]])
    assert not strings & _forbidden(w)
    assert results["list_b"] == [] and results["tags_b"] == []
    assert stats["nodes"] == len(w.a_ids)
    assert {str(n.id) for n in allowed} == w.a_ids
    assert all(degrees[i] == 0 for i in _forbidden(w))
    assert unknown_degree == 0


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_storage_writes_refuse_anything_outside_the_space(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        personal_edge = next(
            e for e in await w.graph.get_edges(w.ids["p_more"], "both") if e.type == EdgeType.CAUSED_BY
        )
        edges_before = len(await w.graph.get_edges(w.ids["p_more"], "both"))
        weight_before = personal_edge.weight
    with embed(w, issuer, ptype, role):
        with pytest.raises(ValueError):
            await w.graph.create_node(MemoryNode(content_summary="personal"))
        with pytest.raises(ValueError):
            await w.graph.create_node(MemoryNode(content_summary="other", space_id=w.b))
        with pytest.raises(ValueError):
            await w.graph.create_edge(MemoryEdge(
                source_id=w.ids["p_more"], target_id=w.ids["p_notes"], type=EdgeType.RELATED_TO,
            ))
        assert not await w.graph.edge_exists(w.ids["p_more"], w.ids["p_notes"], EdgeType.CAUSED_BY)
        for operation in (w.graph.erase_user(issuer), w.graph.erase_space(w.a), w.graph.destroy(issuer)):
            with pytest.raises(PermissionError):
                await operation
        await w.graph.delete_edge(str(personal_edge.id))
        await w.graph.update_edge_weight(str(personal_edge.id), 0.01)
        await w.graph.delete_node(w.ids["p_more"])
        await w.graph.update_node(w.ids["b_q2"], {"pinned": False})
    with principal("alice", orgs=[WORLD_ORG], roles={w.a: "owner"}):
        assert len(await w.graph.get_edges(w.ids["p_more"], "both")) == edges_before
        assert personal_edge.weight == weight_before


# -- read tools ----------------------------------------------------------------


READ_CALLS = [
    ("recall_named", lambda w: w.handler.memory_recall("Bank A pilot starts", k=50, space=w.b)),
    ("recall_spaces", lambda w: w.handler.memory_recall("Bank A pilot starts", k=50, spaces=[w.b, w.a])),
    ("recall_personal", lambda w: w.handler.memory_recall("Bank A pilot notes", k=50, include_personal=True)),
    ("recall_both_args", lambda w: w.handler.memory_recall("Bank A pilot", k=50, space=w.b, spaces=[w.b])),
    ("recall_light", lambda w: w.handler.memory_recall("Bank A pilot", k=50, verbosity="light", space=w.b)),
    ("search", lambda w: w.handler.memory_search("Bank A pilot starts", k=50, space=w.b, include_personal=True)),
    ("enumerate", lambda w: w.handler.memory_search("", k=100, spaces=[w.b], include_personal=True)),
    ("traverse", lambda w: w.handler.memory_traverse(w.ids["a_q1"], depth=10, space=w.b, include_personal=True)),
    ("core", lambda w: w.handler.list_core_memories(space=w.b)),
    ("core_default", lambda w: w.handler.list_core_memories()),
    ("space_memories", lambda w: w.handler.space_memories(w.b, limit=100, include_superseded=True)),
    ("space_tags", lambda w: w.handler.space_tags(w.b)),
    ("space_get", lambda w: w.handler.space_get(w.b)),
    ("space_members", lambda w: w.handler.space_member_list(w.b)),
    ("space_list", lambda w: w.handler.space_list(include_archived=True)),
]


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
@pytest.mark.parametrize("name,call", READ_CALLS, ids=[n for n, _ in READ_CALLS])
async def test_reads_ignore_scope_arguments(world: SpaceWorld, issuer: str, ptype: str, role: str, name: str, call: Any):
    w = world
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        result = await call(w)
    assert "error" not in result, result
    strings = _walk(result)
    assert not strings & _forbidden(w)
    assert w.b not in strings
    assert "Space B" not in strings
    if name == "space_memories":
        assert {i["id"] for i in result["items"]} == w.a_ids
    if name == "recall_named":
        assert {r["space"]["id"] for r in result["results"]} == {w.a}
    if name in ("space_get",):
        assert result["space"]["id"] == w.a
    if name == "space_list":
        assert [s["id"] for s in result["spaces"]] == [w.a]


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_stats_count_only_the_space(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    with embed(w, issuer, ptype, role):
        for kwargs in ({}, {"space": w.b}):
            assert (await w.handler.memory_stats(**kwargs))["nodes"] == len(w.a_ids)


NODE_TOOLS = ["memory_explain", "memory_detail", "memory_traverse", "pin_memory", "unpin_memory", "delete_memory"]


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
@pytest.mark.parametrize("tool", NODE_TOOLS)
@pytest.mark.parametrize("label", ["b_q2", "p_notes"])
async def test_node_tools_answer_not_found_outside_the_space(
    world: SpaceWorld, issuer: str, ptype: str, role: str, tool: str, label: str,
):
    w = world
    node_id = w.ids[label]
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        missing = "00000000-0000-0000-0000-000000000000"
        hidden = await getattr(w.handler, tool)(node_id)
        absent = await getattr(w.handler, tool)(missing)
    assert hidden["code"] == "not_found"
    assert json.dumps(hidden).replace(node_id, "X") == json.dumps(absent).replace(missing, "X")


# -- writes --------------------------------------------------------------------


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
@pytest.mark.parametrize("kwargs", [{}, {"space": "B"}, {"on_behalf_of": {"id": "forged", "via": "api"}},
                                    {"on_behalf_of": "not an object"}], ids=["none", "other_space", "forged", "invalid"])
async def test_store_lands_in_the_space_stamped_with_the_viewer(
    world: SpaceWorld, issuer: str, ptype: str, role: str, kwargs: dict[str, Any],
):
    w = world
    if kwargs.get("space") == "B":
        kwargs = {"space": w.b}
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        out = await w.handler.memory_store("Viewer note about the payments rollout", **kwargs)
        assert "error" not in out, out
        node = await w.graph.get_node(out["node_id"])
    assert node.space_id == w.a
    assert node.on_behalf_of == STAMP
    assert node.original_user_id == issuer
    assert out["space"]["id"] == w.a


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_batch_amend_retire_stamp_and_stay_on_the_space(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    refs = [{"type": "document", "id": "doc_x"}, {"type": "document", "id": "doc_y"}]
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        batch = await w.handler.memory_store_batch(
            [{"content": "Batch note one", "refs": refs}, {"content": "Batch note two"}],
            space=w.b, on_behalf_of={"id": "forged"},
        )
        assert "error" not in batch, batch
        first, second = (r["node_id"] for r in batch["results"])
        amended = await w.handler.memory_amend(second, "Batch note two, corrected", on_behalf_of={"id": "forged"})
        retired = await w.handler.memory_retire_by_ref("document", "doc_x", space=w.b, on_behalf_of={"id": "forged"})
        assert retired == {"retired": 0, "updated": 1, "deleted": 0}
        current = await w.graph.list_space_nodes(w.a, limit=100)
        new_ids = {amended["node_id"], *(str(n.id) for n in current if n.content_full == "Batch note one")}
        stamped = {str(n.id): n.on_behalf_of for n in current if str(n.id) in new_ids}
        with_ref_left = next(n for n in current if n.content_full == "Batch note one")
    assert first != str(with_ref_left.id)
    assert len(stamped) == 2 and all(v == STAMP for v in stamped.values())


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_write_cannot_reach_out_of_the_space(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        for target in ("b_q2", "p_notes"):
            linked = await w.handler.memory_store("Note", related=[{"id": w.ids[target], "type": "supports"}])
            assert linked["code"] == "cross_space_edge"
            amended = await w.handler.memory_amend(w.ids[target], "Rewritten")
            assert amended["code"] == "not_found"
        org_write = await w.handler.memory_store("Note", visibility="org", org_id=WORLD_ORG)
        assert org_write["code"] == "invalid_arguments"
        moved = await w.handler.promote_to_space(w.ids["p_notes"], w.b)
        assert moved["code"] == "not_found"
        to_org = await w.handler.promote_to_org(w.ids["p_notes"], WORLD_ORG)
        assert to_org["code"] == "not_found"
        # The shared ref also exists in B and in personal memory; only A's copies may go.
        retired = await w.handler.memory_retire_by_ref("document", "doc_bank_a", space=w.b, mode="delete")
        assert retired["deleted"] == len(w.a_ids)
    # Nothing outside the space moved or went.
    with principal("bob", orgs=[WORLD_ORG], roles={w.b: "owner"}):
        assert await w.graph.get_node(w.ids["b_q2"]) is not None
    with principal("alice", orgs=[WORLD_ORG], roles={}):
        assert await w.graph.get_node(w.ids["p_notes"]) is not None


async def test_role_and_archive_checks_still_apply(world: SpaceWorld):
    w = world
    with embed(w, "rita", "user", "reader"):
        out = await w.handler.memory_store("Reader tries to write")
        assert out["code"] == "insufficient_role"
        # Read access is fine for a reader.
        assert "error" not in await w.handler.space_memories("ignored")
    assert (await w.alice.call("space_archive", w.a))["space"]["archived"] is True
    with embed(w, "alice", "user", "owner"):
        assert (await w.handler.memory_store("Into an archive"))["code"] == "space_archived"
        assert (await w.handler.memory_recall("Bank A pilot"))["count"] > 0
        assert (await w.handler.space_memories("ignored"))["items"]
        assert "error" not in await w.handler.memory_detail(w.ids["a_q1"])


async def test_issuer_with_no_membership_fails_closed(world: SpaceWorld):
    w = world
    with principal("nora", orgs=[WORLD_ORG], roles={}):
        token = current_embed.set({
            "token_id": TOKEN_ID, "space_id": w.a, "viewer": dict(VIEWER), "permissions": ["read"],
        })
        try:
            assert (await w.handler.memory_recall("Bank A pilot"))["code"] == "space_not_found"
            assert (await w.handler.space_memories(w.a))["code"] == "space_not_found"
        finally:
            current_embed.reset(token)


ALWAYS_REFUSED = [
    ("space_create", lambda w: w.handler.space_create("Another")),
    ("erase_user", lambda w: w.handler.erase_user("alice")),
    ("set_core_preferences", lambda w: w.handler.set_core_preferences(auto=["family"])),
    ("space_update", lambda w: w.handler.space_update(w.a, name="Renamed")),
    ("space_member_add", lambda w: w.handler.space_member_add(w.a, "x", "user", "reader")),
    ("space_member_remove", lambda w: w.handler.space_member_remove(w.a, "svc_a")),
    ("space_archive", lambda w: w.handler.space_archive(w.a)),
    ("space_unarchive", lambda w: w.handler.space_unarchive(w.a)),
    ("space_erase", lambda w: w.handler.space_erase(w.a, "space-a")),
]


@pytest.mark.parametrize("name,call", ALWAYS_REFUSED, ids=[n for n, _ in ALWAYS_REFUSED])
async def test_management_is_refused_even_for_an_owner_issuer(world: SpaceWorld, name: str, call: Any):
    w = world
    with embed(w, "alice", "user", "owner"):
        assert (await call(w))["code"] == "embed_not_allowed"
    assert (await w.registry.get_space(w.a)).name == "Space A"


async def test_context_unbound_again_after_the_block(world: SpaceWorld):
    w = world
    with embed(w, "alice", "user", "owner"):
        pass
    assert current_embed.get() is None
    assert (await w.alice.call("memory_store", "Personal again"))["visibility"] == "private"


# -- audit ---------------------------------------------------------------------


@pytest.mark.parametrize("issuer,ptype,role", ISSUERS)
async def test_audit_detail_carries_the_viewer_and_no_content(world: SpaceWorld, issuer: str, ptype: str, role: str):
    w = world
    secret_note = "Confidential viewer note xyzzy"
    secret_query = "plugh pilot question"
    with embed(w, issuer, ptype, role, extra_roles={w.b: "reader"}):
        stored = await w.handler.memory_store(secret_note, space=w.b)
        await w.handler.memory_recall(secret_query, space=w.b)
        await w.handler.memory_search(secret_query)
        await w.handler.space_memories(w.b)
        await w.handler.space_tags(w.b)
        await w.handler.memory_detail(stored["node_id"])
        await w.handler.memory_explain(stored["node_id"])
        await w.handler.audit_event("embed.mint", w.a, None, {"viewer_id": VIEWER["id"]})
    events = w.audit.events
    assert {e["action"] for e in events} >= {
        "memory.store", "memory.recall", "memory.search", "memory.list", "space.tags", "memory.explain", "embed.mint",
    }
    for event in events:
        assert event["space_id"] == w.a
        assert event["principal_id"] == issuer
        assert event["principal_type"] == ptype
        assert event["detail"]["on_behalf_of"] == VIEWER
        assert event["detail"]["embed_token_id"] == TOKEN_ID
    dumped = json.dumps(events)
    assert "xyzzy" not in dumped and "plugh" not in dumped


async def test_denied_write_audit_names_the_viewer(world: SpaceWorld):
    w = world
    with embed(w, "rita", "user", "reader"):
        await w.handler.memory_store("Nope")
    [event] = w.audit.events
    assert event["action"] == "access.denied"
    assert event["detail"]["on_behalf_of"] == VIEWER
    assert event["detail"]["embed_token_id"] == TOKEN_ID


async def test_no_embed_keys_in_audit_without_a_token(world: SpaceWorld):
    w = world
    await w.alice.call("memory_store", "Plain write", space=w.a)
    [event] = w.audit.events
    assert "on_behalf_of" not in event["detail"] and "embed_token_id" not in event["detail"]
