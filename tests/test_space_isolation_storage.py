"""Storage half of spec 13.2: space isolation enforced by InMemoryGraphProvider itself.

Every protocol read method is called directly on the provider under
``as_principal`` as alice (owner of A), bob (owner of B) and svc_a (service,
writer in A). Space A, space B and both users' personal buckets are seeded with
near-identical content, shared entity names, tags and refs, so any missing
visibility check leaks.
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import FakeEmbedder, principal

from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.models.ref import Ref
from papez.storage.memory import InMemoryGraphProvider, space_bucket_key

ORG = "org_r"
A = "spc_a"
B = "spc_b"
SHARED_REF = Ref(type="document", id="doc-1", label="Pilot plan")


@dataclass
class World:
    graph: InMemoryGraphProvider
    ids: dict[str, str] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    query_embedding: list[float] = field(default_factory=list)

    def of(self, *names: str) -> set[str]:
        return {self.ids[n] for n in names}

    @property
    def all_ids(self) -> list[str]:
        return list(self.ids.values())


# who -> (principal kwargs, visible node names, forbidden node names)
PRINCIPALS: dict[str, dict[str, Any]] = {
    "alice": {"orgs": [ORG], "roles": {A: "owner"}},
    "bob": {"orgs": [ORG], "roles": {B: "owner"}},
    "svc_a": {"orgs": [ORG], "roles": {A: "writer"}, "principal_type": "service"},
}
VISIBLE = {
    "alice": {"a1", "a2", "a3", "p1", "p2", "q2"},
    "bob": {"b1", "b2", "b3", "q1", "q2"},
    "svc_a": {"a1", "a2", "a3"},
}
ALL_NAMES = {"a1", "a2", "a3", "b1", "b2", "b3", "p1", "p2", "q1", "q2"}


def as_who(who: str, **overrides: Any) -> Any:
    kwargs = {**PRINCIPALS[who], **overrides}
    return principal(who, **kwargs)


def _node(content: str, *, space_id: str | None = None, author: str, external_id: str | None = None,
          visibility: Visibility = Visibility.PRIVATE, org_id: str | None = None,
          embedding: list[float] | None = None) -> MemoryNode:
    return MemoryNode(
        content_summary=content[:200],
        content_full=content,
        embedding=embedding,
        entity_refs=["Bank A", "Pilot"],
        tags=["pilot", "bank-a"],
        refs=[SHARED_REF],
        space_id=space_id,
        external_id=external_id,
        visibility=visibility,
        org_id=org_id,
        original_user_id=author,
    )


def _edge(src: str, tgt: str, etype: EdgeType = EdgeType.CAUSED_BY) -> MemoryEdge:
    return MemoryEdge(source_id=uuid.UUID(src), target_id=uuid.UUID(tgt), type=etype, reason="Bank A pilot link")


async def build_world() -> World:
    emb = FakeEmbedder()
    graph = InMemoryGraphProvider()
    w = World(graph=graph)
    specs: list[tuple[str, str, str, dict[str, Any]]] = [
        # name, author, content, extra
        ("a1", "alice", "Bank A pilot starts in Q1", {"space_id": A, "external_id": "ext-1"}),
        ("a2", "svc_a", "Bank A pilot budget approved", {"space_id": A}),
        ("a3", "alice", "Bank A pilot risk log", {"space_id": A}),
        ("b1", "bob", "Bank A pilot starts in Q2", {"space_id": B, "external_id": "ext-1"}),
        ("b2", "bob", "Bank A pilot budget approved", {"space_id": B}),
        ("b3", "bob", "Bank A pilot risk log", {"space_id": B}),
        ("p1", "alice", "Bank A pilot notes", {"external_id": "ext-1"}),
        ("p2", "alice", "Bank A pilot budget notes", {}),
        ("q1", "bob", "Bank A pilot notes", {"external_id": "ext-1"}),
        ("q2", "bob", "Bank A pilot org update", {"visibility": Visibility.ORG, "org_id": ORG}),
    ]
    for name, author, content, extra in specs:
        node = _node(content, author=author, embedding=await emb.embed(content), **extra)
        w.ids[name] = str(node.id)
        owner = author if author in ("alice", "bob") else "alice"
        with as_who(owner):
            await graph.create_node(node)
    i = w.ids
    with as_who("alice"):
        w.edges["a"] = await graph.create_edge(_edge(i["a2"], i["a1"]))
        w.edges["p"] = await graph.create_edge(_edge(i["p2"], i["p1"], EdgeType.SUPPORTS))
        # alice's own edge onto bob's org node lives in alice's bucket
        w.edges["pq"] = await graph.create_edge(_edge(i["p1"], i["q2"], EdgeType.RELATED_TO))
    with as_who("bob"):
        w.edges["b"] = await graph.create_edge(_edge(i["b2"], i["b1"]))
        # org node linked to bob's PRIVATE node: alice may see q2 but never q1
        w.edges["q"] = await graph.create_edge(_edge(i["q2"], i["q1"], EdgeType.RELATED_TO))
    w.query_embedding = await emb.embed("Bank A pilot starts in Q2")  # B's content: the best match
    return w


def _ends(edges: list[MemoryEdge]) -> set[str]:
    return {str(e.source_id) for e in edges} | {str(e.target_id) for e in edges}


ReadCall = Callable[[World], Awaitable[set[str]]]


async def r_get_node(w: World) -> set[str]:
    return {nid for nid in w.all_ids if await w.graph.get_node(nid) is not None}


async def r_get_nodes_by_status(w: World) -> set[str]:
    return {str(n.id) for n in await w.graph.get_nodes_by_status(MemoryStatus.ACTIVE, limit=1000)}


async def r_get_edges(w: World) -> set[str]:
    seen: set[str] = set()
    for nid in w.all_ids:
        for direction in ("both", "out", "outgoing", "in", "incoming"):
            seen |= _ends(await w.graph.get_edges(nid, direction))
    return seen


async def r_get_all_edges(w: World) -> set[str]:
    return _ends(await w.graph.get_all_edges()) | _ends(await w.graph.get_all_edges(w.all_ids))


async def r_traverse(w: World) -> set[str]:
    seen: set[str] = set()
    for nid in w.all_ids:
        seen |= {str(n.id) for n in await w.graph.traverse(nid, 10)}
    return seen


async def r_get_connecting_edges(w: World) -> set[str]:
    return _ends(await w.graph.get_connecting_edges(w.all_ids))


async def r_get_causal_chain(w: World) -> set[str]:
    seen: set[str] = set()
    for nid in w.all_ids:
        for direction in ("upstream", "downstream"):
            chain = await w.graph.get_causal_chain(nid, direction)
            seen |= {str(n.id) for n in chain}
            if chain:
                seen.add(nid)
    return seen


async def r_get_causal_chains_batch(w: World) -> set[str]:
    seen: set[str] = set()
    for direction in ("upstream", "downstream"):
        for seed, chain in (await w.graph.get_causal_chains_batch(w.all_ids, direction)).items():
            seen |= {str(n.id) for n in chain}
            if chain:
                seen.add(seed)
    return seen


async def r_vector_search(w: World) -> set[str]:
    return {str(n.id) for n, _ in await w.graph.vector_search(w.query_embedding, k=100)}


async def r_keyword_search(w: World) -> set[str]:
    return {str(n.id) for n in await w.graph.keyword_search("Bank pilot starts Q2", k=100)}


async def r_find_nodes_by_ref(w: World) -> set[str]:
    return {str(n.id) for n in await w.graph.find_nodes_by_ref("document", "doc-1")}


async def r_find_node_by_external_id(w: World) -> set[str]:
    seen: set[str] = set()
    for sid in (A, B, None):
        node = await w.graph.find_node_by_external_id("ext-1", sid)
        if node is not None:
            seen.add(str(node.id))
    return seen


async def r_get_stats(w: World) -> set[str]:
    # A count-only method: report every node whose existence the counts reveal.
    # Under a correct implementation the counts equal the visible set exactly.
    stats = await w.graph.get_stats()
    visible = {nid for nid in w.all_ids if await w.graph.get_node(nid) is not None}
    edges = await w.graph.get_all_edges()
    if stats["nodes"] != len(visible) or stats["edges"] != len(edges):
        return set(w.all_ids)
    assert stats["total_nodes"] == stats["nodes"] and stats["total_edges"] == stats["edges"]
    return visible


async def r_get_orphans(w: World) -> set[str]:
    return {str(n.id) for n in await w.graph.get_orphans()}


async def r_is_orphan(w: World) -> set[str]:
    return {nid for nid in w.all_ids if not await w.graph.is_orphan(nid)}


async def r_get_degree(w: World) -> set[str]:
    return {nid for nid in w.all_ids if await w.graph.get_degree(nid) > 0}


async def r_get_supportive_degree(w: World) -> set[str]:
    return {nid for nid in w.all_ids if await w.graph.get_supportive_degree(nid) > 0}


async def r_get_causal_weight(w: World) -> set[str]:
    return {nid for nid in w.all_ids if await w.graph.get_causal_weight(nid) > 0}


# Every read method of GraphStorageProvider. Adding a read method means adding it here.
READ_PATHS: dict[str, ReadCall] = {
    "get_node": r_get_node,
    "get_nodes_by_status": r_get_nodes_by_status,
    "get_edges": r_get_edges,
    "get_all_edges": r_get_all_edges,
    "traverse": r_traverse,
    "get_connecting_edges": r_get_connecting_edges,
    "get_causal_chain": r_get_causal_chain,
    "get_causal_chains_batch": r_get_causal_chains_batch,
    "vector_search": r_vector_search,
    "keyword_search": r_keyword_search,
    "find_nodes_by_ref": r_find_nodes_by_ref,
    "find_node_by_external_id": r_find_node_by_external_id,
    "get_stats": r_get_stats,
    "get_orphans": r_get_orphans,
    "is_orphan": r_is_orphan,
    "get_degree": r_get_degree,
    "get_supportive_degree": r_get_supportive_degree,
    "get_causal_weight": r_get_causal_weight,
}


def test_read_path_table_covers_protocol() -> None:
    """Guard: every read-shaped method on the protocol is in READ_PATHS."""
    from papez.storage.base import GraphStorageProvider

    writes_and_lifecycle = {
        "initialize", "destroy", "create_node", "update_node", "delete_node", "create_edge",
        "update_edge_weight", "validate_edge", "delete_edge", "edge_exists", "promote_to_org",
        "atomic_reactivation_update", "erase_user", "space_bucket_ids", "erase_space",
    }
    protocol_methods = {
        name for name in vars(GraphStorageProvider)
        if not name.startswith("_") and callable(getattr(GraphStorageProvider, name))
    }
    assert protocol_methods - writes_and_lifecycle == set(READ_PATHS)


@pytest.mark.asyncio
@pytest.mark.parametrize("who", sorted(PRINCIPALS))
@pytest.mark.parametrize("method", sorted(READ_PATHS))
async def test_every_read_path_respects_scope(method: str, who: str) -> None:
    w = await build_world()
    forbidden = w.of(*(ALL_NAMES - VISIBLE[who]))
    with as_who(who):
        seen = await READ_PATHS[method](w)
    leaked = {name for name, nid in w.ids.items() if nid in seen and nid in forbidden}
    assert not leaked, f"{method} as {who} leaked {sorted(leaked)}"
    assert seen, f"{method} as {who} returned nothing at all (test would be vacuous)"
    # The caller's own space is reachable through every path.
    own_space = {"alice": "a", "svc_a": "a", "bob": "b"}[who]
    assert seen & w.of(*(n for n in VISIBLE[who] if n.startswith(own_space))), method


@pytest.mark.asyncio
async def test_service_never_sees_personal_even_with_include_personal_true() -> None:
    w = await build_world()
    with as_who("svc_a", include_personal=True):
        for method, call in READ_PATHS.items():
            seen = await call(w)
            assert not seen & w.of("p1", "p2", "q1", "q2"), method


@pytest.mark.asyncio
async def test_user_with_include_personal_false_sees_only_scoped_spaces() -> None:
    w = await build_world()
    with as_who("alice", include_personal=False):
        for method, call in READ_PATHS.items():
            seen = await call(w)
            assert seen <= w.of("a1", "a2", "a3"), method


@pytest.mark.asyncio
async def test_explicit_space_ids_param_narrows_scope() -> None:
    w = await build_world()
    with as_who("alice"):
        hits = {str(n.id) for n, _ in await w.graph.vector_search(w.query_embedding, k=100, space_ids=[])}
        assert hits == w.of("p1", "p2", "q2")
        kw = {str(n.id) for n in await w.graph.keyword_search("Bank pilot", k=100, space_ids=[], org_ids=[])}
        assert kw == w.of("p1", "p2")
        assert await w.graph.traverse(w.ids["a1"], 3, space_ids=[]) == []
        assert await w.graph.find_nodes_by_ref("document", "doc-1", space_ids=[A], org_ids=[]) != []
        refs = {str(n.id) for n in await w.graph.find_nodes_by_ref("document", "doc-1", space_ids=[], org_ids=[])}
        assert refs == w.of("p1", "p2")


@pytest.mark.asyncio
async def test_missing_identity_fails_closed() -> None:
    w = await build_world()
    with pytest.raises(RuntimeError):
        await w.graph.get_node(w.ids["a1"])
    with pytest.raises(RuntimeError):
        await w.graph.vector_search(w.query_embedding)


@pytest.mark.asyncio
async def test_vector_and_keyword_search_no_leak() -> None:
    w = await build_world()
    with as_who("alice"):
        vec = [str(n.id) for n, _ in await w.graph.vector_search(w.query_embedding, k=3)]
        kw = [str(n.id) for n in await w.graph.keyword_search("Bank pilot starts Q2", k=3)]
    # B's node is the best match but never appears; k results still come back (filter before top-k).
    assert len(vec) == 3 and len(kw) == 3
    assert not (set(vec) | set(kw)) & w.of("b1", "b2", "b3", "q1")


@pytest.mark.asyncio
async def test_get_node_by_id_from_other_space() -> None:
    w = await build_world()
    with as_who("alice"):
        assert await w.graph.get_node(w.ids["b1"]) is None
        assert await w.graph.get_node(w.ids["q1"]) is None
        assert await w.graph.get_node(w.ids["a1"]) is not None
    with as_who("bob"):
        assert await w.graph.get_node(w.ids["a1"]) is None
    with as_who("svc_a"):
        assert await w.graph.get_node(w.ids["p1"]) is None
        assert await w.graph.get_node(w.ids["q2"]) is None


# -- Edges stay inside a space (spec 6.6, D04) ---------------------------------


@pytest.mark.asyncio
async def test_cross_space_edge_refused_in_storage() -> None:
    w = await build_world()
    before = {k: list(v) for k, v in w.graph._user_edges.items()}
    for who in ("alice", "bob", "svc_a"):
        with as_who(who):
            with pytest.raises(ValueError, match="^cross_space_edge$"):
                await w.graph.create_edge(_edge(w.ids["a1"], w.ids["b1"]))
            with pytest.raises(ValueError, match="^cross_space_edge$"):
                await w.graph.create_edge(_edge(w.ids["b1"], w.ids["a1"]))
    assert {k: list(v) for k, v in w.graph._user_edges.items()} == before


@pytest.mark.asyncio
async def test_space_to_personal_edge_refused() -> None:
    w = await build_world()
    missing = str(uuid.uuid4())
    with as_who("alice"):
        for src, tgt in (("a1", "p1"), ("p1", "a1"), ("a1", "q2"), ("q2", "a1")):
            with pytest.raises(ValueError, match="^cross_space_edge$"):
                await w.graph.create_edge(_edge(w.ids[src], w.ids[tgt]))
        # a missing endpoint paired with a space node is refused too
        with pytest.raises(ValueError, match="^cross_space_edge$"):
            await w.graph.create_edge(_edge(w.ids["a1"], missing))
        # ...but personal -> missing keeps the legacy semantics (caller's bucket)
        eid = await w.graph.create_edge(_edge(w.ids["p1"], missing, EdgeType.RELATED_TO))
    assert any(str(e.id) == eid for e in w.graph._user_edges["alice"])
    assert not await _edge_exists_somewhere(w, w.ids["a1"], w.ids["p1"])


async def _edge_exists_somewhere(w: World, src: str, tgt: str) -> bool:
    return any(
        str(e.source_id) == src and str(e.target_id) == tgt
        for edges in w.graph._user_edges.values() for e in edges
    )


@pytest.mark.asyncio
async def test_same_space_edge_goes_to_space_bucket() -> None:
    w = await build_world()
    with as_who("svc_a"):
        eid = await w.graph.create_edge(_edge(w.ids["a3"], w.ids["a1"], EdgeType.SUPPORTS))
        assert await w.graph.edge_exists(w.ids["a3"], w.ids["a1"], EdgeType.SUPPORTS)
    assert any(str(e.id) == eid for e in w.graph._user_edges[space_bucket_key(A)])
    assert "svc_a" not in w.graph._user_edges or not w.graph._user_edges["svc_a"]
    with as_who("alice"):
        # alice sees the edge svc_a wrote, because both ends are in her scope
        assert eid in {str(e.id) for e in await w.graph.get_edges(w.ids["a1"], "in")}
        assert await w.graph.edge_exists(w.ids["a3"], w.ids["a1"], EdgeType.SUPPORTS)
    with as_who("bob"):
        assert not await w.graph.edge_exists(w.ids["a3"], w.ids["a1"], EdgeType.SUPPORTS)


@pytest.mark.asyncio
async def test_traverse_cannot_cross() -> None:
    w = await build_world()
    # Simulate corrupt/legacy data: an A->B edge injected straight into A's bucket,
    # bypassing create_edge. Visibility must still stop the walk.
    rogue = _edge(w.ids["a1"], w.ids["b1"], EdgeType.RELATED_TO)
    w.graph._user_edges[space_bucket_key(A)].append(rogue)
    w.graph._index_edge(space_bucket_key(A), rogue)
    with as_who("alice"):
        reached = {str(n.id) for n in await w.graph.traverse(w.ids["a1"], 10)}
        assert reached == w.of("a1", "a2")
        assert w.ids["b1"] not in _ends(await w.graph.get_edges(w.ids["a1"], "both"))
        assert w.ids["b1"] not in _ends(await w.graph.get_connecting_edges([w.ids["a1"], w.ids["b1"]]))
        # traverse from an invisible start returns nothing (C18)
        assert await w.graph.traverse(w.ids["b1"], 10) == []
        assert await w.graph.get_causal_chain(w.ids["b1"], "upstream") == []
        assert (await w.graph.get_causal_chains_batch([w.ids["b2"]], "downstream"))[w.ids["b2"]] == []
    with as_who("bob"):
        reached = {str(n.id) for n in await w.graph.traverse(w.ids["b1"], 10)}
        assert reached == w.of("b1", "b2")


@pytest.mark.asyncio
async def test_org_node_edges_require_both_ends_visible() -> None:
    w = await build_world()
    with as_who("alice"):
        ends = _ends(await w.graph.get_edges(w.ids["q2"], "both"))
        assert w.ids["q1"] not in ends  # bob's private node stays hidden
        assert w.ids["p1"] in ends      # alice's own edge onto the org node
    with as_who("bob"):
        ends = _ends(await w.graph.get_edges(w.ids["q2"], "both"))
        assert w.ids["q1"] in ends and w.ids["p1"] not in ends


# -- Erasure (D32 storage half; spec 13.2) --------------------------------------


@pytest.mark.asyncio
async def test_erase_user_keeps_space_memories() -> None:
    w = await build_world()
    a_edges_before = list(w.graph._user_edges[space_bucket_key(A)])
    with as_who("alice"):
        result = await w.graph.erase_user("alice")
    assert set(result) == {"nodes_deleted", "nodes_anonymized", "edges_deleted", "edges_scrubbed",
                           "space_nodes_anonymized"}
    assert result["nodes_deleted"] == 2          # p1, p2
    assert result["nodes_anonymized"] == 0
    assert result["space_nodes_anonymized"] == 2  # a1, a3 written by alice; a2 by svc_a
    a_bucket = w.graph._user_nodes[space_bucket_key(A)]
    assert set(a_bucket) == w.of("a1", "a2", "a3")
    assert a_bucket[w.ids["a1"]].original_user_id == "erased_user"
    assert a_bucket[w.ids["a3"]].original_user_id == "erased_user"
    assert a_bucket[w.ids["a2"]].original_user_id == "svc_a"
    # edge reasons in space buckets are never scrubbed or removed
    assert w.graph._user_edges[space_bucket_key(A)] == a_edges_before
    assert a_edges_before[0].reason == "Bank A pilot link"
    with as_who("svc_a"):
        assert {str(n.id) for n in await w.graph.get_nodes_by_status(MemoryStatus.ACTIVE)} == w.of("a1", "a2", "a3")
    with as_who("bob"):
        # bob's bucket lost alice's edge onto q2 only through alice's own bucket; his edges stay
        assert w.ids["q1"] in _ends(await w.graph.get_edges(w.ids["q2"], "both"))
    assert "alice" not in w.graph._user_nodes


@pytest.mark.asyncio
async def test_erase_space() -> None:
    w = await build_world()
    g = w.graph
    with as_who("alice"):
        await g.create_edge(_edge(w.ids["a3"], w.ids["a2"], EdgeType.RELATED_TO))
    b_nodes = dict(g._user_nodes[space_bucket_key(B)])
    b_edges = list(g._user_edges[space_bucket_key(B)])

    dry = await g.erase_space(A, dry_run=True)
    assert dry == {"nodes": 3, "edges": 2}
    assert set(g._user_nodes[space_bucket_key(A)]) == w.of("a1", "a2", "a3")
    assert len(g._user_edges[space_bucket_key(A)]) == 2
    assert sorted(await g.space_bucket_ids()) == [A, B]

    real = await g.erase_space(A)
    assert real == dry
    assert space_bucket_key(A) not in g._user_nodes and space_bucket_key(A) not in g._user_edges
    assert await g.space_bucket_ids() == [B]
    assert not any(nid in g._node_bucket for nid in w.of("a1", "a2", "a3"))
    assert not any(k[0] == space_bucket_key(A) for k in g._idx_by_ref)
    assert not any(k[0] == space_bucket_key(A) for k in g._idx_by_external_id)
    assert space_bucket_key(A) not in g._idx_by_source and space_bucket_key(A) not in g._idx_by_target
    # B untouched
    assert g._user_nodes[space_bucket_key(B)] == b_nodes
    assert g._user_edges[space_bucket_key(B)] == b_edges
    with as_who("alice"):
        assert await g.get_node(w.ids["a1"]) is None
        assert await g.find_node_by_external_id("ext-1", A) is None
        assert {str(n.id) for n in await g.find_nodes_by_ref("document", "doc-1")} == w.of("p1", "p2", "q2")
    # erasing again, or an unknown space, is a harmless zero
    assert await g.erase_space(A) == {"nodes": 0, "edges": 0}


@pytest.mark.asyncio
async def test_erase_space_removes_foreign_edges_touching_its_nodes() -> None:
    w = await build_world()
    g = w.graph
    # A legacy edge in alice's bucket pointing at an A node (cannot be created via create_edge).
    rogue = _edge(w.ids["p1"], w.ids["a1"], EdgeType.RELATED_TO)
    g._user_edges["alice"].append(rogue)
    g._index_edge("alice", rogue)
    assert (await g.erase_space(A, dry_run=True))["edges"] == 2
    await g.erase_space(A)
    assert str(rogue.id) not in {str(e.id) for e in g._user_edges["alice"]}
    assert str(rogue.id) not in g._edge_bucket


# -- Write-path twin (C01, D01 item 6) -------------------------------------------


@pytest.mark.asyncio
async def test_write_paths_act_on_visible_space_nodes_only() -> None:
    w = await build_world()
    g = w.graph
    ts = datetime.now(UTC)
    b_validated_before = next(e for e in g._user_edges[space_bucket_key(B)] if str(e.id) == w.edges["b"]).last_validated_at
    with as_who("svc_a"):
        await g.update_node(w.ids["a1"], {"pinned": True})
        await g.update_node(w.ids["b1"], {"pinned": True})
        await g.update_node(w.ids["p1"], {"pinned": True})
        await g.atomic_reactivation_update(w.ids["a2"], ts, 0.5)
        await g.atomic_reactivation_update(w.ids["b2"], ts, 0.5)
        await g.store_embedding(w.ids["a3"], [1.0] * 16)
        await g.store_embedding(w.ids["b3"], [1.0] * 16)
        await g.update_edge_weight(w.edges["a"], 0.11)
        await g.update_edge_weight(w.edges["b"], 0.11)
        await g.validate_edge(w.edges["b"])
        with pytest.raises(ValueError, match="invalid_arguments"):
            await g.promote_to_org(w.ids["a1"], ORG)
        await g.promote_to_org(w.ids["b1"], ORG)
        await g.delete_node(w.ids["b3"])
        await g.delete_node(w.ids["p2"])
        await g.delete_edge(w.edges["b"])
        await g.delete_edge(w.edges["p"])
    nodes = {name: g._user_nodes[g._node_bucket[nid]][nid] for name, nid in w.ids.items()}
    assert nodes["a1"].pinned is True
    assert nodes["b1"].pinned is False and nodes["p1"].pinned is False
    assert nodes["a2"].reactivation_count == 1 and nodes["a2"].stability == 1.5
    assert nodes["b2"].reactivation_count == 0
    assert nodes["a3"].embedding == [1.0] * 16
    assert nodes["b3"].embedding != [1.0] * 16
    assert nodes["a1"].visibility == Visibility.PRIVATE and nodes["a1"].org_id is None
    assert nodes["b1"].visibility == Visibility.PRIVATE and nodes["b1"].org_id is None
    edge_a = next(e for e in g._user_edges[space_bucket_key(A)] if str(e.id) == w.edges["a"])
    edge_b = next(e for e in g._user_edges[space_bucket_key(B)] if str(e.id) == w.edges["b"])
    assert edge_a.weight == 0.11 and edge_b.weight != 0.11
    assert edge_b.last_validated_at == b_validated_before
    assert w.edges["p"] in {str(e.id) for e in g._user_edges["alice"]}

    with as_who("svc_a"):
        await g.delete_node(w.ids["a1"])  # visible space node: deleted with its edges
        await g.delete_edge(w.edges["a"])  # already gone with a1: harmless
    assert w.ids["a1"] not in g._user_nodes[space_bucket_key(A)]
    assert not g._user_edges[space_bucket_key(A)]
    assert w.ids["b3"] in g._user_nodes[space_bucket_key(B)]
    assert w.ids["p2"] in g._user_nodes["alice"]


@pytest.mark.asyncio
async def test_promote_to_org_acts_on_visible_personal_node() -> None:
    w = await build_world()
    with as_who("bob"):
        await w.graph.promote_to_org(w.ids["p1"], ORG)  # alice's private node: invisible to bob
    assert w.graph._user_nodes["alice"][w.ids["p1"]].visibility == Visibility.PRIVATE
    with as_who("alice"):
        await w.graph.promote_to_org(w.ids["p1"], ORG)
    node = w.graph._user_nodes["alice"][w.ids["p1"]]
    assert node.visibility == Visibility.ORG and node.org_id == ORG and node.original_user_id == "alice"
    with as_who("bob"):
        assert await w.graph.get_node(w.ids["p1"]) is not None


@pytest.mark.asyncio
async def test_update_node_refuses_space_move() -> None:
    w = await build_world()
    with as_who("alice"):
        with pytest.raises(ValueError, match="immutable"):
            await w.graph.update_node(w.ids["a1"], {"space_id": B})
        await w.graph.update_node(w.ids["a1"], {"space_id": A, "pinned": True})  # unchanged: fine
    assert w.graph._user_nodes[space_bucket_key(A)][w.ids["a1"]].pinned is True
