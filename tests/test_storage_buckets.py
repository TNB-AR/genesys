"""Bucket mechanics of InMemoryGraphProvider: persistence, edge placement, node-scope degree,
visible-set stats/orphans, dangling-edge cleanup, direction vocabularies, atomic()."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from conftest import principal

from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.storage.memory import InMemoryGraphProvider, space_bucket_key

ORG = "org_r"
A = "spc_a"


def _n(text: str, **kw: object) -> MemoryNode:
    return MemoryNode(content_summary=text, content_full=text, **kw)  # type: ignore[arg-type]


def _e(src: MemoryNode, tgt: MemoryNode, etype: EdgeType = EdgeType.SUPPORTS) -> MemoryEdge:
    return MemoryEdge(source_id=src.id, target_id=tgt.id, type=etype)


def test_space_bucket_key_is_module_level() -> None:
    assert space_bucket_key("spc_1") == "space::spc_1"


@pytest.mark.asyncio
async def test_state_file_skips_empty_buckets(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    graph = InMemoryGraphProvider(persist_path=str(path))
    space_node = _n("space note", space_id=A)
    with principal("svc_a", roles={A: "writer"}, principal_type="service"):
        await graph.create_node(space_node)  # lands in space::spc_a, not in svc_a's bucket
        # reads by a service (and the create-on-read `nodes` property) must not litter the file
        _ = graph.nodes
        await graph.get_stats()
        await graph.get_orphans()
        await graph.update_node(str(space_node.id), {"pinned": True})
    with principal("alice"):
        _ = graph.nodes
        _ = graph.edges
        await graph.update_node(str(space_node.id), {"pinned": False})  # invisible: no-op
    saved = json.loads(path.read_text())
    assert list(saved) == [space_bucket_key(A)]
    assert saved[space_bucket_key(A)]["nodes"][str(space_node.id)]["pinned"] is True
    assert "svc_a" in graph._user_nodes  # in memory the create-on-read bucket still exists

    again = InMemoryGraphProvider(persist_path=str(path))
    again._load()
    assert await again.space_bucket_ids() == [A]
    with principal("alice", roles={A: "reader"}):
        assert (await again.get_node(str(space_node.id))) is not None


@pytest.mark.asyncio
async def test_state_file_keeps_bucket_with_only_edges(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    graph = InMemoryGraphProvider(persist_path=str(path))
    with principal("bob", orgs=[ORG]):
        org_node = _n("org note", visibility=Visibility.ORG, org_id=ORG)
        await graph.create_node(org_node)
    with principal("alice", orgs=[ORG]):
        mine = _n("mine")
        await graph.create_node(mine)
        await graph.delete_node(str(mine.id))
        # alice has no nodes left but writes an edge between bob's org node and itself
        await graph.create_edge(_e(org_node, org_node, EdgeType.RELATED_TO))
    saved = json.loads(path.read_text())
    assert saved["alice"]["nodes"] == {} and len(saved["alice"]["edges"]) == 1


@pytest.mark.asyncio
async def test_create_node_without_space_refused_in_space_pass() -> None:
    graph = InMemoryGraphProvider()
    with principal(space_bucket_key(A), roles={A: "owner"}, include_personal=False):
        with pytest.raises(ValueError, match="space bucket"):
            await graph.create_node(_n("derived without space id"))
        nid = await graph.create_node(_n("derived with space id", space_id=A))
        assert graph._node_bucket[nid] == space_bucket_key(A)


@pytest.mark.asyncio
async def test_personal_edge_to_unknown_id_keeps_legacy_semantics() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice"):
        mine = _n("mine")
        await graph.create_node(mine)
        ghost = MemoryEdge(source_id=mine.id, target_id=uuid.uuid4(), type=EdgeType.RELATED_TO)
        eid = await graph.create_edge(ghost)
        assert graph._edge_bucket[eid] == "alice"
        # degree counts the caller's bucket, as before; the dangling edge is not "visible"
        assert await graph.get_degree(str(mine.id)) == 1
        assert await graph.get_edges(str(mine.id), "both") == []
        await graph.delete_edge(eid)  # own-bucket edges are always deletable
        assert graph._user_edges["alice"] == []


@pytest.mark.asyncio
async def test_degree_and_orphan_are_node_scoped() -> None:
    graph = InMemoryGraphProvider()
    with principal("bob", orgs=[ORG]):
        org_node = _n("bob org note", visibility=Visibility.ORG, org_id=ORG)
        bob_private = _n("bob private")
        await graph.create_node(org_node)
        await graph.create_node(bob_private)
        await graph.create_edge(_e(org_node, bob_private, EdgeType.CAUSED_BY))
    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        mine = _n("alice note")
        s1, s2, s3 = _n("s1", space_id=A), _n("s2", space_id=A), _n("s3", space_id=A)
        for n in (mine, s1, s2, s3):
            await graph.create_node(n)
        await graph.create_edge(_e(mine, org_node, EdgeType.RELATED_TO))  # in alice's bucket
        await graph.create_edge(_e(s1, s2, EdgeType.SUPPORTS))  # in space::spc_a
        await graph.create_edge(_e(s1, s3, EdgeType.CONTRADICTS))

        # org node: counted from its OWN bucket (bob's), not the caller's (D04, C08)
        assert await graph.get_degree(str(org_node.id)) == 1
        assert await graph.get_causal_weight(str(org_node.id)) == 1
        assert await graph.get_supportive_degree(str(org_node.id)) == 1
        assert await graph.is_orphan(str(org_node.id)) is False
        # alice's own node counts alice's bucket
        assert await graph.get_degree(str(mine.id)) == 1
        # space nodes count the space bucket only
        assert await graph.get_degree(str(s1.id)) == 2
        assert await graph.get_supportive_degree(str(s1.id)) == 1
        assert await graph.get_causal_weight(str(s1.id)) == 1
        assert await graph.is_orphan(str(s3.id)) is True  # CONTRADICTS is not supportive
        orphans = {str(n.id) for n in await graph.get_orphans()}
        assert orphans == {str(s3.id)}
        # invisible node: counts nothing
        assert await graph.get_degree(str(bob_private.id)) == 0
        assert await graph.is_orphan(str(bob_private.id)) is True

    with principal("svc_a", orgs=[ORG], roles={A: "writer"}, principal_type="service"):
        assert {str(n.id) for n in await graph.get_orphans()} == {str(s3.id)}
        assert await graph.get_degree(str(org_node.id)) == 0
        assert await graph.get_degree(str(s1.id)) == 2


@pytest.mark.asyncio
async def test_get_stats_counts_visible_set() -> None:
    graph = InMemoryGraphProvider()
    with principal("bob", orgs=[ORG]):
        org_node = _n("bob org", visibility=Visibility.ORG, org_id=ORG)
        hidden = _n("bob hidden")
        await graph.create_node(org_node)
        await graph.create_node(hidden)
        await graph.create_edge(_e(org_node, hidden, EdgeType.CAUSED_BY))  # one end hidden from alice
    with principal("carol", roles={"spc_b": "owner"}):
        await graph.create_node(_n("other space", space_id="spc_b"))
    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        mine1, mine2 = _n("m1"), _n("m2")
        s1, s2 = _n("s1", space_id=A), _n("s2", space_id=A)
        for n in (mine1, mine2, s1, s2):
            await graph.create_node(n)
        await graph.create_edge(_e(mine1, mine2, EdgeType.CAUSED_BY))
        await graph.create_edge(_e(mine2, org_node, EdgeType.CAUSED_BY))
        await graph.create_edge(_e(s1, s2, EdgeType.SUPPORTS))
        stats = await graph.get_stats()
        assert stats == {
            "nodes": 5, "edges": 3, "max_causal_weight": 2,
            "total_nodes": 5, "total_edges": 3,
        }
    with principal("alice", orgs=[ORG], roles={A: "owner"}, include_personal=False):
        stats = await graph.get_stats()
        assert stats["nodes"] == 2 and stats["edges"] == 1 and stats["max_causal_weight"] == 1
    with principal("bob", orgs=[ORG]):
        assert (await graph.get_stats())["nodes"] == 2


@pytest.mark.asyncio
async def test_get_nodes_by_status_visible_set() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", roles={A: "owner"}):
        await graph.create_node(_n("mine", status=MemoryStatus.CORE))
        await graph.create_node(_n("space", space_id=A, status=MemoryStatus.CORE))
    with principal("bob"):
        await graph.create_node(_n("bob", status=MemoryStatus.CORE))
        assert len(await graph.get_nodes_by_status(MemoryStatus.CORE)) == 1
    with principal("alice", roles={A: "owner"}):
        assert len(await graph.get_nodes_by_status(MemoryStatus.CORE)) == 2
        assert len(await graph.get_nodes_by_status(MemoryStatus.CORE, limit=1)) == 1


@pytest.mark.asyncio
async def test_get_edges_accepts_both_direction_vocabularies() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice"):
        a, b, c = _n("a"), _n("b"), _n("c")
        for n in (a, b, c):
            await graph.create_node(n)
        out_edge = await graph.create_edge(_e(a, b))
        in_edge = await graph.create_edge(_e(c, a))
        for word in ("out", "outgoing"):
            assert [str(e.id) for e in await graph.get_edges(str(a.id), word)] == [out_edge]
        for word in ("in", "incoming"):
            assert [str(e.id) for e in await graph.get_edges(str(a.id), word)] == [in_edge]
        assert {str(e.id) for e in await graph.get_edges(str(a.id), "both")} == {out_edge, in_edge}
        assert await graph.get_edges(str(a.id), "both", EdgeType.CAUSED_BY) == []


@pytest.mark.asyncio
async def test_org_node_edges_from_other_buckets_are_returned() -> None:
    graph = InMemoryGraphProvider()
    with principal("bob", orgs=[ORG]):
        org_node = _n("bob org", visibility=Visibility.ORG, org_id=ORG)
        bob_org2 = _n("bob org 2", visibility=Visibility.ORG, org_id=ORG)
        await graph.create_node(org_node)
        await graph.create_node(bob_org2)
        bob_edge = await graph.create_edge(_e(org_node, bob_org2))
    with principal("alice", orgs=[ORG]):
        mine = _n("mine")
        await graph.create_node(mine)
        alice_edge = await graph.create_edge(_e(mine, org_node))
        ids = {str(e.id) for e in await graph.get_edges(str(org_node.id), "both")}
        assert ids == {bob_edge, alice_edge}
        assert {str(e.id) for e in await graph.get_all_edges([str(org_node.id)])} == {bob_edge, alice_edge}
        assert {str(e.id) for e in await graph.get_all_edges()} == {bob_edge, alice_edge}
        # both ends are visible to alice, so she may reweight an edge stored in bob's bucket
        await graph.update_edge_weight(bob_edge, 0.2)
    assert next(e for e in graph._user_edges["bob"] if str(e.id) == bob_edge).weight == 0.2
    with principal("eve"):
        await graph.delete_edge(bob_edge)  # eve sees neither end: no-op
    assert any(str(e.id) == bob_edge for e in graph._user_edges["bob"])


@pytest.mark.asyncio
async def test_delete_node_removes_dangling_edges_in_every_bucket() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", orgs=[ORG]):
        org_node = _n("alice org", visibility=Visibility.ORG, org_id=ORG)
        await graph.create_node(org_node)
    with principal("bob", orgs=[ORG]):
        mine = _n("bob note")
        await graph.create_node(mine)
        bob_edge = await graph.create_edge(_e(mine, org_node))
    with principal("alice", orgs=[ORG]):
        await graph.delete_node(str(org_node.id))
    assert graph._user_edges["bob"] == []
    assert bob_edge not in graph._edge_bucket
    assert str(org_node.id) not in graph._node_bucket
    with principal("bob", orgs=[ORG]):
        assert await graph.get_degree(str(mine.id)) == 0


@pytest.mark.asyncio
async def test_destroy_removes_bucket_indexes_and_foreign_edges() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        org_node = _n("alice org", visibility=Visibility.ORG, org_id=ORG, external_id="x-1")
        space_node = _n("space", space_id=A)
        await graph.create_node(org_node)
        await graph.create_node(space_node)
    with principal("bob", orgs=[ORG]):
        mine = _n("bob note")
        await graph.create_node(mine)
        await graph.create_edge(_e(mine, org_node))
    await graph.destroy("alice")
    assert "alice" not in graph._user_nodes
    assert graph._user_edges["bob"] == []
    assert str(org_node.id) not in graph._node_bucket
    assert ("alice", "x-1") not in graph._idx_by_external_id
    # destroy(user) never touches the space bucket
    assert str(space_node.id) in graph._user_nodes[space_bucket_key(A)]


@pytest.mark.asyncio
async def test_atomic_batches_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = InMemoryGraphProvider(persist_path=str(tmp_path / "s.json"))
    calls: list[int] = []
    real = graph._do_save
    monkeypatch.setattr(graph, "_do_save", lambda: (calls.append(1), real())[1])
    with principal("alice", roles={A: "writer"}):
        async with graph.atomic():
            a, b = _n("a", space_id=A), _n("b", space_id=A)
            await graph.create_node(a)
            await graph.create_node(b)
            await graph.create_edge(_e(a, b))
            assert calls == []
        assert calls == [1]
        async with graph.atomic():
            async with graph.atomic():  # nests like defer_saves
                await graph.update_node(str(a.id), {"pinned": True})
            assert calls == [1]
        assert calls == [1, 1]
    assert hasattr(graph, "atomic")


@pytest.mark.asyncio
async def test_rebuild_indexes_restores_bucket_maps() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", roles={A: "owner"}):
        a, b = _n("a", space_id=A), _n("b", space_id=A)
        await graph.create_node(a)
        await graph.create_node(b)
        eid = await graph.create_edge(_e(a, b))
    node_bucket = dict(graph._node_bucket)
    edge_bucket = dict(graph._edge_bucket)
    graph._node_bucket.clear()
    graph._edge_bucket.clear()
    graph._rebuild_indexes()
    assert graph._node_bucket == node_bucket and graph._edge_bucket == edge_bucket
    assert edge_bucket[eid] == space_bucket_key(A)
