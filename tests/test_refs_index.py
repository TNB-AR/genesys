"""Ref index, external-id index and storage-level tag/ref search filters (spec 5.5, 9.2; D09)."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeEmbedder, principal

from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, Visibility
from papez.models.node import MemoryNode
from papez.models.ref import Ref
from papez.storage.memory import InMemoryGraphProvider, space_bucket_key

ORG = "org_r"
A = "spc_a"
B = "spc_b"
DOC = Ref(type="document", id="doc-1")
MEET = Ref(type="meeting", id="m-7")


def _n(text: str, **kw: object) -> MemoryNode:
    return MemoryNode(content_summary=text, content_full=text, **kw)  # type: ignore[arg-type]


def _supersede(new: MemoryNode, old: MemoryNode) -> MemoryEdge:
    """How memory_amend links an amendment: SUPERSEDES directed new (source) -> old (target)."""
    return MemoryEdge(source_id=new.id, target_id=old.id, type=EdgeType.SUPERSEDES, weight=1.0)


def _ids(nodes: list[MemoryNode]) -> set[str]:
    return {str(n.id) for n in nodes}


@pytest.mark.asyncio
async def test_find_nodes_by_ref() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        a1 = _n("a1", space_id=A, refs=[DOC, MEET])
        p1 = _n("p1", refs=[DOC])
        other = _n("other", refs=[Ref(type="document", id="doc-2")])
        for n in (a1, p1, other):
            await graph.create_node(n)
    with principal("bob", orgs=[ORG], roles={B: "owner"}):
        b1 = _n("b1", space_id=B, refs=[DOC])
        q_private = _n("q private", refs=[DOC])
        q_org = _n("q org", refs=[DOC], visibility=Visibility.ORG, org_id=ORG)
        for n in (b1, q_private, q_org):
            await graph.create_node(n)

    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        assert _ids(await graph.find_nodes_by_ref("document", "doc-1")) == _ids([a1, p1, q_org])
        assert _ids(await graph.find_nodes_by_ref("meeting", "m-7")) == _ids([a1])
        assert await graph.find_nodes_by_ref("document", "nope") == []
        assert await graph.find_nodes_by_ref("meeting", "doc-1") == []  # same id, other type
        # explicit scope parameters
        assert _ids(await graph.find_nodes_by_ref("document", "doc-1", org_ids=[], space_ids=[A])) == _ids([a1, p1])
    with principal("bob", orgs=[ORG], roles={B: "owner"}):
        assert _ids(await graph.find_nodes_by_ref("document", "doc-1")) == _ids([b1, q_private, q_org])
    with principal("svc_a", orgs=[ORG], roles={A: "writer"}, principal_type="service"):
        assert _ids(await graph.find_nodes_by_ref("document", "doc-1")) == _ids([a1])


@pytest.mark.asyncio
async def test_ref_index_follows_update_delete_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    graph = InMemoryGraphProvider(persist_path=str(path))
    with principal("alice", roles={A: "owner"}):
        node = _n("a1", space_id=A, refs=[DOC])
        await graph.create_node(node)
        nid = str(node.id)
        # refs given as plain dicts (as a tool might pass them) are coerced to Ref
        await graph.update_node(nid, {"refs": [{"type": "meeting", "id": "m-7"}]})
        assert isinstance(graph._user_nodes[space_bucket_key(A)][nid].refs[0], Ref)
        assert await graph.find_nodes_by_ref("document", "doc-1") == []
        assert _ids(await graph.find_nodes_by_ref("meeting", "m-7")) == {nid}
        assert (space_bucket_key(A), "document", "doc-1") not in graph._idx_by_ref

        reloaded = InMemoryGraphProvider(persist_path=str(path))
        reloaded._load()
        assert _ids(await reloaded.find_nodes_by_ref("meeting", "m-7")) == {nid}

        await graph.delete_node(nid)
        assert await graph.find_nodes_by_ref("meeting", "m-7") == []
        assert graph._idx_by_ref == {}


@pytest.mark.asyncio
async def test_external_id_index_follows_amendment() -> None:
    graph = InMemoryGraphProvider()
    with principal("svc_a", roles={A: "writer"}, principal_type="service"):
        old = _n("v1", space_id=A, external_id="crm-42")
        await graph.create_node(old)
        assert str((await graph.find_node_by_external_id("crm-42", A)).id) == str(old.id)  # type: ignore[union-attr]

        new = _n("v2", space_id=A, external_id="crm-42")
        await graph.create_node(new)
        await graph.create_edge(_supersede(new, old))
        found = await graph.find_node_by_external_id("crm-42", A)
        assert found is not None and str(found.id) == str(new.id)
        # the superseded node keeps its external_id for history
        assert graph._user_nodes[space_bucket_key(A)][str(old.id)].external_id == "crm-42"
        assert graph._idx_by_external_id[(space_bucket_key(A), "crm-42")] == str(new.id)

        # amended into a node WITHOUT the external id: nothing current remains
        newest = _n("v3", space_id=A)
        await graph.create_node(newest)
        await graph.create_edge(_supersede(newest, new))
        assert await graph.find_node_by_external_id("crm-42", A) is None

        # other spaces / personal never match
        assert await graph.find_node_by_external_id("crm-42", B) is None
        assert await graph.find_node_by_external_id("crm-42", None) is None


@pytest.mark.asyncio
async def test_external_id_index_after_delete_and_update() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", roles={A: "owner"}):
        old = _n("v1", space_id=A, external_id="e-1")
        new = _n("v2", space_id=A, external_id="e-1")
        await graph.create_node(old)
        await graph.create_node(new)
        await graph.create_edge(_supersede(new, old))
        await graph.delete_node(str(new.id))  # takes the SUPERSEDES edge with it
        found = await graph.find_node_by_external_id("e-1", A)
        assert found is not None and str(found.id) == str(old.id)

        await graph.update_node(str(old.id), {"external_id": "e-2"})
        assert await graph.find_node_by_external_id("e-1", A) is None
        found = await graph.find_node_by_external_id("e-2", A)
        assert found is not None and str(found.id) == str(old.id)
        assert (space_bucket_key(A), "e-1") not in graph._idx_by_external_id


@pytest.mark.asyncio
async def test_external_id_personal_bucket_is_per_user_and_scoped() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice", orgs=[ORG], roles={A: "owner"}):
        mine = _n("alice ext", external_id="e-1", visibility=Visibility.ORG, org_id=ORG)
        await graph.create_node(mine)
        found = await graph.find_node_by_external_id("e-1", None)
        assert found is not None and str(found.id) == str(mine.id)
    with principal("bob", orgs=[ORG]):
        # bob can see alice's org node but it is not in HIS personal bucket
        assert await graph.find_node_by_external_id("e-1", None) is None
        assert await graph.find_node_by_external_id("e-1", A) is None
    with principal("alice", roles={A: "owner"}):
        space_node = _n("space ext", space_id=A, external_id="e-1")
        await graph.create_node(space_node)
    with principal("bob", roles={B: "owner"}):
        assert await graph.find_node_by_external_id("e-1", A) is None  # not in bob's scope
    with principal("alice", roles={A: "owner"}, include_personal=False):
        assert await graph.find_node_by_external_id("e-1", None) is None
        found = await graph.find_node_by_external_id("e-1", A)
        assert found is not None and str(found.id) == str(space_node.id)


@pytest.mark.asyncio
async def test_search_filters_apply_before_top_k() -> None:
    emb = FakeEmbedder()
    graph = InMemoryGraphProvider()
    query = "quarterly revenue forecast"
    q = await emb.embed(query)
    with principal("alice", roles={A: "owner"}):
        # 20 near-perfect untagged matches would fill any k before a post-filter
        for i in range(20):
            await graph.create_node(_n(f"quarterly revenue forecast {i}", embedding=q, space_id=A))
        tagged = []
        for i in range(3):
            text = f"unrelated revenue note {i}"
            n = _n(text, embedding=await emb.embed(text), space_id=A, tags=["finance"], refs=[DOC])
            tagged.append(n)
            await graph.create_node(n)

        hits = await graph.vector_search(q, k=3, tags_any=["finance", "other"])
        assert _ids([n for n, _ in hits]) == _ids(tagged)
        hits = await graph.vector_search(q, k=3, ref_type="document")
        assert _ids([n for n, _ in hits]) == _ids(tagged)
        assert await graph.vector_search(q, k=3, tags_any=["finance"], ref_type="meeting") == []
        assert len(await graph.vector_search(q, k=3, tags_any=[])) == 3  # empty filter = no filter

        kw = await graph.keyword_search("revenue", k=3, tags_any=["finance"])
        assert _ids(kw) == _ids(tagged)
        kw = await graph.keyword_search("revenue", k=2, ref_type="document")
        assert len(kw) == 2 and _ids(kw) <= _ids(tagged)
        assert await graph.keyword_search("revenue", k=3, tags_any=["nope"]) == []


@pytest.mark.asyncio
async def test_create_node_twice_reindexes() -> None:
    graph = InMemoryGraphProvider()
    with principal("alice"):
        node = _n("x", refs=[DOC], external_id="e-1")
        await graph.create_node(node)
        node2 = node.model_copy(update={"refs": [MEET], "external_id": "e-2"})
        await graph.create_node(node2)  # same id: replaces, indexes follow
        assert await graph.find_nodes_by_ref("document", "doc-1") == []
        assert _ids(await graph.find_nodes_by_ref("meeting", "m-7")) == {str(node.id)}
        assert await graph.find_node_by_external_id("e-1", None) is None
        assert await graph.find_node_by_external_id("e-2", None) is not None
