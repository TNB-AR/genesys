"""Spec 5.5: state files written before refs/tags/spaces load unchanged (uses the WS-L1 fixtures)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import principal

from papez.models.enums import Visibility
from papez.models.node import MemoryNode
from papez.storage.memory import InMemoryGraphProvider, space_bucket_key

FIXTURES = Path(__file__).parent / "fixtures"
FLAT_IDS = {"6f1d2c3a-0b4e-4a5f-9c7d-1e2f3a4b5c6d", "7a2e3d4b-1c5f-4b6a-8d8e-2f3a4b5c6d7e"}
PRIVATE_ID = "9c4a5f6d-3e7b-4d8c-af0a-4b5c6d7e8f90"
ORG_ID_NODE = "ad5b6a7e-4f8c-4e9d-b01b-5c6d7e8f9001"


def _copy(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "state.json"
    shutil.copy(FIXTURES / name, path)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", ["old_state_v1.json", "old_state_per_user.json"])
async def test_load_old_state_file(tmp_path: Path, fixture: str) -> None:
    path = _copy(tmp_path, fixture)
    graph = InMemoryGraphProvider(persist_path=str(path))
    uid = "migrated-user" if fixture == "old_state_v1.json" else "legacy-user"
    with principal(uid, orgs=["org-legacy"]):
        await graph.initialize(uid)
        nodes = graph._user_nodes[uid]
        assert len(nodes) == 2
        for n in nodes.values():
            assert n.refs == [] and n.tags == []
            assert n.space_id is None and n.external_id is None and n.promoted_from is None
        # Indexes were rebuilt: bucket map, edges and the (empty) node indexes.
        assert all(graph._node_bucket[nid] == uid for nid in nodes)
        assert graph._idx_by_ref == {} and graph._idx_by_external_id == {}
        for nid in nodes:
            assert await graph.get_node(nid) is not None
        assert len(await graph.get_all_edges()) == 1
        assert await graph.space_bucket_ids() == []
        # Round trip: saving writes the new fields with their defaults, and loads again.
        await graph.update_node(next(iter(nodes)), {"pinned": True})
    saved = json.loads(path.read_text())
    assert list(saved) == [uid]
    for v in saved[uid]["nodes"].values():
        assert v["refs"] == [] and v["tags"] == [] and v["space_id"] is None
    again = InMemoryGraphProvider(persist_path=str(path))
    again._load()
    assert set(again._user_nodes[uid]) == set(nodes)


@pytest.mark.asyncio
async def test_old_flat_file_migrates_to_current_user(tmp_path: Path) -> None:
    graph = InMemoryGraphProvider(persist_path=str(_copy(tmp_path, "old_state_v1.json")))
    with principal("migrated-user"):
        graph._load()
        assert set(graph._user_nodes["migrated-user"]) == FLAT_IDS
        edges = await graph.get_edges(sorted(FLAT_IDS)[0], "both")
        assert len(edges) == 1


@pytest.mark.asyncio
async def test_old_per_user_file_keeps_org_visibility(tmp_path: Path) -> None:
    graph = InMemoryGraphProvider(persist_path=str(_copy(tmp_path, "old_state_per_user.json")))
    graph._load()
    with principal("colleague", orgs=["org-legacy"]):
        assert await graph.get_node(ORG_ID_NODE) is not None
        assert await graph.get_node(PRIVATE_ID) is None
        # the legacy edge joins the org node to a private node: hidden from the colleague
        assert await graph.get_edges(ORG_ID_NODE, "both") == []
    with principal("svc", orgs=["org-legacy"], principal_type="service"):
        assert await graph.get_node(ORG_ID_NODE) is None
    node = graph._user_nodes["legacy-user"][ORG_ID_NODE]
    assert node.visibility == Visibility.ORG and node.org_id == "org-legacy"


@pytest.mark.asyncio
async def test_load_skips_reserved_keys_and_tolerates_partial_buckets(tmp_path: Path) -> None:
    node = MemoryNode(content_summary="space note", space_id="spc_x")
    personal = MemoryNode(content_summary="personal note")
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "__papez__": {"version": 2, "nodes": {"bogus": {}}},
        space_bucket_key("spc_x"): {"nodes": {str(node.id): node.model_dump(mode="json")}},  # no "edges"
        "alice": {"nodes": {str(personal.id): personal.model_dump(mode="json")}, "edges": []},
        "edges-only-user": {"edges": []},  # no "nodes"
    }))
    graph = InMemoryGraphProvider(persist_path=str(path))
    graph._load()
    assert "__papez__" not in graph._user_nodes
    assert graph._node_bucket[str(node.id)] == space_bucket_key("spc_x")
    assert await graph.space_bucket_ids() == ["spc_x"]
    with principal("alice", roles={"spc_x": "reader"}):
        assert await graph.get_node(str(node.id)) is not None
        assert await graph.get_node(str(personal.id)) is not None
