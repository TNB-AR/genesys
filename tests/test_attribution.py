"""on_behalf_of: viewer validation, storage round trips, output and the MCP boundary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import WORLD_ORG, SpaceWorld, principal

from papez import server as srv
from papez.mcp.tools import MCPToolHandler
from papez.models.attribution import validate_viewer
from papez.models.node import MemoryNode
from papez.models.ref import ValidationFailure
from papez.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider

VIEWER = {"id": "u_123", "email": "pat@example.com", "name": "Pat Lee"}
ATTRIBUTED = {**VIEWER, "via": "api"}


# -- validate_viewer -----------------------------------------------------------


def test_validate_viewer_keeps_known_keys_only():
    assert validate_viewer({**VIEWER, "role": "admin", "via": "embed"}) == VIEWER


def test_validate_viewer_optional_fields_default_to_none():
    assert validate_viewer({"id": "u"}) == {"id": "u", "email": None, "name": None}
    assert validate_viewer({"id": "u", "email": None, "name": None}) == {"id": "u", "email": None, "name": None}


@pytest.mark.parametrize("value", [
    None, "u_123", 7, [], {}, {"email": "a@b.co"},
    {"id": ""}, {"id": 7}, {"id": None}, {"id": "x" * 129},
    {"id": "u", "email": "not-an-email"}, {"id": "u", "email": "a@b"}, {"id": "u", "email": "a b@c.de"},
    {"id": "u", "email": ""}, {"id": "u", "email": 7}, {"id": "u", "email": f"{'a' * 250}@b.co"},
    {"id": "u", "name": 7}, {"id": "u", "name": "x" * 121},
])
def test_validate_viewer_rejects(value: Any):
    with pytest.raises(ValidationFailure) as exc:
        validate_viewer(value)
    assert exc.value.code == "invalid_viewer"


def test_validate_viewer_limits_are_inclusive():
    assert validate_viewer({"id": "x" * 128, "name": "y" * 120})["id"] == "x" * 128
    email = f"{'a' * 249}@b.co"
    assert len(email) == 254
    assert validate_viewer({"id": "u", "email": email})["email"] == email


# -- writes --------------------------------------------------------------------


async def _node(world: SpaceWorld, node_id: str) -> MemoryNode:
    with principal("alice", orgs=[WORLD_ORG], roles=await world.alice.roles()):
        node = await world.graph.get_node(node_id)
    assert node is not None
    return node


async def test_store_round_trip_and_default(space_world: SpaceWorld):
    w = space_world
    with_viewer = await w.svc.call("memory_store", "CRM note on renewal", space=w.a, on_behalf_of=VIEWER)
    assert (await _node(w, with_viewer["node_id"])).on_behalf_of == ATTRIBUTED
    assert (await _node(w, with_viewer["node_id"])).original_user_id == "svc_a"
    plain = await w.svc.call("memory_store", "CRM sync note", space=w.a)
    assert (await _node(w, plain["node_id"])).on_behalf_of is None


async def test_store_refuses_a_bad_viewer_before_writing(space_world: SpaceWorld):
    w = space_world
    before = (await w.alice.call("memory_stats", space=w.a))["nodes"]
    for bad in ({"email": "a@b.co"}, {"id": "u", "via": "phone"}, "u_123"):
        out = await w.svc.call("memory_store", "Should not land", space=w.a, on_behalf_of=bad)
        assert out["code"] == "invalid_viewer"
    assert (await w.alice.call("memory_stats", space=w.a))["nodes"] == before


async def test_amend_uses_its_own_value_and_does_not_inherit(space_world: SpaceWorld):
    w = space_world
    first = await w.svc.call("memory_store", "Original claim", space=w.a, on_behalf_of=VIEWER)
    plain = await w.svc.call("memory_amend", first["node_id"], "Corrected claim")
    assert (await _node(w, plain["node_id"])).on_behalf_of is None
    other = {"id": "u_9", "name": "Sam"}
    second = await w.svc.call("memory_amend", plain["node_id"], "Corrected again", on_behalf_of=other)
    assert (await _node(w, second["node_id"])).on_behalf_of == {"id": "u_9", "email": None, "name": "Sam", "via": "api"}
    assert (await _node(w, first["node_id"])).on_behalf_of == ATTRIBUTED
    bad = await w.svc.call("memory_amend", second["node_id"], "No", on_behalf_of={"name": "no id"})
    assert bad["code"] == "invalid_viewer"


async def test_external_id_amend_does_not_inherit_and_identical_ignores_it(space_world: SpaceWorld):
    w = space_world
    first = await w.svc.call("memory_store", "Doc v1", space=w.a, external_id="doc", on_behalf_of=VIEWER)
    # Same content with another viewer: nothing changed, the stored viewer stays.
    same = await w.svc.call(
        "memory_store", "Doc v1", space=w.a, external_id="doc", on_behalf_of={"id": "someone_else"},
    )
    assert (same["node_id"], same["status"]) == (first["node_id"], "unchanged")
    assert (await _node(w, first["node_id"])).on_behalf_of == ATTRIBUTED
    # A real change without a viewer: the new node has none.
    changed = await w.svc.call("memory_store", "Doc v2", space=w.a, external_id="doc")
    assert changed["status"] == "amended"
    assert (await _node(w, changed["node_id"])).on_behalf_of is None


async def test_batch_applies_to_every_item_and_items_cannot_set_it(space_world: SpaceWorld):
    w = space_world
    out = await w.svc.call(
        "memory_store_batch", [{"content": "One"}, {"content": "Two"}], space=w.a, on_behalf_of=VIEWER,
    )
    for row in out["results"]:
        assert (await _node(w, row["node_id"])).on_behalf_of == ATTRIBUTED
    refused = await w.svc.call(
        "memory_store_batch", [{"content": "Three", "on_behalf_of": VIEWER}], space=w.a,
    )
    assert refused["code"] == "batch_invalid"
    bad = await w.svc.call("memory_store_batch", [{"content": "Four"}], space=w.a, on_behalf_of={"id": ""})
    assert bad["code"] == "invalid_viewer"


async def test_retire_amends_with_the_new_value(space_world: SpaceWorld):
    w = space_world
    sole = await w.svc.call(
        "memory_store", "Sole source memo", space=w.a, refs=[{"type": "doc", "id": "gone"}], on_behalf_of=VIEWER,
    )
    other = {"id": "u_2"}
    assert await w.svc.call("memory_retire_by_ref", "doc", "gone", space=w.a, on_behalf_of=other) == {
        "retired": 1, "updated": 0, "deleted": 0,
    }
    hit = await w.alice.call("memory_detail", sole["node_id"])
    new_id = hit["memory"]["superseded_by"]
    assert (await _node(w, new_id)).on_behalf_of == {"id": "u_2", "email": None, "name": None, "via": "api"}
    bad = await w.svc.call("memory_retire_by_ref", "doc", "gone", space=w.a, on_behalf_of=7)
    assert bad["code"] == "invalid_viewer"


async def test_persistence_round_trip(tmp_path: Path):
    path = tmp_path / "state.json"
    graph = InMemoryGraphProvider(persist_path=str(path))
    handler = MCPToolHandler(graph=graph, embeddings=None, cache=InMemoryCacheProvider())
    with principal("u1", orgs=[]):
        stored = await handler.memory_store("Attributed fact", on_behalf_of=VIEWER)
        plain = await handler.memory_store("Unattributed fact")
    reloaded = InMemoryGraphProvider(persist_path=str(path))
    with principal("u1", orgs=[]):
        await reloaded.initialize("u1")
        assert (await reloaded.get_node(stored["node_id"])).on_behalf_of == ATTRIBUTED
        assert (await reloaded.get_node(plain["node_id"])).on_behalf_of is None


def test_old_state_without_the_field_loads():
    node = MemoryNode.model_validate({"content_summary": "from an old file"})
    assert node.on_behalf_of is None


# -- output --------------------------------------------------------------------


async def test_shown_in_full_recall_search_and_explain(space_world: SpaceWorld):
    w = space_world
    stored = await w.svc.call("memory_store", "Renewal blocked by legal", space=w.a, category="note", on_behalf_of=VIEWER)
    nid = stored["node_id"]

    recall = await w.alice.call("memory_recall", "renewal legal", space=w.a)
    hit = next(r for r in recall["results"] if r["id"] == nid)
    assert hit["on_behalf_of"] == ATTRIBUTED
    assert hit["created_by"] == "svc_a"
    assert hit["category"] == "note"

    search = await w.alice.call("memory_search", "Renewal blocked by legal", space=w.a)
    row = next(r for r in search["results"] if r["id"] == nid)
    assert row["on_behalf_of"] == ATTRIBUTED and row["created_by"] == "svc_a"

    explain = await w.alice.call("memory_explain", nid)
    assert explain["on_behalf_of"] == ATTRIBUTED and explain["created_by"] == "svc_a"


async def test_concise_recall_is_unchanged(space_world: SpaceWorld):
    w = space_world
    await w.svc.call("memory_store", "Renewal blocked by legal", space=w.a, on_behalf_of=VIEWER)
    recall = await w.alice.call("memory_recall", "renewal legal", space=w.a, verbosity="concise")
    assert recall["results"]
    for hit in recall["results"]:
        assert set(hit) <= {
            "id", "summary", "status", "score", "activation", "is_core", "tags", "ref_count", "space",
            "superseded_by", "low_confidence",
        }


# -- the MCP boundary ----------------------------------------------------------


@pytest.mark.parametrize("name", ["memory_store", "memory_store_batch", "memory_amend", "memory_retire_by_ref"])
def test_mcp_cannot_set_on_behalf_of(name: str):
    definition = next(d for d in srv._TOOL_DEFS if d["name"] == name)
    assert "on_behalf_of" not in json.dumps(definition)
    _, required, optional = srv._TOOL_DISPATCH[name]
    assert "on_behalf_of" not in required and "on_behalf_of" not in optional


async def test_on_behalf_of_passed_through_call_tool_is_dropped(space_world: SpaceWorld, monkeypatch: Any):
    w = space_world
    seen: dict[str, Any] = {}

    async def spy(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"ok": True}

    _, required, optional = srv._TOOL_DISPATCH["memory_store"]
    monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_store", (spy, required, optional))
    await srv.call_tool("memory_store", {"content": "x", "space": w.a, "on_behalf_of": VIEWER})
    assert seen["content"] == "x"
    assert "on_behalf_of" not in seen


@pytest.mark.parametrize("viewer", [
    {"id": "a\x00b"},
    {"id": "a\ud800"},
    {"id": "x", "name": "a\x00b"},
    {"id": "x", "name": "\udfff"},
    {"id": "x", "email": "a\x00@b.co"},
    {"id": "x", "email": "a\ud800@b.co"},
    {"id": "x", "email": "a@b.co\n"},
])
def test_viewer_fields_storage_cannot_keep_are_refused(viewer):
    with pytest.raises(ValidationFailure) as exc:
        validate_viewer(viewer)
    assert exc.value.code == "invalid_viewer"


def test_other_control_characters_stay_data():
    viewer = validate_viewer({"id": "a\r\nb", "name": "tab\there\x1b[31m"})
    assert viewer["id"] == "a\r\nb" and viewer["name"] == "tab\there\x1b[31m"
