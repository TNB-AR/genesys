"""Mind Spaces tools (spec 6.8): behaviour, shapes and the spec 6.3 role matrix."""
from __future__ import annotations

from typing import Any

import pytest
from conftest import WORLD_ORG, SpaceWorld, build_space_world, principal

from papez.mcp.tools import MCPToolHandler
from papez.models.enums import PrincipalType, SpaceRole
from papez.models.space import SpaceMember
from papez.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider
from papez.storage.spaces import InMemorySpaceRegistry

SPACE_KEYS = {"id", "slug", "name", "org_id", "description", "forgetting_policy", "llm_inference_enabled",
              "archived", "archived_at", "created_at", "created_by"}


async def _world_with_roles() -> SpaceWorld:
    """space_world plus rita (reader) and walt (writer) in A; nora in the org, not a member."""
    w = await build_space_world()
    for uid, role in (("rita", SpaceRole.READER), ("walt", SpaceRole.WRITER)):
        await w.registry.add_member(SpaceMember(
            space_id=w.a, principal_id=uid, principal_type=PrincipalType.USER, role=role, added_by="alice",
        ))
    return w


ACTORS = {"reader": "rita", "writer": "walt", "owner": "alice", "non_member": "nora"}
RANK = {"reader": 0, "writer": 1, "owner": 2}


def _calls(w: SpaceWorld) -> dict[str, tuple[str, Any]]:
    """tool -> (minimum role, coroutine factory taking the actor)."""
    return {
        "memory_recall": ("reader", lambda a: a.call("memory_recall", "pilot", space=w.a)),
        "memory_search": ("reader", lambda a: a.call("memory_search", "pilot", space=w.a)),
        "memory_traverse": ("reader", lambda a: a.call("memory_traverse", w.ids["a_q1"], space=w.a)),
        "memory_stats": ("reader", lambda a: a.call("memory_stats", space=w.a)),
        "list_core_memories": ("reader", lambda a: a.call("list_core_memories", space=w.a)),
        "space_get": ("reader", lambda a: a.call("space_get", w.a)),
        "space_member_list": ("reader", lambda a: a.call("space_member_list", w.a)),
        "memory_store": ("writer", lambda a: a.call("memory_store", "matrix write", space=w.a)),
        "memory_amend": ("writer", lambda a: a.call("memory_amend", w.ids["a_q1"], "matrix amend")),
        "pin_memory": ("writer", lambda a: a.call("pin_memory", w.ids["a_q1"])),
        "unpin_memory": ("writer", lambda a: a.call("unpin_memory", w.ids["a_q1"])),
        "delete_memory": ("writer", lambda a: a.call("delete_memory", w.ids["a_ctx"])),
        "space_update": ("owner", lambda a: a.call("space_update", w.a, description="d")),
        "space_member_add": ("owner", lambda a: a.call("space_member_add", w.a, "zed", "user", "reader")),
        "space_member_remove": ("owner", lambda a: a.call("space_member_remove", w.a, "svc_a")),
        "space_archive": ("owner", lambda a: a.call("space_archive", w.a)),
        "space_unarchive": ("owner", lambda a: a.call("space_unarchive", w.a)),
        "space_erase_dry_run": ("owner", lambda a: a.call("space_erase", w.a, "space-a", dry_run=True)),
    }


MATRIX_TOOLS = [
    "memory_recall", "memory_search", "memory_traverse", "memory_stats", "list_core_memories", "space_get",
    "space_member_list", "memory_store", "memory_amend", "pin_memory", "unpin_memory", "delete_memory",
    "space_update", "space_member_add", "space_member_remove", "space_archive", "space_unarchive",
    "space_erase_dry_run",
]


@pytest.mark.parametrize("who", list(ACTORS))
@pytest.mark.parametrize("tool", MATRIX_TOOLS)
async def test_role_matrix(tool: str, who: str):
    """Spec 6.3: allowed at or above the minimum role; insufficient_role below; non-members not_found."""
    w = await _world_with_roles()
    need, factory = _calls(w)[tool]
    result = await factory(w.actor(ACTORS[who]))
    if who == "non_member":
        # node-addressed tools can't see the node at all; space-addressed tools can't see the space
        assert result.get("code") in ("space_not_found", "not_found"), result
    elif RANK[who] >= RANK[need]:
        assert "error" not in result, (tool, who, result)
    else:
        assert result.get("code") == "insufficient_role", (tool, who, result)


@pytest.mark.parametrize("who,ok", [("reader", False), ("writer", True), ("owner", True), ("non_member", False)])
async def test_promote_to_space_into_role_matrix(who: str, ok: bool):
    w = await _world_with_roles()
    actor = w.actor(ACTORS[who])
    personal = await actor.call("memory_store", f"{who} personal memory to promote")
    result = await actor.call("promote_to_space", personal["node_id"], w.a)
    if ok:
        assert result["space"]["id"] == w.a
    else:
        assert result["code"] == ("insufficient_role" if who == "reader" else "space_not_found")


async def test_space_erase_owner_only_real():
    w = await _world_with_roles()
    assert (await w.actor("walt").call("space_erase", w.a, "space-a"))["code"] == "insufficient_role"
    assert (await w.alice.call("space_erase", w.a, "space-a"))["erased"]["members"] == 4


# -- space_create -------------------------------------------------------------------


async def _bare_handler() -> MCPToolHandler:
    return MCPToolHandler(
        graph=InMemoryGraphProvider(), embeddings=None, cache=InMemoryCacheProvider(),
        spaces=InMemorySpaceRegistry(),
    )


async def test_space_create_shape_and_defaults():
    h = await _bare_handler()
    with principal("alice", orgs=[WORLD_ORG]):
        res = await h.space_create("Client A — Payments")
    space = res["space"]
    assert set(space) == SPACE_KEYS | {"your_role"}
    assert space["slug"] == "client-a-payments"
    assert space["org_id"] == WORLD_ORG
    assert space["forgetting_policy"] == "retain_all"
    assert space["llm_inference_enabled"] is True
    assert space["archived"] is False and space["archived_at"] is None
    assert space["your_role"] == "owner" and space["created_by"] == "alice"
    assert space["id"].startswith("spc_")
    members = await h.spaces.list_members(space["id"])
    assert [(m.principal_id, m.role) for m in members] == [("alice", SpaceRole.OWNER)]


async def test_space_create_slug_suffix_and_taken():
    h = await _bare_handler()
    with principal("alice", orgs=[WORLD_ORG]):
        first = await h.space_create("Payments")
        second = await h.space_create("Payments")
        third = await h.space_create("Payments")
        taken = await h.space_create("Other", slug="payments")
        bad = await h.space_create("Other", slug="Not A Slug")
    assert [first["space"]["slug"], second["space"]["slug"], third["space"]["slug"]] == [
        "payments", "payments-2", "payments-3"]
    assert taken["code"] == "slug_taken"
    assert bad["code"] == "invalid_arguments"


class _AsyncSlugRegistry:
    """Hosted-registry shape: wraps the in-memory registry, with slugs_for_org
    and erased_at as coroutine functions (as on PostgresSpaceRegistry)."""

    def __init__(self) -> None:
        self.inner = InMemorySpaceRegistry()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def slugs_for_org(self, org_id: str) -> set[str]:
        return self.inner.slugs_for_org(org_id)

    async def erased_at(self, space_id: str) -> str | None:
        return self.inner.erased_at(space_id)


async def test_space_create_derives_slug_with_async_slugs_for_org():
    """Hosted MCP space_create without a slug raised TypeError on an async slugs_for_org."""
    h = MCPToolHandler(
        graph=InMemoryGraphProvider(), embeddings=None, cache=InMemoryCacheProvider(),
        spaces=_AsyncSlugRegistry(),
    )
    with principal("alice", orgs=[WORLD_ORG]):
        first = await h.space_create("Name")
        second = await h.space_create("Name")
    assert [first["space"]["slug"], second["space"]["slug"]] == ["name", "name-2"]


async def test_space_create_org_rules():
    h = await _bare_handler()
    with principal("alice", orgs=["org_1", "org_2"]):
        assert (await h.space_create("X"))["code"] == "invalid_arguments"
        assert (await h.space_create("X", org_id="org_3"))["code"] == "invalid_arguments"
        assert (await h.space_create("X", org_id="org_2"))["space"]["org_id"] == "org_2"
    with principal("solo"):
        assert (await h.space_create("X"))["code"] == "invalid_arguments"
    with principal("alice", orgs=[WORLD_ORG]):
        assert (await h.space_create(""))["code"] == "invalid_arguments"
        assert (await h.space_create("x" * 121))["code"] == "invalid_arguments"
        assert (await h.space_create("X", forgetting_policy="never"))["code"] == "invalid_arguments"
        std = await h.space_create("Std", forgetting_policy="standard", llm_inference_enabled=False)
    assert std["space"]["forgetting_policy"] == "standard"
    assert std["space"]["llm_inference_enabled"] is False


async def test_space_tools_without_registry():
    h = MCPToolHandler(graph=InMemoryGraphProvider(), embeddings=None, cache=InMemoryCacheProvider())
    with principal("alice", orgs=[WORLD_ORG]):
        assert (await h.space_create("X"))["code"] == "invalid_arguments"
        assert (await h.space_list())["code"] == "invalid_arguments"
        assert (await h.memory_recall("x", space="anything"))["code"] == "space_not_found"
        # no registry: personal memory keeps working exactly as before
        assert (await h.memory_store("still works"))["status"] == "stored"


# -- space_list / space_get / space_update ------------------------------------------------


async def test_space_list_sorted_with_counts(space_world: SpaceWorld):
    w = space_world
    await w.alice.call("space_create", "Aardvark", slug="aardvark")
    listed = await w.alice.call("space_list")
    assert [s["name"] for s in listed["spaces"]] == ["Aardvark", "Space A"]
    a = listed["spaces"][1]
    assert a["your_role"] == "owner" and a["member_count"] == 2 and a["memory_count"] == 2
    await w.alice.call("space_archive", "aardvark")
    assert [s["name"] for s in (await w.alice.call("space_list"))["spaces"]] == ["Space A"]
    assert len((await w.alice.call("space_list", include_archived=True))["spaces"]) == 2
    svc_list = await w.svc.call("space_list")
    assert [(s["id"], s["your_role"]) for s in svc_list["spaces"]] == [(w.a, "writer")]


async def test_space_get_by_id_and_slug(space_world: SpaceWorld):
    w = space_world
    by_id = await w.alice.call("space_get", w.a)
    by_slug = await w.alice.call("space_get", "space-a")
    assert by_id == by_slug
    assert by_id["space"]["member_count"] == 2 and by_id["space"]["memory_count"] == 2
    assert by_id["space"]["your_role"] == "owner"
    # a slug only resolves within the caller's own orgs
    outsider = w.actor("eve", orgs=["org_other"])
    assert (await outsider.call("space_get", "space-a"))["code"] == "space_not_found"


async def test_space_update(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("space_update", w.a, name="Renamed", description="desc",
                             forgetting_policy="standard", llm_inference_enabled=False)
    s = res["space"]
    assert (s["name"], s["description"], s["forgetting_policy"], s["llm_inference_enabled"], s["slug"]) == (
        "Renamed", "desc", "standard", False, "space-a")
    assert (await w.alice.call("space_update", w.a, forgetting_policy="bogus"))["code"] == "invalid_arguments"
    assert (await w.alice.call("space_update", w.a, name=""))["code"] == "invalid_arguments"


# -- members ----------------------------------------------------------------------------


async def test_member_add_created_flag_and_shape(space_world: SpaceWorld):
    w = space_world
    added = await w.alice.call("space_member_add", w.a, "carol", "user", "reader")
    assert added["created"] is True
    assert set(added["member"]) == {"principal_id", "principal_type", "role", "added_by", "added_at"}
    assert (added["member"]["role"], added["member"]["added_by"]) == ("reader", "alice")
    changed = await w.alice.call("space_member_add", w.a, "carol", "user", "writer")
    assert changed["created"] is False and changed["member"]["role"] == "writer"
    listed = await w.alice.call("space_member_list", w.a)
    assert {(m["principal_id"], m["role"]) for m in listed["members"]} == {
        ("alice", "owner"), ("svc_a", "writer"), ("carol", "writer")}
    assert (await w.alice.call("space_member_add", w.a, "x", "robot", "reader"))["code"] == "invalid_arguments"
    assert (await w.alice.call("space_member_add", w.a, "x", "user", "admin"))["code"] == "invalid_arguments"


async def test_member_add_principal_not_in_org_hook(space_world: SpaceWorld):
    w = space_world

    async def in_org(org_id: str, principal_id: str) -> bool:
        return principal_id != "outsider"

    w.handler.principal_in_org = in_org
    res = await w.alice.call("space_member_add", w.a, "outsider", "user", "reader")
    assert res["code"] == "principal_not_in_org"
    assert (await w.alice.call("space_member_add", w.a, "insider", "user", "reader"))["created"] is True


async def test_member_add_principal_in_org_hook_covers_services(space_world: SpaceWorld):
    """The hook is asked for service principals too; False creates nothing."""
    w = space_world
    asked: list[tuple[str, str]] = []

    async def in_org(org_id: str, principal_id: str) -> bool:
        asked.append((org_id, principal_id))
        return principal_id != "svc_outside"

    w.handler.principal_in_org = in_org
    res = await w.alice.call("space_member_add", w.a, "svc_outside", "service", "writer")
    assert res["code"] == "principal_not_in_org"
    assert all(m.principal_id != "svc_outside" for m in await w.registry.list_members(w.a))
    ok = await w.alice.call("space_member_add", w.a, "svc_inside", "service", "reader")
    assert ok["created"] is True
    assert asked == [(WORLD_ORG, "svc_outside"), (WORLD_ORG, "svc_inside")]


async def test_member_add_without_hook_adds_any_principal(space_world: SpaceWorld):
    w = space_world
    assert w.handler.principal_in_org is None
    assert (await w.alice.call("space_member_add", w.a, "svc_new", "service", "reader"))["created"] is True
    assert (await w.alice.call("space_member_add", w.a, "zed", "user", "reader"))["created"] is True


async def test_member_remove_of_non_member_is_idempotent(space_world: SpaceWorld):
    w = space_world
    assert await w.alice.call("space_member_remove", w.a, "nobody") == {"removed": True}


async def test_new_member_sees_space_on_next_call(space_world: SpaceWorld):
    w = space_world
    carol = w.actor("carol")
    assert (await carol.call("memory_recall", "pilot", space=w.a))["code"] == "space_not_found"
    await w.alice.call("space_member_add", w.a, "carol", "user", "reader")
    assert {r["id"] for r in (await carol.call("memory_recall", "Bank A pilot", space=w.a))["results"]} == w.a_ids


# -- external_id / on_existing (spec 9.3; WS-L6 extends with the full matrix and batch) --


async def test_on_existing_branches_in_space(space_world: SpaceWorld):
    w = space_world
    first = await w.svc.call("memory_store", "Doc 1 summary", space=w.a, external_id="doc-1",
                             refs=[{"type": "document", "id": "1"}], tags=["t"])
    assert first["status"] == "stored"
    same = await w.svc.call("memory_store", "  Doc 1 summary ", space=w.a, external_id="doc-1",
                            refs=[{"type": "document", "id": "1"}], tags=["T"])
    assert same == {"node_id": first["node_id"], "status": "unchanged", "space": same["space"]}
    skip = await w.svc.call("memory_store", "Doc 1 changed", space=w.a, external_id="doc-1", on_existing="skip")
    assert (skip["node_id"], skip["status"]) == (first["node_id"], "exists")
    err = await w.svc.call("memory_store", "Doc 1 changed", space=w.a, external_id="doc-1", on_existing="error")
    assert err["code"] == "external_id_exists"
    amended = await w.svc.call("memory_store", "Doc 1 changed", space=w.a, external_id="doc-1",
                               refs=[{"type": "document", "id": "1"}], tags=["t"])
    assert amended["status"] == "amended" and amended["superseded"] == first["node_id"]
    # the index follows the amendment
    again = await w.svc.call("memory_store", "Doc 1 changed", space=w.a, external_id="doc-1",
                             refs=[{"type": "document", "id": "1"}], tags=["t"])
    assert (again["node_id"], again["status"]) == (amended["node_id"], "unchanged")
    # external ids are per space: the same id in B (bob) is a new memory
    other = await w.bob.call("memory_store", "Doc 1 summary", space=w.b, external_id="doc-1")
    assert other["status"] == "stored"
    bad = await w.svc.call("memory_store", "x", space=w.a, external_id="", on_existing="amend")
    assert bad["code"] == "invalid_arguments"
    assert (await w.svc.call("memory_store", "x", space=w.a, on_existing="merge"))["code"] == "invalid_arguments"
