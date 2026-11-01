"""Spec 13.5 / 8.1 / 8.2 (WS-L7): audit events emitted by the tool layer (D18)."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pytest

from papez.context import current_client
from papez.engine import config
from papez.models.enums import PrincipalType, SpaceRole
from papez.models.space import SpaceMember

EVENT_KEYS = {"id", "at", "space_id", "principal_id", "principal_type", "action", "node_ids", "detail", "client"}
ID_RE = re.compile(r"^evt_[0-9a-f]{32}$")
AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _clear(world: Any) -> None:
    world.audit.events.clear()


def _summary(world: Any) -> list[tuple[str, str]]:
    return [(e["action"], e["space_id"]) for e in world.audit.events]


async def _add(world: Any, space_id: str, uid: str, role: SpaceRole, ptype: PrincipalType = PrincipalType.USER) -> None:
    await world.registry.add_member(SpaceMember(
        space_id=space_id, principal_id=uid, principal_type=ptype, role=role, added_by="test",
    ))


# ---------------------------------------------------------------------------
# Shape (spec 8.2)
# ---------------------------------------------------------------------------


async def test_event_shape_and_client(space_world):
    w = space_world
    _clear(w)
    from conftest import WORLD_ORG, principal

    roles = await w.alice.roles()
    with principal("alice", orgs=[WORLD_ORG], roles=roles, client={"transport": "mcp", "client_name": "claude"}):
        out = await w.handler.memory_store("Bank A kickoff moved", space=w.a)
    assert "error" not in out
    [ev] = w.audit.events
    assert set(ev) == EVENT_KEYS
    assert ID_RE.match(ev["id"])
    assert AT_RE.match(ev["at"])
    assert ev["space_id"] == w.a
    assert ev["principal_id"] == "alice"
    assert ev["principal_type"] == "user"
    assert ev["action"] == "memory.store"
    assert ev["node_ids"] == [out["node_id"]]
    assert ev["detail"] == {"status": "stored"}
    assert ev["client"] == {"transport": "mcp", "client_name": "claude"}


async def test_service_principal_recorded(space_world):
    w = space_world
    _clear(w)
    out = await w.svc.call("memory_store", "Bank A ingest finished", space=w.a)
    assert "error" not in out
    [ev] = w.audit.events
    assert (ev["principal_id"], ev["principal_type"]) == ("svc_a", "service")
    assert ev["client"] == {"transport": "stdio", "client_name": None}  # whatever the transport bound


async def test_event_ids_unique(space_world):
    w = space_world
    ids = [e["id"] for e in w.audit.events]  # seed events
    assert ids and len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# test_events_for_every_action (spec 13.5)
# ---------------------------------------------------------------------------


async def test_events_for_every_action(space_world):
    w = space_world
    a, b = w.a, w.b
    seen: set[str] = set()

    def expect(events: list[tuple[str, str, list[str] | None, dict[str, Any] | None]]) -> None:
        got = w.audit.events
        assert [(e["action"], e["space_id"]) for e in got] == [(x[0], x[1]) for x in events], got
        for ev, (action, _sid, node_ids, detail) in zip(got, events, strict=True):
            assert set(ev) == EVENT_KEYS
            if node_ids is not None:
                assert ev["node_ids"] == node_ids, (action, ev)
            if detail is not None:
                assert ev["detail"] == detail, (action, ev)
            seen.add(action)
        _clear(w)

    _clear(w)

    # -- writes to a space ----------------------------------------------------
    stored = await w.alice.call("memory_store", "Bank A steering committee is monthly", space=a)
    expect([("memory.store", a, [stored["node_id"]], {"status": "stored"})])

    amended = await w.alice.call("memory_amend", stored["node_id"], "Bank A steering committee is weekly")
    expect([("memory.amend", a, [amended["node_id"], stored["node_id"]], {"status": "amended"})])

    await w.alice.call("pin_memory", w.ids["a_q1"])
    expect([("memory.pin", a, [w.ids["a_q1"]], {})])

    await w.alice.call("unpin_memory", w.ids["a_q1"])
    expect([("memory.unpin", a, [w.ids["a_q1"]], {})])

    await w.alice.call("delete_memory", amended["node_id"])
    expect([("memory.delete", a, [amended["node_id"]], {})])

    promoted = await w.alice.call("promote_to_space", w.ids["p_notes"], a)
    assert "error" not in promoted
    expect([("memory.promote_in", a, [promoted["node_id"]],
             {"promoted_from": w.ids["p_notes"], "keep_original": True})])

    # external_id: first write stores, identical resend is unchanged, a change amends.
    first = await w.alice.call("memory_store", "Bank A SLA is 99.9", space=a, external_id="sla-1")
    expect([("memory.store", a, [first["node_id"]], {"status": "stored"})])
    same = await w.alice.call("memory_store", "Bank A SLA is 99.9", space=a, external_id="sla-1")
    expect([("memory.store", a, [first["node_id"]], {"status": "unchanged"})])
    changed = await w.alice.call("memory_store", "Bank A SLA is 99.95", space=a, external_id="sla-1")
    assert same["status"] == "unchanged" and changed["status"] == "amended"
    expect([("memory.amend", a, [changed["node_id"], first["node_id"]], {"status": "amended"})])

    batch = await w.alice.call("memory_store_batch", [
        {"content": "Bank A batch one", "external_id": "b-1"},
        {"content": "Bank A batch two", "external_id": "b-2"},
    ], space=a)
    r0, r1 = batch["results"]
    expect([
        ("memory.store", a, [r0["node_id"]], {"status": "stored", "batch_index": 0}),
        ("memory.store", a, [r1["node_id"]], {"status": "stored", "batch_index": 1}),
    ])

    retired = await w.alice.call("memory_retire_by_ref", "document", "doc_bank_a", space=a)
    assert retired["retired"] >= 1
    assert [e["action"] for e in w.audit.events] == ["memory.amend"] * retired["retired"]
    assert all(e["detail"] == {"via": "retire_by_ref", "mode": "amend"} for e in w.audit.events)
    expect([("memory.amend", a, None, None)] * retired["retired"])

    # -- reads scoped to a space ---------------------------------------------
    query = "Bank A pilot"
    rec = await w.bob.call("memory_recall", query, space=b)
    rec_ids = [r["id"] for r in rec["results"]]
    assert rec_ids
    expect([("memory.recall", b, rec_ids, {"query_chars": len(query), "results": len(rec_ids)})])

    found = await w.bob.call("memory_search", query, space=b)
    found_ids = [r["id"] for r in found["results"]]
    expect([("memory.search", b, found_ids, {"query_chars": len(query), "results": len(found_ids)})])

    trav = await w.bob.call("memory_traverse", w.ids["b_ctx"])
    trav_ids = [n["id"] for n in trav["nodes"]]
    assert set(trav_ids) == {w.ids["b_ctx"], w.ids["b_q2"]}
    expect([("memory.traverse", b, trav_ids, {"query_chars": 0, "results": 2})])

    await w.bob.call("memory_explain", w.ids["b_q2"])
    expect([("memory.explain", b, [w.ids["b_q2"]], {"query_chars": 0, "results": 1})])

    # -- space lifecycle -----------------------------------------------------
    created = await w.bob.call("space_create", "Space C", slug="space-c")
    c = created["space"]["id"]
    expect([("space.create", c, [], {"org_id": w.org})])

    await w.bob.call("space_update", c, description="private notes", name="Space C2")
    expect([("space.update", c, [], {"fields": ["description", "name"]})])
    await w.bob.call("space_update", c)  # nothing changed: no event
    expect([])

    await w.bob.call("space_archive", c)
    expect([("space.archive", c, [], {})])
    await w.bob.call("space_unarchive", c)
    expect([("space.unarchive", c, [], {})])
    await w.bob.call("space_unarchive", c)  # already active: no change, no event
    expect([])

    # -- membership ----------------------------------------------------------
    added = await w.bob.call("space_member_add", c, "carol", "user", "reader")
    assert added["created"] is True
    expect([("member.add", c, [], {"member_id": "carol", "member_type": "user", "role": "reader"})])
    changed_role = await w.bob.call("space_member_add", c, "carol", "user", "writer")
    assert changed_role["created"] is False
    expect([("member.role_change", c, [],
             {"member_id": "carol", "member_type": "user", "role": "writer", "previous_role": "reader"})])
    await w.bob.call("space_member_remove", c, "carol")
    expect([("member.remove", c, [], {"member_id": "carol", "member_type": "user", "role": "writer"})])
    await w.bob.call("space_member_remove", c, "carol")  # not a member any more: no event
    expect([])

    await w.bob.call("memory_store", "Space C scratch", space=c)
    _clear(w)
    dry = await w.bob.call("space_erase", c, "space-c", dry_run=True)
    assert dry["dry_run"] is True
    expect([])  # a dry run erases nothing
    erased = await w.bob.call("space_erase", c, "space-c")
    assert erased["dry_run"] is False
    expect([("space.erase", c, [], {"nodes": 1, "edges": 0, "members": 1})])

    # -- access.denied -------------------------------------------------------
    denied = await w.bob.call("memory_recall", query, space=a)
    assert denied["code"] == "space_not_found"
    expect([("access.denied", a, [], {"code": "space_not_found", "need_role": None})])

    assert seen == {
        "memory.store", "memory.amend", "memory.delete", "memory.pin", "memory.unpin", "memory.promote_in",
        "memory.recall", "memory.search", "memory.traverse", "memory.explain",
        "space.create", "space.update", "space.archive", "space.unarchive", "space.erase",
        "member.add", "member.remove", "member.role_change", "access.denied",
    }


async def test_retire_delete_mode_emits_memory_delete(space_world):
    w = space_world
    _clear(w)
    out = await w.bob.call("memory_retire_by_ref", "document", "doc_bank_a", space=w.b, mode="delete")
    assert out["deleted"] == 2
    assert sorted(e["node_ids"][0] for e in w.audit.events) == sorted(w.b_ids)
    assert {(e["action"], e["space_id"]) for e in w.audit.events} == {("memory.delete", w.b)}
    assert all(e["detail"] == {"via": "retire_by_ref", "mode": "delete"} for e in w.audit.events)


async def test_space_erase_events_survive_the_space(space_world):
    w = space_world
    await w.bob.call("space_erase", w.b, "space-b")
    events, _ = w.audit.query(w.b)
    actions = {e["action"] for e in events}  # same-millisecond events have no fixed order
    assert {"space.create", "memory.store", "space.erase"} <= actions
    assert await w.registry.get_space(w.b) is None


# ---------------------------------------------------------------------------
# test_no_content_in_events (spec 13.5)
# ---------------------------------------------------------------------------


async def test_no_content_in_events(space_world):
    w = space_world
    marker = f"zqx{uuid.uuid4().hex}"
    a = w.a
    ref = [{"type": "document", "id": "doc_marker", "label": f"label {marker}"}]
    stored = await w.alice.call("memory_store", f"Bank A secret {marker} figure", space=a, refs=ref)
    assert "error" not in stored
    amended = await w.alice.call("memory_amend", stored["node_id"], f"Bank A secret {marker} revised")
    await w.alice.call("memory_store", f"{marker} keyed", space=a, external_id="k-1")
    await w.alice.call("memory_store", f"{marker} keyed v2", space=a, external_id="k-1")
    await w.alice.call("memory_store_batch", [{"content": f"batch {marker}"}], space=a)
    await w.alice.call("pin_memory", amended["node_id"])
    await w.alice.call("unpin_memory", amended["node_id"])
    rec = await w.alice.call("memory_recall", f"secret {marker}", space=a)
    assert any(r["id"] == amended["node_id"] for r in rec["results"])
    await w.alice.call("memory_recall", f"secret {marker}", verbosity="concise")
    await w.alice.call("memory_search", f"secret {marker}", space=a)
    await w.alice.call("memory_traverse", amended["node_id"])
    await w.alice.call("memory_explain", amended["node_id"])
    await w.alice.call("space_update", a, description=f"about {marker}", name=f"Space {marker}")
    personal = await w.alice.call("memory_store", f"personal {marker}")
    await w.alice.call("promote_to_space", personal["node_id"], a)
    await w.alice.call("memory_retire_by_ref", "document", "doc_marker", space=a)
    await w.bob.call("memory_recall", f"secret {marker}", space=a)  # access.denied
    await w.alice.call("delete_memory", amended["node_id"])

    actions = {e["action"] for e in w.audit.events}
    assert {"memory.store", "memory.amend", "memory.recall", "memory.search", "memory.traverse",
            "memory.explain", "space.update", "memory.promote_in", "access.denied", "memory.delete"} <= actions
    dumped = json.dumps(w.audit.events)
    assert marker not in dumped
    assert "secret" not in dumped
    # No slugs or names either (the label of space A must not leak via events).
    assert "space-a" not in dumped and "Space A" not in dumped


# ---------------------------------------------------------------------------
# test_access_denied_logged (spec 13.5)
# ---------------------------------------------------------------------------


async def test_access_denied_logged(space_world):
    w = space_world
    a = w.a
    _clear(w)

    # Non-member, by id and by slug: told "not found", logged against the real id.
    by_id = await w.bob.call("memory_recall", "Bank A", space=a)
    by_slug = await w.bob.call("memory_store", "sneaky", space="space-a")
    missing = await w.bob.call("memory_recall", "Bank A", space="spc_does_not_exist")
    assert json.dumps(by_id) == json.dumps(by_slug) == json.dumps(missing)
    assert by_id["code"] == "space_not_found"
    assert [(e["action"], e["space_id"], e["principal_id"], e["detail"]) for e in w.audit.events] == [
        ("access.denied", a, "bob", {"code": "space_not_found", "need_role": None}),
        ("access.denied", a, "bob", {"code": "space_not_found", "need_role": "writer"}),
    ]  # the missing space logged nothing
    _clear(w)

    # Space management by a non-member.
    got = await w.bob.call("space_get", a)
    assert got["code"] == "space_not_found"
    assert _summary(w) == [("access.denied", a)]
    _clear(w)

    # A reader attempting writes: store (scope check) and pin (node check).
    await _add(w, a, "carol", SpaceRole.READER)
    carol = w.actor("carol")
    store = await carol.call("memory_store", "reader write", space=a)
    assert store["code"] == "insufficient_role"
    pin = await carol.call("pin_memory", w.ids["a_q1"])
    assert pin["code"] == "insufficient_role"
    member = await carol.call("space_member_add", a, "dave", "user", "reader")
    assert member["code"] == "insufficient_role"
    assert [(e["action"], e["space_id"], e["principal_id"], e["detail"]) for e in w.audit.events] == [
        ("access.denied", a, "carol", {"code": "insufficient_role", "need_role": "writer"}),
        ("access.denied", a, "carol", {"code": "insufficient_role", "need_role": "writer"}),
        ("access.denied", a, "carol", {"code": "insufficient_role", "need_role": "owner"}),
    ]
    assert all(e["node_ids"] == [] for e in w.audit.events)
    _clear(w)

    # Archived space: the owner's write is refused and logged.
    await w.bob.call("space_archive", w.b)
    _clear(w)
    arch = await w.bob.call("memory_store", "late note", space=w.b)
    assert arch["code"] == "space_archived"
    amend = await w.bob.call("memory_amend", w.ids["b_q2"], "Bank A pilot starts in Q3")
    assert amend["code"] == "space_archived"
    assert [(e["action"], e["space_id"], e["detail"]["code"]) for e in w.audit.events] == [
        ("access.denied", w.b, "space_archived"),
        ("access.denied", w.b, "space_archived"),
    ]


async def test_access_denied_other_org_and_service(space_world):
    w = space_world
    _clear(w)
    outsider = w.actor("eve", orgs=("org_other",))
    out = await outsider.call("memory_recall", "Bank A", space=w.a)
    assert out["code"] == "space_not_found"
    # By id the real space is found even across orgs; a slug only resolves in the caller's orgs.
    slug = await outsider.call("memory_recall", "Bank A", space="space-a")
    assert slug["code"] == "space_not_found"
    assert _summary(w) == [("access.denied", w.a)]
    _clear(w)

    svc = await w.svc.call("memory_recall", "Bank A", space=w.b)
    assert svc["code"] == "space_not_found"
    [ev] = w.audit.events
    assert (ev["space_id"], ev["principal_id"], ev["principal_type"]) == (w.b, "svc_a", "service")
    _clear(w)

    # space_required carries no space: nothing to log against.
    req = await w.svc.call("memory_store", "no space")
    assert req["code"] == "space_required"
    assert w.audit.events == []


# ---------------------------------------------------------------------------
# test_read_event_per_space_in_scope
# ---------------------------------------------------------------------------


async def test_read_event_per_space_in_scope(space_world, monkeypatch):
    w = space_world
    await _add(w, w.b, "alice", SpaceRole.READER)
    created = await w.alice.call("space_create", "Empty", slug="empty")
    empty = created["space"]["id"]
    query = "Bank A pilot budget"

    for tool in ("memory_recall", "memory_search"):
        _clear(w)
        out = await w.alice.call(tool, query, k=20)
        per_space: dict[str | None, list[str]] = {}
        for r in out["results"]:
            per_space.setdefault((r.get("space") or {}).get("id"), []).append(r["id"])
        assert per_space.get(w.a) and per_space.get(w.b) and per_space.get(None)
        events = {e["space_id"]: e for e in w.audit.events}
        assert len(events) == len(w.audit.events) == 3  # A, B and the empty space; none for personal
        assert set(events) == {w.a, w.b, empty}
        action = tool.replace("_", ".")
        for sid in (w.a, w.b):
            assert events[sid]["action"] == action
            assert events[sid]["node_ids"] == per_space[sid]
            assert events[sid]["detail"] == {"query_chars": len(query), "results": len(per_space[sid])}
        assert events[empty]["detail"] == {"query_chars": len(query), "results": 0}
        assert events[empty]["node_ids"] == []

    # Explicit spaces=[...]: only the named spaces.
    _clear(w)
    out = await w.alice.call("memory_recall", query, spaces=[w.b, "empty"], include_personal=True)
    assert "error" not in out
    assert sorted(_summary(w)) == sorted([("memory.recall", w.b), ("memory.recall", empty)])

    # The node_ids cap applies; the count does not.
    monkeypatch.setattr(config, "AUDIT_READ_NODE_IDS_CAP", 1)
    _clear(w)
    out = await w.alice.call("memory_recall", query, space=w.a, k=20)
    a_hits = [r["id"] for r in out["results"]]
    assert len(a_hits) >= 2
    [ev] = w.audit.events
    assert ev["node_ids"] == a_hits[:1]
    assert ev["detail"]["results"] == len(a_hits)


async def test_read_events_skip_errors_and_invisible_nodes(space_world):
    w = space_world
    _clear(w)
    out = await w.alice.call("memory_traverse", w.ids["b_q2"])  # not visible to alice
    assert out["code"] == "not_found"
    out = await w.alice.call("memory_explain", w.ids["b_q2"])
    assert out["code"] == "not_found"
    out = await w.alice.call("memory_recall", "x", space=w.a, spaces=[w.a])
    assert out["code"] == "invalid_arguments"
    assert w.audit.events == []


# ---------------------------------------------------------------------------
# test_personal_actions_emit_nothing
# ---------------------------------------------------------------------------


async def test_personal_actions_emit_nothing(space_world):
    w = space_world
    _clear(w)
    dave = w.actor("dave")  # no spaces at all
    s1 = await dave.call("memory_store", "Dave likes tea")
    s2 = await dave.call("memory_store", "Dave likes green tea", related_to=[s1["node_id"]])
    am = await dave.call("memory_amend", s2["node_id"], "Dave likes oolong tea")
    await dave.call("pin_memory", s1["node_id"])
    await dave.call("unpin_memory", s1["node_id"])
    await dave.call("memory_recall", "tea")
    await dave.call("memory_search", "tea")
    await dave.call("memory_traverse", s1["node_id"])
    await dave.call("memory_explain", s1["node_id"])
    await dave.call("list_core_memories")
    await dave.call("memory_stats")
    await dave.call("memory_store_batch", [{"content": "Dave batch", "external_id": "d-1"}])
    await dave.call("memory_store", "Dave keyed", external_id="dk")
    await dave.call("memory_store", "Dave keyed v2", external_id="dk")
    await dave.call("delete_memory", am["node_id"])
    assert w.audit.events == []

    # alice is in a space, but her personal-node actions are still personal.
    await w.alice.call("memory_amend", w.ids["p_notes"], "Bank A pilot notes, revised")
    await w.alice.call("pin_memory", w.ids["p_more"])
    await w.alice.call("unpin_memory", w.ids["p_more"])
    await w.alice.call("memory_traverse", w.ids["p_more"])
    await w.alice.call("memory_explain", w.ids["p_more"])
    await w.alice.call("memory_recall", "Bank A", include_personal=True, spaces=[])
    await w.alice.call("memory_retire_by_ref", "document", "doc_bank_a")
    await w.alice.call("delete_memory", w.ids["p_more"])
    assert w.audit.events == []


async def test_no_sink_is_a_noop(space_world):
    w = space_world
    w.handler.audit = None
    out = await w.alice.call("memory_store", "Bank A no sink", space=w.a)
    assert "error" not in out
    denied = await w.bob.call("memory_recall", "Bank A", space=w.a)
    assert denied["code"] == "space_not_found"


async def test_failing_sink_never_breaks_a_tool(space_world):
    w = space_world

    class Broken:
        async def record(self, event: dict[str, Any]) -> None:
            raise RuntimeError("sink down")

    w.handler.audit = Broken()
    out = await w.alice.call("memory_store", "Bank A broken sink", space=w.a)
    assert "error" not in out
    rec = await w.alice.call("memory_recall", "Bank A", space=w.a)
    assert rec["results"]


# ---------------------------------------------------------------------------
# stdio current_client (server.py)
# ---------------------------------------------------------------------------


def test_stdio_client_name_bound_per_call(monkeypatch):
    from papez import server as srv

    assert srv._stdio_client_name() is None  # outside a request: no session
    monkeypatch.setattr(srv, "_stdio_client_name", lambda: "claude-code")
    token = current_client.set({"transport": "stdio", "client_name": None})
    try:
        inner = srv._bind_stdio_client_name()
        assert inner is not None
        assert current_client.get() == {"transport": "stdio", "client_name": "claude-code"}
        current_client.reset(inner)
        assert current_client.get() == {"transport": "stdio", "client_name": None}
    finally:
        current_client.reset(token)

    token = current_client.set({"transport": "mcp", "client_name": None})
    try:
        assert srv._bind_stdio_client_name() is None  # hosted transports bind their own
    finally:
        current_client.reset(token)


@pytest.mark.parametrize("name", ["memory_store", "memory_recall"])
async def test_stdio_call_tool_records_client(monkeypatch, name):
    from papez import server as srv
    from papez.storage.audit import InMemoryAuditSink
    from papez.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider
    from papez.storage.spaces import InMemorySpaceRegistry
    from papez.context import current_org_ids, current_principal_type, current_user_id
    from papez.mcp.tools import MCPToolHandler
    from conftest import TokenEmbedder

    registry = InMemorySpaceRegistry()
    audit = InMemoryAuditSink()
    handler = MCPToolHandler(InMemoryGraphProvider(), TokenEmbedder(), InMemoryCacheProvider(),
                             spaces=registry, audit=audit)
    monkeypatch.setattr(srv, "tools", handler)
    monkeypatch.setattr(srv, "_stdio_client_name", lambda: "claude-code")
    method_dispatch = dict(srv._TOOL_DISPATCH)
    for tool in method_dispatch:
        _, req, opt = method_dispatch[tool]
        method_dispatch[tool] = (getattr(handler, tool), req, opt)
    monkeypatch.setattr(srv, "_TOOL_DISPATCH", method_dispatch)

    tokens = [
        (current_user_id, current_user_id.set(srv.STDIO_LOCAL_USER)),
        (current_principal_type, current_principal_type.set("user")),
        (current_org_ids, current_org_ids.set([srv.STDIO_LOCAL_ORG])),
        (current_client, current_client.set({"transport": "stdio", "client_name": None})),
    ]
    try:
        created = json.loads((await srv.call_tool("space_create", {"name": "Local"}))[0].text)
        sid = created["space"]["id"]
        args = {"content": "local note"} if name == "memory_store" else {"query": "local"}
        await srv.call_tool(name, {**args, "space": sid})
        assert current_client.get() == {"transport": "stdio", "client_name": None}  # reset after the call
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)
    assert [e["client"] for e in audit.events] == [{"transport": "stdio", "client_name": "claude-code"}] * 2
    assert audit.events[0]["action"] == "space.create"
    assert audit.events[1]["action"] == name.replace("_", ".")
    assert audit.events[1]["principal_id"] == srv.STDIO_LOCAL_USER


async def test_erase_user_membership_removal_is_audited(space_world):
    """WS-verify: erase_user drops memberships; each drop is a spec 8.1 member.remove.

    The space where the user is the last user owner keeps the membership
    (spaces_still_owned) and logs nothing.
    """
    world = space_world
    await _add(world, world.a, "dave", SpaceRole.OWNER)
    await _add(world, world.b, "alice", SpaceRole.READER)
    owned = await world.alice.call("space_create", "Alice Only", slug="alice-only")
    _clear(world)

    result = await world.alice.call("erase_user", "alice")

    assert result["spaces_still_owned"] == [owned["space"]["id"]]
    removes = sorted(
        ((e["space_id"], e["detail"]["role"]) for e in world.audit.events if e["action"] == "member.remove"),
    )
    assert removes == sorted([(world.a, "owner"), (world.b, "reader")])
    for e in world.audit.events:
        assert set(e) == EVENT_KEYS
        assert e["detail"]["member_id"] == "alice"
        assert e["detail"]["member_type"] == "user"
        assert e["detail"]["via"] == "erase_user"
    assert owned["space"]["id"] not in {e["space_id"] for e in world.audit.events}
