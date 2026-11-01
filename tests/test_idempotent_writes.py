"""R5 idempotent writes (spec 9, 13.4): external_id + on_existing, batches, retire by ref."""
from __future__ import annotations

from typing import Any

import pytest
from conftest import SpaceWorld

from papez.engine import config
from papez.models.enums import EdgeType
from papez.storage.memory import space_bucket_key

DOC = {"type": "document", "id": "doc_1", "label": "Pilot plan"}
DOC2 = {"type": "document", "id": "doc_2", "label": "Budget sheet"}


def _bucket(w: SpaceWorld, space_id: str | None, uid: str = "alice") -> dict[str, Any]:
    return w.graph._user_nodes.get(space_bucket_key(space_id) if space_id else uid, {})


def _with_external_id(w: SpaceWorld, space_id: str | None, ext: str, uid: str = "alice") -> list[Any]:
    return [n for n in _bucket(w, space_id, uid).values() if n.external_id == ext]


async def _current(w: SpaceWorld, ext: str, space_id: str | None) -> Any:
    return await w.alice.storage("find_node_by_external_id", ext, space_id)


async def _supersedes(w: SpaceWorld, new_id: str) -> list[str]:
    edges = await w.alice.storage("get_edges", new_id, "outgoing", EdgeType.SUPERSEDES)
    return [str(e.target_id) for e in edges if str(e.source_id) == new_id]


# -- memory_store on_existing (spec 9.3) ----------------------------------------


@pytest.mark.parametrize("space", ["space", "personal"])
async def test_on_existing_matrix(space_world: SpaceWorld, space: str):
    w = space_world
    sid = w.a if space == "space" else None
    first = await w.alice.call("memory_store", "Kickoff is on Monday", space=sid, refs=[DOC],
                               tags=["pilot"], external_id="ext-1")
    assert first["status"] == "stored"
    orig = first["node_id"]

    # skip, identical and different content: exists, nothing changes
    for content in ("Kickoff is on Monday", "Kickoff moved to Friday"):
        res = await w.alice.call("memory_store", content, space=sid, refs=[DOC], tags=["pilot"],
                                 external_id="ext-1", on_existing="skip")
        assert {k: res[k] for k in ("node_id", "status")} == {"node_id": orig, "status": "exists"}
    assert len(_with_external_id(w, sid, "ext-1")) == 1

    # error, identical and different: external_id_exists, nothing changes
    for content in ("Kickoff is on Monday", "Kickoff moved to Friday"):
        res = await w.alice.call("memory_store", content, space=sid, refs=[DOC], tags=["pilot"],
                                 external_id="ext-1", on_existing="error")
        assert res["code"] == "external_id_exists" and "error" in res
    assert len(_with_external_id(w, sid, "ext-1")) == 1

    # amend, identical: whitespace trimmed, refs/tags compared as sets
    res = await w.alice.call("memory_store", "  Kickoff is on Monday \n", space=sid, refs=[DOC],
                             tags=["Pilot"], external_id="ext-1")
    assert {k: res[k] for k in ("node_id", "status")} == {"node_id": orig, "status": "unchanged"}
    assert len(_with_external_id(w, sid, "ext-1")) == 1

    # amend, different content / refs / tags: each supersedes the current node
    current = orig
    for kwargs in (
        {"content": "Kickoff moved to Friday", "refs": [DOC], "tags": ["pilot"]},
        {"content": "Kickoff moved to Friday", "refs": [DOC, DOC2], "tags": ["pilot"]},
        {"content": "Kickoff moved to Friday", "refs": [DOC, DOC2], "tags": ["pilot", "q1"]},
        {"content": "Kickoff moved to Friday", "refs": [{**DOC, "label": "Renamed plan"}, DOC2],
         "tags": ["pilot", "q1"]},
    ):
        content = kwargs.pop("content")
        res = await w.alice.call("memory_store", content, space=sid, external_id="ext-1", **kwargs)
        assert res["status"] == "amended", (kwargs, res)
        assert res["superseded"] == current and res["node_id"] != current
        assert await _supersedes(w, res["node_id"]) == [current]
        current = res["node_id"]
        if sid is not None:
            assert res["space"]["id"] == sid
    # sets: same refs and tags in a different order are identical
    res = await w.alice.call("memory_store", "Kickoff moved to Friday", space=sid,
                             refs=[DOC2, {**DOC, "label": "Renamed plan"}], tags=["q1", "pilot"],
                             external_id="ext-1")
    assert res == {"node_id": current, "status": "unchanged"} | ({"space": res["space"]} if sid else {})


async def test_external_id_follows_amendment(space_world: SpaceWorld):
    w = space_world
    old = (await w.alice.call("memory_store", "Rate is 4%", space=w.a, external_id="rate"))["node_id"]
    new = await w.alice.call("memory_store", "Rate is 5%", space=w.a, external_id="rate")
    assert new["status"] == "amended" and new["superseded"] == old
    assert str((await _current(w, "rate", w.a)).id) == new["node_id"]
    old_node = await w.alice.storage("get_node", old)
    assert old_node.external_id == "rate"  # history keeps it
    # memory_amend moves the index too
    amended = await w.alice.call("memory_amend", new["node_id"], "Rate is 6%")
    assert str((await _current(w, "rate", w.a)).id) == amended["node_id"]
    again = await w.alice.call("memory_store", "Rate is 6%", space=w.a, external_id="rate")
    assert again["status"] == "unchanged" and again["node_id"] == amended["node_id"]


async def test_external_id_scoped_to_bucket(space_world: SpaceWorld):
    w = space_world
    a = await w.alice.call("memory_store", "Same id in A", space=w.a, external_id="x")
    p = await w.alice.call("memory_store", "Same id personal", external_id="x")
    b = await w.bob.call("memory_store", "Same id in B", space=w.b, external_id="x")
    bp = await w.bob.call("memory_store", "Same id bob personal", external_id="x")
    assert {r["status"] for r in (a, p, b, bp)} == {"stored"}
    assert len({r["node_id"] for r in (a, p, b, bp)}) == 4


# -- memory_store_batch (spec 9.4) ----------------------------------------------


def _items(n: int, prefix: str = "item") -> list[dict[str, Any]]:
    return [{"content": f"{prefix} number {i}", "external_id": f"{prefix}-{i}", "refs": [DOC], "tags": ["batch"]}
            for i in range(n)]


async def test_batch_stores_in_order(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store_batch", _items(3), space=w.a)
    assert [r["index"] for r in res["results"]] == [0, 1, 2]
    assert {r["status"] for r in res["results"]} == {"stored"}
    for i, r in enumerate(res["results"]):
        node = await w.alice.storage("get_node", r["node_id"])
        assert node.space_id == w.a and node.content_full == f"item number {i}" and node.external_id == f"item-{i}"


async def test_batch_all_or_nothing_validation(space_world: SpaceWorld):
    w = space_world
    before = len(_bucket(w, w.a))
    items = _items(6)
    items[1]["tags"] = ["q2/2026"]
    items[2]["refs"] = [{"type": "Bad Type", "id": "x"}]
    items[3]["related"] = [{"id": w.ids["b_q2"], "type": "supports"}]  # other space
    items[4]["related"] = [{"id": w.ids["a_q1"], "type": "not_a_type"}]
    items[5]["related"] = [{"ref_index": 5, "type": "supports"}]  # not an earlier item
    items.append({"content": "   "})
    res = await w.alice.call("memory_store_batch", items, space=w.a)
    assert res["code"] == "batch_invalid"
    assert res["error"] == "6 items are invalid"
    by_index = {p["index"]: p for p in res["items"]}
    assert set(by_index) == {1, 2, 3, 4, 5, 6}
    assert by_index[1]["code"] == "invalid_tag"
    assert by_index[2]["code"] == "invalid_ref"
    assert by_index[3]["code"] == "cross_space_edge"
    assert by_index[4]["code"] == "invalid_arguments"
    assert by_index[5]["code"] == "invalid_arguments"
    assert by_index[6]["code"] == "invalid_arguments"
    assert all(set(p) == {"index", "code", "error"} for p in res["items"])
    assert len(_bucket(w, w.a)) == before  # nothing stored, not even item 0
    assert await _current(w, "item-0", w.a) is None


async def test_batch_error_mode_conflict_is_a_validation_failure(space_world: SpaceWorld):
    w = space_world
    await w.alice.call("memory_store", "pre-existing", space=w.a, external_id="item-2")
    before = len(_bucket(w, w.a))
    res = await w.alice.call("memory_store_batch", _items(3), space=w.a, on_existing="error")
    assert res["code"] == "batch_invalid"
    assert [(p["index"], p["code"]) for p in res["items"]] == [(2, "external_id_exists")]
    assert len(_bucket(w, w.a)) == before
    dup = await w.alice.call("memory_store_batch",
                             [{"content": "one", "external_id": "d"}, {"content": "two", "external_id": "d"}],
                             space=w.a, on_existing="error")
    assert [(p["index"], p["code"]) for p in dup["items"]] == [(1, "external_id_exists")]


async def test_batch_ref_index(space_world: SpaceWorld):
    w = space_world
    items = [
        {"content": "Decision: ship the pilot", "external_id": "d"},
        {"content": "Because the bank signed", "external_id": "r", "related": [{"ref_index": 0, "type": "supports"}]},
        {"content": "Follow-up on the pilot", "related": [
            {"ref_index": 0, "type": "temporal_sequence"}, {"id": w.ids["a_q1"], "type": "related_to"},
        ]},
    ]
    res = await w.alice.call("memory_store_batch", items, space=w.a)
    ids = [r["node_id"] for r in res["results"]]
    out1 = await w.alice.storage("get_edges", ids[1], "outgoing")
    assert any(str(e.target_id) == ids[0] and e.type == EdgeType.SUPPORTS for e in out1)
    out2 = await w.alice.storage("get_edges", ids[2], "outgoing")
    targets = {(str(e.target_id), e.type) for e in out2}
    assert (ids[0], EdgeType.TEMPORAL_SEQUENCE) in targets
    assert (w.ids["a_q1"], EdgeType.RELATED_TO) in targets


async def test_batch_retry_is_safe(space_world: SpaceWorld):
    w = space_world
    items = _items(4)
    items[2]["related"] = [{"ref_index": 0, "type": "supports"}]
    first = await w.alice.call("memory_store_batch", items, space=w.a)
    count = len(_bucket(w, w.a))
    second = await w.alice.call("memory_store_batch", items, space=w.a)
    assert [r["node_id"] for r in second["results"]] == [r["node_id"] for r in first["results"]]
    assert {r["status"] for r in second["results"]} == {"unchanged"}
    assert len(_bucket(w, w.a)) == count  # no duplicates
    for i in range(4):
        assert len(_with_external_id(w, w.a, f"item-{i}")) == 1
    # changing one item amends only that one
    items[1]["content"] = "item number 1, revised"
    third = await w.alice.call("memory_store_batch", items, space=w.a)
    statuses = [r["status"] for r in third["results"]]
    assert statuses == ["unchanged", "amended", "unchanged", "unchanged"]
    assert third["results"][1]["superseded"] == first["results"][1]["node_id"]
    assert "superseded" not in third["results"][0]


async def test_batch_skip_mode(space_world: SpaceWorld):
    w = space_world
    first = await w.alice.call("memory_store_batch", _items(2), space=w.a)
    changed = [{**i, "content": i["content"] + " changed"} for i in _items(2)]
    res = await w.alice.call("memory_store_batch", changed, space=w.a, on_existing="skip")
    assert [(r["node_id"], r["status"]) for r in res["results"]] == [
        (r["node_id"], "exists") for r in first["results"]
    ]


async def test_batch_duplicate_external_id_in_one_batch_amends(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store_batch",
                             [{"content": "v1", "external_id": "d"}, {"content": "v2", "external_id": "d"}],
                             space=w.a)
    r0, r1 = res["results"]
    assert r0["status"] == "stored" and r1["status"] == "amended" and r1["superseded"] == r0["node_id"]
    assert str((await _current(w, "d", w.a)).id) == r1["node_id"]


async def test_batch_personal_for_users(space_world: SpaceWorld):
    w = space_world
    res = await w.alice.call("memory_store_batch", _items(2, "mine"))
    for r in res["results"]:
        node = await w.alice.storage("get_node", r["node_id"])
        assert node.space_id is None and node.original_user_id == "alice"
    assert await w.bob.storage("get_node", res["results"][0]["node_id"]) is None


async def test_batch_respects_space_role(space_world: SpaceWorld):
    w = space_world
    from papez.models.enums import PrincipalType, SpaceRole
    from papez.models.space import SpaceMember

    await w.registry.add_member(SpaceMember(
        space_id=w.a, principal_id="rita", principal_type=PrincipalType.USER, role=SpaceRole.READER,
        added_by="alice",
    ))
    rita, nora = w.actor("rita"), w.actor("nora")
    before = len(_bucket(w, w.a))
    assert (await rita.call("memory_store_batch", _items(1), space=w.a))["code"] == "insufficient_role"
    assert (await nora.call("memory_store_batch", _items(1), space=w.a)) == {
        "error": "Space not found", "code": "space_not_found",
    }
    assert (await w.bob.call("memory_store_batch", _items(1), space="space-a"))["code"] == "space_not_found"
    assert len(_bucket(w, w.a)) == before
    # writer (service) and owner succeed
    assert "results" in await w.svc.call("memory_store_batch", _items(1, "svc"), space=w.a)
    assert "results" in await w.alice.call("memory_store_batch", _items(1, "own"), space=w.a)
    # archived space refuses writes
    await w.alice.call("space_archive", w.a)
    assert (await w.alice.call("memory_store_batch", _items(1, "late"), space=w.a))["code"] == "space_archived"


async def test_batch_service_requires_space(space_world: SpaceWorld):
    w = space_world
    res = await w.svc.call("memory_store_batch", _items(1))
    assert res["code"] == "space_required"
    assert await w.svc.call("memory_retire_by_ref", "document", "doc_bank_a") == {
        "error": res["error"], "code": "space_required",
    }


async def test_batch_size_limit(space_world: SpaceWorld, monkeypatch: pytest.MonkeyPatch):
    w = space_world
    monkeypatch.setattr(config, "BATCH_MAX_ITEMS", 3)
    before = len(_bucket(w, w.a))
    res = await w.alice.call("memory_store_batch", _items(4), space=w.a)
    assert res["code"] == "invalid_arguments" and "3" in res["error"]
    assert len(_bucket(w, w.a)) == before
    assert len((await w.alice.call("memory_store_batch", _items(3), space=w.a))["results"]) == 3


async def test_batch_default_limit_is_100():
    assert config.BATCH_MAX_ITEMS == 100


async def test_batch_bad_arguments(space_world: SpaceWorld):
    w = space_world
    assert (await w.alice.call("memory_store_batch", "nope", space=w.a))["code"] == "invalid_arguments"
    assert (await w.alice.call("memory_store_batch", _items(1), space=w.a, on_existing="x"))["code"] == (
        "invalid_arguments"
    )
    res = await w.alice.call("memory_store_batch", [{"content": "x", "visibility": "org"}, 5], space=w.a)
    assert res["code"] == "batch_invalid" and [p["index"] for p in res["items"]] == [0, 1]
    assert await w.alice.call("memory_store_batch", [], space=w.a) == {"results": []}


# -- memory_retire_by_ref (spec 9.5) --------------------------------------------


async def _seed_refs(w: SpaceWorld, space_id: str | None) -> dict[str, str]:
    actor = w.alice
    sole = await actor.call("memory_store", "The pilot covers three branches", space=space_id,
                            refs=[DOC], tags=["pilot"], external_id="sole")
    multi = await actor.call("memory_store", "Budget is 2M for the pilot", space=space_id,
                             refs=[DOC, DOC2], tags=["budget"], external_id="multi")
    other = await actor.call("memory_store", "Unrelated memory", space=space_id, refs=[DOC2])
    return {"sole": sole["node_id"], "multi": multi["node_id"], "other": other["node_id"]}


@pytest.mark.parametrize("space", ["space", "personal"])
async def test_retire_by_ref_modes(space_world: SpaceWorld, space: str):
    w = space_world
    sid = w.a if space == "space" else None

    # mode=amend
    ids = await _seed_refs(w, sid)
    res = await w.alice.call("memory_retire_by_ref", "document", "doc_1", space=sid)
    assert res == {"retired": 1, "updated": 1, "deleted": 0}
    retired = await _current(w, "sole", sid)
    assert str(retired.id) != ids["sole"] and await _supersedes(w, str(retired.id)) == [ids["sole"]]
    assert retired.content_full == (
        "[Retired] The pilot covers three branches - source Pilot plan was removed."
    )
    assert "retired" in retired.tags and "pilot" in retired.tags
    updated = await _current(w, "multi", sid)
    assert await _supersedes(w, str(updated.id)) == [ids["multi"]]
    assert [(r.type, r.id) for r in updated.refs] == [("document", "doc_2")]
    assert updated.content_full == "Budget is 2M for the pilot" and updated.tags == ["budget"]
    assert (await w.alice.storage("get_node", ids["other"])) is not None
    # history kept: old nodes still exist and keep the ref
    old = await w.alice.storage("get_node", ids["sole"])
    assert old is not None and old.refs[0].id == "doc_1"
    # repeat is a no-op
    assert await w.alice.call("memory_retire_by_ref", "document", "doc_1", space=sid) == {
        "retired": 0, "updated": 0, "deleted": 0,
    }

    # mode=delete on a fresh source
    sole = await w.alice.call("memory_store", "Only from doc 3", space=sid, refs=[{"type": "document", "id": "doc_3"}])
    multi = await w.alice.call("memory_store", "From doc 3 and 2", space=sid,
                               refs=[{"type": "document", "id": "doc_3"}, DOC2])
    res = await w.alice.call("memory_retire_by_ref", "document", "doc_3", space=sid, mode="delete")
    assert res == {"retired": 0, "updated": 1, "deleted": 1}
    assert await w.alice.storage("get_node", sole["node_id"]) is None
    assert await w.alice.storage("find_nodes_by_ref", "document", "doc_3") == [
        await w.alice.storage("get_node", multi["node_id"])
    ]  # only the superseded original still cites it
    successors = [n for n in _bucket(w, sid).values() if str(n.id) != multi["node_id"]
                  and n.content_full == "From doc 3 and 2"]
    assert len(successors) == 1 and [(r.type, r.id) for r in successors[0].refs] == [("document", "doc_2")]


async def test_retire_by_ref_label_falls_back_to_id(space_world: SpaceWorld):
    w = space_world
    await w.alice.call("memory_store", "A note", space=w.a, refs=[{"type": "ticket", "id": "T-9"}],
                       external_id="t")
    await w.alice.call("memory_retire_by_ref", "ticket", "T-9", space=w.a)
    assert (await _current(w, "t", w.a)).content_full == "[Retired] A note - source T-9 was removed."


async def test_retire_by_ref_scope_and_roles(space_world: SpaceWorld):
    w = space_world
    from papez.models.enums import PrincipalType, SpaceRole
    from papez.models.space import SpaceMember

    await w.registry.add_member(SpaceMember(
        space_id=w.a, principal_id="rita", principal_type=PrincipalType.USER, role=SpaceRole.READER,
        added_by="alice",
    ))
    # seeded memories all cite doc_bank_a in A, B and alice's personal memory
    assert (await w.actor("rita").call("memory_retire_by_ref", "document", "doc_bank_a", space=w.a))["code"] == (
        "insufficient_role"
    )
    assert (await w.bob.call("memory_retire_by_ref", "document", "doc_bank_a", space=w.a))["code"] == (
        "space_not_found"
    )
    assert (await w.alice.call("memory_retire_by_ref", "document", "doc_bank_a", mode="bogus"))["code"] == (
        "invalid_arguments"
    )
    # the service (writer) retires A only
    res = await w.svc.call("memory_retire_by_ref", "document", "doc_bank_a", space=w.a, mode="delete")
    assert res == {"retired": 0, "updated": 0, "deleted": 2}
    for label in ("b_q2", "b_ctx"):
        assert await w.bob.storage("get_node", w.ids[label]) is not None
    for label in ("p_notes", "p_more"):
        assert await w.alice.storage("get_node", w.ids[label]) is not None
    # alice without a space: personal only, never B
    res = await w.alice.call("memory_retire_by_ref", "document", "doc_bank_a")
    assert res == {"retired": 2, "updated": 0, "deleted": 0}
    for label in ("b_q2", "b_ctx"):
        node = await w.bob.storage("get_node", w.ids[label])
        assert node is not None and "retired" not in node.tags


# -- server wiring ---------------------------------------------------------------


def test_server_retryable_only_with_external_ids():
    from papez import server as srv

    assert "memory_store_batch" in srv._TOOL_DISPATCH and "memory_retire_by_ref" in srv._TOOL_DISPATCH
    assert srv._is_retryable("memory_store_batch", {"items": [{"content": "a", "external_id": "1"}]})
    assert not srv._is_retryable("memory_store_batch", {"items": [{"content": "a", "external_id": "1"},
                                                                  {"content": "b"}]})
    assert not srv._is_retryable("memory_store_batch", {"items": []})
    assert not srv._is_retryable("memory_retire_by_ref", {})
    defs = {d["name"]: d for d in srv._TOOL_DEFS}
    assert "retry" in defs["memory_store_batch"]["description"].lower()


async def test_call_tool_marks_failed_batch_retryable(monkeypatch: pytest.MonkeyPatch):
    import json

    from papez import server as srv

    async def boom(**_: Any) -> Any:
        raise RuntimeError("boom")

    method, required, optional = srv._TOOL_DISPATCH["memory_store_batch"]
    monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_store_batch", (boom, required, optional))
    keyed = await srv.call_tool("memory_store_batch", {"items": [{"content": "a", "external_id": "x"}]})
    assert json.loads(keyed[0].text)["retryable"] is True
    unkeyed = await srv.call_tool("memory_store_batch", {"items": [{"content": "a"}]})
    assert json.loads(unkeyed[0].text)["retryable"] is False


# -- Postgres external_id index (WS-pg) -----------------------------------------


def _enforce_postgres_external_id_index(w: SpaceWorld) -> None:
    """Make the in-memory graph refuse what idx_nodes_external_id_current refuses.

    Postgres keeps one current node per (bucket, external_id); a node only stops
    being current when a SUPERSEDES edge marks it (superseded_by, D17). So the
    superseding node cannot be inserted with the external_id before the edge.
    """
    from papez.context import current_user_id

    g = w.graph
    superseded: set[str] = set()
    create_node, create_edge, update_node = g.create_node, g.create_edge, g.update_node

    def clash(bucket: str, ext: str, nid: str) -> bool:
        return any(n.external_id == ext and str(n.id) not in superseded and str(n.id) != nid
                   for n in g._user_nodes.get(bucket, {}).values())

    async def strict_create_node(node: Any) -> str:
        bucket = space_bucket_key(node.space_id) if node.space_id else current_user_id.get()
        if node.external_id is not None and clash(bucket, node.external_id, str(node.id)):
            raise ValueError("external_id_exists")
        return await create_node(node)

    async def strict_create_edge(edge: Any) -> str:
        eid = await create_edge(edge)
        if edge.type == EdgeType.SUPERSEDES:
            superseded.add(str(edge.target_id))
        return eid

    async def strict_update_node(node_id: str, updates: dict[str, Any]) -> None:
        ext = updates.get("external_id")
        for bucket, nodes in g._user_nodes.items():
            if ext is not None and node_id in nodes and clash(bucket, ext, node_id):
                raise ValueError("external_id_exists")
        await update_node(node_id, updates)

    g.create_node, g.create_edge, g.update_node = strict_create_node, strict_create_edge, strict_update_node


@pytest.mark.parametrize("space", ["space", "personal"])
async def test_amend_paths_respect_postgres_external_id_index(space_world: SpaceWorld, space: str):
    w = space_world
    sid = w.a if space == "space" else None
    _enforce_postgres_external_id_index(w)

    first = await w.alice.call("memory_store", "Rate is 4%", space=sid, refs=[DOC], external_id="rate")
    stored = await w.alice.call("memory_store", "Rate is 5%", space=sid, refs=[DOC], external_id="rate")
    assert stored["status"] == "amended" and stored["superseded"] == first["node_id"], stored
    amended = await w.alice.call("memory_amend", stored["node_id"], "Rate is 6%")
    assert amended["status"] == "amended", amended
    batch = await w.alice.call("memory_store_batch", [{"content": "Rate is 7%", "refs": [DOC], "external_id": "rate"}],
                               space=sid)
    assert batch["results"][0]["status"] == "amended", batch
    retired = await w.alice.call("memory_retire_by_ref", "document", "doc_1", space=sid)
    assert retired == {"retired": 1, "updated": 0, "deleted": 0}, retired
    current = await _current(w, "rate", sid)
    assert current.content_full.startswith("[Retired] Rate is 7%")
    assert await _supersedes(w, str(current.id)) == [batch["results"][0]["node_id"]]
    assert len([n for n in _with_external_id(w, sid, "rate")]) == 5  # history keeps the id
