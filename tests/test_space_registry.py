"""InMemorySpaceRegistry (spec 6.8, 7.5; D05) and the providers wiring."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from papez.models import PrincipalType, Space, SpaceMember, SpaceRole
from papez.storage.base import SpaceRegistry
from papez.storage.spaces import InMemorySpaceRegistry, slugify, unique_slug


def _space(slug="client-a", org="org_a", name=None, **kw) -> Space:
    return Space(org_id=org, name=name or slug.replace("-", " ").title(), slug=slug, created_by="u_owner", **kw)


def _member(space_id, pid, role=SpaceRole.WRITER, ptype=PrincipalType.USER) -> SpaceMember:
    return SpaceMember(space_id=space_id, principal_id=pid, principal_type=ptype, role=role, added_by="u_owner")


async def _space_with_owner(reg, slug="client-a", org="org_a", owner="u_owner", **kw) -> Space:
    s = await reg.create_space(_space(slug, org, **kw))
    await reg.add_member(_member(s.id, owner, SpaceRole.OWNER))
    return s


@pytest.fixture
async def reg() -> InMemorySpaceRegistry:
    r = InMemorySpaceRegistry()
    await r.initialize()
    return r


def test_satisfies_protocol():
    r: SpaceRegistry = InMemorySpaceRegistry()
    for name in (
        "initialize", "create_space", "get_space", "get_space_by_slug", "list_spaces_for_principal",
        "list_spaces_for_org", "update_space", "mark_erased", "add_member", "remove_member",
        "list_members", "count_members", "roles_for_principal",
    ):
        assert callable(getattr(r, name))


# ── CRUD ─────────────────────────────────────────────────────────────────

async def test_create_get_and_slug_lookup(reg):
    s = await reg.create_space(_space("client-a"))
    assert s.id.startswith("spc_")
    assert (await reg.get_space(s.id)).slug == "client-a"
    assert (await reg.get_space_by_slug("org_a", "client-a")).id == s.id
    assert await reg.get_space_by_slug("org_b", "client-a") is None
    assert await reg.get_space("spc_missing") is None


async def test_returned_objects_are_copies(reg):
    s = await reg.create_space(_space("client-a"))
    s.name = "mutated"
    got = await reg.get_space(s.id)
    assert got.name != "mutated"


async def test_slug_taken_within_org_only(reg):
    await reg.create_space(_space("client-a", "org_a"))
    with pytest.raises(ValueError, match="^slug_taken$"):
        await reg.create_space(_space("client-a", "org_a"))
    other = await reg.create_space(_space("client-a", "org_b"))
    assert other.org_id == "org_b"


async def test_list_spaces_for_principal_and_org(reg):
    a = await _space_with_owner(reg, "bravo", name="Bravo")
    b = await _space_with_owner(reg, "alpha", name="Alpha")
    c = await _space_with_owner(reg, "charlie", owner="u_other", name="Charlie")
    await reg.add_member(_member(c.id, "u_owner", SpaceRole.READER))
    listed = await reg.list_spaces_for_principal("u_owner")
    assert [(s.slug, r) for s, r in listed] == [
        ("alpha", SpaceRole.OWNER), ("bravo", SpaceRole.OWNER), ("charlie", SpaceRole.READER),
    ]
    assert await reg.list_spaces_for_principal("nobody") == []
    assert [s.id for s in await reg.list_spaces_for_org("org_a")] == [b.id, a.id, c.id]
    assert await reg.list_spaces_for_org("org_b") == []


async def test_update_space(reg):
    s = await _space_with_owner(reg)
    updated = await reg.update_space(s.id, {"name": "New name", "description": "d", "llm_inference_enabled": False,
                                             "forgetting_policy": "standard"})
    assert updated.name == "New name"
    assert updated.llm_inference_enabled is False
    assert updated.forgetting_policy.value == "standard"
    assert (await reg.get_space(s.id)).name == "New name"


async def test_update_space_slug_is_immutable(reg):
    s = await _space_with_owner(reg)
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        await reg.update_space(s.id, {"slug": "other"})
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        await reg.update_space(s.id, {"id": "spc_x"})
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        await reg.update_space(s.id, {"name": ""})
    with pytest.raises(ValueError, match="^space_not_found$"):
        await reg.update_space("spc_missing", {"name": "x"})
    assert (await reg.get_space(s.id)).slug == "client-a"


# ── archived and erased filtering ────────────────────────────────────────

async def test_archived_filtering(reg):
    s = await _space_with_owner(reg)
    live = await _space_with_owner(reg, "live")
    await reg.update_space(s.id, {"archived_at": datetime.now(UTC)})
    assert (await reg.get_space(s.id)).archived
    assert [x.id for x, _ in await reg.list_spaces_for_principal("u_owner")] == [live.id]
    assert {x.id for x, _ in await reg.list_spaces_for_principal("u_owner", include_archived=True)} == {s.id, live.id}
    assert set(await reg.roles_for_principal("u_owner")) == {live.id}
    assert set(await reg.roles_for_principal("u_owner", include_archived=True)) == {s.id, live.id}
    assert [x.id for x in await reg.list_spaces_for_org("org_a")] == [live.id]
    assert len(await reg.list_spaces_for_org("org_a", include_archived=True)) == 2
    # Archived spaces stay addressable by id and slug.
    assert (await reg.get_space_by_slug("org_a", "client-a")).id == s.id
    # Unarchive.
    await reg.update_space(s.id, {"archived_at": None})
    assert s.id in await reg.roles_for_principal("u_owner")


async def test_erased_spaces_vanish(reg):
    s = await _space_with_owner(reg)
    await reg.add_member(_member(s.id, "u_w", SpaceRole.WRITER))
    await reg.mark_erased(s.id)
    assert await reg.get_space(s.id) is None
    assert await reg.get_space_by_slug("org_a", "client-a") is None
    assert await reg.list_spaces_for_principal("u_owner", include_archived=True) == []
    assert await reg.list_spaces_for_org("org_a", include_archived=True) == []
    assert await reg.roles_for_principal("u_owner", include_archived=True) == {}
    assert await reg.roles_for_principal("u_w") == {}
    assert await reg.list_members(s.id) == []
    assert await reg.count_members(s.id) == 0
    assert reg.erased_at(s.id) is not None
    # Row kept: id and slug are not reused.
    with pytest.raises(ValueError, match="^slug_taken$"):
        await reg.create_space(_space("client-a"))
    with pytest.raises(ValueError, match="^space_not_found$"):
        await reg.add_member(_member(s.id, "u_new"))
    await reg.mark_erased(s.id)  # idempotent
    with pytest.raises(ValueError, match="^space_not_found$"):
        await reg.mark_erased("spc_missing")


# ── members ──────────────────────────────────────────────────────────────

async def test_member_upsert_list_count_roles(reg):
    s = await _space_with_owner(reg)
    m = await reg.add_member(_member(s.id, "u_r", SpaceRole.READER))
    assert m.role == SpaceRole.READER
    again = await reg.add_member(_member(s.id, "u_r", SpaceRole.WRITER))
    assert again.role == SpaceRole.WRITER
    assert again.added_at == m.added_at
    assert await reg.count_members(s.id) == 2
    assert {x.principal_id: x.role for x in await reg.list_members(s.id)} == {
        "u_owner": SpaceRole.OWNER, "u_r": SpaceRole.WRITER,
    }
    assert await reg.roles_for_principal("u_r") == {s.id: SpaceRole.WRITER}
    await reg.remove_member(s.id, "u_r")
    assert await reg.roles_for_principal("u_r") == {}
    assert await reg.count_members(s.id) == 1
    await reg.remove_member(s.id, "u_never")  # removing a non-member is a no-op
    assert await reg.get_member(s.id, "u_owner") is not None


async def test_service_principal_cannot_be_owner(reg):
    s = await _space_with_owner(reg)
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        await reg.add_member(_member(s.id, "svc_1", SpaceRole.OWNER, PrincipalType.SERVICE))
    svc = await reg.add_member(_member(s.id, "svc_1", SpaceRole.WRITER, PrincipalType.SERVICE))
    assert svc.principal_type == PrincipalType.SERVICE
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        await reg.add_member(_member(s.id, "svc_1", SpaceRole.OWNER, PrincipalType.SERVICE))


async def test_last_owner_on_remove_and_demote(reg):
    s = await _space_with_owner(reg)
    await reg.add_member(_member(s.id, "svc_1", SpaceRole.WRITER, PrincipalType.SERVICE))
    with pytest.raises(ValueError, match="^last_owner$"):
        await reg.remove_member(s.id, "u_owner")
    with pytest.raises(ValueError, match="^last_owner$"):
        await reg.add_member(_member(s.id, "u_owner", SpaceRole.WRITER))
    # Re-asserting owner on the last owner is fine.
    await reg.add_member(_member(s.id, "u_owner", SpaceRole.OWNER))
    # With a second user owner both operations are allowed.
    await reg.add_member(_member(s.id, "u_two", SpaceRole.OWNER))
    demoted = await reg.add_member(_member(s.id, "u_owner", SpaceRole.READER))
    assert demoted.role == SpaceRole.READER
    with pytest.raises(ValueError, match="^last_owner$"):
        await reg.remove_member(s.id, "u_two")
    await reg.add_member(_member(s.id, "u_owner", SpaceRole.OWNER))
    await reg.remove_member(s.id, "u_two")
    assert await reg.roles_for_principal("u_two") == {}


# ── persistence ──────────────────────────────────────────────────────────

async def test_no_io_in_constructor(tmp_path):
    path = tmp_path / "sub" / "memories.spaces.json"
    InMemorySpaceRegistry(persist_path=path)
    assert not path.parent.exists()


async def test_persistence_round_trip(tmp_path):
    path = tmp_path / "memories.spaces.json"
    r1 = InMemorySpaceRegistry(persist_path=path)
    await r1.initialize()
    assert not path.exists()  # created lazily on the first mutation
    s = await _space_with_owner(r1, description="desc")
    await r1.add_member(_member(s.id, "svc_1", SpaceRole.READER, PrincipalType.SERVICE))
    archived = await _space_with_owner(r1, "old")
    await r1.update_space(archived.id, {"archived_at": datetime.now(UTC)})
    gone = await _space_with_owner(r1, "gone")
    await r1.mark_erased(gone.id)

    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert set(data["spaces"]) == {s.id, archived.id, gone.id}
    assert data["spaces"][gone.id]["erased_at"]
    assert isinstance(data["members"], list)

    r2 = InMemorySpaceRegistry(persist_path=path)
    await r2.initialize()
    got = await r2.get_space(s.id)
    assert got == await r1.get_space(s.id)
    assert got.description == "desc"
    assert (await r2.get_space(archived.id)).archived
    assert await r2.get_space(gone.id) is None
    assert await r2.roles_for_principal("u_owner") == {s.id: SpaceRole.OWNER}
    assert await r2.roles_for_principal("svc_1") == {s.id: SpaceRole.READER}
    assert {m.principal_id for m in await r2.list_members(s.id)} == {"u_owner", "svc_1"}
    with pytest.raises(ValueError, match="^slug_taken$"):
        await r2.create_space(_space("gone"))
    with pytest.raises(ValueError, match="^last_owner$"):
        await r2.remove_member(s.id, "u_owner")


async def test_mutation_before_initialize_keeps_existing_file(tmp_path):
    path = tmp_path / "memories.spaces.json"
    r1 = InMemorySpaceRegistry(persist_path=path)
    first = await r1.create_space(_space("first"))
    r2 = InMemorySpaceRegistry(persist_path=path)
    second = await r2.create_space(_space("second"))  # no initialize() call
    r3 = InMemorySpaceRegistry(persist_path=path)
    await r3.initialize()
    assert await r3.get_space(first.id) is not None
    assert await r3.get_space(second.id) is not None


# ── slug helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(("name", "expected"), [
    ("Client A Payments", "client-a-payments"),
    ("Client A — Payments", "client-a-payments"),
    ("  Hello,   World!  ", "hello-world"),
    ("ACME_corp 2026", "acmecorp-2026"),
    ("---", ""),
])
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_unique_slug():
    assert unique_slug("Client A", []) == "client-a"
    assert unique_slug("Client A", {"client-a"}) == "client-a-2"
    assert unique_slug("client-a", {"client-a", "client-a-2"}) == "client-a-3"
    assert unique_slug("!!!", []) == "space"
    assert unique_slug("x", []) == "x-1"
    long = "a" * 80
    got = unique_slug(long, {"a" * 63})
    assert len(got) <= 63 and got.endswith("-2")
    import re

    from papez.models.space import SLUG_PATTERN
    for candidate in (unique_slug("Ünïcødé ✓", []), got, unique_slug("x", [])):
        assert re.match(SLUG_PATTERN, candidate)


async def test_unique_slug_against_registry(reg):
    await reg.create_space(_space("client-a"))
    slug = unique_slug("Client A", reg.slugs_for_org("org_a"))
    s = await reg.create_space(_space(slug))
    assert s.slug == "client-a-2"


# ── providers wiring ─────────────────────────────────────────────────────

def test_registry_path_for():
    from papez.providers import registry_path_for
    assert str(registry_path_for("/x/data/memories.json")) == "/x/data/memories.spaces.json"
    assert registry_path_for(None) is None
    assert registry_path_for("") is None


def test_get_providers_builds_spaces_and_audit(monkeypatch, tmp_path):
    import papez.providers as prov
    from papez.storage.audit import InMemoryAuditSink

    persist = tmp_path / "memories.json"
    monkeypatch.setenv("GENESYS_PERSIST_PATH", str(persist))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(prov, "_instance", None)
    p = prov.get_providers()
    assert isinstance(p.spaces, InMemorySpaceRegistry)
    assert p.spaces._persist_path == tmp_path / "memories.spaces.json"
    assert isinstance(p.audit, InMemoryAuditSink)
    from papez.engine import config
    assert p.audit.max_events == config.AUDIT_MAX_EVENTS
    assert not (tmp_path / "memories.spaces.json").exists()  # no I/O at construction
    assert prov.get_providers() is p


def test_get_providers_unpersisted_graph_has_unpersisted_registry(monkeypatch):
    import papez.providers as prov

    monkeypatch.setenv("GENESYS_PERSIST_PATH", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(prov, "_instance", None)
    p = prov.get_providers()
    assert p.spaces._persist_path is None
