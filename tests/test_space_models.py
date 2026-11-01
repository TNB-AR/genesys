"""Mind Space models, enums, node fields, context variables, config and the
conftest fixtures (WS-L1), plus loading old state files."""
from __future__ import annotations

import contextvars
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import papez.context as ctx
from papez.engine import config
from papez.models import (
    ForgettingPolicy,
    MemoryNode,
    PrincipalType,
    Space,
    SpaceMember,
    SpaceRole,
)
from papez.models.enums import SPACE_ROLE_RANK, Visibility
from papez.models.space import SLUG_PATTERN
from papez.storage.memory import InMemoryGraphProvider

FIXTURES = Path(__file__).parent / "fixtures"


# -- Enums --------------------------------------------------------------------


def test_enum_values():
    assert [r.value for r in SpaceRole] == ["owner", "writer", "reader"]
    assert [p.value for p in ForgettingPolicy] == ["retain_all", "standard"]
    assert [p.value for p in PrincipalType] == ["user", "service"]
    assert SpaceRole("writer") is SpaceRole.WRITER
    assert SpaceRole.OWNER == "owner"  # str enum: compares equal to context-var strings


def test_space_role_rank_orders_owner_writer_reader():
    assert SPACE_ROLE_RANK == {SpaceRole.READER: 0, SpaceRole.WRITER: 1, SpaceRole.OWNER: 2}
    assert SPACE_ROLE_RANK[SpaceRole("owner")] > SPACE_ROLE_RANK[SpaceRole.WRITER] > SPACE_ROLE_RANK[SpaceRole.READER]
    # lookups by the plain string value work (str enum hashing)
    assert SPACE_ROLE_RANK["writer"] == 1  # type: ignore[index]


# -- Space --------------------------------------------------------------------


def _space(**kw):
    base = {"org_id": "org_r", "name": "Client A payments", "slug": "client-a-payments", "created_by": "alice"}
    base.update(kw)
    return Space(**base)


def test_space_defaults():
    s = _space()
    assert s.id.startswith("spc_") and len(s.id) == 4 + 32
    assert s.forgetting_policy is ForgettingPolicy.RETAIN_ALL
    assert s.llm_inference_enabled is True
    assert s.description is None
    assert s.archived_at is None and s.archived is False
    assert s.created_at.tzinfo is not None
    assert _space().id != s.id


def test_space_archived_property():
    s = _space(archived_at=datetime.now(UTC))
    assert s.archived is True
    assert "archived" not in s.model_dump()  # property, not a stored field


@pytest.mark.parametrize("slug", ["ab", "client-a", "a1", "0-9", "a" * 63, "x-"])
def test_space_slug_valid(slug):
    assert _space(slug=slug).slug == slug


@pytest.mark.parametrize("slug", ["a", "-ab", "Ab", "a_b", "a b", "a" * 64, "", "ünï"])
def test_space_slug_invalid(slug):
    with pytest.raises(ValidationError):
        _space(slug=slug)


def test_space_field_limits():
    assert SLUG_PATTERN == r"^[a-z0-9][a-z0-9-]{1,62}$"
    _space(name="n" * 120, description="d" * 1000)
    for bad in ({"name": ""}, {"name": "n" * 121}, {"description": "d" * 1001}, {"forgetting_policy": "never"}):
        with pytest.raises(ValidationError):
            _space(**bad)
    with pytest.raises(ValidationError):
        Space(name="x", slug="xx", created_by="a")  # org_id required
    with pytest.raises(ValidationError):
        Space(org_id="o", name="x", slug="xx")  # created_by required


def test_space_json_round_trip():
    s = _space(forgetting_policy="standard", llm_inference_enabled=False, archived_at=datetime.now(UTC))
    dumped = s.model_dump(mode="json")
    assert dumped["forgetting_policy"] == "standard"
    again = Space.model_validate(dumped)
    assert again == s and again.archived


# -- SpaceMember --------------------------------------------------------------


def test_space_member():
    m = SpaceMember(space_id="spc_1", principal_id="svc_a", principal_type="service", role="reader", added_by="alice")
    assert m.principal_type is PrincipalType.SERVICE
    assert m.role is SpaceRole.READER
    assert m.added_at.tzinfo is not None
    assert SpaceMember.model_validate(m.model_dump(mode="json")) == m
    with pytest.raises(ValidationError):
        SpaceMember(space_id="s", principal_id="p", principal_type="robot", role="reader", added_by="a")
    with pytest.raises(ValidationError):
        SpaceMember(space_id="s", principal_id="p", principal_type="user", role="admin", added_by="a")


# -- MemoryNode fields --------------------------------------------------------


def test_memory_node_new_field_defaults():
    n = MemoryNode(content_summary="x")
    assert n.refs == [] and n.tags == []
    assert n.space_id is None and n.external_id is None and n.promoted_from is None


def test_memory_node_external_id_max_length():
    assert MemoryNode(content_summary="x", external_id="e" * 256).external_id == "e" * 256
    with pytest.raises(ValidationError):
        MemoryNode(content_summary="x", external_id="e" * 257)


def test_space_node_must_be_private_without_org():
    n = MemoryNode(content_summary="x", space_id="spc_1", promoted_from=str(uuid.uuid4()))
    assert n.visibility is Visibility.PRIVATE and n.org_id is None
    with pytest.raises(ValidationError):
        MemoryNode(content_summary="x", space_id="spc_1", visibility="org")
    with pytest.raises(ValidationError):
        MemoryNode(content_summary="x", space_id="spc_1", org_id="org_r")
    # personal/org nodes are unaffected
    assert MemoryNode(content_summary="x", visibility="org", org_id="org_r").space_id is None


# -- Old state files ----------------------------------------------------------


def _load(tmp_path: Path, fixture: str) -> InMemoryGraphProvider:
    path = tmp_path / "state.json"
    shutil.copy(FIXTURES / fixture, path)
    graph = InMemoryGraphProvider(persist_path=str(path))
    graph._load()
    return graph


def test_old_state_flat_file_loads_with_defaults(tmp_path):
    token = ctx.current_user_id.set("migrated-user")
    try:
        graph = _load(tmp_path, "old_state_v1.json")
    finally:
        ctx.current_user_id.reset(token)
    nodes = graph._user_nodes["migrated-user"]
    assert len(nodes) == 2
    for n in nodes.values():
        assert n.refs == [] and n.tags == []
        assert n.space_id is None and n.external_id is None and n.promoted_from is None
    assert len(graph._user_edges["migrated-user"]) == 1


def test_old_state_per_user_file_loads_with_defaults(tmp_path):
    graph = _load(tmp_path, "old_state_per_user.json")
    nodes = graph._user_nodes["legacy-user"]
    assert len(nodes) == 2
    assert {n.visibility for n in nodes.values()} == {Visibility.PRIVATE, Visibility.ORG}
    for n in nodes.values():
        assert n.refs == [] and n.tags == [] and n.space_id is None
    assert len(graph._user_edges["legacy-user"]) == 1


def test_fixtures_have_no_new_fields():
    for name in ("old_state_v1.json", "old_state_per_user.json"):
        text = (FIXTURES / name).read_text()
        for field in ('"refs"', '"tags"', '"space_id"', '"external_id"', '"promoted_from"'):
            assert field not in text, (name, field)


# -- Context variables and config ---------------------------------------------


def test_new_context_variable_defaults():
    def read():
        assert ctx.current_principal_type.get("user") == "user"
        assert ctx.current_space_roles.get({}) == {}
        assert ctx.current_scope_space_ids.get([]) == []
        assert ctx.current_include_personal.get() is True
        assert ctx.current_client.get({}) == {}
        # no mutable defaults: unset reads without a fallback raise
        for var in (ctx.current_principal_type, ctx.current_space_roles, ctx.current_scope_space_ids, ctx.current_client):
            with pytest.raises(LookupError):
                var.get()

    contextvars.Context().run(read)


def test_new_config_defaults():
    assert config.AUDIT_MAX_EVENTS == 100000
    assert config.BATCH_MAX_ITEMS == 100
    assert config.AUDIT_READ_NODE_IDS_CAP == 50


def test_config_reads_genesys_env(monkeypatch):
    import importlib

    monkeypatch.setenv("GENESYS_MAX_TAGS_PER_MEMORY", "7")
    monkeypatch.setenv("GENESYS_BATCH_MAX_ITEMS", "3")
    try:
        importlib.reload(config)
        assert config.MAX_TAGS_PER_MEMORY == 7
        assert config.BATCH_MAX_ITEMS == 3
    finally:
        monkeypatch.undo()
        importlib.reload(config)
    assert config.MAX_TAGS_PER_MEMORY == 20


# -- conftest fixtures --------------------------------------------------------


def test_environment_is_isolated():
    assert os.environ["OPENAI_API_KEY"] == ""
    assert os.environ["ANTHROPIC_API_KEY"] == ""
    assert os.environ["DATABASE_URL"] == ""
    persist = Path(os.environ["GENESYS_PERSIST_PATH"])
    repo = Path(__file__).resolve().parent.parent
    assert repo not in persist.resolve().parents


def test_as_principal_sets_and_resets_everything(as_principal):
    def run():
        with as_principal("alice", orgs=("org_r", "org_x"), roles={"spc_a": SpaceRole.WRITER, "spc_b": "reader"}):
            assert ctx.current_user_id.get() == "alice"
            assert ctx.current_org_id.get() == "org_r"
            assert ctx.current_org_ids.get() == ["org_r", "org_x"]
            assert ctx.current_principal_type.get() == "user"
            assert ctx.current_space_roles.get() == {"spc_a": "writer", "spc_b": "reader"}
            assert ctx.current_scope_space_ids.get() == ["spc_a", "spc_b"]
            assert ctx.current_include_personal.get() is True
            assert ctx.current_client.get()["transport"] == "stdio"
            with as_principal("svc_a", orgs=["org_r"], roles={"spc_a": "reader"}, principal_type="service"):
                assert ctx.current_user_id.get() == "svc_a"
                assert ctx.current_include_personal.get() is False
            assert ctx.current_user_id.get() == "alice"
            assert ctx.current_include_personal.get() is True
        with pytest.raises(LookupError):
            ctx.current_user_id.get()
        assert ctx.current_space_roles.get({}) == {}
        assert ctx.current_org_ids.get([]) == []
        assert ctx.current_include_personal.get() is True

    contextvars.Context().run(run)


def test_as_principal_include_personal_override(as_principal):
    with as_principal("bob", include_personal=False):
        assert ctx.current_include_personal.get() is False
        assert ctx.current_org_id.get() is None and ctx.current_org_ids.get() == []


async def test_fake_embedder_is_deterministic(fake_embedder):
    a = await fake_embedder.embed("hello")
    assert a == await fake_embedder.embed("hello")
    assert len(a) == fake_embedder.dimension == 16
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert await fake_embedder.embed_batch(["hello"]) == [a]


async def test_forbidden_llm_fails_and_records(forbidden_llm):
    with pytest.raises(pytest.fail.Exception):
        await forbidden_llm.extract_entities("secret")
    assert forbidden_llm.calls == [("extract_entities", ("secret",))]
    # Failed is a BaseException: an engine `except Exception` cannot swallow it
    assert not issubclass(pytest.fail.Exception, Exception)


async def test_recording_llm_records(recording_llm):
    assert await recording_llm.detect_contradiction("a", "b") == (False, 0.0, None)
    assert await recording_llm.infer_causal_edges("n", []) == []
    assert recording_llm.names() == ["detect_contradiction", "infer_causal_edges"]
