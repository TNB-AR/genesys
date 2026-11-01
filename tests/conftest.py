"""Shared test infrastructure (D31).

Everything here is OPT-IN except ``_isolated_environment``: request the
fixtures by name, or import the classes (``from conftest import ForbiddenLLM``).

Credential safety (C35): ``papez.server`` builds its providers at import time
and ``papez.providers`` calls ``load_dotenv()``, which would read the repo's
``.env`` (real API keys and a DATABASE_URL). python-dotenv never overrides a
variable that is already present, so blanking the variables BEFORE any test
module is collected keeps the real values out of the test process. That has
to happen at conftest import (collection imports test modules, which import
``papez.server``, before any fixture runs); the session fixture below re-applies
the values, owns the temporary persist directory and restores the environment
at the end of the session.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from papez.context import (
    current_client,
    current_include_personal,
    current_org_id,
    current_org_ids,
    current_principal_type,
    current_scope_space_ids,
    current_space_roles,
    current_user_id,
    current_user_role,
)
from papez.models.enums import EdgeType

# ---------------------------------------------------------------------------
# Environment isolation (autouse, session)
# ---------------------------------------------------------------------------

_BLANKED_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DATABASE_URL")
# Env var read by papez.providers.get_providers for the in-memory state file.
_PERSIST_VAR = "GENESYS_PERSIST_PATH"

_SAVED_ENV: dict[str, str | None] = {k: os.environ.get(k) for k in (*_BLANKED_VARS, _PERSIST_VAR)}
_SESSION_TMP = tempfile.mkdtemp(prefix="papez-tests-")


def _apply_isolated_env() -> None:
    for key in _BLANKED_VARS:
        os.environ[key] = ""
    os.environ[_PERSIST_VAR] = os.path.join(_SESSION_TMP, "memories.json")


_apply_isolated_env()  # before any test module (and so papez.server) is imported


@pytest.fixture(autouse=True, scope="session")
def _isolated_environment() -> Iterator[str]:
    """Blank credentials, point the persist path at a temp dir; restore afterwards."""
    _apply_isolated_env()
    try:
        yield _SESSION_TMP
    finally:
        for key, value in _SAVED_ENV.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(_SESSION_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Acting as a principal
# ---------------------------------------------------------------------------

_DEFAULT_CLIENT: dict[str, str | None] = {"transport": "stdio", "client_name": None}


@contextmanager
def principal(
    uid: str,
    orgs: Sequence[str] = (),
    roles: Mapping[str, Any] | None = None,
    principal_type: str = "user",
    include_personal: bool | None = None,
    client: Mapping[str, str | None] | None = None,
) -> Iterator[None]:
    """Set every identity/scope context variable for the block, then reset them.

    ``roles`` maps space id -> role (``SpaceRole`` or its string value).
    ``current_scope_space_ids`` is set to every space in ``roles`` (what the
    tool layer does when no ``space``/``spaces`` is named).
    ``include_personal=None`` means ``False`` for services, ``True`` for users.
    """
    role_map = {sid: str(getattr(r, "value", r)) for sid, r in (roles or {}).items()}
    if include_personal is None:
        include_personal = principal_type != "service"
    tokens: list[tuple[Any, Any]] = [
        (current_user_id, current_user_id.set(uid)),
        (current_org_id, current_org_id.set(orgs[0] if orgs else None)),
        (current_org_ids, current_org_ids.set(list(orgs))),
        (current_user_role, current_user_role.set(None)),
        (current_principal_type, current_principal_type.set(principal_type)),
        (current_space_roles, current_space_roles.set(role_map)),
        (current_scope_space_ids, current_scope_space_ids.set(list(role_map))),
        (current_include_personal, current_include_personal.set(include_personal)),
        (current_client, current_client.set(dict(client) if client is not None else dict(_DEFAULT_CLIENT))),
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@pytest.fixture()
def as_principal() -> Callable[..., Any]:
    """``with as_principal("alice", orgs=["org_r"], roles={sid: "writer"}): ...``"""
    return principal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic 16-dim embedder (stable across processes: sha256, not hash())."""

    dimension = 16

    async def embed(self, text: str) -> list[float]:
        h = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vec = [(((h >> (i * 2)) & 3) - 1.5) / 1.5 for i in range(self.dimension)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class ForbiddenLLM:
    """An LLMProvider that fails the test on ANY call (and records it first).

    ``pytest.fail`` raises ``Failed`` (a ``BaseException``), so the engine's
    ``except Exception`` blocks cannot swallow it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _forbid(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        pytest.fail(f"ForbiddenLLM.{name} was called", pytrace=True)

    async def consolidate(self, episodic_memories: list[str]) -> str:
        self._forbid("consolidate", episodic_memories)
        raise AssertionError("unreachable")

    async def detect_contradiction(self, memory_a: str, memory_b: str) -> tuple[bool, float, str | None]:
        self._forbid("detect_contradiction", memory_a, memory_b)
        raise AssertionError("unreachable")

    async def infer_causal_edges(
        self, new_memory: str, existing_memories: list[tuple[str, str]],
    ) -> list[tuple[str, EdgeType, float, str | None]]:
        self._forbid("infer_causal_edges", new_memory, existing_memories)
        raise AssertionError("unreachable")

    async def extract_entities(self, text: str) -> list[str]:
        self._forbid("extract_entities", text)
        raise AssertionError("unreachable")

    async def generate_summary(self, text: str) -> str:
        self._forbid("generate_summary", text)
        raise AssertionError("unreachable")

    async def classify_category(self, text: str) -> str | None:
        self._forbid("classify_category", text)
        raise AssertionError("unreachable")


class RecordingLLM:
    """Positive control: records every call and returns inert results."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def consolidate(self, episodic_memories: list[str]) -> str:
        self.calls.append(("consolidate", (episodic_memories,)))
        return " ".join(episodic_memories)[:200]

    async def detect_contradiction(self, memory_a: str, memory_b: str) -> tuple[bool, float, str | None]:
        self.calls.append(("detect_contradiction", (memory_a, memory_b)))
        return (False, 0.0, None)

    async def infer_causal_edges(
        self, new_memory: str, existing_memories: list[tuple[str, str]],
    ) -> list[tuple[str, EdgeType, float, str | None]]:
        self.calls.append(("infer_causal_edges", (new_memory, existing_memories)))
        return []

    async def extract_entities(self, text: str) -> list[str]:
        self.calls.append(("extract_entities", (text,)))
        return []

    async def generate_summary(self, text: str) -> str:
        self.calls.append(("generate_summary", (text,)))
        return text[:200]

    async def classify_category(self, text: str) -> str | None:
        self.calls.append(("classify_category", (text,)))
        return None


@pytest.fixture()
def forbidden_llm() -> ForbiddenLLM:
    return ForbiddenLLM()


@pytest.fixture()
def recording_llm() -> RecordingLLM:
    return RecordingLLM()


# ---------------------------------------------------------------------------
# space_world (WS-L5): org org_r, users alice/bob, service svc_a, spaces A/B
# ---------------------------------------------------------------------------


class TokenEmbedder:
    """Bag-of-words embedder: near-identical texts get near-identical vectors.

    Unlike ``FakeEmbedder`` (hash of the whole text), this makes "Bank A pilot
    starts in Q1" and "... Q2" nearest neighbours, so a scope leak would show
    up in vector search, auto-link and conflict hints.
    """

    dimension = 64

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            token = token.strip(".,!?;:'\"()")
            if not token:
                continue
            h = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
            vec[h % self.dimension] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


WORLD_ORG = "org_r"


class Actor:
    """Acts as one principal against the shared world.

    Every call reloads the principal's roles from the registry first, exactly
    as the stdio transport does per call (``server.call_tool``), so membership
    changes take effect on the next call.
    """

    def __init__(self, world: SpaceWorld, uid: str, principal_type: str = "user", orgs: Sequence[str] = (WORLD_ORG,)):
        self.world = world
        self.uid = uid
        self.principal_type = principal_type
        self.orgs = list(orgs)

    async def roles(self) -> dict[str, str]:
        roles = await self.world.registry.roles_for_principal(self.uid)
        return {sid: str(getattr(r, "value", r)) for sid, r in roles.items()}

    @contextmanager
    def _ctx(self, roles: Mapping[str, str]) -> Iterator[None]:
        with principal(self.uid, orgs=self.orgs, roles=roles, principal_type=self.principal_type):
            yield

    async def call(self, tool: str, *args: Any, **kwargs: Any) -> Any:
        roles = await self.roles()
        with self._ctx(roles):
            return await getattr(self.world.handler, tool)(*args, **kwargs)

    async def storage(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a graph method directly with this principal's full membership scope."""
        roles = await self.roles()
        with self._ctx(roles):
            return await getattr(self.world.graph, method)(*args, **kwargs)


class SpaceWorld:
    """Shared graph + registry + audit sink + one handler, three actors."""

    def __init__(self, graph: Any, registry: Any, audit: Any, handler: Any) -> None:
        self.graph = graph
        self.registry = registry
        self.audit = audit
        self.handler = handler
        self.org = WORLD_ORG
        self.alice = Actor(self, "alice")
        self.bob = Actor(self, "bob")
        self.svc = Actor(self, "svc_a", principal_type="service")
        self.space_a: Any = None
        self.space_b: Any = None
        # label -> node id for the seeded content
        self.ids: dict[str, str] = {}

    def actor(self, uid: str, principal_type: str = "user", orgs: Sequence[str] = (WORLD_ORG,)) -> Actor:
        return Actor(self, uid, principal_type, orgs)

    @property
    def a(self) -> str:
        return str(self.space_a.id)

    @property
    def b(self) -> str:
        return str(self.space_b.id)

    def ids_in(self, *labels: str) -> set[str]:
        return {self.ids[label] for label in labels}

    @property
    def a_ids(self) -> set[str]:
        return {v for k, v in self.ids.items() if k.startswith("a_")}

    @property
    def b_ids(self) -> set[str]:
        return {v for k, v in self.ids.items() if k.startswith("b_")}

    @property
    def personal_ids(self) -> set[str]:
        return {v for k, v in self.ids.items() if k.startswith("p_")}


async def build_space_world(embedder: Any | None = None, llm: Any | None = None) -> SpaceWorld:
    """Org ``org_r``; spaces A (alice owner, svc_a writer) and B (bob owner).

    Seeds near-identical content with shared tags, entities and refs in A, B
    and alice's personal memory, plus one same-scope edge in each.
    """
    from papez.mcp.tools import MCPToolHandler
    from papez.models.enums import PrincipalType, SpaceRole
    from papez.models.space import SpaceMember
    from papez.storage.audit import InMemoryAuditSink
    from papez.storage.memory import InMemoryCacheProvider, InMemoryGraphProvider
    from papez.storage.spaces import InMemorySpaceRegistry

    graph = InMemoryGraphProvider()
    registry = InMemorySpaceRegistry()
    audit = InMemoryAuditSink()
    handler = MCPToolHandler(
        graph=graph,
        embeddings=embedder if embedder is not None else TokenEmbedder(),
        cache=InMemoryCacheProvider(),
        llm=llm,
        spaces=registry,
        audit=audit,
    )
    world = SpaceWorld(graph, registry, audit, handler)

    created_a = await world.alice.call("space_create", "Space A", slug="space-a")
    created_b = await world.bob.call("space_create", "Space B", slug="space-b")
    world.space_a = await registry.get_space(created_a["space"]["id"])
    world.space_b = await registry.get_space(created_b["space"]["id"])
    await registry.add_member(SpaceMember(
        space_id=world.a, principal_id="svc_a", principal_type=PrincipalType.SERVICE,
        role=SpaceRole.WRITER, added_by="alice",
    ))

    shared_refs = [{"type": "document", "id": "doc_bank_a", "label": "Bank A pilot plan"}]
    shared_tags = ["bank-a", "pilot"]

    async def seed(actor: Actor, label: str, content: str, space: str | None, **extra: Any) -> None:
        result = await actor.call(
            "memory_store", content, space=space, tags=shared_tags, refs=shared_refs, **extra,
        )
        assert "error" not in result, result
        world.ids[label] = result["node_id"]

    await seed(world.alice, "a_q1", "Bank A pilot starts in Q1", world.a)
    await seed(world.alice, "a_ctx", "Bank A pilot budget approved for the payments team", world.a)
    await seed(world.bob, "b_q2", "Bank A pilot starts in Q2", world.b)
    await seed(world.bob, "b_ctx", "Bank A pilot budget approved for the payments team", world.b)
    await seed(world.alice, "p_notes", "Bank A pilot notes", None)
    await seed(world.alice, "p_more", "Bank A pilot budget approved for the payments team", None)

    # Same-scope explicit edges (never across scopes) and shared entity names.
    from papez.models.edge import MemoryEdge

    pairs = [("alice", "a_ctx", "a_q1"), ("bob", "b_ctx", "b_q2"), ("alice", "p_more", "p_notes")]
    for uid, src, dst in pairs:
        actor = world.alice if uid == "alice" else world.bob
        roles = await actor.roles()
        with principal(uid, orgs=[WORLD_ORG], roles=roles):
            await graph.create_edge(MemoryEdge(
                source_id=world.ids[src], target_id=world.ids[dst], type=EdgeType.CAUSED_BY,
                weight=0.9, created_by="user_explicit",
            ))
            await graph.update_node(world.ids[src], {"entity_refs": ["Bank A", "Payments"]})
            await graph.update_node(world.ids[dst], {"entity_refs": ["Bank A", "Payments"]})
    return world


@pytest.fixture()
async def space_world() -> SpaceWorld:
    return await build_space_world()
