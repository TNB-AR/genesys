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
