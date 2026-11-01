from __future__ import annotations

import inspect
import logging
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from papez.context import (
    current_client,
    current_include_personal,
    current_org_ids,
    current_principal_type,
    current_scope_space_ids,
    current_space_roles,
    current_user_id,
    current_user_role,
)
from papez.core_memory.preferences import CoreMemoryPreferences
from papez.core_memory.promoter import evaluate_core_promotion
from papez.models.edge import MemoryEdge
from papez.models.enums import (
    SPACE_ROLE_RANK,
    EdgeType,
    ForgettingPolicy,
    MemoryStatus,
    PrincipalType,
    SpaceRole,
    Visibility,
)
from papez.models.node import MemoryNode
from papez.models.ref import Ref, ValidationFailure, normalize_tags, validate_refs
from papez.models.space import SLUG_PATTERN, Space, SpaceMember
from papez.storage.base import (
    AuditSink,
    CacheProvider,
    EmbeddingProvider,
    EventBusProvider,
    GraphStorageProvider,
    SpaceRegistry,
)

logger = logging.getLogger(__name__)


def _caller_uid() -> str | None:
    """Return the current user ID or None if not set."""
    return current_user_id.get(None)


def _caller_owns_node(node: MemoryNode) -> bool:
    """Check if the current caller is the original owner of a node.

    Raises PermissionError if current_user_id is not set. In multi-user
    mode, the node must have been created by the current user or the
    caller must be an admin within the node's org.
    """
    uid = _caller_uid()
    if uid is None:
        raise PermissionError("current_user_id not set — cannot verify ownership")
    role = current_user_role.get(None)
    if role == "admin" and node.org_id and node.org_id in current_org_ids.get([]):
        return True
    if node.original_user_id:
        return node.original_user_id == uid
    return True


def _is_edge_stale(edge: MemoryEdge) -> bool:
    from papez.engine import config
    if not edge.last_validated_at:
        return False
    days = (datetime.now(UTC) - edge.last_validated_at).days
    return days > config.EDGE_STALE_DAYS


def _stem_plural(term: str) -> str:
    """Strip a trailing plural so a keyword ILIKE substring match still catches
    the singular gold turn — I3-2. ``keyword_search`` matches ``content ILIKE
    '%term%'``, so a query term "cats" never matches a turn that says "cat", and
    "classes" never matches "class". Stemming the query term to its singular
    stem ("cat", "class") keeps matching BOTH forms (the stem is a substring of
    the plural too). Conservative: only long-enough words, never "ss"/"us"/"is"
    endings. A proper noun that gets shortened ("james"->"jame") still ILIKE-
    matches its full form, so the broadening is safe."""
    t = term
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"                         # stories -> story
    if len(t) > 4 and t.endswith(("ses", "xes", "zes", "ches", "shes")):
        return t[:-2]                               # classes -> class, boxes -> box
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]                               # cats -> cat, states -> state
    return t


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO 8601 string, assuming UTC when no timezone is given.

    Node timestamps are always tz-aware, and comparing a tz-aware datetime
    against a naive one raises TypeError — so every ISO timestamp accepted
    from a tool argument must go through this normalization.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _truncate_summary(content: str, limit: int = 200) -> str:
    """Truncate content to a word boundary. This is truncation, not an LLM summary.

    Returns short content verbatim; otherwise cuts at the last whole word that
    fits within ``limit`` (including the trailing ellipsis) so words are never
    split mid-token.
    """
    if len(content) <= limit:
        return content
    cut = content[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def _live_connectivity_factor(causal_weight: int, max_causal_weight: int, is_orphan: bool) -> float:
    """Force-2 (connectivity) computed live for legibility.

    Mirrors the Force-2 block of ``scoring.calculate_decay_score`` exactly — the
    scoring math is not changed here, only surfaced so callers can read the live
    contribution instead of the stored (possibly stale) ``decay_score``.
    """
    import math

    from papez.engine import config

    if max_causal_weight > 0:
        raw = math.log2(1 + causal_weight) / math.log2(1 + max_causal_weight)
        cf = raw ** 2
    else:
        cf = 0.0
    if is_orphan:
        return 0.0
    cf = max(cf, config.MIN_CONNECTIVITY)
    return cf


# ---------------------------------------------------------------------------
# Mind Spaces: error shapes, per-call scope, store request (spec 6-7, 10)
# ---------------------------------------------------------------------------


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Spec Section 10 error dict ``{"error", "code"}`` (plus optional extras)."""
    out: dict[str, Any] = {"error": message, "code": code}
    out.update(extra)
    return out


def _not_found(node_id: str) -> dict[str, Any]:
    """Identical for a missing node and an invisible one (spec 10 ``not_found``)."""
    return {"error": "Node not found", "code": "not_found", "node_id": node_id}


def _space_not_found() -> dict[str, Any]:
    """Never echoes the requested id/slug: a missing space and a non-member's
    space must produce byte-identical bodies (spec 6.7)."""
    return _error("space_not_found", "Space not found")


def _is_service() -> bool:
    return current_principal_type.get(PrincipalType.USER.value) == PrincipalType.SERVICE.value


def _role_value(role: Any) -> str:
    return str(getattr(role, "value", role))


def _ref_dump(ref: Ref) -> dict[str, Any]:
    return ref.model_dump(mode="json", exclude_none=True)


def _ref_identity(ref: Ref) -> tuple[str, str, str | None, str | None]:
    return (ref.type, ref.id, ref.url, ref.label)


def _matches_tags_and_ref_type(node: MemoryNode, tags_any: list[str] | None, ref_type: str | None) -> bool:
    if tags_any and not set(tags_any).intersection(node.tags):
        return False
    if ref_type and not any(r.type == ref_type for r in node.refs):
        return False
    return True


class _Scope:
    """Result of ``MCPToolHandler._resolve_scope`` (D06).

    ``error`` is a ``{"error", "code"}`` dict to return immediately, or None.
    ``space_ids``/``include_personal`` are what was put into the context for
    this call; ``spaces`` maps space id -> ``Space`` (for result labels);
    ``roles`` maps space id -> role for every space in scope. ``reset()``
    restores the context variables and is safe to call more than once.
    """

    def __init__(
        self,
        error: dict[str, Any] | None = None,
        space_ids: list[str] | None = None,
        include_personal: bool = False,
        spaces: dict[str, Space] | None = None,
        roles: dict[str, str] | None = None,
        tokens: list[tuple[Any, Any]] | None = None,
    ) -> None:
        self.error = error
        self.space_ids: list[str] = list(space_ids or [])
        self.include_personal = include_personal
        self.spaces: dict[str, Space] = dict(spaces or {})
        self.roles: dict[str, str] = dict(roles or {})
        self._tokens: list[tuple[Any, Any]] = list(tokens or [])

    def reset(self) -> None:
        while self._tokens:
            var, token = self._tokens.pop()
            var.reset(token)


@dataclass
class _StoreRequest:
    content: str
    source_session: str = ""
    related_to: list[str] | None = None
    created_at: str | None = None
    visibility: str = "private"
    org_id: str | None = None
    related: list[dict[str, Any]] | None = None
    category: str | None = None
    refs: list[Any] | None = None
    tags: list[Any] | None = None
    space_id: str | None = None
    external_id: str | None = None
    on_existing: str = "amend"


@dataclass
class _Validated:
    """Everything ``_write_memory`` needs; produced only by validation (D08)."""

    content: str
    source_session: str
    visibility: Visibility
    org_id: str | None
    category: str | None
    refs: list[Ref]
    tags: list[str]
    space_id: str | None
    created_at: datetime
    related_to: list[str] = field(default_factory=list)
    related: list[tuple[str, EdgeType]] = field(default_factory=list)
    explicit_requested: bool = False
    external_id: str | None = None
    on_existing: str = "amend"
    existing: MemoryNode | None = None
    source_agent: str = "claude"
    promoted_from: str | None = None
    supersedes: tuple[str, str] | None = None  # (old node id, reason)


_ON_EXISTING = ("skip", "amend", "error")
_SLUG_RE = re.compile(SLUG_PATTERN)
_EXTERNAL_ID_MAX = 256


class MCPToolHandler:
    def __init__(
        self,
        graph: GraphStorageProvider,
        embeddings: EmbeddingProvider | None,
        cache: CacheProvider,
        event_bus: EventBusProvider | None = None,
        on_change: Callable[..., Any] | None = None,
        llm: object | None = None,
        spaces: SpaceRegistry | None = None,
        audit: AuditSink | None = None,
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.cache = cache
        self.event_bus = event_bus
        self.on_change = on_change
        self.llm = llm
        self.spaces = spaces
        self.audit = audit
        # Optional org directory hook for space_member_add's principal_not_in_org
        # check: ``async (org_id, principal_id) -> bool``, asked for users and
        # services alike. The library has no org directory, so when this is None
        # the check is skipped (hosted sets it).
        self.principal_in_org: Callable[[str, str], Any] | None = None
        self.preferences = CoreMemoryPreferences(cache)

    def _defer_saves(self) -> AbstractContextManager[None]:
        """Return a context manager that batches saves if the graph supports it."""
        for cls in type(self.graph).__mro__:
            if 'defer_saves' in cls.__dict__:
                return self.graph.defer_saves()  # type: ignore[attr-defined,no-any-return]
        return nullcontext()

    def _atomic(self) -> AbstractAsyncContextManager[Any]:
        """``graph.atomic()`` when the provider defines one, else a no-op (D08).

        Detected on the class (like ``_defer_saves``) so a Mock graph, whose
        attributes all exist, never pretends to be transactional.
        """
        for cls in type(self.graph).__mro__:
            if 'atomic' in cls.__dict__:
                return self.graph.atomic()  # type: ignore[attr-defined,no-any-return]
        return nullcontext()

    async def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self.on_change:
            await self.on_change(event_type, data)

    async def _audit_event(
        self,
        action: str,
        space_id: str,
        node_ids: list[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record one audit event (spec 8.1, 8.2, D18). No-op without a sink.

        Never pass memory content, summaries, slugs or query text: ids,
        counts and lengths only. ``node_ids`` is capped at
        ``config.AUDIT_READ_NODE_IDS_CAP``. A failing sink is logged, never
        raised, so auditing can't break a tool call.
        """
        if self.audit is None:
            return
        from papez.engine import config
        from papez.storage.audit import audit_timestamp, new_event_id

        event: dict[str, Any] = {
            "id": new_event_id(),
            "at": audit_timestamp(),
            "space_id": space_id,
            "principal_id": _caller_uid(),
            "principal_type": current_principal_type.get(PrincipalType.USER.value),
            "action": action,
            "node_ids": list(node_ids or [])[: config.AUDIT_READ_NODE_IDS_CAP],
            "detail": dict(detail or {}),
            "client": dict(current_client.get({})),
        }
        try:
            await self.audit.record(event)
        except Exception:
            logger.warning("audit record failed for %s on %s", action, space_id, exc_info=True)

    async def _audit_denied(self, space_id: str, code: str, need: SpaceRole | None) -> None:
        """``access.denied`` against a real space (spec 8.1)."""
        await self._audit_event(
            "access.denied", space_id, None,
            {"code": code, "need_role": need.value if need is not None else None},
        )

    async def _audit_denied_ref(self, ref: str, need: SpaceRole | None) -> None:
        """``access.denied`` for a ``space_not_found`` answer, if ``ref`` names a real space.

        The caller was told "not found"; the event is logged against the real
        space id (an id anywhere, or a slug in one of the caller's orgs). A
        ref that matches no live space logs nothing.
        """
        if self.audit is None or self.spaces is None or not isinstance(ref, str) or not ref:
            return
        try:
            found = await self.spaces.get_space(ref)
            if found is None:
                for org_id in current_org_ids.get([]):
                    found = await self.spaces.get_space_by_slug(org_id, ref)
                    if found is not None:
                        break
        except Exception:
            logger.warning("could not resolve space for access.denied", exc_info=True)
            return
        if found is not None:
            await self._audit_denied(found.id, "space_not_found", need)

    async def _audit_reads(
        self,
        action: str,
        space_ids: list[str],
        hits: list[tuple[str, str | None]],
        query_chars: int,
    ) -> None:
        """One read event per space in scope, counting that space's results (spec 8.1).

        ``hits`` is ``(node_id, space_id)`` for every returned result; personal
        hits (``space_id`` None) are never audited.
        """
        if self.audit is None:
            return
        for sid in dict.fromkeys(space_ids):
            ids = [nid for nid, hit_space in hits if hit_space == sid]
            await self._audit_event(action, sid, ids, {"query_chars": query_chars, "results": len(ids)})

    @staticmethod
    def _hits_of(results: Any) -> list[tuple[str, str | None]]:
        """``(id, space id)`` pairs from recall/search result rows."""
        out: list[tuple[str, str | None]] = []
        for row in results if isinstance(results, list) else []:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            label = row.get("space")
            out.append((str(row["id"]), label.get("id") if isinstance(label, dict) else None))
        return out

    async def _audit_write_result(
        self, space_id: str | None, result: dict[str, Any], extra: dict[str, Any] | None = None,
    ) -> None:
        """``memory.store`` / ``memory.amend`` for one space write (store, batch item)."""
        if space_id is None or "error" in result or "node_id" not in result:
            return
        status = result.get("status")
        detail: dict[str, Any] = {"status": status, **(extra or {})}
        if status == "amended" and result.get("superseded"):
            await self._audit_event("memory.amend", space_id, [result["node_id"], result["superseded"]], detail)
        else:
            await self._audit_event("memory.store", space_id, [result["node_id"]], detail)

    # -- scope and authorization (D06, D07) -------------------------------------

    async def _lookup_space(self, ref: str, roles: dict[str, str]) -> tuple[Space | None, str] | None:
        """Resolve an id or slug to ``(space, role)`` for the caller, or None.

        Order: exact id in ``current_space_roles``; else (with a registry) an id
        or a slug within each of the caller's orgs, accepted only when the
        caller is a member (archived memberships come from the registry, since
        ``current_space_roles`` holds non-archived spaces only).
        """
        if ref in roles:
            if self.spaces is None:
                return None, roles[ref]
            space = await self.spaces.get_space(ref)
            if space is None:
                return None  # erased (or stale roles): indistinguishable from missing
            return space, roles[ref]
        if self.spaces is None:
            return None
        found = await self.spaces.get_space(ref)
        if found is None:
            for org_id in current_org_ids.get([]):
                found = await self.spaces.get_space_by_slug(org_id, ref)
                if found is not None:
                    break
        if found is None:
            return None
        if found.id in roles:
            return found, roles[found.id]
        uid = _caller_uid()
        if uid is None:
            return None
        all_roles = await self.spaces.roles_for_principal(uid, include_archived=True)
        role = all_roles.get(found.id)
        if role is None or (not found.archived and found.id not in roles):
            # A non-archived membership missing from current_space_roles means
            # the roles were (re)loaded without it: treat as not a member.
            return None
        return found, _role_value(role)

    async def _resolve_scope(
        self,
        space: str | None,
        spaces: list[str] | None,
        include_personal: bool | None,
        need_role: SpaceRole | str | None,
        default_scope: str = "memberships",
        *,
        allow_archived: bool | None = None,
        include_archived_memberships: bool = False,
    ) -> _Scope:
        """Validate the requested scope and set the per-call context (spec 7.4, D06).

        ``default_scope`` applies when neither ``space`` nor ``spaces`` is given:
        ``"memberships"`` = every key of ``current_space_roles`` plus personal;
        ``"personal"`` = personal only. ``allow_archived`` defaults to True for
        reads and False for writes (``need_role`` writer/owner). The caller MUST
        call ``reset()`` on the result (``try/finally``).
        """
        if space is not None and spaces is not None:
            return _Scope(error=_error("invalid_arguments", "pass either `space` or `spaces`, not both"))
        if spaces is not None and (not isinstance(spaces, list) or not all(isinstance(s, str) for s in spaces)):
            return _Scope(error=_error("invalid_arguments", "`spaces` must be a list of space ids or slugs"))
        if include_personal is not None and not isinstance(include_personal, bool):
            return _Scope(error=_error("invalid_arguments", "`include_personal` must be a boolean"))
        need = SpaceRole(need_role) if need_role is not None else None
        is_write = need in (SpaceRole.WRITER, SpaceRole.OWNER)
        if allow_archived is None:
            allow_archived = not is_write
        service = _is_service()
        roles = {sid: _role_value(r) for sid, r in current_space_roles.get({}).items()}

        requested: list[str] | None = None
        if space is not None:
            requested = [space]
        elif spaces is not None:
            requested = list(spaces)

        labels: dict[str, Space] = {}
        scope_roles: dict[str, str] = {}
        scope_ids: list[str] = []
        if requested is not None:
            for ref in requested:
                if not isinstance(ref, str) or not ref:
                    return _Scope(error=_space_not_found())
                hit = await self._lookup_space(ref, roles)
                if hit is None:
                    await self._audit_denied_ref(ref, need)
                    return _Scope(error=_space_not_found())
                found, role = hit
                sid = found.id if found is not None else ref
                if found is not None and found.archived and not allow_archived:
                    await self._audit_denied(sid, "space_archived", need)
                    return _Scope(error=_error("space_archived", "This space is archived and read-only"))
                if need is not None and SPACE_ROLE_RANK[SpaceRole(role)] < SPACE_ROLE_RANK[need]:
                    await self._audit_denied(sid, "insufficient_role", need)
                    return _Scope(error=_error(
                        "insufficient_role", f"This action needs the {need.value} role in the space",
                    ))
                if sid not in scope_ids:
                    scope_ids.append(sid)
                    scope_roles[sid] = role
                    if found is not None:
                        labels[sid] = found
            personal = (not service) and include_personal is True
        else:
            if service and is_write:
                return _Scope(error=_error(
                    "space_required", "A service principal must name a `space` to write to",
                ))
            if default_scope == "memberships":
                scope_roles = dict(roles)
                if include_archived_memberships and self.spaces is not None:
                    uid = _caller_uid()
                    if uid is not None:
                        archived = await self.spaces.roles_for_principal(uid, include_archived=True)
                        for sid, r in archived.items():
                            scope_roles.setdefault(sid, _role_value(r))
                scope_ids = list(scope_roles)
                if self.spaces is not None:
                    for sid in scope_ids:
                        found = await self.spaces.get_space(sid)
                        if found is not None:
                            labels[sid] = found
            personal = (not service) and include_personal is not False

        tokens: list[tuple[Any, Any]] = [
            (current_scope_space_ids, current_scope_space_ids.set(list(scope_ids))),
            (current_include_personal, current_include_personal.set(personal)),
        ]
        return _Scope(
            space_ids=scope_ids, include_personal=personal, spaces=labels, roles=scope_roles, tokens=tokens,
        )

    async def _node_scope(self, include_archived: bool = True) -> _Scope:
        """Scope for node-addressed tools (no ``space`` param): full memberships.

        Archived memberships are included for the write tools so they answer
        ``space_archived`` rather than ``not_found``.
        """
        return await self._resolve_scope(
            None, None, None, None, "memberships", include_archived_memberships=include_archived,
        )

    @contextmanager
    def _narrow_scope(self, space_id: str | None) -> Iterator[None]:
        """Scope a write's storage calls to the new node's own scope (6.6).

        A space node sees only its space; a personal node sees personal + org,
        exactly as before spaces existed.
        """
        t1 = current_scope_space_ids.set([space_id] if space_id else [])
        t2 = current_include_personal.set(space_id is None)
        try:
            yield
        finally:
            current_include_personal.reset(t2)
            current_scope_space_ids.reset(t1)

    async def _authorize_write(self, node: MemoryNode, owner_message: str) -> dict[str, Any] | None:
        """Writer/owner for space nodes; ``_caller_owns_node`` for personal (D07)."""
        if node.space_id is None:
            if not _caller_owns_node(node):
                return _error("insufficient_role", owner_message)
            return None
        sid = node.space_id
        role = {k: _role_value(v) for k, v in current_space_roles.get({}).items()}.get(sid)
        space: Space | None = None
        if self.spaces is not None:
            space = await self.spaces.get_space(sid)
            if space is None:
                return _not_found(str(node.id))
            if role is None:
                uid = _caller_uid()
                if uid is not None:
                    all_roles = await self.spaces.roles_for_principal(uid, include_archived=True)
                    if sid in all_roles and space.archived:
                        role = _role_value(all_roles[sid])
        if role is None:
            return _not_found(str(node.id))
        if space is not None and space.archived:
            await self._audit_denied(sid, "space_archived", SpaceRole.WRITER)
            return _error("space_archived", "This space is archived and read-only")
        if SPACE_ROLE_RANK[SpaceRole(role)] < SPACE_ROLE_RANK[SpaceRole.WRITER]:
            await self._audit_denied(sid, "insufficient_role", SpaceRole.WRITER)
            return _error("insufficient_role", "This action needs the writer role in the space")
        return None

    @staticmethod
    def _space_label(space_id: str | None, labels: dict[str, Space] | None) -> dict[str, Any] | None:
        if space_id is None:
            return None
        space = (labels or {}).get(space_id)
        return {
            "id": space_id,
            "slug": space.slug if space is not None else None,
            "name": space.name if space is not None else None,
        }

    async def _labels_for(self, space_ids: list[str]) -> dict[str, Space]:
        out: dict[str, Space] = {}
        if self.spaces is None:
            return out
        for sid in dict.fromkeys(space_ids):
            found = await self.spaces.get_space(sid)
            if found is not None:
                out[sid] = found
        return out

    # -- writes (D08) -------------------------------------------------------------

    async def memory_store(
        self,
        content: str,
        source_session: str = "",
        related_to: list[str] | None = None,
        created_at: str | None = None,
        visibility: str = "private",
        org_id: str | None = None,
        related: list[dict[str, str]] | None = None,
        category: str | None = None,
        refs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        space: str | None = None,
        external_id: str | None = None,
        on_existing: str = "amend",
    ) -> dict[str, Any]:
        """Store a new memory. Returns the node ID.

        Parameter order note: this is a published API — new parameters are
        appended AFTER the original positional tail (``created_at``,
        ``visibility``, ``org_id``) so existing positional callers keep
        working. Never insert parameters mid-signature.

        Args:
            related_to: Legacy; creates caused_by edges — prefer ``related`` for
                typed edges. Each entry is the id of an existing node.
            related: Typed explicit edges, each ``{"id": <node-id>, "type":
                <edge-type>}``. Direction convention is source = the new node,
                i.e. ``new_node --type--> target`` reads "new node <type>
                target" (e.g. ``supersedes`` → the new node supersedes the
                target). Invalid types are rejected before the node is created.
            category: Free-form classification string. Suggested vocabulary is
                the auto-promote categories (professional, educational, family,
                location) since those interact with core promotion, but any
                string is accepted.
            created_at: Optional ISO 8601 timestamp. Defaults to now.
            visibility: "private" or "org". Defaults to "private".
            org_id: Required when visibility is "org".
            refs: Sources, each ``{type, id, url?, label?}`` (spec 5.2).
            tags: Labels; normalized to lowercase-with-dashes.
            space: Space id or slug to write into (writer or owner).
            external_id: The caller's own id for this memory (unique per space
                among current memories). ``on_existing`` decides what happens
                when a current memory already carries it.

        Validation (visibility, edge types, ids, refs, tags, same-space
        targets) runs completely before anything is written. The result may
        include ``possible_conflicts`` — heuristic hints (lexical
        numeric/negation divergence against auto-link candidates), not verified
        contradictions, and never materialized as edges.
        """
        scope = await self._resolve_scope(space, None, None, SpaceRole.WRITER, "memberships")
        if scope.error is not None:
            return scope.error
        try:
            space_id = scope.space_ids[0] if space is not None and scope.space_ids else None
            validated = await self._validate_store(_StoreRequest(
                content=content,
                source_session=source_session,
                related_to=related_to,
                created_at=created_at,
                visibility=visibility,
                org_id=org_id,
                related=related,
                category=category,
                refs=refs,
                tags=tags,
                space_id=space_id,
                external_id=external_id,
                on_existing=on_existing,
            ))
            if isinstance(validated, dict):
                return validated
            if validated.existing is not None:
                result = await self._apply_on_existing(validated)
            else:
                result = await self._write_memory(validated)
            if space_id is not None and "error" not in result:
                await self._audit_write_result(space_id, result)
                result["space"] = self._space_label(space_id, scope.spaces)
            return result
        finally:
            scope.reset()

    async def _validate_store(self, req: _StoreRequest) -> _Validated | dict[str, Any]:
        """Every check a store needs, with no side effects (D08).

        Targets of ``related``/``related_to`` are resolved under the caller's
        current scope. A space write refuses any target that is not a visible
        node of the same space (``cross_space_edge``); a personal write refuses
        a visible space target and keeps the legacy warn-and-skip for targets
        it cannot see.
        """
        from papez.engine import config

        vis = Visibility(req.visibility)
        space_id = req.space_id
        if space_id is not None and vis == Visibility.ORG:
            return _error("invalid_arguments", "visibility='org' can't be combined with `space`")
        if vis == Visibility.ORG and not req.org_id:
            return _error("invalid_arguments", "org_id required when visibility is 'org'")
        if vis == Visibility.ORG and req.org_id not in current_org_ids.get([]):
            return _error("invalid_arguments", "org_id not in caller's org memberships")
        if not isinstance(req.content, str) or not req.content.strip():
            return _error("invalid_arguments", "content must be a non-empty string")

        parsed_related: list[tuple[str, EdgeType]] = []
        if req.related:
            for item in req.related:
                if not isinstance(item, dict):
                    return _error("invalid_arguments", "each `related` entry must be an object {id, type}")
                raw_type = item.get("type")
                try:
                    etype = EdgeType(raw_type)
                except (ValueError, KeyError):
                    return {
                        "error": f"invalid edge type: {raw_type}",
                        "code": "invalid_arguments",
                        "valid_types": [e.value for e in EdgeType],
                    }
                target = item.get("id")
                if not isinstance(target, str) or not target:
                    return _error("invalid_arguments", "each `related` entry needs an `id`")
                parsed_related.append((target, etype))

        try:
            refs = validate_refs(req.refs, max_refs=config.MAX_REFS_PER_MEMORY)
            tags = normalize_tags(req.tags, max_tag_length=config.MAX_TAG_LENGTH, max_tags=config.MAX_TAGS_PER_MEMORY)
        except ValidationFailure as exc:
            return exc.as_error()

        if req.external_id is not None:
            if not isinstance(req.external_id, str) or not req.external_id or len(req.external_id) > _EXTERNAL_ID_MAX:
                return _error("invalid_arguments", f"external_id must be 1-{_EXTERNAL_ID_MAX} characters")
        if req.on_existing not in _ON_EXISTING:
            return _error("invalid_arguments", "on_existing must be one of 'skip', 'amend', 'error'")

        ts = _parse_iso_utc(req.created_at) if req.created_at else datetime.now(UTC)

        resolved_related_to: list[str] = []
        for target_id in req.related_to or []:
            ok = await self._check_target(target_id, space_id, "related_to")
            if isinstance(ok, dict):
                return ok
            if ok:
                resolved_related_to.append(target_id)
        resolved_related: list[tuple[str, EdgeType]] = []
        for target_id, etype in parsed_related:
            ok = await self._check_target(target_id, space_id, "related")
            if isinstance(ok, dict):
                return ok
            if ok:
                resolved_related.append((target_id, etype))

        existing: MemoryNode | None = None
        if req.external_id is not None:
            existing = await self.graph.find_node_by_external_id(req.external_id, space_id)

        return _Validated(
            content=req.content,
            source_session=req.source_session,
            visibility=Visibility.PRIVATE if space_id is not None else vis,
            org_id=None if space_id is not None else req.org_id,
            category=req.category,
            refs=refs,
            tags=tags,
            space_id=space_id,
            created_at=ts,
            related_to=resolved_related_to,
            related=resolved_related,
            explicit_requested=bool(req.related_to or parsed_related),
            external_id=req.external_id,
            on_existing=req.on_existing,
            existing=existing,
        )

    async def _check_target(self, target_id: str, space_id: str | None, kind: str) -> bool | dict[str, Any]:
        """True = link it, False = skip it (legacy personal rule), dict = refuse."""
        target = await self.graph.get_node(target_id) if isinstance(target_id, str) else None
        if target is None:
            if space_id is not None:
                return _error("cross_space_edge", f"{kind} id {target_id} is not in this space")
            logger.warning(
                "%s target %s not visible to caller %s; skipping edge", kind, target_id, _caller_uid(),
            )
            return False
        if target.space_id != space_id:
            where = "this space" if space_id is not None else "personal memory"
            return _error("cross_space_edge", f"{kind} id {target_id} is not in {where}")
        try:
            uuid.UUID(target_id)
        except ValueError:
            return _error("invalid_arguments", f"invalid node id: {target_id}")
        return True

    async def _apply_on_existing(self, v: _Validated) -> dict[str, Any]:
        """The spec 9.3 matrix for a store whose external_id already exists."""
        existing = v.existing
        assert existing is not None
        existing_id = str(existing.id)
        if v.on_existing == "skip":
            return {"node_id": existing_id, "status": "exists"}
        if v.on_existing == "error":
            return _error("external_id_exists", "A current memory with this external_id already exists",
                          node_id=existing_id)
        same_content = (existing.content_full or existing.content_summary).strip() == v.content.strip()
        same_refs = {_ref_identity(r) for r in existing.refs} == {_ref_identity(r) for r in v.refs}
        same_tags = set(existing.tags) == set(v.tags)
        if same_content and same_refs and same_tags:
            return {"node_id": existing_id, "status": "unchanged"}
        v.supersedes = (existing_id, "external_id update")
        v.source_session = v.source_session or existing.source_session
        v.category = v.category if v.category is not None else existing.category
        v.existing = None
        result = await self._write_memory(v)
        if "error" in result:
            return result
        await self._notify("memory.amended", {"node_id": result["node_id"], "supersedes": existing_id})
        return {"node_id": result["node_id"], "status": "amended", "superseded": existing_id,
                "visibility": result.get("visibility", v.visibility.value)}

    async def _write_memory(self, v: _Validated) -> dict[str, Any]:
        """Create the node and its edges from an already-validated request (D08)."""
        with self._narrow_scope(v.space_id):
            return await self._write_memory_scoped(v)

    async def _write_memory_scoped(self, v: _Validated) -> dict[str, Any]:
        content = v.content
        vis = v.visibility
        org_id = v.org_id
        embedding = await self.embeddings.embed(content) if self.embeddings else []
        summary = _truncate_summary(content)

        ts = v.created_at
        node = MemoryNode(
            status=MemoryStatus.ACTIVE,
            content_summary=summary,
            content_full=content,
            embedding=embedding,
            created_at=ts,
            last_accessed_at=ts,
            last_reactivated_at=ts,
            decay_score=1.0,
            causal_weight=0,
            source_agent=v.source_agent,
            source_session=v.source_session,
            category=v.category,
            visibility=vis,
            org_id=org_id,
            original_user_id=_caller_uid(),
            refs=list(v.refs),
            tags=list(v.tags),
            space_id=v.space_id,
            external_id=v.external_id,
            promoted_from=v.promoted_from,
        )

        node_id = await self.graph.create_node(node)

        # A correction edge first, so the F6 dedupe below never shadows it.
        if v.supersedes is not None:
            old_id, reason = v.supersedes
            await self.graph.create_edge(MemoryEdge(
                source_id=node.id,
                target_id=uuid.UUID(old_id),
                type=EdgeType.SUPERSEDES,
                weight=1.0,
                reason=reason,
                created_by="user_explicit",
            ))

        # Explicit edges from related_to (legacy caused_by; targets pre-checked)
        for target_id in v.related_to:
            await self.graph.create_edge(MemoryEdge(
                source_id=node.id,
                target_id=uuid.UUID(target_id),
                type=EdgeType.CAUSED_BY,
                weight=0.7,
                created_by="user_explicit",
            ))

        # Typed explicit edges from `related` (writer-specified semantics)
        for target_id, etype in v.related:
            await self.graph.create_edge(MemoryEdge(
                source_id=node.id,
                target_id=uuid.UUID(target_id),
                type=etype,
                weight=0.7,
                created_by="user_explicit",
            ))

        # Auto-link to semantically similar existing memories
        possible_conflicts: list[dict[str, Any]] = []
        if embedding:
            try:
                from papez.engine import config
                from papez.engine.contradiction import heuristic_conflict_signal

                org_ids = current_org_ids.get([])
                min_sim = config.resolve_autolink_min_similarity(self.embeddings)
                # The conflict-hint scan uses its OWN (lower) floor: a changed
                # figure between two versions of a fact often lands below the
                # strict "clearly the same topic" auto-link band, so gating the
                # scan on the auto-link floor would silently shrink conflict
                # detection every time that floor is raised.
                conflict_min_sim = config.resolve_conflict_min_similarity(self.embeddings)
                # F6 dedupe: fetch this node's existing neighbors once. ANY edge
                # (any type, either direction) suppresses a duplicate auto-link,
                # so a user_explicit caused_by/supersedes never gets shadowed by
                # a parallel auto_link related_to.
                existing = await self.graph.get_edges(node_id, "both")
                linked_ids = {
                    str(e.target_id) if str(e.source_id) == node_id else str(e.source_id)
                    for e in existing
                }
                # +1 because the query's own top hit is usually the new node.
                # The window is the max of the auto-link fan-out and the
                # (wider) conflict-scan window.
                scan_k = max(config.AUTOLINK_MAX_EDGES + 1, config.CONFLICT_SCAN_K)
                # Scope is narrowed to the node's own scope (_narrow_scope): a
                # space node only ever sees its space, a personal node personal
                # + org, so candidates and conflict hints stay in scope (6.6).
                similar = await self.graph.vector_search(
                    embedding, k=scan_k, org_ids=org_ids
                )
                links_created = 0
                with self._defer_saves():
                    for other_node, score in similar:
                        if str(other_node.id) == node_id:
                            continue
                        if other_node.space_id != v.space_id:
                            continue  # belt and braces: never across a space boundary
                        if score < min_sim and score < conflict_min_sim:
                            continue
                        # Org boundary rule: org nodes only link to same-org nodes
                        if vis == Visibility.ORG and (
                            other_node.visibility != Visibility.ORG or other_node.org_id != org_id
                        ):
                            continue
                        # Heuristic conflict hint — never creates structure.
                        if score >= conflict_min_sim:
                            signal = heuristic_conflict_signal(
                                content, other_node.content_full or other_node.content_summary
                            )
                            if signal:
                                possible_conflicts.append({
                                    "id": str(other_node.id),
                                    "summary": other_node.content_summary,
                                    "signal": signal,
                                })
                        if score < min_sim:
                            continue
                        # F6: skip if this pair is already linked by any edge.
                        if str(other_node.id) in linked_ids:
                            continue
                        if links_created >= config.AUTOLINK_MAX_EDGES:
                            # Fan-out cap reached; keep scanning for conflicts only.
                            continue
                        # F2 incoming-degree cap: fan-out alone doesn't stop a
                        # hub from *accreting* one link per store forever, so
                        # also bound the target's accumulated auto-link degree.
                        target_edges = await self.graph.get_edges(str(other_node.id), "both")
                        target_auto_degree = sum(
                            1 for e in target_edges if e.created_by == "auto_link"
                        )
                        if target_auto_degree >= config.AUTOLINK_MAX_NODE_DEGREE:
                            continue
                        edge = MemoryEdge(
                            source_id=node.id,
                            target_id=other_node.id,
                            type=EdgeType.RELATED_TO,
                            weight=round(score, 4),
                            reason=f"cosine similarity {score:.3f}",
                            created_by="auto_link",
                        )
                        await self.graph.create_edge(edge)
                        linked_ids.add(str(other_node.id))
                        links_created += 1
            except Exception:
                logger.warning("Auto-linking failed for node %s", node_id, exc_info=True)

        # Promote tagged → active if edges were formed (consolidation signal)
        has_edges = (
            v.explicit_requested or v.supersedes is not None or not await self.graph.is_orphan(node_id)
        )
        if has_edges:
            await self.graph.update_node(node_id, {"status": MemoryStatus.ACTIVE})

        if self.event_bus:
            # Identity fields let a Redis-path worker rebuild the context (D10).
            await self.event_bus.publish("memory.created", {
                "node_id": node_id,
                "content_full": content,
                "user_id": _caller_uid(),
                "org_ids": list(current_org_ids.get([])),
                "space_id": v.space_id,
            })

        await self._notify("memory.created", {"node_id": node_id, "content": content[:200]})
        result: dict[str, Any] = {"node_id": node_id, "status": "stored", "visibility": vis.value}
        if possible_conflicts:
            result["possible_conflicts"] = possible_conflicts
        return result

    async def memory_amend(
        self, node_id: str, content: str, reason: str | None = None,
        refs: list[dict[str, Any]] | None = None, tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a correction: create a new memory that supersedes an old one.

        The old memory is kept (its status is unchanged — recall already decays
        superseded hits by SUPERSEDED_DECAY and tags them ``superseded_by``, and
        the transitions engine demotes it naturally as it stops being recalled).
        Mutating status here would bypass that engine and be non-reversible.

        ``refs``/``tags``: None copies the old memory's values; a list replaces
        them exactly. The new node keeps the old node's space and external_id
        (so an external_id follows the amendment).
        """
        scope = await self._node_scope()
        try:
            old = await self.graph.get_node(node_id)
            if not old:
                return _not_found(node_id)
            denied = await self._authorize_write(old, "Only the node owner can amend this memory")
            if denied is not None:
                return denied
            result = await self._amend_node(old, content, reason, refs, tags)
            if "error" in result:
                return result
            new_id = result["node_id"]
            await self._notify("memory.amended", {"node_id": new_id, "supersedes": node_id})
            if old.space_id is not None:
                await self._audit_event("memory.amend", old.space_id, [new_id, node_id], {"status": "amended"})
            return {"node_id": new_id, "supersedes": node_id, "status": "amended"}
        finally:
            scope.reset()

    async def _amend_node(
        self,
        old: MemoryNode,
        content: str,
        reason: str | None,
        refs: list[Any] | None,
        tags: list[Any] | None,
    ) -> dict[str, Any]:
        """Shared amend routine (memory_amend, on_existing="amend"; D08).

        Authorization is the caller's job; this validates the new values and
        writes the superseding node.
        """
        from papez.engine import config

        if not isinstance(content, str) or not content.strip():
            return _error("invalid_arguments", "content must be a non-empty string")
        try:
            new_refs = list(old.refs) if refs is None else validate_refs(refs, max_refs=config.MAX_REFS_PER_MEMORY)
            new_tags = list(old.tags) if tags is None else normalize_tags(
                tags, max_tag_length=config.MAX_TAG_LENGTH, max_tags=config.MAX_TAGS_PER_MEMORY,
            )
        except ValidationFailure as exc:
            return exc.as_error()
        space_id = old.space_id
        return await self._write_memory(_Validated(
            content=content,
            source_session=old.source_session,
            visibility=Visibility.PRIVATE if space_id is not None else old.visibility,
            org_id=None if space_id is not None else old.org_id,
            category=old.category,
            refs=new_refs,
            tags=new_tags,
            space_id=space_id,
            created_at=datetime.now(UTC),
            external_id=old.external_id,
            supersedes=(str(old.id), reason or "amended"),
        ))

    # -- idempotent writes (spec 9.4, 9.5) ----------------------------------------

    async def memory_store_batch(
        self,
        items: list[dict[str, Any]],
        space: str | None = None,
        on_existing: str = "amend",
    ) -> dict[str, Any]:
        """Store up to ``config.BATCH_MAX_ITEMS`` memories into one scope (spec 9.4).

        Every item is validated first (content, refs, tags, category, edge
        types, ``related`` targets in the same scope, ``ref_index`` pointing at
        an earlier item, external_id conflicts under ``on_existing="error"``).
        If any item is invalid nothing is stored and every problem is returned
        as ``batch_invalid`` with ``items: [{index, code, error}]``. Otherwise
        the items are written in order inside ``_atomic()``; a ``related``
        entry ``{"ref_index": i, "type": t}`` links to the node item ``i``
        resolved to. Returns ``{"results": [{index, node_id, status,
        superseded?}]}``.

        Retry safety: when every item carries an ``external_id`` the whole
        batch is idempotent (a resend returns the same node ids with status
        ``unchanged``), so a client may retry it after a timeout.
        """
        from papez.engine import config

        if not isinstance(items, list):
            return _error("invalid_arguments", "`items` must be a list of objects")
        if len(items) > config.BATCH_MAX_ITEMS:
            return _error(
                "invalid_arguments",
                f"a batch holds at most {config.BATCH_MAX_ITEMS} items (got {len(items)})",
            )
        if on_existing not in _ON_EXISTING:
            return _error("invalid_arguments", "on_existing must be one of 'skip', 'amend', 'error'")

        scope = await self._resolve_scope(space, None, None, SpaceRole.WRITER, "memberships")
        if scope.error is not None:
            return scope.error
        try:
            space_id = scope.space_ids[0] if space is not None and scope.space_ids else None
            problems: list[dict[str, Any]] = []
            plans: list[tuple[_Validated, list[tuple[int, EdgeType]]]] = []
            seen_external: set[str] = set()
            for index, item in enumerate(items):
                checked = await self._validate_batch_item(index, item, space_id, on_existing, seen_external)
                if isinstance(checked, dict):
                    problems.append({"index": index, "code": checked.get("code", "invalid_arguments"),
                                     "error": checked.get("error", "invalid item")})
                else:
                    plans.append(checked)
            if problems:
                noun = "item is" if len(problems) == 1 else "items are"
                return _error("batch_invalid", f"{len(problems)} {noun} invalid", items=problems)

            results: list[dict[str, Any]] = []
            failure: dict[str, Any] | None = None
            async with self._atomic():
                for index, (v, index_links) in enumerate(plans):
                    for ref_index, etype in index_links:
                        target = results[ref_index]["node_id"]
                        if target not in {t for t, _ in v.related}:
                            v.related.append((target, etype))
                    if index_links:
                        v.explicit_requested = True
                    if v.external_id is not None:
                        # Re-read at write time: an earlier item of this batch
                        # may carry the same external_id (it then supersedes).
                        v.existing = await self.graph.find_node_by_external_id(v.external_id, space_id)
                    if v.existing is not None:
                        written = await self._apply_on_existing(v)
                    else:
                        written = await self._write_memory(v)
                    if "error" in written:
                        # Validation ran first, so this is not expected; stop
                        # rather than continue on a half-applied batch.
                        failure = _error(
                            written.get("code", "invalid_arguments"),
                            f"item {index} failed after validation: {written.get('error')}",
                            index=index,
                            results=results,
                        )
                        break
                    row: dict[str, Any] = {
                        "index": index, "node_id": written["node_id"], "status": written["status"],
                    }
                    if "superseded" in written:
                        row["superseded"] = written["superseded"]
                    results.append(row)
            # Audit after the atomic block: one event per written item.
            for row in results:
                await self._audit_write_result(space_id, row, {"batch_index": row["index"]})
            if failure is not None:
                return failure
            return {"results": results}
        finally:
            scope.reset()

    _BATCH_ITEM_KEYS = frozenset({
        "content", "refs", "tags", "category", "external_id", "related", "source_session", "created_at",
    })

    async def _validate_batch_item(
        self,
        index: int,
        item: Any,
        space_id: str | None,
        on_existing: str,
        seen_external: set[str],
    ) -> tuple[_Validated, list[tuple[int, EdgeType]]] | dict[str, Any]:
        """Validate one batch item; returns the plan or an error dict."""
        if not isinstance(item, dict):
            return _error("invalid_arguments", "each item must be an object")
        unknown = sorted(set(item) - self._BATCH_ITEM_KEYS)
        if unknown:
            return _error("invalid_arguments", f"unknown item field(s): {', '.join(unknown)}")
        related_raw = item.get("related")
        if related_raw is not None and not isinstance(related_raw, list):
            return _error("invalid_arguments", "`related` must be a list of objects")
        by_id: list[dict[str, Any]] = []
        index_links: list[tuple[int, EdgeType]] = []
        for entry in related_raw or []:
            if isinstance(entry, dict) and "ref_index" in entry:
                if "id" in entry:
                    return _error("invalid_arguments", "a `related` entry takes `id` or `ref_index`, not both")
                ref_index = entry.get("ref_index")
                if not isinstance(ref_index, int) or isinstance(ref_index, bool) or not 0 <= ref_index < index:
                    return _error(
                        "invalid_arguments", f"ref_index {ref_index!r} must point at an earlier item of this batch",
                    )
                raw_type = entry.get("type")
                try:
                    index_links.append((ref_index, EdgeType(raw_type)))
                except (ValueError, KeyError):
                    return _error("invalid_arguments", f"invalid edge type: {raw_type}")
            else:
                by_id.append(entry)
        category = item.get("category")
        if category is not None and not isinstance(category, str):
            return _error("invalid_arguments", "category must be a string")
        source_session = item.get("source_session", "")
        if not isinstance(source_session, str):
            return _error("invalid_arguments", "source_session must be a string")
        created_at = item.get("created_at")
        if created_at is not None:
            if not isinstance(created_at, str):
                return _error("invalid_arguments", "created_at must be an ISO 8601 string")
            try:
                _parse_iso_utc(created_at)
            except ValueError:
                return _error("invalid_arguments", f"invalid created_at: {created_at}")
        content = item.get("content")
        validated = await self._validate_store(_StoreRequest(
            content=content if isinstance(content, str) else "",  # "" -> invalid_arguments
            source_session=source_session,
            created_at=created_at,
            related=by_id or None,
            category=category,
            refs=item.get("refs"),
            tags=item.get("tags"),
            space_id=space_id,
            external_id=item.get("external_id"),
            on_existing=on_existing,
        ))
        if isinstance(validated, dict):
            return validated
        ext = validated.external_id
        if ext is not None:
            if on_existing == "error" and (validated.existing is not None or ext in seen_external):
                return _error("external_id_exists", f"a current memory with external_id {ext!r} already exists")
            seen_external.add(ext)
        return validated, index_links

    async def memory_retire_by_ref(
        self,
        ref_type: str,
        ref_id: str,
        space: str | None = None,
        mode: str = "amend",
    ) -> dict[str, Any]:
        """Retire the memories that cite a removed source (spec 9.5).

        Scope: the named space (writer or owner), or, for a user with no space,
        the caller's own personal memories. A service must name a space.
        Only current (non-superseded) memories are touched.

        ``mode="amend"``: a memory whose only ref is this one is amended to
        ``"[Retired] <summary> - source <label or id> was removed."`` and
        tagged ``retired`` (the ref is kept for provenance; an already retired
        memory is left alone, so a repeat call is a no-op). A memory with other
        refs is amended with this ref removed. ``mode="delete"``: sole-ref
        memories are deleted, the others amended with the ref removed.
        Returns ``{"retired", "updated", "deleted"}``.
        """
        from papez.engine import config

        if mode not in ("amend", "delete"):
            return _error("invalid_arguments", "mode must be 'amend' or 'delete'")
        if not isinstance(ref_type, str) or not ref_type or not isinstance(ref_id, str) or not ref_id:
            return _error("invalid_arguments", "ref_type and ref_id must be non-empty strings")
        scope = await self._resolve_scope(space, None, None, SpaceRole.WRITER, "personal")
        if scope.error is not None:
            return scope.error
        try:
            space_id = scope.space_ids[0] if space is not None and scope.space_ids else None
            counts = {"retired": 0, "updated": 0, "deleted": 0}
            audit_rows: list[tuple[str, list[str]]] = []
            nodes = await self.graph.find_nodes_by_ref(ref_type, ref_id)
            async with self._atomic():
                for node in nodes:
                    if node.space_id != space_id:
                        continue
                    if space_id is None and not _caller_owns_node(node):
                        continue  # org-visible memory of another user
                    nid = str(node.id)
                    incoming = await self.graph.get_edges(nid, "incoming", EdgeType.SUPERSEDES)
                    if any(str(e.target_id) == nid for e in incoming):
                        continue  # superseded: history, not a current memory
                    denied = await self._authorize_write(node, "Only the node owner can retire this memory")
                    if denied is not None:
                        continue
                    matching = [r for r in node.refs if r.type == ref_type and r.id == ref_id]
                    if not matching:
                        continue
                    remaining = [r for r in node.refs if not (r.type == ref_type and r.id == ref_id)]
                    if remaining:
                        result = await self._amend_node(
                            node, node.content_full or node.content_summary,
                            f"ref {ref_type}:{ref_id} removed", [_ref_dump(r) for r in remaining], None,
                        )
                        if "error" in result:
                            return result
                        await self._notify("memory.amended", {"node_id": result["node_id"], "supersedes": nid})
                        audit_rows.append(("memory.amend", [result["node_id"], nid]))
                        counts["updated"] += 1
                    elif mode == "delete":
                        await self.graph.delete_node(nid)
                        await self._notify("memory.deleted", {"node_id": nid})
                        audit_rows.append(("memory.delete", [nid]))
                        counts["deleted"] += 1
                    else:
                        if "retired" in node.tags:
                            continue
                        source = matching[0].label or matching[0].id
                        content = f"[Retired] {node.content_summary} - source {source} was removed."
                        # Keep within the tag limit: ``retired`` always survives.
                        keep = [t for t in node.tags if t != "retired"][: max(config.MAX_TAGS_PER_MEMORY - 1, 0)]
                        result = await self._amend_node(
                            node, content, f"source {ref_type}:{ref_id} removed", None, [*keep, "retired"],
                        )
                        if "error" in result:
                            return result
                        await self._notify("memory.amended", {"node_id": result["node_id"], "supersedes": nid})
                        audit_rows.append(("memory.amend", [result["node_id"], nid]))
                        counts["retired"] += 1
            if space_id is not None:
                for action, ids in audit_rows:
                    await self._audit_event(action, space_id, ids, {"via": "retire_by_ref", "mode": mode})
            return counts
        finally:
            scope.reset()

    async def memory_recall(
        self,
        query: str,
        k: int = 10,
        max_results: int | None = None,
        read_only: bool = False,
        verbosity: str = "full",
        min_results: int | None = None,
        tags_any: list[str] | None = None,
        ref_type: str | None = None,
        space: str | None = None,
        spaces: list[str] | None = None,
        include_personal: bool | None = None,
    ) -> dict[str, Any]:
        """Recall memories by hybrid search; see ``_recall`` for the ranking.

        Scope (spec 6.5/7.4): ``space``/``spaces`` limit the call to named
        spaces (personal memory then only with ``include_personal=True``);
        with neither, every non-archived membership plus personal memory.
        ``tags_any``/``ref_type`` filter candidates BEFORE ranking (storage
        level), so ``k`` matching results still come back.
        """
        from papez.engine import config

        try:
            norm_tags = normalize_tags(
                tags_any, max_tag_length=config.MAX_TAG_LENGTH, max_tags=config.MAX_TAGS_PER_MEMORY,
            ) if tags_any is not None else []
        except ValidationFailure as exc:
            return exc.as_error()
        if ref_type is not None and not isinstance(ref_type, str):
            return _error("invalid_arguments", "ref_type must be a string")
        scope = await self._resolve_scope(space, spaces, include_personal, None, "memberships")
        if scope.error is not None:
            return scope.error
        try:
            out = await self._recall(
                query, k, max_results, read_only, verbosity, min_results,
                tags_any=norm_tags or None, ref_type=ref_type or None, labels=scope.spaces,
            )
            if "error" not in out:
                await self._audit_reads(
                    "memory.recall", scope.space_ids, self._hits_of(out.get("results")),
                    len(query) if isinstance(query, str) else 0,
                )
            return out
        finally:
            scope.reset()

    async def _recall(
        self,
        query: str,
        k: int = 10,
        max_results: int | None = None,
        read_only: bool = False,
        verbosity: str = "full",
        min_results: int | None = None,
        *,
        tags_any: list[str] | None = None,
        ref_type: str | None = None,
        labels: dict[str, Space] | None = None,
    ) -> dict[str, Any]:
        """Recall memories by hybrid search: vector + keyword, ranked by vector similarity.

        verbosity: "full" (default; unchanged payload) or "concise". Concise
        hits carry only id/summary/status/score/activation/is_core (plus
        superseded_by when set); the expensive causal-chain queries are skipped
        entirely. Reactivation writes still happen in both modes (governed by
        read_only, not verbosity).

        min_results: zero-results floor. When the min-similarity gate would
        leave fewer than ``min(k, min_results)`` hits, recall backfills with the
        next-best below-threshold vector hits (each flagged ``low_confidence``)
        so the answerer always has material. Defaults to
        ``config.RECALL_RESULT_FLOOR`` (env ``GENESYS_RECALL_RESULT_FLOOR``);
        pass 0 to disable the backfill and keep strict threshold-only behavior.
        Threshold semantics for normal, well-populated results are unchanged.
        """
        import asyncio

        if verbosity not in ("full", "concise"):
            logger.warning("memory_recall verbosity=%r unrecognized; using 'full'", verbosity)
            verbosity = "full"

        MAX_K = 100
        if k > MAX_K:
            logger.info("memory_recall k=%d capped to %d", k, MAX_K)
            k = MAX_K

        # Extract keyword terms while embedding runs
        _base_stopwords = {
            "what", "where", "who", "how", "why", "which", "does", "did",
            "have", "are", "is", "the", "a", "an",
            "in", "on", "at", "to", "for", "of", "and", "or", "do",
            "from", "that", "this", "with", "about", "some", "any", "many",
            "much", "her", "his", "its", "their", "she", "he", "it", "they",
        }
        _temporal_stopwords = {"when", "has", "had", "was", "were", "been"}
        _temporal_signals = {
            "before", "after", "first", "last", "recently", "ago",
            "year", "month", "date", "time", "during", "until", "since",
        }
        query_lower = query.lower()
        query_words = query_lower.split()
        has_temporal = any(w.strip("?.,!'\"") in _temporal_signals for w in query_words)
        _stopwords = _base_stopwords if has_temporal else _base_stopwords | _temporal_stopwords
        terms = [w for w in query_words if w.strip("?.,!'\"") not in _stopwords and len(w) > 2]

        org_ids = current_org_ids.get([])
        from papez.engine import config as engine_config
        _rrf_on = engine_config.hybrid_rrf_enabled()
        # Run embedding + all keyword searches concurrently. Keyword search
        # coroutines are built once and MUST be awaited on every branch below
        # (no-embedder recall still relies on the keyword path). Under I3-2 the
        # per-term ILIKE query is plural-stemmed ("cats"->"cat") so a singular
        # gold turn is still matched; de-duped so a stem collision is one search.
        _raw_terms = [t.strip("?.,!'\"") for t in terms[:5]]
        if _rrf_on:
            search_terms = list(dict.fromkeys(_stem_plural(t) for t in _raw_terms if t))
        else:
            search_terms = _raw_terms
        # Storage-level filters (D09): only passed when set, so providers that
        # predate them keep working for unfiltered recall.
        search_filters: dict[str, Any] = {}
        if tags_any:
            search_filters["tags_any"] = tags_any
        if ref_type:
            search_filters["ref_type"] = ref_type
        kw_coros = [self.graph.keyword_search(t, k=k, org_ids=org_ids, **search_filters) for t in search_terms]
        if self.embeddings:
            embed_and_kw: list[Any] = await asyncio.gather(self.embeddings.embed(query), *kw_coros)
            embedding: list[float] = embed_and_kw[0]
            kw_results_per_term: list[list[MemoryNode]] = embed_and_kw[1:]
        else:
            embedding = []
            kw_results_per_term = await asyncio.gather(*kw_coros) if kw_coros else []

        # 1. Vector search (needs embedding)
        min_sim = engine_config.resolve_recall_min_similarity(self.embeddings)

        vector_results = (
            await self.graph.vector_search(embedding, k=k, org_ids=org_ids, **search_filters) if embedding else []
        )

        # 2. Collect keyword hits. kw_coverage[nid] = how many distinct query
        # terms matched this node — the lexical-relevance signal I3-2's RRF ranks
        # the keyword list by (a node matching three query terms is a stronger
        # lexical hit than one matching one, independent of its cosine).
        kw_node_ids: set[str] = set()
        kw_nodes_map: dict[str, MemoryNode] = {}
        kw_coverage: dict[str, int] = {}
        for kw_nodes in kw_results_per_term:
            seen_this_term: set[str] = set()
            for node in kw_nodes:
                nid = str(node.id)
                kw_node_ids.add(nid)
                kw_nodes_map[nid] = node
                if nid not in seen_this_term:
                    seen_this_term.add(nid)
                    kw_coverage[nid] = kw_coverage.get(nid, 0) + 1

        # 3. Merge by union, track vector scores and keyword membership
        from papez.engine.scoring import cosine_similarity
        merged: dict[str, dict[str, Any]] = {}

        vector_ids: set[str] = set()
        # Below-threshold vector hits retained as fallback material for the
        # zero-results floor. vector_search returns descending similarity, so
        # this stays "next-best first" and needs no re-sort.
        below_threshold: list[tuple[MemoryNode, float]] = []
        for node, score in vector_results:
            nid = str(node.id)
            is_kw_hit = nid in kw_node_ids
            # A lexical keyword match is its own independent relevance
            # signal (stemmed term overlap with content), so it isn't
            # gated by the vector-similarity floor — only pure vector
            # matches are, to filter embedding noise.
            if score < min_sim and not is_kw_hit:
                below_threshold.append((node, score))
                continue
            vector_ids.add(nid)
            merged[nid] = {"node": node, "vec_score": score, "in_both": is_kw_hit}

        # Add keyword-only results — compute their vector similarity for
        # ranking only (not gated by min_sim; see comment above).
        for nid, node in kw_nodes_map.items():
            if nid not in merged:
                if node.embedding and embedding:
                    vec_score = cosine_similarity(node.embedding, embedding)
                else:
                    vec_score = 0.0
                merged[nid] = {"node": node, "vec_score": vec_score, "in_both": False}

        # Zero-results floor: a query that embeds far from everything (inference
        # or paraphrase questions with little lexical overlap) can leave the
        # min-similarity gate with almost nothing — starving the answerer even
        # though k asked for many. When the surviving union is below the floor,
        # backfill with the next-best below-threshold vector hits, each flagged
        # low_confidence, up to min(k, floor). This is additive and only fires
        # on already-starved results; well-populated results are untouched, so
        # threshold semantics for normal cases are unchanged.
        result_floor = min_results if min_results is not None else engine_config.RECALL_RESULT_FLOOR
        floor_target = min(k, result_floor)
        if len(merged) < floor_target and below_threshold:
            for node, score in below_threshold:
                nid = str(node.id)
                if nid in merged:
                    continue
                merged[nid] = {
                    "node": node,
                    "vec_score": score,
                    "in_both": False,
                    "low_confidence": True,
                }
                if len(merged) >= floor_target:
                    break

        # I3-2 Reciprocal Rank Fusion. Build two rankings over the merged
        # candidates and fuse them, instead of ranking by cosine + a flat +0.1
        # keyword nudge. List V = vector order (by cosine). List K = keyword
        # order (by term COVERAGE desc, cosine as tiebreak). RRF score =
        # 1/(C+rankV) + 1/(C+rankK); normalized to [0,1] so it composes unchanged
        # with the downstream core-injection / spreading / superseded stages
        # (all of which read and write _rank_score on the same scale). A gold
        # turn buried deep in the vector order but high in the coverage order is
        # lifted above a single-term distractor that only ranks well on cosine.
        rrf_c = engine_config.HYBRID_RRF_K
        vec_rank: dict[str, int] = {}
        kw_rank: dict[str, int] = {}
        if _rrf_on:
            for i, (node, _score) in enumerate(vector_results):
                nid = str(node.id)
                if nid not in vec_rank:
                    vec_rank[nid] = i + 1
            kw_candidates = [nid for nid in merged if kw_coverage.get(nid, 0) > 0]
            kw_candidates.sort(
                key=lambda n: (kw_coverage.get(n, 0), merged[n]["vec_score"]),
                reverse=True,
            )
            for i, nid in enumerate(kw_candidates):
                kw_rank[nid] = i + 1
            _rrf_norm = 2.0 / (rrf_c + 1)  # max possible raw score (rank 1 in both)

        # 4. Format results without causal chains first (defer expensive graph queries)
        memories = []
        node_by_id: dict[str, MemoryNode] = {}
        for nid, info in merged.items():
            node = info["node"]
            node_by_id[nid] = node
            mem = self._format_memory_light(node, info["vec_score"], labels)
            if info.get("low_confidence"):
                mem["low_confidence"] = True
            if _rrf_on:
                vr = vec_rank.get(nid)
                kr = kw_rank.get(nid)
                raw = (1.0 / (rrf_c + vr) if vr else 0.0) + (1.0 / (rrf_c + kr) if kr else 0.0)
                rank_score = raw / _rrf_norm
            else:
                rank_score = info["vec_score"] + (
                    engine_config.RRF_KEYWORD_BONUS if info["in_both"] else 0.0
                )
            mem["_rank_score"] = rank_score
            memories.append(mem)

            if self.event_bus and nid in vector_ids:
                await self.event_bus.publish("memory.accessed", {"node_id": nid})

        # Inject core memories not already in results. Auto-promoted core
        # memories are only injected when relevant to the query (avoids
        # noise); explicitly *pinned* memories are always injected — that's
        # the contract of pin_memory (always available regardless of query).
        core_min_sim = engine_config.resolve_core_inject_min_similarity(self.embeddings)
        core_nodes = await self.graph.get_nodes_by_status(MemoryStatus.CORE, limit=50)
        seen_ids = set(merged.keys())
        core_candidates: list[tuple[MemoryNode, float]] = []
        for cnode in core_nodes:
            cid = str(cnode.id)
            if not _matches_tags_and_ref_type(cnode, tags_any, ref_type):
                continue  # the pinned/core injection honours the same filters
            if cid not in seen_ids:
                core_sim = 0.0
                if cnode.embedding and embedding:
                    core_sim = cosine_similarity(cnode.embedding, embedding)
                if cnode.pinned or core_sim >= core_min_sim:
                    core_candidates.append((cnode, core_sim))
        core_candidates.sort(key=lambda x: (x[0].pinned, x[1]), reverse=True)
        for cnode, core_sim in core_candidates[:10]:
            cid = str(cnode.id)
            node_by_id[cid] = cnode
            mem = self._format_memory_light(cnode, core_sim, labels)
            mem["is_core"] = True
            mem["_rank_score"] = core_sim
            memories.append(mem)

        # Batch superseded check + spreading activation: one get_all_edges call
        SUPERSEDED_DECAY = 0.3
        SPREAD_BONUS = 0.05
        all_mem_ids = [m["id"] for m in memories if m.get("id")]
        if all_mem_ids:
            try:
                all_mem_edges = await self.graph.get_all_edges(all_mem_ids)
                superseded_map: dict[str, str] = {}
                mem_id_set = set(all_mem_ids)
                neighbor_counts: dict[str, int] = {}
                for edge in all_mem_edges:
                    tgt = str(edge.target_id)
                    src = str(edge.source_id)
                    # A SUPERSEDES edge is directed new(source) -> old(target):
                    # only the TARGET is superseded. (A previous elif here
                    # marked the superseder itself as superseded whenever the
                    # old node fell out of the result set — down-ranking the
                    # correction it was supposed to prefer.)
                    if edge.type == EdgeType.SUPERSEDES and tgt in node_by_id:
                        superseded_map[tgt] = src
                    # Count edges between result set members for spreading activation
                    if src in mem_id_set and tgt in mem_id_set:
                        neighbor_counts[src] = neighbor_counts.get(src, 0) + 1
                        neighbor_counts[tgt] = neighbor_counts.get(tgt, 0) + 1
                for mem in memories:
                    mem_id = mem.get("id")
                    if not mem_id:
                        continue
                    if mem_id in superseded_map:
                        mem["_rank_score"] *= SUPERSEDED_DECAY
                        mem["superseded_by"] = superseded_map[mem_id]
                    # Spreading activation: boost memories connected to other results
                    spread = neighbor_counts.get(mem_id, 0)
                    if spread > 0:
                        mem["_rank_score"] += SPREAD_BONUS * spread
            except Exception:
                logger.warning("Superseded/spreading check failed", exc_info=True)

        # I3-4 date-anchored reranking. For a question carrying an ABSOLUTE date
        # anchor (a month+year, an explicit date, or a window like "the last two
        # weeks of August 2023"), boost every retrieved turn whose session date
        # or resolved [event: ... -> YYYY-MM-DD] date lands inside that window.
        # BOOST-ONLY and additive: a relative-only or non-date question parses to
        # no anchor and this is a strict no-op, so the [event:]-resolution path
        # is never perturbed. Runs before the sort so a buried date-match can rise
        # into the answer window; a date-match already on top only rises further.
        if engine_config.date_rerank_enabled():
            try:
                from papez.retrieval.date_anchor import (
                    node_matches_anchor,
                    parse_query_date_anchor,
                )
                _anchor = parse_query_date_anchor(query)
                if _anchor is not None:
                    _boost = engine_config.DATE_RERANK_BOOST
                    for mem in memories:
                        _mem_id = mem.get("id")
                        if _mem_id is None:
                            continue
                        _node = node_by_id.get(_mem_id)
                        if _node is None:
                            continue
                        if node_matches_anchor(
                            _node.content_full or _node.content_summary,
                            _node.created_at,
                            _anchor,
                        ):
                            mem["_rank_score"] += _boost
                            mem["date_anchor_boosted"] = True
            except Exception:
                logger.warning("Date-anchored rerank failed", exc_info=True)

        memories.sort(key=lambda m: m["_rank_score"], reverse=True)
        for mem in memories:
            mem.pop("_rank_score", None)

        # Cap final results
        cap = max_results if max_results is not None else k
        memories = memories[:cap]

        # Enrich top results with causal chains (batch fetch). Skipped entirely
        # in concise mode — the chain queries are the main latency cost.
        org_ids_for_chain = current_org_ids.get([])
        mem_ids_for_chain = [m["id"] for m in memories if m.get("id")]
        if verbosity != "concise" and mem_ids_for_chain:
            try:
                has_batch = hasattr(self.graph, "get_causal_chains_batch")
                if has_batch:
                    upstream_map, downstream_map = await asyncio.gather(
                        self.graph.get_causal_chains_batch(mem_ids_for_chain, "upstream", org_ids=org_ids_for_chain),
                        self.graph.get_causal_chains_batch(mem_ids_for_chain, "downstream", org_ids=org_ids_for_chain),
                    )
                else:
                    chain_coros = []
                    for mid in mem_ids_for_chain:
                        chain_coros.append(self.graph.get_causal_chain(mid, "upstream", org_ids=org_ids_for_chain))
                        chain_coros.append(self.graph.get_causal_chain(mid, "downstream", org_ids=org_ids_for_chain))
                    chain_results = await asyncio.gather(*chain_coros, return_exceptions=True)
                    upstream_map = {}
                    downstream_map = {}
                    for i, mid in enumerate(mem_ids_for_chain):
                        up = chain_results[i * 2]
                        down = chain_results[i * 2 + 1]
                        upstream_map[mid] = [] if isinstance(up, BaseException) else up
                        downstream_map[mid] = [] if isinstance(down, BaseException) else down

                for mid in mem_ids_for_chain:
                    upstream = upstream_map.get(mid, [])
                    downstream = downstream_map.get(mid, [])
                    mem = next(m for m in memories if m.get("id") == mid)
                    causal_basis = []
                    causal_chain = []
                    seen_causal: set[str] = set()
                    for n in upstream[:10]:
                        nid_str = str(n.id)
                        if nid_str not in seen_causal:
                            causal_basis.append({"id": nid_str, "summary": n.content_summary, "direction": "upstream"})
                            seen_causal.add(nid_str)
                    for n in downstream[:10]:
                        nid_str = str(n.id)
                        if nid_str not in seen_causal:
                            causal_basis.append({"id": nid_str, "summary": n.content_summary, "direction": "downstream"})
                            seen_causal.add(nid_str)
                    if upstream:
                        for n in reversed(upstream[:10]):
                            causal_chain.append({"id": str(n.id), "summary": n.content_summary})
                        origin = node_by_id.get(mid)
                        if origin:
                            causal_chain.append({"id": mid, "summary": origin.content_summary})
                    mem["causal_basis"] = causal_basis
                    if causal_chain:
                        mem["causal_chain"] = causal_chain
            except Exception:
                logger.warning("Causal chain batch fetch failed", exc_info=True)

        # Update reactivation state + validate co-retrieval edges (skip in read_only mode)
        if not read_only:
            with self._defer_saves():
                now = datetime.now(UTC)
                reactivation_coros = []
                reactivation_mems = []
                for mem in memories:
                    mem_id = mem.get("id")
                    if not mem_id:
                        continue
                    recalled = node_by_id.get(mem_id)
                    if recalled:
                        stability_delta = 0.1 / recalled.stability
                        reactivation_coros.append(
                            self.graph.atomic_reactivation_update(mem_id, now, stability_delta)
                        )
                        reactivation_mems.append((mem, recalled.reactivation_count + 1))
                if reactivation_coros:
                    results = await asyncio.gather(*reactivation_coros, return_exceptions=True)
                    for j, (mem, new_count) in enumerate(reactivation_mems):
                        if not isinstance(results[j], BaseException):
                            mem["reactivation_count"] = new_count

                if len(memories) > 1:
                    recalled_ids = {m["id"] for m in memories if m.get("id")}
                    try:
                        recall_edges = await self.graph.get_all_edges(list(recalled_ids))
                        validate_coros = [
                            self.graph.validate_edge(str(edge.id))
                            for edge in recall_edges
                            if str(edge.source_id) in recalled_ids and str(edge.target_id) in recalled_ids
                        ]
                        if validate_coros:
                            await asyncio.gather(*validate_coros, return_exceptions=True)
                    except Exception:
                        logger.warning("Co-retrieval edge validation failed", exc_info=True)

        if verbosity == "concise":
            concise: list[dict[str, Any]] = []
            for mem in memories:
                c: dict[str, Any] = {
                    "id": mem.get("id"),
                    "summary": mem.get("summary"),
                    "status": mem.get("status"),
                    "score": mem.get("score"),
                    "activation": mem.get("activation"),
                    "is_core": mem.get("is_core", False),
                    "tags": mem.get("tags", []),
                    "ref_count": len(mem.get("refs", [])),
                    "space": mem.get("space"),
                }
                if "superseded_by" in mem:
                    c["superseded_by"] = mem["superseded_by"]
                if mem.get("low_confidence"):
                    c["low_confidence"] = True
                concise.append(c)
            return {"query": query, "results": concise, "count": len(concise)}

        return {"query": query, "results": memories, "count": len(memories)}

    def _format_memory_light(
        self, node: MemoryNode, score: float, labels: dict[str, Space] | None = None,
    ) -> dict[str, Any]:
        """Format a memory node without causal chain queries (for ranking phase)."""
        return {
            "id": str(node.id),
            "content": node.content_full or node.content_summary,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            # `activation` is an alias for `decay_score`: same value, clearer
            # name. The score is an activation/retention weight (retrieval
            # RAISES it), not a countdown to deletion.
            "activation": round(node.decay_score, 4),
            "score": round(score, 4),
            "created_at": node.created_at.isoformat(),
            "causal_basis": [],
            "is_core": node.status == MemoryStatus.CORE,
            "refs": [_ref_dump(r) for r in node.refs],
            "tags": list(node.tags),
            "space": self._space_label(node.space_id, labels),
        }

    @staticmethod
    def _node_passes_filters(node: MemoryNode, filters: dict[str, Any] | None) -> bool:
        """Apply memory_search's post-query filters to a node.

        ISO timestamps are normalized to UTC when tz-naive (matching
        memory_store's created_at handling) — node timestamps are always
        tz-aware, so comparing against a naive datetime would raise TypeError.
        """
        if not filters:
            return True
        if "category" in filters and node.category != filters["category"]:
            return False
        if "entity" in filters and filters["entity"] not in node.entity_refs:
            return False
        if "since" in filters:
            since_dt = _parse_iso_utc(filters["since"])
            if node.created_at < since_dt:
                return False
        if "active_since" in filters:
            active_dt = _parse_iso_utc(filters["active_since"])
            if node.last_reactivated_at < active_dt:
                return False
        # Tag filters expect values already normalized (memory_search does it).
        tags_any = filters.get("tags_any")
        if tags_any and not set(tags_any).intersection(node.tags):
            return False
        tags_all = filters.get("tags_all")
        if tags_all and not set(tags_all).issubset(node.tags):
            return False
        ref_type = filters.get("ref_type")
        ref_id = filters.get("ref_id")
        if ref_type or ref_id:
            # Combined, both must match on the SAME ref (spec 5.4).
            if not any(
                (not ref_type or r.type == ref_type) and (not ref_id or r.id == ref_id)
                for r in node.refs
            ):
                return False
        return True

    @staticmethod
    def _format_search_hit(
        node: MemoryNode, score: float, labels: dict[str, Space] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": str(node.id),
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            "score": round(score, 4),
            "category": node.category,
            "created_at": node.created_at.isoformat(),
            # Provenance / recency fields (additive) so "what changed" answers
            # carry where and when a memory last moved.
            "last_reactivated_at": node.last_reactivated_at.isoformat(),
            "source_session": node.source_session,
            "refs": [_ref_dump(r) for r in node.refs],
            "tags": list(node.tags),
            "space": MCPToolHandler._space_label(node.space_id, labels),
        }

    async def memory_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int = 10,
        space: str | None = None,
        spaces: list[str] | None = None,
        include_personal: bool | None = None,
    ) -> dict[str, Any]:
        """Filtered search; scope as in ``memory_recall`` (spec 6.7)."""
        from papez.engine import config

        if filters is not None and not isinstance(filters, dict):
            return _error("invalid_arguments", "filters must be an object")
        if filters:
            filters = dict(filters)
            if isinstance(filters.get("status"), str):
                filters["status"] = [filters["status"]]
            try:
                for key in ("tags_any", "tags_all"):
                    if filters.get(key) is not None:
                        filters[key] = normalize_tags(
                            filters[key], max_tag_length=config.MAX_TAG_LENGTH,
                            max_tags=config.MAX_TAGS_PER_MEMORY,
                        )
            except ValidationFailure as exc:
                return exc.as_error()
        scope = await self._resolve_scope(space, spaces, include_personal, None, "memberships")
        if scope.error is not None:
            return scope.error
        try:
            out = await self._search(query, filters, k, scope.spaces)
            if "error" not in out:
                await self._audit_reads(
                    "memory.search", scope.space_ids, self._hits_of(out.get("results")),
                    len(query) if isinstance(query, str) else 0,
                )
            return out
        finally:
            scope.reset()

    async def _search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        k: int,
        labels: dict[str, Space] | None,
    ) -> dict[str, Any]:
        """Filtered vector search by status/category/date/entity/tags/refs.

        Enumeration mode (F8): an EMPTY ``query`` skips vector search entirely
        and lists memories by recency (``last_reactivated_at`` descending),
        honoring the same filters. Combined with ``since``/``active_since``
        this answers "what's new/changed since <ts>" without knowing what to
        query for, and works with no embedder configured.
        """
        # Determine status filter (shared by both modes)
        status_filter = None
        if filters and "status" in filters:
            status_filter = [MemoryStatus(s) for s in filters["status"]]

        if not query.strip():
            # Non-vector enumeration path.
            statuses = status_filter if status_filter is not None else list(MemoryStatus)
            nodes_by_id: dict[str, MemoryNode] = {}
            for status in statuses:
                for node in await self.graph.get_nodes_by_status(status, limit=1000):
                    nodes_by_id[str(node.id)] = node
            filtered = [n for n in nodes_by_id.values() if self._node_passes_filters(n, filters)]
            filtered.sort(key=lambda n: n.last_reactivated_at, reverse=True)
            memories = [self._format_search_hit(n, 0.0, labels) for n in filtered[:k]]
            await self._tag_superseded(memories)
            return {"query": query, "results": memories, "count": len(memories)}

        if not self.embeddings:
            return {"query": query, "results": [], "count": 0}
        embedding = await self.embeddings.embed(query)

        # Post-query filters would otherwise shrink a k-limited result set, so
        # over-fetch (k*4, capped at 400) when any are present (D09) and cut to
        # k afterwards; the enumeration path already filters before its cut.
        post_filter_keys = {"category", "entity", "since", "active_since", "tags_any", "tags_all", "ref_type", "ref_id"}
        has_post_filters = bool(filters) and any(key in post_filter_keys for key in (filters or {}))
        fetch_k = max(k, min(k * 4, 400)) if has_post_filters else k
        results = await self.graph.vector_search(embedding, k=fetch_k, status_filter=status_filter)

        # Apply additional filters post-query
        memories = []
        for node, score in results:
            if not self._node_passes_filters(node, filters):
                continue
            memories.append(self._format_search_hit(node, score, labels))
            if len(memories) >= k:
                break

        await self._tag_superseded(memories)
        return {"query": query, "results": memories, "count": len(memories)}

    async def _tag_superseded(self, memories: list[dict[str, Any]]) -> None:
        """Mark search hits that a newer memory supersedes.

        Recall already tags superseded hits; without this, search/enumeration
        consumers (e.g. a change-cursor reader) see a superseded memory as
        indistinguishable from a current one. Direction: new(source) -> old
        (target), so only edge TARGETS get tagged.
        """
        ids = [m["id"] for m in memories if m.get("id")]
        if not ids:
            return
        try:
            by_id = {m["id"]: m for m in memories if m.get("id")}
            for edge in await self.graph.get_all_edges(ids):
                if edge.type == EdgeType.SUPERSEDES:
                    tgt = str(edge.target_id)
                    if tgt in by_id:
                        by_id[tgt]["superseded_by"] = str(edge.source_id)
        except Exception:
            logger.warning("superseded tagging failed", exc_info=True)

    async def memory_traverse(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
        space: str | None = None,
        spaces: list[str] | None = None,
        include_personal: bool | None = None,
    ) -> dict[str, Any]:
        """Subgraph traversal returning connected nodes, within the call's scope.

        A start node outside the scope (or missing) returns ``not_found``.
        """
        scope = await self._resolve_scope(space, spaces, include_personal, None, "memberships")
        if scope.error is not None:
            return scope.error
        try:
            start = await self.graph.get_node(node_id)
            if start is None:
                return _not_found(node_id)
            space_of: dict[str, str | None] = {}
            out = await self._traverse(node_id, depth, edge_types, space_of=space_of)
            if start.space_id is not None and "error" not in out:
                # Edges never cross a space boundary, so a traversal reads
                # exactly one space: the start node's (personal starts: none).
                await self._audit_reads(
                    "memory.traverse", [start.space_id], list(space_of.items()), 0,
                )
            return out
        finally:
            scope.reset()

    async def _traverse(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
        *,
        space_of: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        MAX_DEPTH = 10
        if depth > MAX_DEPTH:
            logger.info("memory_traverse depth=%d capped to %d", depth, MAX_DEPTH)
            depth = MAX_DEPTH

        parsed_types = [EdgeType(t) for t in edge_types] if edge_types else None
        org_ids = current_org_ids.get([])
        nodes = await self.graph.traverse(node_id, depth, parsed_types, org_ids=org_ids)

        # Backends diverge on whether traverse() returns the start node itself:
        # the in-memory provider includes it, Postgres excludes it (WHERE id !=
        # start). The traversal's induced subgraph MUST contain the start node,
        # or every edge incident to it is dropped from `edges` — including the
        # only match when edge_types filters to a start-incident edge. Normalize
        # here so all backends behave identically: ensure the start node leads
        # the result set.
        if not any(str(n.id) == node_id for n in nodes):
            start = await self.graph.get_node(node_id)
            if start is not None:
                nodes = [start, *nodes]

        result_nodes = [
            {
                "id": str(n.id),
                "summary": n.content_summary,
                "status": n.status.value,
                "decay_score": round(n.decay_score, 4),
            }
            for n in nodes
        ]
        if space_of is not None:
            space_of.update({str(n.id): n.space_id for n in nodes})

        # Edges of the induced subgraph among the returned nodes (a deliberate
        # superset of the BFS tree, so callers can render/reconstruct paths).
        # Induced-subgraph + edge-type + visibility filtering all live in the
        # storage layer, so no tool-side re-filtering is needed.
        #
        # get_connecting_edges is a NEW Protocol method — Protocols are
        # structural, not enforced, so external providers (genesys-server's
        # postgres/falkordb/mongo/obsidian backends) may not implement it yet.
        # Same precedent as the get_causal_chains_batch guard in memory_recall:
        # degrade to an empty edge list instead of raising AttributeError and
        # breaking one of the original 11 tools on upgrade.
        node_ids = [str(n.id) for n in nodes]
        result_edges: list[dict[str, Any]] = []
        if hasattr(self.graph, "get_connecting_edges"):
            edges = await self.graph.get_connecting_edges(node_ids, parsed_types, org_ids=org_ids)
            result_edges = [
                {
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "type": e.type.value,
                    "weight": round(e.weight, 4),
                    "created_by": e.created_by,
                }
                for e in edges
            ]

        return {
            "start_node": node_id,
            "depth": depth,
            "nodes": result_nodes,
            "count": len(result_nodes),
            "edges": result_edges,
            "edge_count": len(result_edges),
        }

    async def memory_explain(self, node_id: str) -> dict[str, Any]:
        """Score breakdown, causal basis, and removal impact for a memory.

        Uses the caller's full (non-archived) membership scope; anything not
        visible is ``not_found``.
        """
        scope = await self._node_scope(include_archived=False)
        try:
            out = await self._explain(node_id, scope.spaces)
            label = out.get("space") if "error" not in out else None
            if isinstance(label, dict) and label.get("id"):
                await self._audit_reads("memory.explain", [label["id"]], [(node_id, label["id"])], 0)
            return out
        finally:
            scope.reset()

    async def _explain(self, node_id: str, labels: dict[str, Space]) -> dict[str, Any]:
        node = await self.graph.get_node(node_id)
        if not node:
            return _not_found(node_id)

        causal_weight = await self.graph.get_causal_weight(node_id)
        is_orphan = await self.graph.is_orphan(node_id)

        upstream = await self.graph.get_causal_chain(node_id, "upstream")
        downstream = await self.graph.get_causal_chain(node_id, "downstream")

        all_edges = await self.graph.get_edges(node_id, "both")

        if downstream:
            removal_impact = f"Would orphan {len(downstream)} downstream memories"
        elif is_orphan:
            removal_impact = "Safe to remove — no connections"
        else:
            removal_impact = "Low impact — no downstream dependents"

        # Compute Force-2 (connectivity) and Force-3 (activation) LIVE from this
        # node's current edges/reactivation history, so the reader sees fresh
        # contributions rather than the stored (possibly stale) decay_score.
        import math

        from papez.engine import config
        from papez.engine.scoring import base_level_activation

        stats = await self.graph.get_stats()
        max_causal_weight = int(stats.get("max_causal_weight", 0) or 0)
        connectivity_factor_live = _live_connectivity_factor(
            causal_weight, max_causal_weight, is_orphan
        )
        b_i = base_level_activation(node.reactivation_timestamps, node.created_at)
        activation_factor_live = min(max(math.exp(b_i), 0.0), 1.0)

        return {
            "node_id": node_id,
            "summary": node.content_summary,
            "status": node.status.value,
            "decay_score": round(node.decay_score, 4),
            "activation": round(node.decay_score, 4),
            "score_model": {
                "formula": "decay_score = relevance x connectivity_factor x activation_factor",
                "reading": (
                    "Higher = more strongly retained. This is an activation/retention "
                    "score, not a countdown: retrieval RAISES it. A memory is prune-"
                    "eligible only when it falls below "
                    f"{config.FORGETTING_THRESHOLD} AND is orphaned AND not pinned."
                ),
                "forces": {
                    "relevance": "query-dependent; contributes at recall time, not at rest",
                    "connectivity_factor": round(connectivity_factor_live, 4),
                    "activation_factor": round(activation_factor_live, 4),
                },
                "staleness_note": (
                    "stored decay_score is recomputed by evaluate_transitions "
                    "(background worker in hosted deployments); live forces above "
                    "are computed fresh from this node's current edges and "
                    "reactivation history"
                ),
            },
            "stability": round(node.stability, 4),
            "causal_weight": causal_weight,
            "reactivation_count": node.reactivation_count,
            "reactivation_pattern": node.reactivation_pattern.value,
            "pinned": node.pinned,
            "refs": [_ref_dump(r) for r in node.refs],
            "tags": list(node.tags),
            "space": self._space_label(node.space_id, labels),
            "is_orphan": is_orphan,
            "upstream_count": len(upstream),
            "downstream_count": len(downstream),
            "edge_count": len(all_edges),
            "edges": [
                {
                    "id": str(e.id),
                    "type": e.type.value,
                    "weight": round(e.weight, 4),
                    # "target" is the OTHER END of the edge viewed from this
                    # node (legacy name, kept for compatibility). "direction"
                    # disambiguates: an incoming supersedes edge means the
                    # other node supersedes THIS one, not vice versa.
                    "target": str(e.target_id) if str(e.source_id) == node_id else str(e.source_id),
                    "direction": "outgoing" if str(e.source_id) == node_id else "incoming",
                    "reason": e.reason,
                    "created_by": e.created_by,
                    "last_validated_at": e.last_validated_at.isoformat() if e.last_validated_at else None,
                    "stale": _is_edge_stale(e),
                }
                for e in all_edges
            ],
            "removal_impact": removal_impact,
        }

    async def pin_memory(self, node_id: str) -> dict[str, Any]:
        """Pin a memory to core status (space nodes: writer or owner)."""
        scope = await self._node_scope()
        try:
            node = await self.graph.get_node(node_id)
            if not node:
                return _not_found(node_id)
            denied = await self._authorize_write(node, "Only the node owner can pin this memory")
            if denied is not None:
                return denied

            await self.graph.update_node(node_id, {
                "pinned": True,
                "status": MemoryStatus.CORE,
                "promotion_reason": "user_pinned",
            })
            await self._notify("memory.pinned", {"node_id": node_id})
            if node.space_id is not None:
                await self._audit_event("memory.pin", node.space_id, [node_id], {})
            return {"node_id": node_id, "status": "pinned", "new_status": "core"}
        finally:
            scope.reset()

    async def unpin_memory(self, node_id: str) -> dict[str, Any]:
        """Unpin a memory and re-evaluate core eligibility (space nodes: writer or owner)."""
        scope = await self._node_scope()
        try:
            node = await self.graph.get_node(node_id)
            if not node:
                return _not_found(node_id)
            denied = await self._authorize_write(node, "Only the node owner can unpin this memory")
            if denied is not None:
                return denied

            await self.graph.update_node(node_id, {"pinned": False})
            if node.space_id is not None:
                await self._audit_event("memory.unpin", node.space_id, [node_id], {})

            # Re-evaluate: refresh node after unpin, inside the node's own scope
            # so core promotion counts only that scope's edges.
            with self._narrow_scope(node.space_id):
                refreshed = await self.graph.get_node(node_id)
                if not refreshed:
                    return {"node_id": node_id, "status": "unpinned", "new_status": "unknown"}
                should_stay_core, reason = await evaluate_core_promotion(refreshed, self.graph)

            if not should_stay_core:
                await self.graph.update_node(node_id, {
                    "status": MemoryStatus.ACTIVE,
                    "promotion_reason": None,
                })
                await self._notify("memory.unpinned", {"node_id": node_id, "new_status": "active"})
                return {"node_id": node_id, "status": "unpinned", "new_status": "active"}

            await self._notify("memory.unpinned", {"node_id": node_id, "new_status": "core"})
            return {"node_id": node_id, "status": "unpinned", "new_status": "core", "reason": reason}
        finally:
            scope.reset()

    async def list_core_memories(self, category: str | None = None, space: str | None = None) -> dict[str, Any]:
        """List core memories (personal by default, or one space's), optionally by category."""
        scope = await self._resolve_scope(space, None, None, None, "personal")
        if scope.error is not None:
            return scope.error
        try:
            nodes = await self.graph.get_nodes_by_status(MemoryStatus.CORE, limit=500)
        finally:
            scope.reset()

        if category:
            nodes = [n for n in nodes if n.category == category]

        # Sort by causal_weight descending
        nodes.sort(key=lambda n: n.causal_weight, reverse=True)

        memories = [
            {
                "id": str(n.id),
                "summary": n.content_summary,
                "category": n.category,
                "causal_weight": n.causal_weight,
                "pinned": n.pinned,
                "promotion_reason": n.promotion_reason,
                "created_at": n.created_at.isoformat(),
                "tags": list(n.tags),
                "space": self._space_label(n.space_id, scope.spaces),
            }
            for n in nodes
        ]

        return {"core_memories": memories, "count": len(memories)}

    async def delete_memory(self, node_id: str) -> dict[str, Any]:
        """Hard delete a memory node and all its edges (space nodes: writer or owner)."""
        scope = await self._node_scope()
        try:
            node = await self.graph.get_node(node_id)
            if not node:
                return _not_found(node_id)
            denied = await self._authorize_write(node, "Only the node owner can delete this memory")
            if denied is not None:
                return denied

            await self.graph.delete_node(node_id)
            await self._notify("memory.deleted", {"node_id": node_id})
            if node.space_id is not None:
                await self._audit_event("memory.delete", node.space_id, [node_id], {})
            return {"node_id": node_id, "status": "deleted"}
        finally:
            scope.reset()

    async def memory_stats(self, space: str | None = None) -> dict[str, Any]:
        """Return graph statistics: personal by default, or one space (reader+)."""
        scope = await self._resolve_scope(space, None, None, None, "personal")
        if scope.error is not None:
            return scope.error
        try:
            stats = await self.graph.get_stats()
        finally:
            scope.reset()
        return stats

    async def set_core_preferences(
        self,
        auto: list[str] | None = None,
        approval: list[str] | None = None,
        excluded: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update core memory category preferences."""
        result = await self.preferences.update(auto=auto, approval=approval, excluded=excluded)
        return {"status": "updated", "preferences": result}

    async def promote_to_org(
        self,
        node_id: str,
        org_id: str,
        action: str = "keep_private",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Promote a private memory to org visibility.

        Args:
            action: How to handle cross-boundary edges.
                "keep_private" (default) — edges to private nodes stay but are invisible to other org members.
                "promote_all" — also promote directly linked private nodes (1 level deep).
                "delete_links" — remove edges to private nodes before promoting.
            dry_run: If True, return a preview of what would change without applying it.

        Space memories can't be promoted to org (``invalid_arguments``).
        """
        scope = await self._node_scope()
        try:
            return await self._promote_to_org(node_id, org_id, action, dry_run)
        finally:
            scope.reset()

    async def _promote_to_org(self, node_id: str, org_id: str, action: str, dry_run: bool) -> dict[str, Any]:
        node = await self.graph.get_node(node_id)
        if not node:
            return _not_found(node_id)
        if node.space_id is not None:
            return _error("invalid_arguments", "space memories can't be promoted to org", node_id=node_id)
        if node.visibility == Visibility.ORG:
            return _error("invalid_arguments", "Already org-visible", node_id=node_id)
        if org_id not in current_org_ids.get([]):
            return _error("invalid_arguments", "org_id not in caller's org memberships")
        if not _caller_owns_node(node):
            return _error("insufficient_role", "Only the node owner can promote to org")

        # Check for cross-boundary edges (this node's edges to other private nodes)
        all_edges = await self.graph.get_edges(node_id, "both")
        cross_boundary: list[dict[str, Any]] = []
        for e in all_edges:
            other_id = str(e.target_id) if str(e.source_id) == node_id else str(e.source_id)
            other = await self.graph.get_node(other_id)
            if other and other.visibility == Visibility.PRIVATE:
                cross_boundary.append({
                    "edge_id": str(e.id),
                    "linked_node_id": other_id,
                    "linked_summary": other.content_summary,
                    "edge_type": e.type.value,
                })

        # Compute the diff for each action
        nodes_promoted: list[dict[str, str]] = [{"id": node_id, "summary": node.content_summary}]
        edges_deleted: list[dict[str, str]] = []
        edges_preserved: list[dict[str, str]] = []
        nodes_skipped: list[dict[str, str]] = []

        if action == "delete_links" and cross_boundary:
            for cb in cross_boundary:
                edges_deleted.append({"edge_id": cb["edge_id"], "linked_node_id": cb["linked_node_id"], "linked_summary": cb["linked_summary"]})
        elif action == "promote_all" and cross_boundary:
            for cb in cross_boundary:
                linked = await self.graph.get_node(cb["linked_node_id"])
                if linked and linked.visibility == Visibility.PRIVATE:
                    second_edges = await self.graph.get_edges(cb["linked_node_id"], "both")
                    has_further_private = False
                    for se in second_edges:
                        se_other = str(se.target_id) if str(se.source_id) == cb["linked_node_id"] else str(se.source_id)
                        if se_other == node_id:
                            continue
                        se_node = await self.graph.get_node(se_other)
                        if se_node and se_node.visibility == Visibility.PRIVATE:
                            has_further_private = True
                            break
                    if has_further_private:
                        nodes_skipped.append({"id": cb["linked_node_id"], "summary": cb["linked_summary"], "reason": "has further private links"})
                        continue
                    nodes_promoted.append({"id": cb["linked_node_id"], "summary": cb["linked_summary"]})
        elif action == "keep_private" and cross_boundary:
            for cb in cross_boundary:
                edges_preserved.append({"edge_id": cb["edge_id"], "linked_node_id": cb["linked_node_id"], "linked_summary": cb["linked_summary"]})

        if dry_run:
            return {
                "node_id": node_id,
                "dry_run": True,
                "action": action,
                "nodes_promoted": nodes_promoted,
                "edges_deleted": edges_deleted,
                "edges_preserved": edges_preserved,
                "nodes_skipped": nodes_skipped,
            }

        # Apply changes
        if action == "delete_links" and cross_boundary:
            for cb in cross_boundary:
                await self.graph.delete_edge(cb["edge_id"])
        elif action == "promote_all" and cross_boundary:
            for entry in nodes_promoted[1:]:
                await self.graph.promote_to_org(entry["id"], org_id)

        await self.graph.promote_to_org(node_id, org_id)
        await self._notify("memory.promoted", {"node_id": node_id, "org_id": org_id})
        return {
            "node_id": node_id,
            "status": "promoted",
            "org_id": org_id,
            "action": action,
            "nodes_promoted": nodes_promoted,
            "edges_deleted": edges_deleted,
            "edges_preserved": edges_preserved,
            "nodes_skipped": nodes_skipped,
        }

    async def erase_user(
        self, user_id: str, keep_promoted_nodes: bool = True,
    ) -> dict[str, Any]:
        """GDPR Article 17: erase all data belonging to a user.

        Args:
            keep_promoted_nodes: If True (default), org-promoted nodes are
                anonymized (original_user_id set to "erased_user") and PII
                is scrubbed from connected edge reasons. If False, all nodes
                are deleted regardless of visibility.
        """
        uid = _caller_uid()
        if uid is None:
            raise PermissionError("current_user_id not set — cannot erase")
        role = current_user_role.get(None)
        if uid != user_id and role != "admin":
            return _error("insufficient_role", "Only the user or an admin can erase user data")

        manifest = await self.graph.erase_user(user_id, keep_promoted_nodes=keep_promoted_nodes)
        await self.cache.delete(f"core_preferences:{user_id}")
        result: dict[str, Any] = {
            "status": "erased",
            "user_id": user_id,
            **manifest,
        }
        if self.spaces is not None:
            # D32: drop the user's memberships, except where they are the last
            # user owner (left in place and reported, so the space stays owned).
            still_owned: list[str] = []
            for space, _role in await self.spaces.list_spaces_for_principal(user_id, include_archived=True):
                existing = await self._get_member(space.id, user_id)
                try:
                    await self.spaces.remove_member(space.id, user_id)
                except ValueError as exc:
                    if str(exc) == "last_owner":
                        still_owned.append(space.id)
                        logger.warning("erase_user: %s is the last owner of space %s", user_id, space.id)
                    else:
                        raise
                    continue
                if existing is not None:
                    # Membership changed in a space: spec 8.1 member.remove.
                    await self._audit_event("member.remove", space.id, None, {
                        "member_id": user_id,
                        "member_type": _role_value(existing.principal_type),
                        "role": _role_value(existing.role),
                        "via": "erase_user",
                    })
            result["spaces_still_owned"] = still_owned
        await self._notify("user.erased", {"user_id": user_id, "manifest": manifest})
        return result

    # -- Mind Spaces tools (spec 6.8) ---------------------------------------------

    @staticmethod
    def _space_dict(space: Space, role: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": space.id,
            "slug": space.slug,
            "name": space.name,
            "org_id": space.org_id,
            "description": space.description,
            "forgetting_policy": _role_value(space.forgetting_policy),
            "llm_inference_enabled": space.llm_inference_enabled,
            "archived": space.archived,
            "archived_at": space.archived_at.isoformat() if space.archived_at else None,
            "created_at": space.created_at.isoformat(),
            "created_by": space.created_by,
        }
        if role is not None:
            out["your_role"] = _role_value(role)
        return out

    @staticmethod
    def _member_dict(member: SpaceMember) -> dict[str, Any]:
        return {
            "principal_id": member.principal_id,
            "principal_type": _role_value(member.principal_type),
            "role": _role_value(member.role),
            "added_by": member.added_by,
            "added_at": member.added_at.isoformat(),
        }

    def _registry_error(self) -> dict[str, Any] | None:
        if self.spaces is None:
            return _error("invalid_arguments", "Mind Spaces are not available on this server")
        return None

    async def _memory_count(self, space_id: str) -> int:
        with self._narrow_scope(space_id):
            stats = await self.graph.get_stats()
        return int(stats.get("nodes", stats.get("total_nodes", 0)) or 0)

    async def _space_with_counts(self, space: Space, role: Any) -> dict[str, Any]:
        assert self.spaces is not None
        out = self._space_dict(space, role)
        out["member_count"] = await self.spaces.count_members(space.id)
        out["memory_count"] = await self._memory_count(space.id)
        return out

    async def _space_for(
        self, space: str, need_role: SpaceRole, *, allow_archived: bool | None = None,
    ) -> tuple[Space, str] | dict[str, Any]:
        """Resolve ``space`` for a space-management tool; returns (Space, role) or an error."""
        missing = self._registry_error()
        if missing is not None:
            return missing
        if not isinstance(space, str) or not space:
            return _space_not_found()
        scope = await self._resolve_scope(space, None, None, need_role, allow_archived=allow_archived)
        scope.reset()  # management tools read the registry, not the graph scope
        if scope.error is not None:
            return scope.error
        sid = scope.space_ids[0]
        found = scope.spaces.get(sid)
        if found is None:
            return _space_not_found()
        return found, scope.roles.get(sid, need_role.value)

    async def space_create(
        self,
        name: str,
        slug: str | None = None,
        org_id: str | None = None,
        description: str | None = None,
        forgetting_policy: str = "retain_all",
        llm_inference_enabled: bool = True,
    ) -> dict[str, Any]:
        """Create a space; the caller becomes its first owner. Users only."""
        from papez.storage.spaces import unique_slug

        missing = self._registry_error()
        if missing is not None:
            return missing
        assert self.spaces is not None
        uid = _caller_uid()
        if uid is None:
            raise PermissionError("current_user_id not set — cannot create a space")
        if _is_service():
            return _error("insufficient_role", "Service principals can't create spaces")
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            return _error("invalid_arguments", "name must be 1-120 characters")
        orgs = list(current_org_ids.get([]))
        if org_id is not None:
            if org_id not in orgs:
                return _error("invalid_arguments", "org_id not in caller's org memberships")
        elif len(orgs) == 1:
            org_id = orgs[0]
        elif not orgs:
            return _error("invalid_arguments", "You need to belong to an org to create a space")
        else:
            return _error("invalid_arguments", "You belong to several orgs; pass org_id")
        try:
            policy = ForgettingPolicy(forgetting_policy)
        except ValueError:
            return _error("invalid_arguments", "forgetting_policy must be 'retain_all' or 'standard'")
        if not isinstance(llm_inference_enabled, bool):
            return _error("invalid_arguments", "llm_inference_enabled must be a boolean")
        if slug is not None:
            if not isinstance(slug, str) or not _SLUG_RE.match(slug):
                return _error("invalid_arguments", "slug must match ^[a-z0-9][a-z0-9-]{1,62}$")
            final_slug = slug
        else:
            taken_fn = getattr(self.spaces, "slugs_for_org", None)
            if taken_fn is not None:
                # Sync on InMemorySpaceRegistry, async on hosted registries.
                raw = taken_fn(org_id)
                taken = set(await raw if inspect.isawaitable(raw) else raw)
            else:
                taken = {s.slug for s in await self.spaces.list_spaces_for_org(org_id, include_archived=True)}
            final_slug = unique_slug(name, taken)
        try:
            new_space = Space(
                org_id=org_id,
                name=name,
                slug=final_slug,
                description=description,
                forgetting_policy=policy,
                llm_inference_enabled=llm_inference_enabled,
                created_by=uid,
            )
        except Exception as exc:  # pydantic ValidationError
            return _error("invalid_arguments", f"invalid space: {exc.__class__.__name__}")
        try:
            async with self._atomic():
                created = await self.spaces.create_space(new_space)
                await self.spaces.add_member(SpaceMember(
                    space_id=created.id,
                    principal_id=uid,
                    principal_type=PrincipalType.USER,
                    role=SpaceRole.OWNER,
                    added_by=uid,
                ))
        except ValueError as exc:
            if str(exc) == "slug_taken":
                return _error("slug_taken", "That slug is already used in this org")
            return _error("invalid_arguments", "invalid space")
        await self._audit_event("space.create", created.id, None, {"org_id": created.org_id})
        return {"space": self._space_dict(created, SpaceRole.OWNER)}

    async def space_list(self, include_archived: bool = False) -> dict[str, Any]:
        """Spaces the caller is a member of, sorted by name."""
        missing = self._registry_error()
        if missing is not None:
            return missing
        assert self.spaces is not None
        uid = _caller_uid()
        if uid is None:
            raise PermissionError("current_user_id not set — cannot list spaces")
        pairs = await self.spaces.list_spaces_for_principal(uid, include_archived=bool(include_archived))
        items = [await self._space_with_counts(space, role) for space, role in pairs]
        items.sort(key=lambda d: (str(d["name"]).lower(), d["id"]))
        return {"spaces": items}

    async def space_get(self, space: str) -> dict[str, Any]:
        """One space with member and memory counts (reader+)."""
        hit = await self._space_for(space, SpaceRole.READER)
        if isinstance(hit, dict):
            return hit
        found, role = hit
        return {"space": await self._space_with_counts(found, role)}

    async def space_update(
        self,
        space: str,
        name: str | None = None,
        description: str | None = None,
        forgetting_policy: str | None = None,
        llm_inference_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Change name/description/policies (owner). The slug never changes."""
        hit = await self._space_for(space, SpaceRole.OWNER)
        if isinstance(hit, dict):
            return hit
        found, role = hit
        assert self.spaces is not None
        updates: dict[str, Any] = {}
        if name is not None:
            if not isinstance(name, str) or not name.strip() or len(name) > 120:
                return _error("invalid_arguments", "name must be 1-120 characters")
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if forgetting_policy is not None:
            try:
                updates["forgetting_policy"] = ForgettingPolicy(forgetting_policy)
            except ValueError:
                return _error("invalid_arguments", "forgetting_policy must be 'retain_all' or 'standard'")
        if llm_inference_enabled is not None:
            if not isinstance(llm_inference_enabled, bool):
                return _error("invalid_arguments", "llm_inference_enabled must be a boolean")
            updates["llm_inference_enabled"] = llm_inference_enabled
        if not updates:
            return {"space": self._space_dict(found, role)}
        try:
            updated = await self.spaces.update_space(found.id, updates)
        except ValueError:
            return _error("invalid_arguments", "invalid space update")
        # Field names only: names and descriptions are content, never audited.
        await self._audit_event("space.update", found.id, None, {"fields": sorted(updates)})
        return {"space": self._space_dict(updated, role)}

    async def space_member_add(
        self, space: str, principal_id: str, principal_type: str, role: str,
    ) -> dict[str, Any]:
        """Add a member or change their role (owner)."""
        hit = await self._space_for(space, SpaceRole.OWNER)
        if isinstance(hit, dict):
            return hit
        found, _ = hit
        assert self.spaces is not None
        if not isinstance(principal_id, str) or not principal_id:
            return _error("invalid_arguments", "principal_id is required")
        try:
            ptype = PrincipalType(principal_type)
            new_role = SpaceRole(role)
        except ValueError:
            return _error("invalid_arguments", "principal_type must be user|service and role owner|writer|reader")
        if ptype == PrincipalType.SERVICE and new_role == SpaceRole.OWNER:
            return _error("invalid_arguments", "A service principal can't be an owner")
        if self.principal_in_org is not None:
            if not await self.principal_in_org(found.org_id, principal_id):
                return _error("principal_not_in_org", "That principal doesn't belong to the space's org")
        uid = _caller_uid() or ""
        existing = await self._get_member(found.id, principal_id)
        try:
            member = await self.spaces.add_member(SpaceMember(
                space_id=found.id,
                principal_id=principal_id,
                principal_type=ptype,
                role=new_role,
                added_by=uid,
            ))
        except ValueError as exc:
            if str(exc) == "last_owner":
                return _error("last_owner", "A space must keep at least one user owner")
            return _error("invalid_arguments", "invalid member")
        detail: dict[str, Any] = {
            "member_id": principal_id, "member_type": ptype.value, "role": new_role.value,
        }
        if existing is None:
            await self._audit_event("member.add", found.id, None, detail)
        else:
            detail["previous_role"] = _role_value(existing.role)
            await self._audit_event("member.role_change", found.id, None, detail)
        return {"member": self._member_dict(member), "created": existing is None}

    async def _get_member(self, space_id: str, principal_id: str) -> SpaceMember | None:
        assert self.spaces is not None
        getter = getattr(self.spaces, "get_member", None)
        if getter is not None:
            member: SpaceMember | None = await getter(space_id, principal_id)
            return member
        return next((m for m in await self.spaces.list_members(space_id) if m.principal_id == principal_id), None)

    async def space_member_remove(self, space: str, principal_id: str) -> dict[str, Any]:
        """Remove a member (owner). Takes effect on their next request."""
        hit = await self._space_for(space, SpaceRole.OWNER)
        if isinstance(hit, dict):
            return hit
        found, _ = hit
        assert self.spaces is not None
        existing = await self._get_member(found.id, principal_id) if isinstance(principal_id, str) else None
        try:
            await self.spaces.remove_member(found.id, principal_id)
        except ValueError as exc:
            if str(exc) == "last_owner":
                return _error("last_owner", "A space must keep at least one user owner")
            return _space_not_found()
        if existing is not None:
            await self._audit_event("member.remove", found.id, None, {
                "member_id": principal_id,
                "member_type": _role_value(existing.principal_type),
                "role": _role_value(existing.role),
            })
        return {"removed": True}

    async def space_member_list(self, space: str) -> dict[str, Any]:
        """Members of a space (reader+)."""
        hit = await self._space_for(space, SpaceRole.READER)
        if isinstance(hit, dict):
            return hit
        found, _ = hit
        assert self.spaces is not None
        return {"members": [self._member_dict(m) for m in await self.spaces.list_members(found.id)]}

    async def space_archive(self, space: str) -> dict[str, Any]:
        """Archive a space (owner): read-only, out of default recall."""
        hit = await self._space_for(space, SpaceRole.OWNER)
        if isinstance(hit, dict):
            return hit
        found, role = hit
        assert self.spaces is not None
        updated = await self.spaces.update_space(found.id, {"archived_at": datetime.now(UTC)})
        await self._audit_event("space.archive", found.id, None, {})
        return {"space": self._space_dict(updated, role)}

    async def space_unarchive(self, space: str) -> dict[str, Any]:
        """Unarchive a space (owner)."""
        hit = await self._space_for(space, SpaceRole.OWNER, allow_archived=True)
        if isinstance(hit, dict):
            return hit
        found, role = hit
        assert self.spaces is not None
        if not found.archived:
            return {"space": self._space_dict(found, role)}
        updated = await self.spaces.update_space(found.id, {"archived_at": None})
        await self._audit_event("space.unarchive", found.id, None, {})
        return {"space": self._space_dict(updated, role)}

    async def space_erase(self, space: str, confirm_slug: str, dry_run: bool = False) -> dict[str, Any]:
        """Permanently delete a space, its memories, edges and memberships (owner).

        ``confirm_slug`` must equal the slug. ``dry_run`` reports the counts
        and deletes nothing. The id is never reused; audit rows are kept.
        """
        hit = await self._space_for(space, SpaceRole.OWNER, allow_archived=True)
        if isinstance(hit, dict):
            return hit
        found, _ = hit
        assert self.spaces is not None
        if confirm_slug != found.slug:
            return _error("confirmation_mismatch", "confirm_slug doesn't match the space's slug")
        members = await self.spaces.count_members(found.id)
        if dry_run:
            counts = await self.graph.erase_space(found.id, dry_run=True)
            return {"erased": {"space_id": found.id, "nodes": counts.get("nodes", 0),
                               "edges": counts.get("edges", 0), "members": members}, "dry_run": True}
        async with self._atomic():
            counts = await self.graph.erase_space(found.id)
            await self.spaces.mark_erased(found.id)
        await self._notify("space.erased", {"space_id": found.id})
        # Audit events outlive the space (spec 8.3); counts only.
        await self._audit_event("space.erase", found.id, None, {
            "nodes": counts.get("nodes", 0), "edges": counts.get("edges", 0), "members": members,
        })
        return {"erased": {"space_id": found.id, "nodes": counts.get("nodes", 0),
                           "edges": counts.get("edges", 0), "members": members}, "dry_run": False}

    async def promote_to_space(self, node_id: str, space: str, keep_original: bool = True) -> dict[str, Any]:
        """Copy a personal memory the caller owns into a space (writer or owner; D29).

        The copy keeps content, category, refs, tags, created_at, source_agent
        and source_session, records ``promoted_from``, and gets no edges.
        ``keep_original=False`` deletes the personal original afterwards.
        """
        outer = await self._resolve_scope(None, None, None, None, "personal")
        try:
            original = await self.graph.get_node(node_id)
            if original is None or original.space_id is not None:
                return _not_found(node_id)
            uid = _caller_uid()
            owner = original.original_user_id
            if (owner is not None and owner != uid) or not _caller_owns_node(original):
                return _error("insufficient_role", "Only the node owner can promote this memory")
            inner = await self._resolve_scope(space, None, None, SpaceRole.WRITER)
            if inner.error is not None:
                return inner.error
            try:
                sid = inner.space_ids[0]
                result = await self._write_memory(_Validated(
                    content=original.content_full or original.content_summary,
                    source_session=original.source_session,
                    visibility=Visibility.PRIVATE,
                    org_id=None,
                    category=original.category,
                    refs=list(original.refs),
                    tags=list(original.tags),
                    space_id=sid,
                    created_at=original.created_at,
                    source_agent=original.source_agent,
                    promoted_from=str(original.id),
                ))
                label = self._space_label(sid, inner.spaces)
            finally:
                inner.reset()
            if "error" in result:
                return result
            if not keep_original:
                await self.graph.delete_node(node_id)
                await self._notify("memory.deleted", {"node_id": node_id})
            await self._audit_event("memory.promote_in", sid, [result["node_id"]], {
                "promoted_from": str(original.id), "keep_original": bool(keep_original),
            })
            return {"node_id": result["node_id"], "space": label}
        finally:
            outer.reset()
