"""In-memory ``SpaceRegistry`` with optional sidecar persistence (spec 7.5, D05).

The registry file sits next to the graph state file (``<persist>.spaces.json``)
and has the shape ``{"version": 1, "spaces": {id: dump}, "members": [dumps]}``.
``__init__`` does no I/O: ``initialize()`` loads the file, every mutation
rewrites it, and the file is created on the first mutation.

Errors are ``ValueError`` whose message is a spec Section 10 code
(``slug_taken``, ``last_owner``, ``invalid_arguments``, ``space_not_found``);
the tool layer translates them into ``{"error", "code"}`` dicts.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from papez.models.enums import PrincipalType, SpaceRole
from papez.models.space import SLUG_PATTERN, Space, SpaceMember

logger = logging.getLogger(__name__)

REGISTRY_FILE_VERSION = 1

# Fields a caller may change through update_space. ``slug`` is immutable
# (other systems may store it); ``archived_at`` is how archive/unarchive is done.
_UPDATABLE_FIELDS = frozenset({
    "name", "description", "forgetting_policy", "llm_inference_enabled", "archived_at", "show_service_names",
})

_SLUG_RE = re.compile(SLUG_PATTERN)
_SLUG_MAX_LEN = 63


def slugify(name: str) -> str:
    """Spec 6.8 slug from a name: lowercase, whitespace to ``-``, invalid characters removed.

    Runs of ``-`` collapse to one and leading/trailing ``-`` are stripped. The
    result is cut to 63 characters. It can still be invalid (empty, or one
    character); ``unique_slug`` pads those so the result matches ``SLUG_PATTERN``.
    """
    s = (name or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:_SLUG_MAX_LEN].rstrip("-")


def unique_slug(base: str, taken: Iterable[str]) -> str:
    """Return ``base`` or ``base-2``, ``base-3``, ... whichever is not in ``taken``.

    ``base`` is passed through ``slugify`` first; an empty base becomes
    ``"space"`` and a one-character base is padded with ``-1`` so the result
    always matches ``SLUG_PATTERN``. Suffixed candidates are trimmed so they
    stay within 63 characters.
    """
    taken_set = set(taken)
    slug = slugify(base)
    if not slug:
        slug = "space"
    if len(slug) < 2:
        slug = f"{slug}-1"
    if slug not in taken_set:
        return slug
    n = 2
    while True:
        suffix = f"-{n}"
        candidate = slug[: _SLUG_MAX_LEN - len(suffix)].rstrip("-") + suffix
        if candidate not in taken_set:
            return candidate
        n += 1


def _role(value: Any) -> SpaceRole:
    return value if isinstance(value, SpaceRole) else SpaceRole(value)


def _ptype(value: Any) -> PrincipalType:
    return value if isinstance(value, PrincipalType) else PrincipalType(value)


class InMemorySpaceRegistry:
    """Spaces and memberships held in memory, optionally persisted to a sidecar file.

    Erased spaces keep their row (``erased_at`` set) so ids are never reused,
    but they vanish from every listing, from ``get_space``, from slug lookup
    and from ``roles_for_principal``. Their slug stays reserved within the org.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._spaces: dict[str, Space] = {}
        # space_id -> {principal_id -> member}
        self._members: dict[str, dict[str, SpaceMember]] = {}
        # space_id -> erased_at (ISO string); erased rows stay in _spaces.
        self._erased: dict[str, str] = {}
        self._loaded = False

    # ── persistence ───────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load the sidecar file if it exists. Safe to call more than once."""
        self._load()

    def _load(self) -> None:
        self._loaded = True
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
        except (OSError, ValueError) as exc:
            logger.error("space registry: could not read %s: %s", self._persist_path, exc)
            raise
        spaces: dict[str, Space] = {}
        erased: dict[str, str] = {}
        for sid, raw in (data.get("spaces") or {}).items():
            raw = dict(raw)
            erased_at = raw.pop("erased_at", None)
            space = Space.model_validate(raw)
            spaces[sid] = space
            if erased_at:
                erased[sid] = erased_at
        members: dict[str, dict[str, SpaceMember]] = {}
        for raw in data.get("members") or []:
            m = SpaceMember.model_validate(raw)
            members.setdefault(m.space_id, {})[m.principal_id] = m
        self._spaces, self._erased, self._members = spaces, erased, members

    def _save(self) -> None:
        if self._persist_path is None:
            return
        spaces_out: dict[str, Any] = {}
        for sid, space in self._spaces.items():
            dump = space.model_dump(mode="json")
            if sid in self._erased:
                dump["erased_at"] = self._erased[sid]
            spaces_out[sid] = dump
        data = {
            "version": REGISTRY_FILE_VERSION,
            "spaces": spaces_out,
            "members": [
                m.model_dump(mode="json")
                for by_principal in self._members.values()
                for m in by_principal.values()
            ],
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_name(self._persist_path.name + ".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._persist_path)

    def _ensure_loaded(self) -> None:
        # Mutations before initialize() must not clobber an existing file.
        if not self._loaded:
            self._load()

    # ── helpers ───────────────────────────────────────────────────────────

    def _live(self, space_id: str) -> Space | None:
        if space_id in self._erased:
            return None
        return self._spaces.get(space_id)

    def _require_live(self, space_id: str) -> Space:
        space = self._live(space_id)
        if space is None:
            raise ValueError("space_not_found")
        return space

    def slugs_for_org(self, org_id: str) -> set[str]:
        """Every slug in the org, including archived and erased spaces (slugs are never reused)."""
        self._ensure_loaded()
        return {s.slug for s in self._spaces.values() if s.org_id == org_id}

    def _user_owner_ids(self, space_id: str) -> set[str]:
        return {
            pid for pid, m in self._members.get(space_id, {}).items()
            if m.role == SpaceRole.OWNER and m.principal_type == PrincipalType.USER
        }

    # ── spaces ────────────────────────────────────────────────────────────

    async def create_space(self, space: Space) -> Space:
        self._ensure_loaded()
        if space.id in self._spaces:
            raise ValueError("invalid_arguments")
        if space.slug in self.slugs_for_org(space.org_id):
            raise ValueError("slug_taken")
        stored = space.model_copy(deep=True)
        self._spaces[stored.id] = stored
        self._save()
        return stored.model_copy(deep=True)

    async def get_space(self, space_id: str) -> Space | None:
        self._ensure_loaded()
        space = self._live(space_id)
        return space.model_copy(deep=True) if space else None

    async def get_space_by_slug(self, org_id: str, slug: str) -> Space | None:
        self._ensure_loaded()
        for sid, space in self._spaces.items():
            if space.org_id == org_id and space.slug == slug and sid not in self._erased:
                return space.model_copy(deep=True)
        return None

    async def list_spaces_for_principal(
        self, principal_id: str, include_archived: bool = False,
    ) -> list[tuple[Space, SpaceRole]]:
        self._ensure_loaded()
        out: list[tuple[Space, SpaceRole]] = []
        for sid, by_principal in self._members.items():
            member = by_principal.get(principal_id)
            space = self._live(sid)
            if member is None or space is None:
                continue
            if space.archived and not include_archived:
                continue
            out.append((space.model_copy(deep=True), member.role))
        out.sort(key=lambda pair: (pair[0].name.lower(), pair[0].id))
        return out

    async def list_spaces_for_org(self, org_id: str, include_archived: bool = False) -> list[Space]:
        self._ensure_loaded()
        out = [
            s.model_copy(deep=True) for sid, s in self._spaces.items()
            if s.org_id == org_id and sid not in self._erased and (include_archived or not s.archived)
        ]
        out.sort(key=lambda s: (s.name.lower(), s.id))
        return out

    async def update_space(self, space_id: str, updates: dict[str, Any]) -> Space:
        self._ensure_loaded()
        current = self._require_live(space_id)
        if not isinstance(updates, dict) or "slug" in updates:
            raise ValueError("invalid_arguments")
        unknown = set(updates) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError("invalid_arguments")
        merged = current.model_dump()
        merged.update(updates)
        try:
            updated = Space.model_validate(merged)
        except Exception as exc:  # pydantic ValidationError
            raise ValueError("invalid_arguments") from exc
        self._spaces[space_id] = updated
        self._save()
        return updated.model_copy(deep=True)

    async def mark_erased(self, space_id: str) -> None:
        """Keep the row with ``erased_at`` set and drop its memberships."""
        self._ensure_loaded()
        if space_id not in self._spaces:
            raise ValueError("space_not_found")
        if space_id in self._erased:
            return
        self._erased[space_id] = datetime.now(UTC).isoformat()
        self._members.pop(space_id, None)
        self._save()

    def erased_at(self, space_id: str) -> str | None:
        """ISO timestamp the space was erased, or None. Not part of the protocol."""
        self._ensure_loaded()
        return self._erased.get(space_id)

    # ── members ───────────────────────────────────────────────────────────

    async def add_member(self, member: SpaceMember) -> SpaceMember:
        """Insert or update (upsert on ``(space_id, principal_id)``)."""
        self._ensure_loaded()
        self._require_live(member.space_id)
        role = _role(member.role)
        ptype = _ptype(member.principal_type)
        if ptype == PrincipalType.SERVICE and role == SpaceRole.OWNER:
            raise ValueError("invalid_arguments")
        by_principal = self._members.setdefault(member.space_id, {})
        existing = by_principal.get(member.principal_id)
        if existing is not None:
            owners = self._user_owner_ids(member.space_id)
            if role != SpaceRole.OWNER and owners == {member.principal_id}:
                raise ValueError("last_owner")
            # Keep the original added_at/added_by on an upsert.
            stored = existing.model_copy(update={"role": role, "principal_type": ptype})
        else:
            stored = member.model_copy(update={"role": role, "principal_type": ptype}, deep=True)
        by_principal[member.principal_id] = stored
        self._save()
        return stored.model_copy(deep=True)

    async def remove_member(self, space_id: str, principal_id: str) -> None:
        """Remove a membership. Removing a non-member is a no-op (idempotent)."""
        self._ensure_loaded()
        self._require_live(space_id)
        by_principal = self._members.get(space_id, {})
        if principal_id not in by_principal:
            return
        if self._user_owner_ids(space_id) == {principal_id}:
            raise ValueError("last_owner")
        del by_principal[principal_id]
        self._save()

    async def get_member(self, space_id: str, principal_id: str) -> SpaceMember | None:
        """Not part of the protocol; convenience for ``created`` in ``space_member_add``."""
        self._ensure_loaded()
        if self._live(space_id) is None:
            return None
        m = self._members.get(space_id, {}).get(principal_id)
        return m.model_copy(deep=True) if m else None

    async def list_members(self, space_id: str) -> list[SpaceMember]:
        self._ensure_loaded()
        if self._live(space_id) is None:
            return []
        out = [m.model_copy(deep=True) for m in self._members.get(space_id, {}).values()]
        out.sort(key=lambda m: (m.added_at, m.principal_id))
        return out

    async def count_members(self, space_id: str) -> int:
        self._ensure_loaded()
        if self._live(space_id) is None:
            return 0
        return len(self._members.get(space_id, {}))

    async def roles_for_principal(
        self, principal_id: str, include_archived: bool = False,
    ) -> dict[str, SpaceRole]:
        self._ensure_loaded()
        out: dict[str, SpaceRole] = {}
        for sid, by_principal in self._members.items():
            member = by_principal.get(principal_id)
            space = self._live(sid)
            if member is None or space is None:
                continue
            if space.archived and not include_archived:
                continue
            out[sid] = member.role
        return out
