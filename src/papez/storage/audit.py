"""In-memory ``AuditSink`` (spec 8.2, 8.3, D18).

Events are appended to ``.events`` (oldest first), capped at ``max_events``
(oldest dropped). Not persisted. ``query`` is a test/export helper returning
events newest first with an opaque keyset cursor ``"<at_iso>|<id>"``.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from papez.engine import config


def audit_timestamp(dt: datetime | None = None) -> str:
    """ISO-8601 UTC with milliseconds and ``Z``, e.g. ``2026-11-01T18:04:55.120Z``."""
    dt = (dt or datetime.now(UTC)).astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _as_iso(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return audit_timestamp(value)
    return value


class InMemoryAuditSink:
    """Append-only, capped, in-process audit log."""

    def __init__(self, max_events: int | None = None) -> None:
        cap = config.AUDIT_MAX_EVENTS if max_events is None else max_events
        self.max_events = max(0, int(cap))
        # Oldest first. A plain list so tests can inspect or clear it directly.
        self.events: list[dict[str, Any]] = []

    async def record(self, event: dict[str, Any]) -> None:
        """Append one event. Fills ``id`` and ``at`` when missing; never raises on shape."""
        if self.max_events == 0:
            return
        ev = dict(event)
        ev.setdefault("id", new_event_id())
        ev.setdefault("at", audit_timestamp())
        self.events.append(ev)
        overflow = len(self.events) - self.max_events
        if overflow > 0:
            del self.events[:overflow]

    def query(
        self,
        space_id: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Events for ``space_id``, newest first, ``since <= at <= until``.

        Ordering key is ``(at, id)`` descending. ``cursor`` is the value returned
        as ``next_cursor`` by the previous page; it is ``None`` when there are no
        more events. Raises ``ValueError("invalid_arguments")`` on a malformed
        cursor or a non-positive limit.
        """
        if limit is None or int(limit) <= 0:
            raise ValueError("invalid_arguments")
        limit = int(limit)
        lo = _as_iso(since)
        hi = _as_iso(until)
        after: tuple[str, str] | None = None
        if cursor:
            at, sep, eid = cursor.rpartition("|")
            if not sep or not at or not eid:
                raise ValueError("invalid_arguments")
            after = (at, eid)

        matching = [
            e for e in self.events
            if e.get("space_id") == space_id
            and (lo is None or str(e.get("at", "")) >= lo)
            and (hi is None or str(e.get("at", "")) <= hi)
        ]
        matching.sort(key=lambda e: (str(e.get("at", "")), str(e.get("id", ""))), reverse=True)
        if after is not None:
            matching = [e for e in matching if (str(e.get("at", "")), str(e.get("id", ""))) < after]
        page = matching[:limit]
        next_cursor = None
        if len(matching) > limit:
            last = page[-1]
            next_cursor = f"{last.get('at', '')}|{last.get('id', '')}"
        return [dict(e) for e in page], next_cursor
