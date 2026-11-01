"""InMemoryAuditSink (spec 8.2, 8.3; D18)."""
from __future__ import annotations

import re

import pytest

from papez.storage.audit import InMemoryAuditSink, audit_timestamp
from papez.storage.base import AuditSink


def _ev(i: int, space="spc_a", at=None) -> dict:
    return {
        "id": f"evt_{i:04d}",
        "at": at or f"2026-11-01T18:00:{i:02d}.000Z",
        "space_id": space,
        "principal_id": "u_1",
        "principal_type": "user",
        "action": "memory.recall",
        "node_ids": [],
        "detail": {"query_chars": 3, "results": 0},
        "client": {"transport": "stdio", "client_name": None},
    }


async def test_record_fills_id_and_at():
    sink: AuditSink = InMemoryAuditSink()
    await sink.record({"space_id": "spc_a", "action": "space.create"})
    ev = sink.events[0]
    assert ev["id"].startswith("evt_") and len(ev["id"]) == 36
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", ev["at"])


def test_audit_timestamp_format():
    from datetime import UTC, datetime
    assert audit_timestamp(datetime(2026, 11, 1, 18, 4, 55, 120999, tzinfo=UTC)) == "2026-11-01T18:04:55.120Z"


async def test_record_does_not_alias_caller_dict():
    sink = InMemoryAuditSink()
    ev = _ev(1)
    await sink.record(ev)
    ev["action"] = "changed"
    assert sink.events[0]["action"] == "memory.recall"


async def test_cap_drops_oldest():
    sink = InMemoryAuditSink(max_events=3)
    for i in range(5):
        await sink.record(_ev(i))
    assert [e["id"] for e in sink.events] == ["evt_0002", "evt_0003", "evt_0004"]


async def test_default_cap_from_config(monkeypatch):
    from papez.engine import config
    monkeypatch.setattr(config, "AUDIT_MAX_EVENTS", 7)
    assert InMemoryAuditSink().max_events == 7


async def test_zero_cap_records_nothing():
    sink = InMemoryAuditSink(max_events=0)
    await sink.record(_ev(1))
    assert sink.events == []


async def test_query_filters_space_and_orders_newest_first():
    sink = InMemoryAuditSink()
    for i in range(4):
        await sink.record(_ev(i))
    await sink.record(_ev(9, space="spc_b"))
    events, cursor = sink.query("spc_a")
    assert [e["id"] for e in events] == ["evt_0003", "evt_0002", "evt_0001", "evt_0000"]
    assert cursor is None
    assert [e["id"] for e in sink.query("spc_b")[0]] == ["evt_0009"]
    assert sink.query("spc_none") == ([], None)


async def test_query_since_until():
    sink = InMemoryAuditSink()
    for i in range(6):
        await sink.record(_ev(i))
    events, _ = sink.query("spc_a", since="2026-11-01T18:00:02.000Z", until="2026-11-01T18:00:04.000Z")
    assert [e["id"] for e in events] == ["evt_0004", "evt_0003", "evt_0002"]
    from datetime import UTC, datetime
    events, _ = sink.query("spc_a", since=datetime(2026, 11, 1, 18, 0, 5, tzinfo=UTC))
    assert [e["id"] for e in events] == ["evt_0005"]


async def test_cursor_paging_covers_all_once():
    sink = InMemoryAuditSink()
    same_at = "2026-11-01T18:00:00.000Z"
    for i in range(7):
        await sink.record(_ev(i, at=same_at if i < 3 else None))
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page, cursor = sink.query("spc_a", limit=2, cursor=cursor)
        seen.extend(e["id"] for e in page)
        pages += 1
        if cursor is None:
            break
        assert "|" in cursor
    assert pages == 4
    assert len(seen) == 7 and len(set(seen)) == 7
    assert seen[:4] == ["evt_0006", "evt_0005", "evt_0004", "evt_0003"]
    assert seen[4:] == ["evt_0002", "evt_0001", "evt_0000"]  # ties broken by id, descending


async def test_query_rejects_bad_cursor_and_limit():
    sink = InMemoryAuditSink()
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        sink.query("spc_a", cursor="garbage")
    with pytest.raises(ValueError, match="^invalid_arguments$"):
        sink.query("spc_a", limit=0)
