"""In-memory storage providers for running without Docker/Redis/FalkorDB."""
from __future__ import annotations

import asyncio
import json as _json
from collections import defaultdict
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from papez.context import (
    current_include_personal,
    current_org_ids,
    current_principal_type,
    current_scope_space_ids,
    current_user_id,
)
from papez.models.edge import MemoryEdge
from papez.models.enums import CAUSAL_EDGE_TYPES, SUPPORTIVE_EDGE_TYPES, EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.models.ref import Ref


def _uid() -> str:
    uid = current_user_id.get(None)
    if uid is None:
        raise RuntimeError("No user context — current_user_id not set")
    return uid


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer (no deps)."""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("tion") and len(word) > 5:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    """Split text into stemmed tokens, stripping punctuation."""
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if len(w) > 2]


class InMemoryCacheProvider:
    """CacheProvider backed by a plain dict."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data


SPACE_BUCKET_PREFIX = "space::"


def space_bucket_key(space_id: str) -> str:
    """Bucket key under which a space's nodes and edges live (spec 6.4)."""
    return f"{SPACE_BUCKET_PREFIX}{space_id}"


def _is_space_bucket(bucket_key: str) -> bool:
    return bucket_key.startswith(SPACE_BUCKET_PREFIX)


_DIR_OUT = frozenset({"out", "outgoing", "both"})
_DIR_IN = frozenset({"in", "incoming", "both"})


@dataclass(frozen=True)
class _Scope:
    """What one storage call may see (spec 6.5, D01). Built per call by ``_scope``."""

    uid: str
    org_ids: frozenset[str]
    space_ids: frozenset[str]
    include_personal: bool


def is_visible(node: MemoryNode, bucket_key: str, scope: _Scope) -> bool:
    """The one visibility rule (spec 6.5, D01 item 3).

    A space node is visible iff its space is in scope AND it sits in that
    space's bucket (anything else is corrupt data and stays hidden). A
    personal/org node is visible iff personal memory is in scope and the node
    is in the caller's own bucket or is org-visible to one of the caller's orgs.
    """
    if node.space_id is not None:
        return node.space_id in scope.space_ids and bucket_key == space_bucket_key(node.space_id)
    if not scope.include_personal or _is_space_bucket(bucket_key):
        return False
    if bucket_key == scope.uid:
        return True
    return node.visibility == Visibility.ORG and node.org_id is not None and node.org_id in scope.org_ids


def _ref_key(ref: Any) -> tuple[str, str]:
    if isinstance(ref, Ref):
        return ref.type, ref.id
    return str(ref["type"]), str(ref["id"])


def _coerce_refs(refs: Any) -> list[Ref]:
    return [r if isinstance(r, Ref) else Ref.model_validate(r) for r in (refs or [])]


class InMemoryGraphProvider:
    """GraphStorageProvider backed by plain dicts with per-bucket isolation.

    Buckets: ``_user_nodes[bucket][node_id]`` / ``_user_edges[bucket]`` where a
    bucket is a user id (personal and org nodes, as before) or
    ``space::<space_id>`` (space nodes and the edges between them, spec 6.4).
    ``_node_bucket`` / ``_edge_bucket`` map ids to their bucket so every method
    finds a node's bucket in O(1) (D01). Every read goes through ``is_visible``.

    If ``persist_path`` is set, state is saved to / loaded from that JSON file
    so data survives across process restarts (e.g. separate ``claude -p`` calls).

    Performance: edges are indexed per bucket by source_id and target_id for
    O(degree) lookups instead of O(total_edges) scans. Disk writes are batched
    within ``defer_saves()`` contexts.
    """

    def __init__(self, persist_path: str | None = None):
        self._user_nodes: dict[str, dict[str, MemoryNode]] = {}
        self._user_edges: dict[str, list[MemoryEdge]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._node_locks: dict[str, asyncio.Lock] = {}
        # Edge indexes: {bucket: {node_id: [edges]}}
        self._idx_by_source: dict[str, dict[str, list[MemoryEdge]]] = {}
        self._idx_by_target: dict[str, dict[str, list[MemoryEdge]]] = {}
        # Bucket maps (D01): node id -> bucket key, edge id -> bucket key.
        self._node_bucket: dict[str, str] = {}
        self._edge_bucket: dict[str, str] = {}
        # Node indexes, keyed by bucket so a bucket drops cleanly (D01, spec 5.5/9.2).
        self._idx_by_ref: dict[tuple[str, str, str], set[str]] = {}
        self._idx_by_external_id: dict[tuple[str, str], str] = {}
        # Save batching: defer disk writes until outermost defer_saves() exits
        self._save_depth = 0
        self._dirty = False

    @property
    def nodes(self) -> dict[str, MemoryNode]:
        """Return nodes for the current user."""
        uid = _uid()
        if uid not in self._user_nodes:
            self._user_nodes[uid] = {}
        return self._user_nodes[uid]

    @property
    def edges(self) -> list[MemoryEdge]:
        """Return edges for the current user."""
        uid = _uid()
        if uid not in self._user_edges:
            self._user_edges[uid] = []
        return self._user_edges[uid]

    @edges.setter
    def edges(self, value: list[MemoryEdge]) -> None:
        uid = _uid()
        for edge in self._user_edges.get(uid, []):
            self._edge_bucket.pop(str(edge.id), None)
        self._user_edges[uid] = value
        self._idx_by_source.pop(uid, None)
        self._idx_by_target.pop(uid, None)
        for edge in value:
            self._index_edge(uid, edge)

    # -- Edge index maintenance -------------------------------------------------

    def _index_edge(self, bucket: str, edge: MemoryEdge) -> None:
        src, tgt = str(edge.source_id), str(edge.target_id)
        self._idx_by_source.setdefault(bucket, {}).setdefault(src, []).append(edge)
        self._idx_by_target.setdefault(bucket, {}).setdefault(tgt, []).append(edge)
        self._edge_bucket[str(edge.id)] = bucket

    def _unindex_edge(self, bucket: str, edge: MemoryEdge) -> None:
        eid = str(edge.id)
        src, tgt = str(edge.source_id), str(edge.target_id)
        src_list = self._idx_by_source.get(bucket, {}).get(src)
        if src_list is not None:
            self._idx_by_source[bucket][src] = [e for e in src_list if str(e.id) != eid]
        tgt_list = self._idx_by_target.get(bucket, {}).get(tgt)
        if tgt_list is not None:
            self._idx_by_target[bucket][tgt] = [e for e in tgt_list if str(e.id) != eid]
        if self._edge_bucket.get(eid) == bucket:
            del self._edge_bucket[eid]

    # -- Node index maintenance -------------------------------------------------

    def _index_node(self, bucket: str, node: MemoryNode) -> None:
        nid = str(node.id)
        self._node_bucket[nid] = bucket
        for rtype, rid in {_ref_key(r) for r in node.refs}:
            self._idx_by_ref.setdefault((bucket, rtype, rid), set()).add(nid)
        if node.external_id is not None:
            # Last write wins: an amendment's new node takes the pointer (spec 9.2).
            self._idx_by_external_id[(bucket, node.external_id)] = nid

    def _unindex_refs(self, bucket: str, nid: str, keys: set[tuple[str, str]]) -> None:
        for rtype, rid in keys:
            ids = self._idx_by_ref.get((bucket, rtype, rid))
            if ids is not None:
                ids.discard(nid)
                if not ids:
                    del self._idx_by_ref[(bucket, rtype, rid)]

    def _unindex_external_id(self, bucket: str, nid: str, external_id: str | None) -> None:
        if external_id is None or self._idx_by_external_id.get((bucket, external_id)) != nid:
            return
        del self._idx_by_external_id[(bucket, external_id)]
        best = self._best_external_id_holder(bucket, external_id, exclude=nid)
        if best is not None:
            self._idx_by_external_id[(bucket, external_id)] = best

    def _unindex_node(self, bucket: str, node: MemoryNode) -> None:
        nid = str(node.id)
        if self._node_bucket.get(nid) == bucket:
            del self._node_bucket[nid]
        self._unindex_refs(bucket, nid, {_ref_key(r) for r in node.refs})
        self._unindex_external_id(bucket, nid, node.external_id)

    def _is_superseded(self, bucket: str, nid: str) -> bool:
        """A node is superseded when it is the target of a SUPERSEDES edge (new -> old)."""
        return any(e.type == EdgeType.SUPERSEDES for e in self._idx_by_target.get(bucket, {}).get(nid, []))

    def _best_external_id_holder(self, bucket: str, external_id: str, exclude: str | None = None) -> str | None:
        """Newest non-superseded node in ``bucket`` with ``external_id`` (else newest); O(bucket)."""
        holders = [
            n for nid, n in self._user_nodes.get(bucket, {}).items()
            if n.external_id == external_id and nid != exclude
        ]
        if not holders:
            return None
        holders.sort(key=lambda n: (not self._is_superseded(bucket, str(n.id)), n.created_at), reverse=True)
        return str(holders[0].id)

    def _drop_bucket_indexes(self, bucket: str) -> None:
        for key in [k for k in self._idx_by_ref if k[0] == bucket]:
            del self._idx_by_ref[key]
        for xkey in [k for k in self._idx_by_external_id if k[0] == bucket]:
            del self._idx_by_external_id[xkey]
        for nid in self._user_nodes.get(bucket, {}):
            if self._node_bucket.get(nid) == bucket:
                del self._node_bucket[nid]
        for e in self._user_edges.get(bucket, []):
            if self._edge_bucket.get(str(e.id)) == bucket:
                del self._edge_bucket[str(e.id)]
        self._idx_by_source.pop(bucket, None)
        self._idx_by_target.pop(bucket, None)

    def _rebuild_indexes(self) -> None:
        self._idx_by_source.clear()
        self._idx_by_target.clear()
        self._node_bucket.clear()
        self._edge_bucket.clear()
        self._idx_by_ref.clear()
        self._idx_by_external_id.clear()
        for bucket, edge_list in self._user_edges.items():
            for edge in edge_list:
                self._index_edge(bucket, edge)
        for bucket, nodes in self._user_nodes.items():
            ext_ids: set[str] = set()
            for node in nodes.values():
                self._index_node(bucket, node)
                if node.external_id is not None:
                    ext_ids.add(node.external_id)
            for ext in ext_ids:
                best = self._best_external_id_holder(bucket, ext)
                if best is not None:
                    self._idx_by_external_id[(bucket, ext)] = best

    def _edges_touching(self, node_id: str, uid: str | None = None) -> list[MemoryEdge]:
        """All edges from a single bucket touching a node (by source or target)."""
        if uid is None:
            uid = _uid()
        seen: set[str] = set()
        result: list[MemoryEdge] = []
        for e in self._idx_by_source.get(uid, {}).get(node_id, []):
            eid = str(e.id)
            if eid not in seen:
                result.append(e)
                seen.add(eid)
        for e in self._idx_by_target.get(uid, {}).get(node_id, []):
            eid = str(e.id)
            if eid not in seen:
                result.append(e)
                seen.add(eid)
        return result

    def _purge_edges_touching(self, node_ids: set[str], buckets: list[str] | None = None) -> int:
        """Remove every edge (in ``buckets``, default all) touching ``node_ids``; return the count."""
        removed = 0
        for bucket in list(self._user_edges) if buckets is None else buckets:
            doomed: set[str] = set()
            for nid in node_ids:
                for e in self._edges_touching(nid, bucket):
                    doomed.add(str(e.id))
            if not doomed:
                continue
            kept: list[MemoryEdge] = []
            for e in self._user_edges.get(bucket, []):
                if str(e.id) in doomed:
                    self._unindex_edge(bucket, e)
                    removed += 1
                else:
                    kept.append(e)
            self._user_edges[bucket] = kept
        return removed

    # -- Save batching ----------------------------------------------------------

    @contextmanager
    def defer_saves(self) -> Generator[None, None, None]:
        """Batch disk writes. All mutations inside this context share one write."""
        self._save_depth += 1
        try:
            yield
        finally:
            self._save_depth -= 1
            if self._save_depth == 0 and self._dirty:
                self._do_save()
                self._dirty = False

    @asynccontextmanager
    async def atomic(self) -> AsyncIterator[None]:
        """Group writes so they reach disk in one save (D08; wraps ``defer_saves``).

        The in-memory backend does not roll back on error: callers validate
        everything before entering (``memory_store_batch``), so the block only
        performs writes that are already known to be valid.
        """
        with self.defer_saves():
            yield

    def _save(self) -> None:
        if self._save_depth > 0:
            self._dirty = True
            return
        self._do_save()

    def _do_save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for bucket in list(dict.fromkeys([*self._user_nodes, *self._user_edges])):
            nodes = self._user_nodes.get(bucket, {})
            edges = self._user_edges.get(bucket, [])
            if not nodes and not edges:
                continue  # empty buckets (create-on-read, service principals) are not persisted (C53)
            data[bucket] = {
                "nodes": {nid: n.model_dump(mode="json") for nid, n in nodes.items()},
                "edges": [e.model_dump(mode="json") for e in edges],
            }
        self._persist_path.write_text(_json.dumps(data))

    # -- Scope and visibility ---------------------------------------------------

    def _get_org_ids(self, org_ids: list[str] | None = None) -> list[str]:
        if org_ids is not None:
            return org_ids
        return current_org_ids.get([])

    def _get_scope_space_ids(self, space_ids: list[str] | None = None) -> list[str]:
        if space_ids is not None:
            return space_ids
        return current_scope_space_ids.get([])

    def _scope(self, org_ids: list[str] | None = None, space_ids: list[str] | None = None) -> _Scope:
        """Build the per-call scope; fails closed (RuntimeError) without a user."""
        uid = _uid()
        include_personal = bool(current_include_personal.get(True)) and current_principal_type.get("user") != "service"
        return _Scope(
            uid=uid,
            org_ids=frozenset(self._get_org_ids(org_ids)),
            space_ids=frozenset(self._get_scope_space_ids(space_ids)),
            include_personal=include_personal,
        )

    def _locate(self, node_id: str) -> tuple[str, MemoryNode] | None:
        """Find a node's bucket regardless of visibility (internal only)."""
        uid = current_user_id.get(None)
        if uid is not None:
            own = self._user_nodes.get(uid, {}).get(node_id)
            if own is not None:
                return uid, own
        bucket = self._node_bucket.get(node_id)
        if bucket is not None:
            node = self._user_nodes.get(bucket, {}).get(node_id)
            if node is not None:
                return bucket, node
        return None

    def _visible_located(self, node_id: str, scope: _Scope) -> tuple[str, MemoryNode] | None:
        loc = self._locate(node_id)
        if loc is None or not is_visible(loc[1], loc[0], scope):
            return None
        return loc

    def _visible_node(self, node_id: str, scope: _Scope) -> MemoryNode | None:
        loc = self._visible_located(node_id, scope)
        return loc[1] if loc else None

    def _candidate_buckets(self, scope: _Scope) -> list[str]:
        """Buckets that can hold visible nodes: own, scoped spaces, other users only for org scans."""
        buckets: list[str] = []
        if scope.include_personal:
            buckets.append(scope.uid)
            if scope.org_ids:
                buckets.extend(b for b in self._user_nodes if b != scope.uid and not _is_space_bucket(b))
        buckets.extend(space_bucket_key(sid) for sid in sorted(scope.space_ids))
        return list(dict.fromkeys(buckets))

    def _visible_nodes(self, scope: _Scope | None = None) -> dict[str, MemoryNode]:
        """Every node visible under ``scope`` (default: the context scope)."""
        if scope is None:
            scope = self._scope()
        result: dict[str, MemoryNode] = {}
        for bucket in self._candidate_buckets(scope):
            for nid, node in self._user_nodes.get(bucket, {}).items():
                if nid not in result and is_visible(node, bucket, scope):
                    result[nid] = node
        return result

    def _edge_visible(self, edge: MemoryEdge, scope: _Scope) -> bool:
        return (
            self._visible_located(str(edge.source_id), scope) is not None
            and self._visible_located(str(edge.target_id), scope) is not None
        )

    def _visible_edges_for_node(self, node_id: str, scope: _Scope, direction: str = "both") -> list[MemoryEdge]:
        """Edges from any bucket touching ``node_id`` whose BOTH endpoints are visible."""
        if self._visible_located(node_id, scope) is None:
            return []
        want_out = direction in _DIR_OUT
        want_in = direction in _DIR_IN
        result: list[MemoryEdge] = []
        seen: set[str] = set()
        for bucket in list(self._idx_by_source.keys() | self._idx_by_target.keys()):
            candidates: list[MemoryEdge] = []
            if want_out:
                candidates.extend(self._idx_by_source.get(bucket, {}).get(node_id, []))
            if want_in:
                candidates.extend(self._idx_by_target.get(bucket, {}).get(node_id, []))
            for e in candidates:
                eid = str(e.id)
                if eid in seen:
                    continue
                seen.add(eid)
                if self._edge_visible(e, scope):
                    result.append(e)
        return result

    def _all_edges_in(self, scope: _Scope) -> list[MemoryEdge]:
        """Every edge, from any bucket, whose both endpoints are visible."""
        result: list[MemoryEdge] = []
        seen: set[str] = set()
        for edge_list in self._user_edges.values():
            for e in edge_list:
                eid = str(e.id)
                if eid not in seen and self._edge_visible(e, scope):
                    result.append(e)
                    seen.add(eid)
        return result

    def _node_scope_bucket(self, node_id: str) -> str | None:
        """Bucket whose edges count toward a node's connectivity (D04).

        The node's own bucket when it is visible; the caller's bucket for an
        unknown id (legacy behaviour); ``None`` (count nothing) when the node
        exists but is not visible.
        """
        scope = self._scope()
        loc = self._locate(node_id)
        if loc is None:
            return scope.uid
        if not is_visible(loc[1], loc[0], scope):
            return None
        return loc[0]

    # -- Persistence ------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        raw = _json.loads(self._persist_path.read_text())
        # Support both old format (flat) and new format (per-user)
        if "nodes" in raw and "edges" in raw:
            # Old flat format — migrate to default user
            uid = _uid()
            self._user_nodes[uid] = {nid: MemoryNode(**v) for nid, v in raw.get("nodes", {}).items()}
            self._user_edges[uid] = [MemoryEdge(**e) for e in raw.get("edges", [])]
        else:
            # Per-bucket format: user ids and space::<id> buckets. Keys starting
            # with "__" are reserved for a future envelope and skipped (D05).
            for bucket, bucket_data in raw.items():
                if bucket.startswith("__") or not isinstance(bucket_data, dict):
                    continue
                self._user_nodes[bucket] = {
                    nid: MemoryNode(**v) for nid, v in (bucket_data.get("nodes") or {}).items()
                }
                self._user_edges[bucket] = [MemoryEdge(**e) for e in (bucket_data.get("edges") or [])]
        self._rebuild_indexes()

    async def initialize(self, user_id: str) -> None:
        self._load()

    async def destroy(self, user_id: str) -> None:
        doomed = set(self._user_nodes.get(user_id, {}))
        self._drop_bucket_indexes(user_id)
        self._user_nodes.pop(user_id, None)
        self._user_edges.pop(user_id, None)
        if doomed:
            self._purge_edges_touching(doomed)
        self._node_locks = {k: v for k, v in self._node_locks.items() if k not in doomed}
        self._save()

    async def erase_user(self, user_id: str, keep_promoted_nodes: bool = True) -> dict[str, int]:
        import re
        _PII_RE = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|"
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        )

        user_nodes = self._user_nodes.get(user_id, {})
        nodes_deleted = 0
        nodes_anonymized = 0

        # Partition nodes into deletable vs. promoted-org (anonymize)
        promoted_ids: set[str] = set()
        delete_ids: set[str] = set()
        for nid, node in user_nodes.items():
            if keep_promoted_nodes and node.visibility == Visibility.ORG and node.org_id:
                promoted_ids.add(nid)
            else:
                delete_ids.add(nid)

        # Anonymize promoted nodes
        for nid in promoted_ids:
            node = user_nodes[nid]
            node.original_user_id = "erased_user"
            nodes_anonymized += 1

        # Space memories the user wrote stay with the space; only the author is
        # anonymized (spec 6.4, D32). Counted separately so the four pinned
        # counts keep their meaning.
        space_nodes_anonymized = 0
        for bucket, nodes in self._user_nodes.items():
            if not _is_space_bucket(bucket):
                continue
            for node in nodes.values():
                if node.original_user_id == user_id:
                    node.original_user_id = "erased_user"
                    space_nodes_anonymized += 1

        # Scrub PII from edge reasons connected to promoted nodes (space buckets untouched)
        edges_scrubbed = 0
        for bucket, edge_list in self._user_edges.items():
            if _is_space_bucket(bucket):
                continue
            for e in edge_list:
                src, tgt = str(e.source_id), str(e.target_id)
                if (src in promoted_ids or tgt in promoted_ids) and e.reason and _PII_RE.search(e.reason):
                    e.reason = _PII_RE.sub("[erased]", e.reason)
                    edges_scrubbed += 1

        # Count and remove edges for deleted nodes
        edges_deleted = 0
        all_delete_ids = delete_ids
        user_edges = self._user_edges.get(user_id, [])
        # Count edges being removed from user's own list (only for deleted nodes)
        edges_deleted += sum(
            1 for e in user_edges
            if str(e.source_id) in all_delete_ids or str(e.target_id) in all_delete_ids
        )

        # Remove edges in other users' lists that reference deleted nodes (space buckets untouched)
        for other_uid, edge_list in list(self._user_edges.items()):
            if other_uid == user_id or _is_space_bucket(other_uid):
                continue
            before = len(edge_list)
            self._user_edges[other_uid] = [
                e for e in edge_list
                if str(e.source_id) not in all_delete_ids and str(e.target_id) not in all_delete_ids
            ]
            edges_deleted += before - len(self._user_edges[other_uid])

        # Delete the deletable nodes
        for nid in delete_ids:
            del user_nodes[nid]
        nodes_deleted = len(delete_ids)

        # Clean up user's edge list: remove edges referencing deleted nodes
        if user_id in self._user_edges:
            self._user_edges[user_id] = [
                e for e in self._user_edges[user_id]
                if str(e.source_id) not in all_delete_ids and str(e.target_id) not in all_delete_ids
            ]

        # If no promoted nodes remain, clean up user entirely
        if not promoted_ids:
            self._user_nodes.pop(user_id, None)
            self._user_edges.pop(user_id, None)

        self._node_locks = {k: v for k, v in self._node_locks.items() if k not in delete_ids}
        self._rebuild_indexes()
        self._save()
        return {
            "nodes_deleted": nodes_deleted,
            "nodes_anonymized": nodes_anonymized,
            "edges_deleted": edges_deleted,
            "edges_scrubbed": edges_scrubbed,
            "space_nodes_anonymized": space_nodes_anonymized,
        }

    # -- Node CRUD --------------------------------------------------------------

    async def create_node(self, node: MemoryNode) -> str:
        nid = str(node.id)
        if node.space_id is not None:
            bucket = space_bucket_key(node.space_id)
        else:
            bucket = _uid()
            if _is_space_bucket(bucket):
                # A space maintenance pass runs with current_user_id=space::<id>;
                # a personal node there would be invisible to everyone.
                raise ValueError("a node without space_id cannot be stored in a space bucket")
        previous = self._locate(nid)
        if previous is not None and previous[0] == bucket:
            self._unindex_node(bucket, previous[1])
        self._user_nodes.setdefault(bucket, {})[nid] = node
        self._index_node(bucket, node)
        self._save()
        return nid

    async def get_node(self, node_id: str) -> MemoryNode | None:
        return self._visible_node(node_id, self._scope())

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> None:
        loc = self._visible_located(node_id, self._scope())
        if loc is None:
            return  # invisible or missing: silent no-op (callers pre-check with get_node)
        bucket, node = loc
        if "space_id" in updates and updates["space_id"] != node.space_id:
            raise ValueError("space_id is immutable; copy the memory instead (promote_to_space)")
        if "refs" in updates:
            updates = {**updates, "refs": _coerce_refs(updates["refs"])}
        old_refs = {_ref_key(r) for r in node.refs}
        old_ext = node.external_id
        for k, v in updates.items():
            setattr(node, k, v)
        new_refs = {_ref_key(r) for r in node.refs}
        if old_refs != new_refs:
            self._unindex_refs(bucket, node_id, old_refs - new_refs)
            for rtype, rid in new_refs - old_refs:
                self._idx_by_ref.setdefault((bucket, rtype, rid), set()).add(node_id)
        if node.external_id != old_ext:
            self._unindex_external_id(bucket, node_id, old_ext)
            if node.external_id is not None:
                self._idx_by_external_id[(bucket, node.external_id)] = node_id
        self._save()

    async def delete_node(self, node_id: str) -> None:
        scope = self._scope()
        loc = self._locate(node_id)
        if loc is None:
            # Unknown id: legacy behaviour, drop the caller's edges touching it.
            self._purge_edges_touching({node_id}, [scope.uid])
            self._save()
            return
        bucket, node = loc
        if not is_visible(node, bucket, scope):
            return
        self._unindex_node(bucket, node)
        del self._user_nodes[bucket][node_id]
        # Dangling-edge cleanup in every bucket (D04).
        self._purge_edges_touching({node_id})
        self._node_locks.pop(node_id, None)
        self._save()

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        visible = self._visible_nodes()
        return [n for n in visible.values() if n.status == status][:limit]

    # -- Edge CRUD --------------------------------------------------------------

    def _placement_bucket(self, source_id: str, target_id: str) -> str:
        """Bucket an edge between these endpoints belongs in; raises ``cross_space_edge`` (D04)."""
        src = self._locate(source_id)
        tgt = self._locate(target_id)
        src_space = src[1].space_id if src else None
        tgt_space = tgt[1].space_id if tgt else None
        if src_space is None and tgt_space is None:
            return _uid()  # both personal/org or missing: caller's bucket, as before
        if src is None or tgt is None or src_space != tgt_space:
            raise ValueError("cross_space_edge")
        assert src_space is not None
        return space_bucket_key(src_space)

    async def create_edge(self, edge: MemoryEdge) -> str:
        bucket = self._placement_bucket(str(edge.source_id), str(edge.target_id))
        self._user_edges.setdefault(bucket, []).append(edge)
        self._index_edge(bucket, edge)
        self._save()
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        """Edges touching ``node_id`` with both ends visible.

        ``direction`` accepts ``"out"``/``"outgoing"``, ``"in"``/``"incoming"`` and ``"both"``.
        """
        result = self._visible_edges_for_node(node_id, self._scope(), direction)
        if edge_type is not None:
            result = [e for e in result if e.type == edge_type]
        return result

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        scope = self._scope()
        if node_ids is None:
            return self._all_edges_in(scope)
        seen: set[str] = set()
        result: list[MemoryEdge] = []
        for nid in dict.fromkeys(node_ids):
            for e in self._visible_edges_for_node(nid, scope):
                eid = str(e.id)
                if eid not in seen:
                    result.append(e)
                    seen.add(eid)
        return result

    def _writable_edge(self, edge_id: str) -> tuple[str, MemoryEdge] | None:
        """Locate an edge for a write: own-bucket edges always, others only when both ends are visible."""
        scope = self._scope()
        for bucket in dict.fromkeys([b for b in (self._edge_bucket.get(edge_id), scope.uid) if b is not None]):
            for e in self._user_edges.get(bucket, []):
                if str(e.id) == edge_id:
                    if bucket == scope.uid or self._edge_visible(e, scope):
                        return bucket, e
                    return None
        return None

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        found = self._writable_edge(edge_id)
        if found is not None:
            found[1].weight = weight
            self._save()

    async def validate_edge(self, edge_id: str) -> None:
        found = self._writable_edge(edge_id)
        if found is not None:
            found[1].last_validated_at = datetime.now(UTC)
            self._save()

    async def delete_edge(self, edge_id: str) -> None:
        found = self._writable_edge(edge_id)
        if found is None:
            return
        bucket, edge = found
        self._unindex_edge(bucket, edge)
        self._user_edges[bucket] = [e for e in self._user_edges.get(bucket, []) if str(e.id) != edge_id]
        self._save()

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        try:
            bucket = self._placement_bucket(source_id, target_id)
        except ValueError:
            return False
        if _is_space_bucket(bucket) and self._visible_located(source_id, self._scope()) is None:
            return False
        for e in self._idx_by_source.get(bucket, {}).get(source_id, []):
            if str(e.target_id) == target_id and e.type == edge_type:
                return True
        return False

    # -- Graph queries (indexed) ------------------------------------------------

    async def traverse(
        self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None,
    ) -> list[MemoryNode]:
        scope = self._scope(org_ids, space_ids)
        start = self._visible_node(start_id, scope)
        if start is None:
            return []  # never walk the edges of an invisible start node (C18)
        visited: set[str] = set()
        queue = [(start_id, 0)]
        result: list[MemoryNode] = []
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            node = self._visible_node(nid, scope)
            if node is None:
                continue
            result.append(node)
            if d < depth:
                for e in self._visible_edges_for_node(nid, scope):
                    if edge_types and e.type not in edge_types:
                        continue
                    src, tgt = str(e.source_id), str(e.target_id)
                    next_id = tgt if src == nid else src
                    if next_id not in visited:
                        queue.append((next_id, d + 1))
        return result

    async def get_connecting_edges(
        self, node_ids: list[str], edge_types: list[EdgeType] | None = None, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None,
    ) -> list[MemoryEdge]:
        """Return visible edges of the induced subgraph among ``node_ids``.

        An edge is included only when BOTH of its endpoints are in ``node_ids``
        and — when ``edge_types`` is given — its type is in that set. This is
        the set of *connecting edges* for a traversal result: a deliberate
        superset of the BFS tree, so callers can render/reconstruct paths.

        Visibility is honored: both endpoints must pass ``is_visible`` under the
        scope. The induced-subgraph and edge-type filtering live here (storage),
        not in the tool layer, so every backend that implements this Protocol
        behaves the same.
        """
        scope = self._scope(org_ids, space_ids)
        id_set = set(node_ids)
        result: list[MemoryEdge] = []
        seen: set[str] = set()
        for nid in id_set:
            for e in self._visible_edges_for_node(nid, scope):
                eid = str(e.id)
                if eid in seen:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                if src not in id_set or tgt not in id_set:
                    continue
                if edge_types and e.type not in edge_types:
                    continue
                result.append(e)
                seen.add(eid)
        return result

    def _causal_bfs(self, seed_id: str, direction: str, scope: _Scope) -> list[MemoryNode]:
        result: list[MemoryNode] = []
        if self._visible_node(seed_id, scope) is None:
            return result
        visited: set[str] = set()
        queue = [seed_id]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            for e in self._visible_edges_for_node(nid, scope):
                if e.type not in CAUSAL_EDGE_TYPES:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                next_id = None
                if direction == "upstream" and tgt == nid and src not in visited:
                    next_id = src
                elif direction == "downstream" and src == nid and tgt not in visited:
                    next_id = tgt
                if next_id:
                    node = self._visible_node(next_id, scope)
                    if node is not None:
                        queue.append(next_id)
                        result.append(node)
        return result

    async def get_causal_chain(
        self, node_id: str, direction: str, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None,
    ) -> list[MemoryNode]:
        return self._causal_bfs(node_id, direction, self._scope(org_ids, space_ids))

    async def get_causal_chains_batch(
        self, node_ids: list[str], direction: str, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None,
    ) -> dict[str, list[MemoryNode]]:
        """BFS from every seed under one scope; an invisible seed gets ``[]``."""
        scope = self._scope(org_ids, space_ids)
        return {nid: self._causal_bfs(nid, direction, scope) for nid in node_ids}

    def _bucket_edges_of(self, node_id: str) -> list[MemoryEdge]:
        bucket = self._node_scope_bucket(node_id)
        return [] if bucket is None else self._edges_touching(node_id, bucket)

    async def get_causal_weight(self, node_id: str) -> int:
        return sum(1 for e in self._bucket_edges_of(node_id) if e.type in CAUSAL_EDGE_TYPES)

    async def get_degree(self, node_id: str) -> int:
        return len(self._bucket_edges_of(node_id))

    async def get_supportive_degree(self, node_id: str) -> int:
        return sum(1 for e in self._bucket_edges_of(node_id) if e.type in SUPPORTIVE_EDGE_TYPES)

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_supportive_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        """Visible nodes with zero supportive edges in their own bucket (D03, D04)."""
        scope = self._scope()
        supported: dict[str, set[str]] = {}
        result: list[MemoryNode] = []
        for nid, node in self._visible_nodes(scope).items():
            bucket = self._node_bucket.get(nid, scope.uid)
            if bucket not in supported:
                ids: set[str] = set()
                for e in self._user_edges.get(bucket, []):
                    if e.type in SUPPORTIVE_EDGE_TYPES:
                        ids.add(str(e.source_id))
                        ids.add(str(e.target_id))
                supported[bucket] = ids
            if nid not in supported[bucket]:
                result.append(node)
        return result

    # -- Search -----------------------------------------------------------------

    @staticmethod
    def _passes_search_filters(node: MemoryNode, tags_any: list[str] | None, ref_type: str | None) -> bool:
        """``tags_any``/``ref_type`` predicates, applied BEFORE scoring and the top-k cut.

        An empty or ``None`` filter means "no filter". Tags must already be
        normalized by the caller (``normalize_tags``).
        """
        if tags_any and not set(tags_any).intersection(node.tags):
            return False
        return not ref_type or any(_ref_key(r)[0] == ref_type for r in node.refs)

    async def vector_search(
        self, embedding: list[float], k: int = 10, status_filter: list[MemoryStatus] | None = None, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None, tags_any: list[str] | None = None, ref_type: str | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        import numpy as np

        visible = self._visible_nodes(self._scope(org_ids, space_ids))
        nodes_with_emb: list[MemoryNode] = []
        for node in visible.values():
            if status_filter and node.status not in status_filter:
                continue
            if not self._passes_search_filters(node, tags_any, ref_type):
                continue
            if node.embedding:
                nodes_with_emb.append(node)
        if not nodes_with_emb or k <= 0:
            return []

        query_vec = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matrix = np.array([n.embedding for n in nodes_with_emb], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        sims = (matrix @ query_vec) / (norms * query_norm)

        top_k = min(k, len(sims))
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(nodes_with_emb[i], float(sims[i])) for i in top_indices]

    async def keyword_search(
        self, query: str, entity_refs: list[str] | None = None, k: int = 10, org_ids: list[str] | None = None,
        space_ids: list[str] | None = None, tags_any: list[str] | None = None, ref_type: str | None = None,
    ) -> list[MemoryNode]:
        visible = self._visible_nodes(self._scope(org_ids, space_ids))
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scored: list[tuple[MemoryNode, float]] = []
        for node in visible.values():
            if not self._passes_search_filters(node, tags_any, ref_type):
                continue
            content = (node.content_full or node.content_summary).lower()
            content_tokens = set(content.split())
            hits = sum(1 for t in query_terms if any(t in tok for tok in content_tokens))
            entity_hits = 0
            if node.entity_refs:
                node_entities_lower = [e.lower() for e in node.entity_refs]
                entity_hits = sum(
                    1 for t in query_terms if any(t in ent for ent in node_entities_lower)
                )
            total = hits + entity_hits
            if total > 0:
                scored.append((node, total / len(query_terms)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:k]]

    # -- References, idempotent writes and spaces --------------------------------

    async def find_nodes_by_ref(
        self, ref_type: str, ref_id: str, org_ids: list[str] | None = None, space_ids: list[str] | None = None,
    ) -> list[MemoryNode]:
        """Visible nodes carrying a ref with exactly ``(ref_type, ref_id)``, oldest first."""
        scope = self._scope(org_ids, space_ids)
        found: dict[str, MemoryNode] = {}
        for bucket in self._candidate_buckets(scope):
            for nid in self._idx_by_ref.get((bucket, ref_type, ref_id), set()):
                node = self._user_nodes.get(bucket, {}).get(nid)
                if node is not None and nid not in found and is_visible(node, bucket, scope):
                    found[nid] = node
        return sorted(found.values(), key=lambda n: (n.created_at, str(n.id)))

    async def find_node_by_external_id(self, external_id: str, space_id: str | None) -> MemoryNode | None:
        """Current (non-superseded) visible node with ``external_id`` in the space or caller's bucket."""
        scope = self._scope()
        bucket = space_bucket_key(space_id) if space_id is not None else scope.uid
        nid = self._idx_by_external_id.get((bucket, external_id))
        node = self._user_nodes.get(bucket, {}).get(nid) if nid is not None else None
        if node is None or node.external_id != external_id or self._is_superseded(bucket, str(node.id)):
            # Stale pointer (direct mutation, supersession after indexing): recompute.
            best = self._best_external_id_holder(bucket, external_id)
            if best is None:
                self._idx_by_external_id.pop((bucket, external_id), None)
                return None
            self._idx_by_external_id[(bucket, external_id)] = best
            node = self._user_nodes[bucket][best]
            if self._is_superseded(bucket, best):
                return None
        if not is_visible(node, bucket, scope):
            return None
        return node

    async def space_bucket_ids(self) -> list[str]:
        """Space ids that currently have a non-empty bucket (for maintenance passes)."""
        keys = dict.fromkeys([*self._user_nodes, *self._user_edges])
        return sorted(
            k[len(SPACE_BUCKET_PREFIX):] for k in keys
            if _is_space_bucket(k) and (self._user_nodes.get(k) or self._user_edges.get(k))
        )

    async def erase_space(self, space_id: str, dry_run: bool = False) -> dict[str, int]:
        """Delete a space's bucket, its indexes and any edge elsewhere touching its nodes."""
        bucket = space_bucket_key(space_id)
        node_ids = set(self._user_nodes.get(bucket, {}))
        own_edges = len(self._user_edges.get(bucket, []))
        foreign_edges = 0
        for other in self._user_edges:
            if other == bucket:
                continue
            seen: set[str] = set()
            for nid in node_ids:
                seen.update(str(e.id) for e in self._edges_touching(nid, other))
            foreign_edges += len(seen)
        counts = {"nodes": len(node_ids), "edges": own_edges + foreign_edges}
        if dry_run:
            return counts
        self._drop_bucket_indexes(bucket)
        self._user_nodes.pop(bucket, None)
        self._user_edges.pop(bucket, None)
        if node_ids:
            self._purge_edges_touching(node_ids)
        self._node_locks = {k: v for k, v in self._node_locks.items() if k not in node_ids}
        self._save()
        return counts

    # -- Misc -------------------------------------------------------------------

    async def store_embedding(self, node_id: str, embedding: list[float]) -> None:
        node = self._visible_node(node_id, self._scope())
        if node:
            node.embedding = embedding
            self._save()

    async def promote_to_org(self, node_id: str, org_id: str) -> None:
        loc = self._visible_located(node_id, self._scope())
        if loc is None:
            return
        bucket, node = loc
        if node.space_id is not None:
            # Space nodes are never org-visible (spec 6.2); the tool returns invalid_arguments first.
            raise ValueError("invalid_arguments")
        node.visibility = Visibility.ORG
        node.org_id = org_id
        node.original_user_id = bucket
        self._save()

    def _get_node_lock(self, node_id: str) -> asyncio.Lock:
        if node_id not in self._node_locks:
            self._node_locks[node_id] = asyncio.Lock()
        return self._node_locks[node_id]

    async def atomic_reactivation_update(
        self, node_id: str, timestamp: datetime, stability_delta: float,
    ) -> None:
        lock = self._get_node_lock(node_id)
        async with lock:
            node = self._visible_node(node_id, self._scope())
            if not node:
                return
            node.reactivation_count += 1
            node.reactivation_timestamps.append(timestamp)
            node.stability += stability_delta
            node.last_reactivated_at = timestamp
            self._save()

    async def get_stats(self) -> dict[str, Any]:
        """Counts over the visible set (D03); both key spellings for backend parity."""
        scope = self._scope()
        visible = self._visible_nodes(scope)
        edges = self._all_edges_in(scope)
        causal_counts: dict[str, int] = defaultdict(int)
        for e in edges:
            if e.type in CAUSAL_EDGE_TYPES:
                causal_counts[str(e.source_id)] += 1
                causal_counts[str(e.target_id)] += 1
        max_cw = max(causal_counts.values()) if causal_counts else 0
        return {
            "nodes": len(visible),
            "edges": len(edges),
            "max_causal_weight": max_cw,
            "total_nodes": len(visible),
            "total_edges": len(edges),
        }


class InMemoryEventBusProvider:
    """EventBusProvider that dispatches handlers as background tasks."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Any]] = defaultdict(list)

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        for handler in self._subscribers.get(channel, []):
            asyncio.create_task(self._safe_call(handler, payload))

    async def subscribe(self, channel: str, handler: Any) -> None:
        self._subscribers[channel].append(handler)

    @staticmethod
    async def _safe_call(handler: Any, payload: dict[str, Any]) -> None:
        import logging
        try:
            await handler(payload)
        except Exception:
            logging.getLogger(__name__).exception("Event handler failed")
