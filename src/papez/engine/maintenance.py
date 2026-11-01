"""The maintenance pass: rescore, transition, then prune.

This is the loop that makes forgetting real. ``sweep_for_forgetting`` only
decides; something has to call it on a cadence. The hosted service runs the
same sequence every ten minutes per user; the stdio server runs
``maintenance_loop`` so a local install behaves the same way.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from papez.context import (
    current_include_personal,
    current_org_ids,
    current_principal_type,
    current_scope_space_ids,
    current_user_id,
)
from papez.engine import config
from papez.engine.forgetting import sweep_for_forgetting
from papez.engine.scoring import calculate_decay_score
from papez.models.enums import ForgettingPolicy, MemoryStatus
from papez.storage.base import EmbeddingProvider, GraphStorageProvider, SpaceRegistry
from papez.storage.memory import SPACE_BUCKET_PREFIX, space_bucket_key

logger = logging.getLogger("papez.maintenance")


async def recalculate_decay(graph: GraphStorageProvider, embeddings: EmbeddingProvider | None) -> int:
    """Recompute ``decay_score`` for every non-core node. Returns the count updated."""
    stats = await graph.get_stats()
    max_cw = stats.get("max_causal_weight", 1) or 1
    updated = 0
    for status in (MemoryStatus.ACTIVE, MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC):
        for node in await graph.get_nodes_by_status(status, limit=500):
            if node.status == MemoryStatus.CORE:
                continue
            score = await calculate_decay_score(node, None, None, graph, embeddings, max_cw)  # type: ignore[arg-type]
            await graph.update_node(str(node.id), {"decay_score": score})
            updated += 1
    return updated


@contextmanager
def _context(**values: Any) -> Iterator[None]:
    """Set the named papez.context variables for a block and always reset them."""
    variables: dict[str, contextvars.ContextVar[Any]] = {
        "user_id": current_user_id,
        "principal_type": current_principal_type,
        "org_ids": current_org_ids,
        "scope_space_ids": current_scope_space_ids,
        "include_personal": current_include_personal,
    }
    tokens = [(variables[k], variables[k].set(v)) for k, v in values.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


async def _pass(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider | None,
    llm: Any | None,
    *,
    recalculate: bool,
    prune: bool,
    spaces: SpaceRegistry | None,
) -> dict[str, Any]:
    """Rescore, transition, prune over whatever the current context scopes."""
    result: dict[str, Any] = {"rescored": 0, "transitions": 0, "pruned": []}
    if recalculate:
        result["rescored"] = await recalculate_decay(graph, embeddings)
    # Transitions never call the LLM; the gate is kept for behavioural stability (D11).
    if llm is not None and embeddings is not None:
        from papez.engine.transitions import evaluate_transitions

        result["transitions"] = len(await evaluate_transitions(graph, embeddings, llm))
    if prune:
        result["pruned"] = await sweep_for_forgetting(graph, spaces)
    return result


async def _space_ids_to_maintain(graph: GraphStorageProvider) -> list[str]:
    if not hasattr(graph, "space_bucket_ids"):
        return []
    ids = await graph.space_bucket_ids()
    return [i[len(SPACE_BUCKET_PREFIX):] if i.startswith(SPACE_BUCKET_PREFIX) else i for i in ids]


async def run_maintenance(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider | None,
    llm: Any | None = None,
    *,
    recalculate: bool = True,
    spaces: SpaceRegistry | None = None,
) -> dict[str, Any]:
    """One pass: rescore (optional), evaluate status transitions (only with an
    LLM provider, as transitions need it), then prune. Returns what happened.

    Personal pass (D11): runs for the caller's ``current_user_id`` with
    ``org_ids=[]``, no spaces and personal memory in scope, so it never
    rescores other users' org nodes or any space node. Skipped when no user
    is set (a spaces-only call).

    Space passes: when ``spaces`` is given, each non-empty space bucket whose
    space exists in the registry (archived included, erased excluded) gets
    its own pass as principal ``space::<id>`` with only that space in scope.
    Pruning runs only for ``forgetting_policy == standard`` spaces. Every
    context variable is restored afterwards.

    Returns ``{"rescored", "transitions", "pruned", "spaces"}``; the first
    three describe the personal pass, ``spaces`` maps space id to the same
    three keys for that space.
    """
    result: dict[str, Any] = {"rescored": 0, "transitions": 0, "pruned": [], "spaces": {}}
    uid = current_user_id.get(None)
    if uid is not None and not uid.startswith(SPACE_BUCKET_PREFIX):
        with _context(org_ids=[], scope_space_ids=[], include_personal=True):
            result.update(await _pass(
                graph, embeddings, llm, recalculate=recalculate, prune=True, spaces=spaces,
            ))

    if spaces is None:
        return result
    for space_id in await _space_ids_to_maintain(graph):
        space = await spaces.get_space(space_id)
        if space is None:
            continue  # erased or unknown: never touched
        try:
            with _context(
                user_id=space_bucket_key(space_id),
                principal_type="user",
                org_ids=[],
                scope_space_ids=[space_id],
                include_personal=False,
            ):
                result["spaces"][space_id] = await _pass(
                    graph, embeddings, llm,
                    recalculate=recalculate,
                    prune=space.forgetting_policy == ForgettingPolicy.STANDARD,
                    spaces=spaces,
                )
        except Exception:
            logger.exception("maintenance pass failed for space %s", space_id)
            result["spaces"][space_id] = {"rescored": 0, "transitions": 0, "pruned": [], "error": True}
    return result


async def maintenance_loop(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider | None,
    llm: Any | None = None,
    interval_s: float | None = None,
    spaces: SpaceRegistry | None = None,
) -> None:
    """Run ``run_maintenance`` forever on a cadence. Errors are logged, never fatal."""
    interval = config.MAINTENANCE_INTERVAL_S if interval_s is None else interval_s
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            r = await run_maintenance(graph, embeddings, llm, spaces=spaces)
            if r["pruned"]:
                logger.info("maintenance: rescored %d, transitions %d, pruned %d", r["rescored"], r["transitions"], len(r["pruned"]))
            for sid, sr in r["spaces"].items():
                if sr.get("pruned"):
                    logger.info("maintenance: space %s rescored %d, transitions %d, pruned %d", sid, sr["rescored"], sr["transitions"], len(sr["pruned"]))
        except Exception:
            logger.exception("maintenance pass failed")
