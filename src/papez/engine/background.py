"""Background handlers for LLM-powered memory processing.

All heavy operations (entity extraction, causal inference, contradiction
detection, consolidation) run here as fire-and-forget tasks dispatched
via the event bus.  The synchronous hot path in tools.py stays fast.

Scope (spec 6.9, D10): a handler never inherits the storing request's read
scope. ``on_memory_created`` rebuilds identity from the payload and narrows
storage to exactly one bucket family: the node's space, or personal memory.
One gate decides whether any LLM call may happen for the node: personal
nodes always may, space nodes only when the registry says the space allows
it (no registry means no LLM for space content, fail closed).
"""
from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from papez.context import (
    current_include_personal,
    current_org_ids,
    current_scope_space_ids,
    current_user_id,
)
from papez.models.edge import MemoryEdge
from papez.models.node import MemoryNode
from papez.storage.base import (
    EmbeddingProvider,
    EventBusProvider,
    GraphStorageProvider,
    LLMProvider,
    SpaceRegistry,
)

logger = logging.getLogger(__name__)


@contextmanager
def node_scope(space_id: str | None) -> Iterator[None]:
    """Limit storage reads to one bucket family for the duration of the block.

    ``space_id`` set: only that space (personal memory excluded).
    ``space_id`` None: personal (and org-visible) memory, no spaces.
    Tokens are always reset, so the caller's scope is restored on exit.
    """
    t_spaces = current_scope_space_ids.set([space_id] if space_id else [])
    t_personal = current_include_personal.set(space_id is None)
    try:
        yield
    finally:
        current_include_personal.reset(t_personal)
        current_scope_space_ids.reset(t_spaces)


async def llm_allowed_for(space_id: str | None, spaces: SpaceRegistry | None) -> bool:
    """The one LLM gate (D10): personal nodes yes; space nodes only if the space allows it."""
    if space_id is None:
        return True
    if spaces is None:
        return False
    space = await spaces.get_space(space_id)
    return space is not None and bool(space.llm_inference_enabled)


async def infer_and_link_causal_edges(
    node: MemoryNode,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
) -> list[str]:
    """Ask the LLM for causal links between ``node`` and its nearest neighbours.

    Candidates come only from the node's own scope (its space, or personal
    memory). Every proposed target is re-read with ``get_node`` under that
    scope and must share the node's ``space_id`` before an edge is created;
    a storage ``cross_space_edge`` refusal is logged and skipped. Returns the
    ids of the edges created. The caller is responsible for the LLM gate.
    """
    if not node.embedding:
        return []
    node_id = str(node.id)
    space_id = node.space_id
    created: list[str] = []
    with node_scope(space_id):
        similar = await graph.vector_search(
            node.embedding, k=6, space_ids=[space_id] if space_id else [],
        )
        existing = [
            (str(n.id), n.content_full or n.content_summary)
            for n, _ in similar
            if str(n.id) != node_id and n.space_id == space_id
        ]
        if not existing:
            return []
        content = node.content_full or node.content_summary
        causal_edges = await llm.infer_causal_edges(content, existing)
        for target_id, edge_type, confidence, reason in causal_edges:
            try:
                target_uuid = uuid.UUID(str(target_id))
            except ValueError:
                logger.warning("Causal inference proposed a malformed target id %r for %s", target_id, node_id)
                continue
            target = await graph.get_node(str(target_uuid))
            if target is None or target.space_id != space_id:
                logger.warning("Causal inference target %s is out of scope for %s; skipped", target_id, node_id)
                continue
            if await graph.edge_exists(node_id, str(target_uuid), edge_type):
                continue
            edge = MemoryEdge(
                source_id=node.id,
                target_id=target_uuid,
                type=edge_type,
                weight=round(confidence, 4),
                reason=reason,
                created_by="llm_causal_inference",
            )
            try:
                created.append(await graph.create_edge(edge))
            except ValueError as exc:
                if str(exc) == "cross_space_edge":
                    logger.warning("Skipped cross-space causal edge %s -> %s", node_id, target_id)
                    continue
                raise
            logger.info(
                "Created %s edge %s -> %s (%.2f)",
                edge_type.value, node_id, target_id, confidence,
            )
    return created


async def on_memory_created(
    payload: dict[str, Any],
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
    spaces: SpaceRegistry | None = None,
) -> None:
    """Run all LLM enrichment after a memory is stored.

    ``payload`` is ``{node_id, content_full, user_id, org_ids, space_id}``;
    the last three are optional for older publishers. Identity is taken from
    the payload when present, and the read scope is always narrowed to the
    node's own bucket family; every context variable is restored on exit.
    """
    node_id = payload["node_id"]
    content = payload.get("content_full", "")
    if not content:
        return

    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    try:
        user_id = payload.get("user_id")
        if user_id:
            tokens.append((current_user_id, current_user_id.set(str(user_id))))
        org_ids = payload.get("org_ids")
        if org_ids is not None:
            tokens.append((current_org_ids, current_org_ids.set([str(o) for o in org_ids])))

        if "space_id" in payload:
            space_id: str | None = payload.get("space_id") or None
        else:
            # Older publisher: learn the node's space under the inherited
            # scope, then narrow before anything else reads or writes.
            probe = await graph.get_node(node_id)
            if probe is None:
                return
            space_id = probe.space_id

        with node_scope(space_id):
            node = await graph.get_node(node_id)
            if node is None or node.space_id != space_id:
                return
            if not await llm_allowed_for(node.space_id, spaces):
                logger.info("LLM inference disabled for space %s; skipped enrichment of %s", node.space_id, node_id)
                return
            await _enrich(node, content, graph, llm, embeddings)
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


async def _enrich(
    node: MemoryNode,
    content: str,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
) -> None:
    """Steps 1-5 of ``on_memory_created``; scope and gate are already set."""
    node_id = str(node.id)
    entities: list[str] = []
    save_ctx = graph.defer_saves() if hasattr(graph, "defer_saves") else nullcontext()
    with save_ctx:
        # 1. Entity extraction
        try:
            entities = await llm.extract_entities(content)
            if entities:
                await graph.update_node(node_id, {"entity_refs": entities})
                logger.info("Extracted %d entities for %s", len(entities), node_id)
        except Exception:
            logger.warning("Entity extraction failed for %s", node_id, exc_info=True)

        # 2. Category classification — only fill when the caller didn't set one,
        #    so an explicit memory_store(category=...) is never overwritten.
        if node.category is None:
            try:
                category = await llm.classify_category(content)
                if category:
                    await graph.update_node(node_id, {"category": category})
                    logger.info("Classified %s as '%s'", node_id, category)
            except Exception:
                logger.warning("Category classification failed for %s", node_id, exc_info=True)

        # 3. Causal edge inference (same scope only)
        try:
            await infer_and_link_causal_edges(node, graph, llm, embeddings)
        except Exception:
            logger.warning("Causal inference failed for %s", node_id, exc_info=True)

        # 4. Contradiction detection (requires an embedder to compare content)
        if embeddings is not None:
            try:
                from papez.engine.contradiction import detect_contradictions

                contradictions = await detect_contradictions(node, graph, embeddings, llm)
                if contradictions:
                    logger.info(
                        "Found %d contradictions for %s", len(contradictions), node_id,
                    )
            except Exception:
                logger.warning("Contradiction detection failed for %s", node_id, exc_info=True)

            # 5. Consolidation check for each extracted entity
            try:
                from papez.engine.consolidation import check_and_consolidate

                for entity in entities:
                    result = await check_and_consolidate(
                        entity, graph, llm, embeddings, space_id=node.space_id,
                    )
                    if result:
                        logger.info(
                            "Consolidated memories for entity '%s' -> %s", entity, result,
                        )
            except Exception:
                logger.warning("Consolidation check failed for %s", node_id, exc_info=True)


def register_handlers(
    event_bus: EventBusProvider,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
    spaces: SpaceRegistry | None = None,
) -> None:
    """Wire background handlers to event bus channels."""

    async def _handle_created(payload: dict[str, Any]) -> None:
        try:
            await on_memory_created(payload, graph, llm, embeddings, spaces)
        except Exception:
            logger.exception("Background handler failed")

    if hasattr(event_bus, "_subscribers"):
        event_bus._subscribers["memory.created"].append(_handle_created)
    else:
        import asyncio
        asyncio.run(event_bus.subscribe("memory.created", _handle_created))
