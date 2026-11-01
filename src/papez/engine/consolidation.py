"""Episodic → semantic consolidation."""
from __future__ import annotations

from datetime import UTC, datetime

from papez.context import current_user_id
from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider


async def check_and_consolidate(
    entity_ref: str,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    space_id: str | None = None,
    llm_allowed: bool = True,
) -> str | None:
    """
    If 3+ episodic memories share entity_ref, consolidate into 1 semantic memory.
    Returns new semantic node ID or None.

    Candidates come only from one scope (spec 6.9): the space ``space_id``,
    or personal memory when it is None. The derived node lands in the same
    scope, carries ``space_id``, and is PRIVATE with no org. With
    ``llm_allowed=False`` nothing is read, sent or written.
    """
    if not llm_allowed:
        return None

    from papez.engine.background import node_scope

    with node_scope(space_id):
        return await _consolidate_in_scope(entity_ref, graph, llm, embeddings, space_id)


async def _consolidate_in_scope(
    entity_ref: str,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    space_id: str | None,
) -> str | None:
    # Find episodic nodes with this entity_ref, in this scope only
    episodic_nodes = await graph.get_nodes_by_status(MemoryStatus.EPISODIC, limit=200)
    matching = [
        n for n in episodic_nodes
        if n.space_id == space_id
        and entity_ref.lower() in [e.lower() for e in n.entity_refs]
    ]

    if len(matching) < 3:
        return None

    # Consolidate
    texts = [n.content_full or n.content_summary for n in matching]
    consolidated_text = await llm.consolidate(texts)
    summary = consolidated_text[:200]

    # Generate embedding
    embedding = await embeddings.embed(consolidated_text)

    now = datetime.now(UTC)
    semantic_node = MemoryNode(
        status=MemoryStatus.SEMANTIC,
        content_summary=summary,
        content_full=consolidated_text,
        embedding=embedding,
        created_at=now,
        last_accessed_at=now,
        last_reactivated_at=now,
        entity_refs=[entity_ref],
        space_id=space_id,
        visibility=Visibility.PRIVATE,
        org_id=None,
        original_user_id=current_user_id.get(None),
        source_agent="consolidation",
    )

    node_id = await graph.create_node(semantic_node)

    # Create DERIVED_FROM edges
    for source_node in matching:
        edge = MemoryEdge(
            source_id=semantic_node.id,
            target_id=source_node.id,
            type=EdgeType.DERIVED_FROM,
            weight=0.7,
        )
        await graph.create_edge(edge)

    return node_id
