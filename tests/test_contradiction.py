"""Tests for contradiction detection."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from papez.engine.contradiction import detect_contradictions
from papez.models.enums import MemoryStatus
from papez.models.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {
        "content_summary": "test",
        "content_full": "test content",
        "embedding": [0.1] * 1536,
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestContradiction:
    @pytest.mark.asyncio
    async def test_contradiction_detected(self):
        """Two contradicting memories should create a CONTRADICTS edge."""
        new_node = _make_node(content_full="I work at Meta")
        existing = _make_node(content_full="I work at Google")

        graph = AsyncMock()
        # Storage returns cosine SIMILARITY (higher = more similar)
        graph.vector_search = AsyncMock(return_value=[(existing, 0.95)])
        graph.create_edge = AsyncMock()
        graph.update_node = AsyncMock()

        emb = AsyncMock()
        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(True, 0.95, "Direct contradiction about employer"))

        result = await detect_contradictions(new_node, graph, emb, llm)
        assert len(result) == 1
        assert result[0][1] == 0.95
        # Verify reason is stored on the edge
        edge_arg = graph.create_edge.call_args[0][0]
        assert edge_arg.reason == "Direct contradiction about employer"
        assert edge_arg.created_by == "llm_contradiction"

    @pytest.mark.asyncio
    async def test_no_contradiction_low_confidence(self):
        """LLM says no contradiction → no edge created."""
        new_node = _make_node(content_full="I like Python")
        existing = _make_node(content_full="I like JavaScript")

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(existing, 0.9)])
        graph.create_edge = AsyncMock()

        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(False, 0.2, None))

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 0
        graph.create_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_contradiction_supersedes_core(self):
        """Contradicting a core memory should trigger supersession."""
        new_node = _make_node(content_full="I work at Meta")
        core_node = _make_node(
            content_full="I work at Google",
            status=MemoryStatus.CORE,
        )

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(core_node, 0.95)])
        graph.create_edge = AsyncMock()
        graph.update_node = AsyncMock()

        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(True, 0.9, "Employment changed"))

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 1

        # Should create both CONTRADICTS and SUPERSEDES edges
        assert graph.create_edge.call_count == 2
        # Core node should be demoted to episodic
        graph.update_node.assert_called_once()
        update_args = graph.update_node.call_args[0]
        assert update_args[1]["status"] == MemoryStatus.EPISODIC

    @pytest.mark.asyncio
    async def test_no_contradiction_low_similarity(self):
        """Memories with low similarity shouldn't even be checked by LLM."""
        new_node = _make_node(content_full="I like cooking")
        unrelated = _make_node(content_full="The weather is nice")

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(unrelated, 0.5)])  # 0.5 similarity, below the 0.85 floor
        graph.create_edge = AsyncMock()

        llm = AsyncMock()

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert len(result) == 0
        llm.detect_contradiction.assert_not_called()

    @pytest.mark.asyncio
    async def test_near_duplicate_reaches_llm_and_distant_pair_does_not(self):
        """Similarity semantics: 0.97 is checked, 0.10 (a 'distance' reading of 0.90) is not."""
        new_node = _make_node(content_full="Rent is 2000")
        near = _make_node(content_full="Rent is 2500")
        far = _make_node(content_full="The cat is grey")

        graph = AsyncMock()
        graph.vector_search = AsyncMock(return_value=[(near, 0.97), (far, 0.10)])
        graph.create_edge = AsyncMock()

        llm = AsyncMock()
        llm.detect_contradiction = AsyncMock(return_value=(False, 0.1, None))

        await detect_contradictions(new_node, graph, AsyncMock(), llm)
        assert llm.detect_contradiction.call_count == 1
        assert llm.detect_contradiction.call_args[0][1] == "Rent is 2500"

    @pytest.mark.asyncio
    async def test_llm_not_allowed_skips_everything(self):
        new_node = _make_node(content_full="I work at Meta")
        graph = AsyncMock()
        llm = AsyncMock()

        result = await detect_contradictions(new_node, graph, AsyncMock(), llm, llm_allowed=False)
        assert result == []
        graph.vector_search.assert_not_called()
        llm.detect_contradiction.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_embedding_returns_empty(self):
        """Node without embedding should return empty."""
        new_node = _make_node(embedding=None)
        graph = AsyncMock()

        result = await detect_contradictions(new_node, graph, AsyncMock(), AsyncMock())
        assert result == []
