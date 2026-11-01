"""Spec 13.3: background work (forgetting, contradiction, consolidation,
causal inference, maintenance) respects Mind Spaces (spec 6.9, D10, D11).

Every test runs against the real InMemoryGraphProvider and
InMemorySpaceRegistry. LLM fakes come from conftest: ForbiddenLLM fails the
test on any call (pytest ``Failed`` is a BaseException, so the engine's
``except Exception`` blocks cannot swallow it); RecordingLLM is the positive
control.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from conftest import ForbiddenLLM, RecordingLLM, principal

from papez.context import (
    current_include_personal,
    current_org_ids,
    current_principal_type,
    current_scope_space_ids,
    current_user_id,
)
from papez.engine.background import infer_and_link_causal_edges, on_memory_created, register_handlers
from papez.engine.consolidation import check_and_consolidate
from papez.engine.contradiction import detect_contradictions
from papez.engine.forgetting import sweep_for_forgetting
from papez.engine.maintenance import run_maintenance
from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, ForgettingPolicy, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.models.space import Space
from papez.storage.memory import InMemoryEventBusProvider, InMemoryGraphProvider, space_bucket_key
from papez.storage.spaces import InMemorySpaceRegistry

ORG = "org_r"
VEC = [1.0] + [0.0] * 15          # every node shares this embedding: similarity 1.0


class ConstEmbedder:
    dimension = 16

    async def embed(self, text: str) -> list[float]:
        return list(VEC)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(VEC) for _ in texts]


class ScriptedLLM(RecordingLLM):
    """RecordingLLM that returns entities, contradictions and causal proposals."""

    def __init__(
        self,
        entities: list[str] | None = None,
        contradiction: tuple[bool, float, str | None] = (False, 0.0, None),
        causal: list[tuple[str, EdgeType, float, str | None]] | None = None,
    ) -> None:
        super().__init__()
        self._entities = entities or []
        self._contradiction = contradiction
        self._causal = causal or []

    async def extract_entities(self, text: str) -> list[str]:
        await super().extract_entities(text)
        return list(self._entities)

    async def detect_contradiction(self, memory_a: str, memory_b: str) -> tuple[bool, float, str | None]:
        await super().detect_contradiction(memory_a, memory_b)
        return self._contradiction

    async def infer_causal_edges(
        self, new_memory: str, existing_memories: list[tuple[str, str]],
    ) -> list[tuple[str, EdgeType, float, str | None]]:
        await super().infer_causal_edges(new_memory, existing_memories)
        return list(self._causal)


def _node(text: str, *, space: str | None = None, idle_days: float = 0, **kw: Any) -> MemoryNode:
    when = datetime.now(UTC) - timedelta(days=idle_days)
    return MemoryNode(
        content_summary=text, content_full=text, embedding=list(VEC),
        space_id=space, created_at=when, last_accessed_at=when, **kw,
    )


async def _space(reg: InMemorySpaceRegistry, slug: str, *, policy: ForgettingPolicy = ForgettingPolicy.RETAIN_ALL,
                 llm: bool = True) -> str:
    sp = await reg.create_space(Space(
        org_id=ORG, name=slug, slug=slug, created_by="alice",
        forgetting_policy=policy, llm_inference_enabled=llm,
    ))
    return sp.id


@pytest.fixture()
async def world() -> dict[str, Any]:
    graph = InMemoryGraphProvider()
    reg = InMemorySpaceRegistry()
    return {
        "graph": graph,
        "reg": reg,
        "retain": await _space(reg, "retain-space"),
        "standard": await _space(reg, "standard-space", policy=ForgettingPolicy.STANDARD),
        "nollm": await _space(reg, "no-llm-space", policy=ForgettingPolicy.STANDARD, llm=False),
    }


async def _store(graph: InMemoryGraphProvider, node: MemoryNode) -> MemoryNode:
    await graph.create_node(node)
    return node


def _payload(node: MemoryNode, uid: str = "alice", orgs: list[str] | None = None) -> dict[str, Any]:
    return {
        "node_id": str(node.id), "content_full": node.content_full,
        "user_id": uid, "org_ids": orgs if orgs is not None else [ORG], "space_id": node.space_id,
    }


def _texts_sent(llm: RecordingLLM) -> list[str]:
    """Every string any LLM call received (flattened)."""
    out: list[str] = []
    for _name, args in llm.calls:
        for a in args:
            if isinstance(a, str):
                out.append(a)
            elif isinstance(a, list):
                for item in a:
                    out.extend(item if isinstance(item, tuple) else [item])
    return out


# ---------------------------------------------------------------------------
# Forgetting
# ---------------------------------------------------------------------------


async def test_retain_all_never_pruned(world):
    g, reg, sid = world["graph"], world["reg"], world["retain"]
    orphan = await _store(g, _node("idle orphan in a retain_all space", space=sid, idle_days=90, decay_score=0.0))

    with principal("alice", orgs=[ORG], roles={sid: "writer"}, include_personal=False):
        assert await sweep_for_forgetting(g, reg) == []
        # Without a registry a space node is never pruned either (fail closed).
        assert await sweep_for_forgetting(g) == []

    with principal("alice", orgs=[ORG]):
        result = await run_maintenance(g, ConstEmbedder(), spaces=reg)
    assert result["spaces"][sid]["pruned"] == []
    assert result["spaces"][sid]["rescored"] == 1            # it still decays
    with principal("alice", roles={sid: "reader"}, include_personal=False):
        assert await g.get_node(str(orphan.id)) is not None


async def test_standard_policy_prunes_in_space_scope(world):
    g, reg, sid = world["graph"], world["reg"], world["standard"]
    orphan = await _store(g, _node("idle orphan in a standard space", space=sid, idle_days=90))
    a = await _store(g, _node("idle but linked inside the space", space=sid, idle_days=90))
    b = await _store(g, _node("the other end of that link", space=sid, idle_days=90))
    pinned = await _store(g, _node("idle, unlinked, pinned", space=sid, idle_days=90, pinned=True))
    with principal("alice", roles={sid: "writer"}, include_personal=False):
        await g.create_edge(MemoryEdge(source_id=a.id, target_id=b.id, type=EdgeType.CAUSED_BY))

    # Alice's personal node with a personal edge to another personal node: the
    # personal bucket's edges must not count for the space and vice versa.
    with principal("alice", orgs=[ORG]):
        p1 = await _store(g, _node("personal idle orphan", idle_days=90))
        p2 = await _store(g, _node("personal linked a", idle_days=90))
        p3 = await _store(g, _node("personal linked b", idle_days=90))
        await g.create_edge(MemoryEdge(source_id=p2.id, target_id=p3.id, type=EdgeType.SUPPORTS))
        # A cross-space edge is refused by storage, so it cannot rescue the orphan.
        with pytest.raises(ValueError, match="cross_space_edge"):
            await g.create_edge(MemoryEdge(source_id=p1.id, target_id=orphan.id, type=EdgeType.SUPPORTS))

        result = await run_maintenance(g, ConstEmbedder(), spaces=reg)

    assert result["pruned"] == [str(p1.id)]
    assert result["spaces"][sid]["pruned"] == [str(orphan.id)]
    with principal("alice", orgs=[ORG], roles={sid: "reader"}):
        assert await g.get_node(str(orphan.id)) is None
        for kept in (a, b, pinned, p2, p3):
            assert await g.get_node(str(kept.id)) is not None, kept.content_summary


# ---------------------------------------------------------------------------
# LLM gate
# ---------------------------------------------------------------------------


async def _seed_llm_scenario(g: InMemoryGraphProvider, sid: str) -> MemoryNode:
    for i in range(3):
        await _store(g, _node(f"episodic acme memory {i}", space=sid, status=MemoryStatus.EPISODIC, entity_refs=["acme"]))
    return await _store(g, _node("new acme memory", space=sid))


async def test_llm_disabled_space_never_calls_llm(world):
    g, reg, sid = world["graph"], world["reg"], world["nollm"]
    new = await _seed_llm_scenario(g, sid)
    forbidden = ForbiddenLLM()

    with principal("alice", orgs=[ORG], roles={sid: "writer"}):
        # The real path: the memory.created handler with the registry.
        await on_memory_created(_payload(new), g, forbidden, ConstEmbedder(), reg)
        # No registry: space content never reaches an LLM (fail closed).
        await on_memory_created(_payload(new), g, forbidden, ConstEmbedder())
        # Direct callers are gated too.
        with principal("alice", roles={sid: "writer"}, include_personal=False):
            node = await g.get_node(str(new.id))
            assert node is not None
            assert await detect_contradictions(node, g, ConstEmbedder(), forbidden, llm_allowed=False) == []
            assert await check_and_consolidate("acme", g, forbidden, ConstEmbedder(), space_id=sid, llm_allowed=False) is None
        # Maintenance (rescore/transitions/prune) never calls the LLM.
        await run_maintenance(g, ConstEmbedder(), llm=forbidden, spaces=reg)
    assert forbidden.calls == []

    # Positive control: the identical scenario in an LLM-enabled space DOES call
    # every step, so the negative result above is the gate, not a dead path.
    on = world["retain"]
    new_on = await _seed_llm_scenario(g, on)
    rec = ScriptedLLM(entities=["acme"])
    with principal("alice", orgs=[ORG], roles={on: "writer"}):
        await on_memory_created(_payload(new_on), g, rec, ConstEmbedder(), reg)
    assert set(rec.names()) >= {
        "extract_entities", "classify_category", "infer_causal_edges", "detect_contradiction", "consolidate",
    }


# ---------------------------------------------------------------------------
# Contradiction / consolidation / causal inference stay inside one scope
# ---------------------------------------------------------------------------


async def test_contradiction_only_within_space(world):
    g, a, b = world["graph"], world["retain"], world["standard"]
    in_a = await _store(g, _node("A: the rate limit is 100", space=a))
    in_b = await _store(g, _node("B: the rate limit is 500", space=b))
    with principal("alice", orgs=[ORG]):
        personal = await _store(g, _node("personal: the rate limit is 900"))
    new = await _store(g, _node("A: the rate limit is 200", space=a))

    llm = ScriptedLLM(contradiction=(True, 0.95, "different numbers"))
    # Widest possible caller scope: both spaces and personal memory.
    with principal("alice", orgs=[ORG], roles={a: "writer", b: "writer"}):
        found = await detect_contradictions(new, g, ConstEmbedder(), llm)
        assert [cid for cid, _ in found] == [str(in_a.id)]
        sent = _texts_sent(llm)
        assert "B: the rate limit is 500" not in sent
        assert "personal: the rate limit is 900" not in sent
        edges = await g.get_edges(str(new.id), "out", EdgeType.CONTRADICTS)
        assert {str(e.target_id) for e in edges} == {str(in_a.id)}

        # And a personal node never pairs with any space node.
        llm2 = ScriptedLLM(contradiction=(True, 0.95, "x"))
        mine = await _store(g, _node("personal: the rate limit is 50"))
        found2 = await detect_contradictions(mine, g, ConstEmbedder(), llm2)
        assert [cid for cid, _ in found2] == [str(personal.id)]
        assert not any(t.startswith(("A:", "B:")) for t in _texts_sent(llm2))
    assert str(in_b.id) not in {cid for cid, _ in found + found2}


async def test_consolidation_never_crosses_spaces(world):
    g, a, b = world["graph"], world["retain"], world["standard"]

    def ep(text: str, space: str | None) -> MemoryNode:
        return _node(text, space=space, status=MemoryStatus.EPISODIC, entity_refs=["Acme"])

    a_nodes = [await _store(g, ep(f"A acme {i}", a)) for i in range(2)]
    b_nodes = [await _store(g, ep(f"B acme {i}", b)) for i in range(3)]
    with principal("alice", orgs=[ORG]):
        p_nodes = [await _store(g, ep(f"personal acme {i}", None)) for i in range(3)]

    llm = RecordingLLM()
    with principal("alice", orgs=[ORG], roles={a: "writer", b: "writer"}):
        # Only 2 in A: B's and personal matches never count towards A's threshold.
        assert await check_and_consolidate("acme", g, llm, ConstEmbedder(), space_id=a) is None
        assert llm.calls == []

        a_nodes.append(await _store(g, ep("A acme 2", a)))
        derived_id = await check_and_consolidate("acme", g, llm, ConstEmbedder(), space_id=a)
        assert derived_id is not None
        assert sorted(llm.calls[0][1][0]) == sorted(n.content_full for n in a_nodes)

        derived = await g.get_node(derived_id)
        assert derived is not None
        assert derived.space_id == a
        assert derived.visibility == Visibility.PRIVATE and derived.org_id is None
        assert derived.original_user_id == "alice"
        assert derived.source_agent == "consolidation"
        assert derived_id in g._user_nodes[space_bucket_key(a)]
        assert derived_id not in g._user_nodes.get("alice", {})
        targets = {str(e.target_id) for e in await g.get_edges(derived_id, "out", EdgeType.DERIVED_FROM)}
        assert targets == {str(n.id) for n in a_nodes}

        # Personal consolidation never pulls space nodes in.
        personal_id = await check_and_consolidate("acme", g, RecordingLLM(), ConstEmbedder(), space_id=None)
        assert personal_id is not None
        p_derived = await g.get_node(personal_id)
        assert p_derived is not None and p_derived.space_id is None
        p_targets = {str(e.target_id) for e in await g.get_edges(personal_id, "out", EdgeType.DERIVED_FROM)}
        assert p_targets == {str(n.id) for n in p_nodes}
    assert not {str(n.id) for n in b_nodes} & (targets | p_targets)


async def test_causal_inference_skips_out_of_scope_targets(world):
    g, a, b = world["graph"], world["retain"], world["standard"]
    in_a = await _store(g, _node("A: caused it", space=a))
    in_b = await _store(g, _node("B: unrelated", space=b))
    new = await _store(g, _node("A: the effect", space=a))
    llm = ScriptedLLM(causal=[
        (str(in_a.id), EdgeType.CAUSED_BY, 0.9, "in scope"),
        (str(in_b.id), EdgeType.CAUSED_BY, 0.9, "hallucinated cross-space target"),
        ("not-a-uuid", EdgeType.CAUSED_BY, 0.9, "malformed"),
    ])
    with principal("alice", orgs=[ORG], roles={a: "writer", b: "writer"}):
        created = await infer_and_link_causal_edges(new, g, llm, ConstEmbedder())
        assert len(created) == 1
        (_, (_, existing)), = [c for c in llm.calls if c[0] == "infer_causal_edges"]
        assert {cid for cid, _ in existing} == {str(in_a.id)}
        edges = await g.get_edges(str(new.id), "out")
        assert {str(e.target_id) for e in edges} == {str(in_a.id)}


async def test_background_narrows_scope(world):
    """A personal store by a space member never searches space nodes."""
    g, a = world["graph"], world["retain"]
    space_node = await _store(g, _node("space: secret client detail", space=a))
    with principal("alice", orgs=[ORG]):
        older = await _store(g, _node("personal: older note"))
        new = await _store(g, _node("personal: new note"))

    llm = ScriptedLLM(contradiction=(True, 0.9, "x"), causal=[(str(space_node.id), EdgeType.CAUSED_BY, 0.9, "leak")])
    # The request scope is wide: member of space A, personal memory in scope.
    with principal("alice", orgs=[ORG], roles={a: "writer"}):
        await on_memory_created(_payload(new), g, llm, ConstEmbedder(), world["reg"])
        # Context is restored after the handler.
        assert current_scope_space_ids.get([]) == [a]
        assert current_include_personal.get(True) is True
        # Legacy payload (no identity / space keys) is narrowed the same way.
        legacy = ScriptedLLM()
        await on_memory_created({"node_id": str(new.id), "content_full": new.content_full}, g, legacy, ConstEmbedder())
        edges = await g.get_edges(str(new.id), "out")

    for rec in (llm, legacy):
        assert "space: secret client detail" not in _texts_sent(rec)
        assert "infer_causal_edges" in rec.names()
    (_, (_, existing)), = [c for c in llm.calls if c[0] == "infer_causal_edges"]
    assert {cid for cid, _ in existing} == {str(older.id)}
    assert str(space_node.id) not in {str(e.target_id) for e in edges}

    # Through the event bus, exactly as memory_store publishes it.
    bus = InMemoryEventBusProvider()
    bus_llm = ScriptedLLM()
    register_handlers(bus, g, bus_llm, ConstEmbedder(), world["reg"])
    with principal("alice", orgs=[ORG], roles={a: "writer"}):
        await bus.publish("memory.created", _payload(new))
    for _ in range(5):
        await asyncio.sleep(0)
    assert "infer_causal_edges" in bus_llm.names()
    assert "space: secret client detail" not in _texts_sent(bus_llm)


# ---------------------------------------------------------------------------
# Maintenance passes
# ---------------------------------------------------------------------------


async def test_maintenance_user_pass_never_touches_spaces(world):
    g, sid = world["graph"], world["standard"]
    space_orphan = await _store(g, _node("idle space orphan", space=sid, idle_days=90, decay_score=0.75))
    with principal("bob", orgs=[ORG]):
        bob_org = await _store(g, _node(
            "bob's org-visible note", idle_days=90, decay_score=0.75, visibility=Visibility.ORG, org_id=ORG,
        ))
    with principal("alice", orgs=[ORG]):
        mine = await _store(g, _node("alice's own note", decay_score=0.75))

    # Alice is a member of the space and the request scope includes it, but
    # without a registry only her personal pass runs.
    with principal("alice", orgs=[ORG], roles={sid: "owner"}):
        result = await run_maintenance(g, ConstEmbedder(), llm=RecordingLLM())
        assert result["spaces"] == {}
        assert result["rescored"] == 1
        assert result["pruned"] == []
        assert (await g.get_node(str(space_orphan.id))).decay_score == 0.75
        assert (await g.get_node(str(bob_org.id))).decay_score == 0.75
        assert (await g.get_node(str(mine.id))).decay_score != 0.75


class SpyGraph(InMemoryGraphProvider):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, str, list[str], list[str], bool]] = []

    async def get_orphans(self) -> list[MemoryNode]:
        self.seen.append((
            current_user_id.get(), current_principal_type.get("user"), current_org_ids.get([]),
            current_scope_space_ids.get([]), current_include_personal.get(True),
        ))
        return await super().get_orphans()


async def test_maintenance_space_pass_context_restored():
    g = SpyGraph()
    reg = InMemorySpaceRegistry()
    std = await _space(reg, "std", policy=ForgettingPolicy.STANDARD)
    retain = await _space(reg, "keep")
    archived = await _space(reg, "old", policy=ForgettingPolicy.STANDARD)
    erased = await _space(reg, "gone", policy=ForgettingPolicy.STANDARD)
    for sid in (std, retain, archived, erased):
        await _store(g, _node(f"orphan in {sid}", space=sid, idle_days=90))
    await reg.update_space(archived, {"archived_at": datetime.now(UTC)})
    await reg.mark_erased(erased)

    with principal("svc_1", orgs=[ORG], roles={std: "writer"}, principal_type="service"):
        before = (current_user_id.get(), current_principal_type.get(), current_org_ids.get(),
                  current_scope_space_ids.get(), current_include_personal.get())
        result = await run_maintenance(g, ConstEmbedder(), spaces=reg)
        after = (current_user_id.get(), current_principal_type.get(), current_org_ids.get(),
                 current_scope_space_ids.get(), current_include_personal.get())
    assert before == after

    # Personal pass, then one pass per live space (archived included, erased skipped);
    # sweep only for standard spaces.
    assert g.seen[0] == ("svc_1", "service", [], [], True)
    space_calls = g.seen[1:]
    assert sorted(space_calls) == sorted([
        (space_bucket_key(sid), "user", [], [sid], False) for sid in (std, archived)
    ])
    assert set(result["spaces"]) == {std, retain, archived}
    assert len(result["spaces"][std]["pruned"]) == 1
    assert len(result["spaces"][archived]["pruned"]) == 1
    assert result["spaces"][retain]["pruned"] == []
    assert erased not in result["spaces"]
    with principal("x", roles={erased: "reader"}, include_personal=False):
        assert len(await g.get_nodes_by_status(MemoryStatus.ACTIVE)) == 1   # erased space untouched
