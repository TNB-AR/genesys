from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator

from papez.models.enums import MemoryStatus, ReactivationPattern, Visibility
from papez.models.ref import Ref


class MemoryNode(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: MemoryStatus = MemoryStatus.ACTIVE
    content_summary: str = Field(max_length=200)
    content_full: str | None = None
    content_ref: str | None = None
    embedding: list[float] | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_reactivated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Lifecycle scores
    decay_score: float = 1.0
    causal_weight: int = 0
    reactivation_count: int = 0
    reactivation_pattern: ReactivationPattern = ReactivationPattern.SINGLE
    irrelevance_counter: int = 0

    # Provenance
    source_agent: str = "claude"
    source_session: str = ""

    # Classification
    entity_refs: list[str] = Field(default_factory=list)
    category: str | None = None

    # Stability (increases on successful retrieval, per spaced repetition)
    stability: float = 1.0

    # Core memory
    pinned: bool = False
    promotion_reason: str | None = None

    # Reactivation history
    reactivation_timestamps: list[datetime] = Field(default_factory=list)

    # Organization scoping
    org_id: str | None = None
    visibility: Visibility = Visibility.PRIVATE
    original_user_id: str | None = None

    # Provenance and labels (R1)
    refs: list[Ref] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Mind Space (R2). None = personal memory, exactly as today. Set at
    # creation and never changed; promote_to_space creates a copy.
    space_id: str | None = None

    # Caller's own id for this memory (R5); unique within a space (or per user
    # for personal memories) among non-superseded nodes.
    external_id: str | None = Field(default=None, max_length=256)

    # Id of the personal node this space node was copied from (promote_to_space, D29).
    promoted_from: str | None = None

    @model_validator(mode="after")
    def _space_nodes_are_private(self) -> MemoryNode:
        # Spec 6.2: a node with a space_id is always PRIVATE with no org_id.
        if self.space_id is not None and (self.visibility != Visibility.PRIVATE or self.org_id is not None):
            raise ValueError("a node with space_id must have visibility='private' and org_id=None")
        return self
