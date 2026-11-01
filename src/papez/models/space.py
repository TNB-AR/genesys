"""Mind Space models (spec 6.2)."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from papez.models.enums import ForgettingPolicy, PrincipalType, SpaceRole

SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,62}$"


class Space(BaseModel):
    id: str = Field(default_factory=lambda: f"spc_{uuid.uuid4().hex}")
    org_id: str                                   # the org this space belongs to
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=SLUG_PATTERN)       # unique within the org, e.g. "client-a-payments"
    description: str | None = Field(default=None, max_length=1000)
    forgetting_policy: ForgettingPolicy = ForgettingPolicy.RETAIN_ALL
    llm_inference_enabled: bool = True            # if False, no memory content from this space is sent to an LLM
    created_by: str                               # principal id
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    archived_at: datetime | None = None           # set = archived (read-only, hidden from default recall)

    @property
    def archived(self) -> bool:
        return self.archived_at is not None


class SpaceMember(BaseModel):
    space_id: str
    principal_id: str                             # a user id or a service credential id
    principal_type: PrincipalType
    role: SpaceRole
    added_by: str
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
