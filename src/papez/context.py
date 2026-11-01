"""Request-scoped user and org identity via contextvars."""
from __future__ import annotations

import contextvars
from typing import Any

current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_id")
current_org_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_org_id", default=None)
# No default on purpose: a shared mutable list default is a cross-request
# leak hazard. Readers call current_org_ids.get([]) for a fresh empty list.
current_org_ids: contextvars.ContextVar[list[str]] = contextvars.ContextVar("current_org_ids")
current_user_role: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user_role", default=None)

# --- Mind Spaces and scoped access (spec 7.3) ---------------------------------
# Same "no mutable default" convention as current_org_ids: the variables below
# without a default must be read with an explicit fresh fallback, e.g.
# current_space_roles.get({}), current_scope_space_ids.get([]).

# "user" | "service" (PrincipalType values). No default: readers call
# current_principal_type.get("user").
current_principal_type: contextvars.ContextVar[str] = contextvars.ContextVar("current_principal_type")
# {space_id: role} for every NON-ARCHIVED space the principal is a member of
# (values are SpaceRole strings). No default on purpose (see current_org_ids):
# readers call current_space_roles.get({}) for a fresh empty dict.
current_space_roles: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("current_space_roles")
# The spaces THIS call is limited to; set by the tool layer from the
# `space`/`spaces` params (spec 6.5). No default: readers call
# current_scope_space_ids.get([]).
current_scope_space_ids: contextvars.ContextVar[list[str]] = contextvars.ContextVar("current_scope_space_ids")
# Whether personal (and org-visible) memory is in scope. Storage treats
# current_principal_type == "service" as False regardless of this flag.
current_include_personal: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "current_include_personal", default=True,
)
# Audit client descriptor: {"transport": "stdio"|"mcp"|"rest", "client_name": str | None}.
# No default: readers call current_client.get({}).
current_client: contextvars.ContextVar[dict[str, str | None]] = contextvars.ContextVar("current_client")

# The embed token behind this request, when there is one (embeddable space
# view). Shape: {"token_id", "space_id", "viewer": {"id", "email", "name"},
# "permissions": [...]}. While it is bound, storage and the tool layer stay on
# that one space whatever a caller's arguments say. A shared default of None is
# safe here because the value is never mutated.
current_embed: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("current_embed", default=None)
