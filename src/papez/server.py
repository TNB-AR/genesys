"""Papez MCP Server: stdio transport for Claude Desktop.

Runs on both the mcp 1.x and 2.x Python SDKs. The two differ in how a
low-level ``Server`` learns about tools: 1.x exposes the ``@server.list_tools()``
/ ``@server.call_tool()`` decorators, 2.x replaced them with
``Server.add_request_handler(method, params_type, handler)`` (or ``on_list_tools``
/ ``on_call_tool`` constructor arguments). ``_register_tool_handlers`` picks the
API by feature detection at import time; the tool list, schemas, dispatch and
error payloads below are shared by both paths.
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from papez.context import (
    current_client,
    current_org_ids,
    current_principal_type,
    current_space_roles,
    current_user_id,
)
from papez.providers import get_providers

STDIO_LOCAL_USER = "stdio_local_user"
# The stdio user's single org, so space_create's "defaults to the caller's only
# org" works locally (C15).
STDIO_LOCAL_ORG = "stdio_local_org"

SERVER_NAME = "papez"
SERVER_WEBSITE_URL = "https://github.com/Astrix-Labs/papez"
# serverInfo.icons, as ``mcp.types.Icon`` keyword arguments (``src`` must be an
# https URL or data URI). Empty until a hosted icon asset exists; anything
# listed here is advertised on every SDK whose ``Server`` accepts ``icons``.
SERVER_ICONS: list[dict[str, Any]] = []

providers = get_providers()
tools = providers.tools
# Attach the registry/audit sink when the provider factory did not pass them.
if getattr(tools, "spaces", None) is None:
    tools.spaces = getattr(providers, "spaces", None)
if getattr(tools, "audit", None) is None:
    tools.audit = getattr(providers, "audit", None)


def _package_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("papez")
    except PackageNotFoundError:
        return None


def _build_server() -> Any:
    """Construct the low-level ``Server`` with the serverInfo fields this SDK accepts.

    ``version``, ``website_url`` and ``icons`` were added to ``Server.__init__``
    at different points in the 1.x line and all exist in 2.x. Passing only the
    keywords the installed signature declares keeps the floor at mcp 1.0.
    """
    accepted = inspect.signature(Server.__init__).parameters
    kwargs: dict[str, Any] = {}
    version = _package_version()
    if version and "version" in accepted:
        kwargs["version"] = version
    if "website_url" in accepted:
        kwargs["website_url"] = SERVER_WEBSITE_URL
    if SERVER_ICONS and "icons" in accepted:
        from mcp.types import Icon

        kwargs["icons"] = [Icon(**icon) for icon in SERVER_ICONS]
    return Server(SERVER_NAME, **kwargs)


app = _build_server()

# Tool name → (method, required_args, optional_args_with_defaults)
_TOOL_DISPATCH: dict[str, tuple[Any, ...]] = {
    "memory_store": (tools.memory_store, ["content"], {"source_session": "", "related_to": None, "related": None, "category": None, "visibility": "private", "org_id": None, "refs": None, "tags": None, "space": None, "external_id": None, "on_existing": "amend"}),
    "memory_amend": (tools.memory_amend, ["node_id", "content"], {"reason": None, "refs": None, "tags": None}),
    "memory_recall": (tools.memory_recall, ["query"], {"k": 10, "max_results": None, "verbosity": "full", "tags_any": None, "ref_type": None, "space": None, "spaces": None, "include_personal": None, "category": None, "exclude_superseded": False}),
    "memory_search": (tools.memory_search, ["query"], {"filters": None, "k": 10, "space": None, "spaces": None, "include_personal": None}),
    "memory_traverse": (tools.memory_traverse, ["node_id"], {"depth": 2, "edge_types": None, "space": None, "spaces": None, "include_personal": None}),
    "memory_explain": (tools.memory_explain, ["node_id"], {}),
    "pin_memory": (tools.pin_memory, ["node_id"], {}),
    "unpin_memory": (tools.unpin_memory, ["node_id"], {}),
    "list_core_memories": (tools.list_core_memories, [], {"category": None, "space": None}),
    "delete_memory": (tools.delete_memory, ["node_id"], {}),
    "memory_stats": (tools.memory_stats, [], {"space": None}),
    "set_core_preferences": (tools.set_core_preferences, [], {"auto": None, "approval": None, "excluded": None}),
    "promote_to_org": (tools.promote_to_org, ["node_id", "org_id"], {"action": "keep_private", "dry_run": False}),
    # Mind Spaces (spec 6.8)
    "space_create": (tools.space_create, ["name"], {"slug": None, "org_id": None, "description": None, "forgetting_policy": "retain_all", "llm_inference_enabled": True, "show_service_names": True}),
    "space_list": (tools.space_list, [], {"include_archived": False}),
    "space_get": (tools.space_get, ["space"], {}),
    "space_update": (tools.space_update, ["space"], {"name": None, "description": None, "forgetting_policy": None, "llm_inference_enabled": None, "show_service_names": None}),
    "space_member_add": (tools.space_member_add, ["space", "principal_id", "principal_type", "role"], {}),
    "space_member_remove": (tools.space_member_remove, ["space", "principal_id"], {}),
    "space_member_list": (tools.space_member_list, ["space"], {}),
    "space_archive": (tools.space_archive, ["space"], {}),
    "space_unarchive": (tools.space_unarchive, ["space"], {}),
    "space_erase": (tools.space_erase, ["space", "confirm_slug"], {"dry_run": False}),
    "promote_to_space": (tools.promote_to_space, ["node_id", "space"], {"keep_original": True}),
    # Idempotent writes (spec 9.4, 9.5)
    "memory_store_batch": (tools.memory_store_batch, ["items"], {"space": None, "on_existing": "amend"}),
    "memory_retire_by_ref": (tools.memory_retire_by_ref, ["ref_type", "ref_id"], {"space": None, "mode": "amend"}),
}

# Shared JSON Schema fragments for the Mind Spaces parameters. Never put a
# Python None into a schema: served schemas are compared with exclude_none.
_REFS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Where this memory came from. Each ref points to a document, meeting, ticket, URL or other source.",
    "items": {
        "type": "object", "required": ["type", "id"],
        "properties": {
            "type": {"type": "string", "description": "Kind of source, lowercase, e.g. 'document', 'meeting', 'ticket', 'url'."},
            "id": {"type": "string", "description": "The source's id in its own system."},
            "url": {"type": "string", "description": "Optional link to the source."},
            "label": {"type": "string", "description": "Optional human-readable name, e.g. a file title."},
        },
    },
}
_TAGS_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}, "description": "Labels for filtering, e.g. 'payments', 'bank-a'. Normalized to lowercase-with-dashes."}
_SPACE_WRITE_SCHEMA: dict[str, Any] = {"type": "string", "description": "Space id (spc_...) or slug to write into. Needs the writer or owner role."}
_SPACE_READ_SCHEMA: dict[str, Any] = {"type": "string", "description": "Limit this call to one space (id or slug). Can't be combined with `spaces`."}
_SPACES_READ_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}, "description": "Limit this call to several spaces (ids or slugs). Can't be combined with `space`."}
_INCLUDE_PERSONAL_SCHEMA: dict[str, Any] = {"type": "boolean", "description": "Include your personal memory. Defaults to true, or false when a space is named. Ignored for service principals."}
_SPACE_REF_SCHEMA: dict[str, Any] = {"type": "string", "description": "Space id (spc_...) or slug."}

# Tool definitions as plain dicts (MCP wire shape, camelCase keys). mcp 1.x
# names its model fields camelCase; 2.x names them snake_case with camelCase
# aliases. ``Tool.model_validate`` accepts the wire shape on both, and the
# dicts stay the single source of truth for argument validation.
_TOOL_DEFS: list[dict[str, Any]] = [
    {"name": "memory_store", "description": "Store a new memory in the causal memory graph. Use `related` for writer-specified typed edges (each {id, type}); `related_to` is legacy and always creates caused_by edges. May return `possible_conflicts` — heuristic hints, not verified contradictions.", "inputSchema": {
        "type": "object", "required": ["content"],
        "properties": {
            "content": {"type": "string"},
            "source_session": {"type": "string", "default": ""},
            "related_to": {"type": "array", "items": {"type": "string"}, "description": "Legacy: ids of nodes to link via caused_by. Prefer `related`."},
            "related": {"type": "array", "description": "Typed explicit edges. Direction: new_node --type--> target.", "items": {
                "type": "object", "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["caused_by", "supports", "contradicts", "supersedes", "derived_from", "related_to", "temporal_sequence"]},
                },
            }},
            "category": {"type": "string", "description": "Free-form classification (suggested: professional, educational, family, location)."},
            "visibility": {"type": "string", "enum": ["private", "org"], "default": "private"},
            "org_id": {"type": "string", "description": "Required when visibility is 'org'. Must be an org the caller belongs to."},
            "refs": _REFS_SCHEMA,
            "tags": _TAGS_SCHEMA,
            "space": _SPACE_WRITE_SCHEMA,
            "external_id": {"type": "string", "maxLength": 256, "description": "Your own id for this memory, unique per space among current memories. Makes the store idempotent."},
            "on_existing": {"type": "string", "enum": ["skip", "amend", "error"], "default": "amend", "description": "What to do when a current memory with this external_id exists: skip (return it), amend (supersede it unless identical), error (external_id_exists)."},
        },
    }},
    {"name": "memory_amend", "description": "Record a correction: creates a new memory that supersedes an existing one. The old memory is kept (decayed in recall results), not deleted.", "inputSchema": {
        "type": "object", "required": ["node_id", "content"],
        "properties": {
            "node_id": {"type": "string"},
            "content": {"type": "string"},
            "reason": {"type": "string"},
            "refs": {**_REFS_SCHEMA, "description": "Replace the refs. Omit to copy the old memory's refs."},
            "tags": {**_TAGS_SCHEMA, "description": "Replace the tags. Omit to copy the old memory's tags."},
        },
    }},
    {"name": "memory_recall", "description": "Recall memories using hybrid search (vector + keyword + graph spreading activation).", "inputSchema": {
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 10},
            "max_results": {"type": "integer"},
            "verbosity": {"type": "string", "enum": ["concise", "light", "full"], "default": "full", "description": "concise = id/summary/status/score/activation/is_core/tags/ref_count/space only, no causal chains. light = full hits without the causal chains."},
            "tags_any": {"type": "array", "items": {"type": "string"}, "description": "Only memories with at least one of these tags (applied before ranking)."},
            "ref_type": {"type": "string", "description": "Only memories with a ref of this type."},
            "space": _SPACE_READ_SCHEMA,
            "spaces": _SPACES_READ_SCHEMA,
            "include_personal": _INCLUDE_PERSONAL_SCHEMA,
            "category": {"type": "string", "description": "Only memories in this category."},
            "exclude_superseded": {"type": "boolean", "default": False, "description": "Leave out memories a newer one has superseded."},
        },
    }},
    {"name": "memory_search", "description": "Filtered vector search by status, category, date, or entity. Pass an EMPTY query to enumerate by recency instead (no vector search, no embedder needed): with since/active_since this answers 'what's new/changed since <ts>' without knowing what to query for.", "inputSchema": {
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search text. Empty string switches to enumeration mode: nodes listed by last_reactivated_at descending, honoring the same filters."},
            "filters": {"type": "object", "description": "Supported keys: status (list), category, entity, since (ISO date — created_at >= since; tz-naive treated as UTC), active_since (ISO date — last_reactivated_at >= active_since; tz-naive treated as UTC), tags_any (list: at least one tag), tags_all (list: every tag), ref_type, ref_id (with ref_type, both must match on the same ref). With a non-empty query, results are vector-seeded and k-limited; use an empty query for enumeration."},
            "k": {"type": "integer", "default": 10},
            "space": _SPACE_READ_SCHEMA,
            "spaces": _SPACES_READ_SCHEMA,
            "include_personal": _INCLUDE_PERSONAL_SCHEMA,
        },
    }},
    {"name": "memory_traverse", "description": "Traverse the memory graph from a starting node. Returns reachable nodes AND the edges of the induced subgraph among them (source/target/type/weight/created_by) — a superset of the BFS tree, so paths can be reconstructed.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {
            "node_id": {"type": "string"},
            "depth": {"type": "integer", "default": 2},
            "edge_types": {"type": "array", "items": {"type": "string"}},
            "space": _SPACE_READ_SCHEMA,
            "spaces": _SPACES_READ_SCHEMA,
            "include_personal": _INCLUDE_PERSONAL_SCHEMA,
        },
    }},
    {"name": "memory_explain", "description": "Explain a memory's score breakdown.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "pin_memory", "description": "Pin a memory to core status.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "unpin_memory", "description": "Unpin a memory and re-evaluate core eligibility.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "list_core_memories", "description": "List core memories, optionally filtered by category. Personal ones by default, or one space's with `space`.", "inputSchema": {
        "type": "object",
        "properties": {"category": {"type": "string"}, "space": {"type": "string", "description": "List this space's core memories (id or slug; reader+)."}},
    }},
    {"name": "delete_memory", "description": "Permanently delete a memory node and all its edges.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "memory_stats", "description": "Get graph statistics: personal by default, or one space's with `space`.", "inputSchema": {
        "type": "object", "properties": {"space": {"type": "string", "description": "Stats for this space (id or slug; reader+)."}},
    }},
    {"name": "set_core_preferences", "description": "Configure core memory category preferences.", "inputSchema": {
        "type": "object",
        "properties": {
            "auto": {"type": "array", "items": {"type": "string"}},
            "approval": {"type": "array", "items": {"type": "string"}},
            "excluded": {"type": "array", "items": {"type": "string"}},
        },
    }},
    {"name": "promote_to_org", "description": "Promote a private memory to org visibility. Caller must own the node and belong to the target org.", "inputSchema": {
        "type": "object", "required": ["node_id", "org_id"],
        "properties": {
            "node_id": {"type": "string"},
            "org_id": {"type": "string"},
            "action": {"type": "string", "enum": ["keep_private", "promote_all", "delete_links"], "default": "keep_private"},
            "dry_run": {"type": "boolean", "default": False},
        },
    }},
    # -- Mind Spaces (spec 6.8) ------------------------------------------------
    {"name": "space_create", "description": "Create a Mind Space: a shared memory graph for a team or engagement. You become its first owner. Users only.", "inputSchema": {
        "type": "object", "required": ["name"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "slug": {"type": "string", "description": "Lowercase id for the space, unique in the org. Generated from name if omitted."},
            "org_id": {"type": "string", "description": "Required if you belong to several orgs."},
            "description": {"type": "string"},
            "forgetting_policy": {"type": "string", "enum": ["retain_all", "standard"], "default": "retain_all"},
            "llm_inference_enabled": {"type": "boolean", "default": True, "description": "If false, no memory content from this space is sent to an LLM."},
            "show_service_names": {"type": "boolean", "default": True, "description": "If false, embed author lines don't name the writing credential."},
        },
    }},
    {"name": "space_list", "description": "List the spaces you are a member of, with your role and counts.", "inputSchema": {
        "type": "object",
        "properties": {"include_archived": {"type": "boolean", "default": False}},
    }},
    {"name": "space_get", "description": "Get one space with its member and memory counts.", "inputSchema": {
        "type": "object", "required": ["space"],
        "properties": {"space": _SPACE_REF_SCHEMA},
    }},
    {"name": "space_update", "description": "Change a space's name, description or policies (owner). The slug can't be changed.", "inputSchema": {
        "type": "object", "required": ["space"],
        "properties": {
            "space": _SPACE_REF_SCHEMA,
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string"},
            "forgetting_policy": {"type": "string", "enum": ["retain_all", "standard"]},
            "llm_inference_enabled": {"type": "boolean"},
            "show_service_names": {"type": "boolean"},
        },
    }},
    {"name": "space_member_add", "description": "Add a member to a space, or change an existing member's role (owner).", "inputSchema": {
        "type": "object", "required": ["space", "principal_id", "principal_type", "role"],
        "properties": {
            "space": _SPACE_REF_SCHEMA,
            "principal_id": {"type": "string", "description": "A user id or a service credential id."},
            "principal_type": {"type": "string", "enum": ["user", "service"]},
            "role": {"type": "string", "enum": ["owner", "writer", "reader"], "description": "A service principal can't be owner."},
        },
    }},
    {"name": "space_member_remove", "description": "Remove a member from a space (owner). Takes effect on their next request.", "inputSchema": {
        "type": "object", "required": ["space", "principal_id"],
        "properties": {"space": _SPACE_REF_SCHEMA, "principal_id": {"type": "string"}},
    }},
    {"name": "space_member_list", "description": "List a space's members.", "inputSchema": {
        "type": "object", "required": ["space"],
        "properties": {"space": _SPACE_REF_SCHEMA},
    }},
    {"name": "space_archive", "description": "Archive a space (owner): read-only, hidden from default recall, readable when named explicitly.", "inputSchema": {
        "type": "object", "required": ["space"],
        "properties": {"space": _SPACE_REF_SCHEMA},
    }},
    {"name": "space_unarchive", "description": "Unarchive a space (owner).", "inputSchema": {
        "type": "object", "required": ["space"],
        "properties": {"space": _SPACE_REF_SCHEMA},
    }},
    {"name": "space_erase", "description": "PERMANENTLY delete a space and every memory and edge in it (owner). confirm_slug must equal the space's slug; dry_run returns the counts without deleting.", "inputSchema": {
        "type": "object", "required": ["space", "confirm_slug"],
        "properties": {
            "space": _SPACE_REF_SCHEMA,
            "confirm_slug": {"type": "string"},
            "dry_run": {"type": "boolean", "default": False},
        },
    }},
    {"name": "promote_to_space", "description": "Copy one of your personal memories into a space (writer or owner). Content, refs, tags and category are copied; edges are not.", "inputSchema": {
        "type": "object", "required": ["node_id", "space"],
        "properties": {
            "node_id": {"type": "string"},
            "space": _SPACE_WRITE_SCHEMA,
            "keep_original": {"type": "boolean", "default": True, "description": "If false, delete the personal original after copying."},
        },
    }},
    # -- Idempotent writes (spec 9.4, 9.5) -------------------------------------
    {"name": "memory_store_batch", "description": "Store up to GENESYS_BATCH_MAX_ITEMS (default 100) memories in one call, all into one space (or your personal memory when `space` is omitted; services must name a space). Every item is validated first; if any item is invalid nothing is stored and the error `batch_invalid` lists each problem with its index. Items are then stored in order; an item's `related` entry may be {\"ref_index\": i, \"type\": t} to link to earlier item i. Returns one result per item: {index, node_id, status: stored|exists|unchanged|amended, superseded?}. Retry safety: when every item carries an `external_id` the batch is idempotent, so it is safe to resend the whole batch after a timeout (unchanged items come back with the same node ids). Without external_id a resend creates duplicates.", "inputSchema": {
        "type": "object", "required": ["items"],
        "properties": {
            "items": {"type": "array", "description": "The memories to store, in order.", "items": {
                "type": "object", "required": ["content"],
                "properties": {
                    "content": {"type": "string"},
                    "refs": _REFS_SCHEMA,
                    "tags": _TAGS_SCHEMA,
                    "category": {"type": "string"},
                    "external_id": {"type": "string", "maxLength": 256, "description": "Your own id for this memory; makes the item idempotent."},
                    "source_session": {"type": "string"},
                    "created_at": {"type": "string", "description": "Optional ISO 8601 timestamp."},
                    "related": {"type": "array", "description": "Typed edges from this item: {id, type} for an existing node in the same space, or {ref_index, type} for an earlier item of this batch.", "items": {
                        "type": "object", "required": ["type"],
                        "properties": {
                            "id": {"type": "string"},
                            "ref_index": {"type": "integer", "minimum": 0},
                            "type": {"type": "string", "enum": ["caused_by", "supports", "contradicts", "supersedes", "derived_from", "related_to", "temporal_sequence"]},
                        },
                    }},
                },
            }},
            "space": _SPACE_WRITE_SCHEMA,
            "on_existing": {"type": "string", "enum": ["skip", "amend", "error"], "default": "amend", "description": "Applies to every item whose external_id already exists: skip, amend (unless identical), or error (the whole batch is refused)."},
        },
    }},
    {"name": "memory_retire_by_ref", "description": "Retire the memories that cite a source that was removed. mode=amend (default): a memory whose only ref is this one is amended to a '[Retired] ...' note tagged `retired` (history is kept); a memory with other refs is amended with this ref removed. mode=delete: sole-ref memories are deleted and the ref is removed from the others. Needs writer or owner when `space` is named; without `space` it acts on your personal memory (services must name a space). Returns {retired, updated, deleted}.", "inputSchema": {
        "type": "object", "required": ["ref_type", "ref_id"],
        "properties": {
            "ref_type": {"type": "string", "description": "The removed source's ref type, e.g. 'document'."},
            "ref_id": {"type": "string", "description": "The removed source's id."},
            "space": _SPACE_WRITE_SCHEMA,
            "mode": {"type": "string", "enum": ["amend", "delete"], "default": "amend"},
        },
    }},
]

_TOOL_SCHEMAS: list[Tool] = [Tool.model_validate(definition) for definition in _TOOL_DEFS]


async def list_tools() -> list[Tool]:
    return _TOOL_SCHEMAS


# Tools with no side effects worth worrying about (beyond reactivation
# bookkeeping) — safe for clients to retry on failure. Writes are NOT in this
# set: a failure may have landed after a partial write, so clients should
# reconcile via memory_recall instead of blind-retrying (see README
# "Reliability & retries").
_RETRYABLE_TOOLS = {
    "memory_recall", "memory_search", "memory_traverse", "memory_explain",
    "memory_stats", "list_core_memories",
    "space_list", "space_get", "space_member_list",
}

_logger = logging.getLogger(__name__)


def _is_retryable(name: str, arguments: dict[str, Any]) -> bool:
    """Whether a failed call is safe to resend as-is.

    Read tools always are. ``memory_store_batch`` is too when every item
    carries an ``external_id`` (spec 9.4): a resend then re-matches each item
    instead of duplicating it.
    """
    if name in _RETRYABLE_TOOLS:
        return True
    if name == "memory_store_batch":
        items = arguments.get("items")
        return (
            isinstance(items, list) and bool(items)
            and all(isinstance(i, dict) and isinstance(i.get("external_id"), str) and i["external_id"] for i in items)
        )
    return False


def _error_content(message: str, retryable: bool) -> list[TextContent]:
    payload = {"error": message, "retryable": retryable}
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch a tool call, degrading gracefully instead of hard-failing.

    A memory call is not worth crashing an agent turn over: any tool
    exception is caught and returned as a structured
    ``{"error": ..., "retryable": bool}`` payload rather than propagating as
    a protocol-level MCP failure. ``retryable`` mirrors the README guidance —
    true only for read tools; failed writes should be reconciled, not
    blind-retried.
    """
    if name not in _TOOL_DISPATCH:
        return _error_content(f"Unknown tool: {name}", retryable=False)

    method, required, optional = _TOOL_DISPATCH[name]
    missing = [k for k in required if k not in arguments]
    if missing:
        return _error_content(
            f"missing required argument(s) for {name}: {', '.join(missing)}",
            retryable=False,
        )
    kwargs = {k: arguments[k] for k in required}
    for k, default in optional.items():
        kwargs[k] = arguments.get(k, default)

    roles_token = await _refresh_stdio_space_roles()
    client_token = _bind_stdio_client_name()
    try:
        result = await method(**kwargs)
    except PermissionError as exc:
        return _error_content(str(exc), retryable=False)
    except Exception as exc:
        _logger.exception("Tool %s failed", name)
        return _error_content(
            f"{type(exc).__name__}: {exc}", retryable=_is_retryable(name, arguments)
        )
    finally:
        if client_token is not None:
            current_client.reset(client_token)
        if roles_token is not None:
            current_space_roles.reset(roles_token)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def _stdio_client_name() -> str | None:
    """The MCP client's ``clientInfo.name`` from the initialize handshake, if any."""
    try:
        session = app.request_context.session
    except (LookupError, AttributeError):
        return None
    params = getattr(session, "client_params", None)
    info = getattr(params, "clientInfo", None)
    name = getattr(info, "name", None)
    return name if isinstance(name, str) and name else None


def _bind_stdio_client_name() -> Any:
    """Fill ``current_client["client_name"]`` for this call on stdio (spec 8.2, D18).

    ``main()`` sets ``{"transport": "stdio", "client_name": None}``; the name
    is only known once the client has initialized, so it is read per call
    from the MCP session. Returns the token to reset, or None when unchanged.
    """
    client = current_client.get({})
    if client.get("transport") != "stdio" or client.get("client_name"):
        return None
    name = _stdio_client_name()
    if name is None:
        return None
    return current_client.set({**client, "client_name": name})


async def _load_space_roles(registry: Any, principal_id: str) -> dict[str, str]:
    roles = await registry.roles_for_principal(principal_id)
    return {sid: str(getattr(role, "value", role)) for sid, role in roles.items()}


async def _refresh_stdio_space_roles() -> Any:
    """Reload ``current_space_roles`` from the registry for this call (D06, C15).

    Only on the stdio transport (``current_client`` transport ``"stdio"``, set
    by ``main()``): membership changes then take effect on the very next call.
    Hosted transports bind identity themselves. Returns the context token to
    reset after the call, or None when nothing was set.
    """
    registry = getattr(tools, "spaces", None)
    if registry is None or current_client.get({}).get("transport") != "stdio":
        return None
    if current_principal_type.get("user") != "user":
        return None
    uid = current_user_id.get(None)
    if not uid:
        return None
    try:
        roles = await _load_space_roles(registry, uid)
    except Exception:
        _logger.warning("could not refresh space roles for %s", uid, exc_info=True)
        return None
    return current_space_roles.set(roles)


# --- SDK compatibility layer -------------------------------------------------
#
# mcp 1.x's ``@call_tool()`` decorator does two things around the handler
# before the result reaches the wire: it validates ``arguments`` against the
# tool's ``inputSchema`` (returning an ``isError`` result whose text is
# ``"Input validation error: <message>"``) and it converts any exception into
# an ``isError`` result carrying ``str(exc)``. The 2.x request-handler API does
# neither, so the 2.x adapter below replicates both, and a client sees the
# same bytes whichever SDK the server was installed with.


def _tool_input_schema(name: str) -> dict[str, Any] | None:
    for definition in _TOOL_DEFS:
        if definition["name"] == name:
            schema: dict[str, Any] = definition["inputSchema"]
            return schema
    return None


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Return the 1.x-style validation error text, or None when the arguments pass."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a hard dependency of mcp
        return None
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"Input validation error: {exc.message}"
    return None


def _register_tool_handlers(server: Any) -> str:
    """Attach ``list_tools``/``call_tool`` using whichever API the installed SDK has.

    Returns the mechanism used (``"decorators"`` for 1.x,
    ``"request_handlers"`` for 2.x) so tests and diagnostics can report it.
    """
    if hasattr(server, "list_tools") and hasattr(server, "call_tool"):
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return "decorators"

    if hasattr(server, "add_request_handler"):
        from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams

        def _sdk_error_result(message: str) -> Any:
            # Wire-shape keys: the field is ``isError`` on 1.x and ``is_error`` on 2.x.
            return CallToolResult.model_validate({"content": [{"type": "text", "text": message}], "isError": True})

        async def _on_list_tools(ctx: Any, params: Any) -> Any:
            return ListToolsResult(tools=await list_tools())

        async def _on_call_tool(ctx: Any, params: Any) -> Any:
            name = params.name
            arguments = params.arguments or {}
            try:
                schema = _tool_input_schema(name)
                if schema is not None:
                    problem = _validate_arguments(arguments, schema)
                    if problem is not None:
                        return _sdk_error_result(problem)
                content = await call_tool(name, arguments)
            except Exception as exc:
                _logger.exception("Tool %s failed outside dispatch", name)
                return _sdk_error_result(str(exc))
            return CallToolResult(content=list(content))

        server.add_request_handler("tools/list", PaginatedRequestParams, _on_list_tools)
        server.add_request_handler("tools/call", CallToolRequestParams, _on_call_tool)
        return "request_handlers"

    raise RuntimeError(
        "Unsupported mcp SDK: Server has neither list_tools/call_tool decorators (mcp 1.x) "
        "nor add_request_handler (mcp 2.x)"
    )


SDK_TOOL_API = _register_tool_handlers(app)


async def main() -> None:
    # Stdio identity (spec 7.3, D02): one local user in one local org.
    current_user_id.set(STDIO_LOCAL_USER)
    current_principal_type.set("user")
    current_org_ids.set([STDIO_LOCAL_ORG])
    current_client.set({"transport": "stdio", "client_name": None})
    graph = providers.graph
    await graph.initialize(STDIO_LOCAL_USER)
    spaces = getattr(providers, "spaces", None)
    if spaces is not None:
        await spaces.initialize()
        current_space_roles.set(await _load_space_roles(spaces, STDIO_LOCAL_USER))
    else:
        current_space_roles.set({})

    # Forgetting only happens if something runs the sweep. Rescore, transition
    # and prune on a cadence (GENESYS_MAINTENANCE_INTERVAL_S, default 600s).
    import asyncio

    from papez.engine.maintenance import maintenance_loop

    loop_kwargs: dict[str, Any] = {}
    if spaces is not None and "spaces" in inspect.signature(maintenance_loop).parameters:
        loop_kwargs["spaces"] = spaces  # space passes once WS-L4's loop accepts the registry
    maintenance = asyncio.create_task(
        maintenance_loop(graph, providers.embeddings, providers.llm, **loop_kwargs)
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        maintenance.cancel()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
