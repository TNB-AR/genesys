# Mind Spaces: behaviour reference

This is the full reference for Mind Spaces, references and tags, idempotent writes and the audit log, as shipped in papez 0.7.0. The README has the short version. For a worked example of an outside system feeding a space, see [`integration.md`](integration.md).

## Concepts

| Term | Meaning |
|---|---|
| **Org** | A company or team. Existing Papez concept; org-visible memories are unchanged. |
| **Space** | An isolated memory layer inside one org, for one engagement or project. Ids look like `spc_<hex>`; slugs (`client-a-payments`) are unique within the org and never change. |
| **Personal memory** | A memory with no space. Works exactly as before 0.7.0. |
| **Member** | A user or a service principal added to a space with a role. |
| **Role** | `owner`, `writer` or `reader`. Ordering: `owner > writer > reader`. |
| **Principal** | Whoever makes the request: a `user` or a `service`. |

**The isolation rule:** a memory that belongs to a space can only be seen, linked to, compared with or changed by principals who are members of that space, and only within what their role allows.

Rules for a space memory:

- `space_id` is set when the memory is created and never changes. To move a personal memory into a space, use `promote_to_space`, which makes a copy.
- A space memory is always `visibility="private"` with `org_id=None`. Spaces replace org visibility for that node; `visibility="org"` together with `space` returns `invalid_arguments`.
- `original_user_id` records who wrote it, for audit. Membership, not authorship, controls access.
- Space memories are stored in their own bucket (`space::<space_id>`), never in the author's. `erase_user` therefore never deletes space memories; it replaces the author id on them with `"erased_user"`.

### Users and services

| | User | Service |
|---|---|---|
| Personal memory | Yes | Never. Can't read or write personal memory. |
| Spaces | Any space they are a member of | Exactly the spaces it was granted |
| Can create spaces | Yes, in their org | No (`insufficient_role`) |
| Can be owner | Yes | No (writer or reader only) |

Service principals and their credentials exist only in the hosted product; see the developer docs at [papez.ai/developers](https://papez.ai/developers). The open-source library has no authentication. The stdio server always runs as one local user, `stdio_local_user`, in one local org, `stdio_local_org`, so `space_create` works locally without an `org_id`.

## Role permissions (spec 6.3)

| Action | reader | writer | owner |
|---|:---:|:---:|:---:|
| `memory_recall`, `memory_search`, `memory_traverse`, `memory_explain` in the space | yes | yes | yes |
| `memory_store`, `memory_store_batch`, `memory_amend` into the space | no | yes | yes |
| `memory_retire_by_ref` in the space | no | yes | yes |
| `pin_memory`, `unpin_memory` on space memories | no | yes | yes |
| `delete_memory` on space memories | no | yes | yes |
| `promote_to_space` into the space | no | yes | yes |
| `space_get`, `space_member_list` | yes | yes | yes |
| `space_update` (name, description, policies) | no | no | yes |
| `space_member_add`, `space_member_remove` | no | no | yes |
| `space_archive`, `space_unarchive`, `space_erase` | no | no | yes |

A member without the needed role gets `insufficient_role`. A non-member gets `space_not_found`, exactly as if the space did not exist.

An **archived** space is read-only for everyone: every write above returns `space_archived`, except `space_unarchive` and `space_erase`.

A space always keeps at least one **user** owner. Removing or demoting the last one returns `last_owner`.

Org admins get no rights over space memories. Space nodes have no `org_id`, so org-admin privileges never apply to them.

## Visibility (spec 6.5)

Every read goes through one visibility rule, applied in storage:

```
if node.space_id is not None:
    visible = node.space_id in scope_space_ids
else:
    visible = include_personal and (
        node is in the caller's own bucket
        or (node.visibility == ORG and node.org_id in current_org_ids)
    )
```

- `member_space_ids` is every **non-archived** space the caller is a member of.
- `scope_space_ids` is the set of spaces this call is limited to. It comes from the tool's `space` / `spaces` parameter, or defaults as described below. It is always a subset of the caller's memberships.
- `include_personal` decides whether personal and org memories are in scope. It is always `False` for services, whatever the caller passes. Storage enforces this too.

Every storage read applies the rule: `get_node` returns `None` for an invisible node, searches filter before taking the top `k`, edges are returned only when both ends are visible, traversal never steps onto an invisible node, and stats, orphans and degrees count only what the caller can see. A space node's connectivity is computed from edges inside that space only.

### Read scope per tool

| Tool | No `space`/`spaces` given | `space="x"` or `spaces=[...]` |
|---|---|---|
| `memory_recall`, `memory_search`, `memory_traverse` | All non-archived memberships, plus personal | Only the named spaces. Personal is excluded unless you pass `include_personal=true`. |
| `memory_explain` | All non-archived memberships, plus personal | (no `space` parameter) |
| `list_core_memories`, `memory_stats` | Personal only | That one space only (reader+) |

- `space` and `spaces` together return `invalid_arguments`.
- A space is named by id (`spc_...`) or by slug. Slugs are looked up within the caller's own orgs.
- An unknown space and a space the caller is not a member of return the same body: `{"error": "Space not found", "code": "space_not_found"}`. The error never echoes the name you passed.
- An archived space is out of default recall. Members can still read it by naming it explicitly.
- Every recall, search and explain result carries `"space": {"id", "slug", "name"}`, or `null` for personal and org memories.
- `memory_traverse` from a start node outside the scope returns `{"error": "Node not found", "code": "not_found", "node_id": ...}`.

### Write scope

- `memory_store(space="x")` needs writer or owner, and the space must not be archived.
- A service that writes without `space` gets `space_required`.
- `memory_amend`, `pin_memory`, `unpin_memory` and `delete_memory` act on a node; for a space node the caller needs writer or owner in that node's space. The new node from `memory_amend` keeps the old node's `space_id` and `external_id`. For personal nodes, ownership rules are unchanged.
- `promote_to_org` on a space node returns `invalid_arguments`.

### Edges stay inside a space (spec 6.6)

An edge may only connect two nodes with the same `space_id`: both personal/org, or both in the same space.

- In the tools: on a space write, any `related` / `related_to` target that is not a visible node of the same space returns `cross_space_edge` and nothing is created. On a personal write, a space node as the target returns `cross_space_edge`; an invisible target keeps the old warn-and-skip behaviour.
- In storage: `create_edge` raises `ValueError("cross_space_edge")` when the endpoints' spaces differ. This is the safety net.
- Auto-linking and `possible_conflicts` hints only consider candidates in the same space as the new memory.
- Deleting a node also removes edges in other buckets that pointed at it, so no dangling edges are left.

## Space lifecycle

| Tool | Notes |
|---|---|
| `space_create(name, slug=None, org_id=None, description=None, forgetting_policy="retain_all", llm_inference_enabled=True)` | The caller becomes the first owner. Without `slug`, one is generated from the name (lowercased, spaces to `-`, invalid characters removed, `-2`, `-3`... if taken). A given slug that is already used in the org returns `slug_taken`. `org_id` defaults to the caller's only org; with several orgs it is required. Services can't create spaces. Returns `{"space": {..., "your_role": "owner"}}`. |
| `space_list(include_archived=False)` | The caller's spaces, sorted by name, each with `your_role`, `member_count`, `memory_count`. |
| `space_get(space)` | Reader+. Adds `member_count`, `memory_count`, `your_role`. |
| `space_update(space, name=None, description=None, forgetting_policy=None, llm_inference_enabled=None)` | Owner. The slug can't be changed. |
| `space_member_add(space, principal_id, principal_type, role)` | Owner. Adding an existing member changes their role. A service can't be `owner` (`invalid_arguments`). Returns `{"member": {...}, "created": bool}`. |
| `space_member_remove(space, principal_id)` | Owner. Removing the last user owner returns `last_owner`. Removing a non-member returns `{"removed": true}`. |
| `space_member_list(space)` | Reader+. `{"members": [{principal_id, principal_type, role, added_by, added_at}]}`. |
| `space_archive(space)` | Owner. Sets `archived_at`. The space leaves default recall, stays readable when named, and refuses writes. Archiving an archived space returns `space_archived`. |
| `space_unarchive(space)` | Owner. Clears `archived_at`. On an active space it returns the space unchanged. |
| `space_erase(space, confirm_slug, dry_run=False)` | Owner. Permanently deletes the space and every memory and edge in it. `confirm_slug` must equal the slug exactly (`confirmation_mismatch`). With `dry_run=true` it returns the counts and deletes nothing. Returns `{"erased": {"space_id", "nodes", "edges", "members"}, "dry_run": bool}`. The space id is never reused. Audit events about the space are kept. |
| `promote_to_space(node_id, space, keep_original=True)` | Writer+ in the target space. Copies one of your own personal memories into the space: content, summary, category, refs, tags, creation time, source agent and session. The copy records `promoted_from`. Edges are not copied. With `keep_original=false` the personal original is deleted. Returns `{"node_id", "space"}`. |

Space object shape: `{"id", "slug", "name", "org_id", "description", "forgetting_policy", "llm_inference_enabled", "archived", "archived_at", "created_at", "created_by"}`, plus `your_role` where the caller is a member.

Membership changes take effect on the member's next call. The stdio server reloads the caller's roles from the registry before every tool call.

### Where spaces are stored locally

The in-memory registry is written to a sidecar file next to the graph state file: `GENESYS_PERSIST_PATH=memories.json` gives `memories.spaces.json`, shaped `{"version": 1, "spaces": {...}, "members": [...]}`. Space memories live in the graph file under `space::<id>` keys. Audit events are not persisted by the in-memory sink.

## Background work (spec 6.9)

| Component | Rule |
|---|---|
| Forgetting | A `retain_all` space (the default) never has anything pruned. A `standard` space uses the normal rules, with orphan status counted only from edges inside the space. With no registry, or an unknown or erased space, nothing is pruned. |
| Contradiction detection | Compares only memories in the same space. No LLM call when the space has `llm_inference_enabled=false`. |
| Consolidation | Never merges across spaces, or a space memory with a personal one. No LLM call when `llm_inference_enabled=false`. |
| Causal inference | Candidates come from the same space only. Skipped when `llm_inference_enabled=false`. |
| Entity extraction and category classification | Skipped when `llm_inference_enabled=false`. |
| Core promotion | Hub score and schema match use only the memory's own scope. |
| Maintenance | The stdio server's periodic pass runs a personal pass and then one pass per space with that space's scope. A user pass never rescores space memories, and a space pass never touches personal ones. |
| Embeddings | Still computed for space memories, because recall needs them. The embedding provider sees memory text. See below. |

### Embedding providers see memory text

Every stored memory is embedded, including memories in spaces with `llm_inference_enabled=false`. That flag stops LLM calls; it does not stop embedding. With `GENESYS_EMBEDDER=openai` (the default), memory text is sent to OpenAI. For sensitive spaces, run with `GENESYS_EMBEDDER=local` (`pip install 'papez[local]'`), which computes embeddings on your machine with sentence-transformers.

## References and tags

Every memory can carry `refs` (where it came from) and `tags` (labels).

**Ref:** `{"type", "id", "url"?, "label"?}`.

- `type` matches `^[a-z0-9][a-z0-9_.-]{0,63}$`, e.g. `document`, `meeting`, `crm.contact`.
- `id` is 1-256 characters, `label` up to 200, `url` up to 2048 and must start with `http://` or `https://`.
- Two refs with the same `(type, id)` are duplicates; the first is kept.
- At most `GENESYS_MAX_REFS_PER_MEMORY` (20) refs. A bad ref returns `invalid_ref`; too many returns `too_many_refs`.

**Tag normalization:**

1. Trim whitespace and lowercase.
2. Replace runs of whitespace with a single `-`.
3. Allowed characters: `a-z`, `0-9`, `-`, `_`, `.`, `:`. Anything else returns `invalid_tag`.
4. Drop empty tags.
5. Remove duplicates, keeping the first occurrence.
6. A tag over `GENESYS_MAX_TAG_LENGTH` (50) returns `invalid_tag`; more than `GENESYS_MAX_TAGS_PER_MEMORY` (20) returns `too_many_tags`.

`" Manual Payments "` becomes `manual-payments`, `"Bank A"` becomes `bank-a`, and `"q2/2026"` is rejected.

On any validation failure the tool creates nothing.

**`memory_amend`:** with `refs=None` / `tags=None` the new memory copies the old one's refs and tags; a list replaces them exactly.

**Filters:**

| Where | Key | Matches when |
|---|---|---|
| `memory_recall` | `tags_any` | The memory has at least one of the tags. Applied before ranking, so `k` results still come back when enough match. |
| `memory_recall` | `ref_type` | The memory has a ref of this type |
| `memory_search` `filters` | `tags_any` | At least one of the tags |
| `memory_search` `filters` | `tags_all` | All of the tags |
| `memory_search` `filters` | `ref_type` | A ref of this type |
| `memory_search` `filters` | `ref_id` | A ref with this id. With `ref_type`, both must match on the same ref. |

Filter tag values are normalized the same way as stored tags.

**Output:** full recall hits, search hits and `memory_explain` carry `refs`, `tags` and `space`. Concise recall hits carry `tags`, `ref_count` and `space`.

## Idempotent writes

### `external_id` and `on_existing`

`memory_store(..., external_id=None, on_existing="amend")`. `external_id` (max 256 characters) is the caller's own id for the memory. It is unique among current (not superseded) memories within a space, or within the user's personal memory. When a memory is amended, the id moves to the new node; the superseded node keeps it for history.

| `on_existing` | Content identical? | Result |
|---|---|---|
| `skip` | either | `{"node_id": "<existing>", "status": "exists"}`. Nothing changes. |
| `amend` (default) | yes | `{"node_id": "<existing>", "status": "unchanged"}`. Nothing changes. |
| `amend` | no | Same as `memory_amend` on the existing memory: `{"node_id": "<new>", "status": "amended", "superseded": "<old>", "visibility"}`. |
| `error` | either | `external_id_exists`, with the existing `node_id`. |

"Identical" means the same content after trimming whitespace, the same set of refs (compared by `type`, `id`, `url`, `label`) and the same set of tags.

### `memory_store_batch(items, space=None, on_existing="amend")`

- Up to `GENESYS_BATCH_MAX_ITEMS` (100) items, all into one space, or into personal memory when `space` is omitted. Services must name a space.
- Item keys: `content`, `refs`, `tags`, `category`, `external_id`, `related`, `source_session`, `created_at`. Any other key (for example `visibility` or `org_id`) is an item error with `invalid_arguments`.
- A `related` entry is `{"id", "type"}` for an existing memory in the same space, or `{"ref_index", "type"}` for an earlier item in the same batch. `ref_index` must be an integer smaller than the item's own index.
- Every item is validated first. If any item is invalid, nothing is stored and the call returns `{"error": "<n> items are invalid", "code": "batch_invalid", "items": [{"index", "code", "error"}]}`. Under `on_existing="error"`, an `external_id` that already exists, or that appears twice in the batch, counts as an item error.
- Items are then written in order. The result is `{"results": [{"index", "node_id", "status", "superseded"?}]}`.
- `items` that is not a list, more than the maximum, or an invalid `on_existing` returns `invalid_arguments` for the whole call. An empty list returns `{"results": []}`.
- With `on_existing="amend"` and the same `external_id` twice in one batch, the later item amends the earlier one.

### `memory_retire_by_ref(ref_type, ref_id, space=None, mode="amend")`

For when a source document is deleted. It acts only on current memories (not superseded versions). Writer or owner when `space` is named; without `space` it acts on the caller's own personal memories. Services must name a space.

- `mode="amend"`: a memory whose only ref is this one is amended to `"[Retired] <summary> - source <label or id> was removed."`, tagged `retired`. It keeps the ref for provenance. A memory with other refs is amended with this ref removed. Memories already tagged `retired` are skipped, so calling it again does nothing.
- `mode="delete"`: memories whose only ref is this one are deleted; the ref is removed from the others.
- Returns `{"retired", "updated", "deleted"}`.

### Retries

The `retryable` flag on a failed tool call is `true` for:

- the read tools: `memory_recall`, `memory_search`, `memory_traverse`, `memory_explain`, `memory_stats`, `list_core_memories`, `space_list`, `space_get`, `space_member_list`;
- `memory_store_batch` when every item carries a non-empty `external_id`.

`memory_store` with an `external_id` is idempotent too, so a client can resend it safely; the server still reports `retryable: false` for it, because the flag is per tool and `memory_store` without an `external_id` is not safe to retry.

## Audit events

Every action that touches a space writes one event to the configured `AuditSink`. Personal actions write nothing. The in-memory sink keeps up to `GENESYS_AUDIT_MAX_EVENTS` (100000) events and does not persist them; the hosted product stores them in Postgres, append-only.

Event shape:

```json
{
  "id": "evt_...",
  "at": "2026-11-01T18:04:55.120Z",
  "space_id": "spc_...",
  "principal_id": "user_...",
  "principal_type": "user",
  "action": "memory.recall",
  "node_ids": ["..."],
  "detail": {"query_chars": 64, "results": 2},
  "client": {"transport": "stdio", "client_name": "claude-code"}
}
```

Events never contain memory content, query text, ref labels, space names or slugs. Only ids, counts and lengths.

| Action | When | `detail` |
|---|---|---|
| `memory.store` | A store into a space with status `stored`, `exists` or `unchanged` | `{status}` (+ `batch_index` for batches) |
| `memory.amend` | `memory_amend`, an `on_existing` amend, or a retire in amend mode; `node_ids` are `[new, superseded]` | varies; retire adds `{via: "retire_by_ref", mode}` |
| `memory.delete`, `memory.pin`, `memory.unpin` | Those writes on space memories | |
| `memory.promote_in` | `promote_to_space`; `node_ids` is `[new]` | `{promoted_from, keep_original}` |
| `memory.recall`, `memory.search` | One event per space in scope, including spaces with 0 results | `{query_chars, results}` |
| `memory.traverse`, `memory.explain` | One event for the start or target node's space (edges never cross spaces) | `{query_chars: 0, results}` |
| `space.create` | | `{org_id}` |
| `space.update` | Only when something changed | `{fields}` (names only, never values) |
| `space.archive`, `space.unarchive` | Unarchive of an active space emits nothing | `{}` |
| `space.erase` | Real erase only, never a dry run | `{nodes, edges, members}` |
| `member.add`, `member.role_change`, `member.remove` | | `{member_id, member_type, role, previous_role?}` |
| `access.denied` | `space_not_found`, `insufficient_role` or `space_archived` returned for a real space, logged against the real space id | `{code, need_role}` |

`node_ids` on read events are capped at `GENESYS_AUDIT_READ_NODE_IDS_CAP` (50); the `results` count is not capped. A probe for a space that doesn't exist logs nothing. A failing sink never breaks a tool call.

## Error codes (spec Section 10)

Every new error is `{"error": "<message>", "code": "<code>"}`. Existing errors keep their text and gained a `code`.

| Code | When |
|---|---|
| `invalid_arguments` | A bad combination of parameters: `space` and `spaces` together, `visibility="org"` with `space`, `promote_to_org` on a space node, a service as `owner`, an empty `content`, an unknown batch item key, invalid `org_id` or edge type |
| `invalid_ref` | A ref fails validation |
| `too_many_refs` / `too_many_tags` | Over the limits |
| `invalid_tag` | Disallowed characters or too long |
| `space_not_found` | The space doesn't exist or the caller isn't a member. The two look identical. |
| `space_archived` | A write to an archived space |
| `space_required` | A service wrote without `space` |
| `insufficient_role` | A member without the needed role, a service calling `space_create`, or a non-owner changing a personal memory |
| `last_owner` | Removing or demoting the last user owner |
| `slug_taken` | The slug is already used in that org |
| `principal_not_in_org` | Adding a user from another org. The library has no org directory, so it checks this only when the host sets `MCPToolHandler.principal_in_org`; the hosted product does. |
| `cross_space_edge` | An edge between different spaces, or between a space and personal memory |
| `confirmation_mismatch` | `space_erase` with the wrong `confirm_slug` |
| `external_id_exists` | `on_existing="error"` and the id exists |
| `batch_invalid` | One or more batch items are invalid, with per-item details |
| `not_found` | A node id that doesn't exist or isn't visible. The two look identical. |

Errors from the transport layer itself (an exception inside a tool, a missing required argument) keep the `{"error", "retryable"}` shape.
