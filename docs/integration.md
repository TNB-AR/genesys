# Integrating another system with a Mind Space

This guide shows how a system that holds engagement knowledge (a document store, a CRM, a meeting-notes tool, a ticketing system) feeds a Papez Mind Space, and how people and agents then use it. Tool names and parameters are the ones shipped in papez 0.7.0. The full behaviour reference is [`mind-spaces.md`](mind-spaces.md).

Service credentials, REST endpoints and OAuth belong to the hosted product at `api.papez.ai`; see [papez.ai/developers](https://papez.ai/developers) for their exact shape. The open-source stdio server runs as a single local user, so you can try every step below locally except the service credential.

## 1. The pieces

```
  Source systems                      Papez (hosted)                          People and agents
  --------------                      --------------                          -----------------
  Document store --+                                                      +-- Consultant's AI assistant
  Meeting notes  --+-- service key -> Space "Client A - Payments" <- OAuth (MCP) -+-- Consultant's AI assistant
  CRM / tickets  --+   (writer, one   - memories with refs and tags              +-- New hire (reader at first)
                        space each)   - causal graph and history
                                      - audit log
```

## 2. Set up a space (once per engagement)

An engagement lead (a user) does this in the dashboard or through MCP:

1. **Create the space.**

   ```json
   {"tool": "space_create",
    "arguments": {"name": "Client A - Payments", "slug": "client-a-payments",
                  "forgetting_policy": "retain_all", "llm_inference_enabled": false}}
   ```

   Pass `llm_inference_enabled: false` if the engagement's data must not be sent to an LLM. Memories are still embedded; if the embedding provider must not see the text either, run the server with a local embedder (`GENESYS_EMBEDDER=local`). The creator is the first owner.

2. **Add the team.** `space_member_add` for each consultant (`writer`) and for anyone who only needs to look things up (`reader`):

   ```json
   {"tool": "space_member_add",
    "arguments": {"space": "client-a-payments", "principal_id": "user_7a2", "principal_type": "user", "role": "writer"}}
   ```

3. **Create a service credential** for each source system in the hosted dashboard, e.g. "Document sync - Client A", and store the key in that system's secret store.

4. **Grant it access** with `space_member_add` (or the hosted REST equivalent), `principal_type: "service"`, `role: "writer"`. A service can never be an owner.

## 3. Initial import (once)

The source system sends what it already knows, in batches of up to `GENESYS_BATCH_MAX_ITEMS` (100):

```json
{"tool": "memory_store_batch",
 "arguments": {
   "space": "client-a-payments",
   "on_existing": "amend",
   "items": [
     {"external_id": "doc_8812#kp1",
      "content": "Manual payments of $10,000 or more ship overnight via FedEx Priority. Same-day requests need manager approval.",
      "tags": ["manual-payments", "shipping", "bank-a"],
      "category": "process_rule",
      "refs": [{"type": "document", "id": "doc_8812", "label": "Manual Payments Mail Matrix.pdf",
                "url": "https://docs.example.com/d/doc_8812"}]},
     {"external_id": "mtg_2026_10_14#d1",
      "content": "Client agreed to pilot with Bank A first; Bank B is out of scope until Q2 2027.",
      "tags": ["pilot", "scope", "bank-a", "bank-b"],
      "category": "decision",
      "refs": [{"type": "meeting", "id": "mtg_2026_10_14", "label": "Payments process review, Oct 14"}],
      "related": [{"ref_index": 0, "type": "related_to"}]}
   ]}}
```

- Every item is validated before anything is written. If any item is invalid, nothing is stored and the response is `batch_invalid` with one `{index, code, error}` per problem.
- `related` can point at an existing memory in the same space (`{"id", "type"}`) or at an earlier item in the batch (`{"ref_index", "type"}`).
- The response has one result per item, in order: `{"results": [{"index": 0, "node_id": "...", "status": "stored"}, ...]}`.
- A service must always name `space`; without it the call returns `space_required`.

**Choosing `external_id`:** make it stable and derived from the source, like `<source id>#<item key>`. Then re-running the import is safe, and a changed source item updates its memory instead of duplicating it.

## 4. Keep it in sync (ongoing)

| Event in the source system | Call |
|---|---|
| New item created | `memory_store(content, ..., space="client-a-payments", external_id="<new id>")`, or a batch |
| Item changed | The same call with the **same** `external_id` and `on_existing="amend"`. Papez supersedes the old memory and keeps history: `status: "amended"`, `superseded: "<old id>"`. |
| Item re-sent unchanged (retry, re-import) | The same call. Papez returns `status: "unchanged"` and does nothing. |
| Source item deleted | `memory_retire_by_ref(ref_type="document", ref_id="doc_8812", space="client-a-payments")`. Memories citing only that source become `[Retired] ...` notes tagged `retired`; memories with other sources just lose that ref. `mode="delete"` deletes instead. |
| Network error or timeout | **Retry the same call.** It is idempotent because every item has an `external_id`. For `memory_store_batch` the server marks the failure `retryable: true` when every item has one. |

Without `external_id`, a resend of `memory_store` or `memory_store_batch` creates duplicates. Don't retry those blindly.

## 5. People and agents use it

A consultant adds the Papez MCP connector to their AI assistant and signs in. Their assistant can then:

- **Recall within one engagement:** `memory_recall(query="what did we decide about the Bank B pilot?", space="client-a-payments")`. Each hit carries its `refs`, `tags` and `space`, and superseded versions are marked with `superseded_by`. Naming a space excludes personal memory unless you also pass `include_personal=true`.
- **Filter by label or source:** `memory_recall(..., tags_any=["bank-a"])`, or `memory_search(query, filters={"ref_type": "document", "ref_id": "doc_8812"})`.
- **Add knowledge:** `memory_store(content="Their payments ops lead owns lien validation (from today's call).", space="client-a-payments")`. Writers and owners only.
- **Correct a fact:** `memory_amend(node_id, content, reason=...)` on the outdated memory. The new version stays in the same space and history is kept.
- **Look across engagements:** omit `space`. Recall then covers personal memory plus every non-archived space the user belongs to, and each result says which space it came from.

**Recommended agent instruction** for teams that want agents to stay inside one engagement: "When working on a client, always pass `space` on every Papez call."

## 6. Offboarding

| When | Action |
|---|---|
| A team member leaves the engagement | `space_member_remove(space, principal_id)`. Access ends on their next call. Their contributions stay. |
| A team member leaves the company | `space_member_remove` from every space, then erase their **personal** memory (`MCPToolHandler.erase_user` in the library; the hosted product exposes account deletion). Space memories they wrote are kept, with the author replaced by `erased_user`. |
| The engagement ends | `space_archive(space)`. The space becomes read-only and leaves default recall; members can still read it by naming it. Revoke the engagement's service credentials. |
| The contract requires deletion | `space_erase(space, confirm_slug="client-a-payments", dry_run=true)` to see the counts, then the same call with `dry_run=false`. Export the audit log first if the client wants a record. Audit events survive the erase. |

## What gets audited

Every action above that touches the space writes an audit event: stores, amends, deletes, pins, promotions, one read event per space for each recall or search, lifecycle and membership changes, and `access.denied` for refused calls against a real space. Events hold ids, counts and lengths only, never memory content or query text. The hosted product exports them per space; see [`mind-spaces.md`](mind-spaces.md#audit-events) for the catalogue.
