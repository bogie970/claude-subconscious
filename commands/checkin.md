---
description: Voice Check-in Procedure — process queued voice tasks. Run when you see [Check-in: N voice task(s) queued from voice-{sid}] or when manually invoked.
---

# /checkin — Voice Check-in Procedure

Run this when you see `[Check-in: N voice task(s) queued from voice-{sid}]` prepended to a prompt, or when manually invoked.

1. Parse the `sid` from the bracket header if present (e.g. `voice-1748389200000`).
2. `GET /v1/voice-sessions/{sid}/turns` (auth: bearer LCW token) and INTERNALIZE those turns — they are the voice conversation just had; treat them as conversation context.
3. Resolve **this node's agent slug** — it is the name of the node you are running as (one of `hermes`, `atlas`, `daedalus`). Use the node directory under `claude-nodes/` (the leaf of your cwd path, e.g. `…/claude-nodes/hermes` → `hermes`); if you cannot compute it, default to `hermes`. Then `GET /v1/voice-tasks?agent=<this-node-agent>&state=queued` (and a second call with `state=in_progress`), auth same. **Scope by your own agent so you only claim tasks that this node owns** — a voice task dispatched to another agent (e.g. voice-Atlas) carries `agent='atlas'` and must NOT be pulled into this node's checkin.
4. For each task:
   - If autonomously safe (search L2, save memory, read project doc): execute, then `POST /v1/voice-tasks/{id}/done` with body `{"notes": "<what you did>"}`.
   - If requires Jacob's input or external write: `POST /v1/voice-tasks/{id}/start`, surface explicitly to Jacob.
5. Close the inbox thread: `mcp__node-messenger__complete_task(task_id=<inbox mid for this sid>)`.
6. Print a clear summary block to Jacob (done / in-progress / needs-you).
