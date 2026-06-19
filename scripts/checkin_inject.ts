#!/usr/bin/env tsx
/**
 * UserPromptSubmit hook — voice checkin + voice dispatch marker injector.
 *
 * Checks marker files in ~/.hermes/runtime/:
 *
 *   pending_checkin.<node>.json — per-agent check-in marker written by
 *                            POST /v1/voice/{sid}/checkin (main.py
 *                            _write_pending_checkin_marker). Prompts THIS node
 *                            to run /checkin.
 *   pending_checkin.json        — legacy GLOBAL check-in marker (written only for
 *                            hermes). Drained by the hermes session for
 *                            back-compat / cleanup.
 *
 *   pending_voice_dispatch.json — written by tool_dispatch_task_to_hermes()
 *                            when a user dispatches a task during a live call.
 *
 * #D12 (2026-06-19) — NODE-AWARE ROUTING. The plugin is installed at USER scope,
 * so EVERY Claude session runs this hook. Previously it read only the hardcoded
 * GLOBAL pending_checkin.json, so the first session to submit a prompt consumed
 * it — letting a non-hermes session (e.g. the C:\ control room) STEAL hermes's
 * check-in (observed 2026-06-19: control room ate sid voice-1781887777003).
 * Fix: resolve THIS session's node from cwd/transcript_path; read the per-agent
 * marker for that node; and NEVER consume a marker from a session that resolves
 * to no known node. Backend already writes the per-agent files — reader-side only.
 *
 * Output shape: plain text to stdout; nothing to stderr on normal paths.
 * Fail-soft: any error → exit 0. Never breaks the prompt pipeline.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';

// Test seam: HERMES_RUNTIME_DIR overrides the runtime dir (defaults to the real
// path so production behavior is unchanged). Used by the D12 isolated test.
const RUNTIME_DIR = process.env.HERMES_RUNTIME_DIR || path.join(os.homedir(), '.hermes', 'runtime');
const TTL_SECONDS = 86400; // 24 h
const KNOWN_AGENTS = new Set(['hermes', 'atlas', 'daedalus']);

interface HookInput {
  session_id?: string;
  cwd?: string;
  prompt?: string;
  transcript_path?: string;
}

interface CheckinMarker {
  sid: string;
  task_count: number;
  turn_count?: number;
  ts: number;
  consumed: boolean;
  tmux_injected?: boolean;
  agent?: string;
}

interface DispatchMarker {
  sid: string;
  task: string;
  task_id: string;
  ts: number;
}

/**
 * #D12 — Resolve THIS session's node from the hook input (cwd, then
 * transcript_path), using the same scheme the steward uses. Returns a known
 * agent slug (hermes/atlas/daedalus) or null. null => this session owns no node
 * and MUST NOT consume any node's marker.
 */
function resolveNode(input: HookInput | null): string | null {
  if (!input) return null;
  for (const raw of [input.cwd, input.transcript_path]) {
    if (!raw) continue;
    const s = String(raw).replace(/\\/g, '/').toLowerCase();
    // Direct path form: .../claude-nodes/<node>/...
    let m = s.match(/claude-nodes\/([a-z0-9_-]+)/);
    if (m && KNOWN_AGENTS.has(m[1])) return m[1];
    // Claude Code project-dir form: ...C--Users-jbogi-claude-nodes-<node>...
    m = s.match(/claude-nodes-([a-z0-9_]+)/);
    if (m && KNOWN_AGENTS.has(m[1])) return m[1];
  }
  return null;
}

async function readHookInput(): Promise<HookInput | null> {
  return new Promise((resolve) => {
    let input = '';
    const rl = readline.createInterface({ input: process.stdin });
    rl.on('line', (line) => { input += line; });
    rl.on('close', () => {
      if (!input.trim()) { resolve(null); return; }
      try { resolve(JSON.parse(input)); } catch { resolve(null); }
    });
    setTimeout(() => { rl.close(); }, 1000);
  });
}

/** Consume a marker atomically. Returns the renamed path or null on failure. */
function consumeMarker(markerPath: string): string | null {
  const consumedPath = markerPath.replace('.json', `.consumed-${Date.now()}.json`);
  try {
    fs.renameSync(markerPath, consumedPath);
    return consumedPath;
  } catch {
    return null;
  }
}

function checkinBanner(marker: CheckinMarker): string {
  const n = marker.task_count || 0;
  const turns = marker.turn_count || 0;
  const sid = marker.sid || 'unknown';
  const turnSuffix = turns > 0 ? ` (${turns} turn${turns === 1 ? '' : 's'})` : '';
  const taskWord = n === 1 ? 'task' : 'tasks';
  return n > 0
    ? `[Check-in — merge voice transcript for seamless context, check ${n} queued ${taskWord} — ${sid}${turnSuffix}] run /checkin`
    : `[Check-in — merge voice transcript for seamless context, no tasks — ${sid}${turnSuffix}] run /checkin`;
}

async function main(): Promise<void> {
  // Always drain stdin first to keep the hook protocol happy.
  const input = await readHookInput();
  const node = resolveNode(input);

  const parts: string[] = [];

  // #D12: a session that resolves to no known node (e.g. the C:\ control room)
  // must NEVER consume a node's check-in / dispatch markers. Bail early.
  if (node === null) {
    return;
  }

  // ── 1. Check-in marker (per-agent; hermes also drains the legacy global) ────
  try {
    const perAgent = path.join(RUNTIME_DIR, `pending_checkin.${node}.json`);
    // Only hermes touches the legacy global (back-compat + cleanup). Peer nodes
    // never read it, so they can't steal hermes's check-in.
    const candidates = node === 'hermes'
      ? [perAgent, path.join(RUNTIME_DIR, 'pending_checkin.json')]
      : [perAgent];

    let emittedSid: string | null = null;
    for (const mp of candidates) {
      if (!fs.existsSync(mp)) continue;
      let marker: CheckinMarker | null = null;
      try { marker = JSON.parse(fs.readFileSync(mp, 'utf-8')) as CheckinMarker; } catch { /* unreadable */ }
      if (!marker) continue;

      const now = Date.now() / 1000;
      const stale = !marker.ts || (now - marker.ts) > TTL_SECONDS;
      if (stale) {
        try { fs.renameSync(mp, mp.replace('.json', `.stale-${Date.now()}.json`)); } catch { /* best-effort */ }
        continue;
      }
      // Atomically consume before building output.
      const consumed = consumeMarker(mp);
      if (consumed === null) continue;
      // A live tmux pane already surfaced it (D11.1) — don't double-surface.
      if (marker.tmux_injected === true) continue;
      // Backend writes BOTH per-agent + global for hermes (same sid) — emit once.
      if (emittedSid !== null && emittedSid === marker.sid) continue;
      parts.push(checkinBanner(marker));
      emittedSid = marker.sid;
    }
  } catch { /* silent no-op */ }

  // ── 2. Voice-dispatch marker (per-agent; hermes also drains the legacy global) ─
  try {
    const perAgent = path.join(RUNTIME_DIR, `pending_voice_dispatch.${node}.json`);
    const candidates = node === 'hermes'
      ? [perAgent, path.join(RUNTIME_DIR, 'pending_voice_dispatch.json')]
      : [perAgent];

    let emittedSid: string | null = null;
    for (const mp of candidates) {
      if (!fs.existsSync(mp)) continue;
      let dm: DispatchMarker | null = null;
      try { dm = JSON.parse(fs.readFileSync(mp, 'utf-8')) as DispatchMarker; } catch { /* unreadable */ }
      if (!dm) continue;

      const now = Date.now() / 1000;
      const stale = !dm.ts || (now - dm.ts) > TTL_SECONDS;
      if (stale) {
        try { fs.renameSync(mp, mp.replace('.json', `.stale-${Date.now()}.json`)); } catch { /* best-effort */ }
        continue;
      }
      const consumed = consumeMarker(mp);
      if (consumed === null) continue;
      if (emittedSid !== null && emittedSid === dm.sid) continue;
      const sid = dm.sid || 'unknown';
      const task = dm.task || '(no task text)';
      const taskId = dm.task_id || '?';
      parts.push(`[Voice task — ${sid}] Jacob asked (via voice): "${task}" (task_id=${taskId}). Pick it up.`);
      emittedSid = dm.sid;
    }
  } catch { /* silent no-op */ }

  // ── 3. Emit all banners (if any) ──────────────────────────────────────────
  if (parts.length > 0) {
    process.stdout.write(parts.join('\n') + '\n\n');
  }
}

main().catch(() => { process.exit(0); });
