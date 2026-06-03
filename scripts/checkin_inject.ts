#!/usr/bin/env tsx
/**
 * UserPromptSubmit hook — voice checkin + voice dispatch marker injector.
 *
 * Checks TWO marker files in ~/.hermes/runtime/:
 *
 *   pending_checkin.json   — written by POST /v1/voice/{sid}/checkin after a
 *                            call ends.  Prompts Hermes to run /checkin.
 *
 *   pending_voice_dispatch.json — written by tool_dispatch_task_to_hermes()
 *                            (in voice_tools.py) when a user dispatches a task
 *                            during a live call.  Prompts Hermes to pick up the
 *                            task immediately — same visibility as check-in.
 *
 * Both markers follow the same atomic-consume pattern (rename before emit) and
 * the same 24 h TTL.  Both emit plain text to stdout that Claude Code prepends
 * to the user's prompt context.
 *
 * Output shape: plain text to stdout; nothing to stderr on normal paths.
 * Fail-soft: any error → exit 0. Never breaks the prompt pipeline.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';

const RUNTIME_DIR = path.join(os.homedir(), '.hermes', 'runtime');
const MARKER_PATH = path.join(RUNTIME_DIR, 'pending_checkin.json');
const DISPATCH_MARKER_PATH = path.join(RUNTIME_DIR, 'pending_voice_dispatch.json');
const TTL_SECONDS = 86400; // 24 h

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
}

interface DispatchMarker {
  sid: string;
  task: string;
  task_id: string;
  ts: number;
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

async function main(): Promise<void> {
  // Always drain stdin first to keep the hook protocol happy.
  await readHookInput();

  const parts: string[] = [];

  // ── 1. Check-in marker ─────────────────────────────────────────────────────
  try {
    if (fs.existsSync(MARKER_PATH)) {
      let marker: CheckinMarker | null = null;
      try {
        const raw = fs.readFileSync(MARKER_PATH, 'utf-8');
        marker = JSON.parse(raw) as CheckinMarker;
      } catch { /* unreadable — skip */ }

      if (marker) {
        const now = Date.now() / 1000;
        const stale = !marker.ts || (now - marker.ts) > TTL_SECONDS;
        if (stale) {
          try { fs.renameSync(MARKER_PATH, MARKER_PATH.replace('.json', `.stale-${Date.now()}.json`)); } catch { /* best-effort */ }
        } else {
          // Atomically consume before building output.
          const consumed = consumeMarker(MARKER_PATH);
          if (consumed !== null && marker.tmux_injected !== true) {
            const n = marker.task_count || 0;
            const turns = marker.turn_count || 0;
            const sid = marker.sid || 'unknown';
            const turnSuffix = turns > 0 ? ` (${turns} turn${turns === 1 ? '' : 's'})` : '';
            const taskWord = n === 1 ? 'task' : 'tasks';
            const banner = n > 0
              ? `[Check-in — merge voice transcript for seamless context, check ${n} queued ${taskWord} — ${sid}${turnSuffix}] run /checkin`
              : `[Check-in — merge voice transcript for seamless context, no tasks — ${sid}${turnSuffix}] run /checkin`;
            parts.push(banner);
          }
        }
      }
    }
  } catch { /* silent no-op */ }

  // ── 2. Voice-dispatch marker ───────────────────────────────────────────────
  try {
    if (fs.existsSync(DISPATCH_MARKER_PATH)) {
      let dm: DispatchMarker | null = null;
      try {
        const raw = fs.readFileSync(DISPATCH_MARKER_PATH, 'utf-8');
        dm = JSON.parse(raw) as DispatchMarker;
      } catch { /* unreadable — skip */ }

      if (dm) {
        const now = Date.now() / 1000;
        const stale = !dm.ts || (now - dm.ts) > TTL_SECONDS;
        if (stale) {
          try { fs.renameSync(DISPATCH_MARKER_PATH, DISPATCH_MARKER_PATH.replace('.json', `.stale-${Date.now()}.json`)); } catch { /* best-effort */ }
        } else {
          const consumed = consumeMarker(DISPATCH_MARKER_PATH);
          if (consumed !== null) {
            const sid = dm.sid || 'unknown';
            const task = dm.task || '(no task text)';
            const taskId = dm.task_id || '?';
            const banner =
              `[Voice task — ${sid}] Jacob asked (via voice): "${task}" ` +
              `(task_id=${taskId}). Pick it up.`;
            parts.push(banner);
          }
        }
      }
    }
  } catch { /* silent no-op */ }

  // ── 3. Emit all banners (if any) ──────────────────────────────────────────
  if (parts.length > 0) {
    process.stdout.write(parts.join('\n') + '\n\n');
  }
}

main().catch(() => { process.exit(0); });
