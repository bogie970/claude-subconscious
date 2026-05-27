#!/usr/bin/env npx tsx
/**
 * Stop hook — steward dispatch (S1, async).
 *
 * This is the per-turn TRIGGER half of the LOCKED async execution model
 * (steward.md §3, 2026-05-27). The steward's COMPUTE never runs in a turn's
 * critical path — it runs HERE, in the async Stop hook, AFTER the agent's
 * turn has completed. It:
 *   1. Reads the transcript JSONL (hook stdin gives the path)
 *   2. Diffs against last-dispatched index (own state file, independent of the
 *      subconscious sync state)
 *   3. Builds the steward payload schema (runner.py contract: transcript as
 *      [{role, content, turn_id}], trigger, tool_outputs, active_plan, events)
 *   4. Coalesces with any existing unprocessed steward payload for the session
 *   5. Spawns `python -m lcw.steward.runner <payload>` detached (fire-and-forget)
 *
 * The runner compiles and atomic-writes ~/.hermes/runtime/steward_window.json.
 * In SHADOW mode the agent NEVER reads that window — steward_inject.py only
 * writes a comparison file. So this dispatch firing changes NOTHING the agent
 * sees; it just makes the steward actually RUN per-turn (which it does not
 * today — confirmed: no Stop/UserPromptSubmit hook calls the runner).
 *
 * Mirrors send_messages_to_letta.ts for payload-build + buildPythonSubprocessEnv
 * + coalesce + detached spawn. Fail-open: every error path exits 0.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import {
  buildPythonSubprocessEnv,
  getMode,
  getTempStateDir,
  readBoundedStdinJson,
  recordHookError,
} from './conversation_utils.ts';
import {
  readTranscript,
  extractAllContent,
} from './transcript_utils.ts';
import { getConfig } from './config.ts';

const hermesConfig = getConfig();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'dispatch_steward.log');

// Coalesce: keep pending steward payloads at 1 per session (mirror sender).
const MAX_PAYLOAD_BYTES = 5 * 1024 * 1024;  // 5 MB

interface HookInput {
  session_id: string;
  transcript_path: string;
  stop_hook_active?: boolean;
  cwd: string;
  hook_event_name?: string;
}

interface StewardTurn {
  role: string;
  content: string;
  turn_id: number;
}

interface StewardPayload {
  turn_id: number;
  trigger: string;
  transcript: StewardTurn[];
  tool_outputs: unknown[];
  active_plan: unknown | null;
  events: unknown[];
  // dispatch bookkeeping (ignored by runner.py — it only reads the keys above)
  sessionId: string;
  newLastProcessedIndex: number;
}

function ensureLogDir(): void {
  if (!fs.existsSync(TEMP_STATE_DIR)) {
    fs.mkdirSync(TEMP_STATE_DIR, { recursive: true });
  }
}

const LOG_MAX_BYTES = 10 * 1024 * 1024;

function rotateLogIfNeeded(): void {
  try {
    ensureLogDir();
    if (fs.existsSync(LOG_FILE)) {
      const size = fs.statSync(LOG_FILE).size;
      if (size > LOG_MAX_BYTES) {
        const rotated = LOG_FILE + '.1';
        if (fs.existsSync(rotated)) {
          try { fs.unlinkSync(rotated); } catch { /* ignore */ }
        }
        fs.renameSync(LOG_FILE, rotated);
      }
    }
  } catch {
    // never let log rotation crash the hook
  }
}
rotateLogIfNeeded();

function log(message: string): void {
  try {
    ensureLogDir();
    const timestamp = new Date().toISOString();
    fs.appendFileSync(LOG_FILE, `[${timestamp}] ${message}\n`);
  } catch { /* logging must never throw */ }
}

// --- own dispatch state (independent of subconscious sync state) -------------

function stateFileFor(sessionId: string): string {
  return path.join(TEMP_STATE_DIR, `steward-dispatch-${sessionId}.json`);
}

function loadLastIndex(sessionId: string): number {
  try {
    const f = stateFileFor(sessionId);
    if (fs.existsSync(f)) {
      const v = JSON.parse(fs.readFileSync(f, 'utf-8'));
      const n = Number(v?.lastProcessedIndex);
      if (Number.isFinite(n) && n >= -1) return n;
    }
  } catch { /* fall through */ }
  return -1;
}

function saveLastIndex(sessionId: string, idx: number): void {
  try {
    ensureLogDir();
    const f = stateFileFor(sessionId);
    const tmp = f + '.tmp.' + process.pid;
    fs.writeFileSync(tmp, JSON.stringify({ lastProcessedIndex: idx }), 'utf-8');
    fs.renameSync(tmp, f);
  } catch { /* best effort */ }
}

// --- coalesce (mirror send_messages_to_letta.ts) -----------------------------

function findCoalescePayload(sessionId: string): string | null {
  try {
    const entries = fs.readdirSync(TEMP_STATE_DIR);
    const prefix = `steward-payload-${sessionId}-`;
    const candidates = entries
      .filter(e => e.startsWith(prefix) && e.endsWith('.json'))
      .map(e => path.join(TEMP_STATE_DIR, e));
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
    return candidates[0];
  } catch {
    return null;
  }
}

function coalesceInto(existingPath: string, fresh: StewardPayload): boolean {
  try {
    const existing: StewardPayload = JSON.parse(fs.readFileSync(existingPath, 'utf-8'));
    const mergedTranscript = (existing.transcript || []).concat(fresh.transcript || []);
    const merged: StewardPayload = {
      // The runner compiles the LATEST view, so newer scalars win.
      turn_id: Math.max(existing.turn_id || 0, fresh.turn_id || 0),
      trigger: fresh.trigger,
      transcript: mergedTranscript,
      tool_outputs: (existing.tool_outputs || []).concat(fresh.tool_outputs || []),
      active_plan: fresh.active_plan ?? existing.active_plan ?? null,
      events: (existing.events || []).concat(fresh.events || []),
      sessionId: existing.sessionId,
      newLastProcessedIndex: Math.max(existing.newLastProcessedIndex, fresh.newLastProcessedIndex),
    };
    if (Buffer.byteLength(JSON.stringify(merged), 'utf-8') > MAX_PAYLOAD_BYTES) {
      log(`Coalesce refused — merged payload exceeds ${MAX_PAYLOAD_BYTES} bytes`);
      return false;
    }
    const tmp = existingPath + '.tmp.' + process.pid;
    fs.writeFileSync(tmp, JSON.stringify(merged), 'utf-8');
    fs.renameSync(tmp, existingPath);
    return true;
  } catch (e) {
    log(`Coalesce failed: ${e instanceof Error ? e.message : String(e)}`);
    return false;
  }
}

// --- transcript -> steward turns ---------------------------------------------
// runner.py wants [{role, content, turn_id}]. We collapse each transcript
// message to its plain text (user prompts + assistant text). turn_id is a
// monotonic index over emitted turns — the steward uses it for recency/load.

function buildTurns(messages: any[], startIndex: number): StewardTurn[] {
  const turns: StewardTurn[] = [];
  let turnId = startIndex + 1;
  for (let i = startIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    if (msg?.type !== 'user' && msg?.type !== 'assistant') continue;
    const extracted = extractAllContent(msg);
    if (!extracted.text) continue;
    turns.push({
      role: msg.type === 'user' ? 'user' : 'assistant',
      content: extracted.text,
      turn_id: turnId++,
    });
  }
  return turns;
}

async function readHookInput(): Promise<HookInput> {
  const v = await readBoundedStdinJson<HookInput>(30000);
  if (!v) throw new Error('empty or oversized stdin');
  return v;
}

async function main(): Promise<void> {
  log('='.repeat(60));
  log('dispatch_steward started');

  // Honor the same global kill-switch as the rest of the pipeline.
  const mode = getMode();
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }

  try {
    const hookInput = await readHookInput();
    log(`Hook input: session=${hookInput.session_id} transcript=${hookInput.transcript_path}`);

    if (hookInput.stop_hook_active) {
      log('Stop hook already active, exiting to prevent loop');
      process.exit(0);
    }

    const messages = await readTranscript(hookInput.transcript_path, log);
    log(`Found ${messages.length} messages in transcript`);
    if (messages.length === 0) {
      process.exit(0);
    }

    const lastIndex = loadLastIndex(hookInput.session_id);
    const turns = buildTurns(messages, lastIndex);
    if (turns.length === 0) {
      log('No new turns since last dispatch — skipping');
      process.exit(0);
    }

    const newLastProcessedIndex = messages.length - 1;
    const payload: StewardPayload = {
      turn_id: turns[turns.length - 1].turn_id,
      trigger: 'stop_hook',
      transcript: turns,
      tool_outputs: [],
      active_plan: null,
      events: [],
      sessionId: hookInput.session_id,
      newLastProcessedIndex,
    };

    // Coalesce into an existing pending payload, else write a new one.
    let payloadFile: string;
    const coalesceTarget = findCoalescePayload(hookInput.session_id);
    if (coalesceTarget && coalesceInto(coalesceTarget, payload)) {
      payloadFile = coalesceTarget;
      log(`Coalesced into existing steward payload ${payloadFile}`);
    } else {
      payloadFile = path.join(
        TEMP_STATE_DIR,
        `steward-payload-${hookInput.session_id}-${Date.now()}.json`,
      );
      fs.writeFileSync(payloadFile, JSON.stringify(payload), 'utf-8');
      log(`Wrote NEW steward payload ${payloadFile} (${turns.length} turns)`);
    }

    // Spawn the runner detached: `python -m lcw.steward.runner <payload>`.
    // cwd MUST be hermesRoot so the `lcw` package resolves. Use the same
    // env-builder as the subconscious dispatch.
    const hermesRoot = hermesConfig.hermesRoot;
    const pythonCmd = hermesConfig.pythonPath;
    const workerEnv = buildPythonSubprocessEnv({
      PYTHONPATH: hermesRoot,
      HERMES_ROOT: hermesRoot,
    });

    const child = spawn(
      pythonCmd,
      ['-m', 'lcw.steward.runner', payloadFile],
      {
        detached: true,
        stdio: 'ignore',
        cwd: hermesRoot,
        env: workerEnv,
        windowsHide: true,
      },
    );
    child.unref();

    saveLastIndex(hookInput.session_id, newLastProcessedIndex);

    log(`Spawned steward runner (PID: ${child.pid})`);
    process.exit(0);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);
    if (error instanceof Error && error.stack) {
      log(`Stack: ${error.stack}`);
    }
    recordHookError('dispatch_steward.ts', error);
    process.exit(0);  // fail-open
  }
}

main().catch((e) => {
  try { recordHookError('dispatch_steward.ts', e); } catch {}
  process.exit(0);
});
