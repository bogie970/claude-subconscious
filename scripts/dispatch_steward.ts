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
import * as os from 'os';
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
  // runner.py uses this to load the full recent JSONL tail via
  // _load_recent_transcript_tail(), giving _pack_conversation the complete
  // recent conversation instead of just the incremental delta turns.
  transcript_path: string;
  // AGENT IDENTITY (the fix): which node this Stop hook fired in. runner.py
  // uses this to select BOTH the L1 transcript source and the output window
  // file (steward_window.<agent>.json). Without it every agent compiled as
  // "hermes" and only steward_window.json ever refreshed on a Stop hook —
  // peer windows (.atlas / .daedalus) only updated on the voice-checkout path.
  agent: string;
}

// The three peer nodes each run claude.exe rooted at
// C:/Users/jbogi/claude-nodes/<node>; their session JSONLs live under
// ~/.claude/projects/C--Users-jbogi-claude-nodes-<node>/ . Both the cwd and the
// transcript_path therefore carry the node slug. We derive it from whichever is
// available, validate against the known set, and fall back to 'hermes' ONLY if
// genuinely underivable (keeps the historical default-path behavior).
const KNOWN_AGENTS = new Set(['hermes', 'atlas', 'daedalus']);

function deriveAgent(cwd: string | undefined, transcriptPath: string | undefined): string {
  for (const raw of [cwd, transcriptPath]) {
    if (!raw) continue;
    const s = raw.replace(/\\/g, '/').toLowerCase();
    // Direct cwd form: .../claude-nodes/<node>/...
    let m = s.match(/claude-nodes\/([a-z0-9_-]+)/);
    if (m && KNOWN_AGENTS.has(m[1])) return m[1];
    // CC project-dir form: .../C--Users-jbogi-claude-nodes-<node>/...
    m = s.match(/claude-nodes-([a-z0-9_]+)/);
    if (m && KNOWN_AGENTS.has(m[1])) return m[1];
  }
  return 'hermes';
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
      // Fresh path wins: it's the same live JSONL but we want the newest ref.
      transcript_path: fresh.transcript_path || existing.transcript_path,
      // Agent identity is stable across a session — preserve it through coalesce
      // (fresh wins; both should agree since it's the same node/session).
      agent: fresh.agent || existing.agent || 'hermes',
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

// --- ping-not-spawn signal path (Step 1, flag-gated, fail-open) ---------------
// SYSTEM_REFINE Step 1 (steward.md §7 2026-06-06 LOCKED): instead of spawning a
// fresh `python -m lcw.steward.runner`, POST the SAME payload to the always-on
// backend, which runs the compile on a side thread. The mode is selected by
// STEWARD_DISPATCH (default "spawn" = today's behavior, ZERO change unless set
// to "signal"). When "signal", a POST failure/timeout/connection-refused
// FAILS-OPEN to the spawn path so the steward ALWAYS runs.

const STEWARD_BACKEND_URL = 'http://127.0.0.1:7777/v1/steward/signal';
const SIGNAL_TIMEOUT_MS = 1500;  // short — fire-and-forget; backend returns 202 fast.

function getDispatchMode(): 'spawn' | 'signal' {
  const raw = (process.env.STEWARD_DISPATCH || '').trim().toLowerCase();
  return raw === 'signal' ? 'signal' : 'spawn';  // default + any junk = spawn.
}

function loadLcwToken(): string | null {
  // Token lives in ~/.hermes/lcw_token.json — shape {"token":"<urlsafe>"}.
  try {
    const tokenPath = path.join(os.homedir(), '.hermes', 'lcw_token.json');
    if (!fs.existsSync(tokenPath)) return null;
    const v = JSON.parse(fs.readFileSync(tokenPath, 'utf-8'));
    const tok = typeof v?.token === 'string' ? v.token.trim() : '';
    return tok || null;
  } catch {
    return null;
  }
}

/**
 * POST the payload to the always-on backend (fire-and-forget). Returns true on a
 * 2xx (steward accepted), false on ANY failure — caller fails-open to spawn.
 */
async function signalBackend(payload: StewardPayload): Promise<boolean> {
  const token = loadLcwToken();
  if (!token) {
    log('signal: no bearer token at ~/.hermes/lcw_token.json — fail-open to spawn');
    return false;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SIGNAL_TIMEOUT_MS);
  try {
    const res = await fetch(STEWARD_BACKEND_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (res.status >= 200 && res.status < 300) {
      log(`signal: backend accepted (HTTP ${res.status}) — no spawn`);
      return true;
    }
    log(`signal: backend returned HTTP ${res.status} — fail-open to spawn`);
    return false;
  } catch (e) {
    log(`signal: POST failed (${e instanceof Error ? e.message : String(e)}) — fail-open to spawn`);
    return false;
  } finally {
    clearTimeout(timer);
  }
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
      // Pass the live JSONL path so runner.py can load the full recent tail
      // via _load_recent_transcript_tail().  This fixes the bug where
      // _pack_conversation only received the incremental delta turns (not the
      // full recent 30-turn window), leaving conversation.items nearly empty
      // and breaking voice-session continuity.
      transcript_path: hookInput.transcript_path,
      // Carry agent identity so runner.py writes THIS node's window file
      // (steward_window.<agent>.json) instead of defaulting every node to hermes.
      agent: deriveAgent(hookInput.cwd, hookInput.transcript_path),
    };
    log(`Derived agent=${payload.agent} (cwd=${hookInput.cwd})`);

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

    // ── ping-not-spawn (Step 1) ──────────────────────────────────────────────
    // When STEWARD_DISPATCH=signal, ping the always-on backend instead of
    // spawning. Fail-open: any POST failure falls through to the spawn path
    // below so the steward STILL runs. Default (unset/spawn) skips this entirely
    // → byte-identical to today.
    if (getDispatchMode() === 'signal') {
      const ok = await signalBackend(payload);
      if (ok) {
        saveLastIndex(hookInput.session_id, newLastProcessedIndex);
        log('Signalled backend steward (no spawn) — done');
        process.exit(0);
      }
      log('Signal failed — falling open to spawn path');
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
