#!/usr/bin/env npx tsx
/**
 * Stop hook — local-only.
 *
 * On every Stop event:
 *   1. Read transcript JSONL
 *   2. Diff against last processed index
 *   3. Write payload JSON  (coalesces with existing unprocessed payload — 2026-05-22)
 *   4. Spawn local Python worker (subconscious) detached
 *
 * No Letta cloud calls. No LETTA_API_KEY required.
 *
 * Coalescing (2026-05-22): if there is an existing unprocessed payload for
 * this session (no active lease), merge the new content into it instead of
 * writing a new file. This caps pending-payload count at 1 per session and
 * prevents the accumulation pattern that produced 20+ files for one session.
 */

import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import {
  buildPythonSubprocessEnv,
  loadSyncState,
  saveSyncState,
  getSyncStateFile,
  getMode,
  getTempStateDir,
  readBoundedStdinJson,
  recordHookError,
} from './conversation_utils.ts';
import {
  readTranscript,
  formatMessagesForLetta,
  formatAsXmlTranscript,
} from './transcript_utils.ts';
import { getConfig } from './config.ts';

const hermesConfig = getConfig();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEMP_STATE_DIR = getTempStateDir();
const LOG_FILE = path.join(TEMP_STATE_DIR, 'send_messages.log');

// --- Coalescing constants (2026-05-22) ---------------------------------------
// Matches local_worker.py _LEASE_TTL_SECONDS. Keep in sync.
const LEASE_TTL_SECONDS = 300;
// Max merged payload size before we split and write a new one.
const MAX_PAYLOAD_BYTES = 5 * 1024 * 1024;  // 5 MB

interface HookInput {
  session_id: string;
  transcript_path: string;
  stop_hook_active?: boolean;
  cwd: string;
  hook_event_name?: string;
}

// --- Ecosystem-cwd gate (mirror of aisys/subconscious/ecosystem_gate.py) -----
// Legit ecosystem agents (hermes, atlas, daedalus) all live under claude-nodes/.
// Match Windows + WSL forms, any slash direction, any case.
const ECOSYSTEM_ROOTS = [
  'c:/users/jbogi/claude-nodes',       // native Windows
  '/mnt/c/users/jbogi/claude-nodes',   // WSL mount of the same path
];

function normalizeCwd(cwd: string): string {
  let s = cwd.trim().replace(/\\/g, '/').toLowerCase();
  s = s.replace(/\/+/g, '/');   // collapse duplicate slashes
  s = s.replace(/\/+$/g, '');   // strip trailing slashes
  return s;
}

function cwdIsMissing(cwd: string | undefined | null): boolean {
  // "." is not a sentinel on the TS side (hook always supplies a real cwd),
  // but guard empty/whitespace/undefined anyway.
  return cwd == null || cwd.trim() === '';
}

function isEcosystemCwd(cwd: string | undefined | null): boolean {
  // Override hatch — always write.
  if (process.env.HERMES_MEMORY_ALLOW_ANY_CWD === '1') return true;
  // Fail-safe: missing/empty cwd -> WRITE (caller warns).
  if (cwdIsMissing(cwd)) return true;
  const norm = normalizeCwd(cwd as string);
  return ECOSYSTEM_ROOTS.some(root => norm === root || norm.startsWith(root + '/'));
}

interface LocalPayload {
  sessionId: string;
  cwd: string;
  stateFile: string;
  newLastProcessedIndex: number;
  transcriptXml: string;
}

interface LeaseFile {
  pid: number;
  started_at_epoch: number;
  ttl_seconds?: number;
  host?: string;
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
  ensureLogDir();
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(LOG_FILE, logLine);
}

async function readHookInput(): Promise<HookInput> {
  const v = await readBoundedStdinJson<HookInput>(30000);
  if (!v) throw new Error('empty or oversized stdin');
  return v;
}

// --- Coalescing helpers (2026-05-22) -----------------------------------------

function pidAlive(pid: number): boolean {
  if (pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    // EPERM means process exists but we can't signal — still alive
    return e?.code === 'EPERM';
  }
}

function leaseIsActive(leasePath: string): boolean {
  if (!fs.existsSync(leasePath)) return false;
  try {
    const lease: LeaseFile = JSON.parse(fs.readFileSync(leasePath, 'utf-8'));
    const ttl = lease.ttl_seconds ?? LEASE_TTL_SECONDS;
    const ageSec = (Date.now() / 1000) - (lease.started_at_epoch ?? 0);
    if (ageSec > ttl) return false;  // expired
    return pidAlive(lease.pid);
  } catch {
    return false;
  }
}

/**
 * Find an existing unprocessed payload for this session that no worker
 * is actively processing. Returns the path or null.
 */
function findCoalescePayload(sessionId: string): string | null {
  try {
    const entries = fs.readdirSync(TEMP_STATE_DIR);
    const prefix = `local-payload-${sessionId}-`;
    const candidates = entries
      .filter(e => e.startsWith(prefix) && e.endsWith('.json'))
      .map(e => path.join(TEMP_STATE_DIR, e))
      .filter(p => !leaseIsActive(p + '.lease'));
    if (candidates.length === 0) return null;
    // Prefer oldest — drains accumulated payloads in order.
    candidates.sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
    return candidates[0];
  } catch {
    return null;
  }
}

/**
 * Coalesce new payload content into an existing one. Append transcriptXml,
 * keep the higher newLastProcessedIndex. Returns true on success.
 */
function coalesceInto(existingPath: string, newPayload: LocalPayload): boolean {
  try {
    const existing: LocalPayload = JSON.parse(fs.readFileSync(existingPath, 'utf-8'));
    // Don't grow unboundedly — if merged would exceed MAX_PAYLOAD_BYTES, refuse coalesce
    // and let the caller write a new payload.
    const mergedXml = existing.transcriptXml + newPayload.transcriptXml;
    if (Buffer.byteLength(mergedXml, 'utf-8') > MAX_PAYLOAD_BYTES) {
      log(`Coalesce refused — merged payload would exceed ${MAX_PAYLOAD_BYTES} bytes`);
      return false;
    }
    const merged: LocalPayload = {
      sessionId: existing.sessionId,
      cwd: existing.cwd,
      stateFile: existing.stateFile,
      newLastProcessedIndex: Math.max(existing.newLastProcessedIndex, newPayload.newLastProcessedIndex),
      transcriptXml: mergedXml,
    };
    // Atomic write via tmp + rename
    const tmpPath = existingPath + '.tmp.' + process.pid;
    fs.writeFileSync(tmpPath, JSON.stringify(merged), 'utf-8');
    fs.renameSync(tmpPath, existingPath);
    return true;
  } catch (e) {
    log(`Coalesce failed: ${e instanceof Error ? e.message : String(e)}`);
    return false;
  }
}

async function main(): Promise<void> {
  log('='.repeat(60));
  log('stop_capture started');

  const mode = getMode();
  log(`Mode: ${mode}`);
  if (mode === 'off') {
    log('Mode is off, exiting');
    process.exit(0);
  }

  try {
    log('Reading hook input from stdin...');
    const hookInput = await readHookInput();
    log(`Hook input: session=${hookInput.session_id} transcript=${hookInput.transcript_path} cwd=${hookInput.cwd}`);

    if (hookInput.stop_hook_active) {
      log('Stop hook already active, exiting to prevent loop');
      process.exit(0);
    }

    // --- Ecosystem-cwd gate (primary, earliest skip) ----------------------
    // The hermes-memory plugin is enabled GLOBALLY, so this Stop hook fires in
    // EVERY Claude Code session on the machine and would otherwise write
    // candidate memories into the ONE shared store for unrelated chats. Only
    // dispatch the subconscious extractor for Hermes-ecosystem sessions (cwd
    // under claude-nodes/, covering hermes/atlas/daedalus). Gating here — before
    // reading the transcript and spawning the Python worker — also saves compute.
    // FAIL-SAFE: missing/empty cwd -> proceed (write) + warn; cost asymmetry is
    // breaking all Hermes extraction >> a little pollution. Mirrors the Python
    // gate in aisys/subconscious/ecosystem_gate.py — keep the two in sync.
    if (!isEcosystemCwd(hookInput.cwd)) {
      log(`subconscious: non-ecosystem cwd ${JSON.stringify(hookInput.cwd)}, skipping memory write`);
      process.exit(0);
    }
    if (cwdIsMissing(hookInput.cwd)) {
      log(`subconscious: cwd missing/empty (${JSON.stringify(hookInput.cwd)}) — proceeding (fail-safe write)`);
    }

    log(`Reading transcript from: ${hookInput.transcript_path}`);
    const messages = await readTranscript(hookInput.transcript_path, log);
    log(`Found ${messages.length} messages in transcript`);

    if (messages.length === 0) {
      log('No messages found, exiting');
      process.exit(0);
    }

    const typeCounts: Record<string, number> = {};
    for (const msg of messages) {
      const key = msg.type || msg.role || 'unknown';
      typeCounts[key] = (typeCounts[key] || 0) + 1;
    }
    log(`Message types: ${JSON.stringify(typeCounts)}`);

    const state = loadSyncState(hookInput.cwd, hookInput.session_id, log);

    const newMessages = formatMessagesForLetta(messages, state.lastProcessedIndex, log);

    if (newMessages.length === 0) {
      log('No new messages to send after formatting');
      process.exit(0);
    }

    // Spawn local Python worker (subconscious)
    const pluginRoot = path.resolve(__dirname, '..');
    const pythonDir = path.join(pluginRoot, 'python');
    if (!fs.existsSync(path.join(pythonDir, 'memory'))) {
      log('ERROR: python/memory/ not found — run install.ps1');
      console.error('Subconscious: Python modules missing. Run the install script.');
      process.exit(1);
    }

    const transcriptXml = formatAsXmlTranscript(newMessages);
    const stateFile = getSyncStateFile(hookInput.cwd, hookInput.session_id);

    const localPayload: LocalPayload = {
      sessionId: hookInput.session_id,
      cwd: hookInput.cwd,
      stateFile,
      newLastProcessedIndex: messages.length - 1,
      transcriptXml,
    };

    // --- Coalescing (2026-05-22) ------------------------------------------
    // If there's an existing unprocessed payload for this session (no active
    // lease — previous worker died, hit cap, or was killed), merge into it
    // instead of creating a new file. Caps pending-payload count at 1/session.

    let payloadFile: string;
    const coalesceTarget = findCoalescePayload(hookInput.session_id);
    if (coalesceTarget && coalesceInto(coalesceTarget, localPayload)) {
      payloadFile = coalesceTarget;
      log(`Coalesced into existing payload ${payloadFile} (${transcriptXml.length} new XML chars merged)`);
    } else {
      payloadFile = path.join(TEMP_STATE_DIR, `local-payload-${hookInput.session_id}-${Date.now()}.json`);
      fs.writeFileSync(payloadFile, JSON.stringify(localPayload), 'utf-8');
      log(`Wrote NEW local payload to ${payloadFile} (${transcriptXml.length} chars XML)`);
    }

    const workerScript = path.join(__dirname, 'local_worker.py');
    const pythonCmd = hermesConfig.pythonPath;

    const workerEnv = buildPythonSubprocessEnv({
      PYTHONPATH: pythonDir,
      HERMES_ROOT: hermesConfig.hermesRoot,
    });

    const child = spawn(pythonCmd, [workerScript, payloadFile], {
      detached: true,
      stdio: 'ignore',
      cwd: pythonDir,
      env: workerEnv,
      windowsHide: true,
    });
    child.unref();

    log(`Spawned local worker (PID: ${child.pid})`);
    log('Hook completed (local worker running in background)');
    process.exit(0);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`ERROR: ${errorMessage}`);
    if (error instanceof Error && error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
    console.error(`Error in stop_capture: ${errorMessage}`);
    recordHookError('send_messages_to_letta.ts', error);
    process.exit(0);
  }
}

main().catch((e) => {
  try { recordHookError('send_messages_to_letta.ts', e); } catch {}
  process.exit(0);
});
