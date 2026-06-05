#!/usr/bin/env tsx
/**
 * UserPromptSubmit hook — local-only.
 *
 * Injects on every prompt:
 *   1. Memory block diff (5 letta-style blocks, now "patterns" layer)
 *   2. L2 retrieval (vector search via memory.query_retrieve)
 *
 * (The subconscious "whisper" injection was retired 2026-06-04.)
 *
 * No Letta cloud calls. No LETTA_API_KEY required.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import {
  buildPythonSubprocessEnv,
  loadSyncState,
  saveSyncState,
  SyncState,
  Agent,
  MemoryBlock,
  escapeXmlContent,
  sanitizeBlockLabel,
  formatAllBlocksForStdout,
  cleanLettaFromClaudeMd,
  getMode,
  getTempStateDir,
  recordHookError,
} from './conversation_utils.ts';
import { getLocalAgent } from './local_store.ts';
import { getConfig } from './config.ts';

const hermesConfig = getConfig();
const DEBUG = hermesConfig.debug;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function debug(...args: unknown[]): void {
  if (DEBUG) {
    console.error('[sync debug]', ...args);
  }
}

interface HookInput {
  session_id: string;
  cwd: string;
  prompt?: string;
  transcript_path?: string;
}

const TEMP_STATE_DIR = getTempStateDir();

async function readHookInput(): Promise<HookInput | null> {
  return new Promise((resolve) => {
    let input = '';
    const rl = readline.createInterface({ input: process.stdin });

    rl.on('line', (line) => {
      input += line;
    });

    rl.on('close', () => {
      if (!input.trim()) {
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(input));
      } catch {
        resolve(null);
      }
    });

    setTimeout(() => {
      rl.close();
    }, 1000);
  });
}

function detectChangedBlocks(
  currentBlocks: MemoryBlock[],
  lastBlockValues: { [label: string]: string } | null
): MemoryBlock[] {
  if (!lastBlockValues) {
    return [];
  }
  return currentBlocks.filter(block => {
    const previousValue = lastBlockValues[block.label];
    return previousValue === undefined || previousValue !== block.value;
  });
}

function computeDiff(oldValue: string, newValue: string): { added: string[], removed: string[] } {
  const oldLines = oldValue.split('\n').map(l => l.trim()).filter(l => l);
  const newLines = newValue.split('\n').map(l => l.trim()).filter(l => l);

  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);

  const added = newLines.filter(line => !oldSet.has(line));
  const removed = oldLines.filter(line => !newSet.has(line));

  return { added, removed };
}

function formatChangedBlocksForStdout(
  changedBlocks: MemoryBlock[],
  lastBlockValues: { [label: string]: string } | null
): string {
  if (changedBlocks.length === 0) {
    return '';
  }

  const nonEmptyChanges = changedBlocks.filter(block => block.value && block.value.trim());
  if (nonEmptyChanges.length === 0) {
    return '';
  }

  const formatted = nonEmptyChanges.map(block => {
    const safeLabel = sanitizeBlockLabel(block.label);
    const previousValue = lastBlockValues?.[block.label];

    if (previousValue === undefined) {
      const escapedContent = escapeXmlContent(block.value || '');
      return `<${safeLabel} status="new">\n${escapedContent}\n</${safeLabel}>`;
    }

    const diff = computeDiff(previousValue, block.value || '');

    if (diff.added.length === 0 && diff.removed.length === 0) {
      const escapedContent = escapeXmlContent(block.value || '');
      return `<${safeLabel} status="modified">\n${escapedContent}\n</${safeLabel}>`;
    }

    const diffLines: string[] = [];
    for (const line of diff.removed) {
      diffLines.push(`- ${escapeXmlContent(line)}`);
    }
    for (const line of diff.added) {
      diffLines.push(`+ ${escapeXmlContent(line)}`);
    }

    return `<${safeLabel} status="modified">\n${diffLines.join('\n')}\n</${safeLabel}>`;
  }).join('\n');

  return `<patterns_update>
<!-- Pattern blocks updated since last prompt (showing diff) -->
${formatted}
</patterns_update>`;
}

function _sanitizeCwd(cwd: string): string {
  return cwd.replace(/[\\/:]/g, '-').replace(/^-+/, '');
}

function _markerDir(cwd: string): string {
  return path.join(os.homedir(), '.claude', 'projects', _sanitizeCwd(cwd), 'l1_markers');
}

// Strict marker filename pattern: only files written by our l1_manager.
// Format: l1_evicted_<session_prefix>_<block_id_hex>.md (per-session isolation)
// or legacy: l1_evicted_<block_id_hex>.md (backward compat).
// Rejects any other .md file an attacker (or accidental drop) might place
// in the marker dir.
const MARKER_FILENAME_RE = /^l1_evicted_(?:[\w\-]{1,32}_)?[0-9a-f]{8,32}\.md$/;
const MARKER_MAX_BYTES = 64 * 1024;  // 64 KB hard cap per marker

/**
 * Read and consume L1-evicted marker files. Validates filename pattern
 * strictly, bounds size, and wraps content in an explicit sandbox tag
 * so the model treats it as data rather than instructions.
 *
 * Rejects: filenames not matching MARKER_FILENAME_RE, files >64KB,
 * paths that escape the marker dir.
 */
function consumeL1Markers(cwd: string): string[] {
  const dir = _markerDir(cwd);
  if (!fs.existsSync(dir)) return [];
  const dirResolved = path.resolve(dir);
  const out: string[] = [];
  let entries: string[] = [];
  try {
    entries = fs.readdirSync(dir);
  } catch {
    return [];
  }
  for (const name of entries) {
    if (!MARKER_FILENAME_RE.test(name)) {
      // Silently skip files that don't match our pattern (possibly an attack)
      debug('marker rejected (bad name):', name);
      continue;
    }
    const full = path.join(dir, name);
    // Defense-in-depth: ensure resolved path is still inside the marker dir
    if (!path.resolve(full).startsWith(dirResolved + path.sep)) {
      debug('marker rejected (path traversal):', name);
      continue;
    }
    try {
      const stat = fs.statSync(full);
      if (stat.size > MARKER_MAX_BYTES) {
        debug('marker rejected (too large):', name, stat.size);
        continue;
      }
      const raw = fs.readFileSync(full, 'utf-8');
      // Escape any embedded XML so injected content cannot break out of
      // the wrapper tag. The l1_manager output is already shaped, but
      // user-controlled chunks could contain anything.
      const escaped = raw
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      const wrapped = `<l1_evicted_marker file="${name}">\n${escaped}\n</l1_evicted_marker>`;
      out.push(wrapped);

      // Move out of the active dir so the next run doesn't re-inject
      const consumed = full.replace(/\.md$/, `.consumed-${Date.now()}.md`);
      fs.renameSync(full, consumed);
    } catch (e) {
      debug('marker read failed:', name, e);
    }
  }
  return out;
}

/**
 * Resolve the steward injector mode, MIRRORING aisys/memory/steward_inject.py
 * `_resolve_mode()` precedence EXACTLY (first wins):
 *   1. env STEWARD_INJECT_MODE
 *   2. ~/.hermes/steward.config.yaml  ->  inject_mode: <off|shadow|live>
 *   3. default: "off"
 *
 * Why this lives here: when steward is in `live` mode it injects its own
 * <steward_window> (via the steward_inject.py UserPromptSubmit hook). Running
 * query_retrieve here too would DOUBLE-inject retrieved memory every turn. This
 * helper lets us skip query_retrieve when steward is the live source.
 *
 * No YAML dependency is added: steward.config.yaml is a flat file whose only
 * meaningful key is `inject_mode:`, so we line-scan for it. If the file ever
 * grows nested structure this stays correct as long as `inject_mode` remains a
 * top-level scalar (it is the canonical contract in steward.config.yaml).
 *
 * Fail-open to "off" on ANY error (matches the Python side): if we cannot tell
 * whether steward is live, we keep query_retrieve running so the agent is never
 * left with NO retrieved-memory injector.
 */
function stewardInjectMode(): 'off' | 'shadow' | 'live' {
  const VALID = new Set(['off', 'shadow', 'live']);
  try {
    const env = (process.env.STEWARD_INJECT_MODE || '').trim().toLowerCase();
    if (VALID.has(env)) return env as 'off' | 'shadow' | 'live';

    const cfgPath = path.join(os.homedir(), '.hermes', 'steward.config.yaml');
    if (fs.existsSync(cfgPath)) {
      const raw = fs.readFileSync(cfgPath, 'utf-8');
      for (const line of raw.split(/\r?\n/)) {
        const m = line.match(/^\s*inject_mode\s*:\s*([A-Za-z]+)/);
        if (m) {
          // strip inline comments / quotes already excluded by [A-Za-z]+
          const val = m[1].trim().toLowerCase();
          if (VALID.has(val)) return val as 'off' | 'shadow' | 'live';
          break; // found the key but value invalid -> fall through to default
        }
      }
    }
  } catch {
    // fall through to default
  }
  return 'off';
}

/**
 * Retrieve query-dependent memories from LanceDB vector store.
 * Calls Python subprocess: memory.query_retrieve
 */
function retrieveMemories(prompt: string, cwd: string): string {
  const pythonPath = hermesConfig.pythonPath;
  const pluginRoot = path.resolve(__dirname, '..');
  const pythonDir = path.join(pluginRoot, 'python');

  if (!prompt || prompt.trim().length < 5) {
    debug('retrieveMemories: prompt too short, skipping');
    return '';
  }

  const truncatedPrompt = prompt.slice(0, 8000);

  try {
    const t0 = Date.now();
    const result = spawnSync(
      pythonPath,
      ['-m', 'memory.query_retrieve', truncatedPrompt, '--k', '10', '--format', 'xml'],
      {
        cwd: pythonDir,
        timeout: 15000,
        encoding: 'utf-8',
        env: buildPythonSubprocessEnv({ PYTHONPATH: pythonDir }),
        windowsHide: true,
      }
    );
    const elapsed = Date.now() - t0;

    if (result.stderr) {
      debug(`retrieveMemories stderr (${elapsed}ms):`, result.stderr.trim());
    }

    if (result.status !== 0) {
      debug(`retrieveMemories: python exited with code ${result.status}`);
      return '';
    }

    const output = (result.stdout || '').trim();
    if (!output || output === '<retrieved_memories count="0"/>') {
      debug(`retrieveMemories: no results (${elapsed}ms)`);
      return '';
    }

    debug(`retrieveMemories: got results (${elapsed}ms), ${output.length} chars`);
    return output;
  } catch (err) {
    debug('retrieveMemories error:', err);
    return '';
  }
}

/**
 * ADDITIVE LOGGING (steward S1 shadow). Record the memory ids this live
 * injector actually chose this turn so steward_inject.py can compare its
 * pre-computed window against the truth ("hooks_injected_ids"). This parses
 * the SAME `retrievedXml` we already emit and writes it to a side file — it
 * does NOT change `retrievedXml`, `outputs`, or anything the agent sees.
 * Fully swallowed: a logging failure must never affect the injection.
 */
function recordHookInjection(retrievedXml: string): void {
  try {
    const ids: string[] = [];
    // <memory id="..."> emitted by memory.query_retrieve
    const re = /<memory\s+id="([^"]*)"/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(retrievedXml)) !== null) {
      ids.push(m[1]);
    }
    const runtimeDir = path.join(os.homedir(), '.hermes', 'runtime');
    if (!fs.existsSync(runtimeDir)) fs.mkdirSync(runtimeDir, { recursive: true });
    const out = path.join(runtimeDir, 'last_hook_injection.json');
    const record = { ts: new Date().toISOString(), count: ids.length, ids };
    const tmp = out + '.tmp.' + process.pid;
    fs.writeFileSync(tmp, JSON.stringify(record), 'utf-8');
    fs.renameSync(tmp, out);  // atomic replace — reader sees whole-or-nothing
  } catch {
    // never let shadow logging affect the live injector
  }
}

/**
 * Surface unread hook errors as <hook_error> tags. Tracks cursor in
 * hook_errors.cursor so each entry is shown exactly once. All errors
 * swallowed — this must NEVER raise into the calling hook.
 */
function surfaceHookErrors(): string {
  try {
    const dir = getTempStateDir();
    const file = path.join(dir, 'hook_errors.jsonl');
    const cursorFile = path.join(dir, 'hook_errors.cursor');
    if (!fs.existsSync(file)) return '';

    let cursor = 0;
    try {
      if (fs.existsSync(cursorFile)) {
        const raw = fs.readFileSync(cursorFile, 'utf-8').trim();
        const parsed = parseInt(raw, 10);
        if (Number.isFinite(parsed) && parsed >= 0) cursor = parsed;
      }
    } catch {}

    let stat: fs.Stats;
    try {
      stat = fs.statSync(file);
    } catch {
      return '';
    }

    // If file was rotated (truncated/shrunk), reset cursor to 0.
    if (cursor > stat.size) cursor = 0;
    if (cursor === stat.size) return '';

    let buf = '';
    try {
      const fd = fs.openSync(file, 'r');
      try {
        const len = stat.size - cursor;
        const b = Buffer.alloc(len);
        fs.readSync(fd, b, 0, len, cursor);
        buf = b.toString('utf-8');
      } finally {
        try { fs.closeSync(fd); } catch {}
      }
    } catch {
      return '';
    }

    const lines = buf.split('\n').filter(l => l.trim());
    const tags: string[] = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        const ts = escapeXmlContent(String(entry.ts || ''));
        const script = escapeXmlContent(String(entry.script || 'unknown'));
        const errMsg = escapeXmlContent(String(entry.error || ''));
        const stack = escapeXmlContent(String(entry.stack || ''));
        tags.push(
          `<hook_error script="${script}" ts="${ts}">\n` +
          `<error>${errMsg}</error>\n` +
          `<stack>${stack}</stack>\n` +
          `</hook_error>`
        );
      } catch {
        // skip malformed line
      }
    }

    // Advance cursor regardless of parse failures so we don't loop forever.
    try {
      fs.writeFileSync(cursorFile, String(stat.size), 'utf-8');
    } catch {}

    if (tags.length === 0) return '';
    return (
      `<hook_errors_notice>The following hook(s) failed silently since last prompt. ` +
      `Surface these to the user — async hooks otherwise exit 0 and go unnoticed.</hook_errors_notice>\n` +
      tags.join('\n')
    );
  } catch {
    return '';
  }
}

async function main(): Promise<void> {
  const mode = getMode();
  if (mode === 'off') {
    process.exit(0);
  }

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  try {
    const hookInput = await readHookInput();
    const cwd = hookInput?.cwd || projectDir;
    const sessionId = hookInput?.session_id;

    let state: SyncState | null = null;
    if (sessionId) {
      state = loadSyncState(cwd, sessionId);
    }

    const lastBlockValues = state?.lastBlockValues || null;

    // Local-only: load pattern blocks from disk
    const agent: Agent = getLocalAgent(cwd);
    debug('Loaded', (agent.blocks || []).length, 'blocks');
    const populated = (agent.blocks || []).filter(b => b.value && b.value.trim());
    debug('Non-empty blocks:', populated.map(b => b.label).join(', ') || 'none');

    const changedBlocks = detectChangedBlocks(agent.blocks || [], lastBlockValues);
    debug('Changed blocks:', changedBlocks.length, changedBlocks.map(b => b.label).join(', '));

    // Legacy CLAUDE.md scrub (one-shot migration; will remove this in ~6 months)
    cleanLettaFromClaudeMd(cwd);

    if (state) {
      state.lastBlockValues = {};
      for (const block of agent.blocks || []) {
        state.lastBlockValues[block.label] = block.value;
      }
    }

    const outputs: string[] = [];

    if (mode === 'full') {
      const isFirstPrompt = !lastBlockValues;
      if (isFirstPrompt) {
        outputs.push(formatAllBlocksForStdout(agent, null));
      } else {
        const changedBlocksOutput = formatChangedBlocksForStdout(changedBlocks, lastBlockValues);
        if (changedBlocksOutput) {
          outputs.push(changedBlocksOutput);
        }
      }
    }

    // L2 vector store retrieval.
    //
    // RETIREMENT GATE (2026-06-01): when steward is in `live` inject_mode it
    // injects its own <steward_window> (curated working_set) every turn via the
    // steward_inject.py UserPromptSubmit hook. Running query_retrieve here too
    // would DOUBLE-inject retrieved memory. So when steward is live we SKIP
    // query_retrieve and become a no-op for this section. steward is the SOLE
    // retrieved-memory injector.
    //
    // We still write a truthful EMPTY last_hook_injection.json so that if the
    // operator later flips steward back to `shadow`, its shadow-compare reads a
    // correct (empty) "hooks_injected_ids" instead of a stale set from when
    // query_retrieve was last active. recordHookInjection('') parses to [].
    if (hookInput?.prompt) {
      const mode = stewardInjectMode();
      if (mode === 'live') {
        debug('steward inject_mode=live -> skipping query_retrieve (steward is sole injector)');
        recordHookInjection(''); // truthful empty set
      } else {
        const retrievedXml = retrieveMemories(hookInput.prompt, cwd);
        // ADDITIVE (steward S1 shadow): log the chosen ids before the unchanged
        // emit path. Records [] when nothing was retrieved, so the shadow compare
        // sees the true injected set every turn. Does NOT alter outputs.
        recordHookInjection(retrievedXml);
        if (retrievedXml) {
          outputs.push(retrievedXml);
        }
      }
    }

    // L1-evicted markers (Phase D) — emitted by l1_watch / precompact_safety
    const markers = consumeL1Markers(cwd);
    if (markers.length > 0) {
      outputs.push(...markers);
      outputs.push(`<instruction>L1 manager evicted ${markers.length} chunk-block(s) above. Use memory_recall(query, scope="l1_evict") if you need details from the evicted content.</instruction>`);
    }

    // (Subconscious whisper injection retired 2026-06-04.)

    // Surface any silent hook errors recorded since last prompt
    const hookErrorOutput = surfaceHookErrors();
    if (hookErrorOutput) {
      outputs.push(hookErrorOutput);
    }

    const finalOutput = outputs.join('\n\n');
    debug('Final output length:', finalOutput.length, 'chars');
    console.log(finalOutput);

    // ── Inspector v1 (hermes 2026-05-30): per-fire token accounting log ──
    // Append a record of THIS hook fire to ~/.hermes/logs/hook_injections.jsonl
    // so the Iris Window Inspector can show the REAL hook contribution to the
    // prompt (currently invisible — counted in "uncounted overhead"). Pure
    // observability; output to Claude unchanged. Fail-soft on every step.
    try {
      const tokensEmitted = Math.max(1, Math.floor(finalOutput.length / 4));
      const logDir = path.join(os.homedir(), '.hermes', 'logs');
      if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });
      const logPath = path.join(logDir, 'hook_injections.jsonl');
      const rec = {
        ts: new Date().toISOString(),
        hook_name: 'sync_letta_memory',
        tokens_emitted: tokensEmitted,
        char_count: finalOutput.length,
        sections: outputs.length,
      };
      fs.appendFileSync(logPath, JSON.stringify(rec) + '\n', 'utf-8');
    } catch { /* never let logging affect the hook */ }

    if (state && sessionId) {
      saveSyncState(cwd, state);
    }

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Error syncing memory: ${errorMessage}`);
    recordHookError('sync_letta_memory.ts', error);
    process.exit(0);
  }
}

main().catch((e) => {
  try { recordHookError('sync_letta_memory.ts', e); } catch {}
  process.exit(0);
});
