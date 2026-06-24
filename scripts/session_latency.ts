#!/usr/bin/env tsx
/**
 * Stop hook — per-turn timing and token accounting.
 *
 * Reads the transcript JSONL, finds the last assistant message and the user
 * message immediately before it, computes round-trip latency, and appends ONE
 * JSON line to ~/.hermes/logs/session_latency.jsonl.
 *
 * Fail-soft: any error → exit 0 silently. Never breaks the Stop hook pipeline.
 * Output: nothing to stdout (hook does not need output).
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  readBoundedStdinJson,
  recordHookError,
} from './conversation_utils.ts';

interface HookInput {
  session_id: string;
  transcript_path: string;
  stop_hook_active?: boolean;
  cwd: string;
  hook_event_name?: string;
}

interface TranscriptUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

interface TranscriptMessage {
  type: string;
  timestamp?: string;
  uuid?: string;
  message?: {
    role?: string;
    usage?: TranscriptUsage;
  };
}

interface LatencyRecord {
  session_id: string;
  ts: string;
  turn_id: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
  rtt_ms: number | null;
  transcript_size_bytes: number;
  turn_count: number;
}

const LOG_DIR = path.join(os.homedir(), '.hermes', 'logs');
const LOG_FILE = path.join(LOG_DIR, 'session_latency.jsonl');

function parseIsoMs(ts: string | undefined): number | null {
  if (!ts) return null;
  const n = Date.parse(ts);
  return isFinite(n) ? n : null;
}

async function main(): Promise<void> {
  const hookInput = await readBoundedStdinJson<HookInput>(5000);
  if (!hookInput || hookInput.stop_hook_active) {
    process.exit(0);
  }

  const { session_id, transcript_path } = hookInput;
  if (!transcript_path) {
    process.exit(0);
  }

  // Read the transcript JSONL.
  let raw: string;
  let transcriptSizeBytes = 0;
  try {
    const stat = fs.statSync(transcript_path);
    transcriptSizeBytes = stat.size;
    raw = fs.readFileSync(transcript_path, 'utf-8');
  } catch {
    process.exit(0);
  }

  const lines = raw.split('\n').filter(l => l.trim());
  const turnCount = lines.length;

  // Parse all messages.
  const messages: TranscriptMessage[] = [];
  for (const line of lines) {
    try {
      messages.push(JSON.parse(line) as TranscriptMessage);
    } catch {
      // tolerate bad lines
    }
  }

  // Find the last assistant message.
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (
      m.type === 'assistant' &&
      m.message?.role === 'assistant' &&
      m.message?.usage
    ) {
      lastAssistantIdx = i;
      break;
    }
  }

  if (lastAssistantIdx < 0) {
    process.exit(0);
  }

  const assistantMsg = messages[lastAssistantIdx];
  const usage = assistantMsg.message?.usage ?? {};

  // Find the previous user message before the assistant.
  let prevUserIdx = -1;
  for (let i = lastAssistantIdx - 1; i >= 0; i--) {
    if (messages[i].type === 'user') {
      prevUserIdx = i;
      break;
    }
  }

  const assistantTsMs = parseIsoMs(assistantMsg.timestamp);
  const userTsMs = prevUserIdx >= 0 ? parseIsoMs(messages[prevUserIdx].timestamp) : null;

  let rttMs: number | null = null;
  if (assistantTsMs !== null && userTsMs !== null) {
    rttMs = assistantTsMs - userTsMs;
    // Guard against clock skew producing negative RTT.
    if (rttMs < 0) rttMs = null;
  }

  // Build a short turn_id from the assistant uuid if present.
  const turnId = (assistantMsg.uuid ?? '').slice(0, 8) || String(lastAssistantIdx);

  const record: LatencyRecord = {
    session_id: session_id ?? 'unknown',
    ts: new Date().toISOString(),
    turn_id: turnId,
    input_tokens: usage.input_tokens ?? 0,
    output_tokens: usage.output_tokens ?? 0,
    cache_read_input_tokens: usage.cache_read_input_tokens ?? 0,
    cache_creation_input_tokens: usage.cache_creation_input_tokens ?? 0,
    rtt_ms: rttMs,
    transcript_size_bytes: transcriptSizeBytes,
    turn_count: turnCount,
  };

  try {
    if (!fs.existsSync(LOG_DIR)) {
      fs.mkdirSync(LOG_DIR, { recursive: true });
    }
    fs.appendFileSync(LOG_FILE, JSON.stringify(record) + '\n', 'utf-8');
  } catch {
    // Logging failure must never break the Stop hook.
  }

  process.exit(0);
}

main().catch((e) => {
  try { recordHookError('session_latency.ts', e); } catch {}
  process.exit(0);
});
