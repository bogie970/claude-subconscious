#!/usr/bin/env tsx
/**
 * UserPromptSubmit hook — soft /advance modality reminder.
 *
 * Maintains a turn counter at ~/.hermes/runtime/advance_nudge.counter and emits
 * a gentle nudge every INTERVAL_N turns, encouraging Jacob to use /advance for
 * multi-step build work. Respects /advance-skip (10-turn snooze).
 *
 * Output: plain text written to stdout (same mechanism as checkin_inject.ts).
 * Fail-soft: any error → silent no-op (exit 0). Never breaks the prompt pipeline.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  readBoundedStdinJson,
  recordHookError,
} from './conversation_utils.ts';

interface HookInput {
  session_id?: string;
  cwd?: string;
  prompt?: string;
  transcript_path?: string;
}

interface NudgeCounter {
  count: number;
  last_nudge_at: number;
  snooze_until: number;
}

const RUNTIME_DIR = path.join(os.homedir(), '.hermes', 'runtime');
const COUNTER_FILE = path.join(RUNTIME_DIR, 'advance_nudge.counter');
const LAST_SKIP_FILE = path.join(RUNTIME_DIR, 'advance_nudge.last_skip');
const INTERVAL_N = 6;

function ensureRuntimeDir(): void {
  if (!fs.existsSync(RUNTIME_DIR)) {
    fs.mkdirSync(RUNTIME_DIR, { recursive: true });
  }
}

function loadCounter(): NudgeCounter {
  try {
    if (fs.existsSync(COUNTER_FILE)) {
      const raw = fs.readFileSync(COUNTER_FILE, 'utf-8');
      const parsed = JSON.parse(raw) as NudgeCounter;
      if (
        typeof parsed.count === 'number' &&
        typeof parsed.last_nudge_at === 'number' &&
        typeof parsed.snooze_until === 'number'
      ) {
        return parsed;
      }
    }
  } catch {
    // fall through to defaults
  }
  return { count: 0, last_nudge_at: 0, snooze_until: 0 };
}

function saveCounter(counter: NudgeCounter): void {
  ensureRuntimeDir();
  const tmp = COUNTER_FILE + '.tmp.' + process.pid;
  fs.writeFileSync(tmp, JSON.stringify(counter), 'utf-8');
  fs.renameSync(tmp, COUNTER_FILE);  // atomic replace
}

function writeLastSkip(): void {
  ensureRuntimeDir();
  const tmp = LAST_SKIP_FILE + '.tmp.' + process.pid;
  fs.writeFileSync(tmp, String(Date.now()), 'utf-8');
  fs.renameSync(tmp, LAST_SKIP_FILE);
}

function isSkipCommand(prompt: string | undefined): boolean {
  return (prompt ?? '').trim() === '/advance-skip';
}

async function main(): Promise<void> {
  const hookInput = await readBoundedStdinJson<HookInput>(3000);
  const prompt = hookInput?.prompt;

  // Handle /advance-skip early: write marker, extend snooze, emit nothing.
  if (isSkipCommand(prompt)) {
    try {
      const counter = loadCounter();
      counter.count += 1;
      counter.snooze_until = counter.count + 10;
      writeLastSkip();
      saveCounter(counter);
    } catch {
      // fail-soft
    }
    process.exit(0);
  }

  try {
    const counter = loadCounter();
    counter.count += 1;

    // Determine whether to emit a nudge.
    // Nudge fires when count reaches snooze_until + INTERVAL_N,
    // treating snooze_until=0 as: fire at count >= INTERVAL_N.
    const nextNudgeAt = counter.snooze_until + INTERVAL_N;
    const shouldNudge = counter.count >= nextNudgeAt;

    if (shouldNudge) {
      counter.last_nudge_at = counter.count;
      counter.snooze_until = counter.count;  // restart the interval from here

      const nudgeText = (
        `[ADVANCE-MODALITY LAW @ turn ${counter.count}] For any substantive work this is REQUIRED, not optional: ` +
        `run the advance modality — OPEN (load the active project's plan + state it), work ONLY from the plan, ` +
        `CLOSE (update the plan). Invoke /advance if not already in the loop. Reactive work without the plan loaded IS the drift.`
      );
      process.stdout.write(nudgeText);
    }

    saveCounter(counter);
  } catch {
    // fail-soft: never break the prompt pipeline
  }

  process.exit(0);
}

main().catch((e) => {
  try { recordHookError('advance_nudge.ts', e); } catch {}
  process.exit(0);
});
