#!/usr/bin/env npx tsx
/**
 * Stop hook — mechanical idle-guard (NEVER_IDLE_WORK_CASCADE.md item 1).
 *
 * Makes "idle/holding" turns structurally detectable so the never-idle law is
 * not willpower-dependent. Fires on Stop (main agent finished a turn). Reads the
 * LAST assistant turn from the session transcript and flags a LOOP-IDLE
 * VIOLATION when BOTH hold:
 *   (a) its text matches the banned-holding regex, AND
 *   (b) that turn made ZERO tool calls (no tool_use blocks).
 * On violation it injects a loud re-injection of the sweep directive via the
 * Stop hook's additionalContext so the next tick is forced to descend to a
 * sweep cell.
 *
 * CONTRACT: read JSON on stdin (transcript_path), emit
 * hookSpecificOutput{hookEventName:"Stop", additionalContext} on stdout, exit 0.
 *
 * SAFETY: FAIL-OPEN. Any error (no transcript, parse fail, weird shape) -> emit
 * nothing, exit 0. NEVER blocks the stop (no "decision":"block") — guidance
 * only, so it can never wedge the session. A turn WITH tool calls is never a
 * violation. Ported faithfully from aisys/plugin/scripts/stop_idle_guard.py.
 */

import { readBoundedStdinJson, recordHookError } from './conversation_utils.ts';
import { readTranscript, extractAllContent, TranscriptMessage } from './transcript_utils.ts';

// Banned-holding patterns (case-insensitive). Matches the SOP "THE ONE LAW" list +
// the euphemism catalogue from NEVER_IDLE_ENFORCEMENT_DOCTRINE_2026-06-15.md §4.
// Jacob 2026-06-15: "bake no idle into the loop, don't just use a different word" —
// the dressed-up synonyms ("standing by/ready", "holding the line", "no new signal",
// "not generating busywork", "minimal footprint", "the well is dry") are banned too.
const HOLDING_RE = new RegExp(
  'holding(?: the line| pattern| here)?' +
  '|standing (?:by|ready)' +
  '|on standby' +
  "|ready when you'?re (?:up|back)" +
  '|nothing (?:to do|left|of real value|actionable)' +
  '|no (?:new )?(?:real )?signal' +
  '|not (?:generating|manufacturing) busywork' +
  '|minimal footprint' +
  '|the well is dry' +
  '|exhausted (?:the )?(?:autonomous )?work' +
  "|pivot the moment you'?re back" +
  '|backlog (?:is )?(?:clear|empty)' +
  '|no (?:further )?(?:safe |un-?gated )?(?:work|movement)',
  'i',
);

const INJECTION =
  '⛔ LOOP-IDLE VIOLATION: last turn was a holding no-op (banned-holding ' +
  'text + zero tool calls). The next turn MUST descend to a sweep cell ' +
  '(health / optimization / research × think / do) per ' +
  "NEVER_IDLE_WORK_CASCADE.md. 'Holding' is banned as a turn outcome.";

interface HookInput {
  transcript_path?: string;
}

/**
 * Return the last `assistant` message object from the JSONL transcript.
 * Fail-soft: any error -> null.
 */
async function lastAssistantEntry(transcriptPath: string): Promise<TranscriptMessage | null> {
  try {
    const messages = await readTranscript(transcriptPath);
    let last: TranscriptMessage | null = null;
    for (const msg of messages) {
      const role = msg.type || msg.message?.role || msg.role;
      if (role === 'assistant') {
        last = msg;
      }
    }
    return last;
  } catch {
    return null;
  }
}

/**
 * True iff the turn has banned-holding text AND zero tool_use blocks.
 * A turn WITH any tool call is never a violation, even if its text says
 * "holding".
 */
function isViolation(entry: TranscriptMessage): boolean {
  const extracted = extractAllContent(entry);
  if (extracted.toolUses.length > 0) {
    return false; // any tool call -> never a violation
  }
  const text = extracted.text;
  if (!text || !text.trim()) {
    return false;
  }
  return HOLDING_RE.test(text);
}

function emit(text: string): void {
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'Stop',
      additionalContext: text,
    },
  }) + '\n');
}

async function main(): Promise<void> {
  let hookInput: HookInput | null;
  try {
    hookInput = await readBoundedStdinJson<HookInput>(30000);
  } catch {
    return; // fail-open
  }
  const path = hookInput?.transcript_path;
  if (!path) {
    return; // fail-open
  }
  const entry = await lastAssistantEntry(path);
  if (!entry) {
    return; // fail-open
  }
  if (isViolation(entry)) {
    emit(INJECTION);
  }
}

main().catch((e) => {
  // Absolute fail-open: never crash a turn or block the loop.
  try {
    recordHookError('stop_idle_guard.ts', e);
  } catch {}
  process.exit(0);
});
