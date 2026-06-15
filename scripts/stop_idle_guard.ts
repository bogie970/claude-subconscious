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

function roleOf(m: TranscriptMessage): string {
  return (m.type || m.message?.role || (m as { role?: string }).role || '') as string;
}

function contentBlocks(m: TranscriptMessage): unknown[] {
  const c = m.message?.content ?? (m as { content?: unknown }).content;
  if (Array.isArray(c)) return c;
  if (typeof c === 'string') return [{ type: 'text', text: c }];
  return [];
}

/**
 * A genuine HUMAN prompt = user/human role with NO tool_result blocks. Tool
 * results are recorded as user-role messages too, so they must NOT be treated
 * as a turn boundary — only a real human prompt starts a new turn.
 */
function isHumanPrompt(m: TranscriptMessage): boolean {
  const r = roleOf(m);
  if (r !== 'user' && r !== 'human') return false;
  return !contentBlocks(m).some((b) => (b as { type?: string })?.type === 'tool_result');
}

/**
 * All assistant messages of the CURRENT turn (since the last genuine human
 * prompt), newest first. Fail-soft: any error -> null.
 * FIX 2026-06-15: previously judged only the single last assistant MESSAGE, so a
 * real working turn that ended with a text summary mentioning "holding" (zero
 * tool_use in that final block) false-fired. Now the whole turn is considered.
 */
async function currentTurnAssistant(transcriptPath: string): Promise<TranscriptMessage[] | null> {
  try {
    const messages = await readTranscript(transcriptPath);
    const turn: TranscriptMessage[] = [];
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (isHumanPrompt(m)) break;            // start of this turn reached
      if (roleOf(m) === 'assistant') turn.push(m);
    }
    return turn.length ? turn : null;          // turn[0] = last assistant message
  } catch {
    return null;
  }
}

/**
 * True iff the WHOLE turn made ZERO tool calls AND its final assistant text
 * matches the banned-holding regex. Any tool_use anywhere in the turn -> the
 * turn did real work -> never a violation.
 */
function isViolation(turn: TranscriptMessage[]): boolean {
  for (const m of turn) {
    if (extractAllContent(m).toolUses.length > 0) return false; // real work done
  }
  const lastText = extractAllContent(turn[0]).text; // turn[0] = last assistant msg
  if (!lastText || !lastText.trim()) return false;
  return HOLDING_RE.test(lastText);
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
  const turn = await currentTurnAssistant(path);
  if (!turn) {
    return; // fail-open
  }
  if (isViolation(turn)) {
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
