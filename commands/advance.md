---
description: Advance the active project by the disciplined modality — rigid anti-drift bookends, depth scaled to stakes. Usage: /advance [focus]
---

Advance the active project. The BOOKENDS are non-negotiable (they prevent drift); the MIDDLE scales to the stakes of the work. Optional `$ARGUMENTS` narrows focus.

## RIGID — always, no exceptions (the anti-drift guarantee)
- **OPEN: Load the plan.** Read `~/.<node>/active_project` -> read the FULL project doc from this node's registry (`lcw/projects/<slug>.md` for hermes, `projects/<slug>.md` for other nodes). State the current milestone + open questions. No active project -> stop, tell user to `/project <slug>`.
- **Only do work that is in the plan.** If the work isn't in the plan, it goes INTO the plan first (or gets flagged to Jacob) — never freestyled around it.
- **CLOSE: Update the plan.** After work: update §4 Current state, §6 Tasks, §7 Increment log. Trace what changed to which milestone. Surface anything needing Jacob's ruling; never lock a decision without his okay.

## TRIAGE — pick the depth (do this right after loading the plan)
Assess the focus on two axes — **clarity** (is the approach already obvious?) and **stakes** (live-memory / irreversible / architectural?). Then choose:

- **LIGHT** (clear + low stakes, e.g. a contained fix the plan already specifies): skip planning agents. Do the work directly (yourself or one work agent), then the CLOSE bookend. No ceremony.
- **MEDIUM** (some ambiguity OR moderate stakes): one planning agent to surface approach + tradeoffs; you parse; optional single review; dispatch work.
- **HEAVY** (ambiguous OR high stakes — live memory, architecture, irreversible): the full loop — parallel planning agents (competing angles) -> you parse to the plan -> adversarial review agent -> work agents -> integrate. This is where "massive multi-agent" belongs.

State which tier you chose and why in one line. When unsure, go one tier heavier.

## THE LOOP (run at the chosen depth — iterate, don't force forward)
1. **Plan** (MEDIUM/HEAVY): spawn planning agent(s) for ideas/tradeoffs. They propose only — no production edits.
2. **Parse** (you): decide what enters the plan; update the doc. If planning reveals the premise was wrong or the work is bigger, LOOP BACK to step 1 or re-triage — don't force forward.
3. **Review** (HEAVY, or MEDIUM if risky): adversarial agent pressure-tests the updated plan. If it finds a flaw, loop back to Parse.
4. **Work**: dispatch work agent(s) to execute strictly from the plan. Gate: only what's in the plan.
5. **Integrate**: trust-but-verify (check actual changes, not summaries). Then the CLOSE bookend.

The bookends keep us systematic and drift-free; the triage keeps us from drowning a small task in ceremony. Both matter.
