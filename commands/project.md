---
description: Set or show the active project for this node (clean switch — resets the reload counter so the new project full-loads next turn). Usage: /project <slug> | /project
---

Set or show the active project for THIS node's work-from-plan modality.

Registry location (node-generic): project docs live in this node's registry —
`lcw/projects/` for hermes (the LCW product), or `projects/` for other nodes. The
active-project pointer is `~/.<node>/active_project`. The project-context hook reads
these every turn (it resolves the node from cwd).

Argument: `$ARGUMENTS` (a project slug, or empty to show current).

Steps:
1. If `$ARGUMENTS` is empty: read `~/.<node>/active_project`, report the current project + its status from the registry. Stop.
2. Otherwise validate: confirm `<registry>/<slug>.md` exists (check the registry's README for valid slugs). If not, list the options and ask — do NOT guess.
3. Clean switch:
   - Write the slug to `~/.<node>/active_project`.
   - Delete `~/.<node>/project_context_state.json` (resets the counter so the hook FULL-loads the new project next turn — clean handoff).
4. Read the full `<registry>/<slug>.md` now and confirm: project loaded, status, next milestone, open questions.
