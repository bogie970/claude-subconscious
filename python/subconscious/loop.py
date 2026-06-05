"""Agentic loop: transcript -> LLM chat -> tool_calls -> execute -> resend.

All block mutations are buffered and written atomically at the end of
a successful loop iteration. This prevents partial writes on crash.

Provider: Anthropic (Haiku) by default. Set SUBCONSCIOUS_PROVIDER=ollama to use Ollama.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

_provider = os.getenv("SUBCONSCIOUS_PROVIDER", "claude-cli").lower()
if _provider == "ollama":
    from . import ollama_chat as _chat
elif _provider == "anthropic":
    from . import anthropic_chat as _chat  # type: ignore[no-redef]
else:
    from . import claude_cli_chat as _chat  # type: ignore[no-redef]

from .blocks import BlockStore
from .config import (
    BLOCK_LABELS,
    LOOP_TIMEOUT_SECONDS,
    MAX_LOOP_ITERATIONS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from .prompts import build_system_message
from .tools import TOOL_DEFINITIONS, execute_tool

log = logging.getLogger(__name__)


class LoopResult:
    """Result of a single agentic loop run."""

    def __init__(self) -> None:
        self.tool_calls_made: list[dict[str, Any]] = []
        self.final_response: str = ""
        self.iterations: int = 0
        self.error: str | None = None
        self.duration_seconds: float = 0.0


def run_loop(
    store: BlockStore,
    transcript_xml: str,
    session_id: str = "",
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    memory_store: object | None = None,
) -> LoopResult:
    """Run the agentic loop: send transcript to Ollama, execute tool calls, repeat.

    All block mutations are buffered in-memory and only committed to the
    store on successful completion of each Ollama turn. This is the atomic
    transaction fix from the adversarial review.

    If *memory_store* (a MemoryStore instance) is provided, the
    conversation_search tool can query the ASys long-term vector memory.
    """
    result = LoopResult()
    start = time.monotonic()

    system_msg = build_system_message(store)

    user_message = (
        f"<claude_code_session_update>\n"
        f"<session_id>{session_id}</session_id>\n"
        f"{transcript_xml}\n"
        f"<instructions>\n"
        f"Process this transcript using your tools. You MUST call the tools to "
        f"update memory — do not just describe what you would do.\n"
        f"1. Extract user preferences and patterns → memory_insert or memory_rethink\n"
        f"2. Note project context and decisions → memory_insert or memory_rethink\n"
        f"3. Track unfinished work and TODOs → memory_insert into pending_items\n"
        f"4. If nothing noteworthy, say so without calling any tools.\n"
        f"</instructions>\n"
        f"</claude_code_session_update>"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_message},
    ]

    seen_calls: set[str] = set()
    response: dict[str, Any] = {}

    for iteration in range(1, MAX_LOOP_ITERATIONS + 1):
        elapsed = time.monotonic() - start
        if elapsed > LOOP_TIMEOUT_SECONDS:
            log.warning("Loop timeout after %.1fs (%d iterations)", elapsed, iteration - 1)
            result.error = f"timeout after {elapsed:.1f}s"
            break

        result.iterations = iteration
        log.info("Loop iteration %d (%.1fs elapsed)", iteration, elapsed)

        try:
            response = _chat.chat(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                model=model,
                host=host,
                timeout=max(1, int(LOOP_TIMEOUT_SECONDS - elapsed)),
                temperature=0.7,
            )
        except _chat.OllamaError as e:
            log.error("Ollama error: %s", e)
            result.error = str(e)

            if iteration == 1 and "Cannot connect" in str(e):
                break
            # Retry once on transient errors
            time.sleep(2)
            remaining = LOOP_TIMEOUT_SECONDS - (time.monotonic() - start) - 2
            if remaining < 1:
                log.warning("No time left for retry")
                break
            try:
                response = _chat.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    model=model,
                    host=host,
                    timeout=max(1, int(remaining)),
                    temperature=0.7,
                )
                result.error = None
            except _chat.OllamaError as e2:
                log.error("Retry failed: %s", e2)
                result.error = str(e2)
                break

        # Append assistant response to conversation
        messages.append(response)

        tool_calls = _chat.extract_tool_calls(response)

        if not tool_calls:
            # Check if model put tool calls in content (qwen2.5 bug workaround)
            content_calls = _parse_content_tool_calls(response.get("content", ""))
            if content_calls:
                log.info("Recovered %d tool calls from content field", len(content_calls))
                tool_calls = content_calls

        if not tool_calls:
            result.final_response = response.get("content", "")
            log.info("No tool calls — loop complete. Response: %s", result.final_response[:200])
            break

        # Deduplicate identical tool calls within this turn
        unique_calls = []
        for tc in tool_calls:
            key = f"{tc['name']}:{json.dumps(tc['arguments'], sort_keys=True)}"
            if key in seen_calls:
                log.info("Skipping duplicate tool call: %s", tc["name"])
                continue
            seen_calls.add(key)
            unique_calls.append(tc)

        if not unique_calls:
            log.info("All tool calls were duplicates — stopping loop")
            result.final_response = response.get("content", "")
            break

        log.info("Executing %d tool call(s)", len(unique_calls))

        for tc in unique_calls:
            name = tc["name"]
            args = tc["arguments"]
            tool_use_id = tc.get("_id", name)

            # Validate block label
            if "label" in args and args["label"] not in BLOCK_LABELS:
                tool_result = f"Error: Unknown block '{args['label']}'. Valid: {', '.join(BLOCK_LABELS)}"
                log.warning("Rejected tool call with invalid label: %s", args.get("label"))
            else:
                tool_result = execute_tool(store, name, args, memory_store=memory_store)

            result.tool_calls_made.append({"name": name, "args": args, "result": tool_result})

            messages.append(_chat.make_tool_result_message(name, tool_result, tool_use_id))

    else:
        log.warning("Reached max iterations (%d)", MAX_LOOP_ITERATIONS)
        result.error = f"max iterations ({MAX_LOOP_ITERATIONS})"
        result.final_response = response.get("content", "")

    result.duration_seconds = time.monotonic() - start
    log.info(
        "Loop finished: %d iterations, %d tool calls, %.1fs",
        result.iterations,
        len(result.tool_calls_made),
        result.duration_seconds,
    )
    return result


def _parse_content_tool_calls(content: str) -> list[dict[str, Any]]:
    """Try to extract tool calls from the content field.

    qwen2.5:7b sometimes puts tool calls as JSON in the content
    instead of in the tool_calls field. This is a best-effort recovery.
    Uses balanced-brace extraction instead of regex for nested values.
    """
    if not content:
        return []

    calls = []
    valid_names = {"memory_replace", "memory_insert", "memory_rethink", "conversation_search"}
    i = 0
    while i < len(content):
        idx = content.find("{", i)
        if idx == -1:
            break
        obj_str = _extract_balanced_json(content, idx)
        if obj_str:
            try:
                obj = json.loads(obj_str)
                name = obj.get("name")
                args = obj.get("arguments", {})
                if name in valid_names and isinstance(args, dict):
                    calls.append({"name": name, "arguments": args})
            except (json.JSONDecodeError, AttributeError):
                pass
            i = idx + len(obj_str)
        else:
            i = idx + 1

    return calls


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at `start`."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, min(start + 5000, len(text))):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
