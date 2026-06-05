"""Claude CLI chat wrapper — uses `claude --print` with Max subscription.

Drop-in replacement for ollama_chat. No API key needed — uses the user's
Claude subscription via the CLI. Tool calls are returned as JSON in content
and parsed client-side (same pattern as the qwen2.5 workaround).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from typing import Any

log = logging.getLogger(__name__)

MODEL = os.getenv("SUBCONSCIOUS_MODEL", "haiku")


class OllamaError(Exception):
    """Kept for API compat with loop.py's except clause."""


_claude_bin: str | None = None

def _get_claude_bin() -> str:
    global _claude_bin
    if _claude_bin is None:
        _claude_bin = shutil.which("claude")
        if not _claude_bin:
            raise OllamaError("claude CLI not found in PATH")
    return _claude_bin


def _messages_to_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """Flatten message list into a single prompt for claude --print."""
    parts: list[str] = []

    tool_instructions = ""
    if tools:
        tool_names = []
        tool_specs = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "")
            tool_names.append(name)
            tool_specs.append(json.dumps({
                "name": name,
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }, indent=2))

        tool_instructions = (
            "\n\n<available_tools>\n"
            + "\n".join(tool_specs)
            + "\n</available_tools>\n\n"
            "IMPORTANT: When you want to call a tool, output ONLY a JSON array of tool call objects. "
            "Each object must have exactly these fields: {\"name\": \"tool_name\", \"arguments\": {...}}. "
            "Output the JSON array and NOTHING ELSE — no explanation, no markdown, no text before or after. "
            "If you have no tools to call, respond with normal text (no JSON).\n"
            "Example tool call output:\n"
            '[{"name": "memory_insert", "arguments": {"label": "project_context", "new_str": "example"}}]\n'
        )

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"<system>\n{content}{tool_instructions}</system>")
        elif role == "user":
            parts.append(f"<user>\n{content}\n</user>")
        elif role == "assistant":
            parts.append(f"<assistant>\n{content}\n</assistant>")
        elif role == "tool":
            name = msg.get("name", "unknown")
            parts.append(f"<tool_result name=\"{name}\">\n{content}\n</tool_result>")

    return "\n\n".join(parts)


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "",
    host: str = "",
    timeout: int = 120,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Send a prompt to claude --print. Returns an Ollama-compatible message dict."""
    if not model:
        model = MODEL
    claude_bin = _get_claude_bin()

    prompt = _messages_to_prompt(messages, tools)

    cmd = [
        claude_bin, "--print",
        "--model", model,
        "--output-format", "text",
        "--no-session-persistence",
        "--tools", "",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if platform.system() == "Windows" else 0),
            env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "subconscious-worker"},
        )
    except subprocess.TimeoutExpired:
        raise OllamaError(f"claude CLI timed out after {timeout}s")
    except OSError as e:
        raise OllamaError(f"claude CLI exec failed: {e}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:300]
        raise OllamaError(f"claude CLI error (rc={result.returncode}): {stderr}")

    content = result.stdout.strip()

    log.debug("claude CLI response (%d chars): %s", len(content), content[:200])

    response: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    parsed_calls = _parse_json_tool_calls(content)
    if parsed_calls:
        response["tool_calls"] = [
            {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in parsed_calls
        ]
        response["content"] = ""

    return response


def _parse_json_tool_calls(content: str) -> list[dict[str, Any]]:
    """Try to parse content as a JSON array of tool calls."""
    content = content.strip()
    if not content.startswith("["):
        # Maybe it's wrapped in markdown code block
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start) if "```" in content[start:] else len(content)
            content = content[start:start + (end - start)].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start) if "```" in content[start:] else len(content)
            content = content[start:start + (end - start)].strip()

    if not content.startswith("["):
        return []

    valid_names = {"memory_replace", "memory_insert", "memory_rethink", "conversation_search"}

    try:
        arr = json.loads(content)
        if not isinstance(arr, list):
            return []
        calls = []
        for item in arr:
            name = item.get("name")
            args = item.get("arguments", {})
            if name in valid_names and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
        return calls
    except (json.JSONDecodeError, AttributeError):
        return []


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from response — same interface as ollama_chat."""
    raw = message.get("tool_calls", [])
    if not raw:
        return []

    calls = []
    for tc in raw:
        func = tc.get("function", {})
        name = func.get("name")
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args, "_id": name})
    return calls


def make_tool_result_message(tool_name: str, content: str, tool_use_id: str = "") -> dict[str, Any]:
    """Create a tool result message."""
    return {
        "role": "tool",
        "name": tool_name,
        "content": content,
    }
