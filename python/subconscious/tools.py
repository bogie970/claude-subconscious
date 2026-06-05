"""Tool definitions and executors for the subconscious agentic loop.

Tools are defined in Ollama's tool calling format and executed locally
against the BlockStore and (optionally) the ASys vector memory store.
"""

from __future__ import annotations

import logging
from typing import Any

from .blocks import BlockStore
from .config import BLOCK_LABELS

log = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "memory_replace",
            "description": (
                "Replace a specific string in a memory block with a new string. "
                "Use for precise edits. The old_str must match exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": f"Block to edit. One of: {', '.join(BLOCK_LABELS)}",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to find and replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["label", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": (
                "Insert text into a memory block. By default appends to the end. "
                "Set insert_line to insert at a specific position (0 = beginning)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": f"Block to edit. One of: {', '.join(BLOCK_LABELS)}",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Text to insert.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number to insert at. -1 (default) appends to end.",
                    },
                },
                "required": ["label", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_rethink",
            "description": (
                "Completely rewrite a memory block's contents. Use for large "
                "sweeping changes when the existing content needs major restructuring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": f"Block to rewrite. One of: {', '.join(BLOCK_LABELS)}",
                    },
                    "new_memory": {
                        "type": "string",
                        "description": "Complete new content for the block.",
                    },
                },
                "required": ["label", "new_memory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": (
                "Search the long-term vector memory for past observations, decisions, "
                "patterns, or context. Returns the most relevant memories. Use this to "
                "check if something has been noted before, or to find related context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query. Be specific.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (default 3, max 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(
    store: BlockStore,
    name: str,
    args: dict[str, Any],
    memory_store: Any | None = None,
) -> str:
    """Execute a tool call against the BlockStore. Returns a result message.

    The optional memory_store (a MemoryStore instance) enables the
    conversation_search tool to query the ASys long-term vector memory.
    """
    if name == "memory_replace":
        label = args.get("label", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        if label not in BLOCK_LABELS:
            return f"Error: Unknown block '{label}'. Valid: {', '.join(BLOCK_LABELS)}"
        ok = store.replace(label, old_str, new_str)
        if ok:
            return f"Replaced text in [{label}]."
        return f"Failed: old_str not found in [{label}], or block would exceed limit."

    elif name == "memory_insert":
        label = args.get("label", "")
        new_str = args.get("new_str", "")
        insert_line = args.get("insert_line", -1)
        if label not in BLOCK_LABELS:
            return f"Error: Unknown block '{label}'. Valid: {', '.join(BLOCK_LABELS)}"
        ok = store.insert(label, new_str, insert_line)
        if ok:
            return f"Inserted text into [{label}]."
        return f"Failed: [{label}] would exceed character limit."

    elif name == "memory_rethink":
        label = args.get("label", "")
        new_memory = args.get("new_memory", "")
        if label not in BLOCK_LABELS:
            return f"Error: Unknown block '{label}'. Valid: {', '.join(BLOCK_LABELS)}"
        ok = store.rethink(label, new_memory)
        if ok:
            return f"Rewrote [{label}] completely."
        return f"Failed: new content exceeds character limit for [{label}]."

    elif name == "conversation_search":
        return _handle_conversation_search(args, memory_store)

    else:
        return f"Error: Unknown tool '{name}'."


def _handle_conversation_search(args: dict[str, Any], memory_store: Any | None) -> str:
    """Search the ASys vector memory store and format results."""
    if memory_store is None:
        return "Error: vector memory not available."

    query = args.get("query", "").strip()
    if not query:
        return "Error: empty search query."

    k = min(args.get("k", 3), 5)
    if k < 1:
        k = 1

    try:
        results = memory_store.search(query, k=k)
    except Exception as e:
        log.warning("conversation_search failed: %s", e)
        return f"Error: search failed — {e}"

    if not results:
        return "No results found."

    from .blocks import _sanitize_content

    lines = [f"Found {len(results)} result(s):"]
    for i, scored in enumerate(results, 1):
        rec = scored.record
        created = rec.created_at.strftime("%Y-%m-%d")
        tags_str = ", ".join(rec.tags) if rec.tags else "none"
        content = _sanitize_content(rec.content)
        if len(content) > 300:
            content = content[:297] + "..."
        lines.append(
            f"{i}. [{rec.memory_type.value}] {content}\n"
            f"   (importance: {rec.importance:.1f}, created: {created}, "
            f"tags: {tags_str}, relevance: {scored.relevance:.2f})"
        )
    return "\n".join(lines)
