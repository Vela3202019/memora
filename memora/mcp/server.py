"""
MCP (Model Context Protocol) server for the memory graph.
Exposes retrieval tools over stdio JSON-RPC — no extra dependencies.

Usage:
    memora mcp

Configure in your MCP client (Claude Desktop, Cursor, etc.):
    {
      "mcpServers": {
        "memora": {
          "command": "memora",
          "args": ["mcp"]
        }
      }
    }
"""

import asyncio
import json
import logging
import sys
from typing import Any

from memora.memory.retriever import (
    get_emotional_patterns,
    get_experiences_by_emotion,
    search_experiences,
    search_truths,
)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("mcp-memory")

TOOLS = [
    {
        "name": "search_experiences",
        "description": (
            "Semantically search the user's personal experiences (episodic memory) stored in a knowledge graph. "
            "Returns matching life events with descriptions, dates, locations, significance, and associated emotions. "
            "Use this when the user asks about specific events, moments, memories, or what happened at certain times."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what to search for, e.g. 'times I felt alive', 'travel experiences', 'creative moments'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_truths",
        "description": (
            "Semantically search the user's personal truths (self-knowledge) — abstracted beliefs, patterns, values, "
            "preferences, and goals distilled from their experiences. Returns matching truths "
            "and the source experiences they were derived from."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what personal truths to find, e.g. 'beliefs about relationships', 'patterns around work', 'what I value most'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_experiences_by_emotion",
        "description": (
            "Find all personal experiences that evoked a specific emotion. "
            "Use this when the user asks what makes them feel a certain way, "
            "e.g. 'what makes me feel free?' or 'when do I feel most anxious?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "emotion_name": {
                    "type": "string",
                    "description": "The emotion to look up by name, e.g. 'freedom', 'joy', 'anxiety', 'gratitude', 'aliveness'.",
                },
            },
            "required": ["emotion_name"],
        },
    },
    {
        "name": "get_emotional_patterns",
        "description": (
            "Get an overview of the user's emotional patterns — which emotions come up most often, "
            "their average intensity, and valence (positive/negative). "
            "Use this for broad questions about the user's emotional life, "
            "e.g. 'what emotions do I feel most?' or 'how am I doing emotionally?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

_handler_map = {
    "search_experiences": search_experiences,
    "search_truths": search_truths,
    "get_experiences_by_emotion": get_experiences_by_emotion,
    "get_emotional_patterns": get_emotional_patterns,
}


def _jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _send(msg: dict) -> None:
    """Write a JSON-RPC message to stdout."""
    line = json.dumps(msg, default=str, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


async def handle_request(req: dict) -> None:
    """Route a single JSON-RPC request to the right handler."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    try:
        if method == "initialize":
            _send(_jsonrpc_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "memora-memory-graph", "version": "1.0.0"},
            }))

        elif method == "notifications/initialized":
            pass  # no response needed for notifications

        elif method == "tools/list":
            _send(_jsonrpc_response(req_id, {"tools": TOOLS}))

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = _handler_map.get(tool_name)

            if not handler:
                _send(_jsonrpc_error(req_id, -32601, f"Unknown tool: {tool_name}"))
                return

            result = await handler(**tool_args)
            _send(_jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, default=str, ensure_ascii=False)}],
            }))

        elif method == "ping":
            _send(_jsonrpc_response(req_id, {}))

        else:
            _send(_jsonrpc_error(req_id, -32601, f"Unknown method: {method}"))

    except Exception as e:
        logger.exception("Error handling request %s", method)
        _send(_jsonrpc_error(req_id, -32603, str(e)))


async def main() -> None:
    """Read JSON-RPC lines from stdin, dispatch, respond on stdout."""
    logger.info("MCP Memory Graph server starting on stdio")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            await handle_request(req)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received: %s", line[:200])


if __name__ == "__main__":
    asyncio.run(main())
