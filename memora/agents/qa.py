"""Grounded factual QA over the memory graph.

`ask()` is the recall counterpart to the chat agent's persona layer: terse,
grounded, abstaining. The answering LLM gets Memora's retrieval tools and
decides which to call; it may only answer from what the tools return.

claude-subscription backend: shells out to `claude -p` with Memora's own MCP
server attached, so Claude runs the agentic retrieval loop natively against
the same tools Claude Desktop / Cursor users get.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from memora.config import LLMConfig
from memora.llm.claude_cli import _reply_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise memory-recall assistant answering questions from a \
personal memory graph.

You have tools that search stored memories: experiences (life events), truths \
(beliefs/patterns/preferences/goals), and emotional patterns. Call whichever tools help, \
in as many rounds as you need — e.g. search again with different phrasing, or chain a \
second search on what the first returned.

Rules:
- Answer ONLY from what the tools return. Never use outside knowledge, never guess, \
never fill in details that were not retrieved.
- Be terse and factual: give the answer directly, no preamble, no reflection.
- If the retrieved memories do not contain the answer, reply exactly: I don't know
- For date questions, use the stored dates. For counting questions, count the \
retrieved items."""

# MCP tool names as claude CLI namespaces them: mcp__<server>__<tool>
MCP_SERVER_NAME = "memora"
MCP_TOOL_NAMES = [
    "search_experiences",
    "search_truths",
    "get_experiences_by_emotion",
    "get_emotional_patterns",
]


def _allowed_tools() -> str:
    return ",".join(f"mcp__{MCP_SERVER_NAME}__{t}" for t in MCP_TOOL_NAMES)


def _mcp_config() -> dict:
    """MCP config pointing claude at this environment's memora MCP server."""
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", "memora.mcp.server"],
            }
        }
    }


def _build_prompt(question: str, top_k: int) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"When searching, request top_k={top_k} results per tool call.\n\n"
        f"---\n\nQuestion: {question}"
    )


async def _ask_claude_cli(question: str, top_k: int, model: Optional[str]) -> str:
    if shutil.which("claude") is None:
        raise RuntimeError(
            "`claude` CLI not found. Install Claude Code and log in, "
            "or switch llm.backend in ~/.memora/config.yaml."
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY is set — `claude -p` will bill the API "
            "instead of your subscription. Unset it to use plan credit."
        )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="memora-mcp-", delete=False
    ) as f:
        json.dump(_mcp_config(), f)
        mcp_config_path = f.name
    try:
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--mcp-config", mcp_config_path,
            "--strict-mcp-config",
            "--allowedTools", _allowed_tools(),
        ]
        if model:
            cmd += ["--model", model]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(
            _build_prompt(question, top_k).encode("utf-8")
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {stderr.decode()[:500]}")
        return _reply_text(json.loads(stdout.decode("utf-8"))).strip()
    finally:
        Path(mcp_config_path).unlink(missing_ok=True)


async def ask(
    question: str,
    top_k: int = 5,
    llm_config: Optional[LLMConfig] = None,
) -> str:
    """Answer a factual question strictly from stored memories.

    Returns the answer text, or "I don't know" when the graph doesn't contain it.
    """
    if llm_config is None:
        from memora.config import load_config
        llm_config = load_config().llm
    if llm_config.backend == "claude-subscription":
        return await _ask_claude_cli(question, top_k, llm_config.model)
    raise NotImplementedError(
        f"ask() does not support backend {llm_config.backend!r} yet"
    )
