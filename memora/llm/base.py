"""LLM backend interface: one job — structured JSON completion."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMBackend(ABC):
    @abstractmethod
    async def complete_json(self, system: str, user: str) -> dict:
        """Send prompts, return the parsed JSON object from the response."""

    async def complete_tools(self, system: str, user: str,
                             tools: list, dispatch) -> str:
        """Run an agentic tool-calling loop, return the final text reply.

        `tools` is a list of neutral definitions:
            {"name": str, "description": str, "parameters": <JSON schema>}
        `dispatch(name, args) -> str` executes one tool call (async), returning
        a JSON string. Backends map the neutral defs to their SDK's tool format.
        Optional: backends that don't support tool calling leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling"
        )


def extract_json_block(text: str) -> dict:
    """Parse a JSON object out of raw model text (fences/prose tolerated)."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
