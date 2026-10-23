"""Any OpenAI-compatible endpoint: OpenAI, DeepSeek, Ollama, OpenRouter, Groq."""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import AsyncOpenAI

from memora.llm.base import LLMBackend, extract_json_block


class OpenAICompatBackend(LLMBackend):
    def __init__(self, base_url: Optional[str], model: str,
                 api_key: Optional[str] = None):
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "unused",
            base_url=base_url,
        )

    async def complete_json(self, system: str, user: str) -> dict:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return extract_json_block(response.choices[0].message.content)

    async def complete_tools(self, system: str, user: str,
                             tools: list, dispatch, max_rounds: int = 8) -> str:
        oa_tools = [{"type": "function", "function": t} for t in tools]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for _ in range(max_rounds):
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=oa_tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                result = await dispatch(tc.function.name,
                                        json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        raise RuntimeError(f"tool loop exceeded {max_rounds} rounds")
