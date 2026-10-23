"""Native Anthropic API backend (requires `pip install anthropic`)."""

from __future__ import annotations

from memora.llm.base import LLMBackend, extract_json_block


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model

    async def complete_json(self, system: str, user: str) -> dict:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise RuntimeError(
                "anthropic SDK not installed: pip install anthropic"
            )
        client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        response = await client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[{
                "role": "user",
                "content": user + "\n\nRespond with ONLY the JSON object.",
            }],
        )
        return extract_json_block(response.content[0].text)

    async def complete_tools(self, system: str, user: str,
                             tools: list, dispatch, max_rounds: int = 8) -> str:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise RuntimeError(
                "anthropic SDK not installed: pip install anthropic"
            )
        client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        an_tools = [
            {"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]}
            for t in tools
        ]
        messages = [{"role": "user", "content": user}]
        for _ in range(max_rounds):
            response = await client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=system,
                tools=an_tools,
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                return "".join(
                    b.text for b in response.content if b.type == "text"
                )
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await dispatch(block.name, dict(block.input))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": results})
        raise RuntimeError(f"tool loop exceeded {max_rounds} rounds")
