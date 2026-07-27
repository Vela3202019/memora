"""Native Google Gemini backend (requires `pip install google-genai`)."""

from __future__ import annotations

import asyncio

from memora.llm.base import LLMBackend, extract_json_block


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model

    async def complete_json(self, system: str, user: str) -> dict:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai SDK not installed: pip install google-genai"
            )
        client = genai.Client()  # reads GEMINI_API_KEY
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
            ),
        )
        return extract_json_block(response.text)

    async def complete_tools(self, system: str, user: str,
                             tools: list, dispatch, max_rounds: int = 8) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai SDK not installed: pip install google-genai"
            )
        import json

        client = genai.Client()  # reads GEMINI_API_KEY
        declarations = []
        for t in tools:
            decl = {"name": t["name"], "description": t["description"]}
            # Gemini rejects empty-properties schemas; omit parameters instead
            if t["parameters"].get("properties"):
                decl["parameters"] = t["parameters"]
            declarations.append(types.FunctionDeclaration(**decl))
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
        )
        contents = [types.Content(role="user",
                                  parts=[types.Part.from_text(text=user)])]
        for _ in range(max_rounds):
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )
            calls = response.function_calls or []
            if not calls:
                return response.text or ""
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                result = await dispatch(call.name, dict(call.args or {}))
                parts.append(types.Part.from_function_response(
                    name=call.name, response={"result": json.loads(result)},
                ))
            contents.append(types.Content(role="tool", parts=parts))
        raise RuntimeError(f"tool loop exceeded {max_rounds} rounds")
