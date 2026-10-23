import asyncio
import json
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-dummy")  # memora.agents package import needs it

from memora.agents import qa
from memora.config import LLMConfig


def test_build_prompt_contains_question_and_top_k():
    prompt = qa._build_prompt("What is my dog's name?", top_k=7)
    assert "What is my dog's name?" in prompt
    assert "top_k=7" in prompt
    assert "I don't know" in prompt  # abstention instruction present


def test_mcp_config_points_at_memora_server():
    cfg = qa._mcp_config()
    server = cfg["mcpServers"]["memora"]
    assert server["args"] == ["-m", "memora.mcp.server"]


def test_allowed_tools_covers_all_mcp_tools():
    allowed = qa._allowed_tools()
    for tool in ["search_experiences", "search_truths",
                 "get_experiences_by_emotion", "get_emotional_patterns"]:
        assert f"mcp__memora__{tool}" in allowed


class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b""):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.stdin_data = None

    async def communicate(self, input=None):
        self.stdin_data = input
        return self._stdout, self._stderr


def _run_ask(monkeypatch, envelope, returncode=0):
    """Run ask() with a mocked claude subprocess; return (answer, cmd, proc)."""
    captured = {}

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        proc = _FakeProc(json.dumps(envelope).encode(), returncode=returncode,
                         stderr=b"boom")
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(qa.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(qa.shutil, "which", lambda _: "/usr/local/bin/claude")
    answer = asyncio.run(
        qa.ask("Where does Melanie live?",
               llm_config=LLMConfig(backend="claude-subscription"))
    )
    return answer, captured["cmd"], captured["proc"]


def test_ask_claude_cli_parses_result_event(monkeypatch):
    envelope = [{"type": "system"}, {"type": "result", "result": "  Boston  "}]
    answer, cmd, proc = _run_ask(monkeypatch, envelope)
    assert answer == "Boston"
    # correct CLI wiring
    assert cmd[:2] == ["claude", "-p"]
    assert "--mcp-config" in cmd
    assert "--strict-mcp-config" in cmd
    assert "--allowedTools" in cmd
    assert "mcp__memora__search_experiences" in cmd[cmd.index("--allowedTools") + 1]
    # question reached the prompt on stdin
    assert b"Where does Melanie live?" in proc.stdin_data


def test_ask_claude_cli_failure_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="claude -p failed"):
        _run_ask(monkeypatch, {"result": "x"}, returncode=1)


def test_ask_api_backend_routes_through_complete_tools(monkeypatch):
    """Non-claude backends go through LLMBackend.complete_tools with TOOL_DEFS."""
    captured = {}

    class FakeBackend:
        async def complete_tools(self, system, user, tools, dispatch):
            captured.update(system=system, user=user, tools=tools,
                            dispatch=dispatch)
            return "  Boston  "

    import memora.llm
    monkeypatch.setattr(memora.llm, "get_backend", lambda cfg: FakeBackend())
    answer = asyncio.run(
        qa.ask("Where does Melanie live?", top_k=9,
               llm_config=LLMConfig(backend="openai-compatible"))
    )
    assert answer == "Boston"
    assert captured["system"] == qa.SYSTEM_PROMPT
    assert "top_k=9" in captured["user"]
    assert "Where does Melanie live?" in captured["user"]
    assert captured["tools"] is qa.TOOL_DEFS


def test_dispatch_calls_retriever(monkeypatch):
    from memora.memory import retriever

    async def fake_search(query, top_k=5):
        return [{"description": f"{query}:{top_k}"}]

    monkeypatch.setattr(retriever, "search_experiences", fake_search)
    result = asyncio.run(qa._dispatch("search_experiences",
                                      {"query": "dogs", "top_k": 3}))
    assert json.loads(result) == [{"description": "dogs:3"}]


def test_dispatch_unknown_tool_returns_error():
    result = asyncio.run(qa._dispatch("nope", {}))
    assert "Unknown tool" in json.loads(result)["error"]


def test_tool_defs_match_mcp_server_tools():
    """The neutral defs must stay in lockstep with the MCP server's tool list."""
    from memora.mcp.server import TOOLS as MCP_TOOLS
    assert [t["name"] for t in qa.TOOL_DEFS] == [t["name"] for t in MCP_TOOLS]


def test_openai_compat_tool_loop(monkeypatch):
    """Two-round loop: one tool call, then a final answer."""
    from memora.llm.openai_compat import OpenAICompatBackend

    class Msg:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class ToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.type = "function"
            self.function = type("F", (), {"name": name,
                                           "arguments": arguments})()

    responses = [
        Msg(tool_calls=[ToolCall("tc1", "search_experiences",
                                 '{"query": "dog"}')]),
        Msg(content="Max"),
    ]
    seen = {"messages": None, "calls": []}

    class FakeCompletions:
        async def create(self, **kwargs):
            seen["messages"] = kwargs["messages"]
            msg = responses.pop(0)
            choice = type("C", (), {"message": msg})()
            return type("R", (), {"choices": [choice]})()

    backend = OpenAICompatBackend(base_url=None, model="test", api_key="x")
    backend._client = type("Client", (), {})()
    backend._client.chat = type("Chat", (), {})()
    backend._client.chat.completions = FakeCompletions()

    async def dispatch(name, args):
        seen["calls"].append((name, args))
        return '[{"description": "dog named Max"}]'

    answer = asyncio.run(backend.complete_tools(
        "sys", "Q", qa.TOOL_DEFS, dispatch))
    assert answer == "Max"
    assert seen["calls"] == [("search_experiences", {"query": "dog"})]
    # tool result was fed back into the conversation
    assert any(m.get("role") == "tool" for m in seen["messages"])
