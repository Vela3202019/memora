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


def test_ask_unsupported_backend_raises():
    with pytest.raises(NotImplementedError):
        asyncio.run(qa.ask("q", llm_config=LLMConfig(backend="openai-compatible")))
