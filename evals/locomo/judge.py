"""LoCoMo QA correctness judge.

Cross-family principle: the judge must be a different model family from the
answerer. The answerer is Claude (Sonnet), so the judge is GPT ("openai",
default) or DeepSeek ("deepseek") — never Claude.
"""

from __future__ import annotations

import json
import os

from openai import AsyncOpenAI

JUDGE_BACKENDS = {
    # name: (env var for key, base_url, default model)
    "openai": ("OPENAI_API_KEY", None,
               os.getenv("EVAL_JUDGE_MODEL", "gpt-5.2")),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com",
                 os.getenv("EVAL_JUDGE_MODEL", "deepseek-chat")),
}

_judge = None  # (client, model, name)


def configure_judge(name: str) -> str:
    """Set the judge backend; returns '<name>:<model>' for the manifest."""
    global _judge
    env_var, base_url, model = JUDGE_BACKENDS[name]
    api_key = os.getenv(env_var)
    if not api_key:
        raise SystemExit(f"{env_var} is required for the {name} judge")
    _judge = (AsyncOpenAI(api_key=api_key, base_url=base_url), model, name)
    return f"{name}:{model}"


def get_judge():
    if _judge is None:
        configure_judge("openai")
    return _judge


JUDGE_PROMPT = """You are grading a memory-QA system. You get a QUESTION, the \
GOLD answer, and the system's ANSWER. Decide if the ANSWER is correct.

Rules:
- The ANSWER is correct if it conveys the same fact(s) as GOLD. Wording, \
formatting, and extra harmless detail do not matter. Dates count as correct \
if they identify the same day (or the same month/year when GOLD is only that \
precise).
- The ANSWER is incorrect if it contradicts GOLD, is missing the asked-for \
fact, or answers a different question.
- ABSTAIN-EXPECTED questions (GOLD is "ABSTAIN"): the question is a trap — \
the correct behavior is to say the information isn't known/didn't happen. \
Grade correct if the ANSWER abstains or rejects the premise; grade incorrect \
if it confidently asserts an answer.

Return JSON only: {"correct": true/false, "reason": "<one short sentence>"}"""


async def judge_answer(question: str, gold: str | None, answer: str) -> dict:
    """Grade one answer. gold=None means abstention is the correct behavior."""
    payload = json.dumps({
        "QUESTION": question,
        "GOLD": "ABSTAIN" if gold is None else gold,
        "ANSWER": answer,
    }, ensure_ascii=False)
    client, model, _ = get_judge()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    verdict = json.loads(response.choices[0].message.content)
    return {"correct": bool(verdict.get("correct")),
            "reason": str(verdict.get("reason", ""))}
