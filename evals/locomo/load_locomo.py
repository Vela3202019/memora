"""LoCoMo dataset loader.

Dataset: https://github.com/snap-research/locomo (data/locomo10.json),
downloaded to evals/locomo/data/locomo10.json. 10 conversations between two
named peers, ~2k QA pairs with category labels.

Two harness-critical departures from the normal ingestion path (see spec):
- Transcripts preserve REAL speaker names — never routed through
  ConversationSource._fmt(), which would collapse both peers to [Assistant].
- Extraction uses LOCOMO_PREAMBLE (extract BOTH speakers' memories), not the
  single-user ConversationSource preamble, which would suppress everything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).parent / "data" / "locomo10.json"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# Replaces ConversationSource.prompt_preamble for LoCoMo ingestion: both
# speakers are memory subjects, and names must survive into the extractions.
LOCOMO_PREAMBLE = (
    "This is a conversation between two people. Extract memories for BOTH "
    "named speakers: their experiences, their emotions, their self-knowledge. "
    "Always include the speaker's name in every description (e.g. \"Melanie "
    "adopted a dog\"), and always refer to people by their canonical name — "
    "never pronouns like 'she' or 'he'. Something one speaker merely heard "
    "about from the other is not their own experience; something they did or "
    "felt is."
)


@dataclass
class LocomoQuestion:
    question: str
    answer: Optional[str]  # None for adversarial: correct behavior is to abstain
    category: int
    category_name: str
    evidence: list = field(default_factory=list)


@dataclass
class LocomoSession:
    session_id: int
    date_time: str  # ISO "YYYY-MM-DDTHH:MM:SS"
    transcript: str  # speaker-name-preserving, one "Name: text" line per turn


@dataclass
class LocomoConversation:
    conv_id: str
    speaker_a: str
    speaker_b: str
    sessions: list  # list[LocomoSession], chronological
    questions: list  # list[LocomoQuestion]


def _parse_date_time(raw: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00'."""
    return datetime.strptime(raw, "%I:%M %p on %d %B, %Y").isoformat()


def _format_turn(turn: dict) -> str:
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        text = f"{text} [shares a photo: {caption}]".strip()
    return f"{turn['speaker']}: {text}"


_SESSION_KEY = re.compile(r"^session_(\d+)$")


def _load_conversation(sample: dict) -> LocomoConversation:
    conv = sample["conversation"]
    sessions = []
    for key, turns in conv.items():
        m = _SESSION_KEY.match(key)
        if not m or not isinstance(turns, list):
            continue
        sid = int(m.group(1))
        sessions.append(LocomoSession(
            session_id=sid,
            date_time=_parse_date_time(conv[f"session_{sid}_date_time"]),
            transcript="\n".join(_format_turn(t) for t in turns),
        ))
    sessions.sort(key=lambda s: s.session_id)

    questions = [
        LocomoQuestion(
            question=q["question"],
            answer=None if q["category"] == 5 else str(q.get("answer", "")),
            category=q["category"],
            category_name=CATEGORY_NAMES[q["category"]],
            evidence=q.get("evidence", []),
        )
        for q in sample["qa"]
    ]
    return LocomoConversation(
        conv_id=sample["sample_id"],
        speaker_a=conv["speaker_a"],
        speaker_b=conv["speaker_b"],
        sessions=sessions,
        questions=questions,
    )


DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def load_locomo(path: Path = DATA_PATH) -> list:
    """Load the full dataset as a list[LocomoConversation]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {path} (data/ is gitignored). "
            f"Download it with:\n  curl -sL -o {path} {DOWNLOAD_URL}"
        )
    samples = json.loads(path.read_text())
    return [_load_conversation(s) for s in samples]
