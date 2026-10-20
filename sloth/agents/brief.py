"""The brief agent: read the overnight pile, hand back two lines.

Point it at a folder of chat exports, meeting notes, or log dumps and it
produces the only thing you actually wanted from them: a couple of sentences
and a short list of things that need you specifically.

Uses map-reduce, because a 3B model with an 8k context cannot read your group
chat in one gulp: each chunk is summarised on its own, then the summaries are
summarised. If no model is reachable it falls back to extractive summarising
(word-frequency sentence scoring), which is worse but still beats scrolling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import BriefConfig, SlothConfig
from ..model import Model, ModelOutputError, ModelUnavailable, parse_json
from ..util import TEXT_SUFFIXES, iter_files

AGENT_NAME = "brief"

CHUNK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "at most 4 short factual bullets",
        }
    },
    "required": ["points"],
}

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "array",
            "items": {"type": "string"},
            "description": "the whole night in this many sentences",
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "who_asked": {"type": "string"},
                },
                "required": ["what"],
            },
        },
        "ignorable": {
            "type": "string",
            "description": "one clause on what was noise",
        },
    },
    "required": ["summary", "action_items"],
}

_MAP_SYSTEM = """\
You condense chat logs. Extract only what a busy person would need tomorrow:
decisions, questions aimed at them, deadlines, plans with a time attached.
Drop greetings, jokes, reactions, and link-dumps with no comment.
Be specific: names, times, numbers.

Reply with one JSON object and nothing else:
{"points": ["...", "..."]}
"points" is an array of at most 4 plain sentences. Not objects, not nested
lists, and no other top-level key.
"""

_REDUCE_SYSTEM = """\
You write a morning brief for someone who slept while all of this happened.
Given bullets extracted from overnight messages, write:
- "summary": a JSON array of exactly {lines} string(s), one sentence each.
  An array of {lines}, not one string holding {lines} sentences. Plain and
  concrete, no preamble, no "the group discussed" filler. Lead with what
  changed.
- "action_items": a JSON array of at most {max_items} objects
  {{"what": "...", "who_asked": "..."}}, things that need THIS person. Skip
  anything already handled by someone else.
- "ignorable": one clause naming what was just noise.

Reply with one JSON object and nothing else:
{{"summary": [...], "action_items": [...], "ignorable": "..."}}
"""


@dataclass
class Source:
    """One file that went into the brief."""

    path: Path
    characters: int
    messages: int = 0


@dataclass
class Brief:
    """The finished product."""

    summary: list[str] = field(default_factory=list)
    action_items: list[dict[str, str]] = field(default_factory=list)
    ignorable: str = ""
    sources: list[Source] = field(default_factory=list)
    used_model: bool = False
    #: Why the model was not used, when one was configured but did not work.
    #: ``None`` means no model was asked for in the first place.
    model_error: str | None = None
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def total_characters(self) -> int:
        return sum(s.characters for s in self.sources)

    def render(self, *, color: bool = False) -> str:
        """Plain text, suitable for a terminal or a notification."""
        dim = "\033[2m" if color else ""
        bold = "\033[1m" if color else ""
        reset = "\033[0m" if color else ""

        lines = [f"{bold}While you were asleep{reset}", ""]
        lines.extend(f"  {line}" for line in self.summary or ["Nothing happened."])
        if self.action_items:
            lines.append("")
            lines.append(f"{bold}Needs you{reset}")
            for item in self.action_items:
                who = item.get("who_asked") or ""
                suffix = f" {dim}({who}){reset}" if who else ""
                lines.append(f"  - {item.get('what', '').strip()}{suffix}")
        if self.ignorable:
            lines.append("")
            lines.append(f"{dim}Safely ignored: {self.ignorable}{reset}")

        saved = self.total_characters
        if saved:
            lines.append("")
            lines.append(
                f"{dim}Read {saved:,} characters across {len(self.sources)} file(s) "
                f"so you did not have to.{reset}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        stamp = self.generated_at.strftime("%Y-%m-%d %H:%M")
        out = [f"# Morning brief — {stamp}", ""]
        out.extend(f"{line}" for line in self.summary)
        if self.action_items:
            out += ["", "## Needs you", ""]
            for item in self.action_items:
                who = item.get("who_asked")
                out.append(f"- {item['what']}" + (f" — _{who}_" if who else ""))
        if self.ignorable:
            out += ["", f"_Safely ignored: {self.ignorable}_"]
        if self.sources:
            out += ["", "## Sources", ""]
            for source in self.sources:
                out.append(f"- `{source.path}` ({source.characters:,} chars)")
        out += [
            "",
            f"_Generated locally{'' if self.used_model else ' without a model'}._",
        ]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# reading input
# ---------------------------------------------------------------------------

#: ``Alice: hello`` / ``[09:14] Alice: hello`` / ``09:14 - Alice: hello``
_LINE_RE = re.compile(
    r"^\s*(?:\[?\d{1,2}[:/.]\d{2}(?:[:.]\d{2})?\]?\s*[-–]?\s*)?"
    r"(?P<who>[\w .'\-]{1,32}):\s(?P<text>.+)$"
)

_SENDER_KEYS = ("sender", "from", "author", "name", "user", "sender_name")
_TEXT_KEYS = ("text", "content", "message", "body", "msg")


def collect_sources(
    targets: Sequence[Path], *, recursive: bool = True
) -> list[tuple[Source, str]]:
    """Read every text-ish file under ``targets`` into ``(Source, text)`` pairs."""
    out: list[tuple[Source, str]] = []
    for target in targets:
        target = Path(target).expanduser()
        if target.is_dir():
            paths: Iterable[Path] = iter_files(target, recursive=recursive)
        elif target.is_file():
            paths = [target]
        else:
            continue
        for path in paths:
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text, count = normalise(path, raw)
            if not text.strip():
                continue
            out.append((Source(path=path, characters=len(text), messages=count), text))
    return out


def normalise(path: Path, raw: str) -> tuple[str, int]:
    """Flatten a chat export into ``who: text`` lines. Returns (text, n_messages)."""
    if path.suffix.lower() == ".json":
        flattened = _flatten_json(raw)
        if flattened:
            return "\n".join(flattened), len(flattened)

    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
    messages = sum(1 for line in lines if _LINE_RE.match(line))
    return "\n".join(lines), messages


def _flatten_json(raw: str) -> list[str]:
    """Best-effort read of the many shapes a chat export can take."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        for key in ("messages", "data", "chats", "items", "conversation"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return []
    if not isinstance(data, list):
        return []

    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        who = next(
            (str(item[k]) for k in _SENDER_KEYS if isinstance(item.get(k), str)),
            "someone",
        )
        text = next(
            (str(item[k]) for k in _TEXT_KEYS if isinstance(item.get(k), str)), None
        )
        if text and text.strip():
            out.append(f"{who}: {text.strip()}")
    return out


def chunk_text(text: str, size: int) -> list[str]:
    """Split on line boundaries so a message is never cut in half."""
    if size <= 0:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        if length + len(line) + 1 > size and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# summarising
# ---------------------------------------------------------------------------


def summarise(
    targets: Sequence[Path],
    config: SlothConfig,
    model: Model,
    *,
    recursive: bool = True,
) -> Brief:
    """Produce a :class:`Brief` for everything under ``targets``."""
    collected = collect_sources(targets, recursive=recursive)
    brief = Brief(sources=[source for source, _ in collected])
    if not collected:
        brief.summary = ["Nothing to read. Enjoy the quiet."]
        return brief

    settings = config.brief
    combined = "\n\n".join(
        f"--- {source.path.name} ---\n{text}" for source, text in collected
    )
    chunks = chunk_text(combined, settings.chunk_chars)[: settings.max_chunks]

    if model.available():
        try:
            bullets = _map_chunks(model, chunks)
            if bullets:
                reduced = _reduce(model, bullets, settings)
                brief.summary = reduced["summary"]
                brief.action_items = reduced["action_items"]
                brief.ignorable = reduced["ignorable"]
                brief.used_model = True
                return brief
            brief.model_error = "the model returned no usable points"
        except (ModelUnavailable, ModelOutputError) as exc:
            # Falling back is fine; pretending no model was configured is
            # not. The caller needs to be able to tell the two apart.
            brief.model_error = str(exc)

    brief.summary, brief.action_items = _extractive(combined, settings)
    brief.ignorable = "everything the keyword pass could not rank" + (
        " (model unavailable)" if brief.model_error else " (no model running)"
    )
    return brief


def _map_chunks(model: Model, chunks: Sequence[str]) -> list[str]:
    points: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        raw = model.chat(
            [
                {"role": "system", "content": _MAP_SYSTEM},
                {
                    "role": "user",
                    "content": f"Chunk {index} of {len(chunks)}:\n\n{chunk}",
                },
            ],
            schema=CHUNK_SCHEMA,
        )
        payload = parse_json(raw)
        if isinstance(payload, dict):
            # Unschema'd, the model may pick its own name for the one array
            # it was asked for; take it rather than losing the whole chunk.
            payload = payload.get("points") or _only_list(payload)
        if isinstance(payload, list):
            points.extend(b for b in (_as_bullet(p) for p in payload) if b)
    # Chat threads repeat themselves; the reduce step should not have to.
    return _dedupe(points)


def _only_list(payload: dict) -> list:
    """The single list in a one-key object, if that is what we were given."""
    lists = [v for v in payload.values() if isinstance(v, list)]
    return lists[0] if len(lists) == 1 else []


def _as_bullet(value: Any) -> str:
    """Render one map-step item as a sentence a reduce step can read.

    gpt-oss:120b answers with objects — ``{"task": ..., "notes": ...}`` —
    where a string was asked for. ``str(dict)`` would paste a Python repr,
    quotes and unicode escapes included, into the morning brief.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [
            f"{key}: {str(item).strip()}"
            for key, item in value.items()
            if isinstance(item, str | int | float) and str(item).strip()
        ]
        return "; ".join(parts)
    return str(value).strip()


def _reduce(model: Model, bullets: Sequence[str], settings: BriefConfig) -> dict:
    system = _REDUCE_SYSTEM.format(
        lines=settings.summary_lines, max_items=settings.max_action_items
    )
    raw = model.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(f"- {b}" for b in bullets)},
        ],
        schema=BRIEF_SCHEMA,
    )
    payload = parse_json(raw)
    if not isinstance(payload, dict):
        raise ModelOutputError("reduce step did not return an object")

    summary = payload.get("summary")
    if isinstance(summary, str):
        # Asked for an array of N sentences, an unschema'd model tends to
        # return one string holding all N. Split it back out, or the
        # two-line brief arrives as a single run-on line.
        summary = _sentences(summary)
    summary = [str(s).strip() for s in (summary or []) if str(s).strip()]

    items: list[dict[str, str]] = []
    for item in payload.get("action_items") or []:
        if isinstance(item, str) and item.strip():
            items.append({"what": item.strip()})
        elif isinstance(item, dict) and str(item.get("what", "")).strip():
            items.append(
                {
                    "what": str(item["what"]).strip(),
                    "who_asked": str(item.get("who_asked") or "").strip(),
                }
            )

    return {
        "summary": summary[: settings.summary_lines] or ["Nothing worth waking up for."],
        "action_items": items[: settings.max_action_items],
        "ignorable": str(payload.get("ignorable") or "").strip(),
    }


def _sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping their terminators."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())]
    return [s for s in parts if s]


_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "just",
        "like",
        "really",
        "ok",
        "okay",
        "yeah",
        "yes",
        "no",
        "lol",
        "haha",
        "u",
        "ur",
        "im",
        "dont",
        "cant",
        "thanks",
        "thank",
        "please",
        "hey",
        "hi",
        "hello",
        "good",
        "morning",
        "night",
    ]
)

_ACTION_HINT = re.compile(
    r"\b(can you|could you|please|don't forget|dont forget|deadline|due|by (?:mon|tue|wed|thu|fri|sat|sun|tomorrow|today)|need you|your turn|remind|send me|review|sign|rsvp|confirm)\b",
    re.IGNORECASE,
)


def _extractive(
    text: str, settings: BriefConfig
) -> tuple[list[str], list[dict[str, str]]]:
    """No-model fallback: rank sentences by informative word frequency."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", text)
        if 25 <= len(s.strip()) <= 400
    ]
    if not sentences:
        return (["Too little text to summarise."], [])

    frequency: dict[str, int] = {}
    for sentence in sentences:
        for word in re.findall(r"[a-z']{3,}", sentence.lower()):
            if word not in _STOPWORDS:
                frequency[word] = frequency.get(word, 0) + 1

    def score(sentence: str) -> float:
        words = [w for w in re.findall(r"[a-z']{3,}", sentence.lower())]
        if not words:
            return 0.0
        return sum(frequency.get(w, 0) for w in words) / (len(words) ** 0.5)

    ranked = sorted(sentences, key=score, reverse=True)
    summary = [_without_timestamp(s) for s in _dedupe(ranked)[: settings.summary_lines]]

    actions = [
        {"what": _strip_speaker(s), "who_asked": _speaker(s)}
        for s in sentences
        if _ACTION_HINT.search(s)
    ]
    return summary, _dedupe_items(actions)[: settings.max_action_items]


def _without_timestamp(line: str) -> str:
    """``[22:49] Priya: take the bins out`` -> ``Priya: take the bins out``."""
    match = _LINE_RE.match(line)
    if not match:
        return line
    return f"{match.group('who').strip()}: {match.group('text').strip()}"


def _speaker(line: str) -> str:
    match = _LINE_RE.match(line)
    return match.group("who").strip() if match else ""


def _strip_speaker(line: str) -> str:
    match = _LINE_RE.match(line)
    return (match.group("text") if match else line).strip()


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = re.sub(r"\W+", "", item.lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_items(items: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in items:
        key = re.sub(r"\W+", "", item.get("what", "").lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
