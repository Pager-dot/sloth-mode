"""The tidy agent: look at a junk-drawer folder, propose a home for everything.

Flow:

1. Collect cheap facts about each file (name, extension, size, text excerpt).
2. Ask the local model, in batches, for ``{category, new_name, confidence}``
   constrained to a JSON schema.
3. Validate every answer against the configured taxonomy. Anything the model
   invents, or is unsure about, falls back to a keyword/extension heuristic.
4. Anything still below the confidence threshold goes to ``_Unsure`` rather
   than to a confident-looking wrong folder.
5. Optionally re-examine the unsure ones with the agent harness, which can
   read file contents before deciding.

Nothing moves until :func:`apply_plan`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SlothConfig, TidyConfig
from ..harness import Tool, ToolRegistry, run_agent
from ..journal import Journal
from ..model import Model, ModelOutputError, ModelUnavailable
from ..util import FileFacts, iter_files, read_excerpt, safe_move, slugify_filename

AGENT_NAME = "tidy"

#: Files per model call. Small models lose track past roughly a dozen.
BATCH_SIZE = 10

BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "category": {"type": "string"},
                    "new_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["filename", "category", "confidence"],
            },
        }
    },
    "required": ["files"],
}

_SYSTEM = """\
You file documents for someone who hates filing. For each file you are given,
choose exactly one category from the allowed list and, if the current name is
unhelpful, suggest a better one.

Rules:
- "category" MUST be copied verbatim from the allowed list. Never invent one.
- "new_name" keeps the original extension, uses lowercase-with-hyphens, and
  describes the content (e.g. "acme-invoice-2024-03.pdf"). If the existing
  name is already clear, repeat it unchanged.
- "confidence" is 0.0-1.0. Be honest: use below 0.5 when the filename alone
  could mean several things. Guessing wrong costs more than saying unsure.
- Return one entry per input file, with "filename" matching the input exactly.
- JSON only.
"""


@dataclass
class Proposal:
    """Where one file should go, and how sure we are."""

    path: Path
    category: str
    new_name: str
    confidence: float
    reason: str = ""
    #: ``"model"``, ``"heuristic"``, or ``"deep"`` — shown in the dry run.
    decided_by: str = "heuristic"

    def destination(self, root: Path) -> Path:
        return Path(root) / self.category / self.new_name

    @property
    def renamed(self) -> bool:
        return self.new_name != self.path.name

    def is_noop(self, root: Path) -> bool:
        """True if the file already sits exactly where we would put it."""
        return self.destination(root) == self.path


@dataclass
class TidyPlan:
    """A full proposal for a folder. Read it, then apply it."""

    root: Path
    proposals: list[Proposal] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    used_model: bool = False
    model_error: str | None = None

    @property
    def actionable(self) -> list[Proposal]:
        return [p for p in self.proposals if not p.is_noop(self.root)]

    def by_category(self) -> dict[str, list[Proposal]]:
        grouped: dict[str, list[Proposal]] = {}
        for proposal in self.actionable:
            grouped.setdefault(proposal.category, []).append(proposal)
        return dict(sorted(grouped.items()))

    def to_json(self) -> str:
        return json.dumps(
            {
                "root": str(self.root),
                "used_model": self.used_model,
                "model_error": self.model_error,
                "moves": [
                    {
                        "from": str(p.path),
                        "to": str(p.destination(self.root)),
                        "category": p.category,
                        "confidence": round(p.confidence, 2),
                        "decided_by": p.decided_by,
                        "reason": p.reason,
                    }
                    for p in self.actionable
                ],
                "skipped": [{"path": str(p), "why": why} for p, why in self.skipped],
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def plan(
    root: Path,
    config: SlothConfig,
    model: Model,
    *,
    recursive: bool = False,
    include_hidden: bool = False,
    limit: int | None = None,
    deep: bool = False,
) -> TidyPlan:
    """Build a :class:`TidyPlan` for ``root`` without touching anything."""
    root = Path(root).expanduser()
    tidy_config = config.tidy
    result = TidyPlan(root=root)

    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    known_categories = set(tidy_config.categories) | {tidy_config.unsorted_name}
    facts: list[FileFacts] = []
    for path in iter_files(root, recursive=recursive, include_hidden=include_hidden):
        # Never re-sort files already sitting in one of our own folders.
        relative = path.relative_to(root).parts
        if len(relative) > 1 and "/".join(relative[:-1]) in known_categories:
            result.skipped.append((path, "already filed"))
            continue
        try:
            facts.append(FileFacts.gather(path, tidy_config.excerpt_chars))
        except OSError as exc:
            result.skipped.append((path, f"unreadable ({exc})"))
        if limit is not None and len(facts) >= limit:
            break

    if not facts:
        return result

    decisions: dict[Path, Proposal] = {}
    if model.available():
        try:
            decisions = _ask_model(model, facts, tidy_config)
            result.used_model = True
        except (ModelUnavailable, ModelOutputError) as exc:
            result.model_error = str(exc)

    for fact in facts:
        proposal = decisions.get(fact.path) or heuristic_proposal(fact, tidy_config)
        if proposal.confidence < tidy_config.confidence_threshold:
            fallback = heuristic_proposal(fact, tidy_config)
            if fallback.confidence > proposal.confidence:
                proposal = fallback
        if (
            deep
            and result.used_model
            and proposal.confidence < tidy_config.confidence_threshold
        ):
            proposal = _deep_proposal(model, fact, tidy_config) or proposal
        if proposal.confidence < tidy_config.confidence_threshold:
            proposal.category = tidy_config.unsorted_name
            proposal.reason = proposal.reason or "not confident enough to guess"
        result.proposals.append(proposal)

    result.proposals.sort(key=lambda p: (p.category, p.new_name))
    return result


def _ask_model(
    model: Model, facts: Sequence[FileFacts], config: TidyConfig
) -> dict[Path, Proposal]:
    """Query the model in batches; return only answers that validate."""
    out: dict[Path, Proposal] = {}
    allowed = _category_block(config)
    for batch in _chunk(facts, BATCH_SIZE):
        listing = "\n\n".join(
            f"### file {i + 1}\n{fact.describe()}" for i, fact in enumerate(batch)
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Allowed categories:\n{allowed}\n\nFiles to file:\n\n{listing}"
                ),
            },
        ]
        raw = model.chat(messages, schema=BATCH_SCHEMA)
        payload = _parse_batch(raw)
        by_name = {fact.path.name: fact for fact in batch}
        for item in payload:
            fact = by_name.get(str(item.get("filename", "")).strip())
            if fact is None:
                continue
            proposal = _validate(item, fact, config)
            if proposal is not None:
                out[fact.path] = proposal
    return out


def _parse_batch(raw: str) -> list[dict[str, Any]]:
    from ..model import parse_json

    payload = parse_json(raw)
    if isinstance(payload, dict):
        payload = payload.get("files", [])
    if not isinstance(payload, list):
        raise ModelOutputError(f"expected a list of files, got {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def _validate(
    item: dict[str, Any], fact: FileFacts, config: TidyConfig
) -> Proposal | None:
    """Accept a model answer only if it maps onto the real taxonomy."""
    category = _match_category(str(item.get("category", "")), config)
    if category is None:
        return None

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    new_name = fact.path.name
    if config.rename:
        suggested = str(item.get("new_name") or "").strip()
        if suggested:
            candidate = slugify_filename(suggested)
            # A suggestion that drops or changes the extension is a bug,
            # not a creative choice.
            if fact.suffix and not candidate.endswith(fact.suffix):
                candidate = f"{Path(candidate).stem}{fact.suffix}"
            if candidate and candidate not in (".", ".."):
                new_name = candidate

    return Proposal(
        path=fact.path,
        category=category,
        new_name=new_name,
        confidence=confidence,
        reason=str(item.get("reason") or "")[:200],
        decided_by="model",
    )


def _match_category(value: str, config: TidyConfig) -> str | None:
    """Map a model's category string onto a configured one, or give up."""
    value = value.strip().strip("/")
    if not value:
        return None
    if value in config.categories or value == config.unsorted_name:
        return value
    lowered = value.lower().replace("\\", "/")
    for known in config.categories:
        if known.lower() == lowered:
            return known
    # "Invoices" should still find "Admin/Invoices".
    for known in config.categories:
        if known.lower().split("/")[-1] == lowered.split("/")[-1]:
            return known
    return None


def heuristic_proposal(fact: FileFacts, config: TidyConfig) -> Proposal:
    """Decide with keywords and extensions only. No model, no network."""
    haystack = fact.path.name.lower().replace("_", "-")
    best_category: str | None = None
    best_score = 0
    for category, keywords in config.categories.items():
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score > best_score:
            best_category, best_score = category, score

    if best_category is not None:
        confidence = min(0.8, 0.6 + 0.1 * best_score)
        reason = "filename keyword match"
    elif fact.suffix in config.suffix_map:
        best_category = config.suffix_map[fact.suffix]
        confidence = 0.6
        reason = f"{fact.suffix} files go here by default"
    else:
        best_category = config.unsorted_name
        confidence = 0.2
        reason = "nothing in the name or extension to go on"

    return Proposal(
        path=fact.path,
        category=best_category,
        new_name=slugify_filename(fact.path.name) if config.rename else fact.path.name,
        confidence=confidence,
        reason=reason,
        decided_by="heuristic",
    )


def _deep_proposal(model: Model, fact: FileFacts, config: TidyConfig) -> Proposal | None:
    """Give one stubborn file to the agent harness, which can read it."""
    path = fact.path

    def peek(chars: int = 2000) -> str:
        text = read_excerpt(path, int(chars))
        return text or "(not a text file; nothing to read)"

    def stat() -> dict[str, Any]:
        return {
            "name": path.name,
            "extension": fact.suffix,
            "size_bytes": fact.size,
            "siblings": [p.name for p in list(iter_files(path.parent))[:20]],
        }

    tools = ToolRegistry(
        [
            Tool(
                name="peek_contents",
                description="Read the beginning of the file as text.",
                parameters={
                    "type": "object",
                    "properties": {"chars": {"type": "integer"}},
                },
                run=peek,
            ),
            Tool(
                name="file_stat",
                description="Size, extension, and neighbouring filenames.",
                parameters={"type": "object", "properties": {}},
                run=stat,
            ),
        ]
    )

    task = (
        f"Decide where to file {path.name!r}. Allowed categories:\n"
        f"{_category_block(config)}\n\n"
        "Inspect at most twice, then finish with "
        '{"category": "...", "new_name": "...", "confidence": 0.0, "reason": "..."}'
    )
    try:
        run = run_agent(model, task, tools, max_steps=4, time_budget=90)
    except (ModelUnavailable, ModelOutputError):
        return None

    answer = run.answer
    if isinstance(answer, str):
        try:
            from ..model import parse_json

            answer = parse_json(answer)
        except ModelOutputError:
            return None
    if not isinstance(answer, dict):
        return None

    proposal = _validate(answer, fact, config)
    if proposal is not None:
        proposal.decided_by = "deep"
    return proposal


def _category_block(config: TidyConfig) -> str:
    lines = []
    for category, keywords in config.categories.items():
        hint = ", ".join(keywords[:5])
        lines.append(f"- {category}" + (f"  (e.g. {hint})" if hint else ""))
    lines.append(f"- {config.unsorted_name}  (use when genuinely unsure)")
    return "\n".join(lines)


def _chunk(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------


def apply_plan(plan_: TidyPlan, journal: Journal) -> list[tuple[Path, Path]]:
    """Carry out ``plan_``, recording every move so ``sloth undo`` can revert it."""
    moves: list[tuple[Path, Path]] = []
    batch = journal.new_batch()
    for proposal in plan_.actionable:
        source = proposal.path
        if not source.exists():
            continue
        final = safe_move(source, proposal.destination(plan_.root))
        moves.append((source, final))
    if moves:
        journal.record_moves(AGENT_NAME, moves, batch=batch)
    return moves
