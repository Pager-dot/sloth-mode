"""An append-only journal of every file this tool has moved.

The deal Sloth Mode makes with you: it will rename and reshuffle your
Downloads folder, and you will never have to remember what it did, because
one `sloth undo` puts the last batch back exactly where it was.

The journal is JSONL — greppable, diffable, and readable with `tail`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

MOVE = "move"
UNDO = "undo_batch"


@dataclass
class Entry:
    """One journal line."""

    action: str
    batch: str
    agent: str = ""
    src: str = ""
    dst: str = ""
    ts: float = field(default_factory=time.time)
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Entry:
        data = json.loads(line)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class UndoResult:
    """What an undo did, or would do."""

    batch: str | None
    restored: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def empty(self) -> bool:
        return self.batch is None


class Journal:
    """Append-only move log with batch-level undo."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- writing ----------------------------------------------------------

    def new_batch(self) -> str:
        return uuid.uuid4().hex[:12]

    def append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")

    def record_moves(
        self,
        agent: str,
        moves: Iterable[tuple[Path, Path]],
        batch: str | None = None,
    ) -> str:
        """Log a group of completed moves under one batch id."""
        batch = batch or self.new_batch()
        for src, dst in moves:
            self.append(
                Entry(
                    action=MOVE,
                    batch=batch,
                    agent=agent,
                    src=str(Path(src).resolve() if Path(src).is_absolute() else src),
                    dst=str(Path(dst)),
                )
            )
        return batch

    # -- reading ----------------------------------------------------------

    def entries(self) -> Iterator[Entry]:
        if not self.path.exists():
            return iter(())

        def _iter() -> Iterator[Entry]:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield Entry.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        continue  # a corrupt line should not break undo

        return _iter()

    def batches(self) -> list[str]:
        """Move batch ids, oldest first, excluding ones already undone."""
        undone = set()
        order: list[str] = []
        for entry in self.entries():
            if entry.action == UNDO:
                undone.add(entry.note or entry.batch)
            elif entry.action == MOVE and entry.batch not in order:
                order.append(entry.batch)
        return [b for b in order if b not in undone]

    def batch_moves(self, batch: str) -> list[Entry]:
        return [e for e in self.entries() if e.action == MOVE and e.batch == batch]

    def history(self, limit: int = 20) -> list[tuple[str, int, float, str]]:
        """``(batch, count, newest_ts, agent)`` for recent live batches."""
        rows: dict[str, list[Entry]] = {}
        for entry in self.entries():
            if entry.action == MOVE:
                rows.setdefault(entry.batch, []).append(entry)
        live = set(self.batches())
        out = [
            (batch, len(items), max(e.ts for e in items), items[0].agent)
            for batch, items in rows.items()
            if batch in live
        ]
        out.sort(key=lambda row: row[2], reverse=True)
        return out[:limit]

    # -- undo -------------------------------------------------------------

    def undo_last(self, dry_run: bool = False) -> UndoResult:
        """Reverse the most recent batch that has not been undone yet."""
        batches = self.batches()
        if not batches:
            return UndoResult(batch=None, dry_run=dry_run)
        return self.undo_batch(batches[-1], dry_run=dry_run)

    def undo_batch(self, batch: str, dry_run: bool = False) -> UndoResult:
        result = UndoResult(batch=batch, dry_run=dry_run)
        moves = self.batch_moves(batch)
        # Reverse order, so a batch that moved A->B->C unwinds cleanly.
        for entry in reversed(moves):
            src = Path(entry.src)
            dst = Path(entry.dst)
            if not dst.exists():
                result.skipped.append((dst, "no longer there"))
                continue
            if src.exists():
                result.skipped.append((src, "something else took that name"))
                continue
            if not dry_run:
                src.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dst, src)
            result.restored.append((dst, src))

        if not dry_run and result.restored:
            self.append(
                Entry(
                    action=UNDO,
                    batch=self.new_batch(),
                    note=batch,
                    agent="undo",
                )
            )
        return result

    def prune_vacated_dirs(self, restored: Iterable[tuple[Path, Path]]) -> int:
        """Remove exactly the folders an undo just emptied, and nothing else.

        Walks up from each vacated location, stopping at the folder the file
        was restored into, so an empty directory you created yourself is
        never swept away as a side effect.
        """
        removed = 0
        for vacated, restored_to in restored:
            stop = Path(restored_to).parent.resolve()
            current = Path(vacated).parent.resolve()
            while current != stop and stop in current.parents:
                try:
                    if any(current.iterdir()):
                        break
                    current.rmdir()
                except OSError:
                    break
                removed += 1
                current = current.parent
        return removed

    def prune_empty_dirs(self, root: Path) -> int:
        """Remove directories we created that undo left empty. Returns count."""
        removed = 0
        root = Path(root)
        if not root.is_dir():
            return 0
        for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
            path = Path(dirpath)
            if path == root:
                continue
            # os.walk's listings were taken before we deleted anything, so
            # re-check: a parent becomes empty once its children are gone.
            try:
                if not any(path.iterdir()):
                    path.rmdir()
                    removed += 1
            except OSError:
                pass
        return removed
