"""Small helpers. Stdlib only."""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Files we never touch, no matter how messy the folder is.
DEFAULT_IGNORE = (
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".localized",
)

#: Directory names we never walk into.
DEFAULT_IGNORE_DIRS = (
    ".git",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".sloth",
)

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".rb",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".sh",
    ".sql",
    ".html",
    ".css",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".tiff"}


def human_size(num_bytes: int) -> str:
    """Render a byte count the way a tired human reads it."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"  # pragma: no cover - loop always returns


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def iter_files(
    root: Path,
    recursive: bool = False,
    include_hidden: bool = False,
    ignore: Iterable[str] = DEFAULT_IGNORE,
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
) -> Iterator[Path]:
    """Yield files under ``root``, skipping the usual noise.

    Sorted, so a dry run and the apply that follows agree on the order.
    """
    ignore_set = set(ignore)
    ignore_dir_set = set(ignore_dirs)
    root = Path(root)
    if not root.is_dir():
        return

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in ignore_dir_set and (include_hidden or not d.startswith("."))
            )
            for name in sorted(filenames):
                if name in ignore_set:
                    continue
                if not include_hidden and name.startswith("."):
                    continue
                yield Path(dirpath) / name
    else:
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_file():
                continue
            if entry.name in ignore_set:
                continue
            if not include_hidden and is_hidden(entry):
                continue
            yield entry


_SLUG_STRIP = re.compile(r"[^\w\s.\-]+", re.UNICODE)
_SLUG_SPACE = re.compile(r"[\s_]+")


def slugify_filename(name: str, max_length: int = 72) -> str:
    """Turn a model's suggestion into something a filesystem will accept.

    Keeps a single extension, lowercases, hyphenates, and refuses to return
    an empty string.
    """
    name = unicodedata.normalize("NFKD", name).strip()
    name = name.replace(os.sep, "-")
    if os.altsep:
        name = name.replace(os.altsep, "-")

    stem, dot, suffix = name.rpartition(".")
    if not dot or len(suffix) > 8 or not suffix.isalnum():
        stem, suffix = name, ""

    stem = _SLUG_STRIP.sub("", stem).lower()
    stem = _SLUG_SPACE.sub("-", stem).strip("-.")
    stem = re.sub(r"-{2,}", "-", stem)
    if not stem:
        stem = "untitled"
    stem = stem[:max_length].strip("-.")
    if not stem:
        stem = "untitled"

    suffix = suffix.lower()
    return f"{stem}.{suffix}" if suffix else stem


def unique_destination(dest: Path) -> Path:
    """Return ``dest``, or ``dest`` with a ``-2`` style counter if taken."""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for counter in range(2, 1000):
        candidate = dest.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"gave up finding a free name near {dest}")  # pragma: no cover


def safe_move(src: Path, dest: Path) -> Path:
    """Move ``src`` to ``dest``, never clobbering an existing file."""
    src = Path(src)
    dest = Path(dest)
    if dest.exists() and src.resolve() == dest.resolve():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    final = unique_destination(dest)
    shutil.move(str(src), str(final))
    return final


def read_excerpt(path: Path, max_chars: int = 1200) -> str:
    """Read the first ``max_chars`` of a text-ish file, or return ''."""
    path = Path(path)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars)
    except OSError:
        return ""


@dataclass(frozen=True)
class FileFacts:
    """Everything we know about a file before asking a model about it."""

    path: Path
    size: int
    suffix: str
    modified: float
    excerpt: str = ""

    @classmethod
    def gather(cls, path: Path, excerpt_chars: int = 0) -> FileFacts:
        path = Path(path)
        stat = path.stat()
        return cls(
            path=path,
            size=stat.st_size,
            suffix=path.suffix.lower(),
            modified=stat.st_mtime,
            excerpt=read_excerpt(path, excerpt_chars) if excerpt_chars else "",
        )

    def describe(self) -> str:
        """A compact, token-cheap description for a small local model."""
        lines = [
            f"filename: {self.path.name}",
            f"extension: {self.suffix or '(none)'}",
            f"size: {human_size(self.size)}",
        ]
        if self.excerpt.strip():
            lines.append("first lines of content:")
            lines.append(self.excerpt.strip()[:1200])
        return "\n".join(lines)
