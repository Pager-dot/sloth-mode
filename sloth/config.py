"""Configuration, with defaults chosen so you never have to write a config.

If you do want one, it lives at ``~/.config/sloth/config.toml`` (or
``$SLOTH_CONFIG``) and is read with the stdlib's ``tomllib``:

.. code-block:: toml

    [model]
    name = "gemma3:12b"            # or "gemma3:1b" on a small laptop
    base_url = "http://127.0.0.1:11434"
    # api = "hf"                   # hosted Gemma instead, needs HF_TOKEN

    [tidy]
    confidence_threshold = 0.7

    [tidy.categories]
    "Invoices" = ["invoice", "receipt", "billing", "statement"]
    "Guitar tabs" = ["tab", "chords", "songsterr"]
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .model import ModelConfig

APP_NAME = "sloth"

#: Default destination folders, each with keywords used by the heuristic
#: fallback and offered to the model as hints.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Admin/Invoices": ["invoice", "bill", "billing", "statement", "payment", "receipt"],
    "Admin/Tax": ["tax", "w2", "w-2", "1099", "irs", "hmrc", "p60", "itr"],
    "Admin/Contracts": [
        "contract",
        "agreement",
        "nda",
        "lease",
        "offer-letter",
        "signed",
    ],
    "Admin/Travel": ["boarding", "itinerary", "ticket", "booking", "flight", "hotel"],
    "Admin/Medical": ["prescription", "lab", "medical", "insurance", "claim"],
    "Reading/Papers": ["paper", "arxiv", "preprint", "journal", "thesis"],
    "Reading/Books": ["book", "epub", "mobi", "manual", "handbook"],
    "Reading/Notes": ["notes", "minutes", "draft", "todo", "journal-entry"],
    "Images/Screenshots": ["screenshot", "screen-shot", "capture", "screen_recording"],
    "Images/Photos": ["img", "dsc", "photo", "pxl", "whatsapp-image"],
    "Media/Audio": ["song", "track", "podcast", "album", "mix"],
    "Media/Video": ["video", "clip", "recording", "movie", "episode"],
    "Code/Projects": ["repo", "src", "patch", "diff", "dockerfile", "makefile"],
    "Code/Data": ["dataset", "export", "dump", "csv", "metrics", "results"],
    "Installers": ["installer", "setup", "dmg", "appimage", "msi", "amd64", "x86_64"],
    "Archives": ["archive", "backup", "zip", "tar", "bundle"],
}

#: Extension-only rules, applied when nothing in the name is informative.
DEFAULT_SUFFIX_MAP: dict[str, str] = {
    ".txt": "Reading/Notes",
    ".md": "Reading/Notes",
    ".rtf": "Reading/Notes",
    ".pdf": "Reading/Papers",
    ".epub": "Reading/Books",
    ".mobi": "Reading/Books",
    ".png": "Images/Screenshots",
    ".jpg": "Images/Photos",
    ".jpeg": "Images/Photos",
    ".heic": "Images/Photos",
    ".gif": "Images/Photos",
    ".webp": "Images/Photos",
    ".mp3": "Media/Audio",
    ".wav": "Media/Audio",
    ".flac": "Media/Audio",
    ".m4a": "Media/Audio",
    ".mp4": "Media/Video",
    ".mov": "Media/Video",
    ".mkv": "Media/Video",
    ".webm": "Media/Video",
    ".zip": "Archives",
    ".tar": "Archives",
    ".gz": "Archives",
    ".tgz": "Archives",
    ".bz2": "Archives",
    ".xz": "Archives",
    ".7z": "Archives",
    ".rar": "Archives",
    ".dmg": "Installers",
    ".pkg": "Installers",
    ".deb": "Installers",
    ".rpm": "Installers",
    ".exe": "Installers",
    ".msi": "Installers",
    ".appimage": "Installers",
    ".csv": "Code/Data",
    ".tsv": "Code/Data",
    ".json": "Code/Data",
    ".parquet": "Code/Data",
    ".py": "Code/Projects",
    ".ipynb": "Code/Projects",
    ".patch": "Code/Projects",
    ".diff": "Code/Projects",
    ".docx": "Admin/Contracts",
    ".xlsx": "Code/Data",
    ".pptx": "Reading/Papers",
}

#: Where low-confidence files go, so nothing is ever silently misfiled.
UNSORTED = "_Unsure"


@dataclass
class TidyConfig:
    categories: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_CATEGORIES.items()}
    )
    suffix_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SUFFIX_MAP))
    #: Below this, a file goes to ``_Unsure`` instead of a guess.
    confidence_threshold: float = 0.55
    #: How much of a text file to show the model.
    excerpt_chars: int = 900
    #: Let the model propose better filenames, not just folders.
    rename: bool = True
    unsorted_name: str = UNSORTED


@dataclass
class BriefConfig:
    #: Characters per map-reduce chunk. Small models like small chunks.
    chunk_chars: int = 6000
    max_chunks: int = 24
    #: How many lines the final brief gets. The whole point is "two lines".
    summary_lines: int = 2
    max_action_items: int = 5


@dataclass
class FridgeConfig:
    #: Things you are assumed to always have; they never hit the list.
    staples: list[str] = field(
        default_factory=lambda: ["salt", "pepper", "water", "oil", "sugar"]
    )
    #: What a stocked kitchen should contain, used to compute what's missing.
    want: list[str] = field(
        default_factory=lambda: [
            "milk",
            "eggs",
            "bread",
            "butter",
            "cheese",
            "onions",
            "garlic",
            "tomatoes",
            "rice",
            "pasta",
            "coffee",
            "yoghurt",
            "lemons",
            "frozen peas",
        ]
    )


@dataclass
class SlothConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tidy: TidyConfig = field(default_factory=TidyConfig)
    brief: BriefConfig = field(default_factory=BriefConfig)
    fridge: FridgeConfig = field(default_factory=FridgeConfig)
    state_dir: Path = field(default_factory=lambda: default_state_dir())
    source: Path | None = None

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "journal.jsonl"

    @property
    def brief_dir(self) -> Path:
        return self.state_dir / "briefs"


def default_config_path() -> Path:
    override = os.environ.get("SLOTH_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / APP_NAME / "config.toml"


def default_state_dir() -> Path:
    override = os.environ.get("SLOTH_STATE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return root / APP_NAME


def load_config(path: Path | None = None) -> SlothConfig:
    """Load config from ``path``, falling back to defaults for anything absent."""
    path = Path(path).expanduser() if path else default_config_path()
    config = SlothConfig()
    if not path.is_file():
        return config

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    config.source = path
    config.model = _merge(config.model, data.get("model"))
    config.brief = _merge(config.brief, data.get("brief"))
    config.fridge = _merge(config.fridge, data.get("fridge"))

    tidy_data = dict(data.get("tidy") or {})
    categories = tidy_data.pop("categories", None)
    suffixes = tidy_data.pop("suffix_map", None)
    config.tidy = _merge(config.tidy, tidy_data)
    if isinstance(categories, Mapping):
        # A user-supplied category table replaces the defaults outright;
        # half-inheriting someone else's taxonomy is worse than either.
        config.tidy.categories = {
            str(k): [str(x) for x in (v if isinstance(v, list) else [v])]
            for k, v in categories.items()
        }
    if isinstance(suffixes, Mapping):
        config.tidy.suffix_map.update(
            {str(k).lower(): str(v) for k, v in suffixes.items()}
        )

    state = data.get("state_dir")
    if isinstance(state, str):
        config.state_dir = Path(state).expanduser()
    return config


def _merge(obj: Any, data: Any) -> Any:
    """Return ``obj`` with any recognised keys from ``data`` applied."""
    if not isinstance(data, Mapping):
        return obj
    known = set(obj.__dataclass_fields__)
    updates = {k: v for k, v in data.items() if k in known}
    return replace(obj, **updates) if updates else obj
