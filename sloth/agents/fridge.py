"""The fridge agent: photo in, grocery list out.

Open the fridge, take one photo, close the fridge. A local vision model
(Gemma 3 at 4B and up, or ``qwen2.5vl``, ``llava``, ``moondream`` — whatever
you pulled) names what it can see; we subtract that from the list of things a
working kitchen needs and print what is missing.

This agent is the one place that genuinely needs a model: there is no honest
keyword fallback for "what is in this photograph", so if no vision model is
available it says so instead of inventing a shopping list.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import FridgeConfig, SlothConfig
from ..model import Model, ModelUnavailable
from ..util import IMAGE_SUFFIXES

AGENT_NAME = "fridge"

SEEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "string"},
                    "state": {
                        "type": "string",
                        "description": "plenty | low | nearly empty | unclear",
                    },
                },
                "required": ["name"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["items"],
}

_SYSTEM = """\
You are looking at a photo of the inside of someone's fridge or pantry.
List only food and drink you can actually see. Use plain singular grocery
words ("milk", "eggs", "spinach"), not brand names or descriptions.
Do not guess at items hidden behind others.

Reply with one JSON object and nothing else:
{"items": [{"name": "milk", "quantity": "1 carton", "state": "low"}]}
Every entry needs "name". "quantity" and "state" are optional; "state" is
one of plenty, low, nearly empty, unclear — say so only if the container is
visibly nearly empty.
"""


@dataclass
class Item:
    name: str
    quantity: str = ""
    state: str = ""

    @property
    def running_low(self) -> bool:
        return self.state.lower().strip() in {"low", "nearly empty", "almost empty"}


@dataclass
class GroceryList:
    """What the fridge has, and what it is missing."""

    photos: list[Path] = field(default_factory=list)
    seen: list[Item] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    low: list[str] = field(default_factory=list)
    notes: str = ""

    def render(self) -> str:
        lines = ["Shopping list", ""]
        if not self.missing and not self.low:
            lines.append("  Nothing. The fridge is fine. Go back to the sofa.")
        for name in self.low:
            lines.append(f"  [ ] {name}  (running low)")
        for name in self.missing:
            lines.append(f"  [ ] {name}")
        if self.seen:
            lines += ["", f"Already in there: {', '.join(i.name for i in self.seen)}"]
        if self.notes:
            lines += ["", f"Note: {self.notes}"]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = ["# Shopping list", ""]
        for name in self.low:
            out.append(f"- [ ] {name} _(running low)_")
        for name in self.missing:
            out.append(f"- [ ] {name}")
        if not self.low and not self.missing:
            out.append("- [x] Nothing needed")
        if self.seen:
            out += ["", "## In the fridge", ""]
            for item in self.seen:
                detail = " / ".join(x for x in (item.quantity, item.state) if x)
                out.append(f"- {item.name}" + (f" ({detail})" if detail else ""))
        return "\n".join(out)


def build_list(
    photos: Sequence[Path],
    config: SlothConfig,
    model: Model,
    *,
    want: Sequence[str] | None = None,
) -> GroceryList:
    """Read ``photos`` with a local vision model and diff against the want-list."""
    paths = [Path(p).expanduser() for p in photos]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"no such photo: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"not an image I can send to a vision model: {path}")

    if not model.available():
        raise ModelUnavailable(
            "the fridge agent needs a vision model. Try either:\n"
            "  ollama pull gemma3:4b      (local, ~3 GB, also does the text work)\n"
            "  sloth --api hf fridge ...  (hosted Gemma, needs a free HF_TOKEN)\n"
            "There is no honest way to guess what is in a photo without one."
        )

    result = GroceryList(photos=paths)
    payload = model.json_chat(
        [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": "What food and drink is in this fridge? List what you can see.",
            },
        ],
        SEEN_SCHEMA,
        model=config.model.vision_name,
        images=paths,
    )
    if isinstance(payload, list):
        payload = {"items": payload}
    if not isinstance(payload, dict):
        payload = {"items": []}

    result.seen = _parse_items(payload.get("items"))
    result.notes = str(payload.get("notes") or "").strip()
    result.missing, result.low = diff_pantry(result.seen, config.fridge, want=want)
    return result


#: What a model calls the one field it was asked for, when nothing makes it
#: say "name". gemma4:31b on Ollama Cloud prefers "item".
_NAME_KEYS = ("name", "item", "food", "label", "product", "title")


def _item_name(entry: dict) -> str:
    """The grocery word in ``entry``, whatever key it arrived under."""
    for key in _NAME_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_items(raw: Any) -> list[Item]:
    items: list[Item] = []
    seen_names: set[str] = set()
    for entry in raw or []:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            continue
        name = normalise_name(_item_name(entry))
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        items.append(
            Item(
                name=name,
                quantity=str(entry.get("quantity") or "").strip(),
                state=str(entry.get("state") or "").strip(),
            )
        )
    return sorted(items, key=lambda i: i.name)


def diff_pantry(
    seen: Sequence[Item],
    config: FridgeConfig,
    *,
    want: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(missing, running_low)`` names from the want-list."""
    wanted = [normalise_name(w) for w in (want if want is not None else config.want)]
    staples = {normalise_name(s) for s in config.staples}
    present = {item.name: item for item in seen}

    missing: list[str] = []
    low: list[str] = []
    for name in wanted:
        if not name or name in staples:
            continue
        match = _find(name, present)
        if match is None:
            missing.append(name)
        elif present[match].running_low:
            low.append(name)
    return missing, low


def _find(name: str, present: dict[str, Item]) -> str | None:
    """Match want-list entries to sightings, allowing plurals and compounds."""
    if name in present:
        return name
    singular = name.rstrip("s")
    for candidate in present:
        other = candidate.rstrip("s")
        if singular == other:
            return candidate
        # "frozen peas" is satisfied by "peas"; "milk" by "oat milk".
        if singular and (singular in other.split() or other in singular.split()):
            return candidate
    return None


def normalise_name(name: str) -> str:
    name = re.sub(r"[^\w\s-]", " ", name.lower()).strip()
    name = re.sub(r"\s+", " ", name)
    return name
