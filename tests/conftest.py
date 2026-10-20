"""Shared fixtures. No model server is started; we fake one."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from sloth.config import SlothConfig
from sloth.model import parse_json


class FakeModel:
    """A model that replays canned responses, and records what it was asked.

    Each response is either a string (returned verbatim), a JSON-serialisable
    object (dumped), or a callable taking the message list.
    """

    def __init__(self, responses: Sequence[Any] = (), *, available: bool = True) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._available = available

    # -- the Model protocol ----------------------------------------------

    def available(self) -> bool:
        return self._available

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "schema": schema,
                "model": model,
                "images": list(images or []),
            }
        )
        if not self.responses:
            raise AssertionError(
                f"FakeModel ran out of responses after {len(self.calls)} call(s)"
            )
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    def json_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
        *,
        model: str | None = None,
        images: Sequence[Path] | None = None,
    ) -> Any:
        return parse_json(self.chat(messages, schema=schema, model=model, images=images))

    # -- assertions helpers ----------------------------------------------

    @property
    def last_prompt(self) -> str:
        return "\n".join(str(m.get("content")) for m in self.calls[-1]["messages"])


@pytest.fixture
def fake_model() -> Callable[..., FakeModel]:
    def make(responses: Sequence[Any] = (), **kwargs: Any) -> FakeModel:
        return FakeModel(responses, **kwargs)

    return make


@pytest.fixture
def config(tmp_path: Path) -> SlothConfig:
    cfg = SlothConfig()
    # Resolved, as the CLI always hands it to the agents.
    cfg.model = cfg.model.resolved()
    cfg.state_dir = tmp_path / "state"
    return cfg


@pytest.fixture
def messy_dir(tmp_path: Path) -> Path:
    """A folder that looks like everybody's Downloads."""
    root = tmp_path / "Downloads"
    root.mkdir()
    files = {
        "ACME Invoice March 2024.pdf": b"%PDF-1.4 fake",
        "Screenshot 2024-03-02 at 11.04.21.png": b"\x89PNG fake",
        "notes.txt": b"random thoughts about the garden",
        "dataset_export.csv": b"id,value\n1,2\n",
        "ubuntu-24.04-desktop-amd64.iso": b"iso",
        "thing.qqq": b"mystery",
        ".DS_Store": b"junk",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return root
