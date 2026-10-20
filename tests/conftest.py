"""Shared fixtures. No model server is started; we fake one."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# ---------------------------------------------------------------------------
# A throwaway HTTP server, for the things a FakeModel cannot prove
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """One request the stub server saw."""

    method: str
    path: str
    headers: dict
    body: dict | None = None


@dataclass
class StubServer:
    """A local model server that records what was sent to it.

    `url` goes straight into ModelConfig.base_url; set `reply` to control the
    next response, and read `received` to assert on the wire format.
    """

    url: str
    received: list[Request] = field(default_factory=list)
    reply: dict = field(default_factory=dict)

    @property
    def last(self) -> Request:
        return self.received[-1]

    @property
    def posts(self) -> list[Request]:
        return [r for r in self.received if r.method == "POST"]


@pytest.fixture
def server():
    state = StubServer(url="")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            state.received.append(Request("GET", self.path, dict(self.headers)))
            if self.path == "/api/tags":
                self._respond(
                    {"models": [{"name": "gpt-oss:120b"}, {"name": "gemma4:31b"}]}
                )
            else:
                self._respond({"data": [{"id": "google/gemma-3-4b-it"}]})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            state.received.append(Request("POST", self.path, dict(self.headers), body))

            status = state.reply.get("status", 200)
            content = state.reply.get("content", "{}")
            if status != 200:
                self._respond({"error": state.reply.get("error", "nope")}, status)
            elif self.path == "/api/chat":
                self._respond({"message": {"content": content}})
            else:
                self._respond({"choices": [{"message": {"content": content}}]})

        def log_message(self, *args) -> None:  # keep pytest output clean
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
