"""End-to-end tests against a real HTTP server on localhost.

Everything else fakes the model at the Python level, which leaves the actual
wire format untested: the paths, the auth header, where the JSON schema goes,
how images are attached. This file starts a throwaway server and checks what
`LocalModel` really puts on the socket.

No weights, no network beyond loopback, so it runs anywhere CI runs.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sloth.model import LocalModel, ModelConfig, ModelUnavailable

#: Requests the server saw: (method, path, headers, parsed body or None).
RECEIVED: list[tuple[str, str, dict, dict | None]] = []

#: Set per-test to control what the next POST returns.
NEXT: dict = {}


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
        RECEIVED.append(("GET", self.path, dict(self.headers), None))
        if self.path == "/api/tags":
            self._respond({"models": [{"name": "gemma3:4b"}, {"name": "gemma3:1b"}]})
        else:
            self._respond({"data": [{"id": "google/gemma-3-4b-it"}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        RECEIVED.append(("POST", self.path, dict(self.headers), body))

        if NEXT.get("status", 200) != 200:
            self._respond({"error": NEXT.get("error", "nope")}, NEXT["status"])
        elif self.path == "/api/chat":
            self._respond({"message": {"content": NEXT.get("content", "{}")}})
        else:
            self._respond(
                {"choices": [{"message": {"content": NEXT.get("content", "{}")}}]}
            )

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    RECEIVED.clear()
    NEXT.clear()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_ollama_dialect_on_the_wire(server):
    NEXT["content"] = '{"ok": true}'
    model = LocalModel(ModelConfig(base_url=server, name="gemma3:4b", timeout=10))

    assert model.list_models() == ["gemma3:1b", "gemma3:4b"]
    assert model.available() is True
    assert model.json_chat([{"role": "user", "content": "hi"}], SCHEMA) == {"ok": True}

    _, path, headers, body = RECEIVED[-1]
    assert path == "/api/chat"
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    # Ollama takes the schema in `format`; that is what makes 4B models usable.
    assert body["format"] == SCHEMA
    assert body["options"]["temperature"] == 0.0
    # Ollama needs no credentials, so none are sent.
    assert "Authorization" not in headers


def test_openai_dialect_on_the_wire(server):
    NEXT["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(api="openai", base_url=server, name="local-model", timeout=10)
    )

    assert model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA) == (
        '{"ok": true}'
    )

    _, path, _headers, body = RECEIVED[-1]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_hugging_face_sends_the_token(server):
    NEXT["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(api="hf", base_url=server, api_key="hf_pretend", timeout=10)
    )

    model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)

    _, path, headers, body = RECEIVED[-1]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer hf_pretend"
    assert body["model"] == "google/gemma-3-4b-it"


def test_a_rejected_token_says_so_plainly(server):
    NEXT.update(status=401, error="invalid credentials")
    model = LocalModel(
        ModelConfig(api="hf", base_url=server, api_key="hf_wrong", timeout=10)
    )

    with pytest.raises(ModelUnavailable, match="rejected the token"):
        model.chat([{"role": "user", "content": "hi"}])


def test_server_errors_surface_with_the_status(server):
    NEXT.update(status=500, error="out of memory")
    model = LocalModel(ModelConfig(base_url=server, timeout=10))

    with pytest.raises(ModelUnavailable, match="HTTP 500"):
        model.chat([{"role": "user", "content": "hi"}])
    assert model.available() is True  # listing still works; only chat failed


def test_images_reach_ollama_as_base64(server, tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"\xff\xd8\xff pretend jpeg")
    NEXT["content"] = '{"items": []}'
    model = LocalModel(ModelConfig(base_url=server, timeout=10))

    model.chat(
        [{"role": "user", "content": "what is in here"}],
        model="gemma3:4b",
        images=[photo],
    )

    _, _path, _headers, body = RECEIVED[-1]
    assert body["model"] == "gemma3:4b"
    assert len(body["messages"][-1]["images"]) == 1


def test_images_reach_openai_servers_as_data_urls(server, tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"\xff\xd8\xff pretend jpeg")
    NEXT["content"] = '{"items": []}'
    model = LocalModel(ModelConfig(api="openai", base_url=server, timeout=10))

    model.chat([{"role": "user", "content": "what is in here"}], images=[photo])

    _, _path, _headers, body = RECEIVED[-1]
    parts = body["messages"][-1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_tidy_agent_over_real_http(server, config, messy_dir):
    """The whole validation layer, against a server that answers badly."""
    from sloth.agents import tidy
    from sloth.util import iter_files

    config.model = ModelConfig(api="openai", base_url=server, timeout=10).resolved()
    # Every file gets the same name and a .pdf extension it may not have, and
    # the answer for notes.txt names a category that does not exist.
    answers = {
        "files": [
            {
                "filename": path.name,
                "category": (
                    "Vibes/Whatever" if path.name == "notes.txt" else "Admin/Invoices"
                ),
                "new_name": "server-picked-this.pdf",
                "confidence": 0.93,
            }
            for path in iter_files(messy_dir)
        ]
    }
    NEXT["content"] = json.dumps(answers)

    plan = tidy.plan(messy_dir, config, LocalModel(config.model))

    assert plan.used_model is True
    assert plan.model_error is None
    by_name = {p.path.name: p for p in plan.proposals}
    # The model's renames are honoured, but each file keeps its own extension.
    for name, proposal in by_name.items():
        assert proposal.new_name.endswith(name[name.rfind(".") :].lower())
    # The invented category was discarded in favour of the heuristic.
    assert by_name["notes.txt"].decided_by == "heuristic"
    assert by_name["notes.txt"].category == "Reading/Notes"
    assert [p.decided_by for p in plan.proposals].count("heuristic") == 1


def test_ollama_cloud_sends_the_bearer_token(server):
    """The hosted catalogue needs auth on the same /api/chat the laptop uses."""
    NEXT["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(api="cloud", base_url=server, api_key="ollama_pretend", timeout=10)
    )

    model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)

    _, path, headers, body = RECEIVED[-1]
    assert path == "/api/chat"  # the Ollama dialect, not /v1/...
    assert headers["Authorization"] == "Bearer ollama_pretend"
    assert body["model"] == "gpt-oss:120b"
    assert body["format"] == SCHEMA


def test_a_local_ollama_sends_no_authorization_header(server):
    NEXT["content"] = "{}"
    LocalModel(ModelConfig(base_url=server, timeout=10)).chat(
        [{"role": "user", "content": "hi"}]
    )

    _, _path, headers, _body = RECEIVED[-1]
    assert "Authorization" not in headers


def test_probe_makes_one_real_call(server):
    NEXT["content"] = "ok"
    model = LocalModel(ModelConfig(api="cloud", base_url=server, api_key="k", timeout=10))

    assert model.probe() == "ok"
    assert len([r for r in RECEIVED if r[0] == "POST"]) == 1


def test_probe_surfaces_a_rejected_cloud_token(server):
    NEXT.update(status=401, error="Unauthorized")
    model = LocalModel(
        ModelConfig(api="cloud", base_url=server, api_key="wrong", timeout=10)
    )

    with pytest.raises(ModelUnavailable, match="Ollama rejected the token"):
        model.probe()
