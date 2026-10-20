"""End-to-end tests against a real HTTP server on localhost.

Everything else fakes the model at the Python level, which leaves the actual
wire format untested: the paths, the auth header, where the JSON schema goes,
how images are attached. These start a throwaway server (the `server` fixture
in conftest.py) and check what `LocalModel` really puts on the socket.

No weights, and no traffic that leaves the loopback interface.
"""

from __future__ import annotations

import json

import pytest

from sloth.model import LocalModel, ModelConfig, ModelUnavailable

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def test_ollama_dialect_on_the_wire(server):
    server.reply["content"] = '{"ok": true}'
    model = LocalModel(ModelConfig(base_url=server.url, name="gemma3:4b", timeout=10))

    assert model.list_models() == ["gemma4:31b", "gpt-oss:120b"]
    assert model.available() is True
    assert model.json_chat([{"role": "user", "content": "hi"}], SCHEMA) == {"ok": True}

    request = server.last
    assert request.path == "/api/chat"
    assert request.body["model"] == "gemma3:4b"
    assert request.body["stream"] is False
    # Ollama takes the schema in `format`; that is what makes small models usable.
    assert request.body["format"] == SCHEMA
    assert request.body["options"]["temperature"] == 0.0
    # A local Ollama needs no credentials, so none are sent.
    assert "Authorization" not in request.headers


def test_openai_dialect_on_the_wire(server):
    server.reply["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(api="openai", base_url=server.url, name="local-model", timeout=10)
    )

    assert model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA) == (
        '{"ok": true}'
    )

    assert server.last.path == "/v1/chat/completions"
    assert server.last.body["response_format"]["type"] == "json_schema"
    assert server.last.body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_hugging_face_sends_the_token(server):
    server.reply["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(api="hf", base_url=server.url, api_key="hf_pretend", timeout=10)
    )

    model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)

    assert server.last.path == "/v1/chat/completions"
    assert server.last.headers["Authorization"] == "Bearer hf_pretend"
    assert server.last.body["model"] == "google/gemma-3-4b-it"


def test_ollama_cloud_sends_the_bearer_token(server):
    """The hosted catalogue needs auth on the same /api/chat the laptop uses."""
    server.reply["content"] = '{"ok": true}'
    model = LocalModel(
        ModelConfig(
            api="cloud", base_url=server.url, api_key="ollama_pretend", timeout=10
        )
    )

    model.chat([{"role": "user", "content": "hi"}], schema=SCHEMA)

    assert server.last.path == "/api/chat"  # the Ollama dialect, not /v1/...
    assert server.last.headers["Authorization"] == "Bearer ollama_pretend"
    assert server.last.body["model"] == "gpt-oss:120b"
    assert server.last.body["format"] == SCHEMA


def test_a_request_without_a_key_carries_no_header(server):
    """A secrets proxy may add the header for us, so we must not invent one."""
    server.reply["content"] = "{}"
    model = LocalModel(ModelConfig(api="cloud", base_url=server.url, timeout=10))

    assert model.config.needs_token is True
    model.chat([{"role": "user", "content": "hi"}])

    assert "Authorization" not in server.last.headers


@pytest.mark.parametrize(
    "api,service,help_text",
    [
        ("cloud", "Ollama", "ollama.com/settings/keys"),
        ("hf", "Hugging Face", "huggingface.co/settings/tokens"),
    ],
)
def test_a_rejected_token_says_so_plainly(server, api, service, help_text):
    server.reply.update(status=401, error="Unauthorized")
    model = LocalModel(
        ModelConfig(api=api, base_url=server.url, api_key="wrong", timeout=10)
    )

    with pytest.raises(ModelUnavailable) as caught:
        model.chat([{"role": "user", "content": "hi"}])

    message = str(caught.value)
    assert f"{service} rejected the token" in message
    assert help_text in message  # and how to get a working one


def test_server_errors_surface_with_the_status(server):
    server.reply.update(status=500, error="out of memory")
    model = LocalModel(ModelConfig(base_url=server.url, timeout=10))

    with pytest.raises(ModelUnavailable, match="HTTP 500"):
        model.chat([{"role": "user", "content": "hi"}])
    assert model.available() is True  # listing still works; only chat failed


def test_probe_makes_exactly_one_real_call(server):
    server.reply["content"] = "ok"
    model = LocalModel(
        ModelConfig(api="cloud", base_url=server.url, api_key="k", timeout=10)
    )

    assert model.probe() == "ok"
    assert len(server.posts) == 1


def test_images_reach_ollama_as_base64(server, tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"\xff\xd8\xff pretend jpeg")
    server.reply["content"] = '{"items": []}'
    model = LocalModel(ModelConfig(base_url=server.url, timeout=10))

    model.chat(
        [{"role": "user", "content": "what is in here"}],
        model="gemma3:4b",
        images=[photo],
    )

    assert server.last.body["model"] == "gemma3:4b"
    assert len(server.last.body["messages"][-1]["images"]) == 1


def test_images_reach_openai_servers_as_data_urls(server, tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"\xff\xd8\xff pretend jpeg")
    server.reply["content"] = '{"items": []}'
    model = LocalModel(ModelConfig(api="openai", base_url=server.url, timeout=10))

    model.chat([{"role": "user", "content": "what is in here"}], images=[photo])

    parts = server.last.body["messages"][-1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_tidy_agent_over_real_http(server, config, messy_dir):
    """The whole validation layer, against a server that answers badly."""
    from sloth.agents import tidy
    from sloth.util import iter_files

    config.model = ModelConfig(api="openai", base_url=server.url, timeout=10).resolved()
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
    server.reply["content"] = json.dumps(answers)

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
