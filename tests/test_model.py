import json

import pytest

from sloth.model import (
    LocalModel,
    ModelConfig,
    ModelOutputError,
    ModelUnavailable,
    NullModel,
    _attach_images,
    parse_json,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("```\n[1, 2]\n```", [1, 2]),
        ('Sure! Here you go: {"a": [1, 2]} hope that helps', {"a": [1, 2]}),
        ('[{"b": true}]', [{"b": True}]),
    ],
)
def test_parse_json_tolerates_model_habits(raw, expected):
    assert parse_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "no json here", "{broken", None])
def test_parse_json_raises_on_garbage(raw):
    with pytest.raises(ModelOutputError):
        parse_json(raw)


def test_null_model_is_never_available():
    model = NullModel()
    assert model.available() is False
    with pytest.raises(ModelUnavailable):
        model.chat([{"role": "user", "content": "hi"}])


def test_unreachable_server_raises_model_unavailable():
    # Port 1 is reserved; nothing will answer.
    model = LocalModel(ModelConfig(base_url="http://127.0.0.1:1", timeout=1.0))
    assert model.available() is False
    assert model.has_model("llama3.2") is False
    with pytest.raises(ModelUnavailable, match="could not reach a local model"):
        model.chat([{"role": "user", "content": "hi"}])


def test_config_overrides_ignore_none():
    base = ModelConfig(name="llama3.2:3b", temperature=0.0)
    updated = base.with_overrides(name=None, temperature=0.4)
    assert updated.name == "llama3.2:3b"
    assert updated.temperature == 0.4
    assert base.temperature == 0.0


def test_attach_images_ollama_style(tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"not really a jpeg")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    out = _attach_images(messages, [photo], style="ollama")

    assert "images" in out[-1]
    assert len(out[-1]["images"]) == 1
    assert "images" not in out[0]
    # The originals must not be mutated; agents reuse their prompt lists.
    assert messages[-1] == {"role": "user", "content": "u"}


def test_attach_images_openai_style(tmp_path):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"not really a jpeg")

    out = _attach_images([{"role": "user", "content": "u"}], [photo], style="openai")

    parts = out[-1]["content"]
    assert parts[0] == {"type": "text", "text": "u"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_attach_images_noop_without_images():
    messages = [{"role": "user", "content": "u"}]
    assert _attach_images(messages, None, style="ollama") == messages


def test_fake_model_records_calls(fake_model):
    model = fake_model([{"ok": True}, "plain text"])
    assert json.loads(model.chat([{"role": "user", "content": "a"}])) == {"ok": True}
    assert model.chat([{"role": "user", "content": "b"}], schema={"type": "object"}) == (
        "plain text"
    )
    assert len(model.calls) == 2
    assert model.calls[1]["schema"] == {"type": "object"}
    with pytest.raises(AssertionError):
        model.chat([{"role": "user", "content": "c"}])


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------


def test_resolved_fills_in_ollama_defaults():
    config = ModelConfig().resolved()

    assert config.api == "ollama"
    assert config.name == "gemma3:4b"
    # Gemma 3 at 4B is multimodal, so one pull covers the fridge agent too.
    assert config.vision_name == "gemma3:4b"
    assert config.base_url == "http://127.0.0.1:11434"
    assert config.is_local is True
    assert config.needs_token is False


def test_resolved_fills_in_hugging_face_defaults(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")

    config = ModelConfig(api="hf").resolved()

    assert config.api == "huggingface"
    assert config.name == "google/gemma-3-4b-it"
    assert config.base_url == "https://router.huggingface.co/v1"
    assert config.api_key == "hf_pretend"
    assert config.is_local is False
    assert config.needs_token is False


@pytest.mark.parametrize(
    "variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]
)
def test_any_conventional_token_variable_works(monkeypatch, variable):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "hf_pretend")

    assert ModelConfig(api="huggingface").resolved().api_key == "hf_pretend"


def test_explicit_values_survive_resolution(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-the-environment")

    config = ModelConfig(
        api="huggingface",
        name="google/gemma-3-27b-it",
        base_url="https://example.invalid/v1",
        api_key="explicit",
    ).resolved()

    assert config.name == "google/gemma-3-27b-it"
    assert config.base_url == "https://example.invalid/v1"
    assert config.api_key == "explicit"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        ModelConfig(api="mystery-cloud").resolved()


@pytest.mark.parametrize(
    "api,base_url,expected",
    [
        ("ollama", "", True),
        ("openai", "http://localhost:8080", True),
        ("openai", "http://127.0.0.1:1234", True),
        ("openai", "https://some-gpu-box.example.com", False),
        ("huggingface", "", False),
    ],
)
def test_is_local(api, base_url, expected):
    assert ModelConfig(api=api, base_url=base_url).resolved().is_local is expected


def test_hugging_face_without_a_token_is_unavailable(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    model = LocalModel(ModelConfig(api="hf"))

    assert model.config.needs_token is True
    assert model.available() is False
    # And it says how to fix it, rather than failing somewhere deep in urllib.
    with pytest.raises(ModelUnavailable, match="huggingface.co/settings/tokens"):
        model.chat([{"role": "user", "content": "hi"}])


def test_hugging_face_with_a_token_skips_the_network_check(monkeypatch):
    """`available()` must not make a request for a hosted backend."""
    monkeypatch.setenv("HF_TOKEN", "hf_pretend")
    model = LocalModel(ModelConfig(api="hf"))

    def explode(*_args, **_kwargs):
        raise AssertionError("available() should not call the network")

    monkeypatch.setattr(model, "_request", explode)

    assert model.available() is True


def test_unreachable_message_names_the_backend():
    ollama = LocalModel(ModelConfig(timeout=1.0, base_url="http://127.0.0.1:1"))
    with pytest.raises(ModelUnavailable, match="ollama serve"):
        ollama.chat([{"role": "user", "content": "hi"}])

    server = LocalModel(
        ModelConfig(api="openai", base_url="http://127.0.0.1:1", timeout=1.0)
    )
    with pytest.raises(ModelUnavailable, match="your local server"):
        server.chat([{"role": "user", "content": "hi"}])
