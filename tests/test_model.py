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
