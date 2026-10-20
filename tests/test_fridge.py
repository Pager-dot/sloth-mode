import json

import pytest

from sloth.agents import fridge
from sloth.model import ModelUnavailable, NullModel


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "fridge.jpg"
    path.write_bytes(b"\xff\xd8\xff pretend this is a photo")
    return path


def seen(*names):
    return json.dumps({"items": [{"name": n} for n in names]})


def test_normalise_name():
    assert fridge.normalise_name("  Oat  MILK! ") == "oat milk"
    assert fridge.normalise_name("Cheddar (block)") == "cheddar block"


def test_diff_pantry_lists_what_is_missing(config):
    items = [fridge.Item(name="milk"), fridge.Item(name="eggs")]

    missing, low = diff(items, config)

    assert "bread" in missing
    assert "milk" not in missing
    assert low == []


def test_diff_pantry_ignores_staples(config):
    config.fridge.want = ["salt", "bread"]

    missing, _ = diff([], config)

    assert missing == ["bread"]


def test_diff_pantry_flags_low_items(config):
    items = [fridge.Item(name="milk", state="nearly empty")]

    missing, low = diff(items, config)

    assert low == ["milk"]
    assert "milk" not in missing


def test_diff_pantry_matches_plurals_and_compounds(config):
    config.fridge.want = ["lemons", "frozen peas", "milk"]
    items = [
        fridge.Item(name="lemon"),
        fridge.Item(name="peas"),
        fridge.Item(name="oat milk"),
    ]

    missing, low = diff(items, config)

    assert missing == []


def test_diff_pantry_respects_an_explicit_want_list(config):
    missing, _ = fridge.diff_pantry(
        [fridge.Item(name="beer")], config.fridge, want=["beer", "crisps"]
    )

    assert missing == ["crisps"]


def diff(items, config):
    return fridge.diff_pantry(items, config.fridge)


# ---------------------------------------------------------------------------
# the vision path
# ---------------------------------------------------------------------------


def test_build_list_sends_the_photo_to_the_vision_model(photo, config, fake_model):
    model = fake_model([seen("milk", "eggs", "spinach")])

    result = fridge.build_list([photo], config, model)

    assert [item.name for item in result.seen] == ["eggs", "milk", "spinach"]
    assert "bread" in result.missing
    assert model.calls[0]["images"] == [photo]
    assert model.calls[0]["model"] == config.model.vision_name


def test_build_list_handles_a_bare_list_response(photo, config, fake_model):
    model = fake_model([json.dumps(["milk", {"name": "eggs", "state": "low"}])])

    result = fridge.build_list([photo], config, model)

    assert [item.name for item in result.seen] == ["eggs", "milk"]
    assert result.low == ["eggs"]


def test_build_list_deduplicates_sightings(photo, config, fake_model):
    model = fake_model([seen("Milk", "milk", "", "milk ")])

    result = fridge.build_list([photo], config, model)

    assert [item.name for item in result.seen] == ["milk"]


def test_build_list_refuses_without_a_vision_model(photo, config):
    with pytest.raises(ModelUnavailable, match="ollama pull"):
        fridge.build_list([photo], config, NullModel())


def test_build_list_rejects_non_images(tmp_path, config, fake_model):
    path = tmp_path / "shopping.txt"
    path.write_text("milk")

    with pytest.raises(ValueError, match="vision model"):
        fridge.build_list([path], config, fake_model([seen()]))


def test_build_list_rejects_missing_photos(tmp_path, config, fake_model):
    with pytest.raises(FileNotFoundError):
        fridge.build_list([tmp_path / "nope.jpg"], config, fake_model([seen()]))


def test_render_is_a_checklist(photo, config, fake_model):
    model = fake_model(
        [json.dumps({"items": [{"name": "milk", "state": "low"}], "notes": "it smells"})]
    )

    result = fridge.build_list([photo], config, model)
    text = result.render()

    assert "[ ] milk  (running low)" in text
    assert "[ ] bread" in text
    assert "it smells" in text
    assert "Already in there: milk" in text


def test_markdown_is_a_task_list(photo, config, fake_model):
    model = fake_model([seen("milk")])

    markdown = fridge.build_list([photo], config, model).to_markdown()

    assert markdown.startswith("# Shopping list")
    assert "- [ ] bread" in markdown
    assert "## In the fridge" in markdown


def test_nothing_needed_says_so(photo, config, fake_model):
    config.fridge.want = ["milk"]
    model = fake_model([seen("milk")])

    result = fridge.build_list([photo], config, model)

    assert result.missing == []
    assert "Go back to the sofa" in result.render()
    assert "- [x] Nothing needed" in result.to_markdown()
