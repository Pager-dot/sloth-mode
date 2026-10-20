import json
from pathlib import Path

import pytest

from sloth.agents import brief
from sloth.model import NullModel

CHAT = """\
[23:14] Priya: so the venue cancelled on us for saturday
[23:15] Tom: what?? did they say why
[23:15] Priya: double booking. they offered sunday instead
[23:16] Tom: lol classic
[23:18] Priya: can you call them tomorrow morning and confirm sunday? you have the booking ref
[23:19] Dev: i can bring the speakers either day
[23:20] Priya: also the deposit is due friday, don't forget
[23:21] Tom: 👍
[23:22] Dev: haha
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# reading input
# ---------------------------------------------------------------------------


def test_collect_sources_reads_text_and_counts_messages(tmp_path):
    write(tmp_path, "chat.txt", CHAT)
    write(tmp_path, "photo.png", "not text")

    collected = brief.collect_sources([tmp_path])

    assert len(collected) == 1
    source, text = collected[0]
    assert source.path.name == "chat.txt"
    assert source.messages == 9
    assert "venue cancelled" in text


def test_collect_sources_accepts_a_single_file(tmp_path):
    path = write(tmp_path, "chat.txt", CHAT)

    assert len(brief.collect_sources([path])) == 1


def test_collect_sources_ignores_missing_paths(tmp_path):
    assert brief.collect_sources([tmp_path / "nope"]) == []


def test_normalise_flattens_a_json_export(tmp_path):
    payload = {
        "messages": [
            {"sender_name": "Ada", "content": "the build is green"},
            {"from": "Bob", "text": "nice"},
            {"author": "Cy", "body": ""},
            "a bare string",
        ]
    }
    path = write(tmp_path, "export.json", json.dumps(payload))

    text, count = brief.normalise(path, path.read_text())

    assert "Ada: the build is green" in text
    assert "Bob: nice" in text
    assert "a bare string" in text
    assert count == 3


def test_normalise_falls_back_for_unknown_json(tmp_path):
    path = write(tmp_path, "data.json", json.dumps({"unrelated": 1}))

    text, count = brief.normalise(path, path.read_text())

    assert "unrelated" in text
    assert count == 0


def test_chunk_text_splits_on_line_boundaries():
    text = "\n".join(f"line {i} with some words" for i in range(50))

    chunks = brief.chunk_text(text, 100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 130 for chunk in chunks)
    assert "\n".join(chunks) == text


# ---------------------------------------------------------------------------
# the model path
# ---------------------------------------------------------------------------


def test_map_reduce_brief(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model(
        [
            {
                "points": [
                    "Venue cancelled Saturday, offered Sunday",
                    "Deposit due Friday",
                ]
            },
            {
                "summary": [
                    "The venue cancelled Saturday and offered Sunday instead.",
                    "The deposit is due Friday.",
                ],
                "action_items": [
                    {"what": "Call the venue to confirm Sunday", "who_asked": "Priya"}
                ],
                "ignorable": "emoji and jokes",
            },
        ]
    )

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model
    assert len(result.summary) == 2
    assert result.action_items[0]["who_asked"] == "Priya"
    assert "Call the venue" in result.render()
    assert "emoji and jokes" in result.render()
    assert result.total_characters > 0


def test_summary_is_trimmed_to_the_configured_length(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    config.brief.summary_lines = 1
    model = fake_model(
        [
            {"points": ["a"]},
            {"summary": ["one.", "two.", "three."], "action_items": []},
        ]
    )

    result = brief.summarise([tmp_path], config, model)

    assert result.summary == ["one."]


def test_string_action_items_are_accepted(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model(
        [
            {"points": ["a"]},
            {"summary": "just a string", "action_items": ["call the venue", ""]},
        ]
    )

    result = brief.summarise([tmp_path], config, model)

    assert result.summary == ["just a string"]
    assert result.action_items == [{"what": "call the venue"}]


def test_duplicate_points_are_collapsed(tmp_path, config, fake_model):
    """Chat threads repeat themselves; the reduce prompt should not."""
    write(tmp_path, "chat.txt", CHAT)
    seen = {}

    def reduce_capture(messages):
        seen["bullets"] = messages[-1]["content"]
        return json.dumps({"summary": ["ok"], "action_items": []})

    model = fake_model(
        [
            {
                "points": [
                    "Deposit due Friday",
                    "deposit due friday.",
                    "Venue moved to Sunday",
                ]
            },
            reduce_capture,
        ]
    )

    brief.summarise([tmp_path], config, model)

    assert seen["bullets"].lower().count("deposit due friday") == 1
    assert "Venue moved to Sunday" in seen["bullets"]


def test_chunking_drives_one_call_per_chunk(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    config.brief.chunk_chars = 150
    expected_chunks = len(brief.chunk_text(f"--- chat.txt ---\n{CHAT}".rstrip("\n"), 150))
    model = fake_model(
        [{"points": [f"point {i}"]} for i in range(expected_chunks)]
        + [{"summary": ["done"], "action_items": []}]
    )

    result = brief.summarise([tmp_path], config, model)

    assert result.summary == ["done"]
    assert len(model.calls) == expected_chunks + 1


def test_max_chunks_caps_the_work(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT * 4)
    config.brief.chunk_chars = 100
    config.brief.max_chunks = 2
    model = fake_model(
        [{"points": ["a"]}, {"points": ["b"]}, {"summary": ["done"], "action_items": []}]
    )

    brief.summarise([tmp_path], config, model)

    assert len(model.calls) == 3


def test_model_failure_falls_back_to_extraction(tmp_path, config, fake_model):
    from sloth.model import ModelUnavailable

    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([ModelUnavailable("no server")])

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is False
    assert result.summary


# ---------------------------------------------------------------------------
# the no-model path
# ---------------------------------------------------------------------------


def test_extractive_fallback_finds_the_asks(tmp_path, config):
    write(tmp_path, "chat.txt", CHAT)

    result = brief.summarise([tmp_path], config, NullModel())

    assert result.used_model is False
    assert len(result.summary) == config.brief.summary_lines
    asks = " ".join(item["what"] for item in result.action_items)
    assert "call them tomorrow morning" in asks
    assert "deposit is due friday" in asks.lower()
    # Speaker names are lifted out of the action item and attributed.
    assert any(item["who_asked"] == "Priya" for item in result.action_items)


def test_empty_input_says_so(tmp_path, config):
    result = brief.summarise([tmp_path], config, NullModel())

    assert "Nothing to read" in result.render()
    assert result.sources == []


def test_markdown_output_has_sources_and_actions(tmp_path, config):
    write(tmp_path, "chat.txt", CHAT)

    markdown = brief.summarise([tmp_path], config, NullModel()).to_markdown()

    assert markdown.startswith("# Morning brief")
    assert "## Needs you" in markdown
    assert "chat.txt" in markdown
    assert "without a model" in markdown


def test_render_without_color_has_no_escapes(tmp_path, config):
    write(tmp_path, "chat.txt", CHAT)

    rendered = brief.summarise([tmp_path], config, NullModel()).render()

    assert "\033[" not in rendered


# ---------------------------------------------------------------------------
# Backends that ignore the `format` schema
#
# Ollama Cloud disregards `format` on /api/chat, so the map and reduce steps
# get whatever shape the model preferred. These are verbatim shapes from
# gpt-oss:120b on examples/chats.
# ---------------------------------------------------------------------------


def test_map_step_reads_a_bare_list_of_objects(tmp_path, config, fake_model):
    """The model answers with a list of objects, not {"points": [...]}.

    The bullets must stay readable: feeding Python dict reprs into the
    reduce step is how `{'task': 'Rent', 'details': '...'}` ends up quoted
    inside a morning brief.
    """
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model(
        [
            [
                {"task": "Venue booking", "owner": "Tom", "notes": "confirm Sunday"},
                {"task": "Deposit", "notes": "due Friday"},
            ],
            {"summary": ["ok."], "action_items": []},
        ]
    )

    brief.summarise([tmp_path], config, model)

    reduce_prompt = model.calls[-1]["messages"][-1]["content"]
    assert "Venue booking" in reduce_prompt
    assert "confirm Sunday" in reduce_prompt
    assert "{'" not in reduce_prompt  # no Python reprs
    assert "task:" in reduce_prompt


def test_map_prompt_names_the_points_key(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([{"points": ["a"]}, {"summary": ["ok."], "action_items": []}])

    brief.summarise([tmp_path], config, model)

    assert "points" in model.calls[0]["messages"][0]["content"]


def test_a_run_on_summary_string_is_split_into_the_asked_for_lines(
    tmp_path, config, fake_model
):
    """`"summary"` comes back as one string holding several sentences.

    Without splitting, the two-line brief renders as a single 70-word blob.
    """
    write(tmp_path, "chat.txt", CHAT)
    config.brief.summary_lines = 2
    model = fake_model(
        [
            {"points": ["a"]},
            {
                "summary": (
                    "The boiler engineer is booked for Thursday 09:00-13:00. "
                    "The client review moved to Friday at 14:00. "
                    "Rent is short by $40."
                ),
                "action_items": [],
            },
        ]
    )

    result = brief.summarise([tmp_path], config, model)

    assert len(result.summary) == 2
    assert result.summary[0] == "The boiler engineer is booked for Thursday 09:00-13:00."
    assert result.summary[1] == "The client review moved to Friday at 14:00."


def test_reduce_prompt_asks_for_a_json_array_of_sentences(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([{"points": ["a"]}, {"summary": ["ok."], "action_items": []}])

    brief.summarise([tmp_path], config, model)

    system = model.calls[-1]["messages"][0]["content"].lower()
    assert "array" in system


# ---------------------------------------------------------------------------
# Telling the truth about why the model was not used
# ---------------------------------------------------------------------------


def test_a_failed_model_is_reported_not_silently_swallowed(tmp_path, config, fake_model):
    """A model that errors must not be described as "no model running".

    `tidy` records `model_error` and the CLI warns. `brief` fell back to the
    keyword pass and labelled the result as if no model had been configured,
    which is indistinguishable from `--no-model` output.
    """
    from sloth.model import ModelOutputError

    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([{"points": ["a"]}, ModelOutputError("reduce went wrong")])

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is False
    assert result.model_error is not None
    assert "reduce went wrong" in result.model_error
    assert "model failed" in result.ignorable


def test_no_model_configured_leaves_model_error_unset(tmp_path, config):
    write(tmp_path, "chat.txt", CHAT)

    result = brief.summarise([tmp_path], config, NullModel())

    assert result.used_model is False
    assert result.model_error is None
    assert "model unavailable" in result.ignorable


def test_empty_map_output_is_recorded_as_an_error(tmp_path, config, fake_model):
    """No bullets means the map step produced nothing usable; say so."""
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([{"note": "nothing worth reporting"}])

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is False
    assert result.model_error is not None
    assert "nothing usable" in result.model_error


# -------------------------------------------------------------------------
# degrading loudly
#
# Found against gemma4:31b on Ollama Cloud, which does not enforce the JSON
# schema: it answered the map step with {"action_items": [...]} where the
# schema said {"points": [...]}. Reading only the expected key produced an
# empty list, so the brief fell back to keyword extraction while printing
# "no model running" — a silent failure that looked like success.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"points": ["Deposit due Friday"]},
        {"action_items": ["Deposit due Friday"]},  # the gemma4 shape
        {"bullets": ["Deposit due Friday"]},
        ["Deposit due Friday"],  # a bare list, no wrapper at all
    ],
)
def test_map_step_survives_a_renamed_field(tmp_path, config, fake_model, payload):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([payload, {"summary": ["it worked"], "action_items": []}])

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is True
    assert result.model_error is None
    assert result.summary == ["it worked"]


def test_fenced_json_is_accepted(tmp_path, config, fake_model):
    """Unconstrained backends wrap answers in markdown; parse it anyway."""
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model(
        [
            '```json\n{"points": ["Deposit due Friday"]}\n```',
            '```json\n{"summary": ["it worked"], "action_items": []}\n```',
        ]
    )

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is True
    assert result.summary == ["it worked"]


def test_a_useless_model_reply_is_reported_not_hidden(tmp_path, config, fake_model):
    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([{"nothing": "useful"}])

    result = brief.summarise([tmp_path], config, model)

    assert result.used_model is False
    assert result.model_error is not None
    assert "nothing usable" in result.model_error
    # The reader must not be told "no model running" when one was running.
    assert "model failed" in result.ignorable
    assert result.summary  # the keyword pass still produced something


def test_an_unreachable_model_is_distinguished_from_a_useless_one(
    tmp_path, config, fake_model
):
    from sloth.model import ModelUnavailable

    write(tmp_path, "chat.txt", CHAT)
    model = fake_model([ModelUnavailable("server fell over")])

    result = brief.summarise([tmp_path], config, model)

    assert result.model_error == "server fell over"
    assert "model failed" in result.ignorable


def test_no_model_configured_says_unavailable_not_failed(tmp_path, config):
    write(tmp_path, "chat.txt", CHAT)

    result = brief.summarise([tmp_path], config, NullModel())

    assert result.model_error is None
    assert "model unavailable" in result.ignorable
