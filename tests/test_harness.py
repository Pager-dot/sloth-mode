import json

import pytest

from sloth.harness import Step, Tool, ToolRegistry, run_agent


def _turn(tool, **arguments):
    return json.dumps({"thought": "t", "tool": tool, "arguments": arguments})


@pytest.fixture
def registry():
    calls = []

    def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    def explode() -> None:
        raise RuntimeError("tool went wrong")

    def write_something(value: str) -> str:
        return f"wrote {value}"

    registry = ToolRegistry(
        [
            Tool(
                name="add",
                description="Add two integers.",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
                run=add,
            ),
            Tool(
                name="explode",
                description="Always fails.",
                parameters={"type": "object", "properties": {}},
                run=explode,
            ),
            Tool(
                name="write_something",
                description="Changes the world.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                run=write_something,
                read_only=False,
            ),
        ]
    )
    registry.calls = calls  # type: ignore[attr-defined]
    return registry


def test_registry_rejects_duplicates(registry):
    with pytest.raises(ValueError, match="duplicate"):
        registry.add(Tool(name="add", description="x", parameters={}, run=lambda: None))


def test_prompt_block_lists_tools_and_arguments(registry):
    block = registry.prompt_block()
    assert "- add(a, b): Add two integers." in block
    assert "write_something(value)" in block
    assert registry.names() == ["add", "explode", "write_something"]
    assert "add" in registry and len(registry) == 3


def test_run_agent_calls_tool_then_finishes(fake_model, registry):
    model = fake_model([_turn("add", a=2, b=3), _turn("finish", answer="five")])

    run = run_agent(model, "add 2 and 3", registry)

    assert run.finished
    assert run.answer == "five"
    assert registry.calls == [(2, 3)]  # type: ignore[attr-defined]
    assert [s.tool for s in run.steps] == ["add", "finish"]
    assert run.steps[0].result == 5
    assert "add -> 5" in run.transcript()
    # The tool's result is fed back to the model.
    assert "add returned: 5" in model.last_prompt


def test_tool_errors_are_reported_not_raised(fake_model, registry):
    model = fake_model([_turn("explode"), _turn("finish", answer="gave up")])

    run = run_agent(model, "try it", registry)

    assert run.answer == "gave up"
    assert run.steps[0].error == "RuntimeError: tool went wrong"
    assert "tool went wrong" in model.last_prompt


def test_unknown_tool_is_an_error_step(fake_model, registry):
    model = fake_model([_turn("teleport"), _turn("finish", answer="ok")])

    run = run_agent(model, "do it", registry)

    assert "unknown tool: teleport" in run.steps[0].error


def test_writes_are_gated_by_default(fake_model, registry):
    model = fake_model([_turn("write_something", value="x"), _turn("finish", answer="")])

    run = run_agent(model, "write", registry)

    assert "may not run in read-only mode" in run.steps[0].error


def test_writes_allowed_when_asked(fake_model, registry):
    model = fake_model(
        [_turn("write_something", value="x"), _turn("finish", answer="done")]
    )

    run = run_agent(model, "write", registry, allow_writes=True)

    assert run.steps[0].result == "wrote x"
    assert run.steps[0].error is None


def test_max_steps_stops_the_loop(fake_model, registry):
    model = fake_model([_turn("add", a=1, b=1)] * 10)

    run = run_agent(model, "forever", registry, max_steps=3)

    assert not run.finished
    assert run.stopped_because == "hit max_steps (3)"
    assert len(run.steps) == 3


def test_non_json_reply_gets_one_more_chance(fake_model, registry):
    model = fake_model(["I would love to help!", _turn("finish", answer="recovered")])

    run = run_agent(model, "task", registry)

    assert run.answer == "recovered"
    assert run.steps == [] or run.steps[-1].tool == "finish"


def test_missing_arguments_default_to_empty(fake_model, registry):
    model = fake_model([json.dumps({"tool": "explode"}), json.dumps({"tool": "finish"})])

    run = run_agent(model, "task", registry)

    assert run.finished
    assert run.steps[0].arguments == {}


def test_time_budget_is_respected(fake_model, registry):
    model = fake_model([_turn("add", a=1, b=1)] * 5)

    run = run_agent(model, "task", registry, time_budget=-1)

    assert run.stopped_because == "time budget exhausted"
    assert run.steps == []


def test_on_step_callback_sees_each_step(fake_model, registry):
    seen: list[Step] = []
    model = fake_model([_turn("add", a=1, b=1), _turn("finish", answer="done")])

    run_agent(model, "task", registry, on_step=seen.append)

    assert [s.tool for s in seen] == ["add", "finish"]
