"""A very small agent harness.

Small local models are bad at long free-form reasoning and surprisingly good
at "pick one tool, fill in these fields" when the answer is schema-constrained.
So this harness is deliberately minimal:

* tools are plain Python functions with a JSON-schema argument spec,
* every turn is constrained to one tool call plus a short rationale,
* the loop is bounded by ``max_steps`` and by a wall clock,
* any tool that touches the filesystem is registered read-only here; writes
  go through the journal in :mod:`sloth.journal` instead.

That last point matters: an agent in this project can *look* at anything and
*propose* anything, but it cannot move a file. Only the apply step does that,
after you have seen the plan.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .model import Model, ModelOutputError, parse_json


@dataclass(frozen=True)
class Tool:
    """A capability the agent may use, described well enough to pick it."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    run: Callable[..., Any]
    #: Read-only tools are safe to call in any mode, including dry runs.
    read_only: bool = True

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """A named collection of tools, with a prompt-ready description."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def prompt_block(self) -> str:
        """Render the toolset for a system prompt, compactly."""
        lines = []
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
            args = ", ".join(sorted((tool.parameters.get("properties") or {}).keys()))
            lines.append(f"- {tool.name}({args}): {tool.description}")
        return "\n".join(lines)


@dataclass
class Step:
    """One turn of the loop, kept so you can see what the agent did."""

    index: int
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    note: str = ""


@dataclass
class AgentRun:
    """The outcome of :func:`run_agent`."""

    answer: Any = None
    steps: list[Step] = field(default_factory=list)
    stopped_because: str = "done"

    @property
    def finished(self) -> bool:
        return self.stopped_because == "done"

    def transcript(self) -> str:
        lines = []
        for step in self.steps:
            detail = step.error or _truncate(step.result)
            lines.append(f"{step.index}. {step.tool} -> {detail}")
        return "\n".join(lines)


#: The one response shape the model is allowed to produce.
TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "description": "one short sentence"},
        "tool": {"type": "string", "description": "a tool name, or 'finish'"},
        "arguments": {"type": "object", "description": "arguments for the tool"},
    },
    "required": ["tool", "arguments"],
}

_SYSTEM_TEMPLATE = """\
You are a careful, extremely lazy assistant. You do the minimum work that
fully answers the task, then stop.

Tools:
{tools}
- finish(answer): return the final answer and end the run.

Rules:
- Reply with JSON only: {{"thought": "...", "tool": "...", "arguments": {{...}}}}
- One tool per reply. Keep "thought" under 20 words.
- Call finish as soon as you can answer. Do not explore for its own sake.
- Never invent tool results. If a tool errors, adapt or finish.

{extra}"""


def run_agent(
    model: Model,
    task: str,
    tools: ToolRegistry,
    *,
    max_steps: int = 8,
    time_budget: float = 120.0,
    extra_instructions: str = "",
    allow_writes: bool = False,
    on_step: Callable[[Step], None] | None = None,
) -> AgentRun:
    """Run the loop until the model calls ``finish`` or a budget runs out.

    ``allow_writes`` gates tools marked ``read_only=False``; by default a run
    can read the world but not change it.
    """
    system = _SYSTEM_TEMPLATE.format(
        tools=tools.prompt_block() or "- (none)",
        extra=extra_instructions.strip(),
    ).strip()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    run = AgentRun()
    deadline = time.monotonic() + time_budget

    for index in range(1, max_steps + 1):
        if time.monotonic() > deadline:
            run.stopped_because = "time budget exhausted"
            return run

        raw = model.chat(messages, schema=TURN_SCHEMA)
        try:
            turn = parse_json(raw)
        except ModelOutputError as exc:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"That was not valid JSON ({exc}). Retry."}
            )
            continue
        if not isinstance(turn, Mapping):
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Reply with a JSON object."})
            continue

        name = str(turn.get("tool") or "").strip()
        arguments = turn.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        step = Step(
            index=index,
            tool=name,
            arguments=dict(arguments),
            note=str(turn.get("thought") or "")[:200],
        )

        if name in ("finish", "final_answer", "done"):
            step.result = arguments.get("answer", arguments or None)
            run.steps.append(step)
            if on_step:
                on_step(step)
            run.answer = step.result
            run.stopped_because = "done"
            return run

        try:
            tool = tools.get(name)
            if not tool.read_only and not allow_writes:
                raise PermissionError(f"{name} may not run in read-only mode")
            step.result = tool.run(**step.arguments)
        except Exception as exc:  # tools are user code; surface, don't crash
            step.error = f"{type(exc).__name__}: {exc}"

        run.steps.append(step)
        if on_step:
            on_step(step)

        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": f"{name} returned: {_truncate(step.error or step.result, 1500)}",
            }
        )

    run.stopped_because = f"hit max_steps ({max_steps})"
    return run


def _truncate(value: Any, limit: int = 300) -> str:
    if value is None:
        return "null"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"
