# Contributing

Contributions are welcome, especially lazy ones.

## Getting set up

```bash
make install   # editable install + pytest + ruff
make test      # the suite runs with no model server
make lint
```

## Ground rules

Three of these are not style preferences — they are the promises the tool
makes to the person running it.

1. **Nothing moves without `--apply`.** Any new agent plans first and acts
   second, and the plan is printed in a form a human can skim.
2. **Every change to the filesystem goes through `Journal`.** If `sloth undo`
   cannot reverse it, it does not ship.
3. **The model is never trusted.** Validate its output against something real
   (a configured taxonomy, an existing extension, a clamped number) before
   acting on it. "The model said so" is not a reason.
4. **No runtime dependencies.** The standard library has been enough so far.
   Test and lint tooling is fine; anything that ends up in `install_requires`
   needs a strong argument.
5. **It works with no model.** Add a heuristic fallback, or say plainly that
   the feature needs a model (as `fridge` does) — never invent an answer.

## Adding an agent

Put it in `sloth/agents/`, give it a `plan`-shaped function that returns data
instead of performing actions, add a subcommand in `sloth/cli.py`, and test it
with the `fake_model` fixture from `tests/conftest.py`. Tests must not require
a model server, a network, or a GPU.

## Adding a model backend

`sloth/model.py` is the only file that knows HTTP exists. Implement the
`Model` protocol (`available` and `chat`, plus schema-constrained output if the
server supports it) and keep the rest of the codebase unaware of your vendor.
