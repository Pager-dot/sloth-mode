"""The command line. Short commands, safe defaults, no surprises.

Everything that changes a file requires ``--apply``. Everything that was
changed can be undone with ``sloth undo``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .agents import brief as brief_agent
from .agents import fridge as fridge_agent
from .agents import tidy as tidy_agent
from .config import SlothConfig, default_config_path, load_config
from .journal import Journal
from .model import HF_TOKEN_HELP, LocalModel, Model, ModelUnavailable, NullModel

SLOTH = r"""
      (
       \     ___       sloth mode
        \  .'   `.     ------------
         \/  o o  \    it does the work,
         (   \_/   )   you do the nap
          \  ---  /
       ,--' )   ( `--.
      (___.'     `.___)
"""


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


class Printer:
    """Tiny stdout wrapper: colours when a terminal is watching, else plain."""

    def __init__(self, quiet: bool = False, force_color: bool | None = None) -> None:
        self.quiet = quiet
        if force_color is None:
            self.color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        else:
            self.color = force_color

    def _wrap(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, text: str) -> str:
        return self._wrap(text, "1")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def green(self, text: str) -> str:
        return self._wrap(text, "32")

    def yellow(self, text: str) -> str:
        return self._wrap(text, "33")

    def red(self, text: str) -> str:
        return self._wrap(text, "31")

    def say(self, *parts: str) -> None:
        if not self.quiet:
            print(*parts)

    def always(self, *parts: str) -> None:
        print(*parts)

    def warn(self, message: str) -> None:
        print(self.yellow(f"! {message}"), file=sys.stderr)

    def error(self, message: str) -> None:
        print(self.red(f"error: {message}"), file=sys.stderr)


def parse_duration(value: str) -> float:
    """``90`` / ``30s`` / ``15m`` / ``2h`` -> seconds."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"not a duration: {value!r} (try 15m)")
    amount = float(match.group(1))
    return amount * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sloth",
        description="Open-weight AI that does the chores you keep postponing.",
        epilog=(
            "Local by default. Nothing moves without --apply. "
            "No model to hand? Add --no-model, or --api hf for free hosted Gemma."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sloth {__version__}")
    parser.add_argument("--config", type=Path, help="path to config.toml")
    parser.add_argument("--model", help="model name (default: gemma3:4b)")
    parser.add_argument("--vision-model", help="vision model name for `fridge`")
    parser.add_argument("--base-url", help="server URL (default: Ollama's)")
    parser.add_argument(
        "--api",
        choices=("ollama", "openai", "huggingface", "hf"),
        help=(
            "where the weights run: ollama (default), openai for "
            "llama.cpp/vLLM/LM Studio, or hf for Hugging Face's free tier"
        ),
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="skip inference entirely and use heuristics only",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print results")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    tidy = subparsers.add_parser(
        "tidy",
        help="sort and rename a junk-drawer folder",
        description="Propose a home for every file in a folder. Dry run by default.",
    )
    tidy.add_argument("path", type=Path, nargs="?", default=Path.home() / "Downloads")
    tidy.add_argument("--apply", action="store_true", help="actually move the files")
    tidy.add_argument("-r", "--recursive", action="store_true")
    tidy.add_argument("--include-hidden", action="store_true")
    tidy.add_argument("--limit", type=int, help="look at no more than N files")
    tidy.add_argument(
        "--deep",
        action="store_true",
        help="use the agent harness to read ambiguous files before deciding",
    )
    tidy.add_argument(
        "--threshold",
        type=float,
        help="confidence below which a file goes to _Unsure (default 0.55)",
    )
    tidy.add_argument("--no-rename", action="store_true", help="keep existing filenames")
    tidy.add_argument("--json", action="store_true", help="print the plan as JSON")
    tidy.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
    tidy.set_defaults(func=cmd_tidy)

    brief = subparsers.add_parser(
        "brief",
        help="summarise an overnight pile of chats or notes",
        description="Read text dumps and return a two-line brief plus action items.",
    )
    brief.add_argument("paths", type=Path, nargs="+")
    brief.add_argument("--lines", type=int, help="sentences in the summary (default 2)")
    brief.add_argument("--markdown", action="store_true", help="print markdown")
    brief.add_argument(
        "--save",
        action="store_true",
        help="also write the brief to the state directory",
    )
    brief.add_argument("--no-recursive", action="store_true")
    brief.set_defaults(func=cmd_brief)

    fridge = subparsers.add_parser(
        "fridge",
        help="turn a fridge photo into a shopping list",
        description=(
            "Needs a vision model: `ollama pull gemma3:4b`, or --api hf to borrow one."
        ),
    )
    fridge.add_argument("photos", type=Path, nargs="+")
    fridge.add_argument(
        "--want",
        help="comma-separated list of what a stocked kitchen means to you",
    )
    fridge.add_argument("--markdown", action="store_true")
    fridge.add_argument("-o", "--output", type=Path, help="write the list to a file")
    fridge.set_defaults(func=cmd_fridge)

    undo = subparsers.add_parser("undo", help="put the last batch of moves back")
    undo.add_argument("--dry-run", action="store_true")
    undo.add_argument("--batch", help="undo a specific batch id from `sloth history`")
    undo.set_defaults(func=cmd_undo)

    history = subparsers.add_parser("history", help="what has been moved, and when")
    history.add_argument("-n", "--limit", type=int, default=10)
    history.set_defaults(func=cmd_history)

    nap = subparsers.add_parser(
        "nap",
        help="run tidy on a schedule while you are asleep",
        description="A foreground loop. Pair it with cron, systemd, or just leave it.",
    )
    nap.add_argument("path", type=Path, nargs="?", default=Path.home() / "Downloads")
    nap.add_argument("--every", type=parse_duration, default=3600.0)
    nap.add_argument("--apply", action="store_true", help="move files, not just report")
    nap.add_argument("--rounds", type=int, help="stop after N rounds (default: forever)")
    nap.set_defaults(func=cmd_nap)

    doctor = subparsers.add_parser("doctor", help="check the local setup")
    doctor.set_defaults(func=cmd_doctor)

    return parser


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def resolve(args: argparse.Namespace) -> tuple[SlothConfig, Model, Printer]:
    config = load_config(args.config)
    config.model = config.model.with_overrides(
        name=args.model,
        vision_name=args.vision_model,
        base_url=args.base_url,
        api=args.api,
        # `--api hf` on its own is a complete configuration: resolved() fills
        # in that backend's URL, model names and token.
    ).resolved()
    printer = Printer(quiet=args.quiet)
    model: Model = NullModel() if args.no_model else LocalModel(config.model)
    return config, model, printer


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        print(SLOTH)
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130
    except (
        ModelUnavailable,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        Printer().error(str(exc))
        return 1


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_tidy(args: argparse.Namespace) -> int:
    config, model, out = resolve(args)
    if args.threshold is not None:
        config.tidy.confidence_threshold = args.threshold
    if args.no_rename:
        config.tidy.rename = False

    if args.json:
        out.quiet = True  # stdout must be nothing but the payload
    root = args.path.expanduser()
    out.say(out.dim(f"looking at {root} ..."))
    plan = tidy_agent.plan(
        root,
        config,
        model,
        recursive=args.recursive,
        include_hidden=args.include_hidden,
        limit=args.limit,
        deep=args.deep,
    )

    if args.json:
        out.always(plan.to_json())
        return 0

    _print_plan(plan, out)
    if plan.model_error:
        out.warn(f"no model used: {plan.model_error}")
    elif not plan.used_model:
        out.say(out.dim("decided by heuristics only (no model in play)"))

    if not plan.actionable:
        out.say(out.green("nothing to do. impressive, or concerning."))
        return 0

    if not args.apply:
        out.say("")
        out.say(out.dim("dry run. re-run with --apply to move these files."))
        return 0

    if not args.yes and sys.stdin.isatty():
        answer = input(f"move {len(plan.actionable)} file(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            out.say("left everything exactly where it was.")
            return 0

    journal = Journal(config.journal_path)
    moves = tidy_agent.apply_plan(plan, journal)
    out.say(out.green(f"moved {len(moves)} file(s)."))
    out.say(out.dim("changed your mind? `sloth undo`"))
    return 0


def _print_plan(plan, out: Printer) -> None:
    grouped = plan.by_category()
    if not grouped:
        return
    width = max(
        (len(p.path.name) for ps in grouped.values() for p in ps),
        default=20,
    )
    width = min(width, 48)
    for category, proposals in grouped.items():
        out.say("")
        out.say(out.bold(category + "/"))
        for proposal in proposals:
            name = proposal.path.name
            if len(name) > width:
                name = name[: width - 1] + "…"
            arrow = "→" if proposal.renamed else " "
            target = proposal.new_name if proposal.renamed else ""
            confidence = f"{proposal.confidence:.0%}"
            tag = out.dim(f"[{proposal.decided_by} {confidence}]")
            out.say(f"  {name:<{width}} {arrow} {target:<{width}} {tag}")
    if plan.skipped:
        out.say("")
        out.say(
            out.dim(f"skipped {len(plan.skipped)} file(s) already filed or unreadable")
        )


def cmd_brief(args: argparse.Namespace) -> int:
    config, model, out = resolve(args)
    if args.lines is not None:
        config.brief.summary_lines = max(1, args.lines)

    if args.markdown:
        out.quiet = True  # stdout must be nothing but the document
    out.say(out.dim("reading so you do not have to ..."))
    result = brief_agent.summarise(
        args.paths, config, model, recursive=not args.no_recursive
    )

    if args.markdown:
        out.always(result.to_markdown())
    else:
        out.always("")
        out.always(result.render(color=out.color))

    if args.save:
        config.brief_dir.mkdir(parents=True, exist_ok=True)
        stamp = result.generated_at.strftime("%Y-%m-%d-%H%M")
        path = config.brief_dir / f"brief-{stamp}.md"
        path.write_text(result.to_markdown(), encoding="utf-8")
        out.say("")
        out.say(out.dim(f"saved to {path}"))
    return 0


def cmd_fridge(args: argparse.Namespace) -> int:
    config, model, out = resolve(args)
    want = [w.strip() for w in args.want.split(",") if w.strip()] if args.want else None
    out.say(out.dim("looking in the fridge so you can keep the door shut ..."))
    result = fridge_agent.build_list(args.photos, config, model, want=want)

    text = result.to_markdown() if args.markdown else result.render()
    if args.output:
        args.output.write_text(result.to_markdown(), encoding="utf-8")
        out.say(out.green(f"wrote {args.output}"))
    out.always("")
    out.always(text)
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    config, _, out = resolve(args)
    journal = Journal(config.journal_path)
    result = (
        journal.undo_batch(args.batch, dry_run=args.dry_run)
        if args.batch
        else journal.undo_last(dry_run=args.dry_run)
    )

    if result.empty:
        out.say("nothing to undo. the journal is clean.")
        return 0

    verb = "would restore" if args.dry_run else "restored"
    for came_from, went_back_to in result.restored:
        out.say(f"  {came_from.name} {out.dim('→')} {went_back_to}")
    out.say(out.green(f"{verb} {len(result.restored)} file(s) (batch {result.batch})"))

    if not args.dry_run and result.restored:
        emptied = journal.prune_vacated_dirs(result.restored)
        if emptied:
            out.say(out.dim(f"and removed {emptied} folder(s) it had created"))
    for path, why in result.skipped:
        out.warn(f"skipped {path.name}: {why}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    config, _, out = resolve(args)
    rows = Journal(config.journal_path).history(limit=args.limit)
    if not rows:
        out.say("no moves recorded yet.")
        return 0
    out.say(out.bold(f"{'batch':<14}{'when':<22}{'files':>6}  agent"))
    for batch, count, newest, agent in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))
        out.say(f"{batch:<14}{when:<22}{count:>6}  {agent}")
    out.say("")
    out.say(out.dim("undo any of them with `sloth undo --batch <id>`"))
    return 0


def cmd_nap(args: argparse.Namespace) -> int:
    config, model, out = resolve(args)
    journal = Journal(config.journal_path)
    root = args.path.expanduser()
    every = args.every
    out.always(
        out.dim(f"napping. checking {root} every {int(every)}s. ctrl-c to wake up.")
    )

    round_number = 0
    while args.rounds is None or round_number < args.rounds:
        round_number += 1
        stamp = time.strftime("%H:%M:%S")
        try:
            plan = tidy_agent.plan(root, config, model)
        except (NotADirectoryError, OSError) as exc:
            out.warn(f"{stamp} {exc}")
        else:
            count = len(plan.actionable)
            if count and args.apply:
                moves = tidy_agent.apply_plan(plan, journal)
                out.always(f"{stamp} filed {len(moves)} file(s)")
            elif count:
                out.always(f"{stamp} {count} file(s) could be filed (no --apply)")
            else:
                out.say(out.dim(f"{stamp} nothing new"))

        if args.rounds is not None and round_number >= args.rounds:
            break
        time.sleep(every)
    return 0


def _fix_hints(model_config) -> list[str]:
    """What to type next, depending on which backend went quiet."""
    if model_config.api == "ollama":
        return [
            "try:  ollama serve",
            f"      ollama pull {model_config.name}",
            "      (or `sloth --api hf doctor` to use Hugging Face instead)",
        ]
    if model_config.api == "huggingface":
        return ["check your connection, then:  sloth --api hf doctor"]
    return [f"start your server on {model_config.base_url}, then run doctor again"]


def cmd_doctor(args: argparse.Namespace) -> int:
    config, model, out = resolve(args)
    out.always(SLOTH)
    out.always(out.bold("setup"))
    out.always(f"  sloth           {__version__}")
    out.always(f"  python          {sys.version.split()[0]}")

    config_path = args.config or default_config_path()
    found = "found" if Path(config_path).is_file() else "not found (using defaults)"
    out.always(f"  config          {config_path} {out.dim(f'({found})')}")
    out.always(f"  journal         {config.journal_path}")
    model_config = config.model
    out.always("")
    out.always(out.bold("model"))
    out.always(f"  backend         {model_config.api} @ {model_config.base_url}")
    out.always(f"  text model      {model_config.name}")
    out.always(f"  vision model    {model_config.vision_name}")
    out.always(
        "  privacy         "
        + (
            out.green("stays on this machine")
            if model_config.is_local
            else out.yellow("prompts are sent to " + model_config.base_url)
        )
    )

    if isinstance(model, NullModel):
        out.always(out.yellow("  status          skipped (--no-model)"))
        return 0

    assert isinstance(model, LocalModel)
    if model_config.needs_token:
        out.always(out.red("  status          no token"))
        out.always(f"  {out.dim(HF_TOKEN_HELP)}")
        return 1

    try:
        installed = model.list_models()
    except ModelUnavailable as exc:
        out.always(out.red("  status          unreachable"))
        out.always(f"  {out.dim(str(exc))}")
        out.always("")
        for line in _fix_hints(model_config):
            out.always(f"  {line}")
        return 1

    out.always(out.green(f"  status          reachable, {len(installed)} model(s)"))
    for name in installed[:12]:
        out.always(f"                  {name}")

    missing = [
        name
        for name in (model_config.name, model_config.vision_name)
        if not model.has_model(name)
    ]
    if missing and model_config.api == "ollama":
        out.always("")
        for name in sorted(set(missing)):
            out.always(out.yellow(f"  not pulled: {name}  ->  ollama pull {name}"))
        return 1
    if missing:
        out.always("")
        for name in sorted(set(missing)):
            out.always(out.yellow(f"  {name} is not in this backend's model list"))
        return 1

    out.always("")
    out.always(out.green("  everything it needs is here. go and do nothing."))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
