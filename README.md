<!-- markdownlint-disable MD033 -->
<div align="center">

```
      (
       \     ___
        \  .'   `.        s l o t h   m o d e
         \/  o o  \       ----------------------
         (   \_/   )      it does the work,
          \  ---  /       you do the nap
       ,--' )   ( `--.
      (___.'     `.___)
```

**Local, open-weight AI for the chores you keep putting off.**

Zero dependencies. No API keys. Nothing leaves your machine.

[Why](#why) · [Install](#install) · [The three agents](#the-three-agents) · [How it works](#how-it-works) · [Config](#config) · [Safety](#the-undo-guarantee)

</div>

---

## Why

Sloths are not lazy, they are *efficient*. They do the minimum that keeps them
alive and spend the rest of the day not wasting energy. That is a design
principle, not a personality flaw.

`sloth` is a command line tool that applies it to your computer. It runs an
open-weight model you host yourself and points it at the three jobs everybody
postpones: a Downloads folder that has become a landfill, an overnight pile of
group chats, and the question of whether you still have milk.

Built for the [DEV](https://dev.to) *Lazy Sloth* challenge on International
Sloth Day, 20 October. The open pieces are not decoration — they are the whole
engine:

| Open piece | What it does here |
| --- | --- |
| **Open-weight models** (Llama 3.2, Qwen 3, Qwen2.5-VL, Gemma, Mistral…) | every decision the tool makes |
| **Ollama** / **llama.cpp** / **vLLM** / **LM Studio** | local inference, your choice of server |
| **Schema-constrained decoding** | what makes a 3B model reliable enough to trust with your files |
| **A ~200-line agent harness** ([`sloth/harness.py`](sloth/harness.py)) | bounded tool-calling loop, read-only by default |
| **The Python standard library** | literally the only dependency |

## Install

```bash
git clone https://github.com/Pager-dot/sloth-mode
cd sloth-mode
pip install -e .
```

Then get a model. Anything small works; these are good defaults:

```bash
ollama pull llama3.2:3b     # the text model (~2 GB)
ollama pull qwen2.5vl:3b    # only needed for `sloth fridge`
sloth doctor                # check it can all see each other
```

No Ollama? Point it at any OpenAI-compatible local server instead:

```bash
sloth --api openai --base-url http://127.0.0.1:8080 tidy ~/Downloads
```

**No model at all?** Everything except `fridge` still works — the agents fall
back to keyword and extension heuristics. Try the whole thing without pulling
a single weight:

```bash
make demo
```

## The three agents

### 1. `sloth tidy` — the junk drawer

```console
$ sloth tidy ~/Downloads

Admin/Invoices/
  ACME Invoice March 2024.pdf           → acme-invoice-march-2024.pdf           [model 92%]

Admin/Tax/
  Form 1099-NEC 2023 final.pdf          → form-1099-nec-2023-final.pdf          [model 88%]

Images/Screenshots/
  Screenshot 2024-03-02 at 11.04.21.png → screenshot-2024-03-02-at-11.04.21.png [heuristic 70%]

Reading/Papers/
  attention-is-all-you-need-arxiv.pdf                                           [model 95%]

_Unsure/
  thing.qqq                                                                     [heuristic 20%]

dry run. re-run with --apply to move these files.
```

It reads each filename, extension, size and (for text files) the first few
hundred characters, then picks a folder and a better name. Every row shows who
decided and how confident they were. Nothing moves without `--apply`.

```bash
sloth tidy ~/Downloads                 # look, don't touch (the default)
sloth tidy ~/Downloads --apply          # do it
sloth tidy ~/Downloads --deep           # let the agent read ambiguous files first
sloth tidy ~/Downloads --json           # machine-readable plan, for your own scripts
sloth undo                              # put it all back
```

### 2. `sloth brief` — the overnight pile

```console
$ sloth brief ~/chat-exports

While you were asleep

  The venue cancelled Saturday and offered Sunday instead; Sam is confirming.
  Rent goes out on the 1st and the house is £40 short because of the water bill.

Needs you
  - Take the bins out before Friday (Priya)
  - Transfer your share of the rent (Priya)
  - Review the open PR before Friday — it blocks the release (Ana)

Safely ignored: emoji, the broken coffee machine, 11 messages about a sloth sanctuary

Read 48,210 characters across 6 file(s) so you did not have to.
```

Map-reduce, because a 3B model with an 8k context cannot read your group chat
in one gulp: each chunk is summarised alone, then the summaries are summarised.
Understands plain text logs, Markdown, and the JSON exports WhatsApp /
Messenger / Discord produce. Pair it with `cron` and wake up to it:

```bash
sloth brief ~/chat-exports --save --markdown
```

### 3. `sloth fridge` — the open fridge door

```console
$ sloth fridge fridge.jpg

Shopping list

  [ ] milk  (running low)
  [ ] bread
  [ ] onions
  [ ] coffee

Already in there: butter, cheese, eggs, lemons, spinach, yoghurt
```

One photo, door closed. A local vision model names what it can see, and the
tool subtracts that from your want-list. This is the one agent with no honest
fallback — without a vision model it tells you so instead of inventing a list.

### Plus

```bash
sloth nap ~/Downloads --every 1h --apply   # tidy on a loop while you sleep
sloth history                              # what it moved, and when
sloth undo --batch <id>                    # reverse any batch, not just the last
sloth doctor                               # is the model reachable? is it pulled?
```

## How it works

```
                  ┌──────────────────────────────────────────┐
  your files ───► │  facts: name, extension, size, excerpt   │
  your chats      └───────────────────┬──────────────────────┘
  your photo                          │
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │  LocalModel  (Ollama | OpenAI-compatible) │
                  │  JSON-schema-constrained decoding         │
                  └───────────────────┬──────────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
          ┌───────────────────────┐      ┌────────────────────────┐
          │  validate: category   │      │  agent harness (--deep) │
          │  must exist, ext must │      │  bounded tool loop,     │
          │  survive, 0 ≤ conf ≤1 │      │  read-only tools        │
          └───────────┬───────────┘      └────────────┬───────────┘
                      └───────────────┬───────────────┘
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │  below the threshold?  →  _Unsure/        │
                  │  otherwise  →  a plan you read first      │
                  └───────────────────┬──────────────────────┘
                                      ▼  --apply
                  ┌──────────────────────────────────────────┐
                  │  journal.jsonl  ──►  sloth undo           │
                  └──────────────────────────────────────────┘
```

Four design decisions worth knowing about:

**Small models are fine if you constrain them.** Every call ships a JSON
schema (Ollama's `format`, or `response_format` on OpenAI-compatible servers),
so a 3B model answers in the shape we asked for instead of writing an essay
about how it would be happy to help.

**The model is never trusted.** It can only choose from your configured
taxonomy — invent `Vibes/Whatever` and the answer is discarded. It cannot
change a file extension, and it cannot escape the root directory with a
`../../../etc/passwd` filename. Both are tested.

**Unsure is a valid answer.** Confidence below the threshold sends a file to
`_Unsure/` rather than to a confident-looking wrong folder. A hesitant model
answer loses to a confident extension rule; if both are unsure, nobody guesses.

**Agents propose, they don't act.** Harness tools are read-only unless
explicitly allowed to write, and the only code that moves a file is the apply
step, after you have seen the plan.

## Config

There isn't any, until you want some. Defaults are chosen so `sloth tidy`
works on a stranger's laptop. To override, write
`~/.config/sloth/config.toml`:

```toml
[model]
name = "qwen3:4b"
base_url = "http://127.0.0.1:11434"

[tidy]
confidence_threshold = 0.7
rename = true

# Your taxonomy replaces the default one entirely.
[tidy.categories]
"Admin/Invoices"  = ["invoice", "receipt", "billing"]
"Band/Tabs"       = ["tab", "chords", "songsterr"]
"Thesis/Sources"  = ["paper", "arxiv", "citation"]

[brief]
summary_lines = 2
max_action_items = 5

[fridge]
want = ["milk", "eggs", "bread", "coffee", "hot sauce"]
```

## The undo guarantee

The deal: `sloth` will rearrange your Downloads folder, and you never have to
remember what it did.

Every move is appended to `journal.jsonl` (greppable, `tail`-able, one JSON
object per line) under a batch id. `sloth undo` reverses the last batch,
restores original names, removes exactly the folders it created — and refuses
to overwrite anything that has taken a file's old name since. Run it twice and
it walks back two batches. `sloth history` lists them all.

## Privacy

There is no cloud. `LocalModel` builds its HTTP opener with proxies explicitly
disabled, so a misconfigured `HTTPS_PROXY` cannot quietly ship your invoices
somewhere. The only network traffic is to the model server you started, and
the whole tool works with the Wi-Fi off.

## Development

```bash
make install   # editable install plus pytest and ruff
make test      # 137 tests, no model server required
make lint      # ruff check + format check
make demo      # tidy and brief the example data, heuristics only
```

Tests never start a model: `tests/conftest.py` has a `FakeModel` that replays
canned responses and records what it was asked, which is how the prompts and
the validation layer are tested at all.

## What it is not

It does not read your email, phone home, or run as a daemon you forget about.
`sloth nap` is a foreground loop you can see and kill. It does not delete
anything, ever — the worst it does is move a file somewhere you did not expect,
which is what `sloth undo` is for.

## License

MIT. See [LICENSE](LICENSE).
