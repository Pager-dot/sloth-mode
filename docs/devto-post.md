---
title: "Sloth Mode: open-weight AI that does the chores, and a journal that undoes them"
published: false
tags: ollama, opensource, ai, python
cover_image: https://raw.githubusercontent.com/Pager-dot/sloth-mode/main/docs/demo/sloth-demo.gif
---

*Built for the DEV Lazy Sloth challenge. Sloths are not lazy, they are efficient —
they do the minimum that keeps them alive and spend the rest of the day not
wasting energy. That is a design principle, not a personality flaw.*

🔗 **[github.com/Pager-dot/sloth-mode](https://github.com/Pager-dot/sloth-mode)** ·
📺 **[Watch the 22-second demo](https://youtu.be/REPLACE_WITH_YOUR_VIDEO_ID)**

![Sloth Mode tidying a folder, undoing it, and summarising a night of group chats](https://raw.githubusercontent.com/Pager-dot/sloth-mode/main/docs/demo/sloth-demo.gif)

## What I Built

**Sloth Mode** is a command-line tool (with a small local web GUI) that runs an
open-weight model you host yourself and points it at three jobs everybody
postpones:

- **`sloth tidy`** — reads your junk-drawer folder, proposes a home and a better
  name for every file, and shows you the plan. Nothing moves until you say so.
- **`sloth brief`** — reads an overnight pile of group chats and hands back two
  sentences plus the things that actually need *you*.
- **`sloth fridge`** — one photo of the fridge, door closed, becomes a shopping
  list.

```console
$ sloth tidy ~/Downloads

Admin/Invoices/
  ACME Invoice March 2024.pdf   → acme-invoice-march-2024.pdf   [model 99%]

Reading/Notes/
  notes.txt                     → garden-boiler-notes.txt       [model 85%]

_Unsure/
  thing.qqq                                                     [model 30%]

dry run. re-run with --apply to move these files.
```

Two details carry the whole design. `notes.txt` became `garden-boiler-notes.txt`
because the model *read the file*. And `thing.qqq` went to `_Unsure` at 30%
rather than to a confident-looking wrong folder — **"I don't know" is a valid
answer**, and the threshold enforces it.

**Who would pay for it?** The people who cannot use the cloud tools that already
do this. A freelancer's Downloads folder is full of invoices and signed
contracts. An accountant's is full of other people's tax forms. A clinician's
has patient letters; a researcher's has data under an agreement that forbids
third-party processing. Every "AI file organiser" on the market wants those
documents uploaded. For these users that is not a preference, it is a
disqualification.

The problem it solves is not that sorting files is *hard*. It is that it is
never *urgent*. It loses to everything else, forever, until you have 4,000 files
and no idea which ones matter. Sloth Mode does it while you do something else,
and — crucially — **`sloth undo` puts it all back**, exactly, including the
capitals and spaces in names it had changed. That is what makes it safe enough
to actually run on a folder you care about.

## How I Built It

**The open pieces are not decoration — they are the engine.**

| Open piece | What it does here |
| --- | --- |
| **Gemma 3** (`gemma3:4b`) | the local default; multimodal, so one 3 GB pull covers all three agents |
| **gpt-oss:120b** (Apache-2.0) | 120B parameters via Ollama Cloud, no GPU, no download |
| **Gemma 4** (`gemma4:31b`) | the vision model behind `fridge` |
| **Ollama** / **llama.cpp** / **vLLM** / **LM Studio** | inference, your choice of host |
| **Hugging Face Inference Providers** | the same open weights on a free tier |
| **The Python standard library** | the *only* dependency |

Zero runtime dependencies is itself a laziness feature: `pip install` and you are
done. `urllib` talks to the model, `http.server` serves the GUI, `tomllib` reads
the config, and the undo journal is JSONL you can read with `tail`.

Four backends sit behind one `Model` protocol, so `--api cloud`, `--api hf` or a
bare local Ollama are one flag apart. I wrote a small **agent harness**
(~250 lines) rather than pulling in a framework: tools are plain functions with a
JSON-schema argument spec, the loop is bounded by steps *and* a wall clock, and
**every tool is read-only unless writes are explicitly allowed**. An agent can
look at anything and propose anything; only the apply step touches the disk.

The spine of the whole thing is: **the model is never trusted.**

- It can only choose from your configured taxonomy. Invent `Vibes/Whatever` and
  the answer is discarded for a keyword/extension heuristic.
- It cannot change a file extension.
- It cannot escape the root directory with a `../../../etc/passwd` filename.
- Below the confidence threshold, nobody guesses.
- And if no model is running at all, every agent except `fridge` still works on
  heuristics — `fridge` says so instead of inventing a shopping list.

### Then I ran it against a real model, and four things broke

I had 189 passing tests before the first real inference call. All green. Here is
what they had failed to notice:

**1. The JSON schema was being ignored outright.** Every call ships one. I
assumed it was honoured. It is not — I measured it: a request carrying a schema
that *requires* `{"ok": boolean}`, with a prompt asking for prose, came back as
`'Sloths move slowly through the treetops...'`. Not loosely enforced. Ignored.

**2. So the morning brief silently degraded.** The model answered the map step
with `{"action_items": [...]}` where the schema said `{"points": [...]}`. My code
read only the expected key, got an empty list, and fell back to keyword
extraction — while printing *"no model running"* with a 31B model very much
running. **A silent failure that looks like success is worse than a crash.**

**3. `--deep` was 100% dead.** gpt-oss writes its entire planned trajectory as
several JSON objects in one reply. The parser raised `Extra data`, the harness
retried four times, hit max steps, and returned nothing. Every unsure file, every
run, invisibly.

**4. `fridge` dropped every item it saw** — the model answered `[{"item": "milk"}]`
where the schema said `name`, so the shopping list cheerfully told you to buy the
milk that was visible in the photo.

Same root cause, four faces. The fix was to stop treating the schema as a
contract: accept the field under another name, unwrap markdown fences, take the
first complete JSON value, state the shape in the prompt *as well*, and force the
summary into the sentence count that was asked for. The suite is now **250 tests**
that need no weights and no network — a `FakeModel` for logic, a throwaway HTTP
server for the wire format.

## Why Does Open Innovation Matter?

**Because I could measure the thing instead of believing it.**

That schema discovery is the entire argument. Against a closed API I would have
read the docs, seen that structured output is supported, and shipped a brief that
quietly produced keyword slop for a subset of users. What let me find it was that
every layer was inspectable: I could send a byte-identical request with a valid
schema, a deliberately bogus schema, and no schema at all, diff the three
replies, and *know*. Then I could fix it in my own parser. No ticket, no waiting
for a vendor, no "we don't comment on model internals."

**Open weights let the work happen where the files already are.** The entire
value proposition — tidy the folder full of invoices and contracts — collapses if
step one is "upload your invoices and contracts." Running locally is not a
privacy feature bolted on; it is the only reason the tool is usable by the people
who need it most. The tool is honest about the trade, too: pick a hosted backend
and `sloth doctor` prints `prompts are sent to https://ollama.com` in yellow,
every single run.

**The open ecosystem converged on two wire formats, so swapping is one flag.**
Gemma 3 on a laptop, gpt-oss:120b in the cloud, anything on a llama.cpp server —
the same code, because `/api/chat` and `/v1/chat/completions` are conventions
anyone can implement. That is also an honest escape hatch: nothing here is
load-bearing on one vendor's continued goodwill or pricing.

**And it scales down to nothing.** `gemma3:1b` runs on almost any machine.
`--no-model` runs on none at all. A tool built on a closed API has a hard floor:
no key, no product. This one degrades to keyword heuristics and keeps working on
a train with no signal — which is, after all, the most sloth-like property
available: it does not even need the internet to be awake.

---

*Everything above is in the repo, including the script that recorded the demo.
The bugs are in the git history too — I would rather show the four things that
broke than pretend the first run went well.*

**[github.com/Pager-dot/sloth-mode](https://github.com/Pager-dot/sloth-mode)** — MIT.
