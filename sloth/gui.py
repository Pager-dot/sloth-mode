"""A small local web UI, for people who would rather click than type.

Served by the standard library's ``http.server`` on 127.0.0.1, because adding
a web framework to a zero-dependency tool would be work, and this project is
against work.

Safety, since this process moves files on request:

* it binds to the loopback interface only,
* every mutating call must carry a token minted at startup and embedded in
  the page, which stops another site in your browser from POSTing to
  127.0.0.1 and rearranging your Downloads folder,
* `Plan` never writes. `Apply` carries out a plan the server already computed
  and still holds, so what you approve is exactly what runs — it never
  re-asks the model and hopes for the same answer.
"""

from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .agents import brief as brief_agent
from .agents import tidy as tidy_agent
from .config import SlothConfig
from .journal import Journal
from .model import Model, ModelUnavailable

#: Plans waiting for an Apply, keyed by id. Small and in-memory on purpose:
#: a plan that outlives the process should be recomputed, not trusted.
_PLANS: dict[str, tidy_agent.TidyPlan] = {}
_PLANS_LOCK = threading.Lock()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


class SlothHandler(BaseHTTPRequestHandler):
    server_version = f"sloth/{__version__}"

    # Set by make_server.
    config: SlothConfig
    model: Model
    token: str
    default_path: Path

    # -- plumbing ---------------------------------------------------------

    def _send(self, payload: Any, status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _authorised(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Sloth-Token", ""), self.token)

    def log_message(self, *args) -> None:  # the terminal stays calm
        pass

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        # The URL we hand the user carries ?t=<token>, so compare the path
        # without its query string.
        route = self.path.split("?", 1)[0]
        if route.startswith("/api/"):
            if not self._authorised():
                self._send({"error": "bad or missing token"}, HTTPStatus.FORBIDDEN)
                return
            if route == "/api/status":
                self._send(self._status())
                return
            if route == "/api/history":
                self._send({"batches": self._history()})
                return
            self._send({"error": "no such endpoint"}, HTTPStatus.NOT_FOUND)
            return

        if route in ("/", "/index.html"):
            self._send_html(
                PAGE.replace("__TOKEN__", self.token)
                .replace("__PATH__", str(self.default_path))
                .replace("__VERSION__", __version__)
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._authorised():
            self._send({"error": "bad or missing token"}, HTTPStatus.FORBIDDEN)
            return

        route = self.path.split("?", 1)[0]
        handlers = {
            "/api/plan": self._plan,
            "/api/apply": self._apply,
            "/api/undo": self._undo,
            "/api/brief": self._brief,
        }
        handler = handlers.get(route)
        if handler is None:
            self._send({"error": "no such endpoint"}, HTTPStatus.NOT_FOUND)
            return

        try:
            self._send(handler(self._body()))
        except (
            ModelUnavailable,
            NotADirectoryError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            self._send({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # never take the UI down over one request
            self._send(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # -- actions ----------------------------------------------------------

    def _status(self) -> dict:
        model_config = self.config.model
        return {
            "version": __version__,
            "backend": model_config.api,
            "model": model_config.name,
            "base_url": model_config.base_url,
            "local": model_config.is_local,
            "available": self.model.available(),
            "path": str(self.default_path),
        }

    def _plan(self, body: dict) -> dict:
        root = Path(body.get("path") or self.default_path).expanduser()
        plan = tidy_agent.plan(
            root,
            self.config,
            self.model,
            recursive=bool(body.get("recursive")),
            deep=bool(body.get("deep")),
        )
        plan_id = secrets.token_hex(8)
        with _PLANS_LOCK:
            _PLANS[plan_id] = plan
            # Keep only the few most recent; this is a desk toy, not a server.
            for stale in list(_PLANS)[:-8]:
                _PLANS.pop(stale, None)

        return {
            "plan_id": plan_id,
            "root": str(plan.root),
            "used_model": plan.used_model,
            "model_error": plan.model_error,
            "skipped": len(plan.skipped),
            "moves": [
                {
                    "from": proposal.path.name,
                    "to": proposal.new_name,
                    "category": proposal.category,
                    "confidence": round(proposal.confidence, 2),
                    "decided_by": proposal.decided_by,
                    "reason": proposal.reason,
                    "renamed": proposal.renamed,
                }
                for proposal in plan.actionable
            ],
        }

    def _apply(self, body: dict) -> dict:
        plan_id = str(body.get("plan_id") or "")
        with _PLANS_LOCK:
            plan = _PLANS.pop(plan_id, None)
        if plan is None:
            raise ValueError("that plan has expired; press Plan again")

        journal = Journal(self.config.journal_path)
        moves = tidy_agent.apply_plan(plan, journal)
        return {
            "moved": len(moves),
            "batches": self._history(),
        }

    def _undo(self, body: dict) -> dict:
        journal = Journal(self.config.journal_path)
        result = (
            journal.undo_batch(str(body["batch"]))
            if body.get("batch")
            else journal.undo_last()
        )
        emptied = journal.prune_vacated_dirs(result.restored) if result.restored else 0
        return {
            "restored": len(result.restored),
            "skipped": [[str(p), why] for p, why in result.skipped],
            "folders_removed": emptied,
            "batch": result.batch,
            "batches": self._history(),
        }

    def _brief(self, body: dict) -> dict:
        root = Path(body.get("path") or self.default_path).expanduser()
        lines = body.get("lines")
        if isinstance(lines, int) and lines > 0:
            self.config.brief.summary_lines = min(lines, 6)
        result = brief_agent.summarise([root], self.config, self.model)
        return {
            "summary": result.summary,
            "action_items": result.action_items,
            "ignorable": result.ignorable,
            "used_model": result.used_model,
            "model_error": result.model_error,
            "characters": result.total_characters,
            "sources": len(result.sources),
        }

    def _history(self) -> list[dict]:
        rows = Journal(self.config.journal_path).history(limit=8)
        return [
            {"batch": batch, "files": count, "when": when, "agent": agent}
            for batch, count, when, agent in rows
        ]


def make_server(
    config: SlothConfig,
    model: Model,
    default_path: Path,
    host: str = "127.0.0.1",
    port: int = 8777,
) -> tuple[ThreadingHTTPServer, str]:
    """Build the server and its one-off token. Caller starts it."""
    token = secrets.token_urlsafe(24)

    handler = type(
        "BoundSlothHandler",
        (SlothHandler,),
        {
            "config": config,
            "model": model,
            "token": token,
            "default_path": Path(default_path).expanduser(),
        },
    )
    return ThreadingHTTPServer((host, port), handler), token


def serve(
    config: SlothConfig,
    model: Model,
    default_path: Path,
    host: str = "127.0.0.1",
    port: int = 8777,
    open_browser: bool = True,
    on_ready=None,
) -> None:
    """Run the UI until interrupted."""
    httpd, token = make_server(config, model, default_path, host, port)
    url = f"http://{host}:{httpd.server_port}/?t={token}"
    if on_ready:
        on_ready(url)
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sloth Mode</title>
<style>
  :root {
    --bg: #faf7f2; --panel: #fff; --ink: #241f1a; --soft: #7b7268;
    --line: #e7dfd4; --accent: #6b7f4e; --warn: #b5762f; --bad: #a84a3c;
    --shadow: 0 1px 2px rgba(36,31,26,.06), 0 8px 24px rgba(36,31,26,.06);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #17150f; --panel: #211e17; --ink: #efe8dd; --soft: #a79c8d;
      --line: #322d24; --accent: #9db772; --warn: #d79a4e; --bad: #e0756a;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 0 16px 64px;
  }
  .wrap { max-width: 940px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 16px; padding: 28px 0 18px; }
  pre.sloth { margin: 0; font-size: 10px; line-height: 1.15; color: var(--soft); }
  h1 { font-size: 21px; margin: 0; letter-spacing: -.01em; }
  .sub { color: var(--soft); font-size: 13px; margin-top: 2px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px 18px; margin-bottom: 14px; box-shadow: var(--shadow);
  }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  input[type=text] {
    flex: 1 1 260px; min-width: 0; padding: 9px 11px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--bg); color: var(--ink);
    font: inherit; font-size: 14px;
  }
  button {
    padding: 9px 15px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--panel); color: var(--ink); font: inherit; font-weight: 550;
    cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  label.check { display: flex; align-items: center; gap: 6px; color: var(--soft);
    font-size: 13px; cursor: pointer; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    color: var(--soft); border: 1px solid var(--line); border-radius: 999px;
    padding: 3px 10px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--soft); }
  .dot.ok { background: var(--accent); } .dot.off { background: var(--bad); }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
       color: var(--soft); margin: 0 0 10px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th { text-align: left; font-weight: 600; color: var(--soft); font-size: 11.5px;
       text-transform: uppercase; letter-spacing: .05em; padding: 0 8px 7px 0; }
  td { padding: 6px 8px 6px 0; border-top: 1px solid var(--line); vertical-align: top; }
  td.name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  .cat { display: inline-block; font-size: 12px; padding: 1px 7px; border-radius: 5px;
         background: color-mix(in srgb, var(--accent) 14%, transparent); }
  .cat.unsure { background: color-mix(in srgb, var(--warn) 18%, transparent); }
  .conf { color: var(--soft); font-size: 12px; white-space: nowrap; }
  .by { font-size: 11px; color: var(--soft); text-transform: uppercase;
        letter-spacing: .04em; }
  .msg { font-size: 13.5px; padding: 9px 12px; border-radius: 8px; margin-top: 10px; }
  .msg.ok { background: color-mix(in srgb, var(--accent) 13%, transparent); }
  .msg.bad { background: color-mix(in srgb, var(--bad) 13%, transparent); }
  .msg.warn { background: color-mix(in srgb, var(--warn) 15%, transparent); }
  .muted { color: var(--soft); font-size: 13px; }
  ul.items { margin: 6px 0 0; padding-left: 18px; }
  ul.items li { margin-bottom: 3px; }
  .summary p { margin: 0 0 6px; font-size: 15.5px; }
  .hist { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--soft); }
  .spin { display: inline-block; animation: turn 1.1s linear infinite; }
  @keyframes turn { to { transform: rotate(360deg); } }
  @media (max-width: 620px) {
    pre.sloth { display: none; }
    .conf, .by { display: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <pre class="sloth">      (
       \\     ___
        \\  .'   `.
         \\/  o o  \\
         (   \\_/   )
          \\  ---  /
       ,--' )   ( `--.
      (___.'     `.___)</pre>
    <div>
      <h1>Sloth Mode</h1>
      <div class="sub">It does the work, you do the nap. <span id="status" class="pill"><span class="dot"></span> checking…</span></div>
    </div>
  </header>

  <div class="card">
    <h2>Tidy a folder</h2>
    <div class="row">
      <input type="text" id="path" value="__PATH__" spellcheck="false">
      <button class="primary" id="btn-plan">Plan</button>
      <button id="btn-apply" disabled>Apply</button>
      <button id="btn-undo">Undo</button>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="check"><input type="checkbox" id="recursive"> include subfolders</label>
      <label class="check"><input type="checkbox" id="deep"> deep (read files before deciding)</label>
    </div>
    <div id="tidy-msg"></div>
    <div id="tidy-out"></div>
  </div>

  <div class="card">
    <h2>Morning brief</h2>
    <div class="row">
      <input type="text" id="brief-path" value="__PATH__" spellcheck="false">
      <button id="btn-brief">Summarise</button>
      <label class="check">lines <input type="text" id="lines" value="2" style="width:42px;flex:0 0 42px;text-align:center"></label>
    </div>
    <div id="brief-msg"></div>
    <div id="brief-out" class="summary"></div>
  </div>

  <div class="card">
    <h2>History</h2>
    <div id="history" class="hist">nothing moved yet.</div>
  </div>

  <p class="muted">sloth __VERSION__ · nothing moves until you press Apply · everything is undoable</p>
</div>

<script>
const TOKEN = "__TOKEN__";
const $ = (id) => document.getElementById(id);
let planId = null;

async function call(path, body) {
  const options = {
    headers: { "X-Sloth-Token": TOKEN, "Content-Type": "application/json" },
    method: body === undefined ? "GET" : "POST",
  };
  if (body !== undefined) options.body = JSON.stringify(body);
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ error: "bad response" }));
  if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
  return data;
}

function say(where, text, kind) {
  $(where).innerHTML = text ? `<div class="msg ${kind || "ok"}">${text}</div>` : "";
}

function busy(button, on, label) {
  button.disabled = on;
  button.textContent = on ? "working…" : label;
}

async function refreshStatus() {
  try {
    const s = await call("/api/status");
    const where = s.local ? "on this machine" : s.base_url;
    $("status").innerHTML =
      `<span class="dot ${s.available ? "ok" : "off"}"></span> ${s.model} · ${where}`;
  } catch (err) {
    $("status").innerHTML = `<span class="dot off"></span> ${err.message}`;
  }
}

function renderHistory(batches) {
  if (!batches || !batches.length) { $("history").textContent = "nothing moved yet."; return; }
  $("history").innerHTML = batches
    .map((b) => `${b.batch} · ${b.files} file(s) · ${b.agent}`)
    .join("<br>");
}

function renderPlan(plan) {
  if (!plan.moves.length) {
    $("tidy-out").innerHTML = "";
    say("tidy-msg", "Nothing to do. Impressive, or concerning.", "ok");
    return;
  }
  const rows = plan.moves.map((m) => `
    <tr>
      <td class="name">${m.from}</td>
      <td class="name">${m.renamed ? "→ " + m.to : ""}</td>
      <td><span class="cat ${m.category.startsWith("_") ? "unsure" : ""}">${m.category}</span></td>
      <td class="conf">${Math.round(m.confidence * 100)}%</td>
      <td class="by">${m.decided_by}</td>
    </tr>`).join("");
  $("tidy-out").innerHTML = `<table>
      <tr><th>file</th><th>new name</th><th>goes to</th><th>sure</th><th>decided by</th></tr>
      ${rows}</table>`;
  const how = plan.used_model ? "the model" : "heuristics (no model in play)";
  say("tidy-msg", `${plan.moves.length} file(s) planned by ${how}. Nothing has moved yet.`,
      plan.model_error ? "warn" : "ok");
  if (plan.model_error) {
    say("tidy-msg", `Model unavailable, used heuristics: ${plan.model_error}`, "warn");
  }
}

$("btn-plan").onclick = async () => {
  const button = $("btn-plan");
  busy(button, true, "Plan");
  $("btn-apply").disabled = true;
  try {
    const plan = await call("/api/plan", {
      path: $("path").value,
      recursive: $("recursive").checked,
      deep: $("deep").checked,
    });
    planId = plan.plan_id;
    renderPlan(plan);
    $("btn-apply").disabled = plan.moves.length === 0;
  } catch (err) {
    $("tidy-out").innerHTML = "";
    say("tidy-msg", err.message, "bad");
  } finally {
    busy(button, false, "Plan");
  }
};

$("btn-apply").onclick = async () => {
  const button = $("btn-apply");
  busy(button, true, "Apply");
  try {
    const result = await call("/api/apply", { plan_id: planId });
    say("tidy-msg", `Moved ${result.moved} file(s). Changed your mind? Press Undo.`, "ok");
    $("tidy-out").innerHTML = "";
    renderHistory(result.batches);
    planId = null;
  } catch (err) {
    say("tidy-msg", err.message, "bad");
  } finally {
    busy(button, false, "Apply");
    $("btn-apply").disabled = true;
  }
};

$("btn-undo").onclick = async () => {
  const button = $("btn-undo");
  busy(button, true, "Undo");
  try {
    const result = await call("/api/undo", {});
    say("tidy-msg", result.restored
      ? `Put ${result.restored} file(s) back, and removed ${result.folders_removed} folder(s) it had made.`
      : "Nothing to undo. The journal is clean.", "ok");
    renderHistory(result.batches);
  } catch (err) {
    say("tidy-msg", err.message, "bad");
  } finally {
    busy(button, false, "Undo");
  }
};

$("btn-brief").onclick = async () => {
  const button = $("btn-brief");
  busy(button, true, "Summarise");
  try {
    const result = await call("/api/brief", {
      path: $("brief-path").value,
      lines: parseInt($("lines").value, 10) || 2,
    });
    const items = (result.action_items || [])
      .map((i) => `<li>${i.what}${i.who_asked ? ` <span class="muted">(${i.who_asked})</span>` : ""}</li>`)
      .join("");
    $("brief-out").innerHTML =
      result.summary.map((line) => `<p>${line}</p>`).join("") +
      (items ? `<h2 style="margin-top:14px">Needs you</h2><ul class="items">${items}</ul>` : "") +
      (result.ignorable ? `<p class="muted" style="margin-top:10px">Safely ignored: ${result.ignorable}</p>` : "");
    say("brief-msg",
      `Read ${result.characters.toLocaleString()} characters across ${result.sources} file(s) so you did not have to.`,
      result.model_error ? "warn" : "ok");
    if (result.model_error) {
      say("brief-msg", `Fell back to keyword summarising: ${result.model_error}`, "warn");
    }
  } catch (err) {
    $("brief-out").innerHTML = "";
    say("brief-msg", err.message, "bad");
  } finally {
    busy(button, false, "Summarise");
  }
};

refreshStatus();
call("/api/history").then((d) => renderHistory(d.batches)).catch(() => {});
</script>
</body>
</html>
"""
