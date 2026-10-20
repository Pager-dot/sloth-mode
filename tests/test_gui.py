"""The web UI, driven over real HTTP.

It moves files on request, so the token check gets as much attention as the
happy path.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from sloth.gui import make_server
from sloth.model import NullModel


@pytest.fixture
def ui(config, messy_dir):
    """A running UI with no model in play, pointed at a messy folder."""
    httpd, token = make_server(config, NullModel(), messy_dir, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"

    def call(path, body=None, tok=token):
        request = urllib.request.Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"X-Sloth-Token": tok, "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=10) as response:
            return json.loads(response.read())

    call.base = base
    call.token = token
    call.root = messy_dir
    try:
        yield call
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get_page(url: str) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return response.read().decode()


# ---------------------------------------------------------------------------
# the page itself
# ---------------------------------------------------------------------------


def test_the_page_loads_and_carries_its_token(ui):
    html = get_page(ui.base + "/")

    assert "<title>Sloth Mode</title>" in html
    assert ui.token in html  # the page needs it to call the API
    assert "__TOKEN__" not in html  # and the placeholder is gone
    assert str(ui.root) in html


def test_the_page_needs_no_token_itself(ui):
    """Opening the UI must work from a plain browser address bar."""
    assert "Sloth Mode" in get_page(ui.base + "/index.html")


def test_status_reports_the_backend(ui):
    status = ui("/api/status")

    assert status["model"] == "gemma3:4b"
    assert status["local"] is True
    assert status["available"] is False  # NullModel
    assert status["path"] == str(ui.root)


# ---------------------------------------------------------------------------
# the token is the only thing between a web page and your Downloads folder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "wrong", "almost-right"])
def test_api_calls_without_the_token_are_refused(ui, bad):
    with pytest.raises(urllib.error.HTTPError) as caught:
        ui("/api/status", tok=bad)
    assert caught.value.code == 403


def test_mutating_calls_without_the_token_are_refused(ui):
    before = sorted(p.name for p in ui.root.iterdir())

    for path, body in [
        ("/api/plan", {"path": str(ui.root)}),
        ("/api/apply", {"plan_id": "whatever"}),
        ("/api/undo", {}),
    ]:
        with pytest.raises(urllib.error.HTTPError) as caught:
            ui(path, body, tok="nope")
        assert caught.value.code == 403

    assert sorted(p.name for p in ui.root.iterdir()) == before


def test_unknown_endpoints_404(ui):
    with pytest.raises(urllib.error.HTTPError) as caught:
        ui("/api/please-delete-everything", {})
    assert caught.value.code == 404


# ---------------------------------------------------------------------------
# plan / apply / undo
# ---------------------------------------------------------------------------


def test_plan_changes_nothing_on_disk(ui):
    before = sorted(p.name for p in ui.root.iterdir())

    plan = ui("/api/plan", {"path": str(ui.root)})

    assert plan["moves"]
    assert plan["used_model"] is False
    assert sorted(p.name for p in ui.root.iterdir()) == before
    first = plan["moves"][0]
    assert {"from", "to", "category", "confidence", "decided_by"} <= set(first)


def test_apply_then_undo_roundtrip(ui):
    before = sorted(p.name for p in ui.root.iterdir())

    plan = ui("/api/plan", {"path": str(ui.root)})
    applied = ui("/api/apply", {"plan_id": plan["plan_id"]})

    assert applied["moved"] == len(plan["moves"])
    assert (ui.root / "Admin" / "Invoices").is_dir()
    assert applied["batches"][0]["files"] == applied["moved"]

    undone = ui("/api/undo", {})

    assert undone["restored"] == applied["moved"]
    assert undone["folders_removed"] > 0
    assert sorted(p.name for p in ui.root.iterdir() if p.is_file()) == before
    assert not (ui.root / "Admin").exists()


def test_applying_runs_the_plan_you_were_shown(ui):
    """Apply must not re-ask the model and hope for the same answer."""
    plan = ui("/api/plan", {"path": str(ui.root)})
    shown = {move["to"] for move in plan["moves"]}

    ui("/api/apply", {"plan_id": plan["plan_id"]})

    on_disk = {p.name for p in ui.root.rglob("*") if p.is_file()}
    assert shown <= on_disk


def test_a_plan_can_only_be_applied_once(ui):
    plan = ui("/api/plan", {"path": str(ui.root)})
    ui("/api/apply", {"plan_id": plan["plan_id"]})

    with pytest.raises(urllib.error.HTTPError) as caught:
        ui("/api/apply", {"plan_id": plan["plan_id"]})

    assert caught.value.code == 400
    assert "expired" in json.loads(caught.value.read())["error"]


def test_an_unknown_plan_id_is_refused(ui):
    with pytest.raises(urllib.error.HTTPError) as caught:
        ui("/api/apply", {"plan_id": "made-up"})
    assert caught.value.code == 400


def test_a_bad_path_is_a_clean_error_not_a_crash(ui):
    with pytest.raises(urllib.error.HTTPError) as caught:
        ui("/api/plan", {"path": str(ui.root / "does-not-exist")})

    assert caught.value.code == 400
    assert "not a directory" in json.loads(caught.value.read())["error"]


def test_undo_with_nothing_to_undo(ui):
    result = ui("/api/undo", {})

    assert result["restored"] == 0
    assert result["batch"] is None


# ---------------------------------------------------------------------------
# brief
# ---------------------------------------------------------------------------


def test_brief_endpoint(ui, tmp_path):
    chats = tmp_path / "chats"
    chats.mkdir()
    (chats / "a.txt").write_text(
        "Priya: can you call the landlord tomorrow morning please\n"
        "Tom: sure, and the deposit is due friday\n"
    )

    result = ui("/api/brief", {"path": str(chats), "lines": 2})

    assert len(result["summary"]) <= 2
    assert result["used_model"] is False
    assert result["characters"] > 0


def test_brief_line_count_is_capped(ui, tmp_path):
    chats = tmp_path / "chats"
    chats.mkdir()
    (chats / "a.txt").write_text("Ada: the deployment finished and nothing broke\n")

    result = ui("/api/brief", {"path": str(chats), "lines": 99})

    assert len(result["summary"]) <= 6


def test_history_endpoint(ui):
    assert ui("/api/history")["batches"] == []

    plan = ui("/api/plan", {"path": str(ui.root)})
    ui("/api/apply", {"plan_id": plan["plan_id"]})

    batches = ui("/api/history")["batches"]
    assert len(batches) == 1
    assert batches[0]["agent"] == "tidy"


def test_the_url_we_print_actually_works(ui):
    """The link the CLI prints carries ?t=<token>; it must not 404.

    Found by pointing a real browser at it: the handler was comparing the
    whole path, query string included.
    """
    html = get_page(f"{ui.base}/?t={ui.token}")

    assert "<title>Sloth Mode</title>" in html


def test_api_routes_ignore_query_strings(ui):
    import urllib.parse

    query = urllib.parse.urlencode({"cache": "no"})
    assert ui(f"/api/status?{query}")["model"] == "gemma3:4b"
    assert ui(f"/api/history?{query}")["batches"] == []
