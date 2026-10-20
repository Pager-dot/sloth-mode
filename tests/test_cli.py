import json

import pytest

from sloth.cli import build_parser, main, parse_duration


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Never read the developer's real config or write to their real journal."""
    monkeypatch.setenv("SLOTH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SLOTH_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.mark.parametrize(
    "raw,expected", [("90", 90), ("30s", 30), ("15m", 900), ("2h", 7200)]
)
def test_parse_duration(raw, expected):
    assert parse_duration(raw) == expected


def test_parse_duration_rejects_nonsense():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_duration("soon")


def test_no_command_prints_the_sloth(capsys):
    assert main([]) == 0
    assert "sloth mode" in capsys.readouterr().out


def test_version():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_tidy_dry_run_changes_nothing(messy_dir, capsys):
    before = sorted(p.name for p in messy_dir.iterdir())

    code = main(["--no-model", "tidy", str(messy_dir)])

    out = capsys.readouterr().out
    assert code == 0
    assert "Admin/Invoices/" in out
    assert "dry run" in out
    assert sorted(p.name for p in messy_dir.iterdir()) == before


def test_tidy_json_output(messy_dir, capsys):
    code = main(["--no-model", "tidy", str(messy_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["used_model"] is False
    assert payload["moves"]


def test_tidy_apply_then_undo(messy_dir, capsys):
    before = sorted(p.name for p in messy_dir.iterdir())

    assert main(["--no-model", "tidy", str(messy_dir), "--apply", "--yes"]) == 0
    assert (messy_dir / "Admin" / "Invoices").is_dir()
    assert "moved" in capsys.readouterr().out

    assert main(["undo"]) == 0
    assert "restored" in capsys.readouterr().out
    assert sorted(p.name for p in messy_dir.iterdir() if p.is_file()) == [
        n for n in before
    ]


def test_tidy_apply_respects_a_declined_prompt(messy_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    code = main(["--no-model", "tidy", str(messy_dir), "--apply"])

    assert code == 0
    assert "left everything exactly where it was" in capsys.readouterr().out
    assert not (messy_dir / "Admin").exists()


def test_tidy_threshold_sends_everything_to_unsure(messy_dir, capsys):
    main(["--no-model", "tidy", str(messy_dir), "--threshold", "0.99"])

    out = capsys.readouterr().out
    assert "_Unsure/" in out
    assert "Admin/Invoices/" not in out


def test_tidy_on_a_missing_directory(tmp_path, capsys):
    code = main(["--no-model", "tidy", str(tmp_path / "nope")])

    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_tidy_on_an_empty_directory(tmp_path, capsys):
    (tmp_path / "empty").mkdir()

    code = main(["--no-model", "tidy", str(tmp_path / "empty")])

    assert code == 0
    assert "nothing to do" in capsys.readouterr().out


def test_undo_with_an_empty_journal(capsys):
    assert main(["undo"]) == 0
    assert "nothing to undo" in capsys.readouterr().out


def test_history_empty_then_populated(messy_dir, capsys):
    assert main(["history"]) == 0
    assert "no moves recorded" in capsys.readouterr().out

    main(["--no-model", "tidy", str(messy_dir), "--apply", "--yes"])
    capsys.readouterr()

    assert main(["history"]) == 0
    out = capsys.readouterr().out
    assert "tidy" in out
    assert "batch" in out


def test_brief_runs_without_a_model(tmp_path, capsys):
    (tmp_path / "chat.txt").write_text(
        "Priya: can you confirm the booking tomorrow morning please\n"
        "Tom: sure, the deposit is due friday as well\n"
    )

    code = main(["--no-model", "brief", str(tmp_path), "--markdown"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("# Morning brief")


def test_brief_saves_to_the_state_dir(tmp_path, capsys):
    (tmp_path / "chat.txt").write_text("Ada: the deployment finished, nothing broke\n")

    main(["--no-model", "brief", str(tmp_path), "--save"])

    saved = list((tmp_path / "state" / "briefs").glob("brief-*.md"))
    assert len(saved) == 1
    assert "Morning brief" in saved[0].read_text()
    assert "saved to" in capsys.readouterr().out


def test_fridge_without_a_model_explains_itself(tmp_path, capsys):
    photo = tmp_path / "fridge.jpg"
    photo.write_bytes(b"\xff\xd8\xff")

    code = main(["--no-model", "fridge", str(photo)])

    assert code == 1
    assert "ollama pull" in capsys.readouterr().err


def test_doctor_reports_no_model_mode(capsys):
    code = main(["--no-model", "doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "skipped (--no-model)" in out
    assert "sloth mode" in out


def test_doctor_reports_an_unreachable_server(capsys):
    code = main(["--base-url", "http://127.0.0.1:1", "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "unreachable" in out
    assert "ollama serve" in out


def test_nap_runs_a_bounded_number_of_rounds(messy_dir, capsys):
    code = main(
        ["--no-model", "nap", str(messy_dir), "--every", "1s", "--rounds", "1", "--apply"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "filed" in out
    assert (messy_dir / "Admin" / "Invoices").is_dir()


def test_nap_reports_without_apply(messy_dir, capsys):
    code = main(["--no-model", "nap", str(messy_dir), "--rounds", "1"])

    assert code == 0
    assert "could be filed (no --apply)" in capsys.readouterr().out


def test_config_file_is_honoured(tmp_path, messy_dir, capsys):
    (tmp_path / "config.toml").write_text(
        """
        [tidy]
        confidence_threshold = 0.1

        [tidy.categories]
        "Boring" = ["invoice"]
        "Fun" = ["screenshot"]
        """,
        encoding="utf-8",
    )

    main(["--no-model", "tidy", str(messy_dir)])

    out = capsys.readouterr().out
    assert "Boring/" in out
    assert "Fun/" in out
    assert "Admin/Invoices/" not in out


def test_undo_cleans_up_the_folders_it_created(messy_dir, capsys):
    main(["--no-model", "tidy", str(messy_dir), "--apply", "--yes"])
    capsys.readouterr()

    main(["undo"])

    out = capsys.readouterr().out
    assert "removed" in out
    assert not (messy_dir / "Admin").exists()
    assert (messy_dir / "ACME Invoice March 2024.pdf").exists()


def test_doctor_reports_a_hosted_backend_with_no_visible_token(monkeypatch, capsys):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    # Point at a dead port: no test may touch the real internet.
    code = main(["--api", "hf", "--base-url", "http://127.0.0.1:1", "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "google/gemma-3-4b-it" in out
    # Not a verdict: a proxy may add the header on the way out.
    assert "none here; expecting one upstream" in out
    # A hosted backend must say so out loud.
    assert "prompts are sent to" in out


def test_doctor_names_gemma_for_the_local_backend(capsys):
    code = main(["--base-url", "http://127.0.0.1:1", "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "gemma3:4b" in out
    assert "stays on this machine" in out
    assert "ollama pull gemma3:4b" in out
    assert "--api hf" in out


def test_an_unknown_backend_is_a_clean_error(capsys):
    with pytest.raises(SystemExit):
        main(["--api", "wishful-thinking", "doctor"])


def test_doctor_reports_the_cloud_backend(monkeypatch, capsys):
    for name in ("OLLAMA_API_KEY", "OLLAMA_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    code = main(["--api", "cloud", "--base-url", "http://127.0.0.1:1", "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "gpt-oss:120b" in out
    assert "none here; expecting one upstream" in out
    assert "prompts are sent to" in out


def test_doctor_probes_the_token_on_a_hosted_backend(server, capsys):
    """A public catalogue is not proof the service will talk to us."""
    server.reply["content"] = "ok"

    code = main(["--api", "cloud", "--base-url", server.url, "doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "token           works" in out
    assert "injected upstream" in out  # we hold no key of our own


def test_doctor_fails_when_the_token_is_rejected(server, capsys):
    server.reply.update(status=401, error="Unauthorized")

    code = main(["--api", "cloud", "--base-url", server.url, "doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "token           rejected" in out
    assert "ollama.com/settings/keys" in out  # and how to get a working one


def test_brief_warns_when_it_falls_back(tmp_path, capsys, monkeypatch):
    (tmp_path / "chat.txt").write_text("Ada: the deployment finished\n")

    # A model that is present but answers uselessly must not look like success.
    from sloth.agents import brief as brief_agent

    def useless(*_args, **_kwargs):
        result = brief_agent.Brief()
        result.summary = ["keywords only"]
        result.model_error = "the model returned nothing usable"
        return result

    monkeypatch.setattr(brief_agent, "summarise", useless)
    code = main(["--no-model", "brief", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "fell back to keyword summarising" in captured.err
