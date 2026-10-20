import json

import pytest

from sloth.agents import tidy
from sloth.journal import Journal
from sloth.model import NullModel
from sloth.util import FileFacts


def batch(*entries):
    return json.dumps({"files": list(entries)})


def entry(filename, category, new_name=None, confidence=0.9, reason="because"):
    item = {
        "filename": filename,
        "category": category,
        "confidence": confidence,
        "reason": reason,
    }
    if new_name:
        item["new_name"] = new_name
    return item


# ---------------------------------------------------------------------------
# heuristics (the no-model path)
# ---------------------------------------------------------------------------


def test_heuristic_matches_filename_keywords(messy_dir, config):
    facts = FileFacts.gather(messy_dir / "ACME Invoice March 2024.pdf")

    proposal = tidy.heuristic_proposal(facts, config.tidy)

    assert proposal.category == "Admin/Invoices"
    assert proposal.new_name == "acme-invoice-march-2024.pdf"
    assert proposal.decided_by == "heuristic"
    assert proposal.confidence >= 0.6


def test_heuristic_falls_back_to_extension(messy_dir, config):
    facts = FileFacts.gather(messy_dir / "dataset_export.csv")

    proposal = tidy.heuristic_proposal(facts, config.tidy)

    assert proposal.category == "Code/Data"


def test_heuristic_gives_up_loudly(messy_dir, config):
    facts = FileFacts.gather(messy_dir / "thing.qqq")

    proposal = tidy.heuristic_proposal(facts, config.tidy)

    assert proposal.category == config.tidy.unsorted_name
    assert proposal.confidence < config.tidy.confidence_threshold


def test_plan_without_a_model_still_works(messy_dir, config):
    plan = tidy.plan(messy_dir, config, NullModel())

    categories = {p.path.name: p.category for p in plan.proposals}
    assert categories["ACME Invoice March 2024.pdf"] == "Admin/Invoices"
    assert categories["Screenshot 2024-03-02 at 11.04.21.png"] == "Images/Screenshots"
    assert categories["thing.qqq"] == "_Unsure"
    assert plan.used_model is False
    assert ".DS_Store" not in categories


def test_no_rename_keeps_original_names(messy_dir, config):
    config.tidy.rename = False

    plan = tidy.plan(messy_dir, config, NullModel())

    assert all(not p.renamed for p in plan.proposals)


# ---------------------------------------------------------------------------
# the model path
# ---------------------------------------------------------------------------


def test_model_decisions_are_used(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "scan001.pdf").write_bytes(b"%PDF")
    model = fake_model(
        [batch(entry("scan001.pdf", "Admin/Tax", "tax-return-2023.pdf", 0.92))]
    )

    plan = tidy.plan(root, config, model)

    assert plan.used_model
    assert len(plan.proposals) == 1
    proposal = plan.proposals[0]
    assert proposal.category == "Admin/Tax"
    assert proposal.new_name == "tax-return-2023.pdf"
    assert proposal.decided_by == "model"
    # The allowed taxonomy is in the prompt, so the model can copy it verbatim.
    assert "Admin/Tax" in model.last_prompt


def test_invented_categories_fall_back_to_heuristics(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "march-invoice.pdf").write_bytes(b"%PDF")
    model = fake_model([batch(entry("march-invoice.pdf", "Vibes/Whatever", None, 0.99))])

    plan = tidy.plan(root, config, model)

    assert plan.proposals[0].category == "Admin/Invoices"
    assert plan.proposals[0].decided_by == "heuristic"


def test_short_category_names_are_matched_to_the_full_path(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "x.pdf").write_bytes(b"%PDF")
    model = fake_model([batch(entry("x.pdf", "invoices", None, 0.8))])

    plan = tidy.plan(root, config, model)

    assert plan.proposals[0].category == "Admin/Invoices"


def test_model_cannot_change_the_extension(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "report.pdf").write_bytes(b"%PDF")
    model = fake_model([batch(entry("report.pdf", "Admin/Tax", "report.exe", 0.9))])

    plan = tidy.plan(root, config, model)

    assert plan.proposals[0].new_name == "report.pdf"


def test_model_cannot_escape_the_root_with_a_path(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "report.pdf").write_bytes(b"%PDF")
    model = fake_model(
        [batch(entry("report.pdf", "Admin/Tax", "../../../etc/passwd.pdf", 0.9))]
    )

    plan = tidy.plan(root, config, model)
    destination = plan.proposals[0].destination(root)

    assert ".." not in destination.parts
    assert root in destination.parents


def test_low_confidence_is_rescued_by_the_heuristic(tmp_path, config, fake_model):
    """A hesitant model answer loses to a confident extension rule."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "untitled.pdf").write_bytes(b"%PDF")
    model = fake_model([batch(entry("untitled.pdf", "Admin/Tax", None, 0.2))])

    plan = tidy.plan(root, config, model)

    assert plan.proposals[0].category == "Reading/Papers"
    assert plan.proposals[0].decided_by == "heuristic"


def test_low_confidence_goes_to_unsure(tmp_path, config, fake_model):
    """When neither the model nor the heuristic is sure, nobody guesses."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "untitled.qqq").write_bytes(b"?")
    model = fake_model([batch(entry("untitled.qqq", "Admin/Tax", None, 0.2))])

    plan = tidy.plan(root, config, model)

    assert plan.proposals[0].category == "_Unsure"
    assert "not confident" in plan.proposals[0].reason or plan.proposals[0].reason


def test_unanswered_files_fall_back(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "a-invoice.pdf").write_bytes(b"%PDF")
    (root / "b-screenshot.png").write_bytes(b"\x89PNG")
    # The model answers for only one of the two files.
    model = fake_model([batch(entry("a-invoice.pdf", "Admin/Invoices", None, 0.9))])

    plan = tidy.plan(root, config, model)

    decided = {p.path.name: p.decided_by for p in plan.proposals}
    assert decided["a-invoice.pdf"] == "model"
    assert decided["b-screenshot.png"] == "heuristic"


def test_model_failure_degrades_gracefully(messy_dir, config, fake_model):
    from sloth.model import ModelUnavailable

    model = fake_model([ModelUnavailable("ollama fell asleep")])

    plan = tidy.plan(messy_dir, config, model)

    assert plan.used_model is False
    assert "fell asleep" in plan.model_error
    assert plan.proposals  # heuristics still produced a plan


def test_batches_are_capped(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    for index in range(tidy.BATCH_SIZE + 3):
        (root / f"file{index:02d}.pdf").write_bytes(b"%PDF")
    model = fake_model([batch(), batch()])

    tidy.plan(root, config, model)

    assert len(model.calls) == 2


def test_excerpt_is_sent_for_text_files(tmp_path, config, fake_model):
    root = tmp_path / "d"
    root.mkdir()
    (root / "mystery.txt").write_text("INVOICE 4421 — amount due 320 EUR")
    model = fake_model([batch(entry("mystery.txt", "Admin/Invoices", None, 0.8))])

    tidy.plan(root, config, model)

    assert "amount due 320 EUR" in model.last_prompt


# ---------------------------------------------------------------------------
# applying and undoing
# ---------------------------------------------------------------------------


def test_apply_then_undo_restores_everything(messy_dir, config):
    before = sorted(p.name for p in messy_dir.iterdir())
    plan = tidy.plan(messy_dir, config, NullModel())
    journal = Journal(config.journal_path)

    moves = tidy.apply_plan(plan, journal)

    assert moves
    assert (messy_dir / "Admin" / "Invoices" / "acme-invoice-march-2024.pdf").exists()
    assert not (messy_dir / "ACME Invoice March 2024.pdf").exists()

    journal.undo_last()
    journal.prune_empty_dirs(messy_dir)

    assert sorted(p.name for p in messy_dir.iterdir()) == before


def test_already_filed_files_are_left_alone(messy_dir, config):
    plan = tidy.plan(messy_dir, config, NullModel())
    journal = Journal(config.journal_path)
    tidy.apply_plan(plan, journal)

    second = tidy.plan(messy_dir, config, NullModel(), recursive=True)

    assert second.actionable == []
    assert second.skipped


def test_apply_is_a_noop_for_vanished_files(tmp_path, config):
    root = tmp_path / "d"
    root.mkdir()
    (root / "invoice.pdf").write_bytes(b"%PDF")
    plan = tidy.plan(root, config, NullModel())
    (root / "invoice.pdf").unlink()

    assert tidy.apply_plan(plan, Journal(config.journal_path)) == []


def test_plan_limit(messy_dir, config):
    plan = tidy.plan(messy_dir, config, NullModel(), limit=2)

    assert len(plan.proposals) == 2


def test_plan_rejects_non_directories(tmp_path, config):
    path = tmp_path / "a.txt"
    path.write_text("x")

    with pytest.raises(NotADirectoryError):
        tidy.plan(path, config, NullModel())


def test_plan_json_is_machine_readable(messy_dir, config):
    plan = tidy.plan(messy_dir, config, NullModel())

    payload = json.loads(plan.to_json())

    assert payload["root"] == str(messy_dir)
    assert payload["used_model"] is False
    assert all({"from", "to", "confidence"} <= set(m) for m in payload["moves"])


def test_noop_proposals_are_not_actionable(tmp_path):
    root = tmp_path / "d"
    already_there = root / "Admin/Invoices/acme-invoice.pdf"
    proposal = tidy.Proposal(
        path=already_there,
        category="Admin/Invoices",
        new_name="acme-invoice.pdf",
        confidence=0.9,
    )
    plan = tidy.TidyPlan(root=root, proposals=[proposal])

    assert proposal.is_noop(root)
    assert plan.actionable == []
    assert plan.by_category() == {}
