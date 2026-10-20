from pathlib import Path

from sloth.journal import Journal


def _move(src: Path, dst: Path) -> tuple[Path, Path]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return src, dst


def test_record_and_undo_roundtrip(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("hello")
    moves = [_move(src, tmp_path / "Admin" / "a.txt")]
    journal.record_moves("tidy", moves)

    result = journal.undo_last()

    assert len(result.restored) == 1
    assert src.read_text() == "hello"
    assert not (tmp_path / "Admin" / "a.txt").exists()


def test_dry_run_changes_nothing(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("hello")
    journal.record_moves("tidy", [_move(src, tmp_path / "out" / "a.txt")])

    result = journal.undo_last(dry_run=True)

    assert len(result.restored) == 1
    assert result.dry_run
    assert not src.exists()
    # Still undoable afterwards, because a dry run writes no undo marker.
    assert journal.batches()


def test_undo_walks_back_one_batch_at_a_time(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("1")
    second.write_text("2")
    journal.record_moves("tidy", [_move(first, tmp_path / "A" / "one.txt")])
    journal.record_moves("tidy", [_move(second, tmp_path / "B" / "two.txt")])

    assert len(journal.batches()) == 2

    journal.undo_last()
    assert second.exists() and not first.exists()

    journal.undo_last()
    assert first.exists()
    assert journal.batches() == []

    # A third undo is a no-op, not an error.
    assert journal.undo_last().empty


def test_undo_skips_when_target_name_is_taken(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("original")
    journal.record_moves("tidy", [_move(src, tmp_path / "out" / "a.txt")])
    src.write_text("something new lives here now")

    result = journal.undo_last()

    assert result.restored == []
    assert len(result.skipped) == 1
    assert "took that name" in result.skipped[0][1]
    assert src.read_text() == "something new lives here now"


def test_undo_skips_vanished_files(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("x")
    _, dst = _move(src, tmp_path / "out" / "a.txt")
    journal.record_moves("tidy", [(src, dst)])
    dst.unlink()

    result = journal.undo_last()

    assert result.restored == []
    assert result.skipped[0][1] == "no longer there"


def test_undo_specific_batch(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("1")
    second.write_text("2")
    batch_one = journal.record_moves("tidy", [_move(first, tmp_path / "A" / "one.txt")])
    journal.record_moves("tidy", [_move(second, tmp_path / "B" / "two.txt")])

    journal.undo_batch(batch_one)

    assert first.exists()
    assert not second.exists()
    assert journal.batches() != []


def test_history_is_newest_first(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    for index in range(3):
        path = tmp_path / f"f{index}.txt"
        path.write_text("x")
        journal.record_moves("tidy", [_move(path, tmp_path / "out" / f"f{index}.txt")])

    rows = journal.history()

    assert len(rows) == 3
    assert [row[2] for row in rows] == sorted((row[2] for row in rows), reverse=True)
    assert all(row[3] == "tidy" for row in rows)


def test_corrupt_lines_are_ignored(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    src = tmp_path / "a.txt"
    src.write_text("x")
    journal.record_moves("tidy", [_move(src, tmp_path / "out" / "a.txt")])
    with path.open("a") as handle:
        handle.write("this is not json\n\n")

    assert len(list(journal.entries())) == 1
    assert journal.undo_last().restored


def test_entries_on_missing_journal(tmp_path):
    assert list(Journal(tmp_path / "nope.jsonl").entries()) == []
    assert Journal(tmp_path / "nope.jsonl").history() == []


def test_prune_empty_dirs(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "f.txt").write_text("x")

    removed = Journal(tmp_path / "j.jsonl").prune_empty_dirs(tmp_path)

    assert removed == 2
    assert (tmp_path / "keep" / "f.txt").exists()
    assert not (tmp_path / "a").exists()


def test_prune_vacated_dirs_only_removes_what_undo_emptied(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("x")
    _move(src, tmp_path / "Admin" / "Invoices" / "a.txt")
    mine = tmp_path / "my-own-empty-dir"
    mine.mkdir()
    journal.record_moves("tidy", [(src, tmp_path / "Admin" / "Invoices" / "a.txt")])
    result = journal.undo_last()

    removed = journal.prune_vacated_dirs(result.restored)

    assert removed == 2
    assert not (tmp_path / "Admin").exists()
    assert mine.is_dir()
    assert src.exists()


def test_prune_vacated_dirs_keeps_non_empty_folders(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    src = tmp_path / "a.txt"
    src.write_text("x")
    dst = tmp_path / "Admin" / "a.txt"
    _move(src, dst)
    (tmp_path / "Admin" / "something-else.txt").write_text("y")
    journal.record_moves("tidy", [(src, dst)])
    result = journal.undo_last()

    assert journal.prune_vacated_dirs(result.restored) == 0
    assert (tmp_path / "Admin").is_dir()
