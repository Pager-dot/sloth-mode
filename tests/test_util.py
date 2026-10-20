from pathlib import Path

import pytest

from sloth import util


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ACME Invoice March 2024.pdf", "acme-invoice-march-2024.pdf"),
        ("  spaced   out.TXT ", "spaced-out.txt"),
        ("weird///path.png", "weird-path.png"),
        ("no-extension", "no-extension"),
        ("!!!.pdf", "untitled.pdf"),
        ("Résumé final FINAL.docx", "resume-final-final.docx"),
        ("archive.tar.gz", "archive.tar.gz"),
        ("under_scored_name.md", "under-scored-name.md"),
    ],
)
def test_slugify_filename(raw, expected):
    assert util.slugify_filename(raw) == expected


def test_slugify_truncates_but_keeps_extension():
    name = util.slugify_filename("x" * 300 + ".pdf", max_length=20)
    assert name.endswith(".pdf")
    assert len(Path(name).stem) <= 20


def test_human_size():
    assert util.human_size(512) == "512B"
    assert util.human_size(2048) == "2.0KB"
    assert util.human_size(5 * 1024**3) == "5.0GB"


def test_unique_destination_adds_counter(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("one")
    assert util.unique_destination(first).name == "a-2.txt"
    (tmp_path / "a-2.txt").write_text("two")
    assert util.unique_destination(first).name == "a-3.txt"


def test_safe_move_never_clobbers(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("new")
    dest_dir = tmp_path / "out"
    dest_dir.mkdir()
    (dest_dir / "src.txt").write_text("existing")

    final = util.safe_move(src, dest_dir / "src.txt")

    assert final.name == "src-2.txt"
    assert (dest_dir / "src.txt").read_text() == "existing"
    assert final.read_text() == "new"
    assert not src.exists()


def test_iter_files_skips_noise(messy_dir):
    names = [p.name for p in util.iter_files(messy_dir)]
    assert ".DS_Store" not in names
    assert "notes.txt" in names
    assert names == sorted(names)


def test_iter_files_recursive(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / "top.txt").write_text("x")
    (tmp_path / "sub" / "deeper" / "low.txt").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")

    flat = {p.name for p in util.iter_files(tmp_path)}
    deep = {p.name for p in util.iter_files(tmp_path, recursive=True)}

    assert flat == {"top.txt"}
    assert deep == {"top.txt", "low.txt"}
    assert "junk.js" not in deep


def test_iter_files_on_missing_dir(tmp_path):
    assert list(util.iter_files(tmp_path / "nope")) == []


def test_read_excerpt_only_reads_text(tmp_path):
    text = tmp_path / "a.md"
    text.write_text("hello world")
    binary = tmp_path / "a.png"
    binary.write_bytes(b"\x89PNG")

    assert util.read_excerpt(text) == "hello world"
    assert util.read_excerpt(binary) == ""
    assert util.read_excerpt(text, max_chars=5) == "hello"


def test_file_facts_describe(tmp_path):
    path = tmp_path / "invoice.txt"
    path.write_text("Amount due: 42")
    facts = util.FileFacts.gather(path, excerpt_chars=100)

    described = facts.describe()
    assert "invoice.txt" in described
    assert "Amount due: 42" in described
    assert facts.suffix == ".txt"
