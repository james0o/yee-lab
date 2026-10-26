"""A run survives a power cut: what it wrote loads without the line cut off, the next line
does not join it, two runs never work in one directory, and a run past its end does nothing."""

import fcntl
import json
from datetime import datetime, timedelta

import pytest
import typer

from yeesearch import gp
from yeesearch.cli import _time
from yeesearch.pool import Archive, append_jsonl, read_jsonl


def test_a_line_cut_off_is_left_out_and_the_next_one_starts_anew(tmp_path):
    path = tmp_path / "evals.jsonl"
    path.write_text(json.dumps({"method": "a", "J": 0.1}) + "\n" + '{"method": "b", "J"')
    assert read_jsonl(path) == [{"method": "a", "J": 0.1}]
    archive = Archive(path)
    assert "a" in archive and "b" not in archive
    archive.add("c", {"J": 0.2})
    archive.close()
    assert [entry["method"] for entry in read_jsonl(path)] == ["a", "c"]
    with append_jsonl(path) as file:
        file.write("{}\n")
    assert len(read_jsonl(path)) == 3


def test_one_run_at_a_time_in_a_directory(tmp_path):
    with (tmp_path / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="another run"):
            gp.run(tmp_path, hours=1)


def test_a_run_past_its_end_does_nothing(tmp_path):
    gp.run(tmp_path, hours=1, until=datetime.now() - timedelta(minutes=1))
    assert not (tmp_path / "evals.jsonl").exists()


def test_the_end_of_a_run_is_a_time():
    assert _time("2026-10-27 19:00") == datetime(2026, 10, 27, 19, 0)
    soon = _time("19:00")
    assert soon.time().hour == 19 and datetime.now() < soon <= datetime.now() + timedelta(days=1)
    with pytest.raises(typer.BadParameter):
        _time("tomorrow")
