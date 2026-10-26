import json
import random

from yeesearch import gp
from yeesearch.grammar import Node, check


def _one(name: str, J: float, behaviour: str) -> gp.Individual:
    return gp.Individual(Node("dac"), name, J, behaviour)


def test_only_the_best_of_methods_that_err_alike_stays():
    individuals = [_one("a", 0.3, "x"), _one("b", 0.1, "x"), _one("c", 0.2, "y"), _one("d", 0.05, "z")]
    kept = gp._distinct(random.Random(0), None, None, individuals, population=3)
    assert [one.source for one in kept] == ["d", "b", "c"]


def test_random_methods_are_new_and_typed_right():
    rng = random.Random(3)
    taken = {gp.source(tree) for tree in gp._random(rng, 20)}
    more = gp._random(rng, 30, taken)
    assert len(more) == 30 and all(check(tree) for tree in more)
    assert not taken & {gp.source(tree) for tree in more}


def test_a_short_run_logs_the_train_and_validation_best_and_starts_again(tmp_path, monkeypatch):
    """With STALL = 1 the second generation without a better method starts a new cycle
    from random trees; every generation logs the best on both sets."""
    monkeypatch.setattr(gp, "STALL", 1)
    gp.run(tmp_path, hours=1, population=12, workers=2, generations=3)
    records = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [record["generation"] for record in records] == [1, 2, 3]
    assert any(record["restart"] for record in records)
    for record in records:
        assert record["best_ever"] <= record["best"] and record["distinct"] <= 12
        assert record["validation_best"] is not None and record["validation_method"]
    hall = json.loads((tmp_path / "hall.json").read_text())
    assert hall and all(entry["validation"] is not None for entry in hall)
