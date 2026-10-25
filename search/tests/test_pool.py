import json
import math

from yeelab.build.methods import METHODS

from yeesearch.pool import Archive, Workers


def test_scores_are_archived(tmp_path):
    archive = Archive(tmp_path / "evals.jsonl")
    scorer = Workers(workers=2, timeout=120)
    try:
        sources = [repr(METHODS["fptp"]), repr(METHODS["schulze"])]
        results = scorer.methods(sources, "train", 8, archive)
        fptp, schulze = (results[source] for source in sources)
        assert schulze["J"] < fptp["J"] < math.inf
        assert len(fptp["errors"]["sincere"]) == 20
    finally:
        scorer.close()
        archive.close()
    again = Archive(tmp_path / "evals.jsonl")
    assert again[sources[0]]["J"] == fptp["J"]
    again.close()


def test_a_method_over_its_time_scores_inf():
    """The worker is killed: preparing the training set alone takes longer than this."""
    scorer = Workers(workers=1, timeout=0.5)
    try:
        result = scorer.methods([repr(METHODS["star"])], "train", 32, Archive())[repr(METHODS["star"])]
    finally:
        scorer.close()
    assert result["J"] == math.inf and "over" in result["error"]


def test_preparing_when_a_worker_starts_does_not_count_against_the_time():
    scorer = Workers(workers=1, timeout=5, prepared=(("train", 8),))
    try:
        result = scorer.methods([repr(METHODS["fptp"])], "train", 8, Archive())[repr(METHODS["fptp"])]
    finally:
        scorer.close()
    assert result["J"] < math.inf and result["seconds"] < 5


def test_a_failing_method_scores_inf():
    scorer = Workers(workers=1, timeout=60)
    try:
        result = scorer.methods(["Highest(Tally(Plurality()), n=2)"], "train", 8, Archive())
    finally:
        scorer.close()
    entry = result["Highest(Tally(Plurality()), n=2)"]
    assert entry["J"] == math.inf and "not a winner" in entry["error"]
    assert json.loads(json.dumps(entry))["J"] == math.inf
