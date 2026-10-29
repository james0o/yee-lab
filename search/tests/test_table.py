import ast

from yeesearch.pool import append_jsonl
from yeesearch.table import short, write_table


def test_short_names_whole_subtrees_longest_first():
    assert short("Highest(Tally(Plurality()))") == "fptp"
    assert short("Unbeaten(Margins(Pairwise()), among=Highest(Tally(Plurality()), n=2))") == "two_round"
    assert short("Highest(Tally(Plurality()), n=2)") == "top2"  # not fptp: n=2 is inside its brackets
    assert short("Unbeaten(StrongestPaths(Margins(Pairwise())))") == "schulze"
    assert short("Highest(Weakest(Margins(Pairwise())), n=3)") == "Highest(Weakest(margins), n=3)"


def test_table_is_python_best_first(tmp_path):
    with append_jsonl(tmp_path / "evals.jsonl") as file:
        file.write('{"method": "Highest(Tally(Plurality()))", "J": 0.1, "sincere": 0.1, "burial": 0.1, "size": 2}\n')
        file.write('{"method": "Highest(Tally(BordaCount()))", "J": 0.05, "sincere": 0.05, "burial": 0.05, "size": 2}\n')
        file.write('{"method": "Undecided()", "J": Infinity, "error": "failed"}\n')
    out = tmp_path / "methods.py"
    assert write_table([tmp_path], out) == 3
    text = out.read_text()
    ast.parse(text)
    rows = [line for line in text.splitlines() if line.startswith("(")]
    assert rows[0].endswith("borda)") and rows[1].endswith("fptp)") and rows[2].startswith("(     inf")
    assert write_table([tmp_path], out, max_j=0.07) == 1
