"""The morning after a run: the best methods found and every method of
yeelab.build.methods scored on the test set, on a finer grid, which is the real result.

Besides J at the share of the training (SHARE), the burial error at other shares, and J
over the groups of the test set: the random configurations, the hand-made ones and those
out of the training distribution (more candidates, other voters)."""

import json
import math
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from yeelab.build.methods import METHODS

from yeesearch.configs import SETS
from yeesearch.evaluate import NODE_COST, SCENARIOS, SHARE, objective
from yeesearch.pool import Archive, Workers, append_jsonl, parse, read_jsonl, synced

TEST_SIZE = 32  # Spearman's rho of the known methods' order against 64 is 0.999 already at 24
RUNS = Path("runs")
SHARES = (SHARE, 0.2, 0.45)
GROUPS = {"random": "random3-", "hand-made": "test", "ood": "ood-"}
console = Console(width=120)


def best_found(run: Path, top: int) -> list[str]:
    """The `top` methods of the hall of fame, by their J on the validation set, those
    without one last."""
    hall = json.loads((run / "hall.json").read_text())
    return [entry["method"] for entry in sorted(hall, key=lambda e: (e["validation"] is None, e["validation"] or 0,
                                                                       e["J"]))[:top]]


def summary(source: str, per_config: list[dict | None], share: str) -> dict:
    """J, sincere and burial error of a method at `share` over the configurations of
    `per_config` (score_config's results), inf where it failed on any."""
    if any(result is None for result in per_config):
        return {"J": math.inf, "sincere": math.inf, "burial": math.inf}
    scenario_errors = {scenario: [result[share][scenario] for result in per_config]
                       for scenario in SCENARIOS}
    return objective(parse(source), scenario_errors)


def run_report(run: Path | None, top: int = 20, size: int = TEST_SIZE, workers: int = 4, set_name: str = "test",
               cache_dir: Path = RUNS):
    """Scores the best methods of the run (none for None) and the methods of
    yeelab.build.methods on a set, writes report.json into the run and prints a table.
    What a method scores on a configuration is cached in cache_dir/<set>-<size>.jsonl
    for all runs, so the known methods are scored once and a stopped report goes on."""
    found = best_found(run, top) if run is not None else []
    names = {repr(method): name for name, method in METHODS.items()}
    sources = list(dict.fromkeys(found + list(names)))
    configs = SETS[set_name]()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _Cache(cache_dir / f"{set_name}-{size}.jsonl")
    todo = {index: [source for source in sources if (config.name, source) not in cache.results]
            for index, config in enumerate(configs)}
    console.print(f"{sum(map(len, todo.values()))} of {len(sources) * len(configs)} scores to compute")
    workers_ = Workers(workers)
    try:
        for index, result in workers_.configs(set_name, size, todo, SHARES):
            for source in todo[index]:
                cache.add(configs[index].name, source, result.get(source))
            console.print(f"configuration {index + 1}/{len(configs)} {configs[index].name}", end="\r")
    finally:
        workers_.close()
        cache.close()
    console.print()  # past the progress line
    rows = []
    for source in sources:
        per_config = [cache.results.get((config.name, source)) for config in configs]
        row = {"method": source, "name": names.get(source, f"found {found.index(source) + 1}"
                                                   if source in found else "")}
        for share in SHARES:
            row[str(share)] = summary(source, per_config, str(share))
        for group, prefix in GROUPS.items():
            picked = [k for k, config in enumerate(configs) if config.name.startswith(prefix)]
            if picked:
                row[group] = summary(source, [per_config[k] for k in picked], str(SHARE))["J"]
        rows.append(row)
    rows.sort(key=lambda row: row[str(SHARE)]["J"])
    if run is not None:
        (run / f"report-{set_name}-{size}.json").write_text(json.dumps(rows, indent=1))
    _print(rows, set_name, size)
    return rows


class _Cache:
    """{(configuration name, source): score_config's result for it} in a JSON-lines file."""

    def __init__(self, path: Path):
        self.results = {(entry["config"], entry["method"]): entry["result"] for entry in read_jsonl(path)}
        self._file = append_jsonl(path)

    def add(self, config: str, source: str, result: dict | None):
        self.results[config, source] = result
        self._file.write(json.dumps({"config": config, "method": source, "result": result}) + "\n")
        self._file.flush()

    def close(self):
        synced(self._file)
        self._file.close()


def _print(rows: list[dict], set_name: str, size: int):
    """The table of the methods by J, and the expressions of the methods found below it."""
    groups = [group for group in GROUPS if any(group in row for row in rows)]
    table = Table(title=f"{set_name} set, grid {size}: J = mean(0.4 sincere + 0.6 burial) + {NODE_COST} size",
                  caption="burial: at 30 % of the voters, then at 20 and 45 %; the groups: J on part of the set")
    for column in ("", "J", "sincere", "burial", *(f"{100 * s:.0f} %" for s in SHARES[1:]), *groups, "size"):
        table.add_column(column, justify="left" if not column else "right", no_wrap=True)
    for row in rows:
        main = row[str(SHARE)]
        table.add_row(row["name"], f"{main['J']:.4f}", f"{main['sincere']:.4f}", f"{main['burial']:.4f}",
                      *(f"{row[str(s)]['burial']:.4f}" for s in SHARES[1:]),
                      *(f"{row[group]:.4f}" for group in groups), str(main.get("size", "")))
    console.print(table)
    for row in rows:
        if row["name"].startswith("found"):
            console.print(f"{row['name']}: {row['method']}", soft_wrap=True)


def grid_agreement(set_name: str = "validation", sizes=(16, 64), workers: int = 4) -> dict:
    """How well the order of the methods of yeelab.build.methods by J on the coarse grid
    of the training matches that on finer ones: Spearman's rho against the last size."""
    sources = [repr(method) for method in METHODS.values()]
    names = {repr(method): name for name, method in METHODS.items()}
    scores = {}
    scorer = Workers(workers, timeout=600)
    try:
        for size in sizes:
            results = scorer.methods(sources, set_name, size, Archive())
            scores[size] = np.array([results[source]["J"] for source in sources])
    finally:
        scorer.close()
    finite = np.isfinite(np.vstack(list(scores.values()))).all(axis=0)
    reference = scores[sizes[-1]]
    rho = {size: float(spearmanr(scores[size][finite], reference[finite]).statistic) for size in sizes}
    table = Table(title=f"J of the methods on the {set_name} set by grid")
    table.add_column("")
    for size in sizes:
        table.add_column(str(size))
    for k in np.argsort(reference):
        table.add_row(names[sources[k]], *(f"{scores[size][k]:.4f}" for size in sizes))
    table.add_row("Spearman", *(f"{rho[size]:.3f}" for size in sizes))
    console.print(table)
    return rho
