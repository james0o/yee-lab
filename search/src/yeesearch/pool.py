"""Scoring methods in worker processes, each with a time limit (pebble kills a worker that
runs over it), and an archive of every method scored, so that a run can go on where it
stopped and no method is scored twice.

A method travels as its repr, the expression that builds it (blocks are frozen dataclasses
and eval(repr(block)) == block), which is also its key in the archive."""

import json
import math
import multiprocessing
import os
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from pebble import ProcessPool, ProcessExpired

import yeelab.build
import yeelab.threads
from yeelab.build.blocks import CombinedCondition, Threshold, WinnerComparison

from yeesearch.configs import SETS
from yeesearch.evaluate import SHARE, Prepared, errors, objective, warm

NAMESPACE = {name: getattr(yeelab.build, name) for name in yeelab.build.__all__}
NAMESPACE |= {"CombinedCondition": CombinedCondition, "Threshold": Threshold, "WinnerComparison": WinnerComparison}
TIMEOUT = 60.0  # seconds a method may take on the training set
FAILED = math.inf  # J of a method that fails or runs over its time
BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

_prepared: dict[tuple[str, int], list[Prepared]] = {}  # per worker: the configurations of a set on a grid


def parse(source: str):
    """The block of a method's repr."""
    return eval(source, dict(NAMESPACE))  # noqa: S307 -- our own reprs


class _Inline:
    """An executor that runs in the calling thread: a worker is one thread, and handing each
    chunk of a kernel to a thread of its own costs more than the kernel on the small grids
    of the search."""

    @staticmethod
    def map(fn, *iterables):
        return map(fn, *iterables)


def _start_worker(prepared: tuple[tuple[str, int], ...] = ()):
    """One thread per worker for yeelab's kernels, the workers being the parallelism, and
    the configurations of the sets on their grids `prepared` ready before the first method:
    preparing the training set takes tens of seconds, which must not count against the
    time limit of a method (a worker killed for it would prepare again, and run over again)."""
    yeelab.threads.pool = _Inline()
    for set_name, size in prepared:
        warm(prepare(set_name, size))


def prepare(set_name: str, size: int) -> list[Prepared]:
    key = (set_name, size)
    if key not in _prepared:
        _prepared[key] = [Prepared.of(config, size) for config in SETS[set_name]()]
    return _prepared[key]


def score_method(source: str, set_name: str, size: int) -> dict:
    """objective() of the method on the configurations of a set, with the errors of every scenario."""
    start = time.perf_counter()
    method = parse(source)
    scenario_errors = errors(method, prepare(set_name, size))
    result = objective(method, scenario_errors)
    rounded = {name: [round(value, 6) for value in values] for name, values in scenario_errors.items()}
    return result | {"errors": rounded, "seconds": round(time.perf_counter() - start, 3)}


def score_config(set_name: str, index: int, size: int, sources: list[str], shares=(SHARE,)) -> dict:
    """{source: {share: {scenario: error}}} of the methods on one configuration of a set
    (the sincere error the same at every share), None where a method fails: the many
    configurations of the test set one at a time, so that a worker never holds the voters
    of more than one."""
    prepared = Prepared.of(SETS[set_name]()[index], size)
    out = {}
    for source in sources:
        try:
            method = parse(source)
            out[source] = {str(share): {name: prepared.scorer.relative(winner)
                                        for name, winner in prepared.winners(method, share).items()}
                           for share in shares}
        except Exception:  # noqa: BLE001 -- a baseline that cannot run here scores nothing
            out[source] = None
    return out


def read_jsonl(path: Path) -> list[dict]:
    """The entries of a JSON-lines file, without a last line cut off (by a power cut)."""
    entries = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def append_jsonl(path: Path):
    """The file opened for appending lines, a line cut off before ended first, so that the
    next one does not join it."""
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as file:
            file.write("\n")
    return path.open("a")


def synced(file):
    """Written to the disk itself, not only to the system's cache: what survives a power cut."""
    file.flush()
    os.fsync(file.fileno())


class Archive:
    """Every method scored in a run, by source, in a JSON-lines file (None: in memory only)."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.results: dict[str, dict] = {}
        if path is not None:
            for entry in read_jsonl(path):
                self.results[entry["method"]] = entry
        self._file = None if path is None else append_jsonl(path)

    def add(self, source: str, result: dict):
        entry = {"method": source} | result
        self.results[source] = entry
        if self._file is not None:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()

    def sync(self):
        if self._file is not None:
            synced(self._file)

    def __contains__(self, source: str) -> bool:
        return source in self.results

    def __getitem__(self, source: str) -> dict:
        return self.results[source]

    def close(self):
        if self._file is not None:
            synced(self._file)
            self._file.close()


class Workers:
    """A pool of worker processes that score methods on a set of configurations."""

    def __init__(self, workers: int = 4, timeout: float = TIMEOUT, prepared: tuple[tuple[str, int], ...] = ()):
        """`prepared`: the (set, grid) every worker prepares when it starts, outside the time limit."""
        self.timeout = timeout
        for name in BLAS_THREADS:  # the workers inherit it: one thread each, as for yeelab's kernels
            os.environ[name] = "1"
        self.pool = ProcessPool(max_workers=workers, initializer=_start_worker, initargs=(tuple(prepared),),
                                context=multiprocessing.get_context("forkserver"))

    def methods(self, sources: list[str], set_name: str, size: int, archive: Archive) -> dict[str, dict]:
        """{source: result} of the methods, from the archive or scored now (and archived)."""
        todo = list(dict.fromkeys(source for source in sources if source not in archive))
        futures = {source: self.pool.schedule(score_method, (source, set_name, size), timeout=self.timeout)
                   for source in todo}
        for source, future in futures.items():
            try:
                result = future.result()
            except FutureTimeout:
                result = {"J": FAILED, "error": f"over {self.timeout} s"}
            except ProcessExpired as error:
                result = {"J": FAILED, "error": f"worker died: {error}"}
            except Exception as error:  # noqa: BLE001 -- a method that fails scores inf
                result = {"J": FAILED, "error": f"{type(error).__name__}: {error}"}
            archive.add(source, result)
        return {source: archive[source] for source in sources}

    def configs(self, set_name: str, size: int, todo: dict[int, list[str]], shares, timeout: float | None = None):
        """score_config() of the configurations of a set, {index: sources} of them, in
        order: (index, result)."""
        futures = {index: self.pool.schedule(score_config, (set_name, index, size, sources, shares), timeout=timeout)
                   for index, sources in todo.items() if sources}
        for index, future in futures.items():
            try:
                yield index, future.result()
            except (FutureTimeout, ProcessExpired) as error:
                yield index, {"error": repr(error)}

    def close(self, stop: bool = False):
        """Waits for the scheduled work, or with `stop` kills the workers."""
        if stop:
            self.pool.stop()
        else:
            self.pool.close()
        self.pool.join()
