"""yee-search: gp to search overnight, report to score the result on the test set, score
for one method, grids to check that the coarse grid of the training orders the methods
as a fine one does."""

from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
console = Console()


@app.command()
def gp(out: Path = typer.Option(..., help="directory of the run; an existing one is continued"),
       hours: float = typer.Option(10.0, help="how long to run"),
       population: int = typer.Option(1000, help="methods in a generation"),
       workers: int = typer.Option(4, help="worker processes"),
       seed: int = typer.Option(0),
       timeout: float = typer.Option(60.0, help="seconds a method may take on the training set"),
       generations: int | None = typer.Option(None, help="stop after this many generations"),
       until: str | None = typer.Option(None, help='stop at this time instead of after --hours: "19:00" (the next '
                                                     'one) or "2026-10-27 19:00"'),
       seed_from: Path | None = typer.Option(None, help="a directory of yee-search conditional: the best methods of its validation.json "
                                                 "(of each ranking) join the first generation of a new run, with "
                                                 "the library's known methods and random ones; ignored when the run "
                                                 "continues one in --out")):
    """Genetic programming over the grammar, against burial by 30 % of the voters."""
    from yeesearch.gp import run

    run(out, hours, population, workers, seed, timeout, generations, _time(until) if until else None, seed_from)


def _time(text: str) -> datetime:
    """"HH:MM", the next such time, or "YYYY-MM-DD HH:MM"."""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        pass
    try:
        clock = datetime.strptime(text, "%H:%M").time()
    except ValueError:
        raise typer.BadParameter(f'expected "HH:MM" or "YYYY-MM-DD HH:MM", got {text!r}') from None
    now = datetime.now()
    when = datetime.combine(now.date(), clock)
    return when if when > now else when + timedelta(days=1)


@app.command()
def report(run: Path = typer.Argument(None, help="directory of a run; none: only the known methods"),
           top: int = typer.Option(20, help="methods of the hall of fame to score"),
           size: int = typer.Option(32, help="grid"),
           workers: int = typer.Option(4),
           set_name: str = typer.Option("test", "--set", help="train, validation or test")):
    """The best methods of a run and the known methods on the test set."""
    from yeesearch.report import run_report

    run_report(run, top, size, workers, set_name)


@app.command()
def score(method: str = typer.Argument(..., help="the expression of a method, as repr prints it"),
          set_name: str = typer.Option("train", "--set"),
          size: int = typer.Option(16)):
    """J of one method, in this process."""
    from yeesearch.pool import prepare, parse
    from yeesearch.evaluate import errors, objective

    block = parse(method)
    result = objective(block, errors(block, prepare(set_name, size)))
    console.print(result)


@app.command()
def conditional(out: Path = typer.Option(..., help="directory of the search; the maps in it are reused"),
                max_size: int = typer.Option(8, help="blocks of the enumerated primitives"),
                same_size: int | None = typer.Option(None, help="blocks of a condition X == Y at most; none: any"),
                combine: int = typer.Option(1000, help="best conditions combined with & and |"),
                deeper: int = typer.Option(300, help="best Conditionals of each ranking taken a level deeper"),
                deeper_conditions: int = typer.Option(5000, help="best conditions of the level deeper"),
                top_train: int = typer.Option(200, help="best of the training scored on the validation set"),
                top_test: int = typer.Option(20, help="best of the validation of each ranking scored on the test set"),
                workers: int = typer.Option(4),
                estimate: bool = typer.Option(False, help="only count the Conditionals and time them"),
                summary: bool = typer.Option(False, help="only write summary.md again from a finished search"),
                pixel_median: str = typer.Option("geometric", help='"geometric" as the web UI starts, or "marginal"'),
                maps: Path | None = typer.Option(None, help="directory of the maps (default: OUT/maps)")):
    """Every Conditional(c, A, B) of the primitives (the grammar up to --max-size blocks,
    the known methods, the best of the GP runs), scored from their maps."""
    from yeesearch.conditional import run, summary_again

    if summary:
        summary_again(out, top_test)
        return
    run(out, Path("runs"), max_size, same_size, combine, deeper, deeper_conditions, top_train, top_test, workers,
        estimate, pixel_median, maps)


@app.command()
def table(runs: list[Path] = typer.Argument(..., help="directories of GP runs"),
          out: Path | None = typer.Option(None, help="the file (default: methods.py in the first run)"),
          max_j: float | None = typer.Option(None, help="only the methods of J at most this (a smaller file)")):
    """Every method the runs scored, best first, written short (fptp, nanson, ...) in a
    Python file to read and search in an editor."""
    from yeesearch.table import write_table

    out = out or runs[0] / "methods.py"
    console.print(f"{write_table(runs, out, max_j)} methods: {out}")


@app.command()
def find(run: Path = typer.Argument(..., help="directory of a GP run, or a file of yee-search table"),
         patterns: list[str] = typer.Argument(..., help='what a method holds, all of them: nanson "among=top2"'),
         regex: bool = typer.Option(False, "--regex", "-e", help="the patterns as regular expressions"),
         limit: int = typer.Option(50, "--limit", "-n", help="best matches shown; 0: all"),
         aliases: bool = typer.Option(False, "--aliases", "-a", help="show what score6, cmp6, ... stand for"),
         pager: bool = typer.Option(False, "--pager", "-p", help="in less, to scroll")):
    """The methods a run scored that hold the patterns, best first, highlighted: as
    grep over yee-search table, which it writes first if the run has no methods.py."""
    from rich import box
    from rich.table import Table
    from rich.text import Text

    from yeesearch.table import find as found_in, highlighted, layout, matchers, row, write_table

    table_path = run if run.is_file() else run / "methods.py"
    if not table_path.exists():
        console.print(f"writing {table_path} first (a minute or two)", style="dim")
        write_table([run], table_path)
    alias_lines, found = found_in(table_path, patterns, regex)
    shown = found if limit == 0 else found[:limit]
    matches = [match.pattern for match in matchers(patterns, regex)]

    def write():
        if aliases:
            for line in alias_lines:
                console.print(highlighted(line), soft_wrap=True)
        table = Table(box=box.SIMPLE_HEAD, show_lines=True, header_style="bold", pad_edge=False)
        table.add_column("method", overflow="fold", ratio=1)
        for name in ("J(0.3)", "sincere", "burial", "size", "#"):
            table.add_column(name, justify="right", style="dim" if name == "#" else "cyan", no_wrap=True)
        for rank, line in shown:
            *numbers, method = row(line)
            table.add_row(highlighted(layout(method), matches), *numbers, str(rank))
        console.print(table)
        console.print(Text(f"{len(found)} methods match" + (f", the best {len(shown)} shown" if len(shown) < len(found)
                                                            else ""), style="dim"))

    if pager:
        with console.pager(styles=True):
            write()
    else:
        write()


@app.command()
def grids(set_name: str = typer.Option("validation", "--set"),
          sizes: list[int] = typer.Option([16, 24, 64], "--size"),
          workers: int = typer.Option(4)):
    """J of the known methods on several grids, and Spearman's rho against the finest."""
    from yeesearch.report import grid_agreement

    grid_agreement(set_name, tuple(sizes), workers)
