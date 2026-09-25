import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from const import CANDIDATES, DEVIATION, PIXELS
from methods import METHODS
from ranking_cells import (
    generate_ranking_probabilities,
    read_cached_ranking_probabilities,
)

PLOTS = Path("plots")
console = Console()


def plot_yee_diagram(winners: np.ndarray, candidates: np.ndarray, title: str) -> Path:
    colors = list(plt.cm.tab20.colors[: len(candidates)])
    fig, ax = plt.subplots()
    ax.imshow(
        winners.T,
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=len(candidates) - 1,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="nearest",
    )
    ax.scatter(
        candidates[:, 0],
        candidates[:, 1],
        c=colors,
        s=40,
        edgecolors="#000000",
        linewidths=0.6,
        zorder=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("x median")
    ax.set_ylabel("y median")
    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / f"{title}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _generate_with_progress(pixels: int, deviation: float):
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"generating rankings ({pixels}x{pixels})", total=None)

        def update(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        return generate_ranking_probabilities(
            CANDIDATES, pixels, deviation, progress=update
        )


def main(
    pixels: int = typer.Option(PIXELS, "--pixels", "-p", help="Pixels per axis."),
    deviation: float = typer.Option(
        DEVIATION, "--deviation", "-d", help="Mean absolute deviation from the median."
    ),
    methods: list[str] = typer.Option(
        list(METHODS),
        "--method",
        "-m",
        help=f"Method to run, repeatable. Available: {', '.join(METHODS)}.",
    ),
    plot: bool = typer.Option(True, help="Save a Yee diagram for each method."),
    regenerate: bool = typer.Option(
        False, help="Ignore cached rankings and generate them again."
    ),
) -> None:
    """Compute Yee diagrams with Beta-distributed voters."""
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise typer.BadParameter(
            f"unknown method(s) {', '.join(unknown)}; choose from {', '.join(METHODS)}",
            param_hint="--method",
        )

    start = time.perf_counter()
    profile = None
    if not regenerate:
        profile = read_cached_ranking_probabilities(CANDIDATES, pixels, deviation)
    step = "load"
    if profile is None:
        profile = _generate_with_progress(pixels, deviation)
        step = "generate"
    rankings, probs = profile
    console.print(f"[bold]{step:<9}[/bold] {time.perf_counter() - start:.4f} s")

    for name in methods:
        with console.status(f"{name}..."):
            start = time.perf_counter()
            winners = METHODS[name](rankings, probs)
            computed = time.perf_counter()
            if plot:
                plot_yee_diagram(winners, CANDIDATES, name)
            plotted = time.perf_counter()
        line = f"[bold cyan]{name:<9}[/bold cyan] {computed - start:.4f} s"
        if plot:
            line += f"  [dim](plot {plotted - computed:.4f} s)[/dim]"
        console.print(line)


if __name__ == "__main__":
    typer.run(main)
