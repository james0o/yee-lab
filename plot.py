import time
from functools import partial
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

from distributions import Distribution, model
from methods import CYCLE, METHODS, voronoi
from ranking_cells import NODES, SPREAD, Spread, effective_nodes

PIXELS = 400  # per axis, default of --pixels
DEVIATION = 0.2  # default of --deviation
CANDIDATES = np.array([
    [0.6,0.35],
    [0.25,0.4],
    [0.35,0.3],
    [0.5, 0.5],
    [0.3, 0.7]], dtype=np.float64)
PLOTS = Path("plots")
DIAGRAMS = ["voronoi", *METHODS]  # voronoi needs no voters, so it has no model folder
console = Console()
app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def plot_yee_diagram(
    winners: np.ndarray, candidates: np.ndarray, title: str, path: Path
) -> Path:
    colors = list(plt.cm.tab20.colors[: len(candidates)])
    # CYCLE gets the extra last colour, black
    winners = np.where(winners == CYCLE, len(candidates), winners)
    fig, ax = plt.subplots()
    ax.imshow(
        winners.T,
        cmap=ListedColormap(colors + ["#000000"]),
        vmin=0,
        vmax=len(candidates),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _generate_with_progress(module, pixels: int, deviation: float, nodes: int, **options):
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
        nodes = effective_nodes(pixels, nodes)
        grid = f"{nodes}x{nodes} nodes" if nodes else f"{pixels}x{pixels} exact"
        task = progress.add_task(f"generating rankings ({grid})", total=None)

        def update(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        return module.generate_ranking_probabilities(
            CANDIDATES, pixels, deviation, nodes, progress=update, **options
        )


@app.command()
def main(
    pixels: int = typer.Option(PIXELS, "--pixels", "-p", help="Pixels per axis."),
    deviation: float = typer.Option(
        DEVIATION,
        "--deviation",
        "-d",
        help="Mean absolute deviation from the median at the centre pixel "
        "(everywhere for normal voters and for --spread mean_abs).",
    ),
    distribution: Distribution = typer.Option(
        "beta", help="Voter distribution around each pixel."
    ),
    spread: Spread = typer.Option(
        SPREAD,
        help="Beta only: what stays the same for every pixel. mean_abs (legacy): E|X - m|; "
        "rms: sqrt(E (X - m)^2); tapered: (a + b) times (4m(1 - m))^0.2. "
        "All agree at the centre pixel.",
    ),
    nodes: int = typer.Option(
        NODES,
        "--nodes",
        "-n",
        help="Chebyshev nodes per axis where probabilities are computed exactly "
        "and then interpolated to the pixels; 0 = exact at every pixel.",
    ),
    methods: list[str] = typer.Option(
        DIAGRAMS,
        "--method",
        "-m",
        help=f"Method to run, repeatable. Available: {', '.join(DIAGRAMS)}.",
    ),
    plot: bool = typer.Option(True, help="Save a Yee diagram for each method."),
    regenerate: bool = typer.Option(
        False, help="Ignore cached rankings and generate them again."
    ),
) -> None:
    """Compute Yee diagrams with Beta or normally distributed voters."""
    unknown = [name for name in methods if name not in DIAGRAMS]
    if unknown:
        raise typer.BadParameter(
            f"unknown method(s) {', '.join(unknown)}; choose from {', '.join(DIAGRAMS)}",
            param_hint="--method",
        )
    try:
        effective_nodes(pixels, nodes)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--nodes") from error

    def run(name, compute, title, path):
        with console.status(f"{name}..."):
            start = time.perf_counter()
            winners = compute()
            computed = time.perf_counter()
            if plot:
                plot_yee_diagram(winners, CANDIDATES, title, path)
            plotted = time.perf_counter()
        line = f"[bold cyan]{name:<9}[/bold cyan] {computed - start:.4f} s"
        if plot:
            line += f"  [dim](plot {plotted - computed:.4f} s)[/dim]"
        console.print(line)

    if "voronoi" in methods:
        run("voronoi", partial(voronoi, CANDIDATES, pixels), "voronoi", PLOTS / "voronoi.png")
    voting = [name for name in methods if name != "voronoi"]
    if not voting:
        return

    module, options = model(distribution, spread)
    label = f"beta_{spread}" if distribution == "beta" else distribution
    # plots/normal/<method>.png, plots/beta/<spread>/<method>.png
    folder = PLOTS / "beta" / spread if distribution == "beta" else PLOTS / "normal"
    start = time.perf_counter()
    profile = None
    if not regenerate:
        profile = module.read_cached_ranking_probabilities(
            CANDIDATES, pixels, deviation, nodes, **options
        )
    step = "load"
    if profile is None:
        profile = _generate_with_progress(module, pixels, deviation, nodes, **options)
        step = "generate"
    rankings, probs = profile
    console.print(f"[bold]{step:<9}[/bold] {time.perf_counter() - start:.4f} s")

    for name in voting:
        run(name, partial(METHODS[name], rankings, probs), f"{name}_{label}", folder / f"{name}.png")

if __name__ == "__main__":
    app()
