"""Export one Yee diagram as a PNG or SVG.

Run `uv run python -m yeelab.plot --help` for options.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from yeelab.distributions import Distribution
from yeelab.margin.regions import MARGINS, regions
from yeelab.margin.shares import Model
from yeelab.ranking_cells import SPREADS, Spread
from yeelab.voting import CYCLE

CANDIDATES = np.array([
    [0.6, 0.35],
    [0.25, 0.4],
    [0.35, 0.3],
    [0.5, 0.5],
    [0.3, 0.7],
])
COLORS = list(plt.get_cmap("tab20").colors)


def _candidate(value: str) -> tuple[float, float]:
    try:
        x, y = (float(coordinate) for coordinate in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be written as X,Y") from error
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("candidate coordinates must be between 0 and 1")
    return x, y


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Yee diagram as PNG or SVG.")
    parser.add_argument(
        "--method",
        choices=["voronoi", *MARGINS],
        default="irv",
        help="voting method (default: irv)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg"),
        default="png",
        help="output image format (default: png)",
    )
    parser.add_argument(
        "--svg",
        dest="format",
        action="store_const",
        const="svg",
        help="write SVG (short for --format svg)",
    )
    parser.add_argument("--output", type=Path, help="output path (default: <method>.<format>)")
    parser.add_argument(
        "--distribution",
        choices=("beta", "normal"),
        default="beta",
        help="voter distribution (default: beta)",
    )
    parser.add_argument(
        "--deviation",
        type=float,
        default=0.2,
        help="voter spread (default: 0.2)",
    )
    parser.add_argument(
        "--spread",
        choices=SPREADS,
        default="rms",
        help="Beta spread rule (default: rms)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=320,
        help="region tracing resolution per axis (default: 320)",
    )
    parser.add_argument(
        "--candidate",
        type=_candidate,
        action="append",
        dest="candidates",
        metavar="X,Y",
        help="candidate position in the unit square; repeat to set the layout",
    )
    return parser


def _draw(ax, method: str, candidates: np.ndarray, model: Model | None, size: int) -> None:
    for region in regions(method, candidates, model, size):
        winner = region["winner"]
        color = "#000000" if winner == CYCLE else COLORS[winner]
        for polygon in region["polygons"]:
            rings = [MplPath(np.reshape(ring, (-1, 2)), closed=True) for ring in polygon]
            path = MplPath.make_compound_path(*rings)
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor=color, lw=0.2))

    colors = COLORS[:len(candidates)]
    ax.scatter(*candidates.T, c=colors, s=45, edgecolors="black", linewidths=0.8, zorder=3)
    for index, (x, y) in enumerate(candidates):
        ax.annotate(chr(ord("A") + index), (x + 0.015, y + 0.015), weight="bold", fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    candidates = np.asarray(args.candidates if args.candidates else CANDIDATES, dtype=np.float64)
    if not 2 <= len(candidates) <= len(COLORS):
        _parser().error(f"provide between 2 and {len(COLORS)} candidates")
    if args.size < 2:
        _parser().error("--size must be at least 2")
    if not 0 < args.deviation:
        _parser().error("--deviation must be greater than 0")

    model = None if args.method == "voronoi" else Model(
        args.distribution, args.deviation,
        args.spread if args.distribution == "beta" else None,
    )
    output = args.output or Path(f"{args.method}.{args.format}")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    try:
        _draw(ax, args.method, candidates, model, args.size)
        fig.savefig(output, format=args.format, dpi=200)
    finally:
        plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
