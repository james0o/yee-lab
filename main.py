import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import time
from pathlib import Path

from const import CANDIDATES, DISTANCE, N_PART, PIXELS
from methods import ElectionConfig, run_all

NO_CONDORCET_COLOR = "#c8c8c8"

def plot_yeediagram(data: np.ndarray, candidates: np.ndarray, title: str) -> None:
    n_colors = int(data.max()) + 1
    candidate_colors = list(plt.cm.tab20.colors[:len(candidates)])
    colors = candidate_colors[:n_colors]
    if n_colors > len(candidates):
        colors.append(NO_CONDORCET_COLOR)
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots()
    ax.imshow(
        data.T,
        cmap=cmap,
        vmin=0,
        vmax=n_colors - 1,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
    )
    for i, (x, y) in enumerate(candidates):
        ax.scatter(
        x,
        y,
        c=[candidate_colors[i]],
        s=40,
        edgecolors="#000000",
        linewidths=0.6,
        zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_title(title)
    ax.set_xlabel("x median")
    ax.set_ylabel("y median")
    fig.savefig(f"plots/{title}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main(argv: list[str] | None = None):
    """Create Yee diagram (as PNG) for each voting method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels", type=int, default=PIXELS)
    parser.add_argument("--n-part", type=int, default=N_PART)
    parser.add_argument("--distance", type=float, default=DISTANCE)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    args = parser.parse_args(argv)

    plots_path = Path("plots")
    plots_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    candidates = np.asarray(CANDIDATES, dtype=np.float64)
    results = run_all(
        ElectionConfig(
            pixels=args.pixels,
            n_part=args.n_part,
            distance=args.distance,
            candidates=candidates,
            cache_root=args.cache_root,
        )
    )
    for method_name, winners in results.items():
        plot_yeediagram(winners, candidates, method_name)
    end_time = time.time()
    print(f"Time taken to generate all plots: {end_time - start_time:.2f} seconds")
    

if __name__ == "__main__":
    main()
