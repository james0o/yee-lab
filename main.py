from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from const import CANDIDATES
from methods import fptp

PLOTS = Path("plots")


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


def main() -> None:
    import time
    start = time.perf_counter()
    print(plot_yee_diagram(fptp(), CANDIDATES, "fptp"))
    print(f"Time taken, {time.perf_counter() - start:.4f} s")


if __name__ == "__main__":
    main()
