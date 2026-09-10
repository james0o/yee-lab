import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, time
from const import CANDIDATES, N_CANDIDATES

CANDIDATE_COLORS = list(plt.cm.tab20.colors[:N_CANDIDATES])
NO_CONDORCET_COLOR = "#c8c8c8"

def plot_yeediagram(data: np.ndarray, title: str) -> None:
    n_colors = int(data.max()) + 1
    colors = CANDIDATE_COLORS[:n_colors]
    if n_colors > N_CANDIDATES:
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
    for i, (x, y) in enumerate(CANDIDATES):
        ax.scatter(
        x,
        y,
        c=[CANDIDATE_COLORS[i]],
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
    plt.savefig(f"plots/{title}.png", dpi=300, bbox_inches="tight")

def main():
    """Create Yee diagram (as PNG) for each voting method."""

    if not os.path.exists('plots'):
        os.makedirs('plots')

    start_time = time.time()
    for methods in os.listdir("winners"):
        method_name = methods.replace(".npy", "")  # Remove the .npy extension
        plot_yeediagram(np.load(f"winners/{methods}"), method_name)
    end_time = time.time()
    print(f"Time taken to generate all plots: {end_time - start_time:.2f} seconds")
    

if __name__ == "__main__":
    main()
