"""Figures and numbers of the chapter "Beta versus normal voters" in math.typ.

Run from the repository root: `uv run python docs/figures.py`.
Writes docs/figures/*.png and prints the numbers quoted in the text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.special import betainc, ndtr, ndtri
from scipy.stats import beta as beta_dist

import normal
import ranking_cells
from const import CANDIDATES
from methods import CYCLE, condorcet_cycle, fptp, ideal, irv, schulze

PIXELS = 300
DEVIATION = 0.3
NAMES = "ABCDE"
A, B, C, D, E = range(5)
# the palette of the cell figure in math.typ, black for Condorcet cycles
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#000000"]
FIGURES = Path(__file__).parent / "figures"
MEDIANS = ranking_cells.pixel_medians(PIXELS)
MODELS = {"beta": ranking_cells, "normal": normal}


def profile(name):
    model = MODELS[name]
    cached = model.read_cached_ranking_probabilities(CANDIDATES, PIXELS, DEVIATION)
    if cached is None:
        cached = model.generate_ranking_probabilities(CANDIDATES, PIXELS, DEVIATION)
    return cached


def show(ax, winners, title, alpha=1.0):
    winners = np.where(winners == CYCLE, 5, winners)
    ax.imshow(winners.T, cmap=ListedColormap(PALETTE), vmin=0, vmax=5, origin="lower",
              extent=(0, 1, 0, 1), interpolation="nearest", alpha=alpha)
    ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=45, edgecolors="k", linewidths=1, zorder=3)
    for name, (x, y) in zip(NAMES, CANDIDATES):
        ax.annotate(name, (x + 0.015, y + 0.015), weight="bold", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)


def first_choice_shares(rankings, probs, alive):
    """Share of voters whose first choice among `alive` is each candidate."""
    first = np.array([next(c for c in r if c in alive) for r in rankings])
    return {c: probs[..., first == c].sum(axis=-1) for c in alive}


UPPER = np.linspace(0.5, 0.9985, 400)
# solved as one sweep: beta_params_at needs the continuation from 1/2 outwards
UPPER_PARAMS = ranking_cells.beta_params_at(UPPER, DEVIATION)


def beta_marginal(median):
    """Frozen Beta distribution with the given median."""
    a, b = UPPER_PARAMS[np.argmin(abs(UPPER - max(median, 1 - median)))]
    return beta_dist(a, b) if median >= 0.5 else beta_dist(b, a)


def compare(profiles):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.9))
    methods = [("IRV", irv), ("Schulze", schulze), ("Condorcet winner", condorcet_cycle)]
    tie = _ties()
    for row, name in enumerate(profiles):
        rankings, probs = profiles[name]
        voronoi = ideal(rankings, probs)
        for col, (label, method) in enumerate(methods):
            winners = method(rankings, probs)
            show(axes[row, col], winners, f"{name}: {label}")
        sch, cyc = schulze(rankings, probs), condorcet_cycle(rankings, probs)
        print(f"{name}: schulze != Voronoi on {np.sum((sch != voronoi) & ~tie)} pixels, "
              f"cycles {np.sum((cyc == CYCLE) & ~tie)} (of {np.sum(~tie)} without ties); "
              f"areas Voronoi {[round(np.mean(voronoi == c), 3) for c in range(5)]} "
              f"Schulze {[round(np.mean(sch == c), 3) for c in range(5)]}")
    fig.tight_layout()
    fig.savefig(FIGURES / "compare.png", dpi=150)
    plt.close(fig)


def _ties():
    """Pixels whose centre is equidistant (to rounding) from its two nearest candidates."""
    centres = np.stack(np.meshgrid(MEDIANS, MEDIANS, indexing="ij"), axis=-1)
    dist = np.sort(np.linalg.norm(centres[..., None, :] - CANDIDATES, axis=-1), axis=-1)
    tie = dist[..., 1] - dist[..., 0] < 1e-12
    print(f"pixels on a bisector (ties): {tie.sum()}")
    return tie


def irv_round(profiles):
    rankings, probs = profiles["beta"]
    shares = first_choice_shares(rankings, probs, (A, D, E))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    show(ax, irv(rankings, probs), "Beta IRV and the round 3 ties among A, D, E", alpha=0.55)
    x, y = np.meshgrid(MEDIANS, MEDIANS, indexing="ij")
    for (p, q), color, style in [((A, D), "red", "-"), ((D, E), "black", "--"), ((A, E), "purple", ":")]:
        ax.contour(x, y, shares[p] - shares[q], levels=[0], colors=color, linestyles=style, linewidths=2)
        ax.plot([], [], color=color, linestyle=style, label=f"{NAMES[p]} = {NAMES[q]}")
    t = np.linspace(0, 1, 50)
    ax.plot(t, 0.425 + (t - 0.55) * 2 / 3, color="gray", lw=1, label="bisectors D|A, D|E")
    ax.plot(t, t + 0.2, color="gray", lw=1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "irv_round.png", dpi=150)
    plt.close(fig)

    i = int(0.55 * PIXELS)
    for name in profiles:
        rankings, probs = profiles[name]
        shares = first_choice_shares(rankings, probs, (A, D, E))
        winners = irv(rankings, probs)
        print(f"{name}: column x = {MEDIANS[i]:.3f}, round 3 shares among A, D, E")
        for y in np.arange(0.62, 1.0, 0.06):
            j = min(int(y * PIXELS), PIXELS - 1)
            extra = ""
            if name == "beta":
                extra = f"  P(Y<0.2) = {beta_marginal(MEDIANS[j]).cdf(0.2):.3f}"
            print(f"  y = {MEDIANS[j]:.3f}: winner {NAMES[winners[i, j]]}  "
                  + "  ".join(f"{NAMES[c]} {shares[c][i, j]:.3f}" for c in (A, D, E)) + extra)


def fptp_edge(profiles):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, name in zip(axes, profiles):
        rankings, probs = profiles[name]
        winners = fptp(rankings, probs)
        show(ax, winners, f"{name}: FPTP")
        print(f"{name}: FPTP boundaries near the left edge")
        for x in (0.2, 0.08, 0.03, 0.005):
            col = winners[int(x * PIXELS)]
            be = [round(MEDIANS[j], 3) for j in range(PIXELS - 1) if {col[j], col[j + 1]} == {B, E}]
            cb = [round(MEDIANS[j], 3) for j in range(PIXELS - 1) if {col[j], col[j + 1]} == {C, B}]
            extra = ""
            if name == "beta":
                extra = f"  P(X<0.02) = {beta_marginal(x).cdf(0.02):.3f}"
            print(f"  x = {x}: B|E at y {be}, C|B at y {cb}{extra}")
    fig.tight_layout()
    fig.savefig(FIGURES / "fptp_edge.png", dpi=150)
    plt.close(fig)


def beta_shapes():
    print("Beta marginals, deviation 0.3")
    for m in (0.5, 0.7, 0.9, 0.98):
        dist = beta_marginal(m)
        a, b = dist.args
        print(f"  median {m}: a {a:.3f} b {b:.3f} mean {dist.mean():.3f} "
              f"P(X<0.1) {dist.cdf(0.1):.2f} P(X>0.9) {dist.sf(0.9):.2f}")


def pull_example(samples=4_000_000, seed=0):
    """Share preferring A to D at pixels on A's side of their bisector."""
    rng = np.random.default_rng(seed)
    n = CANDIDATES[A] - CANDIDATES[D]
    offset = (CANDIDATES[A] @ CANDIDATES[A] - CANDIDATES[D] @ CANDIDATES[D]) / 2
    sigma = normal.sigma_from_deviation(DEVIATION)
    from scipy.special import ndtr
    for pixel in [(0.95, 0.55), (0.9, 0.6), (0.8, 0.45)]:
        bx, by = beta_marginal(pixel[0]), beta_marginal(pixel[1])
        x, y = bx.rvs(samples, random_state=rng), by.rvs(samples, random_state=rng)
        share = np.mean(n[0] * x + n[1] * y > offset)
        dist = (n @ pixel - offset) / np.linalg.norm(n)
        print(f"  pixel {pixel}: distance {dist:+.3f}, Beta share {share:.3f}, "
              f"normal share {ndtr(dist / sigma):.3f}, Beta mean ({bx.mean():.3f}, {by.mean():.3f})")


# ---------------------------------------------------------------- round edges

# the seven candidates of the FPTP example (B removed, three added on the left)
SEVEN = np.array([(0.6, 0.35), (0.35, 0.3), (0.5, 0.5), (0.3, 0.7),
                  (0.15, 0.31), (0.07, 0.45), (0.04, 0.62)])
SEVEN_PALETTE = ["#4e79a7", "#76b7b2", "#e15759", "#b07aa1", "#59a14f", "#edc948", "#f28e2b"]


def voronoi_lines(ax, candidates, **style):
    """Borders of the Voronoi cells (the `ideal` diagram), drawn as contours."""
    t = np.linspace(0, 1, 800)
    x, y = np.meshgrid(t, t, indexing="ij")
    nearest = np.linalg.norm(np.stack([x, y], -1)[..., None, :] - candidates, axis=-1).argmin(-1)
    for c in range(len(candidates)):
        ax.contour(x, y, (nearest == c).astype(float), levels=[0.5], **style)


def blur_corner():
    """Three candidates whose cells meet at (1/2, 1/2) with angles 90, 135, 135 degrees.
    The 90 degree cell is the quadrant {x < 1/2, y < 1/2}, whose share is exactly
    Phi((1/2 - x) / sigma) Phi((1/2 - y) / sigma)."""
    deviation = 0.2
    sigma = normal.sigma_from_deviation(deviation)
    a = 0.3
    three = np.array([(0.5 - a / 2, 0.5 - a / 2), (0.5 + a / 2, 0.5 - a / 2), (0.5 - a / 2, 0.5 + a / 2)])
    colors = ["#4e79a7", "#e15759", "#59a14f"]
    pixels = 200
    rankings, probs = normal.ranking_probabilities(three, pixels, deviation, 0)
    m = ranking_cells.pixel_medians(pixels)
    x, y = np.meshgrid(m, m, indexing="ij")
    quadrant = ndtr((0.5 - x) / sigma) * ndtr((0.5 - y) / sigma)
    shares = {c: probs[..., rankings[:, 0] == c].sum(-1) for c in range(3)}
    print(f"blur_corner: sigma {sigma:.4f}; max |quadrant formula - exact share| "
          f"{abs(quadrant - shares[0]).max():.1e}; at the vertex shares "
          f"{[round(float(np.interp(0.5, m, shares[c][:, np.searchsorted(m, 0.5)])), 3) for c in range(3)]}; "
          f"Phi^-1(1/sqrt 2) = {ndtri(2 ** -0.5):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    ax = axes[0]
    ax.imshow((x < 0.5) & (y < 0.5), cmap="Greys", origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1.6)
    ax.set_title("cell of the first candidate (quadrant)", fontsize=10)
    ax = axes[1]
    ax.imshow(quadrant.T, cmap="Greys", origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1.6)
    cs = ax.contour(x, y, quadrant, levels=[0.25, 0.375, 0.5, 0.75], colors="#4e79a7", linewidths=1.2)
    ax.clabel(cs, fontsize=7)
    ax.plot([0.5, 0.5, 0], [0, 0.5, 0.5], color="k", lw=0.8, ls="--")
    ax.set_title(r"its share: blurred, $\Phi(\frac{1/2-x}{\sigma})\Phi(\frac{1/2-y}{\sigma})$", fontsize=10)
    ax = axes[2]
    winners = fptp(rankings, probs)
    ax.imshow(winners.T, cmap=ListedColormap(colors), vmin=0, vmax=2, origin="lower",
              extent=(0, 1, 0, 1), interpolation="nearest")
    voronoi_lines(ax, three, colors="k", linewidths=0.8, linestyles="--")
    ax.scatter(*three.T, c=colors, s=45, edgecolors="k", zorder=3)
    ax.set_title(f"normal FPTP, $\\sigma$ = {sigma:.2f} (dashed: Voronoi)", fontsize=10)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "blur_corner.png", dpi=150)
    plt.close(fig)


def seven_fptp():
    """FPTP for seven candidates: Voronoi, normal with a small and a large blur, Beta."""
    pixels = 200
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    panels = [("Voronoi (no blur)", None, None), ("normal, d = 0.1", normal, 0.1),
              ("normal, d = 0.3", normal, 0.3), ("Beta, d = 0.3", ranking_cells, 0.3)]
    for ax, (title, model, deviation) in zip(axes, panels):
        if model is None:
            m = ranking_cells.pixel_medians(pixels)
            centres = np.stack(np.meshgrid(m, m, indexing="ij"), -1)
            winners = np.linalg.norm(centres[..., None, :] - SEVEN, axis=-1).argmin(-1)
        elif model is normal:
            rankings, probs = normal.ranking_probabilities(SEVEN, pixels, deviation)
            winners = fptp(rankings, probs)
        else:
            medians = ranking_cells.node_medians(pixels)
            rankings, probs = ranking_cells.compute_ranking_probabilities(
                SEVEN, ranking_cells.beta_params_at(medians, deviation))
            winners = fptp(rankings, ranking_cells.interpolate_to_pixels(probs, medians, pixels))
        ax.imshow(winners.T, cmap=ListedColormap(SEVEN_PALETTE), vmin=0, vmax=6, origin="lower",
                  extent=(0, 1, 0, 1), interpolation="nearest")
        voronoi_lines(ax, SEVEN, colors="k", linewidths=0.6, linestyles="--")
        ax.scatter(*SEVEN.T, c=SEVEN_PALETTE, s=40, edgecolors="k", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "seven_fptp.png", dpi=150)
    plt.close(fig)


def sample_voters(model, pixel, n, rng):
    """n voters of the pixel with the given median (Beta) or mean (normal)."""
    if model == "normal":
        return rng.normal(pixel, normal.sigma_from_deviation(DEVIATION), (n, 2))
    return np.column_stack([beta_marginal(m).rvs(n, random_state=rng) for m in pixel])


def first_choice(points, candidates, among):
    among = np.asarray(among)
    return among[np.linalg.norm(points[:, None] - candidates[among], axis=-1).argmin(axis=1)]


def voters(ax, points, choice, palette, pixel, title):
    order = np.random.default_rng(0).permutation(len(points))  # no colour drawn on top
    ax.scatter(*points[order].T, c=np.asarray(palette)[choice[order]], s=1.5, linewidths=0)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=1))
    ax.scatter(*pixel, marker="+", s=150, c="k", zorder=4, linewidths=2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def voters_irv(profiles, n=6000):
    """Voters of pixels on the column x = 0.55, coloured by first choice among A, D, E."""
    rng = np.random.default_rng(1)
    i = int(0.55 * PIXELS)
    ys = (0.70, 0.80, 0.86, 0.98)
    t = np.linspace(-0.35, 1.35, 10)
    fig, axes = plt.subplots(2, len(ys), figsize=(3.1 * len(ys), 7.0))
    for row, name in enumerate(profiles):
        rankings, probs = profiles[name]
        shares = first_choice_shares(rankings, probs, (A, D, E))
        for ax, y in zip(axes[row], ys):
            j = min(int(y * PIXELS), PIXELS - 1)
            pixel = (MEDIANS[i], MEDIANS[j])
            points = sample_voters(name, pixel, n, rng)
            share = "  ".join(f"{NAMES[c]} {shares[c][i, j]:.3f}" for c in (A, D, E))
            voters(ax, points, first_choice(points, CANDIDATES, (A, D, E)), PALETTE, pixel,
                   f"{name} ({pixel[0]:.2f}, {pixel[1]:.2f})\n{share}")
            ax.plot(t, 0.425 + (t - 0.55) * 2 / 3, color="k", lw=0.6)
            ax.plot(t, t + 0.2, color="k", lw=0.6)
            ax.scatter(*CANDIDATES[[A, D, E]].T, c=[PALETTE[c] for c in (A, D, E)], s=40,
                       edgecolors="k", zorder=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "voters_irv.png", dpi=150)
    plt.close(fig)


def voters_wall(n=6000):
    """Voters of pixels approaching the left wall, coloured by first choice (FPTP)."""
    rng = np.random.default_rng(2)
    xs = (0.2, 0.08, 0.02)
    fig, axes = plt.subplots(2, len(xs), figsize=(3.1 * len(xs), 7.0))
    for row, name in enumerate(("beta", "normal")):
        for ax, x in zip(axes[row], xs):
            pixel = (x, 0.4)
            points = sample_voters(name, pixel, 200_000, rng)
            choice = first_choice(points, CANDIDATES, range(5))
            share = "  ".join(f"{NAMES[c]} {np.mean(choice == c):.2f}" for c in (B, C, E))
            voters(ax, points[:n], choice[:n], PALETTE, pixel,
                   f"{name} ({x}, 0.4)\n{share}")
            voronoi_lines(ax, CANDIDATES, colors="k", linewidths=0.6)
            ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=40, edgecolors="k", zorder=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "voters_wall.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    profiles = {name: profile(name) for name in MODELS}
    beta_shapes()
    pull_example()
    compare(profiles)
    irv_round(profiles)
    fptp_edge(profiles)
    blur_corner()
    seven_fptp()
    voters_irv(profiles)
    voters_wall()
