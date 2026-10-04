"""Figures and numbers of math.typ.

Run from the repository root: `uv run python docs/figures.py`.
Writes docs/figures/*.png and prints the numbers quoted in the text. Beta voters use
the default spread rule (ranking_cells.SPREAD) except where the rules are compared.
`uv run python docs/figures.py search` runs only the slow search over random candidate
layouts of the chapter on the shapes of win regions.
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.special import betainc, ndtr, ndtri
from scipy.stats import beta as beta_dist

from yeelab import normal, ranking_cells
from yeelab.approval import AVG, GAP, coverage
from yeelab.build import AvgApproval, GapApproval, Highest, Tally, Voters
from yeelab.margin.geometric import geometric_median_at, geometric_medians
from yeelab.margin.regions import MARGINS, regions, winners
from yeelab.margin.shares import Model, pairwise_shares
from yeelab.pixels import beta as pixel_beta, normal as pixel_normal
from yeelab.pixels.methods import _pairwise_preferences, borda, condorcet, fptp, irv, schulze, voronoi
from yeelab.voting import CYCLE
from yeelab.web.app import DEVIATION as UI_DEVIATION, DRAG_GRID, FINAL_GRID

# candidates A-E of the document
CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
PIXELS = 300
DEVIATION = 0.3
SPREAD = ranking_cells.SPREAD
# the spread rules in the order of the document: default first, legacy last
RULES = ("rms", "mean_abs")
assert set(RULES) == set(ranking_cells.SPREADS) and RULES[0] == SPREAD
# one line style per rule as well as a colour, so a rule is never told by colour alone
RULE_STYLE = {
    "rms": {"color": "#4e79a7", "linestyle": "-"},
    "mean_abs": {"color": "#e15759", "linestyle": ":"},
}
NAMES = "ABCDE"
A, B, C, D, E = range(5)
# the palette of the cell figure in math.typ, black for Condorcet cycles
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#000000"]
FIGURES = Path(__file__).parent / "figures"
MEDIANS = ranking_cells.pixel_medians(PIXELS)
MODELS = {"beta": pixel_beta, "normal": pixel_normal}


def profile(name, spread=SPREAD):
    model = MODELS[name]
    options = {"spread": spread} if name == "beta" else {}
    cached = model.read_cached_ranking_probabilities(CANDIDATES, PIXELS, DEVIATION, **options)
    if cached is None:
        cached = model.generate_ranking_probabilities(CANDIDATES, PIXELS, DEVIATION, **options)
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


def beta_marginal(median, spread=SPREAD):
    """Frozen Beta distribution of one coordinate with the given median."""
    (a, b), = ranking_cells.beta_params_at([median], DEVIATION, spread)
    return beta_dist(a, b)


def _ties():
    """Pixels whose centre is equidistant (to rounding) from its two nearest candidates."""
    centres = np.stack(np.meshgrid(MEDIANS, MEDIANS, indexing="ij"), axis=-1)
    dist = np.sort(np.linalg.norm(centres[..., None, :] - CANDIDATES, axis=-1), axis=-1)
    return dist[..., 1] - dist[..., 0] < 1e-12

# ---------------------------------------------------------------- spread rules

def spread_shapes():
    """Beta marginals towards a wall under each spread rule."""
    a0 = ranking_cells.centre_shape(DEVIATION)
    print(f"spread rules, deviation {DEVIATION}: a0 {a0:.3f}, "
          f"centre RMS {0.5 / np.sqrt(2 * a0 + 1):.3f}, P(X<0.1) {betainc(a0, a0, 0.1):.3f}")
    for spread in RULES:
        for m in (0.5, 0.7, 0.9, 0.98):
            dist = beta_marginal(m, spread)
            a, b = dist.args
            mean_abs = a / (a + b) * (1 - 2 * betainc(a + 1, b, m))  # E|X - m|, median m
            rms = np.sqrt(dist.var() + (dist.mean() - m) ** 2)
            print(f"  {spread:9s} median {m}: a {a:.3f} b {b:.3f} a+b {a + b:.3f} "
                  f"mean {dist.mean():.3f} E|X-m| {mean_abs:.3f} RMS {rms:.3f} "
                  f"P(X<0.1) {dist.cdf(0.1):.3f} P(X>0.9) {dist.sf(0.9):.3f}")


def spread_densities():
    """Density of one coordinate of a pixel's voters under each rule, medians towards a wall."""
    medians = (0.5, 0.7, 0.9, 0.98)
    x = np.linspace(0.0005, 0.9995, 2000)
    fig, axes = plt.subplots(1, len(medians), figsize=(12, 3.1), sharey=True)
    for ax, m in zip(axes, medians):
        ax.axvline(m, color="0.6", lw=0.8)
        for spread in RULES:
            ax.plot(x, beta_marginal(m, spread).pdf(x), lw=1.8, label=spread, **RULE_STYLE[spread])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3)
        ax.set_title(f"median {m}", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("density", fontsize=9)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "spread_densities.png", dpi=150)
    plt.close(fig)


def cycle_counts():
    """Share of pixels (without ties) with a cycle or with Schulze away from Voronoi."""
    tie, nearest = _ties(), voronoi(CANDIDATES, PIXELS)
    print(f"pixels on a bisector (ties): {tie.sum()} of {tie.size}")
    for deviation in (0.2, DEVIATION):
        models = [("normal", pixel_normal, {})] + [(f"beta {s}", pixel_beta, {"spread": s}) for s in RULES]
        for label, model, options in models:
            rankings, probs = model.ranking_probabilities(CANDIDATES, PIXELS, deviation, **options)
            sch, cyc = schulze(rankings, probs), condorcet(rankings, probs)
            print(f"  deviation {deviation} {label:14s}: cycles {np.sum((cyc == CYCLE) & ~tie):5d} "
                  f"({np.sum((cyc == CYCLE) & ~tie) / np.sum(~tie):.2%}), Schulze != Voronoi "
                  f"{np.sum((sch != nearest) & ~tie):5d} ({np.sum((sch != nearest) & ~tie) / np.sum(~tie):.1%}), "
                  f"area of D {np.mean(sch == D):.3f} (Voronoi {np.mean(nearest == D):.3f})")


def spread_rules():
    """FPTP, IRV and Condorcet winner for each spread rule, default candidates."""
    fig, axes = plt.subplots(len(RULES), 3, figsize=(10, 3.4 * len(RULES)))
    for row, spread in enumerate(RULES):
        rankings, probs = profile("beta", spread)
        for col, (label, winners) in enumerate([
                ("FPTP", fptp(rankings, probs)), ("IRV", irv(rankings, probs)),
                ("Condorcet winner", condorcet(rankings, probs))]):
            show(axes[row, col], winners, f"{spread}: {label}")
    fig.tight_layout()
    fig.savefig(FIGURES / "spread_rules.png", dpi=130)
    plt.close(fig)


def irv_round():
    """Beta IRV with the round 3 ties among A, D, E, under the legacy and the default rule."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    x, y = np.meshgrid(MEDIANS, MEDIANS, indexing="ij")
    t = np.linspace(0, 1, 50)
    i = int(0.55 * PIXELS)
    for ax, spread in zip(axes, ("mean_abs", SPREAD)):
        rankings, probs = profile("beta", spread)
        shares = first_choice_shares(rankings, probs, (A, D, E))
        winners = irv(rankings, probs)
        show(ax, winners, f"{spread}: IRV and the round 3 ties among A, D, E", alpha=0.55)
        for (p, q), color, style in [((A, D), "red", "-"), ((D, E), "black", "--"), ((A, E), "purple", ":")]:
            ax.contour(x, y, shares[p] - shares[q], levels=[0], colors=color, linestyles=style, linewidths=2)
            ax.plot([], [], color=color, linestyle=style, label=f"{NAMES[p]} = {NAMES[q]}")
        ax.plot(t, 0.425 + (t - 0.55) * 2 / 3, color="gray", lw=1, label="bisectors D|A, D|E")
        ax.plot(t, t + 0.2, color="gray", lw=1)
        ax.legend(loc="lower right", fontsize=8)

        print(f"{spread}: column x = {MEDIANS[i]:.3f}, round 3 shares among A, D, E")
        for yy in np.arange(0.62, 1.0, 0.06):
            j = min(int(yy * PIXELS), PIXELS - 1)
            print(f"  y = {MEDIANS[j]:.3f}: winner {NAMES[winners[i, j]]}  "
                  + "  ".join(f"{NAMES[c]} {shares[c][i, j]:.3f}" for c in (A, D, E))
                  + f"  P(Y<0.2) = {beta_marginal(MEDIANS[j], spread).cdf(0.2):.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "irv_round.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- Beta versus normal

def beta_shapes():
    print(f"Beta marginals ({SPREAD}), deviation {DEVIATION}")
    for m in (0.5, 0.7, 0.9, 0.98):
        dist = beta_marginal(m)
        a, b = dist.args
        print(f"  median {m}: a {a:.3f} b {b:.3f} mean {dist.mean():.3f} "
              f"P(X<0.1) {dist.cdf(0.1):.3f} P(X>0.9) {dist.sf(0.9):.3f}")


def compare(profiles):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.9))
    methods = [("IRV", irv), ("Schulze", schulze), ("Condorcet winner", condorcet)]
    for row, name in enumerate(profiles):
        rankings, probs = profiles[name]
        for col, (label, method) in enumerate(methods):
            show(axes[row, col], method(rankings, probs), f"{name}: {label}")
    fig.tight_layout()
    fig.savefig(FIGURES / "compare.png", dpi=150)
    plt.close(fig)


def pull_example(samples=4_000_000, seed=0):
    """Share preferring A to D at pixels on A's side of their bisector."""
    rng = np.random.default_rng(seed)
    n = CANDIDATES[A] - CANDIDATES[D]
    offset = (CANDIDATES[A] @ CANDIDATES[A] - CANDIDATES[D] @ CANDIDATES[D]) / 2
    sigma = normal.sigma_from_deviation(DEVIATION)
    print(f"pull towards the centre ({SPREAD})")
    for pixel in [(0.95, 0.55), (0.9, 0.6), (0.8, 0.45)]:
        bx, by = beta_marginal(pixel[0]), beta_marginal(pixel[1])
        x, y = bx.rvs(samples, random_state=rng), by.rvs(samples, random_state=rng)
        share = np.mean(n[0] * x + n[1] * y > offset)
        dist = (n @ pixel - offset) / np.linalg.norm(n)
        print(f"  pixel {pixel}: distance {dist:+.3f}, Beta share {share:.3f}, "
              f"normal share {ndtr(dist / sigma):.3f}, Beta mean ({bx.mean():.3f}, {by.mean():.3f})")


def fptp_edge(profiles):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, name in zip(axes, profiles):
        rankings, probs = profiles[name]
        show(ax, fptp(rankings, probs), f"{name}: FPTP")
    fig.tight_layout()
    fig.savefig(FIGURES / "fptp_edge.png", dpi=150)
    plt.close(fig)

    models = [("normal", profiles["normal"])] + [(f"beta {s}", profile("beta", s)) for s in RULES]
    models.append((f"beta {SPREAD}, deviation 0.2",
                   pixel_beta.ranking_probabilities(CANDIDATES, PIXELS, 0.2, spread=SPREAD)))
    for label, (rankings, probs) in models:
        winners = fptp(rankings, probs)
        print(f"{label}: FPTP borders of B near the left edge")
        for x in (0.2, 0.08, 0.03, 0.005):
            col = winners[int(x * PIXELS)]
            be = [round(MEDIANS[j], 3) for j in range(PIXELS - 1) if {col[j], col[j + 1]} == {B, E}]
            cb = [round(MEDIANS[j], 3) for j in range(PIXELS - 1) if {col[j], col[j + 1]} == {C, B}]
            extra = ""
            if label.startswith("beta") and "deviation" not in label:
                extra = f"  P(X<0.02) = {beta_marginal(x, label.split()[1]).cdf(0.02):.3f}"
            print(f"  x = {x}: B|E at y {be}, C|B at y {cb}{extra}")

# ---------------------------------------------------------------- blurred cells

# the seven candidates of the FPTP example (B removed, three added on the left)
SEVEN = np.array([(0.6, 0.35), (0.35, 0.3), (0.5, 0.5), (0.3, 0.7),
                  (0.15, 0.31), (0.07, 0.45), (0.04, 0.62)])
SEVEN_PALETTE = ["#4e79a7", "#76b7b2", "#e15759", "#b07aa1", "#59a14f", "#edc948", "#f28e2b"]


def voronoi_lines(ax, candidates, **style):
    """Borders of the Voronoi cells (pixels.methods.voronoi), drawn as contours."""
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
    rankings, probs = pixel_normal.ranking_probabilities(three, pixels, deviation, 0)
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
    panels = [("Voronoi (no blur)", None, None, {}), ("normal, D = 0.1", pixel_normal, 0.1, {}),
              ("normal, D = 0.3", pixel_normal, 0.3, {}),
              (f"Beta ({SPREAD}), D = 0.3", pixel_beta, 0.3, {"spread": SPREAD})]
    for ax, (title, model, deviation, options) in zip(axes, panels):
        if model is None:
            winners = voronoi(SEVEN, pixels)
        else:
            rankings, probs = model.ranking_probabilities(SEVEN, pixels, deviation, **options)
            winners = fptp(rankings, probs)
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


def voters_wall(n=6000):
    """Voters of pixels approaching the left wall, coloured by first choice (FPTP)."""
    rng = np.random.default_rng(2)
    xs = (0.2, 0.08, 0.02)
    fig, axes = plt.subplots(2, len(xs), figsize=(3.1 * len(xs), 7.0))
    for row, name in enumerate(("beta", "normal")):
        for ax, x in zip(axes[row], xs):
            pixel = (x, 0.4)
            points = sample_voters(name, pixel, 200_000, rng)
            choice = np.linalg.norm(points[:, None] - CANDIDATES, axis=-1).argmin(axis=1)
            share = "  ".join(f"{NAMES[c]} {np.mean(choice == c):.2f}" for c in (B, C, E))
            voters(ax, points[:n], choice[:n], PALETTE, pixel, f"{name} ({x}, 0.4)\n{share}")
            voronoi_lines(ax, CANDIDATES, colors="k", linewidths=0.6)
            ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=40, edgecolors="k", zorder=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "voters_wall.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- shapes of win regions

EIGHT = np.ones((3, 3), bool)  # 8-neighbour pixels count as connected


def pieces(mask, minsize=1):
    """Sizes of the 8-connected pieces of a pixel mask, largest first."""
    labels, n = ndimage.label(mask, structure=EIGHT)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return sorted((int(s) for s in sizes if s >= minsize), reverse=True), labels


def concave_pixels(mask, margin=3):
    """Pixels inside the convex hull of the region, not in it, and more than `margin`
    pixels away from it. For a convex region pixelation leaves only pixels within about
    one pixel of it, so any such pixel shows that the region is not convex."""
    points = np.argwhere(mask)
    if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        return 0
    hull = MplPath(points[ConvexHull(points).vertices])
    inside = hull.contains_points(np.argwhere(np.ones_like(mask))).reshape(mask.shape)
    return int(np.sum(inside & (ndimage.distance_transform_edt(~mask) > margin)))


def spans(labels, k, medians):
    where = np.argwhere(labels == k)
    lo, hi = medians[where.min(axis=0)], medians[where.max(axis=0)]
    return f"x {lo[0]:.3f}-{hi[0]:.3f}, y {lo[1]:.3f}-{hi[1]:.3f}"


# three candidates on a line: IRV squeezes the middle one out
LINE = np.array([(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)])
LINE_NAMES = "LMR"
LINE_PALETTE = ["#4e79a7", "#e15759", "#59a14f"]


def collinear():
    """FPTP and IRV for three collinear candidates, normal voters (centre squeeze)."""
    deviation, pixels = 0.3, 300
    rankings, probs = pixel_normal.ranking_probabilities(LINE, pixels, deviation, 0)
    m = ranking_cells.pixel_medians(pixels)
    row = pixels // 2
    shares = {c: probs[:, row, rankings[:, 0] == c].sum(-1) for c in range(3)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for ax, (label, method) in zip(axes, (("FPTP", fptp), ("IRV", irv))):
        winners = method(rankings, probs)
        ax.imshow(winners.T, cmap=ListedColormap(LINE_PALETTE), vmin=0, vmax=2, origin="lower",
                  extent=(0, 1, 0, 1), interpolation="nearest")
        voronoi_lines(ax, LINE, colors="k", linewidths=0.8, linestyles="--")
        ax.scatter(*LINE.T, c=LINE_PALETTE, s=45, edgecolors="k", zorder=3)
        for name, (x, y) in zip(LINE_NAMES, LINE):
            ax.annotate(name, (x + 0.015, y + 0.03), weight="bold", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"normal {label}, D = {deviation} (dashed: Voronoi)", fontsize=10)
        line = winners[:, row]
        runs = [(m[i], m[j - 1], LINE_NAMES[line[i]]) for i, j in
                zip(np.r_[0, np.flatnonzero(np.diff(line)) + 1], np.r_[np.flatnonzero(np.diff(line)) + 1, pixels])]
        print(f"collinear {label}: " + ", ".join(f"{w} on x {a:.3f}-{b:.3f}" for a, b, w in runs))
    ax = axes[2]
    for c, style in zip(range(3), ("-", "--", ":")):
        ax.plot(m, shares[c], color=LINE_PALETTE[c], linestyle=style, lw=2, label=f"$s_{LINE_NAMES[c]}$")
    squeezed = irv(rankings, probs)[:, row] == 1
    ax.fill_between(m, 0, 1, where=squeezed, color=LINE_PALETTE[1], alpha=0.12, lw=0,
                    label="M wins IRV")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("pixel median x (any y)", fontsize=9)
    ax.set_title("first-choice shares", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "collinear.png", dpi=150)
    plt.close(fig)
    for x in (0.4, 0.5):
        i = np.argmin(abs(m - x))
        print(f"  x = {m[i]:.3f}: " + "  ".join(f"{LINE_NAMES[c]} {shares[c][i]:.3f}" for c in range(3)))


# A in a thin strip between N and S; NW, NE, SW, SE take the voters far above and below
SPLIT = np.array([(0.5, 0.5), (0.5, 0.6), (0.5, 0.4), (0.3, 0.95), (0.7, 0.95), (0.3, 0.05), (0.7, 0.05)])
SPLIT_NAMES = ["A", "N", "S", "NW", "NE", "SW", "SE"]
SPLIT_PALETTE = ["#e15759", "#4e79a7", "#59a14f", "#9cb9d6", "#76b7b2", "#b3d6ad", "#edc948"]


def disconnected_fptp():
    """FPTP region of A in two pieces, normal and Beta voters, D = 0.12."""
    deviation, pixels = 0.12, 401  # odd: the middle pixel column is x = 1/2
    m = ranking_cells.pixel_medians(pixels)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, (name, model, options) in zip(axes, (("normal", pixel_normal, {}),
                                                 (f"Beta ({SPREAD})", pixel_beta, {"spread": SPREAD}))):
        nodes = 0 if model is pixel_normal else ranking_cells.NODES
        rankings, probs = model.ranking_probabilities(SPLIT, pixels, deviation, nodes, **options)
        winners = fptp(rankings, probs)
        sizes, labels = pieces(winners == 0)
        print(f"split {name}: FPTP region of A in {len(sizes)} pieces {sizes}: "
              + "; ".join(spans(labels, k, m) for k in range(1, len(sizes) + 1)))
        # the whole pixel column x = 1/2
        i = pixels // 2
        column = np.stack([probs[i][:, rankings[:, 0] == c].sum(-1) for c in range(len(SPLIT))])
        lead = column[0] - column[1:].max(axis=0)
        print(f"  column x = {m[i]}: A's largest lead over the best rival {lead.max():+.4f} "
              f"(y = {m[lead.argmax()]:.3f}); at A's own position A {column[0, i]:.3f}, "
              f"best rival {SPLIT_NAMES[1 + column[1:, i].argmax()]} {column[1:, i].max():.3f}")
        ax.imshow(winners.T, cmap=ListedColormap(SPLIT_PALETTE), vmin=0, vmax=6, origin="lower",
                  extent=(0, 1, 0, 1), interpolation="nearest")
        voronoi_lines(ax, SPLIT, colors="k", linewidths=0.6, linestyles="--")
        ax.axvline(0.5, color="k", lw=0.8, ls=":")
        ax.scatter(*SPLIT.T, c=SPLIT_PALETTE, s=40, edgecolors="k", zorder=3)
        for label, (x, y) in zip(SPLIT_NAMES, SPLIT):
            ax.annotate(label, (x + 0.015, y + 0.015), weight="bold", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name} FPTP, D = {deviation}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "disconnected_fptp.png", dpi=150)
    plt.close(fig)


# c_1, ..., c_4 of the Schulze example
NOTCH = np.array([(0.457, 0.407), (0.254, 0.289), (0.334, 0.641), (0.235, 0.464)])
NOTCH_SEGMENT = np.array([(0.351, 0.519), (0.314, 0.342)])


def schulze_notch():
    """Beta Schulze region of c_1 that is not convex, and the cycle pockets behind it."""
    deviation, pixels = 0.3, PIXELS
    rankings, probs = pixel_beta.ranking_probabilities(NOTCH, pixels, deviation, spread=SPREAD)
    sch, cyc = schulze(rankings, probs), condorcet(rankings, probs)
    pairwise = _pairwise_preferences(rankings, probs)
    print(f"notch: concave pixels of c_1's Schulze region {concave_pixels(sch == 0)}, "
          f"cycle pixels {np.sum(cyc == CYCLE)}")
    # exact values along the segment
    t = np.linspace(0, 1, 11)
    points = NOTCH_SEGMENT[0] + t[:, None] * (NOTCH_SEGMENT[1] - NOTCH_SEGMENT[0])
    medians = np.unique(points.round(6))
    r, p = pixel_beta.compute_ranking_probabilities(
        NOTCH, ranking_cells.beta_params_at(medians, deviation, SPREAD))
    s, c, d = schulze(r, p), condorcet(r, p), _pairwise_preferences(r, p)
    for tt, (x, y) in zip(t, points):
        i, j = np.searchsorted(medians, round(x, 6)), np.searchsorted(medians, round(y, 6))
        dd = d[i, j]
        defeats = ", ".join(f"c_{u + 1}>c_{v + 1} {dd[u, v]:.3f}" for u in range(4) for v in range(4)
                            if dd[u, v] > 0.5)
        print(f"  t = {tt:.1f} ({x:.3f}, {y:.3f}): Schulze c_{s[i, j] + 1}, Condorcet "
              f"{'cycle' if c[i, j] == CYCLE else f'c_{c[i, j] + 1}'}; {defeats}")

    x, y = np.meshgrid(MEDIANS, MEDIANS, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    for ax, winners, title in ((axes[0], sch, f"Beta ({SPREAD}) Schulze, D = {deviation}"),
                               (axes[1], cyc, "Condorcet winner (black: cycle), detail")):
        ax.imshow(np.where(winners == CYCLE, 5, winners).T, cmap=ListedColormap(PALETTE), vmin=0, vmax=5,
                  origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
        ax.scatter(*NOTCH.T, c=PALETTE[:4], s=45, edgecolors="k", zorder=3)
        for k, (cx, cy) in enumerate(NOTCH):
            ax.annotate(f"$c_{k + 1}$", (cx + 0.012, cy + 0.012), weight="bold", fontsize=10)
        ax.set_title(title, fontsize=10)
    axes[0].plot(*NOTCH_SEGMENT.T, color="k", lw=1.5, marker="o", ms=4)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    ax = axes[1]
    ax.contour(x, y, (sch == 0).astype(float), levels=[0.5], colors="white", linewidths=2)
    ax.contour(x, y, pairwise[..., 0, 3] - 0.5, levels=[0], colors="#edc948", linewidths=1.5, linestyles="--")
    ax.scatter(*NOTCH_SEGMENT.T, c="white", edgecolors="k", s=30, zorder=4)
    ax.plot([], [], color="white", lw=2, label="border of $c_1$'s Schulze region")
    ax.plot([], [], color="#edc948", lw=1.5, ls="--", label="$c_1$ and $c_4$ tie head to head")
    ax.scatter([], [], c="white", edgecolors="k", s=30, label="ends of the segment")
    ax.legend(fontsize=8, loc="lower right", facecolor="0.85")
    ax.set_xlim(0.2, 0.45)
    ax.set_ylim(0.28, 0.58)
    fig.tight_layout()
    fig.savefig(FIGURES / "schulze_notch.png", dpi=150)
    plt.close(fig)


def stochastic_order():
    """Is P(X <= t) non-increasing in the median for every t? Largest violation for medians
    that a 400-pixel grid reaches."""
    m = np.linspace(0.5, 1 - 1 / 800, 8000)
    t = np.linspace(0.001, 0.999, 999)
    for spread in RULES:
        for deviation in (0.2, 0.3):
            ab = ranking_cells.beta_params_at(m, deviation, spread)
            F = betainc(ab[:, :1], ab[:, 1:], t)
            later_max = np.maximum.accumulate(F[::-1], axis=0)[::-1]   # max of F over larger medians
            gain = later_max - F
            start = m[1:][(np.diff(F, axis=0) > 1e-12).any(axis=1)].min()  # first local rise
            print(f"stochastic order, {spread:8s} D = {deviation}: P(X <= t) grows with the median "
                  f"from median {start:.3f} on, by at most {gain.max():.3f}")

# ---------------------------------------------------------------- web UI: margins and polygons

# the web UI's defaults: Beta voters, the default rule, its deviation
UI_MODEL = Model("beta", UI_DEVIATION, SPREAD)
# the medians of the margin examples: at candidate D, and in a pocket of Condorcet cycles
MARGIN_POINTS = {"m_1": (0.5, 0.5), "m_2": (0.32, 0.545)}
# the part of the square around B's holes in the polygon example
HOLE_WINDOW = ((0.225, 0.365), (0.365, 0.49))


def _signed_area(ring):
    x, y = np.reshape(ring, (-1, 2)).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def margin_example():
    """Shares, winner and margin of every method (margin/regions.py MARGINS) at
    MARGIN_POINTS."""
    for label, point in MARGIN_POINTS.items():
        rankings, probs = pixel_beta.compute_ranking_probabilities(
            CANDIDATES, ranking_cells.beta_params_at(np.array(point), UI_DEVIATION, SPREAD))
        probs = probs[0, 1]  # the median (point[0], point[1])
        first = np.array([probs[rankings[:, 0] == c].sum() for c in range(5)])
        d = _pairwise_preferences(rankings, probs[None, None])[0, 0]
        print(f"margins at {label} = {point} (Beta {SPREAD}, D = {UI_DEVIATION})")
        print("  first choices " + "  ".join(f"{n} {s:.4f}" for n, s in zip(NAMES, first)))
        print("  Borda scores  " + "  ".join(f"{n} {s:.4f}" for n, s in zip(NAMES, d.sum(-1))))
        for i in range(5):
            print(f"  pi_{NAMES[i]}. " + "  ".join(f"{d[i, j]:.4f}" for j in range(5)))
        # IRV round by round, as voting.irv_rounds
        alive = list(range(5))
        while len(alive) > 1:
            tally = first_choice_shares(rankings, probs, alive)
            low, second = sorted(alive, key=lambda c: tally[c])[:2]
            print("  IRV round " + "  ".join(f"{NAMES[c]} {tally[c]:.4f}" for c in alive)
                  + f": {NAMES[low]} out, gap to {NAMES[second]} {tally[second] - tally[low]:.4f}")
            alive.remove(low)
        # Baldwin and Nanson round by round: Borda scores among the remaining, sums of d
        for rule, below_mean in (("Baldwin", False), ("Nanson", True)):
            alive = list(range(5))
            while len(alive) > 1:
                score = {c: sum(d[c, e] for e in alive if e != c) for c in alive}
                mean = sum(score.values()) / len(alive)
                out = [c for c in alive if score[c] <= mean] if below_mean else [min(alive, key=score.get)]
                gap = (min(abs(s - mean) for s in score.values()) if below_mean
                       else sorted(score.values())[1] - score[out[0]])
                print(f"  {rule} round " + "  ".join(f"{NAMES[c]} {score[c]:.4f}" for c in alive)
                      + (f" (mean {mean:.4f})" if below_mean else "")
                      + f": {', '.join(NAMES[c] for c in out)} out, gap {gap:.4f}")
                alive = [c for c in alive if c not in out] or alive[:1]
        lead = d - d.T
        print("  narrowest head-to-head result " + "  ".join(
            f"{NAMES[i]} {np.delete(lead[i], i).min():+.4f}" for i in range(5)))
        # Schulze path strengths from the margins d - d^T (build.blocks.StrongestPaths)
        p = np.maximum(lead, 0.0)
        for k in range(5):
            p = np.maximum(p, np.minimum(p[:, k, None], p[None, k, :]))
        print("  Schulze links " + "  ".join(f"{NAMES[i]}>{NAMES[j]} {lead[i, j]:.4f}"
                                             for i in range(5) for j in range(5) if lead[i, j] > 0))
        print("  Schulze paths " + "  ".join(f"{NAMES[i]}>{NAMES[j]} {p[i, j]:.4f}"
                                             for i in range(5) for j in range(5) if i != j and p[i, j] > 0))
        print("  beaten by (max_f p_fe - p_ef) " + "  ".join(
            f"{NAMES[e]} {(p[:, e] - p[e, :]).max():.4f}" for e in range(5)))
        voters = Voters(first[None], d[None], rankings, probs[None])
        for name, method in (("FPTP", "fptp"), ("Borda", "borda"), ("Baldwin", "baldwin"),
                             ("Nanson", "nanson"), ("Condorcet", "condorcet"),
                             ("Schulze", "schulze"), ("Minimax", "minimax"), ("Black", "black"),
                             ("IRV", "irv")):
            winner, margin = MARGINS[method].evaluate(voters)
            print(f"  {name:9s} winner {'cycle' if winner[0] == CYCLE else NAMES[winner[0]]}, "
                  f"margin {margin[0]:.4f}")


def margin_fields():
    """The margin of FPTP and IRV over the square, with the borders of the regions."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), layout="constrained")
    norm = LogNorm(1e-4, 0.3, clip=True)
    for ax, (label, method) in zip(axes, (("FPTP", "fptp"), ("IRV", "irv"))):
        coords, _, margin = winners(method, CANDIDATES, UI_MODEL, FINAL_GRID)
        image = ax.pcolormesh(coords, coords, margin.T, norm=norm, cmap="gray", shading="nearest",
                              rasterized=True)
        for region in regions(method, CANDIDATES, UI_MODEL, FINAL_GRID):
            for polygon in region["polygons"]:
                for ring in polygon:
                    ax.plot(*np.reshape(ring, (-1, 2)).T, color="#edc948", lw=0.9)
        ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=45, edgecolors="k", zorder=3)
        for name, (x, y) in zip(NAMES, CANDIDATES):
            ax.annotate(name, (x + 0.015, y + 0.015), weight="bold", fontsize=9, color="#edc948")
        for (name, (x, y)), offset in zip(MARGIN_POINTS.items(), ((0.02, -0.065), (-0.085, 0.02))):
            ax.scatter(x, y, marker="+", color="#e15759", s=90, lw=1.5, zorder=4)
            ax.annotate(f"${name}$", (x + offset[0], y + offset[1]), color="#e15759", fontsize=11,
                        weight="bold", zorder=4)
        ax.plot([], [], color="#edc948", lw=0.9, label="border between winners")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{label}: margin $\\mu$, Beta ({SPREAD}), D = {UI_DEVIATION}", fontsize=10)
    fig.colorbar(image, ax=axes, shrink=0.85, label="$\\mu$ (log scale)")
    fig.savefig(FIGURES / "margins.png", dpi=150)
    plt.close(fig)


def _draw_regions(ax, shapes):
    """Fill every polygon even-odd, as the UI does; outer rings solid, holes dashed on top."""
    for region in shapes:
        for polygon in region["polygons"]:
            rings = [np.reshape(ring, (-1, 2)) for ring in polygon]
            path = MplPath.make_compound_path(*[MplPath(ring, closed=True) for ring in rings])
            ax.add_patch(PathPatch(path, facecolor=PALETTE[region["winner"]], alpha=0.6, lw=0))
    for k in (0, 1):
        for region in shapes:
            for polygon in region["polygons"]:
                for ring in polygon[:1] if k == 0 else polygon[1:]:
                    ax.plot(*np.reshape(ring, (-1, 2)).T, color="k" if k == 0 else "#e15759",
                            lw=0.8 if k == 0 else 1.2, ls="-" if k == 0 else "--")


def polygon_example():
    """IRV regions as the web UI receives them while dragging: rings, orientation, holes."""
    coords, winner, margin = winners("irv", CANDIDATES, UI_MODEL, DRAG_GRID)
    shapes = regions("irv", CANDIDATES, UI_MODEL, DRAG_GRID)
    print(f"IRV polygons, Beta {SPREAD}, D = {UI_DEVIATION}, grid {DRAG_GRID}")
    total = 0.0
    for region in shapes:
        for polygon in region["polygons"]:
            areas = [_signed_area(ring) for ring in polygon]
            closed = all(ring[:2] == ring[-2:] for ring in polygon)
            total += sum(areas)
            print(f"  {NAMES[region['winner']]}: polygon of {len(polygon)} ring(s), vertices "
                  f"{[len(ring) // 2 - 1 for ring in polygon]}, signed areas "
                  + ", ".join(f"{a:+.6f}" for a in areas) + ("" if closed else " (not closed)"))
    print(f"  sum of all signed areas {total:.5f}; at grid {FINAL_GRID} " + "{:.5f}".format(sum(
        _signed_area(ring) for region in regions("irv", CANDIDATES, UI_MODEL, FINAL_GRID)
        for polygon in region["polygons"] for ring in polygon)))
    # B's holes are rings of C, reversed
    holes = [np.reshape(ring, (-1, 2)) for region in shapes if region["winner"] == B
             for polygon in region["polygons"] for ring in polygon[1:]]
    outers = [np.reshape(polygon[0], (-1, 2)) for region in shapes if region["winner"] == C
              for polygon in region["polygons"]]
    for hole in holes:
        match = [k for k, outer in enumerate(outers) if set(map(tuple, outer)) == set(map(tuple, hole))]
        print(f"  hole of B with {len(hole) - 1} vertices, lowest point {hole[hole[:, 1].argmin()]}, "
              f"= outer ring of C's polygon {match}")
    # one crossing on the border of B's larger hole: a grid edge along x from B to C
    (left, bottom), (right, top) = holes[0].min(axis=0), holes[0].max(axis=0)
    for j in np.flatnonzero((coords > bottom) & (coords < top)):
        i = np.flatnonzero((winner[:-1, j] == B) & (winner[1:, j] == C)
                           & (coords[:-1] > left - 0.01) & (coords[1:] < right + 0.01))
        if len(i):
            i = i[0]
            alpha, beta = margin[i, j], margin[i + 1, j]
            t = alpha / (alpha + beta)
            x = coords[i] + t * (coords[i + 1] - coords[i])
            print(f"  crossing at y = {coords[j]:.6f}: B wins at x = {coords[i]:.6f} with margin "
                  f"{alpha:.5f}, C at x = {coords[i + 1]:.6f} with margin {beta:.5f}; t = {t:.4f}, "
                  f"x = {x:.6f}, a vertex of the hole: {np.abs(holes[0] - (x, coords[j])).max(1).min() < 1e-6}")
            break

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    (x0, x1), (y0, y1) = HOLE_WINDOW
    for ax in axes:
        _draw_regions(ax, shapes)
        ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=45, edgecolors="k", zorder=3)
        for name, (x, y) in zip(NAMES, CANDIDATES):
            ax.annotate(name, (x + 0.006, y + 0.006), weight="bold", fontsize=9)
        ax.set_aspect("equal")
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1))
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"IRV, Beta ({SPREAD}), D = {UI_DEVIATION}: polygons on a "
                      f"{DRAG_GRID} grid", fontsize=10)
    ax = axes[1]
    inside = (coords >= x0) & (coords <= x1), (coords >= y0) & (coords <= y1)
    gx, gy = np.meshgrid(coords[inside[0]], coords[inside[1]], indexing="ij")
    ax.scatter(gx, gy, c=[PALETTE[w] for w in winner[np.ix_(*inside)].ravel()], s=5, zorder=2)
    for hole in holes:  # arrows along B's holes: clockwise
        step = max(len(hole) // 8, 1)
        for k in range(0, len(hole) - 1, step):
            ax.annotate("", hole[k + 1], hole[k], zorder=4,
                        arrowprops={"arrowstyle": "-|>", "color": "#e15759", "lw": 1.2})
    ax.plot([], [], color="k", lw=0.8, label="outer ring (counter-clockwise)")
    ax.plot([], [], color="#e15759", lw=1.2, ls="--", label="hole of B (clockwise)")
    ax.legend(fontsize=8, loc="upper left", facecolor="white", framealpha=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title("detail: grid points coloured by winner", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "polygons.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- pixels at the geometric median

OUTERMOST = MEDIANS[0]  # the median closest to a wall


def _border_distances(model):
    """How far the pairwise majority borders are from the bisectors, at every point where
    a border crosses a line of the pixel grid: (with pixels at the medians along the axes,
    with pixels at the geometric medians). A crossing is placed along the grid edge by
    linear interpolation, and along the moved edge for the geometric medians, as
    margin/regions.py does."""
    above = ranking_cells.interpolate_to(pairwise_shares(CANDIDATES, model), model.medians, MEDIANS,
                                         model.transform)
    drawn = [np.stack(np.meshgrid(MEDIANS, MEDIANS, indexing="ij"), axis=-1),
             geometric_medians(model, MEDIANS)]
    distances = [[], []]
    for c, e in combinations(range(len(CANDIDATES)), 2):
        lead = above[..., c, e].astype(np.float64) - 0.5
        direction = CANDIDATES[e] - CANDIDATES[c]
        offset = (CANDIDATES[e] @ CANDIDATES[e] - CANDIDATES[c] @ CANDIDATES[c]) / 2
        for axis in (0, 1):
            before, after = np.moveaxis(lead, axis, 0)[:-1], np.moveaxis(lead, axis, 0)[1:]
            crossing = np.sign(before) != np.sign(after)
            t = before[crossing] / (before[crossing] - after[crossing])
            for points, out in zip(drawn, distances):
                points = np.moveaxis(points, axis, 0)
                p = points[:-1][crossing] + t[:, None] * (points[1:] - points[:-1])[crossing]
                out.append(np.abs(p @ direction - offset) / np.linalg.norm(direction))
    return [np.concatenate(d) for d in distances]


def _fill_regions(ax, shapes):
    """Every polygon filled even-odd in the colour of its winner; black: no Condorcet winner."""
    for region in shapes:
        for polygon in region["polygons"]:
            rings = [np.reshape(ring, (-1, 2)) for ring in polygon]
            path = MplPath.make_compound_path(*[MplPath(ring, closed=True) for ring in rings])
            ax.add_patch(PathPatch(path, facecolor=PALETTE[region["winner"]], lw=0))


def geometric_median():
    """Pixels at the geometric median of their voters: where g moves a pixel, the strip it
    leaves along the walls, and the Condorcet borders against the bisectors."""
    print("geometric median g of the voters with medians (m, 1/2) and (m, m): its first coordinate")
    for deviation in (0.2, DEVIATION):
        model = Model("beta", deviation, SPREAD)
        for m in (0.5, 0.7, 0.9, 0.98, 1 - OUTERMOST):
            (a, b), = ranking_cells.beta_params_at([m], deviation, SPREAD)
            g = geometric_median_at(model, [[m, 0.5], [m, m]])[:, 0]
            print(f"  D = {deviation} median {m:.4f}: g {g[0]:.3f} (m, 1/2), {g[1]:.3f} (m, m); "
                  f"mean {a / (a + b):.3f}")

    print("strip along the walls: distance of g from the wall for the outermost medians, "
          "in the middle of a wall and at a corner; part of the square that is no geometric median")
    for spread in RULES:
        for deviation in (0.1, 0.2, DEVIATION, 0.4):
            model = Model("beta", deviation, spread)
            g = geometric_medians(model, MEDIANS)
            outline = np.concatenate([g[:, 0], g[-1, 1:], g[-2::-1, -1], g[0, -2:0:-1]])
            closer = geometric_median_at(model, [[1e-6, 0.5], [1e-6, 1e-6]])[:, 0]
            print(f"  {spread:8s} D = {deviation}: middle {g[0, PIXELS // 2, 0]:.3f}, corner {g[0, 0, 0]:.3f}, "
                  f"empty {1 - _signed_area(outline):.1%}; with a median of 1e-6: "
                  f"middle {closer[0]:.3f}, corner {closer[1]:.4f}")

    print("distance of the pairwise majority borders from the bisectors, mean (largest): "
          "pixels at the medians along the axes -> at the geometric medians")
    for spread in RULES:
        for deviation in (0.2, DEVIATION):
            at_median, at_geometric = _border_distances(Model("beta", deviation, spread))
            print(f"  {spread:8s} D = {deviation}: {at_median.mean():.4f} ({at_median.max():.3f}) -> "
                  f"{at_geometric.mean():.4f} ({at_geometric.max():.3f}), "
                  f"{at_geometric.mean() / at_median.mean():.0%} of it, at {len(at_median)} crossings")

    model = Model("beta", DEVIATION, SPREAD)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (pixel_median, title) in zip(axes, (("marginal", "at the medians along the axes"),
                                                ("geometric", "at the geometric medians"))):
        _fill_regions(ax, regions("condorcet", CANDIDATES, model, FINAL_GRID, pixel_median))
        voronoi_lines(ax, CANDIDATES, colors="white", linewidths=0.9)
        ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=45, edgecolors="k", linewidths=1, zorder=3)
        for name, (x, y) in zip(NAMES, CANDIDATES):
            ax.annotate(name, (x + 0.015, y + 0.015), weight="bold", fontsize=9)
        ax.set_title(f"Condorcet winner, pixels {title}", fontsize=10)
    ax = axes[2]
    lines = np.concatenate([[OUTERMOST], np.arange(1, 10) / 10, [1 - OUTERMOST]])
    along = np.linspace(OUTERMOST, 1 - OUTERMOST, 400)
    for k, m in enumerate(lines):
        style = {"color": "k", "lw": 1.2} if k in (0, len(lines) - 1) else {"color": "0.55", "lw": 0.7}
        fixed = np.full_like(along, m)
        ax.plot(*geometric_median_at(model, np.column_stack([fixed, along])).T, **style)
        ax.plot(*geometric_median_at(model, np.column_stack([along, fixed])).T, **style)
    ax.set_title("lines of equal $m_x$ and of equal $m_y$, moved by $g$", fontsize=10)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES / "geometric.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- approval voting

# the approval ballots: name, label, cut and the method that tallies them; with two levels
# they are the ballots of the web UI's score_dh and score_avg
APPROVALS = [("approval_gap", "largest gap", GAP, Highest(Tally(GapApproval()))),
             ("approval_avg", "mean distance", AVG, Highest(Tally(AvgApproval())))]


def approval():
    """The voters who approve D, and the diagrams, for the approval ballots."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
    lines = np.linspace(0, 1, 401)
    shade = LinearSegmentedColormap.from_list("approves", ["white", PALETTE[D]])
    for k, (name, label, cut, method) in enumerate(APPROVALS):
        cover = coverage(lines, lines, CANDIDATES, cut, 4)
        print(f"{name}: approved by the voters of this part of the square " + "  ".join(
            f"{name} {cover[c].mean():.3f}" for c, name in enumerate(NAMES))
            + f"; approved candidates per voter {cover.sum(axis=0).mean():.3f}")
        ax = axes[k]
        ax.imshow(cover[D].T, cmap=shade, vmin=0, vmax=1, origin="lower", extent=(0, 1, 0, 1), alpha=0.6)
        ax.scatter(*CANDIDATES.T, c=PALETTE[:5], s=45, edgecolors="k", linewidths=1, zorder=3)
        for name, (x, y) in zip(NAMES, CANDIDATES):
            ax.annotate(name, (x + 0.015, y + 0.015), weight="bold", fontsize=9)
        ax.set_title(f"voters who approve D: {label}", fontsize=10)
        winner = winners(method, CANDIDATES, UI_MODEL, FINAL_GRID)[1][1:-1, 1:-1]
        show(axes[2 + k], winner, f"{name}: {label}")
    compared = [("fptp", "fptp"), ("borda", "borda"), ("schulze", "schulze"),
                *[(name, method) for name, _, _, method in APPROVALS], ("score_range", "score_range")]
    for name, method in compared:
        winner = winners(method, CANDIDATES, UI_MODEL, FINAL_GRID)[1][1:-1, 1:-1]
        print(f"{name}: share of the square won " + "  ".join(
            f"{name} {(winner == c).mean():.3f}" for c, name in enumerate(NAMES)))
    fig.savefig(FIGURES / "approval.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    if sys.argv[1:] == ["search"]:
        shape_search()
        sys.exit()
    spread_shapes()
    spread_densities()
    cycle_counts()
    spread_rules()
    irv_round()
    profiles = {name: profile(name) for name in MODELS}
    beta_shapes()
    compare(profiles)
    pull_example()
    fptp_edge(profiles)
    blur_corner()
    seven_fptp()
    voters_wall()
    collinear()
    disconnected_fptp()
    schulze_notch()
    stochastic_order()
    margin_example()
    margin_fields()
    polygon_example()
    geometric_median()
    approval()
