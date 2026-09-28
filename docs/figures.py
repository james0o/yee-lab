"""Figures and numbers of math.typ.

Run from the repository root: `uv run python docs/figures.py`.
Writes docs/figures/*.png and prints the numbers quoted in the text. Beta voters use
the default spread rule (ranking_cells.SPREAD) except where the rules are compared.
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
from methods import CYCLE, condorcet_cycle, fptp, irv, schulze, voronoi

# candidates A-E of the document
CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
PIXELS = 300
DEVIATION = 0.3
SPREAD = ranking_cells.SPREAD
# the spread rules in the order of the document: default first, legacy last
RULES = ("rms", "tapered", "mean_abs")
assert set(RULES) == set(ranking_cells.SPREADS) and RULES[0] == SPREAD
# one line style per rule as well as a colour, so a rule is never told by colour alone
RULE_STYLE = {
    "rms": {"color": "#4e79a7", "linestyle": "-"},
    "tapered": {"color": "#59a14f", "linestyle": "--"},
    "mean_abs": {"color": "#e15759", "linestyle": ":"},
}
NAMES = "ABCDE"
A, B, C, D, E = range(5)
# the palette of the cell figure in math.typ, black for Condorcet cycles
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#000000"]
FIGURES = Path(__file__).parent / "figures"
MEDIANS = ranking_cells.pixel_medians(PIXELS)
MODELS = {"beta": ranking_cells, "normal": normal}


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
        models = [("normal", normal, {})] + [(f"beta {s}", ranking_cells, {"spread": s}) for s in RULES]
        for label, model, options in models:
            rankings, probs = model.ranking_probabilities(CANDIDATES, PIXELS, deviation, **options)
            sch, cyc = schulze(rankings, probs), condorcet_cycle(rankings, probs)
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
                ("Condorcet winner", condorcet_cycle(rankings, probs))]):
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
    methods = [("IRV", irv), ("Schulze", schulze), ("Condorcet winner", condorcet_cycle)]
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
                   ranking_cells.ranking_probabilities(CANDIDATES, PIXELS, 0.2, spread=SPREAD)))
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
    """Borders of the Voronoi cells (methods.voronoi), drawn as contours."""
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
    panels = [("Voronoi (no blur)", None, None, {}), ("normal, D = 0.1", normal, 0.1, {}),
              ("normal, D = 0.3", normal, 0.3, {}),
              (f"Beta ({SPREAD}), D = 0.3", ranking_cells, 0.3, {"spread": SPREAD})]
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

if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
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
