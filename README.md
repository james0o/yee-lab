# YeeLab

A Yee diagram compares voting methods. Candidates are fixed points $c_1, \dots, c_C$ in the unit square $[0,1]^2$. The square is split into $n \times n$ pixels, and every pixel is its own election:

- the voters of the pixel are spread around the pixel centre $m$,
- every voter ranks the candidates by distance (closest first),
- the pixel is coloured by the winner of that election.

## What is different here

In the original Yee diagram the voters of a pixel are normal, $\mathcal{N}(m, \sigma^2 I)$, so some of them end up outside the square. Here each coordinate is Beta distributed instead, so every voter stays inside the square:

$$X \sim \text{Beta}(a, b), \qquad Y \sim \text{Beta}(a', b'), \qquad X, Y \text{ independent.}$$

$(a, b)$ are chosen per pixel so that

1. the pixel centre is the **median**: $F(m; a, b) = \tfrac12$,
2. the spread is the same for every pixel: $\sqrt{\mathbb{E}(X - m)^2}$ equals its value at the centre pixel, where $\mathbb{E}|X - \tfrac12| = D$.

$D$ is the `--deviation` of the plots and the *Deviation* slider of the UI (default $0.2$).

Why it matters: with normal voters every Condorcet method draws exactly the Voronoi diagram of the candidates. With Beta voters the median is a median only along the axes, so the skew of the distribution decides diagonal head-to-head races. Condorcet regions get pulled towards the centre and Condorcet cycles appear.

Voting methods: FPTP, IRV, Borda, Schulze, and `condorcet_cycle`, which marks pixels without a Condorcet winner in black. How the ranking probabilities are computed (exactly, without sampling voters) is described in [docs/math.pdf](docs/math.pdf).

## Running it

Requires [uv](https://docs.astral.sh/uv/).

### Plots

[yeelab/pixels/plot.py](src/yeelab/pixels/plot.py) saves one PNG per method into `src/yeelab/pixels/plots/`:

```sh
uv run python -m yeelab.pixels.plot                         # all methods, Beta voters
uv run python -m yeelab.pixels.plot -m irv -m schulze       # only some methods
uv run python -m yeelab.pixels.plot --distribution normal   # original Yee model
uv run python -m yeelab.pixels.plot -d 0.3 -p 800           # deviation 0.3, 800x800 pixels
uv run python -m yeelab.pixels.plot -h                      # all options
```

Ranking probabilities are cached in `src/yeelab/pixels/cache/`, so a second run with the same settings is fast (`--regenerate` ignores the cache). Candidates are set in [yeelab/pixels/plot.py](src/yeelab/pixels/plot.py).

### Web UI

```sh
uv run fastapi dev
```

Then open http://127.0.0.1:8000. [yeelab/web/app.py](src/yeelab/web/app.py) serves the page in [yeelab/web/ui/index.html](src/yeelab/web/ui/index.html) and computes the diagrams. The diagram is drawn as curves, not pixels: the backend returns the region of every winner as polygons (`POST /api/regions`). It computes only the shares the method needs, caches everything a drag does not change, runs the hot loops as compiled [numba](https://numba.pydata.org/) kernels, and traces each border as the zero set of the winner's margin. The details are in the last chapter of [docs/math.pdf](docs/math.pdf). The very first start compiles the kernels, which takes a few seconds; numba caches them afterwards.

- **Candidates:** drag one to move it, click empty space to add one (up to 8), right-click to remove one. The diagram follows the drag, typically within 5–40 ms; IRV with 8 candidates, which needs every ranking cell, within about 0.1–0.2 s.
- **Method:** Voronoi (no voters, the reference), FPTP, IRV, Borda, Schulze or Condorcet cycle.
- **Voters:** Beta, or normal for the original Yee model.
- **Deviation:** $D$ from $0$ to $0.4$. At $0$ every voter sits at their pixel, so every method draws the Voronoi diagram.
- **Hover** over the square to see the voters of that pixel: their 2D density over the square and their distribution along $x$ above it.

## Code

The code is the package `yeelab` in [src/yeelab/](src/yeelab/). It computes a diagram in two ways, which never import each other:

- `yeelab/margin/` — the web UI's way: only the shares each method needs, a winner with a margin that is 0 on every border, and the regions as polygons traced along that zero set.
- `yeelab/pixels/` — the original way: the complete ranking profile of every pixel, cached on disk, the winner of every pixel, and the plot CLI. [docs/figures.py](docs/figures.py) and the tests use it, the tests as the reference for `margin/`.

Both have a `methods.py`, each with the methods in the form it needs. What both use is at the top of the package: the voter models and ranking cells (`ranking_cells.py`, `normal.py`), `voting.py` (the IRV rounds and the Condorcet-cycle code) and `threads.py`. `yeelab/web/` is the FastAPI app and the page, on top of `margin/`.

`uv run pytest` runs the tests; `uv run python docs/figures.py` regenerates the figures of [docs/math.pdf](docs/math.pdf).

