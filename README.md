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

[plot.py](plot.py) saves one PNG per method into `plots/`:

```sh
uv run python plot.py                         # all methods, Beta voters
uv run python plot.py -m irv -m schulze       # only some methods
uv run python plot.py --distribution normal   # original Yee model
uv run python plot.py -d 0.3 -p 800           # deviation 0.3, 800x800 pixels
uv run python plot.py -h                      # all options
```

Ranking probabilities are cached in `cache/`, so a second run with the same settings is fast (`--regenerate` ignores the cache). Candidates are set in [plot.py](plot.py).

### Web UI

```sh
uv run fastapi dev
```

Then open http://127.0.0.1:8000. [main.py](main.py) serves the page in [ui/index.html](ui/index.html) and computes the diagrams. The diagram is drawn as curves, not pixels: the backend returns the region of every winner as polygons (`POST /api/regions`). It computes only the shares the method needs, caches everything a drag does not change, and traces each border as the zero set of the winner's margin. The details are in the last chapter of [docs/math.pdf](docs/math.pdf).

- **Candidates:** drag one to move it, click empty space to add one (up to 8), right-click to remove one. The diagram follows the drag, typically within 10–40 ms. IRV is the exception: it needs every ranking cell, so it lags (about 0.1 s for 5 candidates, 0.8 s for 8).
- **Method:** Voronoi (no voters, the reference), FPTP, IRV, Borda, Schulze or Condorcet cycle.
- **Voters:** Beta, or normal for the original Yee model.
- **Deviation:** $D$ from $0$ to $0.4$. At $0$ every voter sits at their pixel, so every method draws the Voronoi diagram.
- **Hover** over the square to see the voters of that pixel: their 2D density over the square and their distribution along $x$ above it.

