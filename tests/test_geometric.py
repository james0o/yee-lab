"""Checks of pixels at the geometric median of their voters (margin/geometric.py).

g(m), the geometric median of the voters with the medians m along the axes, is checked
against sampled voters and against a finer voter grid, for the symmetries of the voters,
and for being one to one. The diagram drawn with it (regions.py, /api/regions) must be
the usual one with every election moved from m to g(m), leave the strip along the walls
empty, and bring the Condorcet border between two candidates closer to their bisector.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.path import Path as MplPath

from yeelab.margin import shares
from yeelab.margin.geometric import (
    PIXEL_MEDIANS,
    geometric_median_at,
    geometric_medians,
    node_table,
    pixels_at,
)
from yeelab.margin.regions import MARGINS, grid, nearest, regions, winners
from yeelab.margin.shares import PIXELS, Model
from yeelab.ranking_cells import SPREADS, beta_params_at, pixel_medians
from yeelab.web.app import app

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
D, E = 3, 4  # their bisector is y = x + 0.2
MODEL = Model("beta", 0.25, "rms")
MODELS = [Model("beta", deviation, spread) for deviation in (0.05, 0.25, 0.4) for spread in SPREADS]
CENTRES = pixel_medians(PIXELS)
OUTERMOST = CENTRES[0]  # the median closest to a wall
RANDOM = np.random.default_rng(0).uniform(OUTERMOST, 1 - OUTERMOST, (40, 2))


def _id(model):
    return f"{model.deviation}-{model.spread}"


# ---------------------------------------------------------------- g

@pytest.mark.parametrize("model", MODELS, ids=_id)
def test_centre_is_its_own_geometric_median(model):
    """The voters of the centre are symmetric around it. On a line of symmetry, a
    median of 1/2 along one axis, the geometric median stays on the line."""
    np.testing.assert_allclose(geometric_median_at(model, [[0.5, 0.5]]), [[0.5, 0.5]], atol=1e-12)
    np.testing.assert_allclose(geometric_medians(model, np.array([0.5]))[0, 0], [0.5, 0.5], atol=1e-12)
    on_line = np.column_stack([np.full(len(RANDOM), 0.5), RANDOM[:, 1]])
    np.testing.assert_allclose(geometric_median_at(model, on_line)[:, 0], 0.5, atol=1e-12)


@pytest.mark.parametrize("model", MODELS, ids=_id)
def test_mirroring_and_swapping_the_axes_carry_over(model):
    """Voters mirrored along x, or with x and y swapped, have the mirrored or swapped
    geometric median. Computed at each point: nothing is filled in by symmetry."""
    g = geometric_median_at(model, RANDOM)
    mirrored = geometric_median_at(model, RANDOM * [-1, 1] + [1, 0])
    np.testing.assert_allclose(mirrored, g * [-1, 1] + [1, 0], atol=1e-10)
    np.testing.assert_allclose(geometric_median_at(model, RANDOM[:, ::-1]), g[:, ::-1], atol=1e-10)


@pytest.mark.parametrize("model", MODELS, ids=_id)
def test_interpolated_medians_are_symmetric_too(model):
    g = geometric_medians(model, CENTRES)
    np.testing.assert_allclose(g[::-1], g * [-1, 1] + [1, 0], atol=1e-10)
    np.testing.assert_allclose(g.transpose(1, 0, 2), g[..., ::-1], atol=1e-10)


def _weiszfeld(points, start):
    """Geometric median of sampled voters: Weiszfeld's iteration, the mean of the points
    weighted by 1 / distance, until it moves by less than 1e-7."""
    p = np.asarray(start, dtype=np.float64)
    for _ in range(500):
        weights = 1 / np.linalg.norm(points - p, axis=1)
        moved = weights @ points / weights.sum()
        if np.abs(moved - p).max() < 1e-7:
            return moved
        p = moved
    raise AssertionError("Weiszfeld's iteration did not settle")


@pytest.mark.parametrize("model", [MODEL, Model("beta", 0.1, "rms"), Model("beta", 0.3, "mean_abs")], ids=_id)
@pytest.mark.parametrize("median", [(0.75, 0.95), (0.95, 0.5), (0.3, 0.7), (OUTERMOST, OUTERMOST)])
def test_matches_sampled_voters(model, median):
    """A million voters put their geometric median within about 3e-4 of the exact one;
    g moves a pixel near a wall by a few hundredths."""
    (ax, bx), (ay, by) = beta_params_at(np.array(median), model.deviation, model.spread)
    rng = np.random.default_rng(1)
    voters = np.column_stack([rng.beta(ax, bx, 1_000_000), rng.beta(ay, by, 1_000_000)])
    sampled = _weiszfeld(voters, voters.mean(axis=0))
    np.testing.assert_allclose(geometric_median_at(model, [median])[0], sampled, atol=1.5e-3)
    index = np.searchsorted(CENTRES, median)  # the nearest pixels: interpolated from the nodes
    interpolated = geometric_medians(model, CENTRES[index])[0, 1]
    assert np.abs(interpolated - sampled).max() < 1.5e-3 + 1 / PIXELS


def test_agrees_with_the_measured_medians():
    """Deviation 0.25, rms: the geometric medians measured on samples before g existed."""
    g = geometric_median_at(MODEL, [[0.75, 0.95], [0.95, 0.5], [0.3, 0.7]])
    np.testing.assert_allclose(g, [[0.732, 0.891], [0.887, 0.5], [0.322, 0.678]], atol=1e-3)
    shift = np.linalg.norm(geometric_medians(Model("beta", 0.1, "rms"), CENTRES)
                           - np.stack(np.meshgrid(CENTRES, CENTRES, indexing="ij"), axis=-1), axis=-1)
    assert 0.016 < shift.max() < 0.021  # deviation 0.1: at most about 0.016 was measured


@pytest.fixture
def finer_voter_grid(monkeypatch):
    """The voter grid with cells a quarter as wide, for this test only."""
    shares._cached_voter_grid.cache_clear()
    monkeypatch.setattr(shares, "APPROVAL_CELLS", 4 * shares.APPROVAL_CELLS)
    yield
    shares._cached_voter_grid.cache_clear()


@pytest.mark.parametrize("model, error", [(Model("beta", 0.05, "rms"), 6e-5), (Model("beta", 0.1, "rms"), 2.5e-5),
                                          (MODEL, 5e-6), (Model("beta", 0.4, "mean_abs"), 5e-6)], ids=str)
def test_voter_grid_is_fine_enough(model, error, request):
    """The sum over the voter grid against the same sum over cells a quarter as wide,
    at random medians and next to the walls and corners."""
    walls = [[OUTERMOST, OUTERMOST], [OUTERMOST, 0.5], [1 - OUTERMOST, OUTERMOST], [0.2, 1 - OUTERMOST]]
    points = np.concatenate([RANDOM, walls])
    g = geometric_median_at(model, points)
    request.getfixturevalue("finer_voter_grid")
    assert np.abs(geometric_median_at(model, points) - g).max() < error


@pytest.mark.parametrize("model", MODELS, ids=_id)
def test_interpolation_from_the_nodes_matches_g(model):
    """At pixels between the nodes, the outermost ones included."""
    index = np.concatenate([np.random.default_rng(2).integers(0, PIXELS, (60, 2)),
                            [[0, 0], [0, PIXELS - 1], [0, PIXELS // 2], [1, 3]]])
    interpolated = geometric_medians(model, CENTRES)[index[:, 0], index[:, 1]]
    np.testing.assert_allclose(interpolated, geometric_median_at(model, CENTRES[index]), atol=5e-5)


@pytest.mark.parametrize("model", MODELS, ids=_id)
def test_is_one_to_one(model):
    """Neighbouring pixels keep their order along both axes and every cell of the moved
    pixel grid keeps its orientation: the grid does not fold over."""
    g = geometric_medians(model, CENTRES)
    along_x, along_y = np.diff(g, axis=0), np.diff(g, axis=1)
    assert along_x[..., 0].min() > 0 and along_y[..., 1].min() > 0
    cells = along_x[:, :-1, 0] * along_y[:-1, :, 1] - along_x[:, :-1, 1] * along_y[:-1, :, 0]
    assert cells.min() > 0


def test_normal_voters_have_it_at_their_pixel():
    model = Model("normal", 0.2)
    np.testing.assert_array_equal(geometric_median_at(model, RANDOM), RANDOM)
    g = geometric_medians(model, CENTRES)
    np.testing.assert_array_equal(g[7, 100], [CENTRES[7], CENTRES[100]])


def test_node_table_is_built_once():
    assert node_table(MODEL) is node_table(MODEL)
    assert not node_table(MODEL)[1].flags.writeable


# ---------------------------------------------------------------- The strip along the walls

@pytest.mark.parametrize("deviation, corner, middle", [(0.1, 0.0066, 0.0180), (0.25, 0.0136, 0.0521)])
def test_no_voters_have_their_geometric_median_next_to_a_wall(deviation, corner, middle):
    """The strip g does not reach (rms rule): narrowest at the corners, widest in the
    middle of a wall. Every geometric median is pulled towards the centre."""
    g = geometric_medians(Model("beta", deviation, "rms"), CENTRES)
    reach = g[0, :, 0]  # from the wall x = 0 to the geometric medians of its outermost pixels
    assert reach.min() == pytest.approx(corner, abs=2e-4) and reach.argmin() in (0, PIXELS - 1)
    assert reach.max() == pytest.approx(middle, abs=2e-4)
    assert abs(reach.argmax() - PIXELS / 2) <= 1
    assert g.min() == pytest.approx(corner, abs=2e-4) and g.max() == pytest.approx(1 - corner, abs=2e-4)
    towards_centre = np.abs(g - 0.5) <= np.abs(np.stack(np.meshgrid(CENTRES, CENTRES, indexing="ij"), -1) - 0.5)
    assert towards_centre.all()


def test_pixels_at_inverts_g():
    """The pixel found for a point has its voters' geometric median within two pixels
    of it (g moves neighbours up to twice as far apart); the points of the strip have
    none."""
    at = pixels_at(MODEL)
    assert at.shape == (PIXELS, PIXELS)
    middle = PIXELS // 2
    assert at[middle, middle] == middle * PIXELS + middle
    g = geometric_medians(MODEL, CENTRES).reshape(-1, 2)
    points = np.stack(np.meshgrid(CENTRES, CENTRES, indexing="ij"), axis=-1)
    found = at >= 0
    distance = np.linalg.norm(g[at[found]] - points[found], axis=-1)
    assert distance.max() < 2 / PIXELS
    assert 0.17 < (~found).mean() < 0.18
    # the strip: 0.0136 wide at a corner, 0.0521 in the middle of a wall
    assert not found[:4, :4].any() and found[5, 5]
    assert not found[:15, middle].any() and found[16:middle, middle].all()

    # The UI gets the quarter next to the origin and mirrors it. That finds a pixel as
    # close as the lookup's own (not always the same one: on a diagonal two are equally close).
    k, l = np.divmod(at[:middle, :middle], PIXELS)
    k, l = np.concatenate([k, PIXELS - 1 - k[::-1]]), np.concatenate([l, l[::-1]])
    k, l = np.concatenate([k, k[:, ::-1]], axis=1), np.concatenate([l, PIXELS - 1 - l[:, ::-1]], axis=1)
    np.testing.assert_array_equal(found, np.concatenate([found[:middle], found[middle - 1::-1]]))
    np.testing.assert_array_equal(found, np.concatenate([found[:, :middle], found[:, middle - 1::-1]], axis=1))
    mirrored = np.linalg.norm(g[(k * PIXELS + l)[found]] - points[found], axis=-1)
    np.testing.assert_allclose(mirrored, distance, atol=1e-9)


# ---------------------------------------------------------------- Regions

def _rasterize(shapes, points):
    """Winner at the `points` (K, 2) from the polygons, filled even-odd; -2 = none."""
    out = np.full(len(points), -2)
    for region in shapes:
        inside = np.zeros(len(points), dtype=bool)
        for polygon in region["polygons"]:
            for ring in polygon:
                inside ^= MplPath(np.reshape(ring, (-1, 2))).contains_points(points)
        out[inside] = region["winner"]
    return out


def _area(ring):
    x, y = np.reshape(ring, (-1, 2)).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


@pytest.mark.parametrize("method", ["fptp", "irv", "condorcet", "approval"])
def test_regions_are_the_same_elections_moved_by_g(method):
    """The winner drawn at g(m) is the winner of the pixel with the medians m, the one
    the usual diagram draws at m. The polygons tile the image of g and nothing else."""
    pixels = 150
    centres = pixel_medians(pixels)
    points = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1).reshape(-1, 2)
    usual = _rasterize(regions(method, CANDIDATES, MODEL, 300), points)
    shapes = regions(method, CANDIDATES, MODEL, 300, "geometric")
    moved = geometric_medians(MODEL, centres)
    # pixels may differ only right at a border
    assert (_rasterize(shapes, moved.reshape(-1, 2)) == usual).mean() > 0.995

    rings = [(polygon[0], polygon[1:]) for region in shapes for polygon in region["polygons"]]
    assert all(_area(outer) > 0 for outer, _ in rings)
    assert all(_area(hole) < 0 for _, holes in rings for hole in holes)
    edge = geometric_medians(MODEL, np.unique(grid(300, PIXELS)[1]))
    image = np.concatenate([edge[:, 0], edge[-1, 1:], edge[-2::-1, -1], edge[0, -2:0:-1]])
    total = sum(_area(r) for outer, holes in rings for r in [outer, *holes])
    assert total == pytest.approx(_area(image), abs=1e-3) and total < 0.83
    # nothing is drawn in the strip, not even next to a corner
    vertices = np.concatenate([np.reshape(r, (-1, 2)) for outer, holes in rings for r in [outer, *holes]])
    assert vertices.min() >= edge.min() - 1e-6 and vertices.max() <= edge.max() + 1e-6
    assert (_rasterize(shapes, np.array([[0.02, 0.5], [0.5, 0.97], [0.01, 0.01]])) == -2).all()


def _border(shapes, one, other):
    """The vertices on the rings of both winners: the crossings their regions share."""
    def vertices(winner):
        return {tuple(vertex) for region in shapes if region["winner"] == winner
                for polygon in region["polygons"] for ring in polygon
                for vertex in np.reshape(ring, (-1, 2)).tolist()}
    return np.array(sorted(vertices(one) & vertices(other)))


@pytest.mark.parametrize("spread", SPREADS)
def test_condorcet_border_is_closer_to_the_bisector(spread):
    """Deviation 0.25: between D (0.5, 0.5) and E (0.3, 0.7) the border should be the
    line y = x + 0.2. With pixels at the medians along the axes it is 0.045 to 0.075
    from it (rms), with pixels at geometric medians 0.025 to 0.039: about half the bend
    is gone. The rest is the skew of the voters and stays."""
    model = Model("beta", 0.25, spread)
    distance = {}
    for pixel_median in PIXEL_MEDIANS:
        border = _border(regions("condorcet", CANDIDATES, model, 320, pixel_median), D, E)
        assert len(border) > 100
        distance[pixel_median] = np.abs(border[:, 1] - border[:, 0] - 0.2) / np.sqrt(2)
    for measure in (np.mean, np.max, np.min):
        assert measure(distance["geometric"]) < 0.6 * measure(distance["marginal"])
    if spread == "rms":
        assert 0.044 < distance["marginal"].min() and distance["marginal"].max() < 0.077
        assert 0.024 < distance["geometric"].min() and distance["geometric"].max() < 0.040


def test_only_beta_voters_are_drawn_elsewhere():
    """Normal voters have both medians at their pixel, and so have no voters at all."""
    normal = Model("normal", 0.2)
    for method in ("irv", "voronoi"):
        assert regions(method, CANDIDATES, normal, 64, "geometric") == regions(method, CANDIDATES, normal, 64)
        assert regions(method, CANDIDATES, None, 64, "geometric") == regions(method, CANDIDATES, None, 64)
        assert regions(method, CANDIDATES, MODEL, 64, "geometric") != regions(method, CANDIDATES, MODEL, 64)
        assert regions(method, CANDIDATES, MODEL, 64, "marginal") == regions(method, CANDIDATES, MODEL, 64)


def test_voronoi_is_drawn_only_where_the_methods_are():
    """With pixels at geometric medians the Voronoi diagram has the outline of a
    method's diagram and nothing in the strip. Inside, it is the Voronoi diagram: the
    nearest candidate, with borders on the bisectors."""
    shapes = regions("voronoi", CANDIDATES, MODEL, 320, "geometric")
    method = regions("condorcet", CANDIDATES, MODEL, 320, "geometric")

    def total(shapes):
        return sum(_area(ring) for region in shapes for polygon in region["polygons"] for ring in polygon)

    assert total(shapes) == pytest.approx(total(method), abs=1e-5) and total(shapes) < 0.83
    assert (_rasterize(shapes, np.array([[0.02, 0.5], [0.5, 0.97], [0.01, 0.01]])) == -2).all()

    points = geometric_medians(MODEL, pixel_medians(150)).reshape(-1, 2)
    winner, margin = nearest(CANDIDATES, points)
    clear = margin > 1e-6  # not right on a border
    np.testing.assert_array_equal(_rasterize(shapes, points)[clear], winner[clear])
    border = _border(shapes, D, E)
    distance = np.abs(border[:, 1] - border[:, 0] - 0.2) / np.sqrt(2)
    # rounded to 6 decimals; the grid cell of a corner of three regions is only close
    assert len(border) > 100 and np.median(distance) < 1e-6 and distance.max() < 2e-3


@pytest.mark.parametrize("method", MARGINS)
def test_every_method_is_drawn_at_geometric_medians(method):
    """Winners and margins are those of the usual grid, so every region is there."""
    _, winner, _ = winners(method, CANDIDATES, MODEL, 64)
    shapes = regions(method, CANDIDATES, MODEL, 64, "geometric")
    assert {region["winner"] for region in shapes} == set(np.unique(winner).tolist())
    assert all(region["polygons"] for region in shapes)


# ---------------------------------------------------------------- API

def test_api_draws_pixels_at_either_median():
    client = TestClient(app)
    config = client.get("/api/config").json()
    assert [info["name"] for info in config["pixel_medians"]] == ["marginal", "geometric"]
    assert config["pixel_median"] == "marginal"

    def shapes(**settings):
        request = {"candidates": CANDIDATES.tolist(), "method": "condorcet", "deviation": 0.25,
                   "grid": 64, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes() == shapes(pixel_median="marginal") == regions("condorcet", CANDIDATES, MODEL, 64)
    assert shapes(pixel_median="geometric") == regions("condorcet", CANDIDATES, MODEL, 64, "geometric")
    assert shapes(pixel_median="geometric") != shapes()
    assert shapes(pixel_median="geometric", distribution="normal") == shapes(distribution="normal")
    assert shapes(method="voronoi", pixel_median="geometric") == regions("voronoi", CANDIDATES, MODEL, 64, "geometric")
    assert shapes(method="voronoi", pixel_median="geometric") != shapes(method="voronoi")
    request = {"candidates": CANDIDATES.tolist(), "method": "condorcet", "pixel_median": "mean"}
    assert client.post("/api/regions", json=request).status_code == 422


def test_api_returns_a_quarter_of_the_pixels_of_each_point():
    client = TestClient(app)
    response = client.get("/api/geometric", params={"deviation": 0.25, "spread": "rms"})
    assert response.status_code == 200
    half = PIXELS // 2
    np.testing.assert_array_equal(np.reshape(response.json()["pixels"], (half, half)),
                                  pixels_at(MODEL)[:half, :half])
    assert client.get("/api/geometric", params={"deviation": 0.25}).json() == response.json()
    assert client.get("/api/geometric", params={"deviation": 0.26}).status_code == 422
    assert client.get("/api/geometric", params={"deviation": 0.25, "spread": "wide"}).status_code == 422
    assert client.get("/api/geometric").status_code == 422
