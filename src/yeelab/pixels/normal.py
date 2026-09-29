"""Ranking probabilities of every pixel for normal voters, exact and cached.

The same functions as pixels/beta.py. The cells are normal.normal_cells, and each
cell's probability is the signed sum of its edges' triangles (normal.triangle_terms).
"""

from pathlib import Path

import numpy as np

from yeelab.normal import node_medians, normal_cells, sigma_from_deviation, triangle_terms
from yeelab.pixels.beta import interpolate_to_pixels as _interpolate_to_pixels
from yeelab.pixels.cache import (
    DEFAULT_CACHE_ROOT,
    candidate_hash,
    load_node_probabilities,
    metadata,
    save_node_probabilities,
    value_token,
)
from yeelab.ranking_cells import NODES, effective_nodes


def interpolate_to_pixels(probs, medians, pixels):
    """Probabilities (pixels, pixels, R) from probabilities (N, N, R) at node_medians."""
    return _interpolate_to_pixels(probs, medians, pixels, transform=lambda m: m)

# ---------------------------------------------------------------- Probabilities

def compute_ranking_probabilities(candidates, medians, deviation, progress=None):
    """Returns (rankings (R, C), probabilities (N, N, R)); [i, j] is the pixel with
    mean (medians[i], medians[j]).

    progress: optional callable(done, total), called after each cell.
    """
    sigma = sigma_from_deviation(deviation)
    polygons, rankings = normal_cells(candidates, sigma)
    medians = np.asarray(medians, dtype=np.float64)
    # every edge of every cell (shared edges twice) in one parallel call
    segments = np.concatenate([np.hstack([poly, np.roll(poly, -1, axis=0)]) for poly in polygons])
    terms = triangle_terms(segments, medians, sigma)
    cells = np.repeat(np.arange(len(polygons)), [len(poly) for poly in polygons])
    probs = np.zeros((len(polygons), len(medians), len(medians)))
    np.add.at(probs, cells, terms)
    if progress is not None:
        progress(len(polygons), len(polygons))
    return rankings, np.clip(np.moveaxis(probs, 0, -1), 0.0, 1.0)


def ranking_probabilities(candidates, pixels, deviation, nodes=NODES,
                          progress=None):
    """(rankings, probabilities (pixels, pixels, R)) computed at the node medians
    and interpolated, without the cache."""
    medians = node_medians(pixels, nodes)
    rankings, probs = compute_ranking_probabilities(candidates, medians, deviation, progress)
    return rankings, interpolate_to_pixels(probs, medians, pixels)

# ---------------------------------------------------------------- Cache

def rankings_path(pixels, deviation, candidates, nodes, cache_root=DEFAULT_CACHE_ROOT):
    name = (f"P{pixels}_D{value_token(deviation)}_N{effective_nodes(pixels, nodes)}"
            f"_C{candidate_hash(candidates)}.npz")
    return Path(cache_root) / "normal" / name


def _rankings_metadata(pixels, deviation, candidates, nodes):
    return metadata(
        "rankings",
        distribution="normal",
        pixels=int(pixels),
        deviation=float(deviation),
        nodes=effective_nodes(pixels, nodes),
        candidate_hash=candidate_hash(candidates),
    )


def read_cached_ranking_probabilities(
    candidates,
    pixels,
    deviation,
    nodes=NODES,
    cache_root=DEFAULT_CACHE_ROOT,
):
    """(rankings, probabilities) from the cache, or None if not cached."""
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, candidates, nodes, cache_root)
    saved = load_node_probabilities(
        path, _rankings_metadata(pixels, deviation, candidates, nodes), candidates
    )
    if saved is None:
        return None
    rankings, probs, medians = saved
    return rankings, interpolate_to_pixels(probs, medians, pixels)


def generate_ranking_probabilities(
    candidates,
    pixels,
    deviation,
    nodes=NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    progress=None,
):
    """Compute (rankings, probabilities) and save the node probabilities to the cache."""
    candidates = np.asarray(candidates, dtype=np.float64)
    medians = node_medians(pixels, nodes)
    rankings, probs = compute_ranking_probabilities(candidates, medians, deviation, progress)
    path = rankings_path(pixels, deviation, candidates, nodes, cache_root)
    save_node_probabilities(
        path, _rankings_metadata(pixels, deviation, candidates, nodes),
        rankings, probs, medians, candidates,
    )
    return rankings, interpolate_to_pixels(probs, medians, pixels)
