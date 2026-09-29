"""The pixel pipeline: the complete ranking profile of every pixel of an n x n grid.

    beta, normal    ranking probabilities at the nodes, interpolated to the pixels and
                    cached on disk (cache.py, into cache/); the same functions for both
    methods         the voting methods on the profile, winner per pixel
    plot            the CLI that saves them as PNGs into plots/

The web UI does not use it. docs/figures.py does, and the tests check margin/ against it.
"""
