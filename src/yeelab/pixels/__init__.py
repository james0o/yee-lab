"""The pixel pipeline: the complete ranking profile of every pixel of an n x n grid.

    beta, normal    ranking probabilities at the nodes, interpolated to the pixels and
                    cached on disk (cache); the same functions for both distributions
    methods         the voting methods on the profile, winner per pixel
    plot            the CLI that saves them as PNGs

The web UI does not use it. docs/figures.py does, and the tests check the UI's shares
and regions against it.
"""
