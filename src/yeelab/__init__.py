"""Yee diagrams with Beta-distributed voters.

Two ways to compute a diagram, each in its own folder, and what both use here:

    margin/         the web UI's: only the shares a method needs, winners with a margin
                    that is 0 on every border, regions as polygons (its zero set)
    pixels/         the original pipeline: the complete ranking profile of every pixel,
                    cached, winners per pixel, PNGs; docs/figures.py and the tests use it
    web/            the FastAPI app and the page, on margin/

margin/ and pixels/ never import each other. Shared by both:

    ranking_cells   Beta voters of a pixel (spread rules), the cells of equal ranking,
                    interpolation from the Chebyshev nodes
    normal          normal voters: sigma, their cells, exact triangle probabilities
    distributions   the voter distributions
    voting          the Condorcet cycle winner code, the compiled rounds of IRV
    threads         the thread pool of the compiled kernels
"""
