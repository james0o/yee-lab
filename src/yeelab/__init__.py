"""Yee diagrams with Beta-distributed voters.

The web UI (web/) draws the win regions as polygons, from the modules here:

    ranking_cells   Beta voters of a pixel (spread rules), the cells of equal ranking,
                    interpolation from the Chebyshev nodes
    normal          normal voters: sigma, their cells, exact triangle probabilities
    beta_tables     tabulated Beta CDF and quantile, compiled edge integrals
    shares          the shares each method needs, with the edge cache
    methods         the methods on those shares, with a margin that is 0 on borders
    regions         win regions as polygons
    threads         the thread pool of the compiled kernels

pixels/ is the pipeline the UI replaced: the complete profile of every pixel, cached on
disk, the methods on it and the plot CLI. docs/figures.py and the tests use it.
"""
