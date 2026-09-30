"""Diagrams fast enough to follow a dragged candidate: the web UI's computation.

    shares          only the shares each method needs, edge integrals cached
    beta_tables     tabulated Beta CDF and quantile, compiled edge integrals
    regions         the shares a method needs on a grid, and its win regions as
                    polygons, traced as the zero set of the margin (the methods,
                    winner and a margin that is 0 on borders: yeelab.build)
"""
