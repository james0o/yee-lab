"""Diagrams fast enough to follow a dragged candidate: the web UI's computation.

    shares          only the shares each method needs, edge integrals cached
    beta_tables     tabulated Beta CDF and quantile, compiled edge integrals
    geometric       the geometric median of the voters of a pixel, where the pixel can
                    be drawn instead of at their medians along the axes
    regions         the shares a method needs on a grid, and its win regions as
                    polygons, traced as the zero set of the margin (the methods,
                    winner and a margin that is 0 on borders: yeelab.build)
"""
