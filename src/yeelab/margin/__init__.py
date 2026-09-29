"""Diagrams fast enough to follow a dragged candidate: the web UI's computation.

    shares          only the shares each method needs, edge integrals cached
    beta_tables     tabulated Beta CDF and quantile, compiled edge integrals
    methods         the methods on those shares: winner and a margin that is 0 on borders
    regions         win regions as polygons, traced as the zero set of the margin
"""
