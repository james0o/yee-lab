"""Voter distributions around each pixel: Beta (ranking_cells.py) or normal (normal.py)."""

from typing import Literal, get_args

Distribution = Literal["beta", "normal"]
DISTRIBUTIONS = get_args(Distribution)
