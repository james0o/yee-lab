"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import BordaCount, Eliminate, Highest, Plurality, Schulze, Tally

    borda   = Highest(Tally(BordaCount()))
    fptp    = Highest(Tally(Plurality()))
    irv     = Eliminate(Tally(Plurality()), how="min")
    baldwin = Eliminate(Tally(BordaCount()), how="min")
    nanson  = Eliminate(Tally(BordaCount()), how="mean")
    schulze = Schulze()

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs

    blocks      the blocks, their types (Ballot, Scores, Winner) and how each computes
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.build.blocks import (
    Ballot,
    BordaCount,
    Eliminate,
    Highest,
    Plurality,
    Schulze,
    Scores,
    Tally,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Share, Voters

__all__ = [
    "Ballot", "BordaCount", "Eliminate", "Highest", "Plurality", "Schulze", "Scores",
    "Tally", "Winner", "METHODS", "FIRST", "PAIRWISE", "PROFILE", "Share", "Voters",
]
