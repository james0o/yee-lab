"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (BordaCount, Eliminate, Fallback, Highest, Margins, Pairwise,
                              Plurality, StrongestPaths, Tally, Unbeaten, Weakest)

    borda   = Highest(Tally(BordaCount()))
    fptp    = Highest(Tally(Plurality()))
    irv     = Eliminate(Tally(Plurality()), how="min")
    baldwin = Eliminate(Tally(BordaCount()), how="min")
    nanson  = Eliminate(Tally(BordaCount()), how="mean")

    M = Margins(Pairwise())
    schulze = Unbeaten(StrongestPaths(M))
    condorcet_cycle = Unbeaten(M)
    minimax = Highest(Weakest(M))
    black   = Fallback(condorcet_cycle, borda)
    koth    = Unbeaten(M, against=fptp, order=Tally(Plurality()))  # king of the hill

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs

    blocks      the blocks, their types (Ballot, Duels, Links, Scores, Winner) and how
                each computes
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.build.blocks import (
    Ballot,
    BordaCount,
    Duels,
    Eliminate,
    Fallback,
    Highest,
    Links,
    Margins,
    Pairwise,
    Plurality,
    Scores,
    StrongestPaths,
    Tally,
    Unbeaten,
    Weakest,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Share, Voters

__all__ = [
    "Ballot", "BordaCount", "Duels", "Eliminate", "Fallback", "Highest", "Links", "Margins",
    "Pairwise", "Plurality", "Scores", "StrongestPaths", "Tally", "Unbeaten", "Weakest",
    "Winner", "METHODS", "FIRST", "PAIRWISE", "PROFILE", "Share", "Voters",
]
