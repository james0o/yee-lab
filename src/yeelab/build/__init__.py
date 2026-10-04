"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (BordaCount, Eliminate, Fallback, Highest, Margins, Mix,
                              Pairwise, Plurality, Runoff, Score, ScoreAvg, ScoreDH,
                              StrongestPaths, Tally, Unbeaten, Weakest)

    borda   = Highest(Tally(BordaCount()))
    fptp    = Highest(Tally(Plurality()))
    irv     = Eliminate(Tally(Plurality()), how="min")
    baldwin = Eliminate(Tally(BordaCount()), how="min")
    nanson  = Eliminate(Tally(BordaCount()), how="mean")

    M = Margins(Pairwise())
    schulze = Unbeaten(StrongestPaths(M))
    condorcet = Unbeaten(M)
    minimax = Highest(Weakest(M))
    black   = Fallback(condorcet, borda)
    koth    = Unbeaten(M, against=fptp, order=Tally(Plurality()))  # king of the hill
    king_runoff = Runoff(M, koth, irv)  # their winners, one on one

    # the highest mean score of 0 to 5, each voter's in proportion to the distances
    score_range = Highest(Tally(Score(6)))
    # ... with the mean distance in the middle of the scale
    score_avg = Highest(Tally(ScoreAvg(6)))
    # ... the steps from 5 to 0 shared out among the gaps between neighbours, each to the
    # largest gap / (its steps + delta): D'Hondt's rule at delta = 1, the default 0.8
    score_dh = Highest(Tally(ScoreDH(6)))

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs

    blocks      the blocks, their types (Ballot, CandidateTotals, PairDiffs, PairShares,
                Winner) and how each computes
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.build.blocks import (
    Ballot,
    BordaCount,
    CandidateTotals,
    Eliminate,
    Fallback,
    Highest,
    Margins,
    Mix,
    PairDiffs,
    PairShares,
    Pairwise,
    Plurality,
    Runoff,
    Score,
    ScoreAvg,
    ScoreDH,
    StrongestPaths,
    Tally,
    Unbeaten,
    Weakest,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Scored, Share, Voters

__all__ = [
    "Ballot", "BordaCount", "CandidateTotals", "Eliminate", "Fallback", "Highest", "Margins", "Mix",
    "PairDiffs", "PairShares", "Pairwise", "Plurality", "Runoff", "Score", "ScoreAvg", "ScoreDH",
    "StrongestPaths", "Tally", "Unbeaten", "Weakest", "Winner", "METHODS",
    "FIRST", "PAIRWISE", "PROFILE", "Scored", "Share", "Voters",
]
