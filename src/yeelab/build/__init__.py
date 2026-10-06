"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (BordaCount, Eliminate, Fallback, Highest, Margins, Mix,
                              Pairwise, Plurality, Runoff, Score, ScoreAvg, ScoreCluster,
                              ScoreDH, ScoreHybrid, StrongestPaths, Tally, Unbeaten, Weakest)

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

    # the highest mean score of 0 to 5, each voter's by where the distance is between the
    # closest and the farthest candidate, that part of the way to the power 1.5
    score = Highest(Tally(Score(6, power=1.5)))
    # the other ways to score of yeelab.score, for comparisons (not in the web UI)
    ScoreAvg(6), ScoreDH(6, delta=0.8), ScoreHybrid(6, delta=0.8), ScoreCluster(6, mu=0.1, kappa=2.0)

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
    ScoreCluster,
    ScoreDH,
    ScoreHybrid,
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
    "PairDiffs", "PairShares", "Pairwise", "Plurality", "Runoff", "Score", "ScoreAvg", "ScoreCluster",
    "ScoreDH", "ScoreHybrid", "StrongestPaths", "Tally", "Unbeaten", "Weakest", "Winner", "METHODS",
    "FIRST", "PAIRWISE", "PROFILE", "Scored", "Share", "Voters",
]
