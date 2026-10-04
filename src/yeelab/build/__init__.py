"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (Approval, AvgApproval, BordaCount, Eliminate, Fallback,
                              GapApproval, Highest, Margins, Mix, Pairwise, Plurality, Runoff,
                              Score, StrongestPaths, Tally, Unbeaten, Weakest)

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

    # approved by the most voters, each approving down to their largest gap
    approval_gap = Highest(Tally(GapApproval()))
    # ... each approving those closer than the mean distance
    approval_avg = Highest(Tally(AvgApproval()))
    # the highest mean score of 0 to 5, each voter's in proportion to the distances
    score = Highest(Tally(Score(6)))

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs

    blocks      the blocks, their types (Ballot, CandidateTotals, PairDiffs, PairShares,
                Winner) and how each computes
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.approval import AVG, GAP, HALF, Cut
from yeelab.build.blocks import (
    Approval,
    AvgApproval,
    Ballot,
    BordaCount,
    CandidateTotals,
    Eliminate,
    Fallback,
    GapApproval,
    Highest,
    Margins,
    Mix,
    PairDiffs,
    PairShares,
    Pairwise,
    Plurality,
    Runoff,
    Score,
    StrongestPaths,
    Tally,
    Unbeaten,
    Weakest,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Approved, Scored, Share, Voters

__all__ = [
    "Approval", "AvgApproval", "Ballot", "BordaCount", "CandidateTotals", "Eliminate", "Fallback",
    "GapApproval", "Highest", "Margins", "Mix", "PairDiffs", "PairShares", "Pairwise", "Plurality", "Runoff",
    "Score", "StrongestPaths", "Tally", "Unbeaten", "Weakest", "Winner", "METHODS",
    "FIRST", "PAIRWISE", "PROFILE", "AVG", "GAP", "HALF", "Approved",
    "Scored", "Cut", "Share", "Voters",
]
