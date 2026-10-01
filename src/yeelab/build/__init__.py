"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (Approval, BordaCount, Eliminate, Fallback, GapApproval, Highest,
                              Margins, Pairwise, Plurality, Runoff, StrongestPaths, Tally,
                              Unbeaten, Weakest)

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

    approval     = Highest(Tally(Approval()))     # approved by the most voters, each approving half
    approval_gap = Highest(Tally(GapApproval()))  # ... each approving down to their largest gap

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs

    blocks      the blocks, their types (Ballot, CandidateTotals, PairDiffs, PairShares,
                Winner) and how each computes
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.approval import GAP, HALF, Cut
from yeelab.build.blocks import (
    Approval,
    Ballot,
    BordaCount,
    CandidateTotals,
    Eliminate,
    Fallback,
    GapApproval,
    Highest,
    Margins,
    PairDiffs,
    PairShares,
    Pairwise,
    Plurality,
    Runoff,
    StrongestPaths,
    Tally,
    Unbeaten,
    Weakest,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Approved, Share, Voters

__all__ = [
    "Approval", "Ballot", "BordaCount", "CandidateTotals", "Eliminate", "Fallback", "GapApproval",
    "Highest", "Margins", "PairDiffs", "PairShares", "Pairwise", "Plurality", "Runoff",
    "StrongestPaths", "Tally", "Unbeaten", "Weakest", "Winner", "METHODS", "FIRST", "PAIRWISE",
    "PROFILE", "GAP", "HALF", "Approved", "Cut", "Share", "Voters",
]
