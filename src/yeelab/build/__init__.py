"""Voting methods built from small blocks, like in Scratch but with Python constructors:

    from yeelab.build import (AntiPlurality, BordaCount, Bullet, Burial, Compromise, D21, DescendingCoalitions,
                              Conditional, Eliminate,
                              Highest, Lead, Margins, Mix, Pairwise, Plurality, Score,
                              ScoreAvg, ScoreCluster, ScoreComparisons, ScoreDH, ScoreHybrid,
                              Smith, StrongestPaths, Tally, Top, Unbeaten, Weakest)

    borda   = Highest(Tally(BordaCount()))
    fptp    = Highest(Tally(Plurality()))   # = Eliminate(Tally(Plurality()), how="all")
    irv     = Eliminate(Tally(Plurality()), how="min")
    baldwin = Eliminate(Tally(BordaCount()), how="min")
    nanson  = Eliminate(Tally(BordaCount()), how="mean")
    dac     = DescendingCoalitions()        # descending acquiescing coalitions
    anti    = Highest(Tally(AntiPlurality()))

    M = Margins(Pairwise())
    schulze = Unbeaten(StrongestPaths(M))
    condorcet = Unbeaten(M)
    minimax = Highest(Weakest(M))
    black   = Conditional(condorcet, when_true=condorcet, when_false=borda)
    koth    = Unbeaten(M, against=fptp, order=Tally(Plurality()))  # king of the hill
    # the two finalists head to head; a tie is a draw (no one, voting.CYCLE)
    two_round = Unbeaten(M, among=Highest(Tally(Plurality()), n=2))
    # FPTP where someone has 55% of the first choices, else the two-round system; Borda
    # where its winner leads by 2% of the C - 1 points, else Schulze (totals from 0 to 1)
    Conditional(Top(Tally(Plurality())) > 0.55, fptp, two_round)
    Conditional(Lead(Tally(BordaCount())) > 0.02, borda, schulze)
    # instant runoff in the Smith set, the smallest set of candidates who each beat every
    # candidate outside it
    smith_irv = Eliminate(Tally(Plurality()), how="min", among=Smith(M))
    # the same where a candidate beats another only with 55% of the voters: IRV decides
    # among all who are not beaten so
    Eliminate(Tally(Plurality()), how="min", among=Smith(M, majority=0.55))

    # the highest mean score of 0 to 5, each voter's by where the distance is between the
    # closest and the farthest candidate, that part of the way to the power 1.5
    ballot = Score(6, power=1.5)
    score = Highest(Tally(ballot))
    # majority judgment: the highest median score, ties by Balinski and Laraki's tie-break
    mj    = Highest(Tally(ballot, statistic="median"))
    star  = Unbeaten(ScoreComparisons(ballot), among=Highest(Tally(ballot), n=2))
    smith_score = Highest(Tally(ballot), among=Smith(M))  # the highest mean score in the Smith set
    # any of them where 30% of the voters bullet vote: their first choice first and all
    # other candidates equal below it
    Bullet(borda, share=0.3)
    # 30% of the voters rank the frontrunner they like less of the two with the most
    # first choices last
    Burial(borda, share=0.3)
    # 30% of the voters whose first choice is neither frontrunner rank the one they like
    # more first
    Compromise(fptp, share=0.3)
    # the other ways to score of yeelab.score, for comparisons (not in the web UI)
    ScoreAvg(6), ScoreDH(6, delta=0.8), ScoreHybrid(6, delta=0.8), ScoreCluster(6, mu=0.1, kappa=2.0)
    # Janeček's D21: at most 2 plus votes and 1 minus vote, the minus vote only after 2 plus votes
    d21   = Highest(Tally(D21(max_positive=2, max_negative=1, negative_after=2, power=1.5)))

    winner, margin = nanson.evaluate(voters)  # Voters with the shares in nanson.needs
    chosen, margin = Highest(Tally(ballot), n=2).select(voters)  # several candidates: a mask

    blocks      the blocks, their types (Ballot, CandidateTotals, PairDiffs, PairShares,
                Winner, Measure, Condition) and how each computes
    judgment    the majority judgment of a score ballot, compiled: the order and the
                totals of Tally(ballot, statistic="median")
    strategic   the methods where some of the voters vote strategically: Bullet, Burial,
                Compromise
    rounds      one round of Eliminate, compiled: who goes at every point
    voters      Voters: the shares at every point that a method is evaluated on
    methods     the methods above by name (METHODS); margin/regions.py draws them
"""

from yeelab.build.blocks import (
    AntiPlurality,
    Ballot,
    BordaCount,
    CandidateTotals,
    D21,
    DescendingCoalitions,
    Eliminate,
    Conditional,
    Highest,
    Lead,
    Margins,
    Measure,
    Mix,
    PairDiffs,
    PairShares,
    Pairwise,
    Plurality,
    Score,
    ScoreComparisons,
    ScoreAvg,
    ScoreCluster,
    ScoreDH,
    ScoreHybrid,
    Smith,
    StrongestPaths,
    Tally,
    Top,
    Unbeaten,
    Undecided,
    Union,
    Weakest,
    Winner,
)
from yeelab.build.methods import METHODS
from yeelab.build.strategic import Bullet, Burial, Compromise
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Ballots, Graded, Scored, ScoredPairwise, Share, Voters

__all__ = [
    "AntiPlurality", "Ballot", "Ballots", "BordaCount", "Bullet", "Burial", "CandidateTotals", "Compromise",
    "D21", "DescendingCoalitions",
    "Conditional", "Eliminate", "Highest", "Lead", "Margins", "Measure", "Mix", "PairDiffs", "PairShares",
    "Pairwise", "Plurality", "Score", "ScoreComparisons", "ScoreAvg", "ScoreCluster", "ScoreDH",
    "ScoreHybrid", "Smith", "StrongestPaths", "Tally", "Top", "Unbeaten", "Undecided", "Union", "Weakest",
    "Winner", "METHODS",
    "FIRST", "PAIRWISE", "PROFILE", "Graded", "Scored", "ScoredPairwise", "Share", "Voters",
]
