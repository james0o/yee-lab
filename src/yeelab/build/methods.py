"""The voting methods built from blocks (blocks.py), by name, in the order the web UI
lists them (margin/regions.py MARGINS)."""

from yeelab.build.blocks import (
    AntiPlurality,
    BordaCount,
    D21,
    DescendingCoalitions,
    Conditional,
    Eliminate,
    Highest,
    Margins,
    Pairwise,
    Plurality,
    Score,
    ScoreComparisons,
    StrongestPaths,
    Tally,
    Unbeaten,
    Undecided,
    Weakest,
    Winner,
)

fptp = Highest(Tally(Plurality()))                   # first past the post
irv = Eliminate(Tally(Plurality()), how="min")       # instant runoff
dac = DescendingCoalitions()                         # descending acquiescing coalitions
borda = Highest(Tally(BordaCount()))                 # Borda count
baldwin = Eliminate(Tally(BordaCount()), how="min")  # the lowest Borda score out
nanson = Eliminate(Tally(BordaCount()), how="mean")  # every Borda score <= the mean out

P = Pairwise()
M = Margins(P)
# the two with the most first choices, head to head; a tie is a draw (no one)
two_round = Unbeaten(M, among=Highest(Tally(Plurality()), n=2))
schulze = Unbeaten(StrongestPaths(M))                # no one beats them along the strongest paths
condorcet = Unbeaten(M)                              # the Condorcet winner; CYCLE where there is none
minimax = Highest(Weakest(M))                        # the smallest worst defeat
black = Conditional(condorcet != Undecided(), condorcet, borda)  # the Condorcet winner, else Borda
# king of the hill: the fptp winner, or the most first choices among those who beat it
koth = Unbeaten(M, against=fptp, order=Tally(Plurality()))

# the highest mean score, of 0 to 5: each voter's by where the distance is between the
# closest and the farthest candidate, that part of the way to the power SCORE_POWER
SCORE_POWER = 1.5  # the web UI's default; the one power that fits real ballots best (docs/math.pdf)
score_ballot = Score(6, power=SCORE_POWER)
score = Highest(Tally(score_ballot))
# majority judgment on the same ballot: the highest median score, a tie of medians by
# Balinski and Laraki's tie-break for the voters of a point as a distribution
mj = Highest(Tally(score_ballot, statistic="median"))
# STAR: the two highest mean scores, head to head on the scores of each ballot (equal
# scores abstain by default); a tie is a draw
star = Unbeaten(ScoreComparisons(score_ballot), among=Highest(Tally(score_ballot), n=2))

top3 = Highest(Tally(Plurality()), n=3)
top2 = Highest(Tally(AntiPlurality()), n=2, among=top3)
system321 = Unbeaten(ScoreComparisons(Score(3, power=SCORE_POWER, fixed=True)), among=top2)

# Janeček's D21: at most 2 plus votes and 1 minus vote, the minus vote only after 2 plus votes
d21 = Highest(Tally(D21(max_positive=2, max_negative=1, negative_after=2, power=SCORE_POWER)))


#-----------------------------------experimental methods, do not tests-----------------------------------
TESTED = frozenset(globals())  # the names above the wall: the tests check only the methods among them
star_two_round = Unbeaten(ScoreComparisons(score_ballot), among= score | two_round)
finalists = Conditional(
    (fptp == two_round) & (two_round != condorcet),
    when_true=score | condorcet,
    when_false=score | two_round,
)

d20 = Highest(Tally(D21(max_positive=2, max_negative=0, power=SCORE_POWER)))


sun_two_round = Unbeaten(ScoreComparisons(score_ballot), among = finalists)
score_two = Conditional((fptp == score), when_true=score, when_false=two_round)


smj_fin = Conditional(
   (mj == two_round),
    when_true=mj,
    when_false=score

)
smj_two = Unbeaten(ScoreComparisons(score_ballot), among= smj_fin | two_round)
new = Conditional(
    (Highest(Tally(Plurality()), n=2)) == (Highest(Tally(score_ballot), n=2)), 
    when_true=two_round, 
    when_false = Conditional(
        (score == fptp) & (score == condorcet),
        when_true=score,
        when_false=Undecided()
    )
)

three_round = Eliminate(Tally(Plurality()), how="min", among=top3)

adhoc = Conditional(
    fptp == score, 
    when_true=score, 
    when_false=Conditional(
        two_round == score,
        when_true=score,
        when_false= Conditional(
            score == koth,
            when_true=koth,
            when_false=Undecided()
        )
    )
)


METHODS: dict[str, Winner] = {
    "fptp": fptp,
    "two_round": two_round,
    'three_round': three_round,
    "irv": irv,
    "dac": dac,
    "borda": borda,
    "baldwin": baldwin,
    "nanson": nanson,
    "schulze": schulze,
    "condorcet": condorcet,
    "minimax": minimax,
    "black": black,
    "koth": koth,
    "score": score,
    "star": star,
    "star_two_round": star_two_round,
    "system321": system321,
    "mj": mj,
    "d21": d21,
    "d20": d20,
    "smj_two": smj_two,
    "new": new,
    "adhoc": adhoc,
}
