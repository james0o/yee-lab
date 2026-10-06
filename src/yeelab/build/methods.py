"""The voting methods built from blocks (blocks.py), by name, in the order the web UI
lists them (margin/regions.py MARGINS)."""

from yeelab.build.blocks import (
    BordaCount,
    Eliminate,
    Fallback,
    Highest,
    Margins,
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

fptp = Highest(Tally(Plurality()))                   # first past the post
irv = Eliminate(Tally(Plurality()), how="min")       # instant runoff
borda = Highest(Tally(BordaCount()))                 # Borda count
baldwin = Eliminate(Tally(BordaCount()), how="min")  # the lowest Borda score out
nanson = Eliminate(Tally(BordaCount()), how="mean")  # every Borda score <= the mean out

P = Pairwise()
M = Margins(P)
schulze = Unbeaten(StrongestPaths(M))                # no one beats them along the strongest paths
condorcet = Unbeaten(M)                              # the Condorcet winner; CYCLE where there is none
minimax = Highest(Weakest(M))                        # the smallest worst defeat
black = Fallback(condorcet, borda)                   # the Condorcet winner, else Borda
# king of the hill: the fptp winner, or the most first choices among those who beat it
koth = Unbeaten(M, against=fptp, order=Tally(Plurality()))
king_runoff = Runoff(M, koth, irv)                   # the king of the hill against the irv winner

# the highest mean score, of 0 to 5: each voter's by where the distance is between the
# closest and the farthest candidate, that part of the way to the power SCORE_POWER
SCORE_POWER = 1.5  # the web UI's default; the one power that fits real ballots best (docs/math.pdf)
score = Highest(Tally(Score(6, power=SCORE_POWER)))

METHODS: dict[str, Winner] = {
    "fptp": fptp,
    "irv": irv,
    "borda": borda,
    "baldwin": baldwin,
    "nanson": nanson,
    "schulze": schulze,
    "condorcet": condorcet,
    "minimax": minimax,
    "black": black,
    "koth": koth,
    "king_runoff": king_runoff,
    "score": score,
}
