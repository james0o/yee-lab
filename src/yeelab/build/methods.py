"""The voting methods built from blocks (blocks.py), by name, in the order the web UI
lists them (margin/regions.py MARGINS)."""

from yeelab.build.blocks import (
    Approval,
    BordaCount,
    Eliminate,
    Fallback,
    GapApproval,
    Highest,
    Margins,
    Mix,
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

approval = Highest(Tally(Approval()))                # approved by the most voters, each approving half
approval_gap = Highest(Tally(GapApproval()))         # ... each approving down to their largest gap


def mixed_approval(half: float) -> Winner:
    """Approval by both kinds of voters: the share `half` of them approve half of the
    candidates, the others down to their largest gap. It is approval_gap at 0 and
    approval at 1."""
    return Highest(Tally(Mix(GapApproval(), Approval(), share=half)))


approval_mix = mixed_approval(0.5)                   # ... half of them each way


score = Highest(Tally(Score(6)))                     # the highest mean score, of 0 to 5

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
    "approval": approval,
    "approval_gap": approval_gap,
    "approval_mix": approval_mix,
    "score": score,
}
