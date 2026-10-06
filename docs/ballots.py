"""The score ballot of score.py against real ballots: the numbers of math.typ
(@sec-score-data).

Run from the repository root:

    uv run --with pandas --with rdata --with pyreadstat python docs/ballots.py

The first run downloads the public data sets into docs/data/ (not in git). Each pairs,
for the same people and the same candidates, a fine opinion or a distance with a
coarser ballot or rating:

    Voter Autrement 2017, online (Zenodo 1199545, ODbL): 0-100 opinions of the 11
        candidates of the French presidential election, and an approval ballot and
        evaluation ballots on 0/1/2, -1/0/1, 0/1/2/3 and -1/0/1/2
    Voter Autrement 2022 (Zenodo 10998451, ODbL): 0-100 opinions of 12 candidates,
        approval, score ballots on 3 and 4 levels and majority judgment on 5 and 7
    Votare Altrimenti 2022 (Mendeley Data dgsd5yb7zp, CC BY 4.0): 0-100 opinions of 14
        Italian parties, approval, a score ballot 0-4 and an evaluative ballot 0-4
    SensoMineR 1.28 (CRAN): perfume_ideal and cream_id, consumers who rate products and
        give the perceived and the ideal intensity of each attribute
    CSES Integrated Module Dataset (cses.org, doi:10.7804/cses.imd.2024-02-27): the
        left-right placement of the respondent and of up to nine parties, and the like
        or dislike of each party, 0-10
    Kuhlmann et al. 2017 (OSF gvqjs): 26 personality items on a slider 1-101 and on a
        Likert scale 1-5, the same items by the same people
    Zhang et al., CES-D 8 (OSF 5uwcp): 8 items on two sliders 0-100, a Likert scale 0-4
        and a scale 0-14, the same items by the same people

A voter's distance to a candidate is 100 minus the opinion, the distance between the
perceived and the ideal profile of a product, the distance on the left-right scale, or
the top of the slider minus the answer. The ballot of score.py depends on the distances
only up to a common scale and shift, so any such distance gives the same ballot. Prints
the numbers quoted in the text.
"""

import io
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

from yeelab.score import AVG, CLUSTER, DHONDT, HYBRID, RANGE, from_distances

DATA = Path(__file__).parent / "data"
ZENODO_2022 = "https://zenodo.org/records/10998451/files/{}?download=1"
MENDELEY = ("https://data.mendeley.com/public-files/datasets/dgsd5yb7zp/files/"
            "bb6279ec-e6ba-4e89-95ce-a05d7b298114/file_downloaded")
ZHANG = "https://osf.io/download/45pnh/?view_only=ad027d57b2ea45ab976b9882dea98f93"
SOURCES = {  # local name: url
    "voter_autrement_2017_online.csv": "https://zenodo.org/records/1199545/files/merged.csv?download=1",
    "SensoMineR_1.28.tar.gz": "https://cran.r-project.org/src/contrib/Archive/SensoMineR/SensoMineR_1.28.tar.gz",
    "cses_imd_csv.zip": "https://cses.org/wp-content/uploads/2024/02/cses_imd_csv.zip",
    "votare2022/survey_en.csv": MENDELEY,
    "kuhlmann2017/data.sav": "https://osf.io/download/vr3wh/",
    "zhang_cesd/data.csv": ZHANG,
    **{f"va2022/{name}.csv": ZENODO_2022.format(f"{name}.csv") for name in (
        "notes", "approval", "scores_3_pos", "scores_3_neg", "scores_4_pos", "scores_4_neg",
        "majority_judgement_5", "majority_judgement_7")},
}
POWERS = np.round(np.arange(0.8, 3.001, 0.05), 2)  # the powers searched
SHOWN = np.round(np.arange(1.0, 2.501, 0.1), 2)    # those of the summary table
UI = (1.0, 1.5, 2.0)                               # the slider's ends and default


def fetch(name):
    """docs/data/<name>, downloaded once."""
    path = DATA / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCES[name]}")
        with urllib.request.urlopen(SOURCES[name]) as response:
            path.write_bytes(response.read())
    return path


@dataclass
class Ballots:
    """The distances r and the real ballots b (n, C) of one ballot kind; groups of
    voters with different numbers of candidates (CSES) are separate arrays."""

    name: str
    election: bool  # ballots of a voting experiment, or ratings without a stake
    levels: int
    r: list
    b: list

    @property
    def voters(self):
        return sum(len(r) for r in self.r)


def kept(r, b, levels):
    """The voters with every distance and score, distances not all alike, and a ballot
    that uses its lowest and its highest score, as the ballot of score.py does. A
    rating on more than four levels that does not is stretched to them first: few
    people use the whole of a long scale when nothing is at stake."""
    r, b = np.asarray(r, dtype=float), np.asarray(b, dtype=float)
    ok = ~np.isnan(r).any(axis=1) & ~np.isnan(b).any(axis=1) & (np.ptp(np.nan_to_num(r), axis=1) > 0)
    r, b = r[ok], b[ok]
    if levels > 4:
        ok = np.ptp(b, axis=1) > 0
        r, low, high = r[ok], b[ok].min(axis=1, keepdims=True), b[ok].max(axis=1, keepdims=True)
        b = np.rint((b[ok] - low) / (high - low) * (levels - 1))
    ok = (b.min(axis=1) == 0) & (b.max(axis=1) == levels - 1)
    return r[ok], b[ok].astype(int)


def ballots(name, election, levels, r, b):
    r, b = kept(r, b, levels)
    return Ballots(name, election, levels, [r], [b])

# ---------------------------------------------------------------- data


def voter_autrement_2017():
    merged = pd.read_csv(fetch("voter_autrement_2017_online.csv"), comment="#", low_memory=False)
    names = [column[len("Approval "):] for column in merged.columns[1:12]]

    def numbers(prefix):
        return merged[[f"{prefix} {name}" for name in names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    r = 100 - numbers("Evaluation Continuous")
    out = [ballots("VA 2017 approval", True, 2, r, numbers("Approval"))]
    approved = numbers("Approval")
    voted = merged["official vote"][~np.isnan(approved).any(axis=1) & ~np.isnan(r).any(axis=1)]
    print(f"Voter Autrement 2017: {len(merged)} participants, {len(voted)} with an approval ballot and every "
          "opinion; their official first-round vote "
          + "  ".join(f"{name} {share:.0%}" for name, share in voted.value_counts(normalize=True)[:3].items()))
    for scale, levels, low in (("0/1/2", 3, 0), ("-1/0/1", 3, -1), ("0/1/2/3", 4, 0), ("-1/0/1/2", 4, -1)):
        out.append(ballots(f"VA 2017 {scale}", True, levels, r, numbers(f"Evaluation {scale}") - low))
    return out


def voter_autrement_2022():
    notes = pd.read_csv(fetch("va2022/notes.csv")).set_index("id")
    print(f"Voter Autrement 2022: {len(notes)} participants with every opinion")
    good = ["Insuffisant", "Passable", "Assez bien", "Bien", "Très bien"]
    out = []
    for name, label, levels, low, grades in (
            ("approval", "approval", 2, 0, None), ("scores_3_pos", "0/1/2", 3, 0, None),
            ("scores_3_neg", "-1/0/1", 3, -1, None), ("scores_4_pos", "0/1/2/3", 4, 0, None),
            ("scores_4_neg", "-1/0/1/2", 4, -1, None), ("majority_judgement_5", "MJ 5 grades", 5, 0, good),
            ("majority_judgement_7", "MJ 7 grades", 7, 0, ["A rejeter", *good, "Excellent"])):
        ballot = pd.read_csv(fetch(f"va2022/{name}.csv")).set_index("id")
        both = ballot.index.intersection(notes.index)
        ballot = ballot.loc[both, notes.columns]
        if grades:
            ballot = ballot.replace({grade: k for k, grade in enumerate(grades)})
        out.append(ballots(f"VA 2022 {label}", True, levels, 100 - notes.loc[both].to_numpy(float),
                           ballot.apply(pd.to_numeric, errors="coerce").to_numpy(float) - low))
    return out


def votare_altrimenti_2022():
    survey = pd.read_csv(fetch("votare2022/survey_en.csv"), low_memory=False)
    print(f"Votare Altrimenti 2022: {len(survey)} respondents")
    parties = range(1, 15)

    def numbers(prefix):
        return survey[[f"{prefix}_{k}" for k in parties]].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)

    opinion = numbers("D7")
    opinion[opinion < 0] = np.nan  # -1: no opinion
    approval = survey[[f"D3_{k}" for k in parties]]
    approved = np.where(approval.isna(), np.nan, (approval == "Selected").to_numpy(float))
    words = ["Severely Insufficient", "Insufficient", "Acceptable", "Good", "Excellent"]
    worded = survey[[f"D6a_{k}" for k in parties]].replace({word: k for k, word in enumerate(words)})
    evaluative = np.where(survey["Rand"].str.startswith("EVa").to_numpy()[:, None],
                          worded.apply(pd.to_numeric, errors="coerce").to_numpy(float), numbers("D6b"))
    r = 100 - opinion
    return [ballots("Italy 2022 approval", True, 2, r, approved),
            ballots("Italy 2022 score 0-4", True, 5, r, numbers("D5")),
            ballots("Italy 2022 evaluative 0-4", True, 5, r, evaluative)]


def ideal_profiles():
    """The distance between the perceived and the ideal profile of each product, and its
    liking."""
    out = []
    with tarfile.open(fetch("SensoMineR_1.28.tar.gz")) as tar:
        for name, person, product, levels in (("perfume_ideal", "user", "product", 9),
                                              ("cream_id", "juge", "produit", 11)):
            raw = tar.extractfile(f"SensoMineR/data/{name}.rda").read()
            frame = rdata.read_rda(io.BytesIO(raw))[name].sort_values([person, product])
            ideal = [column for column in frame.columns if str(column).startswith("id_")]
            perceived = [frame.columns[frame.columns.get_loc(column) - 1] for column in ideal]
            n, c = frame[person].nunique(), frame[product].nunique()
            gap = frame[perceived].to_numpy(float) - frame[ideal].to_numpy(float)
            r = np.sqrt((gap ** 2).sum(axis=1)).reshape(n, c)
            liking = frame["liking"].to_numpy(float).reshape(n, c)
            out.append(ballots(f"{name} (liking)", False, levels, r, liking))
    return out


def cses():
    """The left-right distance to each party as the respondent placed it, and the like
    or dislike, 0 to 10, grouped by the number of rated parties (3 to 9)."""
    parties = "ABCDEFGHI"
    columns = ["IMD3006"] + [f"IMD3007_{p}" for p in parties] + [f"IMD3008_{p}" for p in parties]
    with zipfile.ZipFile(fetch("cses_imd_csv.zip")) as archive, archive.open("cses_imd.csv") as file:
        frame = pd.read_csv(file, usecols=columns)
    me = frame["IMD3006"].to_numpy(float)
    placed = frame[[f"IMD3007_{p}" for p in parties]].to_numpy(float)
    liked = frame[[f"IMD3008_{p}" for p in parties]].to_numpy(float)
    rated = (me <= 10)[:, None] & (placed <= 10) & (liked <= 10)  # 95 to 99: missing
    out = Ballots("CSES (like-dislike)", False, 11, [], [])
    for count in range(3, len(parties) + 1):
        rows = rated.sum(axis=1) == count
        r, b = kept(np.abs(placed - me[:, None])[rows][rated[rows]].reshape(-1, count),
                    liked[rows][rated[rows]].reshape(-1, count), 11)
        out.r.append(r)
        out.b.append(b)
    return [out]


def psychology():
    k = pd.read_spss(fetch("kuhlmann2017/data.sav"), convert_categoricals=False)
    z = pd.read_csv(fetch("zhang_cesd/data.csv"))
    print(f"Kuhlmann et al.: {len(k)} people; Zhang et al.: {len(z)} people")
    items = [f"{scale}_{i:02d}" for scale, n in (("Gew", 12), ("Erl", 8), ("Nar", 6)) for i in range(1, n + 1)]
    out = [ballots("Kuhlmann Likert 1-5", False, 5, 101 - k[[f"VA_{i}" for i in items]].to_numpy(float),
                   k[[f"LI_{i}" for i in items]].to_numpy(float) - 1)]

    def items_of(form):
        return z[[f"cesd{form}{i}" for i in range(1, 9)]].to_numpy(float)

    for slider in "cd":
        out.append(ballots(f"Zhang slider {slider}, Likert 0-4", False, 5, 100 - items_of(slider), items_of("a")))
        out.append(ballots(f"Zhang slider {slider}, scale 0-14", False, 15, 100 - items_of(slider), items_of("b")))
    return out

# ---------------------------------------------------------------- the power of Score


def part_of_the_way(r):
    far, near = r.max(axis=1, keepdims=True), r.min(axis=1, keepdims=True)
    return (far - r) / (far - near)


def misses(data: Ballots, ballot):
    """Mean over the voters of the mean distance, in points, between the ballot ballot(r,
    levels) and the real one."""
    return np.mean(np.concatenate([np.abs(ballot(r, data.levels) - b).mean(axis=1)
                                   for r, b in zip(data.r, data.b)]))


def score_misses(data: Ballots, power):
    return misses(data, lambda r, levels: from_distances(r, levels, RANGE, power=power))


def power_of_score(sets):
    print(f"\n{'data':32s} {'L':>2s} {'voters':>7s}  points missed at p = 1, 1.5, 2   best p")
    table = {}
    for data in sets:
        missed = np.array([score_misses(data, p) for p in POWERS])
        table[data.name] = missed
        at = {p: missed[list(POWERS).index(p)] for p in UI}
        best = POWERS[missed.argmin()]
        print(f"{data.name:32s} {data.levels:2d} {data.voters:7d}  " + "  ".join(f"{at[p]:.3f}" for p in UI)
              + f"   {best:.2f} ({missed.min() / at[1.0] - 1:+.1%}; p = 1.5 {at[1.5] / at[1.0] - 1:+.1%})")
    for election in (True, False):
        names = [data.name for data in sets if data.election == election]
        change = np.array([[table[n][list(POWERS).index(p)] / table[n][list(POWERS).index(1.0)] - 1
                            for p in SHOWN] for n in names])
        print(f"\n{'elections' if election else 'ratings'}, {len(names)} ballot kinds: the change of the "
              f"points missed against p = 1")
        print("  p                " + " ".join(f"{p:6.1f}" for p in SHOWN))
        print("  mean             " + " ".join(f"{v:+6.1%}" for v in change.mean(axis=0)))
        print("  worst kind       " + " ".join(f"{v:+6.1%}" for v in change.max(axis=0)))
        print("  kinds better     " + " ".join(f"{v:6d}" for v in (change < 0).sum(axis=0)))
    return table


def where_voters_cut(data: Ballots):
    """Approval: the share of the decisions right when approving above a part of the way,
    and the cut of Score at each p."""
    r, y = data.r[0], data.b[0]
    part = part_of_the_way(r)
    lowest = np.where(y == 1, part, np.inf).min(axis=1)
    highest = np.where(y == 0, part, -np.inf).max(axis=1)
    print(f"\n{data.name}: approved per voter {y.sum(axis=1).mean():.2f}; part of the way of the farthest "
          f"approved: median {np.median(lowest):.2f}, of the closest not approved {np.median(highest):.2f}; "
          f"every approved closer than every other {np.mean(lowest > highest):.0%}")
    cuts = np.round(np.arange(0.5, 0.91, 0.05), 2)
    right = [np.mean((part > cut) == y) for cut in cuts]
    print("  approving above a part of the way: " + "  ".join(f"{c:.2f} {a:.1%}" for c, a in zip(cuts, right)))
    for p in UI:
        a = from_distances(r, 2, RANGE, power=p)
        print(f"  Score p = {p:g}: cut at {0.5 ** (1 / p):.2f}, approves {a.sum(axis=1).mean():.2f}, decisions "
              f"right {np.mean(a == y):.1%}, whole ballots {np.mean((a == y).all(axis=1)):.1%}")


def by_kind_of_ballot(table, sets):
    """The best single p of the approval ballots, of the scales from 0 and of those with a
    neutral 0 (-1/0/1, -1/0/1/2), of the elections."""
    groups = {"approval": [d for d in sets if d.election and d.levels == 2],
              "scales from 0": [d for d in sets if d.election and d.levels > 2 and "-1" not in d.name],
              "scales with a neutral 0": [d for d in sets if d.election and "-1" in d.name]}
    print()
    for label, group in groups.items():
        relative = np.mean([table[d.name] / table[d.name][list(POWERS).index(1.0)] for d in group], axis=0)
        print(f"{label} ({len(group)} kinds): best single p {POWERS[relative.argmin()]:.2f}, "
              f"{relative.min() - 1:+.1%}; at 1.5 {relative[list(POWERS).index(1.5)] - 1:+.1%}")


def halfway(sets):
    """The scales with a neutral middle and three levels: the candidates exactly halfway
    between the closest and the farthest (an opinion of 50 between 0 and 100), and the
    score Score gives them at p = 1, 1.5 and 2: halfway is the border of the middle
    score at p = 2, and np.round takes the half to the even score, 0."""
    print()
    for data in sets:
        if data.levels == 3 and "-1" in data.name:
            r, b = data.r[0], data.b[0]
            half = np.isclose(part_of_the_way(r), 0.5)
            print(f"{data.name}: candidates exactly halfway {half.mean():.1%}, of them given the middle score "
                  f"{np.mean(b[half] == 1):.0%}; points missed on them at p = 1, 1.5, 2: " + ", ".join(
                      f"{np.abs(from_distances(r, 3, RANGE, power=p) - b)[half].mean():.2f}" for p in UI))


def power_by_spread(sets, rng):
    """A p chosen per voter from where the other candidates are on the way (mean part,
    in quintiles), fitted on half the voters of each kind and tested on the other half,
    against one p fitted the same way and against p = 1.5."""
    fixed, adapted, default, own = [], [], [], []
    for data in sets:
        r, b = data.r[0], data.b[0]
        train = rng.random(len(r)) < 0.5
        part = part_of_the_way(r)
        errors = np.stack([np.abs(np.rint(part ** p * (data.levels - 1)) - b).mean(axis=1) for p in POWERS], 1)
        base = errors[~train, list(POWERS).index(1.0)].mean()
        best = errors[train].mean(axis=0).argmin()
        inner = np.sort(part, axis=1)[:, 1:-1].mean(axis=1)
        quintile = np.digitize(inner, np.quantile(inner[train], [0.2, 0.4, 0.6, 0.8]))
        chosen = np.array([errors[train & (quintile == q)].mean(axis=0).argmin() for q in range(5)])[quintile]
        fixed.append(errors[~train, best].mean() / base - 1)
        adapted.append(errors[~train][np.arange((~train).sum()), chosen[~train]].mean() / base - 1)
        default.append(errors[~train, list(POWERS).index(1.5)].mean() / base - 1)
        own.append(errors[~train].min(axis=1).mean() / base - 1)
    print(f"\np from the spread of the candidates, elections, on the voters not fitted: p = 1.5 "
          f"{np.mean(default):+.1%}, one p per kind {np.mean(fixed):+.1%}, p per quintile of the mean "
          f"part {np.mean(adapted):+.1%}, each voter's own best p {np.mean(own):+.1%} (against p = 1)")

# ---------------------------------------------------------------- other assumptions


def normalized(u):
    low, high = u.min(axis=1, keepdims=True), u.max(axis=1, keepdims=True)
    return (u - low) / (high - low)


# Other ways from distances to scores, one parameter shared by every kind of ballot:
# the utility u(r), scored as Score scores the part of the way, round(u (L - 1)) on u from
# 0 at the farthest to 1 at the closest.
UTILITIES = {
    "exp(-k (1 - part))": (lambda r, k: normalized(np.exp(-k * (1 - part_of_the_way(r)))),
                           np.round(np.arange(0.2, 4.01, 0.2), 2)),
    "exp(-(r / (s r_max))^2)": (lambda r, s: normalized(np.exp(-(r / (s * r.max(axis=1, keepdims=True))) ** 2)),
                                np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1, 2, 5])),
    "-log(r + c r_max)": (lambda r, c: normalized(-np.log(r + c * r.max(axis=1, keepdims=True))),
                          np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 3, 10, 100])),
    "-r^q": (lambda r, q: normalized(-(r ** q)), np.round(np.arange(0.1, 1.01, 0.05), 2)),
}


def other_utilities(table, sets):
    elections = [d for d in sets if d.election]
    base = {d.name: table[d.name][list(POWERS).index(1.0)] for d in elections}
    print("\nother utilities of the distance, one parameter for all elections: change against p = 1, "
          "mean (worst kind)")
    relative = np.mean([table[d.name] / base[d.name] for d in elections], axis=0)
    worst = np.max([table[d.name] / base[d.name] for d in elections], axis=0)
    k = relative.argmin()
    print(f"  {'part^p':26s} {relative[k] - 1:+.1%} ({worst[k] - 1:+.1%}) at {POWERS[k]:g}")
    for name, (utility, grid) in UTILITIES.items():
        changes = np.array([[misses(d, lambda r, levels, t=t: np.rint(utility(r, t) * (levels - 1)))
                             / base[d.name] - 1 for t in grid] for d in elections])
        k = changes.mean(axis=0).argmin()
        print(f"  {name:26s} {changes[:, k].mean():+.1%} ({changes[:, k].max():+.1%}) at {grid[k]:g}")


RULES = {"mean distance in the middle (AVG)": AVG, "largest gaps (DHONDT)": DHONDT,
         "gaps of the closer half (HYBRID)": HYBRID, "clusters kept (CLUSTER)": CLUSTER}


def other_rules(table, sets):
    """The other rules of score.py against Score with p = 1 and 1.5."""
    elections = [d for d in sets if d.election]
    base = {d.name: table[d.name][list(POWERS).index(1.0)] for d in elections}
    print("\nthe other rules of score.py, elections: change against p = 1, mean (worst kind, best kind)")
    rows = {f"Score, p = {p:g}": [table[d.name][list(POWERS).index(p)] / base[d.name] - 1 for d in elections]
            for p in (1.5,)}
    for label, rule in RULES.items():
        rows[label] = [misses(d, lambda r, levels, rule=rule: from_distances(r, levels, rule)) / base[d.name] - 1
                       for d in elections]
    at = rows["Score, p = 1.5"]
    for label, change in rows.items():
        beats = [d.name for d, a, c in zip(elections, at, change) if c < a]
        print(f"  {label:36s} {np.mean(change):+.1%} ({np.max(change):+.1%}, {np.min(change):+.1%})"
              + ("" if label.startswith("Score") else f"; better than p = 1.5 on {len(beats)}: {', '.join(beats)}"))


def squares(sets):
    """Score (p = 1) on the distances to an exponent: 2 would be the squared distances."""
    print("\ndistances to an exponent, ratings with a distance of their own: points missed")
    for data in sets:
        if data.name.startswith(("perfume", "cream", "CSES")):
            print(f"  {data.name:24s} " + "  ".join(
                f"{q:g}: {misses(data, lambda r, levels, q=q: from_distances(r ** q, levels)):.3f}"
                for q in (0.5, 0.75, 1, 2)))


if __name__ == "__main__":
    sets = (voter_autrement_2017() + voter_autrement_2022() + votare_altrimenti_2022() + ideal_profiles()
            + cses() + psychology())
    table = power_of_score(sets)
    where_voters_cut(sets[0])
    by_kind_of_ballot(table, sets)
    halfway(sets)
    power_by_spread([d for d in sets if d.election], np.random.default_rng(0))
    other_utilities(table, sets)
    other_rules(table, sets)
    squares(sets)
