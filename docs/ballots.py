"""The score rules of score.py against real ballots: the numbers of math.typ
(@sec-score-data).

Run from the repository root:

    uv run --with pandas --with rdata python docs/ballots.py

The first run downloads the public data sets into docs/data/ (not in git):

    Voter Autrement 2017, online experiment (Zenodo record 1199545, ODbL): approval
        ballots and 0-100 opinions of the 11 candidates of the French presidential
        election, from the same participants
    SensoMineR 1.28 (CRAN): perfume_ideal and cream_id, consumers who rate products and
        give the perceived and the ideal intensity of each attribute
    CSES Integrated Module Dataset (cses.org, doi:10.7804/cses.imd.2024-02-27): the
        left-right placement of the respondent and of up to nine parties, and the like
        or dislike of each party

A voter's distance to a candidate is 100 minus the opinion of Voter Autrement, the
distance between the perceived and the ideal profile of a product, or the distance on
the left-right scale. Every rule depends on the distances only up to a common scale and
shift, so any such distance gives the same ballot. Prints the numbers quoted in the text.
"""

import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

from yeelab.score import AVG, DELTA, DHONDT, HYBRID, RANGE, from_distances

DATA = Path(__file__).parent / "data"
VOTER_AUTREMENT = "https://zenodo.org/records/1199545/files/merged.csv?download=1"
SENSOMINER = "https://cran.r-project.org/src/contrib/Archive/SensoMineR/SensoMineR_1.28.tar.gz"
CSES = "https://cses.org/wp-content/uploads/2024/02/cses_imd_csv.zip"
RULES = {"range": RANGE, "avg": AVG, "dhondt": DHONDT, "hybrid": HYBRID}


def fetch(url, name):
    """The file at `url`, downloaded once into DATA as `name`."""
    path = DATA / name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


def ballots(r, levels):
    """The ballot of each rule for the distances r (n, C), as a part of the top score."""
    return {name: from_distances(r, levels, rule, DELTA) / (levels - 1) for name, rule in RULES.items()}

# ---------------------------------------------------------------- approval


def voter_autrement():
    """Approvals y and opinions u, (n, 11), of the participants who had an approval
    ballot and gave every candidate an opinion; without the empty and the full ballots,
    which no rule casts, and the opinions all alike."""
    merged = pd.read_csv(fetch(VOTER_AUTREMENT, "voter_autrement_2017_online.csv"), comment="#",
                         low_memory=False)
    names = [column[len("Approval "):] for column in merged.columns[1:12]]
    y = merged[[f"Approval {name}" for name in names]].apply(pd.to_numeric, errors="coerce")
    u = merged[[f"Evaluation Continuous {name}" for name in names]].apply(pd.to_numeric, errors="coerce")
    both = y.notna().all(axis=1) & u.notna().all(axis=1)
    y, u = y[both].to_numpy(int), u[both].to_numpy(float)
    kept = (y.sum(axis=1) > 0) & (y.sum(axis=1) < len(names)) & (np.ptp(u, axis=1) > 0)
    print(f"Voter Autrement: {len(merged)} participants, {both.sum()} with an approval ballot and every "
          f"opinion; empty ballots {np.mean(y.sum(axis=1) == 0):.1%}, full {np.mean(y.sum(axis=1) == 11):.1%}, "
          f"kept {kept.sum()}")
    official = merged["official vote"][both][kept].value_counts(normalize=True)
    print("  their official first-round vote: " + "  ".join(f"{name} {share:.0%}" for name, share in official[:4].items()))
    return y[kept], u[kept]


def approval():
    y, u = voter_autrement()
    r = 100 - u
    print(f"approved per voter {y.sum(axis=1).mean():.2f} (median {np.median(y.sum(axis=1)):.0f})")
    approve = {name: ballot.astype(int) for name, ballot in ballots(r, 2).items()}
    for name, a in approve.items():
        wrong = a != y
        print(f"  {name:7s} approves {a.sum(axis=1).mean():.2f}  correct {1 - wrong.mean():.1%}  "
              f"whole ballot {(~wrong).all(axis=1).mean():.1%}  too many {(a > y).sum(axis=1).mean():.2f}  "
              f"too few {(a < y).sum(axis=1).mean():.2f}")
    # where DHONDT cuts: its gap starts closer than the midrange, or not
    s = np.sort(r, axis=1)
    mid = (s[:, :1] + s[:, -1:]) / 2
    start = s[np.arange(len(s)), np.diff(s, axis=1).argmax(axis=1)]
    far = start >= mid[:, 0]
    for label, group in (("largest gap starts closer than the midrange", ~far),
                         ("largest gap starts at or beyond it", far)):
        print(f"  {label}: {group.mean():.0%} of the voters, approved {y[group].sum(axis=1).mean():.2f}")
        for name in ("range", "dhondt", "hybrid"):
            a = approve[name][group]
            print(f"    {name:7s} approves {a.sum(axis=1).mean():.2f}  correct {(a == y[group]).mean():.1%}  "
                  f"whole ballot {(a == y[group]).all(axis=1).mean():.1%}")
    # where the voters cut, as a part of the way of RANGE
    part = (s[:, -1:] - r) / (s[:, -1:] - s[:, :1])
    lowest = np.where(y == 1, part, np.inf).min(axis=1)
    highest = np.where(y == 0, part, -np.inf).max(axis=1)
    print(f"  part of the way of the farthest approved: median {np.median(lowest):.2f}; of the closest "
          f"not approved: median {np.median(highest):.2f}; every approved closer than every other: "
          f"{np.mean(lowest > highest):.0%}")
    cuts = np.round(np.arange(0.5, 0.91, 0.05), 2)
    correct = [np.mean((part > cut) == y) for cut in cuts]
    best = int(np.argmax(correct))
    print("  approving above a part of the way: " + "  ".join(f"{c:.2f} {a:.1%}" for c, a in zip(cuts, correct))
          + f"; best {cuts[best]:.2f}, approving {(part > cuts[best]).sum(axis=1).mean():.2f}")

# ---------------------------------------------------------------- graded


def ideal_profiles():
    """(name, distances (n, products), ratings (n, products), levels) of the two data
    sets of the ideal profile method: the distance between the perceived and the ideal
    profile, both given with each product."""
    archive = fetch(SENSOMINER, "SensoMineR_1.28.tar.gz")
    out = []
    with tarfile.open(archive) as tar:
        for name, person, product, levels in (("perfume_ideal", "user", "product", 9),
                                              ("cream_id", "juge", "produit", 11)):
            raw = tar.extractfile(f"SensoMineR/data/{name}.rda").read()
            frame = rdata.read_rda(io.BytesIO(raw))[name].sort_values([person, product])
            ideal = [column for column in frame.columns if str(column).startswith("id_")]
            perceived = [frame.columns[frame.columns.get_loc(column) - 1] for column in ideal]
            n, c = frame[person].nunique(), frame[product].nunique()
            gap = frame[perceived].to_numpy(float) - frame[ideal].to_numpy(float)
            r = np.sqrt((gap ** 2).sum(axis=1)).reshape(n, c)
            out.append((name, r, frame["liking"].to_numpy(float).reshape(n, c), levels,
                        list(frame[product].iloc[:c])))
    return out


def cses():
    """[(distances, ratings)] of the CSES respondents, grouped by their number of rated
    parties (3 to 9): the left-right distance to each party as the respondent placed it,
    and the like or dislike, 0 to 10."""
    parties = "ABCDEFGHI"
    columns = ["IMD3006"] + [f"IMD3007_{p}" for p in parties] + [f"IMD3008_{p}" for p in parties]
    with zipfile.ZipFile(fetch(CSES, "cses_imd_csv.zip")) as archive:
        with archive.open("cses_imd.csv") as file:
            frame = pd.read_csv(file, usecols=columns)
    me = frame["IMD3006"].to_numpy(float)
    placed = frame[[f"IMD3007_{p}" for p in parties]].to_numpy(float)
    liked = frame[[f"IMD3008_{p}" for p in parties]].to_numpy(float)
    rated = (me <= 10)[:, None] & (placed <= 10) & (liked <= 10)  # 95 to 99: missing
    groups = []
    for count in range(3, len(parties) + 1):
        rows = rated.sum(axis=1) == count
        r = np.abs(placed - me[:, None])[rows][rated[rows]].reshape(-1, count)
        groups.append((r, liked[rows][rated[rows]].reshape(-1, count)))
    print(f"CSES: {len(frame)} respondents, {sum(len(r) for r, _ in groups)} with 3 or more rated parties")
    return groups


def normalized(groups):
    """Only the voters with distances and ratings not all alike, the ratings of each
    from 0 at the lowest to 1 at the highest."""
    out = []
    for r, rating in groups:
        kept = (np.ptp(r, axis=1) > 0) & (np.ptp(rating, axis=1) > 0)
        r, rating = r[kept], rating[kept]
        low = rating.min(axis=1, keepdims=True)
        out.append((r, (rating - low) / (rating.max(axis=1, keepdims=True) - low)))
    return out


def errors(groups, levels, power=1.0):
    """Mean over the voters of the mean distance between each rule's ballot and the
    normalized ratings, as a part of the scale; distances to the power `power`."""
    per_voter = {name: [] for name in RULES}
    for r, rating in groups:
        for name, ballot in ballots(r ** power, levels).items():
            per_voter[name].append(np.abs(ballot - rating).mean(axis=1))
    return {name: np.concatenate(values).mean() for name, values in per_voter.items()}


def graded():
    sets = []
    for name, r, rating, levels, products in ideal_profiles():
        print(f"{name}: {r.shape[0]} consumers, {r.shape[1]} products, ratings {rating.min():.0f} to "
              f"{rating.max():.0f}; top rating used by {np.mean(rating.max(axis=1) == rating.max()):.0%}, "
              f"bottom by {np.mean(rating.min(axis=1) == rating.min()):.0%}")
        groups = normalized([(r, rating)])
        if name == "perfume_ideal":
            _, normal = groups[0]
            twice = [np.abs(normal[:, products.index(a)] - normal[:, products.index(a + "2")]).mean()
                     for a in ("PurePoison", "Shalimar")]
            print(f"  the same perfume twice: {np.mean(twice) * (levels - 1):.2f} points apart on average")
        sets.append((name, groups, levels))
    sets.append(("cses", normalized(cses()), 11))
    for name, groups, levels in sets:
        points = levels - 1
        print(f"{name}: {sum(len(r) for r, _ in groups)} voters; mean error in points of the scale (0 to {points}): "
              + "  ".join(f"{rule} {error * points:.2f}" for rule, error in errors(groups, levels).items()))
        print("  distances to the power: " + "  ".join(
            f"{power} {errors(groups, levels, power)['range'] * points:.2f}" for power in (0.5, 0.75, 1.0, 2.0))
            + " (range)")


if __name__ == "__main__":
    approval()
    graded()
