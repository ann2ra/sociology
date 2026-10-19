"""
PISA 2025 analysis: how parental education, family wealth and cultural
capital relate to students' maths scores in each country.

Model (a separate weighted regression for each country, weight W_FSTUWT):

    maths_score = a
                  + b1[education]  (basic / secondary / higher)
                  + b2[wealth]     (within-country quartile of the WEALTH index: Q1..Q4)
                  + b3[culture]    (within-country quartile of the CULTURE index: Q1..Q4)

Each factor is categorical (dummy variables) with the lowest category as the
baseline, so a = prediction for "basic education + poorest 25% + lowest
cultural capital". The regression is run separately for each of the 10
plausible values (PV1-PV10MATH) and the coefficients are averaged, as PISA
methodology recommends.

We build both indices ourselves (the official HOMEPOS is not used because it
mixes money with books):

  WEALTH  (economic capital) - cars (ST251Q01JA),
          rooms with a bath or shower (ST251Q03JA), rooms with a toilet (ST251Q04JA);
  CULTURE (cultural capital, after P. Bourdieu) - books (ST255Q01JA),
          musical instruments (ST251Q06JA), works of art (ST251Q07JA).

The split is visible in the data itself: instruments and art correlate more
with books (0.28-0.30) than with cars and bathrooms (0.15-0.22).
Index = mean of standardised (z) values; at least 2 of 3 answers are required.

We also fit a "standardised" model
    score = a + β1·z(PAREDINT) + β2·z(WEALTH) + β3·z(CULTURE)
to compare which factor is stronger (points per +1 standard deviation).

Run:     python analysis/analyze.py
Output:  docs/data.js (read by the website)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "PISA2025_math_SES_clean.csv.gz"
OUT = ROOT / "docs" / "data.js"

PV = [f"PV{i}MATH" for i in range(1, 11)]
MIN_STUDENTS = 500  # skip the model if fewer students have complete data

# HISCED: 1 no qualification, 2 ISCED 2                      -> "basic"
#         3-5 ISCED 3-4 (upper secondary, post-secondary)  -> "secondary"
#         6-9 ISCED 5-8 (tertiary)                         -> "higher"
EDU_BINS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}

WEALTH_ITEMS = ["ST251Q01JA", "ST251Q03JA", "ST251Q04JA"]
CULTURE_ITEMS = ["ST255Q01JA", "ST251Q06JA", "ST251Q07JA"]

# Q-codes are parts of countries (selected regions), not whole countries.
# Mapping them onto the map is an assumption; check the PISA 2025 report.
MAP_CODE = {"TAP": "TWN", "KSV": "XKX", "QUA": "UKR", "QCI": "CHN"}


def build_index(d, items):
    """Mean of the items' z-values.

    Some countries did not ask all the questions (e.g. QCI has no questions
    on instruments and art, the Netherlands none on cars and toilets). There
    the index is built from the questions that were asked, and at least 2
    answers are required (or 1 if the country asked only one question).
    Returns the index and, for each country, how many items it used."""
    z = pd.DataFrame(index=d.index)
    for c in items:
        ok = d[c].notna()
        mu = np.average(d.loc[ok, c], weights=d.W_FSTUWT[ok])
        sd = np.sqrt(np.average((d.loc[ok, c] - mu) ** 2, weights=d.W_FSTUWT[ok]))
        z[c] = (d[c] - mu) / sd
    index = pd.Series(np.nan, index=d.index)
    used = {}
    for cnt, rows in d.groupby("CNT").groups.items():
        zc = z.loc[rows]
        avail = [c for c in items if zc[c].notna().mean() > 0.5]
        used[cnt] = len(avail)
        if not avail:
            continue
        idx = zc[avail].mean(axis=1)
        idx[zc[avail].notna().sum(axis=1) < min(2, len(avail))] = np.nan
        index.loc[rows] = idx
    return index, used


def wquantile_bins(x, w, n=4, seed=2025):
    """Weighted quartile number (0..n-1) for each student.
    The index has few distinct values (3 questions with a few options each),
    so many students share the same value. To make each group exactly 25%,
    such students are ordered at random (fixed seed, so the result is
    reproducible)."""
    tie = np.random.default_rng(seed).random(len(x))
    order = np.lexsort((tie, x.values))
    cw = np.cumsum(w.values[order])
    cw = cw / cw[-1]
    bins = np.empty(len(x), dtype=int)
    bins[order] = np.minimum((cw * n - 1e-9).astype(int), n - 1)
    return bins


def wls(X, y, w):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def wmean(x, w):
    return float(np.average(x, weights=w))


def r2(X, Y, w, coefs):
    out = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        pred = X @ coefs[j]
        ybar = np.average(y, weights=w)
        out.append(1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - ybar) ** 2))
    return float(np.mean(out))


def fit(X, Y, w):
    """Coefficients averaged over the 10 plausible values, and R²."""
    coefs = np.array([wls(X, Y[:, j], w) for j in range(Y.shape[1])])
    return coefs.mean(axis=0), round(r2(X, Y, w, coefs), 3)


def levels(c, start, k):
    return [0.0] + [round(v, 1) for v in c[start:start + k]]


def country_model(g):
    res = {
        "n": int(len(g)),
        "mean": round(wmean(g[PV].mean(axis=1), g.W_FSTUWT), 1),
    }
    m = g.dropna(subset=["HISCED", "PAREDINT", "WEALTH", "CULTURE"]).copy()
    res["n_model"] = int(len(m))
    if len(m) < MIN_STUDENTS:
        return res

    w = m.W_FSTUWT.values
    edu = m.HISCED.astype(int).map(EDU_BINS).values
    wealth = wquantile_bins(m.WEALTH, m.W_FSTUWT)
    culture = wquantile_bins(m.CULTURE, m.W_FSTUWT)
    Y = m[PV].values

    # Categorical model: intercept, edu1, edu2, w1, w2, w3, c1, c2, c3
    X = np.column_stack([
        np.ones(len(m)),
        edu == 1, edu == 2,
        wealth == 1, wealth == 2, wealth == 3,
        culture == 1, culture == 2, culture == 3,
    ]).astype(float)
    c, res["r2"] = fit(X, Y, w)
    res["a"] = round(c[0], 1)
    res["edu"] = levels(c, 1, 2)
    res["wealth"] = levels(c, 3, 3)
    res["culture"] = levels(c, 6, 3)

    # The same model WITHOUT cultural capital: it is not controlled for, so
    # part of its association with scores "moves" to education and wealth.
    c2, r2_2 = fit(X[:, :6], Y, w)
    res["noculture"] = {
        "a": round(c2[0], 1),
        "edu": levels(c2, 1, 2),
        "wealth": levels(c2, 3, 3),
        "r2": r2_2,
    }

    # Share of students in each education group (for warnings on the site)
    res["edu_share"] = [round(float(w[edu == k].sum() / w.sum()), 3) for k in range(3)]

    # Standardised model: points per +1 SD of each factor
    def z(v):
        mu = np.average(v, weights=w)
        sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
        return (v - mu) / sd

    Xs = np.column_stack([
        np.ones(len(m)), z(m.PAREDINT.values), z(m.WEALTH.values), z(m.CULTURE.values),
    ])
    cs, _ = fit(Xs, Y, w)
    res["std"] = {"edu": round(cs[1], 1), "wealth": round(cs[2], 1), "culture": round(cs[3], 1)}
    return res


def main():
    cols = ["CNT", "W_FSTUWT", "HISCED", "PAREDINT"] + WEALTH_ITEMS + CULTURE_ITEMS + PV
    d = pd.read_csv(SRC, usecols=cols)
    d["WEALTH"], wealth_used = build_index(d, WEALTH_ITEMS)
    d["CULTURE"], culture_used = build_index(d, CULTURE_ITEMS)

    countries = {}
    for cnt, g in d.groupby("CNT"):
        r = country_model(g)
        r["map"] = MAP_CODE.get(cnt, cnt)
        # How many of the 3 items went into each index in this country
        r["items"] = {"wealth": wealth_used[cnt], "culture": culture_used[cnt]}
        if "a" in r:
            print(f"{cnt} {r['mean']:.0f}  wealth {r['wealth'][3]:+.0f}, education {r['edu'][2]:+.0f},"
                  f" culture {r['culture'][3]:+.0f}")
        else:
            print(f"{cnt} {r['mean']:.0f}  - not enough data for the model")
        countries[cnt] = r

    payload = {
        "source": "OECD PISA 2025 students; weighted regression, 10 plausible values",
        "wealth_items": "cars, rooms with a bath or shower, rooms with a toilet",
        "culture_items": "books, musical instruments, works of art",
        "levels": {
            "edu": ["basic (up to grade 9)", "secondary (school / college)", "higher (university)"],
            "wealth": ["poorest 25%", "below average", "above average", "richest 25%"],
            "culture": ["lowest 25%", "below average", "above average", "highest 25%"],
        },
        "countries": countries,
    }
    full = [c for c in countries.values() if "a" in c]
    print(f"\nWealth stronger than education: {sum(c['wealth'][3] > c['edu'][2] for c in full)}"
          f" of {len(full)} countries; without cultural capital in the model: "
          f"{sum(c['noculture']['wealth'][3] > c['noculture']['edu'][2] for c in full)}")
    print(f"Culture stronger than wealth: {sum(c['culture'][3] > c['wealth'][3] for c in full)}"
          f" of {len(full)}; than education: {sum(c['culture'][3] > c['edu'][2] for c in full)}")
    OUT.write_text(
        "// Generated by analysis/analyze.py - do not edit by hand\n"
        "window.PISA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"\nSaved {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
