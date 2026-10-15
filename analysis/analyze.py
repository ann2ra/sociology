"""
Аналіз PISA 2025: як освіта батьків, достаток і книжки вдома
пов'язані з балом з математики в кожній країні.

Модель (для кожної країни окремо, зважена регресія з вагою W_FSTUWT):

    бал_математики = a
                     + b1[освіта]     (базова / середня / вища)
                     + b2[достаток]   (квартиль HOMEPOS у країні: Q1..Q4)
                     + b3[книжки]     (0–10 / 11–100 / понад 100)

Кожен фактор — категорії (dummy-змінні), базова категорія = найнижча,
тож a = прогноз для «базова освіта + найбідніші 25% + 0–10 книжок».
Регресію рахуємо окремо для кожного з 10 plausible values (PV1–PV10MATH)
і усереднюємо коефіцієнти — так рекомендує методика PISA.

Додатково рахуємо «стандартизовану» модель
    бал = a + β1·z(PAREDINT) + β2·z(HOMEPOS) + β3·z(книжки)
щоб порівняти, який фактор сильніший (бали за +1 стандартне відхилення).

Запуск:  python analysis/analyze.py
Результат: docs/data.js (його читає сайт)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "PISA2025_math_SES_clean.csv.gz"
OUT = ROOT / "docs" / "data.js"

PV = [f"PV{i}MATH" for i in range(1, 11)]
MIN_STUDENTS = 500  # менше учнів з повними даними — модель не рахуємо

# HISCED: 1 немає кваліфікації, 2 ISCED 2 → «базова»
#         3–5 ISCED 3–4 (старша школа, післясередня) → «середня»
#         6–9 ISCED 5–8 (вища)                       → «вища»
EDU_BINS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
# ST255Q01JA: 1–2 → 0–10 книжок, 3–4 → 11–100, 5–7 → понад 100
BOOK_BINS = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}

# Q-коди — це частини країн (окремі регіони), а не країни цілком.
# Прив'язка до карти — припущення, перевірте у звіті PISA 2025.
MAP_CODE = {"TAP": "TWN", "KSV": "XKX", "QUA": "UKR", "QCI": "CHN"}


def wquantile_bins(x, w, n=4):
    """Номер зваженого квартиля (0..n-1) для кожного учня."""
    order = np.argsort(x.values)
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


def country_model(g):
    res = {
        "n": int(len(g)),
        "mean": round(wmean(g[PV].mean(axis=1), g.W_FSTUWT), 1),
    }
    m = g.dropna(subset=["HISCED", "HOMEPOS", "ST255Q01JA", "PAREDINT"]).copy()
    res["n_model"] = int(len(m))
    if len(m) < MIN_STUDENTS:
        return res

    w = m.W_FSTUWT.values
    edu = m.HISCED.astype(int).map(EDU_BINS).values
    books = m.ST255Q01JA.astype(int).map(BOOK_BINS).values
    wealth = wquantile_bins(m.HOMEPOS, m.W_FSTUWT)
    Y = m[PV].values

    # Категорійна модель: intercept, edu1, edu2, w1, w2, w3, b1, b2
    X = np.column_stack([
        np.ones(len(m)),
        edu == 1, edu == 2,
        wealth == 1, wealth == 2, wealth == 3,
        books == 1, books == 2,
    ]).astype(float)
    coefs = np.array([wls(X, Y[:, j], w) for j in range(len(PV))])
    c = coefs.mean(axis=0)
    res["a"] = round(c[0], 1)
    res["edu"] = [0.0, round(c[1], 1), round(c[2], 1)]
    res["wealth"] = [0.0, round(c[3], 1), round(c[4], 1), round(c[5], 1)]
    res["books"] = [0.0, round(c[6], 1), round(c[7], 1)]
    res["r2"] = round(r2(X, Y, w, coefs), 3)

    # Та сама модель БЕЗ книжок: тоді «достаток» включає і книжки,
    # бо HOMEPOS сам рахує книжки як частину домашнього майна.
    X2 = X[:, :6]
    coefs2 = np.array([wls(X2, Y[:, j], w) for j in range(len(PV))])
    c2 = coefs2.mean(axis=0)
    res["nobooks"] = {
        "a": round(c2[0], 1),
        "edu": [0.0, round(c2[1], 1), round(c2[2], 1)],
        "wealth": [0.0, round(c2[3], 1), round(c2[4], 1), round(c2[5], 1)],
        "r2": round(r2(X2, Y, w, coefs2), 3),
    }

    # Частка учнів у кожній групі освіти / книжок (для підказок на сайті)
    res["edu_share"] = [round(float(w[edu == k].sum() / w.sum()), 3) for k in range(3)]
    res["books_share"] = [round(float(w[books == k].sum() / w.sum()), 3) for k in range(3)]

    # Стандартизована модель: бали за +1 SD кожного фактора
    def z(v):
        mu = np.average(v, weights=w)
        sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
        return (v - mu) / sd

    Xs = np.column_stack([
        np.ones(len(m)), z(m.PAREDINT.values), z(m.HOMEPOS.values),
        z(m.ST255Q01JA.values.astype(float)),
    ])
    cs = np.array([wls(Xs, Y[:, j], w) for j in range(len(PV))]).mean(axis=0)
    res["std"] = {"edu": round(cs[1], 1), "wealth": round(cs[2], 1), "books": round(cs[3], 1)}
    return res


def main():
    cols = ["CNT", "W_FSTUWT", "HISCED", "PAREDINT", "ST255Q01JA", "HOMEPOS"] + PV
    d = pd.read_csv(SRC, usecols=cols)
    countries = {}
    for cnt, g in d.groupby("CNT"):
        r = country_model(g)
        print(cnt, r["mean"], "" if "a" in r else "— замало даних для моделі")
        r["map"] = MAP_CODE.get(cnt, cnt)
        if "a" in r:
            print(f"  із книжками: достаток {r['wealth'][3]:+.0f}, освіта {r['edu'][2]:+.0f};"
                  f" без книжок: достаток {r['nobooks']['wealth'][3]:+.0f}, освіта {r['nobooks']['edu'][2]:+.0f}")
        countries[cnt] = r

    payload = {
        "source": "OECD PISA 2025, учні; зважена регресія, 10 plausible values",
        "levels": {
            "edu": ["базова (до 9 класів)", "середня (школа / коледж)", "вища (університет)"],
            "wealth": ["найбідніші 25%", "нижче середнього", "вище середнього", "найбагатші 25%"],
            "books": ["0–10 книжок", "11–100 книжок", "понад 100 книжок"],
        },
        "countries": countries,
    }
    full = [c for c in countries.values() if "a" in c]
    print(f"\nДостаток сильніший за освіту: із книжками у "
          f"{sum(c['wealth'][3] > c['edu'][2] for c in full)} з {len(full)} країн, без книжок у "
          f"{sum(c['nobooks']['wealth'][3] > c['nobooks']['edu'][2] for c in full)} з {len(full)}")
    OUT.write_text(
        "// Згенеровано analysis/analyze.py — не редагуйте вручну\n"
        "window.PISA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"\nЗбережено {OUT} ({OUT.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
