"""
Аналіз PISA 2025: як освіта батьків, достаток і культурний капітал сім'ї
пов'язані з балом з математики в кожній країні.

Модель (для кожної країни окремо, зважена регресія з вагою W_FSTUWT):

    бал_математики = a
                     + b1[освіта]     (базова / середня / вища)
                     + b2[достаток]   (квартиль індексу WEALTH у країні: Q1..Q4)
                     + b3[культура]   (квартиль індексу CULTURE у країні: Q1..Q4)

Кожен фактор — категорії (dummy-змінні), базова категорія = найнижча,
тож a = прогноз для «базова освіта + найбідніші 25% + найнижчий культурний
капітал». Регресію рахуємо окремо для кожного з 10 plausible values
(PV1–PV10MATH) і усереднюємо коефіцієнти, як рекомендує методика PISA.

Обидва індекси будуємо самі (офіційний HOMEPOS не використовуємо, бо він
змішує гроші з книжками):

  WEALTH  (економічний капітал) — автомобілі (ST251Q01JA),
          кімнати з ванною (ST251Q03JA), кімнати з туалетом (ST251Q04JA);
  CULTURE (культурний капітал, П. Бурдьє) — книжки (ST255Q01JA),
          музичні інструменти (ST251Q06JA), твори мистецтва (ST251Q07JA).

Такий поділ видно в самих даних: інструменти й мистецтво корелюють
з книжками (0.28–0.30) сильніше, ніж з машинами та ваннами (0.15–0.22).
Індекс = середнє стандартизованих (z) значень; потрібно щонайменше 2 з 3
відповідей.

Додатково рахуємо «стандартизовану» модель
    бал = a + β1·z(PAREDINT) + β2·z(WEALTH) + β3·z(CULTURE)
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

WEALTH_ITEMS = ["ST251Q01JA", "ST251Q03JA", "ST251Q04JA"]
CULTURE_ITEMS = ["ST255Q01JA", "ST251Q06JA", "ST251Q07JA"]

# Q-коди — це частини країн (окремі регіони), а не країни цілком.
# Прив'язка до карти — припущення, перевірте у звіті PISA 2025.
MAP_CODE = {"TAP": "TWN", "KSV": "XKX", "QUA": "UKR", "QCI": "CHN"}


def build_index(d, items):
    """Середнє z-значень питань.

    Деякі країни не ставили частину питань (напр. у QCI немає питань про
    інструменти й мистецтво, у Нідерландах — про машини й туалети). Тоді
    індекс у цій країні рахуємо з тих питань, які там були, і вимагаємо
    щонайменше 2 відповіді (або 1, якщо в країні було лише одне питання).
    Повертає індекс і, для кожної країни, скільки питань у ньому використано."""
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
    """Номер зваженого квартиля (0..n-1) для кожного учня.
    Індекс має небагато різних значень (3 питання з кількома варіантами),
    тож багато учнів мають однакове значення. Щоб групи були рівно по 25%,
    таких учнів упорядковуємо випадково (фіксований seed — результат
    відтворюваний)."""
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
    """Коефіцієнти, усереднені по 10 plausible values, і R²."""
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

    # Категорійна модель: intercept, edu1, edu2, w1, w2, w3, c1, c2, c3
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

    # Та сама модель БЕЗ культурного капіталу: його не контролюємо, тож
    # частина його зв'язку з балами «переходить» до освіти й достатку.
    c2, r2_2 = fit(X[:, :6], Y, w)
    res["noculture"] = {
        "a": round(c2[0], 1),
        "edu": levels(c2, 1, 2),
        "wealth": levels(c2, 3, 3),
        "r2": r2_2,
    }

    # Частка учнів у кожній групі освіти (для попереджень на сайті)
    res["edu_share"] = [round(float(w[edu == k].sum() / w.sum()), 3) for k in range(3)]

    # Стандартизована модель: бали за +1 SD кожного фактора
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
        # Скільки з 3 питань увійшло в кожен індекс у цій країні
        r["items"] = {"wealth": wealth_used[cnt], "culture": culture_used[cnt]}
        if "a" in r:
            print(f"{cnt} {r['mean']:.0f}  достаток {r['wealth'][3]:+.0f}, освіта {r['edu'][2]:+.0f},"
                  f" культура {r['culture'][3]:+.0f}")
        else:
            print(f"{cnt} {r['mean']:.0f}  — замало даних для моделі")
        countries[cnt] = r

    payload = {
        "source": "OECD PISA 2025, учні; зважена регресія, 10 plausible values",
        "wealth_items": "автомобілі, кімнати з ванною, кімнати з туалетом",
        "culture_items": "книжки, музичні інструменти, твори мистецтва",
        "levels": {
            "edu": ["базова (до 9 класів)", "середня (школа / коледж)", "вища (університет)"],
            "wealth": ["найбідніші 25%", "нижче середнього", "вище середнього", "найбагатші 25%"],
            "culture": ["найнижчий (25%)", "нижче середнього", "вище середнього", "найвищий (25%)"],
        },
        "countries": countries,
    }
    full = [c for c in countries.values() if "a" in c]
    print(f"\nДостаток сильніший за освіту: у {sum(c['wealth'][3] > c['edu'][2] for c in full)}"
          f" з {len(full)} країн; без культурного капіталу в моделі — у "
          f"{sum(c['noculture']['wealth'][3] > c['noculture']['edu'][2] for c in full)}")
    print(f"Культура сильніша за достаток: у {sum(c['culture'][3] > c['wealth'][3] for c in full)}"
          f" з {len(full)}; за освіту: у {sum(c['culture'][3] > c['edu'][2] for c in full)}")
    OUT.write_text(
        "// Згенеровано analysis/analyze.py — не редагуйте вручну\n"
        "window.PISA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"\nЗбережено {OUT} ({OUT.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
