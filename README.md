# PISA 2025: cleaned data (maths + socio-economic status)

Source: OECD PISA 2025, student file `CY09_MS_STU_PUF.sav`.

- `PISA2025_math_SES_clean.sav` — SPSS, with variable names and value labels
- `PISA2025_math_SES_clean.csv.gz` — the same dataset as CSV (gzip-compressed)

747,068 students, 89 countries, 25 columns.

| Column | Description |
|---|---|
| CNTSTUID | Student ID (no duplicates) |
| CNT | Country code |
| PV1MATH … PV10MATH | 10 plausible values of the maths score |
| W_FSTUWT | Final student weight |
| HISCED | Highest education level of the parents (ISCED, 1–9) |
| P1ISCED, P2ISCED | Education of parent/guardian 1 and 2 (replace MISCED/FISCED in PISA 2025) |
| PAREDINT | Parental education in years of schooling |
| ST255Q01JA | Number of books at home (1–7) |
| ST251Q01JA | Cars, vans, trucks (1 = none, 2 = one, 3 = two, 4 = three or more) |
| ST251Q03JA | Rooms with a bath or shower (1–4) |
| ST251Q04JA | Rooms with a toilet (1–4) |
| ST251Q06JA | Musical instruments (1–4) |
| ST251Q07JA | Works of art (1–4) |
| HOMEPOS | Home possessions index (WLE); PISA 2025 has no WEALTH index, and HOMEPOS includes books |
| ESCS | Index of economic, social and cultural status |

## Cleaning
- Missing-value codes (95–99, 9997; 5–9 for ST251) were replaced with empty values.
- ST251Q02JA and ST251Q05JA are not in PISA 2025.
- 8,653 students without maths scores were removed.
- Students with incomplete family data were kept.

---

# Project: "Money or Education?"

**Website:** https://ann2ra.github.io/sociology/

An interactive map: choose parental education, the family's economic capital (cars, bathrooms, toilets) and cultural capital (books, musical instruments, works of art), and see a child's predicted PISA 2025 maths score in each country.

## Structure

| File | What it is |
|---|---|
| `analysis/analyze.py` | Analysis script: weighted regressions `score = a + b1·education + b2·wealth + b3·culture` for each country (10 plausible values) |
| `docs/data.js` | Analysis output (coefficients for all countries), read by the website |
| `docs/index.html` | The website: map, sliders, heatmap, findings, solutions, methods |
| `English_version_1.pdf` | The group's report |

## How to re-run the analysis

```bash
pip install pandas numpy
python analysis/analyze.py
```

## How to view the site locally

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

## How to enable GitHub Pages (once)

Settings → Pages → Build and deployment → Source: **Deploy from a branch** →
Branch: `main`, folder `/docs` → Save. The site appears at the link above within 1–2 minutes.
