# FIXES.md

What was broken in this repository, what I changed, which claims I retracted,
and which numbers moved and why.

This file is for you to read once and then delete. The parts worth keeping are
already folded into the README's Validation and Limitations sections.

---

## 1. The target variable was not underpricing

This is the whole story. Everything else in this document is secondary.

`src/scraper_ipo_calendar.py` mapped stockanalysis.com's `Return` column to
`first_day_return_pct`. The cached HTML header reads:

```
[IPO Date, Symbol, Company Name, IPO Price, Current, Return]
```

`Return` is the return from the offer price to the price **on the day the page
was scraped** (around April 2025), not to the first-day close. The project's
target was a multi-year holding-period return labelled as underpricing.

| Ticker | Repo's "underpricing" | Actual first-day return |
|---|---:|---:|
| PLTR | +1873.7% | +31.0% |
| DASH | +73.3% | +85.8% |
| ABNB | +110.0% | +112.8% |

The sample-level tell was visible in the repo's own data: median −4.5%, 25th
percentile −68.2%, minimum exactly −100%. No IPO sample has that shape.

The `collapse_floor = -0.50` filter inside `test_h1_litigious_paradox`, with
its comment about "the upstream stockanalysis.com price field returning a
*current* price for delisted tickers", was half a diagnosis. The mechanism was
identified correctly and then treated as an outlier problem affecting some
rows, rather than as the definition of the column affecting all of them.

**Every figure, hypothesis test and headline number in the old README was
computed on this variable.**

### What I replaced it with

`(first_day_close − offer_price) / offer_price`, from Yahoo Finance, needing
three corrections the old price scraper did not make:

1. **Yahoo's OHLC is split-adjusted even with `auto_adjust=False`.** That flag
   only disables the dividend adjustment. NVDA's 2024-01-02 close is reported
   as $48.17 against an actual $481.68. For IPOs this matters constantly:
   CrowdStrike's first-day close reports as $14.50 against $58.00.
2. **Yahoo lists a split before it restates the series.** NeuroSense carries a
   1-for-20 dated three days before this build with un-restated prices;
   applying the factor turned a −32.7% return into −96.6%.
   `split_is_applied()` tests whether the price series actually steps at the
   split date and drops splits that have not landed.
3. **Tickers get reused.** ASPL's history starts in 2007 at a flat $0.34 and
   belongs to a different issuer than the 2020 SPAC that listed under it.

Validated against eight IPOs with publicly known first-day returns (ABNB,
DASH, CRWD, RBLX, PLTR, LYFT, UBER, SNOW), all within 0.1pp. Committed as
`tests/test_prices.py`.

---

## 2. Claims retracted

| Old claim | Status |
|---|---|
| "H1 — Litigious paradox: **Reject H0** (ρ = −0.206, p = 0.0003)" | **Retracted.** ρ = −0.047, p = 0.31, n = 474. Fails to reject, and the interval spans zero in every one of 14 robustness specifications. |
| "The bars step *down* monotonically from +48.3% (Q1) to +6.0% (Q5)" | **Retracted.** Quintile medians are +14.3%, +16.8%, +15.0%, +11.1%, +5.1%, not monotone, and the bootstrap intervals overlap completely. |
| "Litigious is the *only* LM category with a significant negative ρ" | **Retracted.** No LM category reaches significance. The largest \|ρ\| across all seven is 0.089 (Positive, p = 0.054, not significant after correction). |
| "the only LM category whose Spearman p-value survives a Bonferroni correction" | **Retracted.** Nothing survives Bonferroni; the smallest corrected p across the seven categories is 0.379. |
| "H3 — ρ = −0.080, p = 0.21, directionally consistent" | **Retracted.** ρ = −0.012, p = 0.81. Not even directionally informative. |
| "H4 — VIX variance: **Reject H0** (W = 4.18, p = 0.015)" | **Changed.** Levene now gives W = 2.39, p = 0.093 — but the rank-based Fligner–Killeen gives p = 4.7e−09 with IQR rising 35.9% → 34.9% → 56.4% across VIX terciles. The conclusion survives on a robust test; the original statistic does not. |
| "H6 — Text features: **Reject H0** (χ²(7) = 14.18, p = 0.048, ΔBIC = −29.1)" | **Retracted.** χ²(7) = 7.30, p = 0.40. Adjusted R² is 0.023 with and without text features. |
| "H2 / H5 — **Skipped** (Ritter rank merge unpopulated)" | **Repaired and now reported.** See §4. H2 fails to reject. H5 rejects on Levene but not on a robust test. |
| "Every IPO is mapped to a real GICS sector — no `Unknown` / `Other` buckets" | **Retracted as misleading.** True only because the classifier forced 29.2% of rows into Industrials with no matching rule. |
| "Industrials = 494" presented as a finding | **Retracted.** 96.3% of rows labelled Industrials got there via the silent fallback. |
| "`data/external/underwriter_ranks.csv` — Carter-Manaster ranks (Ritter)" | **Retracted.** It was a 55-row hand-assigned tier list, not Ritter's file. Replaced with the real `Underwriter-Rank.xls`, SHA-256 pinned. |
| "1,655 IPOs" as the analysis sample | **Corrected.** 1,773 calendar entries; 709 have a computable first-day return; 474 also have a prospectus. |
| "text signals are individually weak but add stabilising lift when combined non-linearly by the model" | **Retracted.** No model beats a constant out-of-sample, so there is no lift to attribute. |

---

## 3. Leakage found and removed

| Where | What it was |
|---|---|
| `add_market_features` | VIX and NASDAQ read with `index <= ipo_date`, i.e. the listing day's own close. Now the last trading day strictly before, with `market_data_date` recorded so the lag is auditable. |
| `add_market_features` | Hot-market dummy thresholded the trailing issue count against the **full-sample** tercile. Now an expanding-window quantile. |
| `compute_prospectus_uniqueness` | Each filing compared against a sector mean over the **whole corpus**, including filings that did not exist yet. Now only earlier same-sector filings. |
| `encode_categoricals` | `sector_encoded` and `lead_underwriter_encoded` were the per-category mean of the target over the full dataset. Fed to the model **and** used as a control in the H1 regression, where conditioning on a function of the dependent variable distorts every other coefficient. Removed; sector is now fixed effects fitted inside each fold. |
| `winsorise_target` | Full-sample 1st/99th percentiles. Winsorisation now happens inside the fold, from training-fold limits only, and only in the secondary model table. |

One caveat I could not remove: the SEC submissions API returns the
registrant's *current* SIC assignment, not the one in force at the IPO. It is
in the README's leakage audit.

---

## 4. H2 and H5 repaired

The old README said the Ritter merge was "unpopulated". The actual position
was worse: **there is no underwriter column anywhere in the data.** The
calendar scrape never captured one, so no amount of name normalisation could
have fixed the merge.

Both tests are now live, from a different source:

- The syndicate is read off the S-1 cover page and matched through a committed
  alias table, no fuzzy matching.
- Two false-positive sources needed handling: the "Copies to:" block names
  outside counsel by personal name (*Michael Benjamin* of Latham & Watkins
  matched Benjamin Securities; *Morrison Cohen LLP* matched Cohen & Co), and
  some issuers are themselves brokers (Futu Holdings matched Futu Securities).
- Ranks come from Ritter's actual `Underwriter-Rank.xls`, reshaped to one row
  per underwriter-year because his ranks are period-specific.

**Match rate: 415 of 474 filings (87.6%) name a recognised underwriter; 405
(85.4%) resolve to a Ritter rank.** A hand audit of 12 random filings found
the extracted lead correct in every case where the saved text contained a
cover page (11 of 12; the twelfth had captured a signature page instead).

---

## 5. Additional findings not in the brief

- **The corrected sample contains zero SPACs.** All 580 name-matched SPACs
  fail the price fetch, because Yahoo purges the shell ticker after a de-SPAC
  or liquidation. The brief asked for an ex-SPAC robustness check on H1; the
  check is vacuous because the sample is already entirely ex-SPAC. This is a
  large hole and is stated prominently in Limitations.
- **`prospectus_uniqueness` correlates −0.97 with log document length.** The
  Hanley–Hoberg implementation is close to a length measure, which is a flaw
  in the feature rather than a finding.
- **The LM loader would have silently produced empty modal categories.** It
  looked for `StrongModal`/`WeakModal`; the master dictionary spells them
  `Strong_Modal`/`Weak_Modal`. It happened to work only because the committed
  copy used the older single `Modal` column.
- **`sector_for_sic('2834.0')` returned `Unclassified`.** A SIC read from a
  pandas column containing nulls arrives as a float string, which `int()`
  rejects. Found by a test written for this review.
- **`tokenise`'s docstring example was wrong.** It claimed `"firm's"` survives
  intact; the apostrophe is stripped and the token splits into `firm`, `s`.
- **`mean_squared_error(squared=False)` in the old `evaluate()`** was removed
  in scikit-learn 1.6, so the model pipeline would not have run on a current
  install at all.
- **`data/raw/.cache/` was 2.4 MB of tracked HTML**, which is why GitHub
  reported the repository's language as HTML.
- **`data/external/lm_dictionary.csv` was 9.2 MB**, 40% of the repository, of
  which the code read seven binary columns.

## 6. Where the brief was wrong on inspection

Two places, both because the brief was written against the old README rather
than the data:

1. **"the defects are about what is missing and what is broken, not about
   fabrication."** The target variable was wrong, so every published result
   was a measurement of something other than what it claimed to measure. That
   is a more serious category of defect than the brief anticipated.
2. **"H2 — Skipped (Ritter rank merge unpopulated) … diagnose why it is empty
   — almost certainly underwriter name normalisation."** The merge was empty
   because there was no `lead_underwriter` column in the dataset at all. Name
   normalisation was necessary but not sufficient; the names had to be
   recovered from the prospectuses first.

The brief also asked for a SHAP beeswarm. I published the mean-|SHAP| bar
chart and skipped the beeswarm: on a model with negative out-of-sample R², a
beeswarm invites exactly the causal misreading the brief warned about, and it
adds nothing the bar chart does not already say.

---

## 7. Still outstanding

- **The CI badge is not in the README yet.** The workflow has never run: I
  cannot push from here. Once `.github/workflows/ci.yml` runs green on
  `main`, add to the README's badge row:
  ```
  [![CI](https://github.com/leeyawnnn/ipo_underpricing_model/actions/workflows/ci.yml/badge.svg)](https://github.com/leeyawnnn/ipo_underpricing_model/actions/workflows/ci.yml)
  ```
- **Repository description and topics** need setting through the web UI or:
  ```
  gh repo edit leeyawnnn/ipo_underpricing_model \
    --description "Empirical study of US IPO first-day returns (2019-2024): SEC S-1 text features via Loughran-McDonald, deal and market-regime controls, hypothesis tests, and out-of-sample model evaluation." \
    --add-topic ipo --add-topic underpricing --add-topic sec-edgar \
    --add-topic textual-analysis --add-topic loughran-mcdonald \
    --add-topic empirical-finance --add-topic python --add-topic lightgbm
  ```
- **Verify the language badge flips to Python** on the profile page after the
  push. `.gitattributes` and the cache purge should do it, but linguist
  recomputes on push and it is worth checking.
