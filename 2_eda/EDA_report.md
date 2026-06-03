# Exploratory Data Analysis: Free-to-Play vs Paid Player Retention on Steam

**MATH 189 — Group 41 — Week 8 (EDA / Summary Statistics)**

## Research question

For top Steam games, do free-to-play titles sustain stronger long-term player retention than paid titles, as measured through their monthly player-count trends?

## Data

The analysis draws on four scraped files: a free-to-play game list (50 games), a paid game list (50 games), and the corresponding SteamCharts monthly player-count histories. After reshaping, the player histories cover **2012-07 through 2026-04** (up to 166 monthly observations per game). The two metrics available per game-month are `peak_player_count` and `average_player_count`; we use the monthly **average concurrent players** as the primary activity measure because it reflects sustained engagement rather than one-off spikes.

Three free-to-play games (AppIDs 409160, 365300, 1121910) appear in the list but have no player history and are excluded, leaving **47 free-to-play and 50 paid** games. Four game-months with a non-positive average were dropped as inactive/no-data periods. The 2011–2018 release filter described in the proposal is **not** applied here, per the current analysis scope; release-date coverage is documented separately in `release_window_audit.xlsx`.

## Engineered retention features

Because the question is about *retention*, not raw popularity, each game is reduced to a small set of features (see `per_game_features.csv`):

- **`peak_avg`** — highest monthly average concurrent players (popularity / scale).
- **`peak_position`** — where in the observed life the peak falls (0 = first month, 1 = most recent), capturing early-spike vs late-bloom behavior.
- **`retention_ratio`** — mean activity over the most recent 12 months divided by the historical peak; the headline long-term-retention proxy (1 = still at peak, 0 = abandoned).
- **`months_above_half_peak`** — share of months sustaining at least 50% of peak (breadth of the active plateau).

## Methods

Methods follow the course toolkit referenced in the weekly summaries: descriptive statistics and confidence intervals (Week 2); normality diagnostics via QQ-plots and the Kolmogorov–Smirnov test, plus skew handling by log transformation (Week 4); association between quantitative variables via Pearson and Spearman correlation (Weeks 4–5); and a group-comparison preview using the Welch two-sample t-test and the Mann–Whitney U test, with a bootstrap confidence interval for the difference in medians. The inferential tests here are exploratory and set up the formal Week 9 analysis.

## Findings

### 1. Player counts are heavily right-skewed and approximately log-normal

Raw peak averages are extremely skewed (skew ≈ 3.5 free, 4.0 paid) and fail the KS normality test (p < 0.0001). A base-10 log transform removes most of the skew (≈ 0.4 free, 0.8 paid) and is consistent with normality (KS p = 0.54 free, 0.44 paid). **Implication:** all scale-based comparisons and any regression should operate on the log scale or use rank-based tests — using raw counts would let a few giants (e.g. Dota 2, PUBG) dominate. See `figures/01_distribution_peak_players.png` and `figures/02_qqplot_log_peak.png`.

### 2. Free games are bigger at the top but not in the middle

Mean `peak_avg` is higher for free games (114k vs 95k), driven by a handful of mega-hits, but the **median is the reverse** — paid 31.7k vs free 7.2k. The typical free-to-play game in this sample is smaller than the typical paid game; the free group's average is inflated by extreme outliers. This is exactly why the median and rank-based methods are the honest summaries here.

### 3. Free games peak earlier; paid games hold a larger share of their peak

Free games reach their peak very early (median `peak_position` = 0.13, i.e. ~13% into their observed life), consistent with a launch-spike pattern. Paid games peak later and more evenly (median 0.32). On the headline metric, paid games retain a higher fraction of their peak: median `retention_ratio` of **0.281 (paid)** vs **0.150 (free)**. See `figures/03_retention_and_peakpos_box.png`.

### 4. Retention curves confirm the pattern

The decay-from-peak curve (`figures/07_decay_from_peak.png`) is the cleanest view: both groups fall sharply in the first ~12 months, but paid games then stabilize around 0.2–0.3 of peak while free games continue decaying toward 0.1–0.15. The alternative alignment by months-since-first-observation (`figures/04_retention_curves.png`) shows the same early-life gap but should be read with care — at large horizons it is distorted by survivorship and by paid games' later peak timing (see Limitations).

### 5. Among paid games, bigger means stickier-proportion is *weaker*

`retention_ratio` is essentially uncorrelated with popularity for free games (Pearson r = 0.13 on log peak, p = 0.38; Spearman ρ = 0.07). For paid games there is a modest **negative** association (Pearson r = −0.34, p = 0.017; Spearman ρ = −0.35, p = 0.012): the largest paid hits give back a larger share of their peak over time. See `figures/06_peak_vs_retention.png`.

### 6. Group-comparison preview (for Week 9)

Comparing `retention_ratio` between groups: the difference in medians (paid − free ≈ 0.13) is suggestive but **not yet significant** at α = 0.05. The Welch t-test gives t = −1.04, p = 0.30; the more appropriate skew-robust Mann–Whitney U gives p = 0.068; and the bootstrap 95% CI for median(free) − median(paid) is [−0.238, 0.026], which includes 0. The Mann–Whitney result is the one to trust given the skew, and it sits right at the margin — a larger or scope-corrected sample may sharpen it.

## Summary

The exploratory evidence leans **against** the proposal's hypothesis that free-to-play games retain better long-term. In this sample, free games show the expected larger launch spikes but **weaker** proportional retention, while paid games peak later and hold a higher share of their peak. The gap is directionally consistent across the retention ratio, the peak-position metric, and both retention curves, but is only marginally significant so far — motivating the formal Week 9 testing.

## Limitations

1. **Sample composition.** The lists were ranked by current estimated ownership without the 2011–2018 release filter; half the 100 games fall outside that window (see `release_window_audit.xlsx`). Re-scoping the sample could change the comparison.
2. **Left truncation / survivorship.** SteamCharts coverage begins 2012-07, so games launched earlier have no true launch data, and only long-lived games contribute at long horizons — which inflates the tail of the months-since-first-observation curve.
3. **No review or genre/multiplayer data yet.** The proposal's review-score trends and online/multiplayer covariates are not in the scraped files, so the planned `free-to-play × multiplayer` interaction cannot yet be explored.
4. **Pricing-model and edition artifacts.** Some titles carry re-release/edition store dates rather than original launch dates, and price is treated as a fixed current label.

## Next steps (Week 9)

Run the formal Welch t-test and Mann–Whitney U on `retention_ratio` (log-transformed where appropriate) with confidence intervals; add review-score and multiplayer covariates once scraped; and fit a regression of `retention_ratio` on free-to-play status, log launch size, release year, and a free-to-play × multiplayer interaction, using the regression diagnostics and multicollinearity checks from Weeks 5–7.

## Reproducibility

`python eda/eda_player_retention.py` regenerates every figure, `per_game_features.csv`, `group_summary_stats.csv`, and `diagnostics.txt`.
