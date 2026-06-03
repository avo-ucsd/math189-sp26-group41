"""
Week 8 EDA -- free-to-play vs paid player retention on Steam.

Builds per-game retention features from the monthly player-count series, produces
summary-statistic tables and figures, and runs the exploratory diagnostics that
motivate the Week 9 inferential tests.

Statistical methods referenced (from weekly course summaries):
  W2  descriptive stats, sampling distribution, confidence intervals
  W4  normality diagnostics (QQ-plot, Kolmogorov-Smirnov), skew handling via log transform
  W4-5 association between two quantitative variables (Pearson / Spearman correlation)
  W2/4 group comparison setup (Welch t-test on log scale / Mann-Whitney U)

Outputs (eda/):
  per_game_features.csv      one row per game with engineered retention features
  group_summary_stats.csv    grouped descriptive statistics
  diagnostics.txt            normality tests, correlations, group-comparison preview
  figures/*.png              seven EDA figures

Run from repo root:  python eda/eda_player_retention.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
EDA = REPO / "eda"
FIG = EDA / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FREE_C, PAID_C = "#2E86C1", "#E67E22"
RECENT_WINDOW = 12  # months used to define "long-term / recent" activity
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load():
    fp = pd.read_csv(DATA / "top50_free_games_player_data.csv"); fp["group"] = "Free-to-play"
    pp = pd.read_csv(DATA / "top50_paid_games_player_data.csv"); pp["group"] = "Paid"
    players = pd.concat([fp, pp], ignore_index=True)
    players["ym"] = pd.to_datetime(players["year_month"], format="%Y-%m")
    # cleaning: drop non-positive activity rows (closed servers / no data months)
    n0 = len(players)
    players = players[players["average_player_count"] > 0].copy()
    dropped = n0 - len(players)

    fl = pd.read_csv(DATA / "top50_free_games_list.csv"); fl["group"] = "Free-to-play"
    pl = pd.read_csv(DATA / "top50_paid_games_list.csv"); pl["group"] = "Paid"
    pl = pl.rename(columns={"Base Price": "base_price"})
    fl["base_price"] = "Free"
    lst = pd.concat([fl, pl], ignore_index=True)
    lst["release_date"] = pd.to_datetime(lst["Release Date"], errors="coerce")
    lst["release_year"] = lst["release_date"].dt.year
    return players, lst, dropped


# --------------------------------------------------------------------------- #
# Per-game feature engineering
# --------------------------------------------------------------------------- #
def per_game_features(players):
    rows = []
    for (gid, grp), g in players.groupby(["game_id", "group"]):
        g = g.sort_values("ym")
        avg = g["average_player_count"].to_numpy()
        peak_avg = avg.max()
        i_peak = int(avg.argmax())
        n = len(g)
        recent = g.tail(RECENT_WINDOW)["average_player_count"].mean()
        rows.append(dict(
            game_id=gid, group=grp, game_name=g["game_name"].iloc[0],
            n_months=n,
            first_month=g["ym"].min(), last_month=g["ym"].max(),
            peak_avg=peak_avg,
            peak_concurrent=g["peak_player_count"].max(),
            mean_avg=avg.mean(), median_avg=float(np.median(avg)),
            # when in the game's observed life the peak occurs (0 = start, 1 = end)
            peak_position=i_peak / (n - 1) if n > 1 else 0.0,
            recent_avg=recent,
            # long-term retention: recent activity as a share of historical peak
            retention_ratio=recent / peak_avg if peak_avg > 0 else np.nan,
            # share of months sustaining >= 50% of peak (breadth of the plateau)
            months_above_half_peak=float(np.mean(avg >= 0.5 * peak_avg)),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary statistics table (W2 descriptive stats)
# --------------------------------------------------------------------------- #
def group_summary(feat):
    metrics = ["n_months", "peak_avg", "recent_avg", "retention_ratio",
               "peak_position", "months_above_half_peak"]
    out = []
    for grp, g in feat.groupby("group"):
        for m in metrics:
            s = g[m].dropna()
            out.append(dict(group=grp, metric=m, n=len(s),
                            mean=s.mean(), sd=s.std(ddof=1),
                            median=s.median(), q1=s.quantile(.25),
                            q3=s.quantile(.75), min=s.min(), max=s.max(),
                            skew=stats.skew(s)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Normalized retention curves: align to months-since-first-observation,
# normalize each game by its own peak, take the median curve per group.
# --------------------------------------------------------------------------- #
def retention_curves(players, max_t=120):
    curves = {}
    for grp, gg in players.groupby("group"):
        mat = []
        for gid, g in gg.groupby("game_id"):
            g = g.sort_values("ym")
            avg = g["average_player_count"].to_numpy(dtype=float)
            pk = avg.max()
            if pk <= 0:
                continue
            norm = avg / pk
            t = np.arange(len(norm))
            s = pd.Series(norm[:max_t], index=t[:max_t])
            mat.append(s)
        df = pd.DataFrame(mat)
        curves[grp] = dict(median=df.median(), q1=df.quantile(.25), q3=df.quantile(.75),
                           count=df.count())
    return curves


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_distributions(feat):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        d = feat.loc[feat.group == grp, "peak_avg"]
        ax[0].hist(d, bins=np.logspace(0, 6, 25), alpha=.55, color=col, label=grp)
        ax[1].hist(np.log10(d), bins=20, alpha=.55, color=col, label=grp)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Peak monthly avg concurrent players")
    ax[0].set_ylabel("Number of games")
    ax[0].set_title("Peak player counts (log x)")
    ax[1].set_xlabel("log10(peak avg players)")
    ax[1].set_ylabel("Number of games")
    ax[1].set_title("Log-transformed")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "01_distribution_peak_players.png"); plt.close(fig)


def fig_qq(feat):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, (grp, col) in zip(ax, [("Free-to-play", FREE_C), ("Paid", PAID_C)]):
        d = np.log10(feat.loc[feat.group == grp, "peak_avg"].dropna())
        stats.probplot(d, dist="norm", plot=a)
        a.get_lines()[0].set_markerfacecolor(col); a.get_lines()[0].set_markeredgecolor(col)
        a.get_lines()[0].set_markersize(5); a.get_lines()[1].set_color("k")
        a.set_title(f"{grp}: log10(peak avg)")
    fig.tight_layout(); fig.savefig(FIG / "02_qqplot_log_peak.png"); plt.close(fig)


def fig_retention_box(feat):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = ["Free-to-play", "Paid"]
    cols = [FREE_C, PAID_C]
    for a, metric, title in [(ax[0], "retention_ratio", "Retention ratio (recent 12mo / peak)"),
                             (ax[1], "peak_position", "Peak position in game life (0=launch, 1=now)")]:
        data = [feat.loc[feat.group == g, metric].dropna() for g in groups]
        bp = a.boxplot(data, tick_labels=groups, patch_artist=True, widths=.55, showmeans=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(.55)
        for med in bp["medians"]:
            med.set_color("k")
        a.set_title(title)
    fig.tight_layout(); fig.savefig(FIG / "03_retention_and_peakpos_box.png"); plt.close(fig)


def fig_curves(curves):
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        c = curves[grp]
        x = c["median"].index.to_numpy()
        ax.plot(x, c["median"].to_numpy(), color=col, lw=2.2, label=f"{grp} (median)")
        ax.fill_between(x, c["q1"].to_numpy(), c["q3"].to_numpy(), color=col, alpha=.15)
    ax.set_xlabel("Months since first observed month")
    ax.set_ylabel("Avg players / own peak")
    ax.set_title("Normalized retention curves (median, IQR band)")
    ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_retention_curves.png"); plt.close(fig)


def fig_decay_from_peak(players, max_t=96, min_games=10):
    """Retention measured as decay from each game's own peak month. Aligning at
    the peak removes the peak-timing / survivorship artifact that distorts the
    months-since-first-observation curve."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        mat = []
        for _, g in players[players.group == grp].groupby("game_id"):
            g = g.sort_values("ym")
            avg = g["average_player_count"].to_numpy(dtype=float)
            i = int(avg.argmax())
            after = avg[i:]
            pk = after[0]
            if pk <= 0:
                continue
            mat.append(pd.Series((after / pk)[:max_t]))
        df = pd.DataFrame(mat)
        n = df.count()
        keep = n[n >= min_games].index
        ax.plot(keep, df.median()[keep].values, color=col, lw=2.3, label=f"{grp} (n>={min_games})")
        ax.fill_between(keep, df.quantile(.25)[keep], df.quantile(.75)[keep], color=col, alpha=.15)
    ax.set_xlabel("Months since each game's own peak")
    ax.set_ylabel("Avg players / peak")
    ax.set_title("Decay from peak (median, IQR band)")
    ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "07_decay_from_peak.png"); plt.close(fig)


def fig_aggregate_timeseries(players):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        g = players[players.group == grp].groupby("ym")["average_player_count"].median()
        ax.plot(g.index, g.values, color=col, lw=1.9, label=grp)
    ax.set_yscale("log")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Median avg concurrent players (log)")
    ax.set_title("Median game activity over time")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "05_aggregate_timeseries.png"); plt.close(fig)


def fig_scatter(feat):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        d = feat[feat.group == grp]
        ax.scatter(d["peak_avg"], d["retention_ratio"], color=col, alpha=.7,
                   s=42, edgecolor="white", label=grp)
    ax.set_xscale("log")
    ax.set_xlabel("Peak monthly avg players (log)")
    ax.set_ylabel("Retention ratio (recent 12mo / peak)")
    ax.set_title("Popularity vs long-term retention")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "06_peak_vs_retention.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Diagnostics (normality, correlation, group-comparison preview)
# --------------------------------------------------------------------------- #
def diagnostics(feat):
    lines = []
    def p(s): lines.append(s); print(s)

    p("=== NORMALITY (W4): Kolmogorov-Smirnov vs fitted normal ===")
    for grp in ["Free-to-play", "Paid"]:
        for label, x in [("peak_avg (raw)", feat.loc[feat.group == grp, "peak_avg"]),
                         ("log10(peak_avg)", np.log10(feat.loc[feat.group == grp, "peak_avg"]))]:
            x = x.dropna()
            ks = stats.kstest((x - x.mean()) / x.std(ddof=1), "norm")
            p(f"  {grp:<13} {label:<18} KS D={ks.statistic:.3f}  p={ks.pvalue:.4f}  skew={stats.skew(x):.2f}")

    p("\n=== ASSOCIATION (W4-5): peak popularity vs retention_ratio ===")
    for grp in ["Free-to-play", "Paid"]:
        d = feat[feat.group == grp].dropna(subset=["peak_avg", "retention_ratio"])
        pr = stats.pearsonr(np.log10(d["peak_avg"]), d["retention_ratio"])
        sr = stats.spearmanr(d["peak_avg"], d["retention_ratio"])
        p(f"  {grp:<13} Pearson(log peak, ret) r={pr.statistic:.3f} p={pr.pvalue:.4f} | "
          f"Spearman rho={sr.statistic:.3f} p={sr.pvalue:.4f}")

    p("\n=== GROUP COMPARISON PREVIEW (motivates W9 inference) ===")
    fr = feat.loc[feat.group == "Free-to-play", "retention_ratio"].dropna()
    pa = feat.loc[feat.group == "Paid", "retention_ratio"].dropna()
    p(f"  retention_ratio  Free median={fr.median():.3f}  Paid median={pa.median():.3f}")
    welch = stats.ttest_ind(fr, pa, equal_var=False)
    mwu = stats.mannwhitneyu(fr, pa, alternative="two-sided")
    p(f"  Welch t-test (W2): t={welch.statistic:.3f}  p={welch.pvalue:.4f}")
    p(f"  Mann-Whitney U (W4, robust to skew): U={mwu.statistic:.0f}  p={mwu.pvalue:.4f}")
    # bootstrap CI for difference in medians (W2 CIs / resampling)
    rng = np.random.default_rng(189)
    diffs = [np.median(rng.choice(fr, len(fr))) - np.median(rng.choice(pa, len(pa)))
             for _ in range(10000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p(f"  Bootstrap 95% CI for median(Free)-median(Paid): [{lo:.3f}, {hi:.3f}]")
    return "\n".join(lines)


def main():
    players, lst, dropped = load()
    print(f"Cleaning: dropped {dropped} rows with non-positive average_player_count")
    feat = per_game_features(players)
    feat = feat.merge(lst[["AppID", "release_year", "base_price", "Estimated Owners"]],
                      left_on="game_id", right_on="AppID", how="left").drop(columns="AppID")
    feat.to_csv(EDA / "per_game_features.csv", index=False)

    summ = group_summary(feat)
    summ.to_csv(EDA / "group_summary_stats.csv", index=False)

    curves = retention_curves(players)
    fig_distributions(feat); fig_qq(feat); fig_retention_box(feat)
    fig_curves(curves); fig_decay_from_peak(players)
    fig_aggregate_timeseries(players); fig_scatter(feat)

    diag = diagnostics(feat)
    (EDA / "diagnostics.txt").write_text(diag)

    print(f"\nGames analysed: {feat.group.value_counts().to_dict()}")
    print(f"Wrote per_game_features.csv, group_summary_stats.csv, diagnostics.txt, "
          f"{len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
