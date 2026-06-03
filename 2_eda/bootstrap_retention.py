"""
Bootstrap confidence intervals for retention uncertainty.

Method: cluster bootstrap by game (resample games with replacement, then take
the statistic), B = 10,000 resamples, 95% percentile CIs.

Why cluster bootstrap: within a single game the monthly observations are highly
autocorrelated, so resampling rows would understate uncertainty. Games are the
independent units here, so we resample games. This is the standard approach
for hierarchical / panel data and matches the Week 2 CI machinery applied at
the correct level of clustering.

Outputs (eda/):
  bootstrap_summary.csv             point estimates + 95% percentile CIs for
                                    per-group mean & median retention_ratio,
                                    and the free-minus-paid difference
  figures/08_decay_curve_ci.png     decay-from-peak curves with bootstrap CI bands
  figures/09_retention_forest.png   forest plot of the summary CIs

Run from repo root:  python eda/bootstrap_retention.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
EDA = REPO / "eda"
FIG = EDA / "figures"
FIG.mkdir(parents=True, exist_ok=True)

B = 10_000               # bootstrap resamples
ALPHA = 0.05             # 95% CI -> percentiles 2.5 / 97.5
SEED = 189
MAX_T_CURVE = 96         # months past peak shown on the decay curve
MIN_GAMES_AT_T = 10      # only draw the curve where >= this many games contribute
RECENT_WINDOW = 12

FREE_C, PAID_C = "#2E86C1", "#E67E22"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


# --------------------------------------------------------------------------- #
# Load and reshape
# --------------------------------------------------------------------------- #
def load_players():
    fp = pd.read_csv(DATA / "top50_free_games_player_data.csv"); fp["group"] = "Free-to-play"
    pp = pd.read_csv(DATA / "top50_paid_games_player_data.csv"); pp["group"] = "Paid"
    df = pd.concat([fp, pp], ignore_index=True)
    df["ym"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    return df[df["average_player_count"] > 0].copy()


def retention_per_game(players):
    """Return per-game retention_ratio = mean(last 12 mo) / peak (avg series)."""
    rows = []
    for (gid, grp), g in players.groupby(["game_id", "group"]):
        g = g.sort_values("ym")
        avg = g["average_player_count"].to_numpy(dtype=float)
        pk = avg.max()
        recent = g.tail(RECENT_WINDOW)["average_player_count"].mean()
        if pk > 0:
            rows.append((gid, grp, recent / pk))
    return pd.DataFrame(rows, columns=["game_id", "group", "retention_ratio"])


def peak_aligned_matrix(players, group, max_t=MAX_T_CURVE):
    """Build an (n_games, max_t) matrix of (avg / own_peak) aligned at each
    game's own peak month. Missing slots are NaN."""
    rows = []
    for _, g in players[players.group == group].groupby("game_id"):
        g = g.sort_values("ym")
        avg = g["average_player_count"].to_numpy(dtype=float)
        i = int(avg.argmax())
        after = avg[i:][:max_t]
        pk = after[0]
        if pk <= 0:
            continue
        norm = np.full(max_t, np.nan)
        norm[: len(after)] = after / pk
        rows.append(norm)
    return np.asarray(rows)


# --------------------------------------------------------------------------- #
# Percentile bootstrap helpers
# --------------------------------------------------------------------------- #
def ci_pct(arr, alpha=ALPHA):
    return np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def boot_group_stat(values, stat, rng, B=B):
    """Cluster-bootstrap a single-group statistic of a 1d array of game-level values."""
    n = len(values)
    idx = rng.integers(0, n, size=(B, n))
    return stat(values[idx], axis=1)


def boot_diff(free_vals, paid_vals, stat, rng, B=B):
    """Two-sample stratified bootstrap of stat(free) - stat(paid)."""
    nf, np_ = len(free_vals), len(paid_vals)
    idf = rng.integers(0, nf, size=(B, nf))
    idp = rng.integers(0, np_, size=(B, np_))
    return stat(free_vals[idf], axis=1) - stat(paid_vals[idp], axis=1)


def boot_curve_ci(matrix, rng, B=B, alpha=ALPHA):
    """Pointwise percentile CIs for the median of a (n_games, T) matrix.
    Resamples games (rows) with replacement; takes nanmedian column-wise."""
    n, T = matrix.shape
    lo = np.full(T, np.nan); hi = np.full(T, np.nan)
    valid_n = np.sum(~np.isnan(matrix), axis=0)
    for t in range(T):
        if valid_n[t] < MIN_GAMES_AT_T:
            continue
        col = matrix[:, t]
        # vectorize: draw all B samples of size n in one go
        idx = rng.integers(0, n, size=(B, n))
        sampled = col[idx]
        meds = np.nanmedian(sampled, axis=1)
        lo[t], hi[t] = np.percentile(meds[~np.isnan(meds)], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, valid_n


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def build_summary(ret):
    rng = np.random.default_rng(SEED)
    fr = ret.loc[ret.group == "Free-to-play", "retention_ratio"].to_numpy()
    pa = ret.loc[ret.group == "Paid", "retention_ratio"].to_numpy()

    rows = []
    for stat_name, stat in [("mean", np.mean), ("median", np.median)]:
        # per-group CIs
        for label, vals in [("Free-to-play", fr), ("Paid", pa)]:
            pt = stat(vals)
            boot = boot_group_stat(vals, stat, rng)
            lo, hi = ci_pct(boot)
            rows.append(dict(quantity=f"{stat_name} retention_ratio",
                             group=label, n_games=len(vals),
                             point_estimate=pt, ci_low=lo, ci_high=hi))
        # difference Free - Paid
        pt = stat(fr) - stat(pa)
        boot = boot_diff(fr, pa, stat, rng)
        lo, hi = ci_pct(boot)
        rows.append(dict(quantity=f"{stat_name} retention_ratio",
                         group="Free minus Paid", n_games=f"{len(fr)} vs {len(pa)}",
                         point_estimate=pt, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_decay_curve_ci(players, rng):
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col in [("Free-to-play", FREE_C), ("Paid", PAID_C)]:
        M = peak_aligned_matrix(players, grp)
        med = np.nanmedian(M, axis=0)
        lo, hi, n_t = boot_curve_ci(M, rng)
        valid = ~np.isnan(lo)
        t = np.arange(M.shape[1])
        ax.plot(t[valid], med[valid], color=col, lw=2.3,
                label=f"{grp} median (n_max={M.shape[0]})")
        ax.fill_between(t[valid], lo[valid], hi[valid], color=col, alpha=.18)
    ax.set_xlabel("Months since each game's own peak")
    ax.set_ylabel("Avg players / peak")
    ax.set_title("Decay from peak: median with 95% bootstrap CI band")
    ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "08_decay_curve_ci.png"); plt.close(fig)


def fig_forest(summary):
    order = [("Free-to-play", "mean retention_ratio"),
             ("Paid", "mean retention_ratio"),
             ("Free minus Paid", "mean retention_ratio"),
             ("Free-to-play", "median retention_ratio"),
             ("Paid", "median retention_ratio"),
             ("Free minus Paid", "median retention_ratio")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    for i, (grp, q) in enumerate(order):
        r = summary[(summary.group == grp) & (summary.quantity == q)].iloc[0]
        is_diff = grp == "Free minus Paid"
        c = "#555" if is_diff else (FREE_C if grp == "Free-to-play" else PAID_C)
        ax.errorbar(r.point_estimate, i,
                    xerr=[[r.point_estimate - r.ci_low], [r.ci_high - r.point_estimate]],
                    fmt="o", color=c, ecolor=c, capsize=4, lw=2, markersize=7)
        ax.text(r.ci_high + 0.012, i,
                f"{r.point_estimate:+.3f}  [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                if is_diff else f"{r.point_estimate:.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]",
                va="center", fontsize=9, color="#222")
    ax.axvline(0, ls="--", color="#bbb", lw=1)
    labels = [f"{q.split()[0].capitalize()}  -  {grp}" for grp, q in order]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("retention_ratio  (free minus paid for difference rows)")
    ax.set_title("Bootstrap 95% percentile CIs (B = 10,000, cluster-by-game)")
    # add headroom for the text labels on the right
    xmax = max(summary.ci_high) + 0.18
    xmin = min(summary.ci_low) - 0.05
    ax.set_xlim(xmin, xmax)
    fig.tight_layout(); fig.savefig(FIG / "09_retention_forest.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    players = load_players()
    ret = retention_per_game(players)

    summary = build_summary(ret)
    summary_out = summary.copy()
    for c in ["point_estimate", "ci_low", "ci_high"]:
        summary_out[c] = summary_out[c].round(4)
    summary_out.to_csv(EDA / "bootstrap_summary.csv", index=False)

    fig_decay_curve_ci(players, rng)
    fig_forest(summary)

    # console echo
    pd.set_option("display.width", 200, "display.max_columns", 10)
    print(f"Cluster bootstrap by game, B={B}, seed={SEED}, 95% percentile CIs\n")
    print(summary_out.to_string(index=False))
    print(f"\nWrote bootstrap_summary.csv, figures/08_decay_curve_ci.png, "
          f"figures/09_retention_forest.png")


if __name__ == "__main__":
    main()
