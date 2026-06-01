"""
Windowed retention with bootstrap CIs.

Splits the "retention ratio" into multiple post-peak time windows so short-term
and long-term retention can be compared on their own footing. Within each
window we compute one retention value per game (mean of avg-players in that
window divided by the game's peak), then cluster-bootstrap by game to get
percentile CIs.

Windows (months after each game's own peak month):
    1-6     short-term
    7-12    medium-short
    13-24   medium
    25-48   medium-long
    49-96   long-term

A game contributes to a window only if at least one of its observations falls
inside that window. min_games_per_group_per_window controls the cutoff below
which a window-group cell is dropped.

Outputs (eda/):
    bootstrap_windowed.csv             tidy table of point estimates + 95% CIs
    figures/10_windowed_retention.png  two-panel forest plot

Run from repo root:  python eda/bootstrap_windowed.py
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

B = 10_000
ALPHA = 0.05
SEED = 189
MIN_GAMES = 10                       # per group, per window
WINDOWS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 96)]
WIN_LABEL = {w: f"{w[0]}-{w[1]} mo" for w in WINDOWS}

FREE_C, PAID_C, DIFF_C = "#2E86C1", "#E67E22", "#444444"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


# --------------------------------------------------------------------------- #
def load_players():
    fp = pd.read_csv(DATA / "top50_free_games_player_data.csv"); fp["group"] = "Free-to-play"
    pp = pd.read_csv(DATA / "top50_paid_games_player_data.csv"); pp["group"] = "Paid"
    df = pd.concat([fp, pp], ignore_index=True)
    df["ym"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    return df[df["average_player_count"] > 0].copy()


def windowed_retention(players):
    """For each game, compute mean(avg in [lo, hi] months post-peak) / peak."""
    rows = []
    for (gid, grp), g in players.groupby(["game_id", "group"]):
        g = g.sort_values("ym")
        avg = g["average_player_count"].to_numpy(dtype=float)
        i = int(avg.argmax())
        post = avg[i + 1:]                     # months strictly after peak
        peak = avg[i]
        rec = dict(game_id=gid, group=grp)
        for lo, hi in WINDOWS:
            seg = post[lo - 1: hi]              # months lo..hi post-peak (1-indexed)
            rec[WIN_LABEL[(lo, hi)]] = (seg.mean() / peak) if len(seg) > 0 and peak > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def ci(arr, alpha=ALPHA):
    return np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def boot_group_median(vals, rng, B=B):
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    return np.median(vals[idx], axis=1)


def boot_diff_median(fr, pa, rng, B=B):
    nf, npd = len(fr), len(pa)
    idf = rng.integers(0, nf, size=(B, nf))
    idp = rng.integers(0, npd, size=(B, npd))
    return np.median(fr[idf], axis=1) - np.median(pa[idp], axis=1)


def build_table(wret):
    rng = np.random.default_rng(SEED)
    rows = []
    for w in WINDOWS:
        col = WIN_LABEL[w]
        fr = wret.loc[wret.group == "Free-to-play", col].dropna().to_numpy()
        pa = wret.loc[wret.group == "Paid", col].dropna().to_numpy()
        # per-group medians
        for label, vals in [("Free-to-play", fr), ("Paid", pa)]:
            if len(vals) < MIN_GAMES:
                rows.append(dict(window=col, group=label, n_games=len(vals),
                                 point_estimate=np.nan, ci_low=np.nan, ci_high=np.nan))
                continue
            boot = boot_group_median(vals, rng)
            lo, hi = ci(boot)
            rows.append(dict(window=col, group=label, n_games=len(vals),
                             point_estimate=float(np.median(vals)), ci_low=lo, ci_high=hi))
        # difference free - paid
        if len(fr) >= MIN_GAMES and len(pa) >= MIN_GAMES:
            boot = boot_diff_median(fr, pa, rng)
            lo, hi = ci(boot)
            rows.append(dict(window=col, group="Free minus Paid",
                             n_games=f"{len(fr)} vs {len(pa)}",
                             point_estimate=float(np.median(fr) - np.median(pa)),
                             ci_low=lo, ci_high=hi))
        else:
            rows.append(dict(window=col, group="Free minus Paid",
                             n_games=f"{len(fr)} vs {len(pa)}",
                             point_estimate=np.nan, ci_low=np.nan, ci_high=np.nan))
    return pd.DataFrame(rows)


def fig_windowed(tbl):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw=dict(width_ratios=[1, 1]))
    wins = [WIN_LABEL[w] for w in WINDOWS]
    y = np.arange(len(wins))

    # Left: per-group medians with CIs
    for offset, grp, c in [(-0.18, "Free-to-play", FREE_C), (0.18, "Paid", PAID_C)]:
        sub = tbl[tbl.group == grp].set_index("window").loc[wins]
        ax1.errorbar(sub["point_estimate"].values, y + offset,
                     xerr=[sub["point_estimate"].values - sub["ci_low"].values,
                           sub["ci_high"].values - sub["point_estimate"].values],
                     fmt="o", color=c, ecolor=c, capsize=4, lw=2, markersize=7, label=grp)
    ax1.set_yticks(y); ax1.set_yticklabels(wins)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Median retention ratio  (mean of window / peak)")
    ax1.set_title("Per-group medians by post-peak window")
    ax1.legend(loc="lower right")

    # Right: free minus paid with CIs, 0 reference
    sub = tbl[tbl.group == "Free minus Paid"].set_index("window").loc[wins]
    ax2.axvline(0, ls="--", color="#bbb", lw=1)
    ax2.errorbar(sub["point_estimate"].values, y,
                 xerr=[sub["point_estimate"].values - sub["ci_low"].values,
                       sub["ci_high"].values - sub["point_estimate"].values],
                 fmt="o", color=DIFF_C, ecolor=DIFF_C, capsize=4, lw=2, markersize=7)
    for i, (pt, lo, hi) in enumerate(zip(sub.point_estimate, sub.ci_low, sub.ci_high)):
        ax2.text(hi + 0.015, i, f"{pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                 va="center", fontsize=9, color="#222")
    ax2.set_yticks(y); ax2.set_yticklabels(wins)
    ax2.invert_yaxis()
    pad = 0.05
    ax2.set_xlim(min(sub.ci_low) - pad, max(sub.ci_high) + 0.35)
    ax2.set_xlabel("Free minus Paid  (median retention ratio)")
    ax2.set_title("Free-vs-Paid gap by window")

    fig.suptitle("Windowed retention with 95% bootstrap CIs (B = 10,000, cluster-by-game)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "10_windowed_retention.png", bbox_inches="tight")
    plt.close(fig)


def main():
    players = load_players()
    wret = windowed_retention(players)
    tbl = build_table(wret)
    out = tbl.copy()
    for c in ["point_estimate", "ci_low", "ci_high"]:
        out[c] = out[c].round(4)
    out.to_csv(EDA / "bootstrap_windowed.csv", index=False)
    fig_windowed(tbl)

    pd.set_option("display.width", 200, "display.max_columns", 10)
    print(f"Cluster bootstrap by game, B={B}, seed={SEED}, 95% percentile CIs")
    print(out.to_string(index=False))
    print("\nWrote bootstrap_windowed.csv and figures/10_windowed_retention.png")


if __name__ == "__main__":
    main()
