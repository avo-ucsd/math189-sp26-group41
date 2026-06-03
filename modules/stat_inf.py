import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats

def boot_stat(x, stat, B, rng):
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return stat(x[idx], axis=1)

def boot_diff(a, b, stat, B, rng):
    return boot_stat(a, stat, B, rng) - boot_stat(b, stat, B, rng)

def pct_ci(samples, alpha=0.05):
    return np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

WINDOWS = [(1, 6), (7, 12), (13, 24), (25, 48), (49, 96)]

def windowed_retention(player_df, group_label):
    df = player_df.copy()
    df['ym'] = pd.to_datetime(df['year_month'])
    df = df[df['average_player_count'] > 0]
    rows = []
    for gid, g in df.groupby('game_id'):
        g = g.sort_values('ym')
        a = g['average_player_count'].to_numpy(float)
        i = int(a.argmax())
        post, peak = a[i+1:], a[i]
        rec = {'game_id': gid, 'group': group_label}
        for lo, hi in WINDOWS:
            seg = post[lo-1:hi]
            rec[f'{lo}-{hi}'] = (seg.mean() / peak) if len(seg) > 0 and peak > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

"""
Section 4.5
"""
def load_review_scores():
    review_scores = pd.read_csv("data/review_scores_paired.csv")
    review_scores = review_scores.rename(columns={
        "Title": "game_name",
        "List": "group",
        "ReleaseScore": "release_score",
        "LatestScore": "latest_score",
        "ReleaseTotal": "release_total_reviews",
        "LatestTotal": "latest_total_reviews",
    })
    review_scores["group"] = review_scores["group"].replace({
        "free": "Free-to-play",
        "paid": "Paid",
    })
    review_scores["release_score"] = pd.to_numeric(review_scores["release_score"], errors="coerce")
    review_scores["latest_score"] = pd.to_numeric(review_scores["latest_score"], errors="coerce")
    review_scores["release_total_reviews"] = pd.to_numeric(review_scores["release_total_reviews"], errors="coerce")
    review_scores["latest_total_reviews"] = pd.to_numeric(review_scores["latest_total_reviews"], errors="coerce")
    review_scores["score_change"] = review_scores["latest_score"] - review_scores["release_score"]
    return review_scores


def run_paired_test(group_frame):
    paired_frame = group_frame.dropna(subset=["release_score", "latest_score"]).copy()
    release_scores = paired_frame["release_score"]
    latest_scores = paired_frame["latest_score"]
    score_changes = paired_frame["score_change"]
    n_games = len(paired_frame)

    mean_change = score_changes.mean()
    standard_deviation = score_changes.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(n_games)
    t_statistic, p_value = stats.ttest_rel(latest_scores, release_scores)
    t_critical = stats.t.ppf(0.975, n_games - 1)
    confidence_interval = (
        mean_change - t_critical * standard_error,
        mean_change + t_critical * standard_error,
    )

    return pd.Series({
        "games": n_games,
        "release_mean": release_scores.mean(),
        "latest_mean": latest_scores.mean(),
        "mean_change": mean_change,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "ci_low": confidence_interval[0],
        "ci_high": confidence_interval[1],
        "cohens_d": mean_change / standard_deviation,
        "shapiro_p": stats.shapiro(score_changes).pvalue,
        "wilcoxon_p": stats.wilcoxon(latest_scores, release_scores).pvalue,
        "shifted_games": int(paired_frame["Shifted"].sum()),
        "capped_games": int(paired_frame["Capped"].sum()),
    })

"""
Section 4.6
"""
def _parse_owner_token(token: str) -> float:
    token = str(token).strip().replace(",", "").replace("+", "")
    if token.endswith("M"):
        return float(token[:-1]) * 1_000_000
    if token.endswith("K"):
        return float(token[:-1]) * 1_000
    return float(token)


def parse_owner_range(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if "-" in text:
        low, high = [part.strip() for part in text.split("-")]
        return (_parse_owner_token(low) + _parse_owner_token(high)) / 2
    return _parse_owner_token(text)


def load_tables():
    free_players = pd.read_csv("data/top50_free_games_player_data.csv")
    paid_players = pd.read_csv("data/top50_paid_games_player_data.csv")
    free_list = pd.read_csv("data/top50_free_games_list.csv")
    paid_list = pd.read_csv("data/top50_paid_games_list.csv")

    free_players["group"] = "Free-to-play"
    paid_players["group"] = "Paid"
    players = pd.concat([free_players, paid_players], ignore_index=True)
    players["year_month"] = pd.to_datetime(players["year_month"], format="%Y-%m")
    players = players.loc[players["average_player_count"] > 0].copy()

    free_list = free_list.rename(columns={
        "AppID": "game_id",
        "Game Title": "game_name",
        "Release Date": "release_date",
        "Estimated Owners": "estimated_owners",
        "Peak Concurrent Players (24h)": "launch_peak_concurrent_players",
    })
    paid_list = paid_list.rename(columns={
        "AppID": "game_id",
        "Game Title": "game_name",
        "Release Date": "release_date",
        "Base Price": "base_price",
        "Estimated Owners": "estimated_owners",
        "Peak Concurrent Players (24h)": "launch_peak_concurrent_players",
    })
    free_list["base_price"] = 0.0
    free_list["group"] = "Free-to-play"
    paid_list["group"] = "Paid"

    games = pd.concat([free_list, paid_list], ignore_index=True)
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
    games["release_year"] = games["release_date"].dt.year
    games["estimated_owners_mid"] = games["estimated_owners"].apply(parse_owner_range)
    games["base_price"] = pd.to_numeric(
        games["base_price"].astype(str).str.replace("$", "", regex=False),
        errors="coerce",
    )
    games["log_estimated_owners"] = np.log10(games["estimated_owners_mid"].where(games["estimated_owners_mid"] > 0))
    games["log_launch_peak_concurrent_players"] = np.log10(
        games["launch_peak_concurrent_players"].where(games["launch_peak_concurrent_players"] > 0)
    )

    return players, games

def build_game_features(players, games):
    rows = []
    for (game_id, group), group_frame in players.groupby(["game_id", "group"]):
        group_frame = group_frame.sort_values("year_month")
        values = group_frame["average_player_count"].to_numpy(dtype=float)
        peak_avg = values.max()
        recent_avg = group_frame.tail(12)["average_player_count"].mean()
        rows.append({
            "game_id": game_id,
            "group": group,
            "n_months": len(group_frame),
            "first_month": group_frame["year_month"].min(),
            "last_month": group_frame["year_month"].max(),
            "peak_avg": peak_avg,
            "recent_avg": recent_avg,
            "retention_ratio": recent_avg / peak_avg if peak_avg > 0 else np.nan,
            "peak_position": int(values.argmax()) / (len(values) - 1) if len(values) > 1 else 0.0,
            "months_above_half_peak": np.mean(values >= 0.5 * peak_avg),
        })

    features = pd.DataFrame(rows).merge(
        games[[
            "game_id",
            "game_name",
            "release_year",
            "estimated_owners_mid",
            "log_estimated_owners",
            "launch_peak_concurrent_players",
            "log_launch_peak_concurrent_players",
            "base_price",
        ]],
        on="game_id",
        how="left",
    )
    features["is_free_to_play"] = (features["group"] == "Free-to-play").astype(int)
    features["log_retention_ratio"] = np.log(features["retention_ratio"])
    features["release_year_centered"] = features["release_year"] - features["release_year"].mean()
    return features


def summarize_by_group(features, columns):
    records = []
    for column in columns:
        for group_name, group_frame in features.groupby("group"):
            series = group_frame[column].dropna()
            records.append({
                "group": group_name,
                "metric": column,
                "n": int(series.shape[0]),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "max": series.max(),
            })
    return pd.DataFrame(records)