import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

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