import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from pathlib import Path
from scipy import stats

def compute_retention(df):
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['year_month'])
    df_sorted = df.sort_values('year_month')
    
    first_obs = df_sorted.groupby('game_name').first()
    last_obs = df_sorted.groupby('game_name').last()
    
    retention = pd.DataFrame({
        'first_players': first_obs['average_player_count'],
        'last_players': last_obs['average_player_count']
    })
    retention['retention_ratio'] = retention['last_players'] / retention['first_players']
    return retention