import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load original enriched data
df = pd.read_csv('/Users/perrygregg/Downloads/NDHockey_enriched.csv')

# Filter to games with travel data
df_model = df[df['travel_intensity'].notna()].copy()

# Calculate opponent strength (same as in multivariate script)
def calculate_opponent_strength(df):
    """Calculate opponent strength as rolling average of opponent's goals allowed"""
    
    team_stats = []
    for team_id in df['team_id'].unique():
        team_data = df[df['team_id'] == team_id].sort_values('game_date')
        team_data['goals_allowed_rolling'] = team_data['sGoalsAllowed'].rolling(
            window=3, min_periods=1
        ).mean()
        team_stats.append(team_data)
    
    team_stats_df = pd.concat(team_stats, ignore_index=False).sort_index()
    
    opponent_strength_list = []
    for idx, row in df.iterrows():
        game_id = row['game_id']
        team_id = row['team_id']
        
        opponent_data = df[(df['game_id'] == game_id) & (df['team_id'] != team_id)]
        
        if len(opponent_data) > 0:
            opponent_idx = opponent_data.index[0]
            opponent_strength = team_stats_df.loc[opponent_idx, 'goals_allowed_rolling']
        else:
            opponent_strength = np.nan
        
        opponent_strength_list.append(opponent_strength)
    
    return opponent_strength_list

df_model['opponent_strength'] = calculate_opponent_strength(df_model)

# Select variables for correlation analysis
variables = [
    'travel_intensity',
    'travel_distance',
    'rest_days',
    'timezone_shift',
    'opponent_strength',
    'is_away',
    'scoring_margin',
    'is_win',
    'team_score',
    'sGoalsAllowed'
]

# Create analysis dataset
corr_df = df_model[variables].dropna()

# Calculate correlation matrix
corr_matrix = corr_df.corr()

# Create figure with better styling
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with enhanced visuals
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=1.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            vmin=-1,
            vmax=1,
            ax=ax,
            annot_kws={"size": 9, "weight": "bold"})

# Enhance labels
ax.set_title('OLS Model Variables: Correlation Matrix\nND Hockey Travel Intensity Analysis',
             fontsize=14, fontweight='bold', pad=20)

# Rotate labels for readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Improve layout
plt.tight_layout()

# Save figure
plt.savefig('/Users/perrygregg/Downloads/ols_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ols_correlation_heatmap.png")

# Also create a focused heatmap highlighting the OLS model variables
print("\nCreating focused OLS model heatmap...")

ols_vars = [
    'travel_intensity',
    'opponent_strength',
    'is_away',
    'rest_days',
    'timezone_shift',
    'scoring_margin'
]

ols_corr = corr_df[ols_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(ols_corr,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=2,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            vmin=-1,
            vmax=1,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"})

ax.set_title('OLS Model Variables Only: Correlation Matrix\nMultivariate Regression Analysis',
             fontsize=13, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/ols_model_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ols_model_correlation_heatmap.png")

print("\nCorrelation Summary:")
print(f"Sample size: {len(corr_df)}")
print(f"\nFull correlation matrix shape: {corr_matrix.shape}")
print(f"OLS model variables correlation shape: {ols_corr.shape}")
print("\nOLS Model Correlations:")
print(ols_corr)
