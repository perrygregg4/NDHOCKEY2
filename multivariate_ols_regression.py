import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load enriched data
df = pd.read_csv('/Users/perrygregg/Downloads/NDHockey_enriched.csv')

print("="*90)
print("MULTIPLE LINEAR REGRESSION (OLS): Travel Intensity on Scoring Margin")
print("Controlling for Opponent Strength, Home/Away, Rest Days, Timezone Shifts")
print("="*90)

# ===== 1. ENGINEER OPPONENT STRENGTH =====
print("\n" + "="*90)
print("STEP 1: ENGINEER OPPONENT STRENGTH VARIABLE")
print("="*90)

# Opponent strength = average opponent goals allowed across all their games
# This measures how good the opponent's defense is (higher = stronger opponent)

def calculate_opponent_strength(df):
    """
    Calculate opponent strength as a rolling average of opponent's goals allowed
    This represents how strong the opposing team's defense is
    """
    
    # Group by team to get their defensive stats
    team_stats = []
    for team_id in df['team_id'].unique():
        team_data = df[df['team_id'] == team_id].sort_values('game_date')
        
        # Calculate rolling average of goals allowed (defense strength)
        team_data['goals_allowed_rolling'] = team_data['sGoalsAllowed'].rolling(
            window=3, min_periods=1
        ).mean()
        
        team_stats.append(team_data)
    
    team_stats_df = pd.concat(team_stats, ignore_index=False).sort_index()
    
    # Now for each game, find opponent's rolling average goals allowed
    opponent_strength_list = []
    for idx, row in df.iterrows():
        game_id = row['game_id']
        team_id = row['team_id']
        
        # Find opponent in same game
        opponent_data = df[(df['game_id'] == game_id) & (df['team_id'] != team_id)]
        
        if len(opponent_data) > 0:
            opponent_idx = opponent_data.index[0]
            opponent_strength = team_stats_df.loc[opponent_idx, 'goals_allowed_rolling']
        else:
            opponent_strength = np.nan
        
        opponent_strength_list.append(opponent_strength)
    
    return opponent_strength_list

df['opponent_strength'] = calculate_opponent_strength(df)

# Alternative simpler measure: opponent's average goals scored
# (offensive strength of the opponent)
def get_opponent_avg_goals(df):
    """Alternative: opponent's average goals scored (offensive strength)"""
    opponent_goals = []
    for idx, row in df.iterrows():
        game_id = row['game_id']
        team_id = row['team_id']
        
        opponent_data = df[(df['game_id'] == game_id) & (df['team_id'] != team_id)]
        if len(opponent_data) > 0:
            opp_goals = opponent_data.iloc[0]['team_score']
        else:
            opp_goals = np.nan
        
        opponent_goals.append(opp_goals)
    
    return opponent_goals

df['opponent_goals_scored'] = get_opponent_avg_goals(df)

# Use opponent strength as the primary measure
print(f"\nOpponent Strength Statistics (Defense - Goals Allowed Average):")
print(df['opponent_strength'].describe())

print(f"\nOpponent Goals Scored Statistics (Offense):")
print(df['opponent_goals_scored'].describe())

# ===== 2. PREPARE DATA FOR REGRESSION =====
print("\n" + "="*90)
print("STEP 2: PREPARE DATA FOR MULTIVARIATE OLS")
print("="*90)

# Filter to games with travel data
df_model = df[df['travel_intensity'].notna()].copy()

# Select variables for regression
X_vars = ['travel_intensity', 'opponent_strength', 'is_away', 'rest_days', 'timezone_shift']
y_var = 'scoring_margin'

# Create analysis dataset
analysis_df = df_model[X_vars + [y_var]].dropna()

print(f"\nSample Size: {len(analysis_df)} games")
print(f"Variables: {X_vars}")
print(f"Dependent Variable: {y_var}")

print(f"\nDescriptive Statistics:")
print(analysis_df.describe())

# ===== 3. FIT MULTIVARIATE OLS MODEL =====
print("\n" + "="*90)
print("STEP 3: FIT MULTIPLE LINEAR REGRESSION (OLS)")
print("="*90)

# Prepare X and y
X = analysis_df[X_vars]
y = analysis_df[y_var]

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions and residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Model fit statistics
n = len(y)
k = X.shape[1]  # number of predictors
ss_total = np.sum((y - y.mean())**2)
ss_residual = np.sum(residuals**2)
ss_regression = ss_total - ss_residual

r2 = 1 - (ss_residual / ss_total)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
f_stat = (ss_regression / k) / (ss_residual / (n - k - 1))
f_pval = 1 - stats.f.cdf(f_stat, k, n - k - 1)

rmse = np.sqrt(np.sum(residuals**2) / (n - k - 1))

print(f"\nMODEL FIT STATISTICS:")
print(f"  R² Score: {r2:.4f}")
print(f"  Adjusted R²: {adj_r2:.4f}")
print(f"  F-Statistic: {f_stat:.4f} (p-value: {f_pval:.6f})")
print(f"  RMSE: {rmse:.4f}")

# Calculate standard errors and t-statistics
# Using the formula: SE = RMSE / sqrt(diagonal of X'X inverse)
try:
    X_with_const = np.column_stack([np.ones(len(X)), X])
    xxt_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    se_coefficients = rmse * np.sqrt(np.diag(xxt_inv))[1:]  # skip intercept SE for clarity
    
    t_stats = model.coef_ / se_coefficients
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
except:
    se_coefficients = np.zeros(k)
    t_stats = np.zeros(k)
    p_values = np.ones(k)

# ===== 4. REGRESSION OUTPUT TABLE =====
print("\n" + "="*90)
print("REGRESSION COEFFICIENTS & HYPOTHESIS TESTS")
print("="*90)

print(f"\n{'Variable':<25} {'Coefficient':>15} {'Std Error':>15} {'t-stat':>12} {'p-value':>12} {'Significance':<5}")
print("-"*90)

for i, var in enumerate(X_vars):
    coef = model.coef_[i]
    se = se_coefficients[i] if i < len(se_coefficients) else np.nan
    t_stat = t_stats[i] if i < len(t_stats) else np.nan
    p_val = p_values[i] if i < len(p_values) else np.nan
    
    # Significance markers
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    elif p_val < 0.10:
        sig = '.'
    else:
        sig = 'ns'
    
    print(f"{var:<25} {coef:>15.6f} {se:>15.6f} {t_stat:>12.4f} {p_val:>12.6f} {sig:<5}")

print(f"\n{'Intercept':<25} {model.intercept_:>15.6f}")
print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10, ns = not significant")

# ===== 5. HYPOTHESIS TESTS =====
print("\n" + "="*90)
print("KEY HYPOTHESIS TEST: Travel Intensity Effect")
print("="*90)

travel_idx = X_vars.index('travel_intensity')
travel_coef = model.coef_[travel_idx]
travel_se = se_coefficients[travel_idx]
travel_t = t_stats[travel_idx]
travel_p = p_values[travel_idx]

print(f"\nH0: Travel Intensity has NO effect on Scoring Margin (β₁ = 0)")
print(f"H1: Travel Intensity HAS an effect on Scoring Margin (β₁ ≠ 0)")
print(f"\nTest Statistic: t = {travel_t:.4f}")
print(f"P-value: {travel_p:.6f}")

if travel_p < 0.05:
    direction = "POSITIVE" if travel_coef > 0 else "NEGATIVE"
    print(f"\n✓ REJECT H0 at α=0.05")
    print(f"  Travel intensity has a SIGNIFICANT {direction} effect on scoring margin")
    print(f"  Interpretation: Each 0.1 increase in travel intensity → {travel_coef*0.1:.4f} goal difference")
else:
    print(f"\n✗ FAIL TO REJECT H0 at α=0.05")
    print(f"  After controlling for opponent strength, home/away, rest days, and timezone shifts,")
    print(f"  travel intensity does NOT have a statistically significant effect on scoring margin")
    print(f"  (The earlier positive correlation was likely due to confounding variables)")

# ===== 6. EFFECT INTERPRETATION =====
print("\n" + "="*90)
print("COEFFICIENT INTERPRETATION")
print("="*90)

print(f"\nTRAVEL INTENSITY (β₁ = {travel_coef:.4f}):")
if travel_p < 0.05:
    print(f"  • Each 0.1 unit increase → {travel_coef*0.1:+.4f} goals in scoring margin")
    print(f"  • Moving from Low (0.2) to High (0.6) intensity → {travel_coef*0.4:+.4f} goals")
else:
    print(f"  • NOT statistically significant (p = {travel_p:.4f})")
    print(f"  • Effect is likely explained by other variables in the model")

print(f"\nOPPONENT STRENGTH (β₂ = {model.coef_[X_vars.index('opponent_strength')]:.4f}):")
opp_p = p_values[X_vars.index('opponent_strength')]
if opp_p < 0.05:
    print(f"  • Playing stronger opponents → {model.coef_[X_vars.index('opponent_strength')]:+.4f} goal margin change")
    print(f"  • This makes intuitive sense: harder opponents = lower scoring margin")
else:
    print(f"  • NOT statistically significant")

print(f"\nHOME/AWAY (β₃ = {model.coef_[X_vars.index('is_away')]:.4f}):")
away_p = p_values[X_vars.index('is_away')]
if away_p < 0.05:
    print(f"  • Being away → {model.coef_[X_vars.index('is_away')]:+.4f} goal margin change")
else:
    print(f"  • NOT statistically significant (p = {away_p:.4f})")

print(f"\nREST DAYS (β₄ = {model.coef_[X_vars.index('rest_days')]:.4f}):")
rest_p = p_values[X_vars.index('rest_days')]
if rest_p < 0.05:
    print(f"  • Each additional rest day → {model.coef_[X_vars.index('rest_days')]:+.4f} goal margin change")
else:
    print(f"  • NOT statistically significant")

print(f"\nTIMEZONE SHIFT (β₅ = {model.coef_[X_vars.index('timezone_shift')]:.4f}):")
tz_p = p_values[X_vars.index('timezone_shift')]
if tz_p < 0.05:
    print(f"  • Each hour of timezone shift → {model.coef_[X_vars.index('timezone_shift')]:+.4f} goal margin change")
else:
    print(f"  • NOT statistically significant")

# ===== 7. MODEL DIAGNOSTICS =====
print("\n" + "="*90)
print("MODEL DIAGNOSTICS")
print("="*90)

# Normality of residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test for Normality:")
print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print(f"  ✓ Residuals appear normally distributed")
else:
    print(f"  ⚠ Residuals may not be normally distributed")

# Heteroscedasticity (Breusch-Pagan approximation)
fitted_values = y_pred
abs_residuals = np.abs(residuals)
correlation = np.corrcoef(fitted_values, abs_residuals)[0, 1]
print(f"\nHeteroscedasticity Check (residuals vs fitted):")
print(f"  Correlation: {correlation:.4f}")
if abs(correlation) < 0.3:
    print(f"  ✓ Homoscedasticity assumption appears reasonable")
else:
    print(f"  ⚠ Possible heteroscedasticity detected")

# ===== 8. CREATE VISUALIZATIONS =====
print("\n" + "="*90)
print("Generating diagnostic plots...")
print("="*90)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Travel Intensity vs Scoring Margin (with regression line)
axes[0, 0].scatter(X['travel_intensity'], y, alpha=0.6, s=100, color='steelblue')
z = np.polyfit(X['travel_intensity'], y, 1)
p = np.poly1d(z)
x_line = np.linspace(X['travel_intensity'].min(), X['travel_intensity'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')
axes[0, 0].set_xlabel('Travel Intensity', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Scoring Margin', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Travel Intensity vs Scoring Margin\n(p={travel_p:.4f}, {"Sig***" if travel_p<0.05 else "Not Sig"})', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Opponent Strength vs Scoring Margin
axes[0, 1].scatter(X['opponent_strength'], y, alpha=0.6, s=100, color='coral')
z2 = np.polyfit(X['opponent_strength'].dropna(), y[X['opponent_strength'].notna()], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(X['opponent_strength'].min(), X['opponent_strength'].max(), 100)
axes[0, 1].plot(x_line2, p2(x_line2), "r--", linewidth=2, label='Trend')
axes[0, 1].set_xlabel('Opponent Strength (Avg Goals Allowed)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Scoring Margin', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Opponent Strength vs Scoring Margin', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Home/Away effect
away_margins = y[X['is_away'] == 1]
home_margins = y[X['is_away'] == 0]
positions = [1, 2]
axes[0, 2].boxplot([home_margins, away_margins], positions=positions, labels=['Home', 'Away'],
                    patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0, 2].set_ylabel('Scoring Margin', fontsize=11, fontweight='bold')
axes[0, 2].set_title(f'Home vs Away Games\n(β={model.coef_[X_vars.index("is_away")]:.4f}, p={away_p:.4f})', 
                     fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Residuals vs Fitted Values
axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=100, color='steelblue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Residuals vs Fitted Values\n(Check for Heteroscedasticity)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot\n(Check for Normality)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Histogram of Residuals
axes[1, 2].hist(residuals, bins=12, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[1, 2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1, 2].set_title(f'Distribution of Residuals\n(Shapiro-Wilk p={shapiro_p:.4f})', 
                     fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/multivariate_regression_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multivariate_regression_diagnostics.png")

# Create coefficient plot
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by absolute value of coefficients
coef_df = pd.DataFrame({
    'Variable': X_vars,
    'Coefficient': model.coef_,
    'P-Value': p_values,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in p_values]
})
coef_df = coef_df.sort_values('Coefficient', ascending=True)

colors = ['steelblue' if sig == 'Yes' else 'lightgray' for sig in coef_df['Significant']]
bars = ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors, edgecolor='black', alpha=0.8)

ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Multiple Linear Regression Coefficients\n(Blue = Significant at p<0.05)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (idx, row) in enumerate(coef_df.iterrows()):
    ax.text(row['Coefficient'], i, f"  {row['Coefficient']:.4f} ", 
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/regression_coefficients_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: regression_coefficients_plot.png")

# ===== 9. CONCLUSION =====
print("\n" + "="*90)
print("CONCLUSION")
print("="*90)

print(f"\nModel explains {r2*100:.1f}% of variance in scoring margin (R² = {r2:.4f})")

if travel_p < 0.05:
    direction = "INCREASES" if travel_coef > 0 else "DECREASES"
    print(f"\n✓ TRAVEL INTENSITY {direction} SCORING MARGIN")
    print(f"  • Effect is statistically significant (p = {travel_p:.6f})")
    print(f"  • After controlling for opponent strength, home/away, rest days, and timezone,")
    print(f"    travel intensity still has a meaningful effect")
    print(f"  • Coefficient: {travel_coef:.6f} (per 1.0 unit increase)")
else:
    print(f"\n✗ NO SIGNIFICANT TRAVEL INTENSITY EFFECT")
    print(f"  • The positive bivariate correlation was CONFOUNDED by:")
    
    # Check which variables are significant
    confounders = []
    for i, var in enumerate(X_vars):
        if i != travel_idx and p_values[i] < 0.05:
            confounders.append(f"{var} (p={p_values[i]:.4f})")
    
    if confounders:
        print(f"    - {', '.join(confounders)}")
    else:
        print(f"    - No statistically significant confounders (small sample size effect)")
    
    print(f"  • Travel intensity coefficient: {travel_coef:.6f} (p = {travel_p:.4f})")
    print(f"  • This suggests the earlier positive relationship was spurious")

print("\n" + "="*90)
