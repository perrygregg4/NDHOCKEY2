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

print("="*80)
print("LINEAR REGRESSION ANALYSIS: Travel Intensity & Team Performance")
print("="*80)

# Filter to games with travel data (non-first games)
df_with_travel = df[df['travel_intensity'].notna()].copy()

print(f"\nAnalyzing {len(df_with_travel)} games with travel data")
print(f"(First games excluded as they have no travel indicator)")

# ===== MODEL 1: SCORING MARGIN =====
print("\n" + "="*80)
print("MODEL 1: SCORING MARGIN vs TRAVEL INTENSITY")
print("="*80)

# Prepare data
X1 = df_with_travel[['travel_intensity', 'is_away', 'rest_days', 'timezone_shift']].dropna()
y1 = df_with_travel.loc[X1.index, 'scoring_margin']

print(f"\nSample size: {len(X1)} games")

# Fit model
model1 = LinearRegression()
model1.fit(X1, y1)

# Predictions and residuals
y1_pred = model1.predict(X1)
residuals1 = y1 - y1_pred
r2_score1 = model1.score(X1, y1)
adj_r2_1 = 1 - (1 - r2_score1) * (len(y1) - 1) / (len(y1) - X1.shape[1] - 1)

# Calculate RMSE and standard error
rmse1 = np.sqrt(np.mean(residuals1**2))
std_error1 = rmse1 / np.sqrt(len(y1))

# T-statistics and p-values
t_stats1 = model1.coef_ / (rmse1 / np.sqrt(np.diag(np.linalg.inv(X1.T @ X1))))
p_values1 = 2 * (1 - stats.t.cdf(np.abs(t_stats1), len(y1) - X1.shape[1]))

print("\nModel Performance:")
print(f"  R² Score: {r2_score1:.4f}")
print(f"  Adjusted R²: {adj_r2_1:.4f}")
print(f"  RMSE: {rmse1:.4f}")

print("\nRegression Coefficients:")
print(f"  {'Variable':<25} {'Coefficient':>15} {'Std Error':>15} {'t-stat':>12} {'p-value':>12}")
print(f"  {'-'*80}")
feature_names1 = ['travel_intensity', 'is_away', 'rest_days', 'timezone_shift']
for i, feature in enumerate(feature_names1):
    coef = model1.coef_[i]
    t_stat = t_stats1[i]
    p_val = p_values1[i]
    se = rmse1 / np.sqrt(np.diag(np.linalg.inv(X1.T @ X1)))[i]
    significance = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'ns'
    print(f"  {feature:<25} {coef:>15.6f} {se:>15.6f} {t_stat:>12.4f} {p_val:>12.4f} {significance}")

print(f"\n  Intercept: {model1.intercept_:.6f}")

print("\nInterpretation:")
if model1.coef_[0] < 0:
    print(f"  • A 0.1 increase in travel intensity → {model1.coef_[0]*0.1:.4f} goal scoring margin")
    print("    (High travel intensity is associated with LOWER scoring margin)")
else:
    print(f"  • A 0.1 increase in travel intensity → {model1.coef_[0]*0.1:.4f} goal scoring margin")
    print("    (High travel intensity is associated with HIGHER scoring margin)")

print(f"  • Away games: {model1.coef_[1]:+.4f} goal difference")
print(f"  • Each additional rest day: {model1.coef_[2]:+.4f} goal difference")
print(f"  • Each hour timezone shift: {model1.coef_[3]:+.4f} goal difference")

# ===== MODEL 2: WIN/LOSS =====
print("\n" + "="*80)
print("MODEL 2: WIN PROBABILITY vs TRAVEL INTENSITY (Logistic via OLS)")
print("="*80)

X2 = df_with_travel[['travel_intensity', 'is_away', 'rest_days', 'timezone_shift']].dropna()
y2_raw = df_with_travel.loc[X2.index, 'is_win']
y2 = y2_raw.fillna(0).astype(int)

print(f"\nSample size: {len(X2)} games")
print(f"Win rate overall: {y2.mean()*100:.1f}%")

# Fit linear probability model
model2 = LinearRegression()
model2.fit(X2, y2)

y2_pred = model2.predict(X2)
residuals2 = y2 - y2_pred
r2_score2 = model2.score(X2, y2)
adj_r2_2 = 1 - (1 - r2_score2) * (len(y2) - 1) / (len(y2) - X2.shape[1] - 1)
rmse2 = np.sqrt(np.mean(residuals2**2))

# T-statistics and p-values
t_stats2 = model2.coef_ / (rmse2 / np.sqrt(np.diag(np.linalg.inv(X2.T @ X2))))
p_values2 = 2 * (1 - stats.t.cdf(np.abs(t_stats2), len(y2) - X2.shape[1]))

print("\nModel Performance:")
print(f"  R² Score: {r2_score2:.4f}")
print(f"  Adjusted R²: {adj_r2_2:.4f}")
print(f"  RMSE: {rmse2:.4f}")

print("\nRegression Coefficients (Linear Probability Model):")
print(f"  {'Variable':<25} {'Coefficient':>15} {'Std Error':>15} {'t-stat':>12} {'p-value':>12}")
print(f"  {'-'*80}")
feature_names2 = ['travel_intensity', 'is_away', 'rest_days', 'timezone_shift']
for i, feature in enumerate(feature_names2):
    coef = model2.coef_[i]
    t_stat = t_stats2[i]
    p_val = p_values2[i]
    se = rmse2 / np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)))[i]
    significance = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'ns'
    print(f"  {feature:<25} {coef:>15.6f} {se:>15.6f} {t_stat:>12.4f} {p_val:>12.4f} {significance}")

print(f"\n  Intercept: {model2.intercept_:.6f}")

print("\nInterpretation (percentage point change in win probability):")
if model2.coef_[0] < 0:
    print(f"  • A 0.1 increase in travel intensity → {model2.coef_[0]*0.1*100:+.1f}% change in win probability")
    print("    (High travel intensity is associated with LOWER win probability)")
else:
    print(f"  • A 0.1 increase in travel intensity → {model2.coef_[0]*0.1*100:+.1f}% change in win probability")
    print("    (High travel intensity is associated with HIGHER win probability)")

print(f"  • Being away: {model2.coef_[1]*100:+.1f}% change in win probability")
print(f"  • Each additional rest day: {model2.coef_[2]*100:+.1f}% change in win probability")
print(f"  • Each hour timezone shift: {model2.coef_[3]*100:+.1f}% change in win probability")

# ===== CORRELATION ANALYSIS =====
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_vars = ['travel_intensity', 'travel_distance', 'rest_days', 'timezone_shift', 
             'scoring_margin', 'is_win', 'team_score', 'sGoalsAllowed']
corr_data = df_with_travel[corr_vars].dropna()

print(f"\nCorrelation Matrix ({len(corr_data)} observations):")
corr_matrix = corr_data.corr()

# Print significant correlations
print("\nSignificant Correlations with Scoring Margin:")
margin_corrs = corr_data.corr()['scoring_margin'].sort_values(ascending=False)
for var, corr in margin_corrs.items():
    if var != 'scoring_margin':
        # Calculate p-value
        n = len(corr_data)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        significance = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
        print(f"  {var:<25}: {corr:>8.4f} (p={p_val:.4f}) {significance}")

print("\nSignificant Correlations with Win Rate:")
win_corrs = corr_data.corr()['is_win'].sort_values(ascending=False)
for var, corr in win_corrs.items():
    if var != 'is_win':
        n = len(corr_data)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        significance = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
        print(f"  {var:<25}: {corr:>8.4f} (p={p_val:.4f}) {significance}")

# ===== CREATE VISUALIZATIONS =====
print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Travel Intensity vs Scoring Margin
axes[0, 0].scatter(df_with_travel['travel_intensity'], df_with_travel['scoring_margin'], 
                   alpha=0.6, s=100, c='steelblue')
z1 = np.polyfit(X1['travel_intensity'], y1, 1)
p1 = np.poly1d(z1)
x_line = np.linspace(df_with_travel['travel_intensity'].min(), 
                     df_with_travel['travel_intensity'].max(), 100)
axes[0, 0].plot(x_line, p1(x_line), "r--", linewidth=2, label='Trend line')
axes[0, 0].set_xlabel('Travel Intensity (0-1)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Scoring Margin (Goals)', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Travel Intensity vs Scoring Margin\nR² = {r2_score1:.3f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Rest Days vs Scoring Margin
axes[0, 1].scatter(df_with_travel['rest_days'], df_with_travel['scoring_margin'], 
                   alpha=0.6, s=100, c='forestgreen')
z2 = np.polyfit(X1['rest_days'].dropna(), y1[X1['rest_days'].notna()], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df_with_travel['rest_days'].min(), 
                      df_with_travel['rest_days'].max(), 100)
axes[0, 1].plot(x_line2, p2(x_line2), "r--", linewidth=2, label='Trend line')
axes[0, 1].set_xlabel('Rest Days', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Scoring Margin (Goals)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Rest Days vs Scoring Margin', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Travel Distance vs Scoring Margin
axes[1, 0].scatter(df_with_travel['travel_distance'], df_with_travel['scoring_margin'], 
                   alpha=0.6, s=100, c='coral')
axes[1, 0].set_xlabel('Travel Distance (Miles)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Scoring Margin (Goals)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Travel Distance vs Scoring Margin', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Win Rate by Travel Intensity Category
win_rates = df_with_travel.groupby('travel_intensity_category')['is_win'].mean() * 100
categories = ['Low', 'Medium', 'High', 'First Game']
win_data = []
for cat in categories:
    if cat in win_rates.index:
        win_data.append(win_rates[cat])
    else:
        win_data.append(0)

colors_bars = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
bars = axes[1, 1].bar(categories, win_data, color=colors_bars, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Win Rate by Travel Intensity Category', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim(0, 100)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/travel_intensity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: travel_intensity_analysis.png")

# Create correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix: Travel & Performance Variables', 
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/perrygregg/Downloads/travel_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: travel_correlation_heatmap.png")

# ===== SUMMARY REPORT =====
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\nKey Findings:")
if abs(model1.coef_[0]) > 0.5:
    print(f"  ✓ Travel intensity has SIGNIFICANT impact on scoring margin")
    print(f"    (Coefficient: {model1.coef_[0]:.4f})")
else:
    print(f"  ⚠ Travel intensity has LIMITED impact on scoring margin")
    print(f"    (Coefficient: {model1.coef_[0]:.4f})")

if r2_score1 > 0.3:
    print(f"  ✓ Model 1 explains {r2_score1*100:.1f}% of variance in scoring margin")
elif r2_score1 > 0.1:
    print(f"  ⚠ Model 1 explains only {r2_score1*100:.1f}% of variance (limited predictive power)")
else:
    print(f"  ✗ Model 1 explains only {r2_score1*100:.1f}% of variance (poor fit)")

print(f"\n  • Away games have {model2.coef_[1]*100:+.1f}% win probability impact")
print(f"  • Rest days matter: {model2.coef_[2]*100:+.1f}% per additional day")
print(f"  • Timezone shifts affect wins: {model2.coef_[3]*100:+.1f}% per hour")

print("\nRecommendations for Further Analysis:")
print("  1. Include interaction terms (e.g., travel_intensity × is_away)")
print("  2. Test non-linear relationships with polynomial terms")
print("  3. Use logistic regression for binary outcome (win/loss)")
print("  4. Control for opponent strength / team quality")
print("  5. Analyze by season and conference strength")
print("  6. Consider lagged effects (travel impact on next 2-3 games)")

print("\n" + "="*80)
