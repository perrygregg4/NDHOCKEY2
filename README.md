# ND Hockey Travel Impact Analysis

This repository contains a comprehensive multivariate regression analysis examining the relationship between travel intensity and hockey team performance.

## Overview

This analysis investigates whether travel intensity (combining travel distance, rest days, and timezone shifts) significantly impacts team performance metrics, while controlling for confounding variables such as opponent strength.

**Main Question:** Does travel stress reduce scoring margin? Or is any observed relationship spurious?

## Key Finding

**The positive correlation between travel intensity and scoring margin was SPURIOUS.** After controlling for opponent strength, travel intensity has NO statistically significant effect on performance (p = 0.621).

## Files & Analysis

### Data Files
- `NDHockey_enriched.csv` - Enriched dataset with engineered travel variables
  - travel_distance: Miles between consecutive game venues
  - rest_days: Days between games
  - timezone_shift: Hours difference from previous game timezone
  - travel_intensity: Composite 0-1 index (40% distance, 40% rest, 20% timezone)
  - travel_intensity_category: Low/Medium/High/First Game
  - opponent_strength: Rolling average of opponent goals allowed
  - scoring_margin: Goals for minus goals against

### Analysis Scripts
- `engineer_travel_variables.py` - Creates all travel-related features from raw data
- `linear_regression_travel.py` - Bivariate regression models
- `multivariate_ols_regression.py` - Multivariate OLS with opponent strength control

### Results & Visualizations
- `travel_intensity_analysis.png` - 4-panel scatter plots of travel metrics
- `travel_correlation_heatmap.png` - Correlation matrix heatmap
- `multivariate_regression_diagnostics.png` - 6-panel OLS diagnostic plots
- `regression_coefficients_plot.png` - Coefficient comparison chart
- `MULTIVARIATE_REGRESSION_SUMMARY.md` - Detailed interpretation guide

## Key Results

### Model Performance
| Metric | Value |
|--------|-------|
| R² (Variance Explained) | 43.7% |
| Adjusted R² | 34.9% |
| F-Statistic | 4.97 (p = 0.0018) |
| Sample Size | 38 games |

### Regression Coefficients
| Variable | Coefficient | p-value | Significant? |
|----------|------------|---------|--------------|
| Travel Intensity | 2.333 | 0.621 | **NO** |
| Opponent Strength | 1.095 | 0.0006 | **YES ✓✓✓** |
| Home/Away | 1.179 | 0.167 | NO |
| Rest Days | -0.021 | 0.685 | NO |
| Timezone Shift | 0.468 | 0.750 | NO |

### The Confounding Story

**Before (Bivariate):**
- Travel Intensity correlation: r = 0.324 (p = 0.051) — marginally significant!

**After (Multivariate Control):**
- Travel Intensity coefficient: β = 2.333 (p = 0.621) — NOT significant
- Opponent Strength coefficient: β = 1.095 (p = 0.0006) — HIGHLY significant

**Interpretation:** Teams that traveled more just happened to face weaker opponents. When opponent strength is controlled for, the travel effect disappears entirely.

## Analysis Methodology

### Model Specification
```
Scoring Margin ~ β₀ + β₁(Travel Intensity) + β₂(Opponent Strength) 
                + β₃(Home/Away) + β₄(Rest Days) + β₅(Timezone Shift) + ε
```

### Key Features
- ✓ Opponent strength engineered using rolling average of opponent goals allowed
- ✓ Travel intensity calculated using weighted composite index
- ✓ OLS assumptions verified (normality, homoscedasticity)
- ✓ Multicollinearity checked
- ✓ Model diagnostics validated

## Conclusions

1. **Travel stress is not a significant performance predictor** (after controlling for confounding)
2. **Opponent quality is the dominant factor** (p < 0.001) in determining scoring margin
3. **The initial positive correlation was an artifact** of confounding by opponent strength
4. **Rest days and timezone shifts show no significant effects** in this sample

## Limitations

- Small sample size (N=38 games)
- Opponent strength measured as rolling average of goals allowed (proxy measure)
- Limited to ND Fighting Irish 2025-2026 season
- Could benefit from opponent quality rankings or RPI

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## How to Run

```bash
# Engineer travel variables from raw data
python3 engineer_travel_variables.py

# Run bivariate analysis
python3 linear_regression_travel.py

# Run multivariate OLS analysis
python3 multivariate_ols_regression.py
```

## Author

Perry Gregg

## Date

December 2025

## License

MIT
