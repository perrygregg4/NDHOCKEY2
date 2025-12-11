# Multiple Linear Regression Analysis: Travel Intensity & Scoring Margin

## Executive Summary

**The initial positive correlation between travel intensity and scoring margin was SPURIOUS.**

When we control for opponent strength, home/away status, rest days, and timezone shifts, **travel intensity has NO statistically significant effect on scoring margin** (p = 0.621).

---

## Model Specification

### Equation
```
Scoring Margin ~ β₀ + β₁(Travel Intensity) + β₂(Opponent Strength) + β₃(Home/Away) + β₄(Rest Days) + β₅(Timezone Shift) + ε
```

### Sample
- **N = 38 games** (only games with prior travel history)
- **Dependent Variable:** Scoring Margin (Goals For - Goals Against)
- **Predictors:** 5 independent variables

---

## Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.4372 |
| Adjusted R² | 0.3492 |
| F-Statistic | 4.97 (p = 0.0018) |
| RMSE | 2.47 goals |

**Interpretation:** The model explains 43.7% of variance in scoring margin. This is a substantial improvement over the bivariate models (R² = 0.18), indicating that opponent strength is a critical control variable.

---

## Regression Results

### Coefficients & Significance Tests

| Variable | Coefficient | Std Error | t-stat | p-value | Significant? |
|----------|------------|-----------|--------|---------|--------------|
| **Travel Intensity** | **2.333** | **4.670** | **0.500** | **0.621** | **NO** |
| **Opponent Strength** | **1.095** | **0.286** | **3.834** | **0.0006** | **YES ✓** |
| Home/Away (is_away) | 1.179 | 0.835 | 1.413 | 0.167 | NO |
| Rest Days | -0.021 | 0.050 | -0.410 | 0.685 | NO |
| Timezone Shift | 0.468 | 1.457 | 0.321 | 0.750 | NO |
| **Intercept** | **-5.292** | — | — | — | — |

**Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, . p<0.10

---

## Key Finding: The Confounding Story

### Before (Bivariate Correlation)
```
Travel Intensity → +0.324 correlation with Scoring Margin (p = 0.051)
→ Appeared marginally significant!
```

### After (Multivariate OLS Control)
```
Travel Intensity → β = 2.333 (p = 0.621)
→ NOT significant after controlling for other factors
```

### Why the Difference?

**Opponent Strength is a Confounder:**
- Teams that travel more may face weaker opponents on average
- This creates a spurious positive correlation: Travel → Higher Scoring Margin
- But this is really: Weaker Opponents → Higher Scoring Margin

When we control for opponent strength (p = 0.0006, highly significant), the travel intensity effect disappears.

---

## Individual Variable Interpretations

### 1. **Travel Intensity (NOT Significant)**
- **Coefficient:** 2.333 (p = 0.621)
- **Interpretation:** Each 0.1 increase in travel intensity is associated with +0.233 goal margin, but this is NOT statistically significant
- **Conclusion:** Travel intensity does NOT have a meaningful effect on performance when opponent strength is controlled for

### 2. **Opponent Strength (HIGHLY SIGNIFICANT ✓✓✓)**
- **Coefficient:** 1.095 (p = 0.0006) ***
- **Interpretation:** Each 1.0 increase in opponent strength (goals allowed average) → +1.095 goal scoring margin
- **Why it seems counterintuitive:** Higher opponent strength = team allows more goals = weaker defense. The positive coefficient suggests we're playing against weaker opponents when our scoring margin is higher
- **This is the dominant driver of scoring margin variation**

### 3. **Home/Away (Marginally Not Significant)**
- **Coefficient:** 1.179 (p = 0.167)
- **Interpretation:** Away games → +1.179 goal margin
- **Note:** This is counterintuitive and not statistically significant
- **Possible explanation:** Small sample size (38 games)

### 4. **Rest Days (NOT Significant)**
- **Coefficient:** -0.021 (p = 0.685)
- **Interpretation:** Each additional rest day → -0.021 goal margin
- **Conclusion:** Rest does NOT have a significant effect (in this small sample)

### 5. **Timezone Shift (NOT Significant)**
- **Coefficient:** 0.468 (p = 0.750)
- **Interpretation:** Each hour of timezone shift → +0.468 goal margin
- **Conclusion:** Timezone adjustments do NOT significantly impact performance

---

## Model Diagnostics

### Normality of Residuals ✓
- **Shapiro-Wilk Test:** p = 0.278
- **Interpretation:** Residuals are normally distributed (meets OLS assumption)

### Homoscedasticity ✓
- **Residuals vs Fitted:** r = -0.131
- **Interpretation:** Constant variance assumption appears satisfied

### Overall Model Validity
- ✓ F-statistic significant (p = 0.0018)
- ✓ Normality assumption met
- ✓ Homoscedasticity assumption met
- ✓ No severe multicollinearity issues

---

## Statistical Conclusion

### Hypothesis Test Results

**H₀:** Travel Intensity has NO effect on Scoring Margin (β₁ = 0)
**H₁:** Travel Intensity HAS an effect on Scoring Margin (β₁ ≠ 0)

**Test Result:** FAIL TO REJECT H₀
- t-statistic: 0.500
- p-value: 0.621
- Conclusion: Insufficient evidence to conclude travel intensity affects scoring margin

---

## Substantive Interpretation

### What This Means

1. **The earlier positive correlation was CONFOUNDED**
   - Opponent strength explains the relationship
   - High travel intensity didn't improve performance; teams just faced weaker opponents

2. **Travel stress does NOT significantly impact team performance**
   - After controlling for meaningful variables, no travel effect remains
   - Other factors (opponent quality, team ability) dominate

3. **Opponent strength is the CRITICAL factor**
   - Highly significant (p = 0.0006)
   - Much stronger effect than any travel variable
   - Teams perform better against weaker opponents (obvious, but now quantified)

4. **Rest days, timezone shifts, and home/away status are not significant predictors**
   - In this sample, these variables don't meaningfully predict scoring margin
   - May be due to small sample size (N=38)

---

## Limitations & Considerations

1. **Sample Size:** Only 38 games (first game excluded, missing data)
2. **Opponent Strength Measurement:** Using rolling average of goals allowed as proxy
3. **Multicollinearity:** Variables may be correlated (though not severely)
4. **Model Specification:** Could include interaction terms (e.g., Travel × Home/Away)
5. **Alternative Measures:** Could use opponent quality ratings, conference strength, etc.

---

## Recommendations for Further Analysis

1. **Expand Sample:** Include multiple seasons to increase power
2. **Better Opponent Strength:** Use preseason rankings, RPI, or Pairwise rankings
3. **Interaction Terms:** Test Travel Intensity × Home/Away, Travel × Opponent Strength
4. **Lagged Effects:** Does travel impact the NEXT game, not the current one?
5. **Non-linear Effects:** Travel intensity measured in categories (Low/Medium/High)
6. **Win/Loss Model:** Use logistic regression for binary outcome
7. **Position-Specific Analysis:** Does travel affect forwards vs defensemen differently?

---

## Files Generated

1. `multivariate_regression_diagnostics.png` - 6-panel diagnostic plot
2. `regression_coefficients_plot.png` - Coefficient comparison chart
3. `multivariate_ols_regression.py` - Complete analysis script

---

## Conclusion

**Travel intensity does NOT have a statistically significant effect on scoring margin after controlling for opponent strength and other factors.** The initial positive correlation was due to confounding: teams that travel more may face systematically weaker opponents, creating a spurious relationship. When this confounder is removed, the travel effect disappears entirely.

This finding suggests that **travel stress is not the binding constraint on team performance** at the college hockey level. Other factors—particularly opponent quality—are far more important.
