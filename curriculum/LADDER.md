# Learning Regression Through the Sugar Paradox

A ladder curriculum. Each rung teaches one STAT 538 concept using the
Sugar Paradox analysis panel (37 SSA countries, 2010-2022, 962 rows).
Every exercise produces a real number from the paper. By the top you
understand both the statistics and every claim in your own publication.

Data: `data/derived/analysis_panel.csv`
Work in: `curriculum/exercises/`

---

## Rung 1: Vectors and Correlation
**STAT 538 Unit:** M1 (Simple Linear Regression basics)
**Paper connection:** The raw bivariate r = 0.677 (Figure 1A)

### What to learn
Each column of data is a vector. The correlation between two vectors is
the cosine of the angle between them (after centering). When r = 0.677,
the angle between centered sugar-supply and obesity vectors is about 47
degrees.

### Exercise
1. Load the panel. Filter to women, average each variable over 2018-2020
   to get 37 country means.
2. Center both sugar and obesity (subtract their means).
3. Compute the dot product of the centered vectors. Divide by the product
   of their lengths. That number is r.
4. Verify: `scipy.stats.pearsonr` gives the same answer.
5. Compute the angle in degrees: `arccos(r) * 180 / pi`.

### What you should see
r = 0.677. Angle ~ 47 degrees. Countries with high sugar supply and
high obesity point in roughly the same direction in 37-dimensional
observation space.

### Paper number produced
Figure 1A bivariate r = 0.677, p < 0.001.

---

## Rung 2: Simple Regression as Projection
**STAT 538 Unit:** M1 (The projection view)
**Paper connection:** Pooled OLS sugar coefficient, t = 19.84

### What to learn
Regression takes Y (obesity, a vector in 481-dimensional space for the
full women's panel) and finds the closest point in the column space of X.
"Closest" means perpendicular drop. The fitted values are the shadow.
The residuals are the height. R-squared is how much shorter the
residual vector is compared to the centered Y vector.

### Exercise
1. Use the full women's panel (481 rows). Fit `obesity ~ sugar + oil`
   with OLS (no fixed effects).
2. Extract fitted values and residuals.
3. Verify: residuals are orthogonal to each predictor
   (`np.dot(residuals, sugar)` is ~0).
4. Compute R-squared two ways: (a) from the model, (b) as
   `1 - (||residuals||^2 / ||centered_Y||^2)`.
5. Look at the sugar coefficient and its t-statistic.

### What you should see
Sugar beta = 0.630, t = 19.84, R^2 ~ 0.49. The projection explains
about half the variance. The sugar coefficient looks huge and
overwhelmingly significant.

### Paper number produced
Table 1 row 1: Pooled OLS, beta = 0.630, t = 19.84.

### What to notice
This is the number that makes sugar look like it drives obesity.
Every rung from here forward will erode it.

---

## Rung 3: Multiple Regression and Partial Correlation
**STAT 538 Unit:** M4 (Multiple Linear Regression)
**Paper connection:** Partial r = 0.496 (Figure 1B)

### What to learn
In multiple regression, each coefficient measures the relationship
between that predictor and Y AFTER removing what the other predictors
explain. The partial correlation is the correlation between the
residualized X and the residualized Y.

### Exercise
1. Use the 37 country means (2018-2020 averages).
2. Regress sugar on GDP + urbanization + oil. Save residuals (call
   them sugar_resid).
3. Regress obesity on GDP + urbanization + oil. Save residuals (call
   them obesity_resid).
4. Correlate sugar_resid with obesity_resid. That is the partial r.
5. Plot sugar_resid vs obesity_resid. That is Figure 1B.

### What you should see
Partial r = 0.496 (p = 0.002). Even after removing GDP, urbanization,
and vegetable oil, sugar supply still predicts obesity across countries.
The cross-sectional case looks strong.

### Paper number produced
Figure 1B: partial r = 0.50, p = 0.002.

### Concept check
Why is the partial r (0.496) lower than the bivariate r (0.677)?
Because GDP and urbanization explain some of the sugar-obesity overlap.
But 0.496 is still strong. The cross-sectional signal survives
controlling for observable confounders.

---

## Rung 4: The F-test as Comparing Projections
**STAT 538 Unit:** M5 (Hypothesis Testing in MLR)
**Paper connection:** Overall model F-test, partial F for sugar

### What to learn
Every F-test compares two projections. The small model projects Y onto
a smaller subspace. The big model projects onto a bigger one. F measures
whether the improvement in fit (per added dimension) exceeds what you
would expect from noise.

### Exercise
1. Fit the reduced model: `obesity ~ oil` (no sugar).
2. Fit the full model: `obesity ~ sugar + oil`.
3. Compute the partial F-statistic by hand:
   `F = [(RSS_reduced - RSS_full) / 1] / [RSS_full / (n - p)]`
4. Verify: this equals the square of the sugar t-statistic from Rung 2.
5. Use `scipy.stats.f` to get the p-value from the F-distribution.

### What you should see
t^2 = F exactly. The t-test for a single predictor IS the F-test
comparing "model with sugar" to "model without sugar." They are the
same geometric operation.

### Paper number produced
Not a specific paper number, but the foundation for understanding why
t = 19.84 (Rung 2) will collapse to t = 0.25 (Rung 5).

---

## Rung 5: Fixed Effects as Expanding the Subspace
**STAT 538 Unit:** M4/M5 (MLR, the big conceptual leap)
**Paper connection:** Country FE sugar t = 0.25 (Table 1 row 2)

### What to learn
Adding 36 country dummy variables expands the column space of X from
3 dimensions (intercept + sugar + oil) to 39 dimensions. The projection
of Y onto this bigger subspace is much closer to Y (R^2 jumps from
0.49 to 0.96). But sugar's UNIQUE contribution -- the improvement from
adding sugar to a subspace that already contains country dummies --
shrinks to nearly nothing.

This is the core of the paradox. Between-country differences in obesity
levels are large and correlated with sugar. But those level differences
are absorbed by country dummies. What's left for sugar to explain is
only within-country variation -- and there, sugar does nothing.

### Exercise
1. Fit `obesity ~ sugar + oil + C(iso3)` (country fixed effects).
2. Note: R^2 ~ 0.96. The country dummies alone explain 95.7%.
3. Look at the sugar coefficient: beta ~ 0.035, t ~ 0.25.
4. Compute the partial F for sugar in this model. It is tiny.
5. Compare: in Rung 2, adding sugar to an intercept-only model was
   huge. Here, adding sugar to a model that already has country FE
   is nothing.

### What you should see
Sugar beta drops from 0.630 to 0.035. t drops from 19.84 to 0.25.
The signal was entirely between-country. Within countries, sugar supply
does not predict obesity.

### Paper number produced
Table 1 row 2: Country FE, beta = 0.035, t = 0.25.

### The question this raises
Is the null because sugar truly does not matter within countries? Or
because there is not enough within-country variation to detect it?
The next rungs investigate.

---

## Rung 6: Two-Way Fixed Effects and Variance Decomposition
**STAT 538 Unit:** M4/M5 (MLR, nested model comparison)
**Paper connection:** TWFE t = -0.64, 99.4% variance absorbed (Figure 6)

### What to learn
Year fixed effects absorb the shared upward trend in obesity common
to all countries. After adding 12 year dummies (on top of 36 country
dummies), the model explains 99.43% of variance. The residual is 0.57%.
Any food variable must compete in that 0.57%.

### Exercise
1. Fit `obesity ~ sugar + oil + C(iso3) + C(year)` (TWFE).
2. Note R^2 = 0.9943. Residual = 0.57%.
3. Sugar coefficient: beta = -0.029, t = -0.64.
4. Now fit WITHOUT sugar: `obesity ~ C(iso3) + C(year)`. Note R^2
   is still 0.9943. Sugar adds essentially nothing.
5. Compute the variance decomposition:
   - Entity FE alone: R^2 = 95.7%
   - Entity + year FE: R^2 = 99.4%
   - Year FE incremental: 3.7%
   - Residual: 0.6%

### What you should see
Entity FE absorb level differences (95.7%). Year FE absorb the shared
trend (3.7%). Together: 99.4%. Sugar competes in the remaining 0.6%.
That space is dominated by measurement noise.

### Paper number produced
Table 1 row 3: TWFE, beta = -0.029, t = -0.64.
Figure 6: 95.7% / 3.7% / 0.6% decomposition.

---

## Rung 7: The Independence Assumption and Serial Correlation
**STAT 538 Unit:** M2/M7 (Assumptions and Diagnostics)
**Paper connection:** AR(1) = 0.90-0.98, SE inflation 2.6-2.9x

### What to learn
The assumption cascade (your Unit 5): linearity > independence >
constant variance > normality. In panel data, the independence
assumption fails because observations within the same country are
serially correlated. A country with high obesity in 2015 has high
obesity in 2016. This inflates the apparent precision of OLS standard
errors.

### Exercise
1. Fit the TWFE model from Rung 6. Extract residuals.
2. For each country, compute the AR(1) correlation of residuals
   (correlate residuals[t] with residuals[t+1]).
3. Average across countries. This is the mean within-country
   autocorrelation of TWFE residuals.
4. Now compare standard errors:
   - Fit TWFE with default (non-clustered) SEs.
   - Fit TWFE with country-clustered SEs.
   - Compute the ratio: clustered SE / non-clustered SE.

### What you should see
Mean AR(1) somewhere between 0.90 and 0.98 (depending on whether
you use sugar+oil or sugar-only TWFE). SE inflation: 2.6-2.9x.
The non-clustered t for sugar might be 1.5-2.0 -- looking borderline.
Clustered t: 0.25-0.64. The difference is entirely from correcting the
independence violation.

### Paper number produced
Section 4.2: SE inflation 2.6-2.9x. Mean AR(1) of residuals.

### Why this matters for the paper's argument
Published studies that do not cluster standard errors in country-level
panels overstate significance by this factor. A coefficient with
non-clustered t = 2.0 has clustered t ~ 0.7. The paper names
Lin et al. (2018) as an example.

---

## Rung 8: Detrending and First Differences
**STAT 538 Unit:** M3 (Transformations) / M7 (Diagnostics for MLR)
**Paper connection:** Detrended t = -0.13, FD r = -0.03

### What to learn
Detrending and first-differencing are transformations that remove
specific kinds of confounding. Detrending removes each country's
linear time trend (a transformation on the data before fitting).
First-differencing replaces levels with year-to-year changes. Both
destroy the between-country signal AND the shared trend, leaving only
short-run deviations.

### Exercise
1. Detrend: for each country, fit a linear trend to obesity, sugar,
   and oil over time. Subtract the fitted trend. Refit the regression
   on detrended data.
2. First-difference: for each country, compute year-to-year changes
   in obesity and sugar. Correlate the changes.
3. Long-difference: compute 2010-to-2020 changes for each country.
   Correlate the changes across countries.

### What you should see
- Detrended sugar t = -0.13 (dead null).
- First differences r = -0.03 (dead null).
- Long differences r = -0.19 (null, p = 0.25).
Three different ways to strip out trends. All null.

### Paper number produced
Table 1 row 4: Detrended, t = -0.13.
Section 3.2: FD r = -0.03, LD r = -0.19.

---

## Rung 9: The Mundlak/CRE Decomposition
**STAT 538 Unit:** M5 (Hypothesis Testing) / Causal inference preview
**Paper connection:** Within t = -0.66, Between t = 3.46

### What to learn
The correlated random effects model includes both the time-varying
sugar supply AND its country mean as regressors. The coefficient on
the time-varying part estimates the within-country association. The
coefficient on the mean estimates the between-country association.
This cleanly separates the two sources of variation.

### Exercise
1. For each country, compute mean sugar over all years.
2. Compute within-country deviation: sugar_within = sugar - country_mean.
3. Fit: `obesity ~ sugar_within + sugar_between + oil_within + oil_between + C(year)`.
4. Cluster SEs by country.
5. Read the within and between sugar coefficients.

### What you should see
Within-country sugar: t = -0.66 (null).
Between-country sugar: t = 3.46 (significant).
The entire cross-sectional association is between-country. Within
countries, sugar and obesity are unrelated.

### Paper number produced
Section 3.2: CRE within t = -0.66, between t = 3.46.

### The causal inference lesson
Between-country variation is confounded by development level, food
system structure, urbanization history -- everything that differs
between rich and poor countries. Within-country variation strips this
out. The fact that only between-country sugar predicts obesity means
the cross-sectional association reflects development gradients, not
dietary effects.

---

## Rung 10: Simulation -- Bootstrap and Permutation
**STAT 538 Unit:** M5 (Simulation-based inference)
**Paper connection:** Bootstrap p = 0.80/0.58, Permutation p = 0.58

### What to learn
When you don't trust the asymptotic distribution (here: only 37
clusters, strong serial correlation), you can build the null
distribution yourself. Permutation asks: "if sugar had no
within-country relationship with obesity, what would the t-statistic
look like?" Bootstrap asks: "how stable is my estimate?"

### Exercise
1. Wild cluster bootstrap: for 999 iterations, multiply each country's
   residuals by random +1/-1 (Rademacher weights), reconstruct Y,
   refit, store t. Compute the fraction of |t*| >= |t_observed|.
2. Permutation test: for 999 iterations, shuffle sugar values within
   each country (breaking temporal order but preserving cross-country
   structure), refit TWFE, store t. Compute the fraction of
   |t*| >= |t_observed|.
3. Plot both null distributions as histograms. Mark the observed t.

### What you should see
Bootstrap p ~ 0.80 (one-way FE), ~ 0.58 (TWFE). Permutation p ~ 0.58.
The observed t = -0.64 sits comfortably in the middle of the null
distribution. There is no signal.

### Paper number produced
Section 3.2: bootstrap and permutation p-values.

---

## Rung 11: The Positive Control
**STAT 538 Unit:** M7 (Diagnostics -- does my method work?)
**Paper connection:** GDP detrended t = -4.12 (Figure 5)

### What to learn
If no food variable survives within-country tests, maybe the method
is too weak to detect anything. A positive control tests this. GDP
per capita should behave differently from food supply: cross-sectionally
positive (richer countries are fatter), but within-country the
relationship could reverse (short-run GDP growth might not immediately
increase obesity, or might even reduce it through health investment).

### Exercise
1. Cross-sectional: correlate log GDP with obesity (2018-2020 means).
2. TWFE: fit `obesity ~ log_GDP + C(iso3) + C(year)`, clustered.
3. Detrended: remove country-specific linear trends from both obesity
   and log GDP, then regress.
4. Compare the three.

### What you should see
- Cross-sectional: r = 0.73 (strongly positive -- richer countries are fatter).
- TWFE: t = -0.28 (null -- absorbed by trends).
- Detrended: beta = -1.10, t = -4.12 (significantly NEGATIVE).

GDP reverses sign after detrending. This proves the method has power.
Food supply variables show nothing at any specification.

### Paper number produced
Section 5.4: GDP cross-sectional r = 0.73, detrended t = -4.12.
Figure 5.

---

## Rung 12: The Placebo Test
**STAT 538 Unit:** M7 (Diagnostics) / Causal inference
**Paper connection:** Soybean oil case study (Figure 4, Table 2)

### What to learn
A placebo test uses a variable that should NOT have a causal effect
and checks whether it passes the same tests the real variable passes.
If the placebo passes, the tests cannot distinguish real from fake.
Soybean oil supply in SSA has no plausible causal link to obesity,
yet it passes cross-sectional and one-way FE tests.

### Exercise
1. Extract soybean oil supply from the FAOSTAT zip (see script 05).
2. Run the full diagnostic battery:
   - Cross-sectional r
   - One-way FE (clustered)
   - TWFE (clustered)
   - Lead-lag test (does lead == lag? If so, co-trending)
   - Detrended
   - First differences
   - Oster bounds (delta)
3. Classify each test as pass/fail.

### What you should see
- Cross-sectional: r = 0.36, p = 0.03 (PASSES)
- One-way FE: t = 2.09 (PASSES)
- TWFE: t = 0.70 (fails)
- Lead-lag ratio: 0.99 (pure co-trending)
- Detrended: t = 1.62 (fails)
- First differences: r = 0.04 (fails)
- Oster delta: 17.6 (misleadingly high)

### Paper number produced
Table 2, Figure 4. The entire Section 4.4.

### The lesson
Standard robustness tests (cross-sectional, one-way FE, Oster bounds)
cannot distinguish co-trending from causation. Only tests that remove
trends (TWFE, detrending, FD, lead-lag) detect the artifact.

---

## Rung 13: The Food-Group Cascade
**STAT 538 Unit:** M8 (Model Selection) / M5 (Multiple testing)
**Paper connection:** 6/10 -> 2/10 -> 0/10 -> 0/10 (Section 3.3)

### What to learn
If the null were specific to sugar, you might blame measurement.
But the null applies to ALL 10 FAOSTAT food groups. The cascade
pattern -- many cross-sectional positives, fewer one-way FE positives,
zero TWFE survivors, zero FD survivors -- is the signature of trend
domination, not of one bad variable.

### Exercise
1. Extract all 10 food groups from FAOSTAT (see script 03b).
2. For each, compute: cross-sectional r, one-way FE t, TWFE t, FD r.
3. Count how many are significant at 5% under each specification.
4. Identify which food groups survive one-way FE (meat, vegetables).
5. Verify they collapse under TWFE.

### What you should see
Cross-sectional: 6/10 significant.
One-way FE: 2/10 significant (meat t = 3.05, vegetables t = 4.13).
TWFE: 0/10.
First differences: 0/10.

### Paper number produced
Section 3.3: the cascade.

---

## Rung 14: The Diabetes Borderline Case
**STAT 538 Unit:** M7 (Diagnostics for MLR -- sensitivity analysis)
**Paper connection:** Diabetes TWFE t = 1.92 (Section 3.4)

### What to learn
Diabetes has more residual variance after FE (4.4% vs 0.6% for
obesity). This means there is more room for a food variable to find
something. And indeed, the TWFE sugar-diabetes coefficient is
borderline (t = 1.92, p = 0.055). But detrending kills it (t = 0.26).
The borderline signal is residual co-trending exploiting the extra room.

### Exercise
1. Fit the full specification cascade for diabetes (see script 11):
   pooled OLS, country FE, TWFE, detrended, FD, long diff, CRE.
2. Compute the variance decomposition for diabetes.
3. Compare: obesity residual = 0.6%, diabetes residual = 4.4%.
4. Note: TWFE t = 1.92 (borderline). Detrended t = 0.26 (dead null).

### What you should see
The TWFE result looks tempting. But detrending and FD both kill it.
The 7.7x more residual variance explains why TWFE can find a borderline
signal that more demanding specifications reject.

### Paper number produced
Section 3.4: diabetes TWFE t = 1.92, detrended t = 0.26.

### Why this matters
This is the one place a hostile reviewer will push. You need to
understand exactly why the borderline result does not survive, and
the variance decomposition is the explanation.

---

## Rung 15: Cross-Country Change and Initial Conditions
**STAT 538 Unit:** M4 (MLR interpretation) / M5 (Hypothesis testing)
**Paper connection:** Initial obesity t = 3.45 (Section 6, Table 3)

### What to learn
If food supply does not predict which countries gained more obesity,
what does? A cross-country change regression (N = 37, one observation
per country) tests this. The only significant predictor is initial
(2010) obesity level. Countries that started fatter got fatter faster.
This is divergence, not convergence.

### Exercise
1. For each country, compute 2010-to-2020 changes in obesity, sugar,
   oil, GDP, urbanization, and the 2010 initial obesity level.
2. Fit: `d_obesity ~ d_sugar + d_oil + d_GDP + d_urban + init_obesity`.
3. Note: only init_obesity is significant (t = 3.45, R^2 = 0.41).
4. Sugar change: t = -0.22 (null).

### What you should see
Initial conditions predict divergence. Food supply changes predict
nothing. Cross-country obesity differences widen over time because
of structural conditions (built environment, food retail, employment
patterns), not because of what people eat.

### Paper number produced
Table 3, Section 6: init_obesity t = 3.45, R^2 = 0.41.

---

## The View from the Top

After 15 rungs you have:
- Reproduced every major number in the paper by hand
- Understood projection, F-tests, fixed effects, serial correlation,
  clustering, detrending, first differences, CRE decomposition,
  bootstrap, permutation, positive controls, placebo tests, model
  comparison, and variance decomposition
- Seen each STAT 538 concept applied to real data with real stakes
- Built the intuition for WHY the paradox exists (trend domination +
  tiny residual variance) rather than just knowing THAT it exists

The paper is the curriculum. The curriculum is the paper.

---

## Module Mapping

| Rung | STAT 538 Module | Paper Section |
|------|----------------|---------------|
| 1 | M1 | Fig 1A |
| 2 | M1 | Table 1 row 1 |
| 3 | M4 | Fig 1B |
| 4 | M5 | (foundation) |
| 5 | M4/M5 | Table 1 row 2 |
| 6 | M4/M5 | Table 1 row 3, Fig 6 |
| 7 | M2/M7 | Section 4.2 |
| 8 | M3/M7 | Table 1 row 4, Section 3.2 |
| 9 | M5 | Section 3.2 |
| 10 | M5 | Section 3.2 |
| 11 | M7 | Section 5.4, Fig 5 |
| 12 | M7 | Section 4.4, Fig 4 |
| 13 | M8/M5 | Section 3.3 |
| 14 | M7 | Section 3.4 |
| 15 | M4/M5 | Section 6 |
