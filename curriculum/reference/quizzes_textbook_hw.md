# Quizzes, Textbook, and Homework Reference

Complete reference for building the Sugar Paradox interactive textbook.
Sources: STAT 538 quizzes (3, 5), homework submissions (HW1-HW5),
Sheather textbook (Ch 2, 3, 5, 6, 7, 8), CURRICULUM.md, and LADDER.md.

---

## Quizzes

### Quiz 3 (Module 3: Diagnostics and Transformations)

**Topics covered:** Inverse response plot, Box-Cox transformations, multivariate
power transformations, Cook's distance, influential point removal, diagnostic
plot interpretation.

**Question types:** Computational (run R code, read output), interpretation
(compare diagnostic plots before/after), conceptual (what does lambda mean).

**Key skills tested:**
- Use `inverseResponsePlot()` to find optimal lambda for response transformation
- Interpret the four standard diagnostic plots (residuals vs fitted, QQ, Scale-Location, residuals vs leverage)
- Use `powerTransform()` for multivariate Box-Cox on both X and Y simultaneously
- Interpret `testTransform()` output (hypothesis tests on specific lambda values)
- Compute and interpret Cook's distance; identify the most influential observations
- Refit models after removing influential points and compare diagnostics

**Actual questions:**
1. (5 pts) Use `inverseResponsePlot` on `lm(x ~ y, data = cathedral)` to find optimal lambda. Report to two decimal places. *Answer: lambda = -2.18*
2. (5 pts) After transformation (R^2 improves .41 to .446), which diagnostic plot improved most? *Answer: Residuals vs Fitted*
3. (5 pts) Use `powerTransform(cbind(x, y) ~ 1)` to find simultaneous optimal transformation. Test (0,0) and (1,1). *Answer: (1,1) -- no transformation is adequate (p = 0.385)*
4. (5 pts) Identify points with largest Cook's distance in `lm(I(1/x^2) ~ y)`. *Answer: St Asaph (D = 1.017) and Bath (D = 0.566)*
5. (5 pts) Refit without those two points. Which diagnostics improved? *Answer: All four plots improved, though violations may remain*

**Sugar Paradox exercises that would prepare for this:**
- Rung 7 (Independence/serial correlation) -- diagnostic plot interpretation
- Rung 8 (Detrending/transformations) -- understanding when and why to transform
- Rung 11 (Positive control) -- comparing diagnostic batteries across specifications

---

### Quiz 5 -- Reading Quiz (Module 5: Hypothesis Testing, Permutation, Bootstrap)

**Topics covered:** Overall F-test meaning, partial F-test for categorical
predictors via `anova()`, handling NAs in nested model comparison, permutation
test mechanics (shuffle Y), bootstrap CI construction (percentile method),
residual vs pairs bootstrap, definition of p in F-test.

**Question types:** Conceptual (true/false with careful wording traps),
interpretation (read ANOVA output), code debugging (fix a permutation test).

**Key skills tested:**
- Distinguish "all coefficients equal zero" from "all coefficients equal each other"
- Know that p includes the intercept (total number of parameters)
- Understand that ANOVA table is useful but not "vitally important" per Faraway
- Understand partial vs marginal effects (beta_Area means different things in simple vs multiple regression)
- Understand permutation test mechanics: shuffle Y to simulate the null
- Understand bootstrap does NOT require restrictive assumptions
- Distinguish residual bootstrap (designed experiments, X fixed) from pairs bootstrap (observational, X random)
- Know that bootstrap CI endpoints come from percentiles of the bootstrap coefficient distribution

**Actual questions:**
1. (5 pts) "The overall F-test tests that all regression coefficients are the same." True/False? *Answer: False (it tests all = 0, not all = each other)*
2. (5 pts) What is p in the F-test? (a) Including intercept, (b) Excluding intercept? *Answer: (a) Including intercept*
3. (5 pts) "The ANOVA table is vitally important" per Faraway. True/False? *Answer: False (Faraway shows multiple routes to F)*
4. (5 pts) Does H0: beta_Area = 0 mean the same in simple vs full model? *Answer: (b) No -- in full model it's conditional on other variables*
5. (5 pts) Permutation test avoids normality assumption. True/False? *Answer: True*
6. (5 pts) Permutation test uses F-statistic as measure of association. True/False? *Answer: True*
7. (5 pts) Permutation test simulates the alternative hypothesis. True/False? *Answer: False (it simulates the null)*
8. (5 pts) Bootstrap requires many restrictive assumptions. True/False? *Answer: False*
9. (5 pts) Residual bootstrap is most appropriate for non-experimental data. True/False? *Answer: False (residual bootstrap is for designed experiments; pairs bootstrap is for observational data)*
10. (5 pts) Bootstrap CI endpoints are determined by: (a) t-distribution, (b) residual percentiles, (c) percentiles of bootstrap coefficient distribution, (d) error term quantiles? *Answer: (c)*

### Quiz 5 -- Problem Solving (Module 5: Prestige Data)

**Topics covered:** Overall F-test computation, partial F-test for categorical
predictors, NA handling in model comparison, permutation test code, bootstrap CI
comparison.

**Question types:** Run R code and report numbers, interpret ANOVA output,
debug code, compare intervals.

**Key skills tested:**
- Run `summary(lm(...))` and extract the overall F-statistic from the last line
- Interpret `anova(small, big)` output for nested model comparison
- Understand why `!is.na(Prestige$type)` is needed (ensure both models use same rows)
- Fix a permutation test by wrapping `sample()` around the response variable
- Run bootstrap code and compare bootstrap CI upper endpoints to standard CI endpoints

**Actual questions:**
1. (5 pts) Report the overall F-statistic for `prestige ~ education + income + women + type` on Prestige data. *Answer: read from summary output, ~105.3 on 5 and 92 df*
2. (5 pts) Using anova(fit.p2, fit.p), is type significant? Choices: (a) reject null that coefficients = 0, (b) reject null that coefficients != 0, (c) don't reject. *Answer: (a) -- p = 0.0045, reject H0: both type coefficients = 0*
3. (5 pts) Why use `Prestige[!is.na(Prestige$type),]`? *Answer: (d) -- ensures both fit.p and fit.p2 use the same rows. Without it, fit.p drops NA rows (uses type) but fit.p2 keeps them (doesn't use type)*
4. (5 pts) Fix permutation test code. *Answer: (b) -- add sample() around prestige (shuffle Y, not X)*
5. (5 pts) Which coefficient's upper CI endpoint changed most from standard to bootstrap? *Answer: run the code and compare 97.5% columns*

**Sugar Paradox exercises that would prepare for these:**
- Rung 4 (F-test as comparing projections) -- the conceptual foundation for Q1-Q2
- Rung 5 (Fixed effects as expanding subspace) -- nested model comparison, partial F
- Rung 9 (CRE decomposition) -- within vs between, partial effects
- Rung 10 (Bootstrap and permutation) -- simulation-based inference mechanics

---

## Homework Summary

### HW1 (Sheather Ch 2: Simple Linear Regression)

**Problems:**
- S2.1: Playbill data -- CI for slope, test H0: beta_0 = 10000, prediction interval at x = 400000, evaluate "next week = this week" rule
- S2.2: Economic indicators -- CI for slope (negative association), confidence interval for E(Y|X=4)
- S2.5: Compare two models (RSS, SSreg) from scatterplot appearance
- S2.7: Why 95% of observations fall outside the 95% CI (CI is for the line, not individual points)
- Key definitions: CI, PI, explanatory/predictor/response/dependent variable, predicted/fitted value, line of best fit, residual

**Concepts exercised:**
- Confidence interval for beta_1 via `confint()`
- Hypothesis testing for specific parameter values (t-test with non-zero null)
- Prediction intervals via `predict(..., interval = "prediction")`
- Confidence intervals for the mean response via `predict(..., interval = "confidence")`
- RSS vs SSreg decomposition (SST = SSreg + RSS)
- CI for the line vs PI for individual observations

**Key skills:**
- Mechanically extract CIs, PIs, and test statistics from R output
- Interpret whether specific values fall inside/outside intervals
- Distinguish CI (uncertainty about the line) from PI (uncertainty about individual observations)
- Decompose total variability into explained and unexplained

**Sugar Paradox equivalent:**
- Rung 1 (correlation) and Rung 2 (simple regression as projection) teach the same concepts
- Rung 2 exercise: fit pooled OLS, extract t-statistics and R^2 -- same skills as S2.1/S2.2
- The PI vs CI distinction maps to the "is the signal about the mean or about individual countries?" question in the panel analysis

---

### HW2 (Sheather Ch 3 + Faraway Ch 6: Diagnostics and Transformations)

**Problems:**
- S3.1: Airfare data -- critique analyst who only checked R^2/p-values without diagnostics, identify curved residuals, suggest quadratic fix
- S3.4: Shipping data -- diagnose nonlinearity + heteroscedasticity + non-normality in Time ~ Tonnage, explain why PI at high tonnage is too short, evaluate log(Time) ~ Tonnage^0.25 transformation
- F6.1: SAT data (total ~ expend + ratio + salary + takers) -- check constant variance (Breusch-Pagan), normality (Shapiro-Wilk), leverage (2p/n threshold), outliers (|r_i| > 2), influence (Cook's D > 4/n)
- F6.4: Swiss data (Fertility ~ Catholic) -- full diagnostic battery, log-log transformation, quadratic model comparison

**Concepts exercised:**
- The full diagnostic checklist: residuals vs fitted, QQ, Scale-Location, residuals vs leverage
- Box-Cox transformation and its relationship to lambda
- Breusch-Pagan test for heteroscedasticity (`ncvTest()`)
- Shapiro-Wilk test for normality
- Leverage points (hat values, 2p/n threshold)
- Outliers (standardized residuals > 2)
- Cook's distance (4/n threshold, combines leverage and outlier status)
- Structural nonlinearity vs distributional problems -- why quadratic beats transformation when the relationship is U-shaped

**Key skills:**
- Read and interpret all four diagnostic plots
- Run formal tests (Breusch-Pagan, Shapiro-Wilk) alongside visual inspection
- Distinguish structural problems (wrong model form) from distributional problems (non-normality, heteroscedasticity)
- Choose between transformation and polynomial based on the nature of the violation
- Identify high-leverage, outlier, and influential observations

**Sugar Paradox equivalent:**
- Rung 7 (serial correlation diagnostics) -- the assumption cascade in action
- Rung 8 (detrending) -- transformation as a diagnostic tool
- Rung 11 (positive control) -- validating that the method works before trusting the null

**Key definitions from HW2:**
- Standardized residual: e_i / (s * sqrt(1 - h_ii))
- Leverage: h_ii, diagonal of hat matrix, threshold 2p/n
- Outlier: |standardized residual| > 2
- Bad leverage point: high leverage + outlier (pulls the line where it shouldn't go)
- Cook's distance: D_i = (r_i^2 / p) * (h_ii / (1 - h_ii))
- Constant variance: Var(epsilon_i) = sigma^2 for all i
- Elasticity: in log-log model, beta_1 = % change in Y per 1% change in X

---

### HW3 (Sheather Ch 3: Transformations, Polynomial Regression)

**Problems:**
- S3.3A: AdRevenue data -- develop sqrt(Y) ~ sqrt(X) model with Box-Cox justification, prediction intervals at 0.5M and 20M circulation, back-transformation, weaknesses (Parade leverage)
- S3.3B: Same data -- cubic polynomial (no transformation), nested F-tests for degree selection, prediction intervals, weaknesses (Parade leverage near 1.0, extrapolation danger)
- S3.3C: Compare Part A vs Part B -- why sqrt-sqrt is preferred (lower leverage, fixes heteroscedasticity, simpler, better extrapolation)
- S3.7: Why inverse response plot fails when X is highly skewed (narrow range of fitted values can't distinguish transformations)
- S3.8: Diamond price data -- untransformed linear model, log-log model with elasticity interpretation, comparison (log-log better diagnostics despite slightly lower R^2)

**Concepts exercised:**
- Box-Cox guided transformation selection
- Polynomial regression as alternative to transformation
- Nested F-tests for polynomial degree (anova(linear, quadratic, cubic))
- Back-transformation of prediction intervals (squaring sqrt-scale PIs gives asymmetric intervals)
- Comparing models on different response scales (R^2 not directly comparable)
- Leverage in polynomial models (extreme points have extreme leverage)
- Elasticity interpretation of log-log models (slope = % change per % change)
- Limitations of inverse response plot (IRP fails with skewed X)

**Key skills:**
- Choose between transformation and polynomial based on diagnostic evidence
- Compute and back-transform prediction intervals
- Evaluate tradeoffs: simpler model with better diagnostics vs more complex model with higher R^2
- Interpret polynomial coefficients vs elasticity from log-log
- Understand when graphical transformation diagnostics (IRP) can fail

**Sugar Paradox equivalent:**
- Rung 8 (detrending as transformation) -- the concept that transformations change what signal you can see
- Rung 14 (diabetes borderline case) -- comparing specifications when residual variance differs

---

### HW4 (Faraway Ch 2 + Sheather Ch 5: Multiple Regression, ANCOVA)

**Problems:**
- F2.6: Cheddar cheese (taste ~ Acetic + H2S + Lactic) -- report coefficients, verify cor(fitted, Y)^2 = R^2, no-intercept model R^2 problem, QR decomposition
- F2.7: Wafer data (orthogonal factorial design) -- treatment contrasts, zero correlations in orthogonal design, coefficient stability when dropping predictors
- S5.2: Houston Chronicle (repeating first grade ~ low income, by year) -- ANCOVA: test association, test year effect after controlling for poverty, test interaction (different slopes by year)
- S5.3: Chateau Latour (Quality ~ EndofHarvest * Rain) -- interaction interpretation, marginal effects (days to decrease quality by 1 point with/without rain)

**Concepts exercised:**
- Multiple regression coefficients and their interpretation
- R^2 as cor(fitted, observed)^2 for models with intercept
- The misleading R^2 from no-intercept models (uses SST_0 = sum(y_i^2))
- QR decomposition as numerically stable alternative to normal equations
- Orthogonal designs: dropping predictors doesn't change remaining coefficients
- ANCOVA: combining continuous and categorical predictors
- Interaction terms: the slope of one predictor depends on the level of another
- update() for model comparison, anova() for nested F-tests

**Key skills:**
- Extract and interpret coefficients from `summary(lm(...))`
- Understand why R^2 changes (or doesn't) when modifying the model
- Use `model.matrix()` to see the actual design matrix
- Set up and interpret ANCOVA models
- Compute marginal effects from interaction models (slope = main effect + interaction * level)

**Sugar Paradox equivalent:**
- Rung 3 (partial correlation) -- multiple regression controlling for confounders
- Rung 5 (fixed effects as expanding subspace) -- adding country dummies is the panel analog of ANCOVA
- Rung 9 (CRE decomposition) -- interaction between within and between components
- Rung 15 (cross-country change regression) -- multiple regression with initial conditions

---

### HW5 (Faraway Ch 3: Testing in Multiple Regression)

**Problems (template -- answers not filled in submission):**
- F3.2: Cheddar cheese -- identify significant predictors at 5%, refit with exp(Acetic) and exp(H2S), compare models (can't use F-test for non-nested), interpret a 0.01 increase in log(H2S), convert log-scale change to percentage
- F3.4: SAT data -- test beta_salary = 0, test all three predictors = 0, add takers, show t^2 = F equivalence
- F3.5: Derive formula relating R^2 and the F-statistic: F = (R^2 / (p-1)) / ((1-R^2) / (n-p))
- F3.6: Happy MBA data -- permutation test for money predictor, overlay t-density on permutation histogram, compare permutation and parametric results
- F3.7: Punting data -- test beta_RStr = beta_LStr (subspace test), confidence region for (beta_RStr, beta_LStr), test total leg strength sufficiency, left-right symmetry (simultaneous test)

**Concepts exercised:**
- Individual t-tests vs overall F-test
- Non-nested model comparison (can't use F-test; use AIC/BIC)
- Log-scale coefficient interpretation (0.01 change on log scale ~ 1% change on original scale)
- The R^2-to-F formula
- Permutation test implementation and comparison to parametric
- Subspace tests / linear contrasts (beta_1 = beta_2)
- Confidence regions (2D joint confidence set for two parameters)
- Left-right symmetry testing (simultaneous equality constraints)

**Key skills:**
- Distinguish nested from non-nested model comparisons
- Implement permutation tests from scratch in R
- Set up and interpret linear contrast tests
- Understand the relationship between R^2, F, and individual t-tests
- Convert between log and percentage interpretations

**Sugar Paradox equivalent:**
- Rung 4 (F-test as comparing projections) -- the core mechanism tested here
- Rung 10 (bootstrap and permutation) -- F3.6's permutation test is exactly this
- Rung 9 (CRE decomposition) -- the within/between split is a linear contrast

---

## Textbook Key Formulas and Results

### Chapter 2: Simple Linear Regression

**The model:**
$$Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad E(\varepsilon | X) = 0, \quad \text{Var}(Y | X = x) = \sigma^2$$

**Least squares estimates:**
$$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**Fitted values and residuals:**
$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i, \quad e_i = y_i - \hat{y}_i$$

**Standard error of beta_1:**
$$\text{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}}, \quad s^2 = \frac{\sum e_i^2}{n - 2}$$

**Confidence interval for beta_1:**
$$\hat{\beta}_1 \pm t_{n-2, \alpha/2} \cdot \text{SE}(\hat{\beta}_1)$$

**Prediction interval for new Y at x_0:**
$$\hat{y}_0 \pm t_{n-2, \alpha/2} \cdot s\sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}}$$

**Confidence interval for E(Y | x_0):**
$$\hat{y}_0 \pm t_{n-2, \alpha/2} \cdot s\sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}}$$

**Key distinction:** PI includes the "+1" inside the square root (individual observation noise). CI does not. PI is always wider.

**R-squared:**
$$R^2 = \frac{\text{SSreg}}{\text{SST}} = 1 - \frac{\text{RSS}}{\text{SST}}, \quad \text{SST} = \text{SSreg} + \text{RSS}$$

---

### Chapter 3: Diagnostics and Transformations (SLR)

**Anscombe's quartet:** Four datasets with identical regression output but completely different scatter plots. Lesson: ALWAYS plot the data.

**The four diagnostic plots:**
1. Residuals vs Fitted -- checks linearity (look for curves) and constant variance (look for funnel)
2. Normal QQ -- checks normality (points should hug diagonal)
3. Scale-Location -- checks constant variance (should be flat)
4. Residuals vs Leverage -- identifies influential points (near Cook's contours)

**Standardized residual:**
$$r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}}$$

**Box-Cox transformation:**
The family $Y^{(\lambda)} = (Y^\lambda - 1)/\lambda$ for $\lambda \neq 0$, and $\log(Y)$ for $\lambda = 0$.
Box-Cox finds the $\lambda$ maximizing the profile log-likelihood. Special cases:
- $\lambda = 1$: no transformation
- $\lambda = 0.5$: square root
- $\lambda = 0$: log
- $\lambda = -1$: reciprocal

**Inverse Response Plot (IRP):** Fits $Y^\lambda$ against $\hat{Y}$ from the linear model. Picks the lambda minimizing RSS. Can fail when X is skewed (insufficient spread in $\hat{Y}$ to distinguish lambda values).

**Multivariate Box-Cox:** `powerTransform(cbind(Y1, Y2) ~ 1)` finds simultaneous optimal transformations. Test specific lambda vectors with `testTransform()`.

**Log transformation interpretation:**
- log-log model: $\beta_1$ = elasticity (% change in Y per 1% change in X)
- log-level model: $100 \cdot \beta_1$ = % change in Y per unit change in X (approximate)

---

### Chapter 5: Multiple Linear Regression

**The model:**
$$Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon$$

**Matrix formulation:**
$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$$

**Fitted values and hat matrix:**
$$\hat{\mathbf{Y}} = \mathbf{H}\mathbf{Y}, \quad \mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$$

**Polynomial regression:** Special case where predictors are $x, x^2, x^3, \ldots$. Use nested F-tests (anova) to select polynomial degree.

**R-squared always increases** when adding predictors (geometrically: the projection subspace expands, so the shadow of Y gets closer). Hence: use adjusted R^2 or AIC/BIC for model comparison.

**Adjusted R-squared:**
$$R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{SST}/(n-1)}$$

**Prediction and confidence intervals in MLR:** Same structure as SLR but use the full X matrix. CI for the mean is narrower; PI adds individual noise.

---

### Chapter 6: Diagnostics and Transformations (MLR)

**Leverage in multiple regression:**
$$h_{ii} = \text{diag}(\mathbf{H})_i, \quad \text{threshold: } \frac{2(p+1)}{n}$$

High leverage = unusual predictor combination (not necessarily outlier).

**Standardized residual:**
$$r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}}$$

**Cook's distance:**
$$D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1 - h_{ii}}$$

Combines outlier status (large $r_i$) and leverage ($h_{ii}$). Threshold: $4/n$.

**The 2x2 grid of influence:**
| | Small residual | Large residual |
|---|---|---|
| Low leverage | Harmless | Outlier (not influential) |
| High leverage | Pulls line (good leverage) | BAD: influential outlier |

**Added variable plots (partial regression plots):**
- Regress Y on everything except $X_j$ -> get residuals $e_{Y|X_{-j}}$
- Regress $X_j$ on everything except $X_j$ -> get residuals $e_{X_j|X_{-j}}$
- Plot $e_{Y|X_{-j}}$ vs $e_{X_j|X_{-j}}$
- Slope = $\hat{\beta}_j$ (the partial coefficient)

**Variance Inflation Factor (VIF):**
$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$
where $R^2_j$ is R-squared from regressing $X_j$ on all other predictors.
VIF > 10 indicates severe multicollinearity.

**Marginal model plots:** Compare the model's prediction of $E(Y|X_j)$ against a nonparametric smoother. If they diverge, the model is misspecified.

**ANCOVA / Interaction models:** Including categorical predictors and their interactions with continuous predictors. Key: `anova(small, big)` for nested model comparison requires both models fitted on exactly the same rows (handle NAs explicitly).

---

### Chapter 7: Variable Selection

**Four criteria for evaluating subsets:**

1. **Adjusted R-squared:** Highest is best, but tends to overfit.

2. **AIC (Akaike Information Criterion):**
$$\text{AIC} = n \ln(\text{RSS}/n) + 2p$$
Lowest is best. Penalizes complexity less aggressively than BIC.

3. **BIC (Bayesian Information Criterion):**
$$\text{BIC} = n \ln(\text{RSS}/n) + p \ln(n)$$
Lowest is best. Penalizes complexity more for $n > e^2 \approx 7.4$.

4. **Mallows' Cp:**
$$C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2_{\text{full}}} - n + 2p$$
Good models have $C_p \approx p$.

**Search strategies:**
- All subsets: evaluate all $2^m$ models (exponential, only feasible for moderate m)
- Forward selection: start empty, add the predictor that improves fit most
- Backward elimination: start full, drop the predictor that hurts fit least
- Stepwise: combine forward and backward

**Key insight:** Overfitting vs underfitting tradeoff. Too few predictors = biased predictions (model is wrong). Too many = high variance (model fits noise). AIC/BIC balance this.

**Post-selection inference warning:** If you select a model by testing many subsets, the p-values from that selected model are inflated. The variables look more significant than they are because they were cherry-picked.

---

### Chapter 8: Logistic Regression

**When to use:** Response is binary (0/1) or binomial (counts of successes).

**The logistic function:**
$$\theta(x) = \frac{\exp(\beta_0 + \beta_1 x)}{1 + \exp(\beta_0 + \beta_1 x)}$$

**Log-odds (logit) link:**
$$\log\left(\frac{\theta}{1 - \theta}\right) = \beta_0 + \beta_1 x$$

**Odds ratio interpretation:** A one-unit increase in $x$ multiplies the odds by $e^{\beta_1}$.

**Fitting:** Maximum likelihood, not least squares. The variance depends on the mean ($\theta(1-\theta)$), so OLS is inappropriate.

**Deviance (analog of RSS):**
$$D = -2 \sum [y_i \log(\hat{\theta}_i) + (1 - y_i)\log(1 - \hat{\theta}_i)]$$

**Testing:**
- Wald test (analog of t-test): $z = \hat{\beta}/\text{SE}(\hat{\beta})$
- Likelihood ratio test (analog of F-test): compare deviances of nested models
- Drop-in-deviance test: $\Delta D \sim \chi^2_q$ where q = number of dropped parameters

**Diagnostics:** Deviance residuals replace ordinary residuals. Hosmer-Lemeshow test for goodness of fit.

---

## LADDER.md Curriculum Mapping

The LADDER.md defines 15 rungs that map the Sugar Paradox analysis to STAT 538 modules. Here is how they align with the course material, textbook chapters, and homework.

### Module M1 (Simple Linear Regression) -- Sheather Ch 2, HW1

| Rung | Concept | Textbook | Homework |
|------|---------|----------|----------|
| 1 | Vectors and correlation (bivariate r = 0.677) | Ch 2: correlation, scatter plots | HW1 S2.1 (CI for slope), S2.2 (negative association) |
| 2 | Simple regression as projection (pooled OLS t = 19.84) | Ch 2: least squares, R^2, fitted values | HW1 S2.1c (prediction), S2.7 (CI vs PI) |

### Module M3 (Transformations) -- Sheather Ch 3, HW2-HW3

| Rung | Concept | Textbook | Homework |
|------|---------|----------|----------|
| 8 | Detrending and first differences as transformations | Ch 3: Box-Cox, log transforms | HW3 S3.3 (sqrt-sqrt, polynomial), S3.8 (log-log, elasticity) |

### Module M4 (Multiple Regression) -- Sheather Ch 5, Faraway Ch 2, HW4

| Rung | Concept | Textbook | Homework |
|------|---------|----------|----------|
| 3 | Multiple regression and partial correlation (partial r = 0.496) | Ch 5: MLR, partial coefficients | HW4 F2.6 (cheddar), S5.2 (ANCOVA) |
| 5 | Fixed effects as expanding the subspace (t drops 19.84 to 0.25) | Ch 5: adding predictors to column space | HW4 F2.7 (orthogonal design), S5.3 (interaction) |
| 6 | Two-way FE and variance decomposition (99.4% absorbed) | Ch 5: SST decomposition | HW4 S5.2 (year effect in ANCOVA) |
| 15 | Cross-country change regression (init_obesity t = 3.45) | Ch 5: MLR interpretation | HW4 F2.6 (multiple predictors) |

### Module M5 (Hypothesis Testing) -- Faraway Ch 3, HW5, Quiz 5

| Rung | Concept | Textbook | Homework/Quiz |
|------|---------|----------|---------------|
| 4 | F-test as comparing projections (t^2 = F) | Ch 5 pp.129-130 (nested F-test) | HW5 F3.4 (demonstrate t^2 = F), F3.5 (R^2-F formula) |
| 9 | CRE decomposition (within t = -0.66, between t = 3.46) | Ch 5: partial coefficients | HW5 F3.7 (subspace tests, symmetry) |
| 10 | Bootstrap and permutation (p = 0.80/0.58) | Faraway Ch 3.3 (permutation) | HW5 F3.6 (permutation test for money), Quiz 5 Q4-Q7 |

### Module M2/M7 (Diagnostics) -- Sheather Ch 3/6, HW2, Quiz 3

| Rung | Concept | Textbook | Homework/Quiz |
|------|---------|----------|---------------|
| 7 | Serial correlation, SE inflation (AR(1) = 0.90-0.98) | Ch 6: diagnostics, leverage, influence | HW2 F6.1 (SAT diagnostics), F6.4 (Swiss diagnostics) |
| 11 | Positive control (GDP detrended t = -4.12) | Ch 6: added variable plots | HW2 F6.1e (influential points) |
| 12 | Placebo test (soybean oil case study) | Ch 6: marginal model plots | Quiz 3 Q4-Q5 (Cook's D, refitting) |

### Module M8 (Model Selection) -- Sheather Ch 7

| Rung | Concept | Textbook | Homework |
|------|---------|----------|----------|
| 13 | Food-group cascade (6/10 -> 0/10) | Ch 7: AIC, BIC, all-subsets, overfitting | (no direct HW, but Ch 7 material) |

### Module M7 (Sensitivity Analysis)

| Rung | Concept | Textbook | Homework |
|------|---------|----------|----------|
| 14 | Diabetes borderline case (TWFE t = 1.92, detrended t = 0.26) | Ch 6: variance decomposition, diagnostics | HW2 F6.4f-g (transformation vs quadratic) |

### Coverage summary

All 15 rungs map to STAT 538 modules M1-M8. The mapping is dense: each rung typically touches 2-3 homework problems and 1-2 textbook chapters. The ladder's progression (pooled OLS -> country FE -> TWFE -> detrending -> simulation -> positive control -> placebo -> cascade) mirrors the course's progression (SLR -> diagnostics -> transformations -> MLR -> testing -> diagnostics-MLR -> model selection), but reordered around a single dataset rather than a parade of unrelated examples.

The CURRICULUM.md's 16-unit structure (5 atomic operations + 3 diagnostic units + 4 MLR units + 3 beyond-the-line + 1 synthesis) provides the theoretical scaffolding. The LADDER.md provides the applied practice. Together they cover:

- **Fully covered by both:** Projection (U2/R2), F-tests (U3/R4), diagnostics (U6/R7,R11,R12), transformations (U8/R8), multiple regression (U9/R3,R5,R6), simulation (U4/R10), model selection (U12/R13)
- **Covered by CURRICULUM.md only:** Vectors/inner products (U1), assumption cascade theory (U5), multicollinearity/VIF (U10), contrasts (U11), prediction intervals (U13), logistic regression (U14), causal inference DAGs (U15), synthesis (U16)
- **Covered by LADDER.md only:** Panel-specific methods (CRE decomposition R9, variance decomposition R6, serial correlation R7, placebo tests R12, positive controls R11, cross-country change R15)

The interactive textbook should integrate both: use CURRICULUM.md's theoretical structure as the chapter organization, and LADDER.md's exercises as the computational backbone within each chapter.
