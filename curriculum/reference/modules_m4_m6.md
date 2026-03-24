# Modules M4-M6 Reference

Source: STAT 538 (Regression), Zaretzki, University of Tennessee. Textbook: Faraway, *Linear Models with R*.

---

## M4: Multiple Linear Regression

### M4-V1: Multiple Linear Regression

**Key concepts taught:**

- MLR as fitting a hyperplane through a multivariable point cloud
- Linear vs. nonlinear models: "linear" means parameters enter linearly, not necessarily the predictors (e.g., $Y_i = \beta_0 + \beta_1 x_1 + \beta_2 \log x_2 + \beta_3 x_1 x_2 + e_i$ is linear; $Y_i = \beta_0 + \beta_1 x_1^{\beta_2}$ is not)
- Matrix formulation of regression: $Y = X\beta + e$
- Design matrix $X$ (n x (p+1), first column all ones for intercept)
- Least squares estimation via normal equations: $\hat{\beta} = (X^TX)^{-1}X^Ty$
- Hat/leverage matrix: $H = X(X^TX)^{-1}X^T$, projects $y$ onto column space of $X$
- Variance of coefficient estimates: $\text{var}(\hat{\beta}) = \sigma^2(X^TX)^{-1}$
- Residual variance estimate: $\hat{\sigma}^2 = RSS/(n-p)$
- Gauss-Markov theorem: OLS is BLUE (Best Linear Unbiased Estimator) when errors are uncorrelated with equal variance
- $R^2 = 1 - RSS/SST = \text{cor}(\hat{y}, y)^2$; the correlation-based definition is preferred when no intercept
- Identifiability: $(X^TX)$ must be invertible; fails when columns of $X$ are linearly dependent
- Correlation between coefficient estimates: off-diagonal elements of $\sigma^2(X^TX)^{-1}$; estimates change as variables are added/removed

**R functions introduced:**

- `lm(Y ~ X1 + X2 + ..., data)` -- fit multiple linear regression
- `summary(lmod)` / `sumary(lmod)` (faraway shorthand) -- coefficient table, $R^2$, F-stat
- `model.matrix(~ X1 + X2 + ..., data)` -- extract the design matrix $X$
- `crossprod(x, x)` -- compute $X^TX$
- `solve(xtx)` -- matrix inverse $(X^TX)^{-1}$
- `solve(xtx, xty)` -- solve normal equations directly (numerically better)
- `names(lmod)` / `names(summary(lmod))` -- inspect stored components of lm objects
- `summary(lmod)$sigma` -- residual standard deviation $\hat{\sigma}$
- `summary(lmod)$cov.unscaled` -- $(X^TX)^{-1}$, the unscaled covariance matrix

**Datasets used:**

- `gala` (faraway) -- Galapagos Islands: Species count predicted by Area, Elevation, Nearest, Scruz, Adjacent. Core example throughout M4-M5.
- `changeover_times` (Sheather) -- loaded but not heavily used in slides
- `production` (Sheather) -- loaded from web
- `invoice` (Sheather) -- loaded from web

**Key formulas:**

- Model: $Y_i = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p + e_i$
- Matrix form: $Y = X\beta + e$
- Normal equations: $X^TX\hat{\beta} = X^Ty$
- OLS estimator: $\hat{\beta} = (X^TX)^{-1}X^Ty$
- Fitted values: $\hat{y} = X\hat{\beta} = H y$ where $H = X(X^TX)^{-1}X^T$
- Residual sum of squares: $e^Te = (y - X\beta)^T(y - X\beta)$
- Variance of estimates: $\text{var}(\hat{\beta}) = \sigma^2(X^TX)^{-1}$
- Unbiased variance estimate: $\hat{\sigma}^2 = RSS/(n-p)$
- $R^2 = 1 - RSS/SST$

**Exercises / in-class activities:**

- Compute $\hat{\beta}$ manually via matrix algebra on `gala` data and compare to `lm()` output
- Inspect `names(lmod)` and `names(summary(lmod))` to understand what R stores
- Demonstrate identifiability failure by adding `Adiff = Area - Adjacent` (linearly dependent column); R silently drops the redundant variable

**Important pedagogical notes:**

- R handles identifiability by removing variables in reverse order of appearance -- "you need to look hard for error message"
- $R^2$ alone is not reliable (Anscombe quartet reference)
- Coefficient estimates depend on which other variables are in the model due to correlation between $\hat{\beta}_i$ and $\hat{\beta}_j$

**Sugar Paradox connection:**

- Script 02 (cross-sectional regressions with controls) is exactly the MLR framework: sugar supply as one predictor, GDP/urbanization/trade openness as others in $Y = X\beta + e$
- Script 03 (fixed-effects models) adds 36 country dummies to the design matrix $X$ -- this is MLR with a very wide $X$, and the FE coefficient on sugar is a partial correlation controlling for all country-level heterogeneity
- The identifiability discussion applies directly: in FE models, any time-invariant country characteristic is perfectly collinear with the country dummy, so it drops out. This is the FE identification tradeoff.
- Coefficient interpretation depends on controls -- the sugar coefficient changes as we add GDP, urbanization, etc. This is the cascade logic of Script 03.

---

### M4-V2: Orthogonality and Dummy Variables

**Key concepts taught:**

- Covariance and correlation matrices for random vectors and for $\hat{\beta}$
- Orthogonality: $X_1^T X_2 = 0$ implies $\hat{\beta}_1$ and $\hat{\beta}_2$ can be estimated independently
- When variables are orthogonal, adding/removing variables does not change coefficient estimates
- Orthogonality is achieved through designed experiments, not typically observational data
- Dummy variables: encoding categorical factors as 0/1 indicator columns
- Shifted intercept model: $Y_i = \beta_0 + \beta_1 X + \beta_2 I(z=\text{group2}) + \beta_3 I(z=\text{group3}) + e_i$ -- common slope, different intercepts per group
- Interaction model: allowing slopes to vary by group via product terms $X \cdot I(z=\text{group})$
- Baseline/reference group: R drops the first factor level as the reference category

**R functions introduced:**

- `summary(lmod, cor=TRUE)` -- show correlation between coefficient estimates
- `factor(variable)` -- convert numeric to factor for dummy coding
- `lm(Y ~ X + factor(Z), data)` -- shifted intercept model (additive dummies)
- `lm(Y ~ X * factor(Z), data)` -- interaction model (slopes vary by group)
- `cov()` -- covariance matrix of predictors
- `coef(model)` -- extract coefficients for manual plotting

**Datasets used:**

- `gala` (faraway) -- continued from M4-V1
- `odor` (faraway) -- designed experiment with orthogonal variables (temp, gas, pack predicting odor)
- Wisconsin Hospital Costs (WHD) -- 526 hospital groups, DRG codes 209/391/430. Y = log(total charges/discharges), X = log(# discharges). Demonstrates economies of scale and group-specific intercepts/slopes.

**Key formulas:**

- Covariance matrix: $\text{cov}(\hat{\beta}) = \sigma^2(X^TX)^{-1}$
- Orthogonality condition: $X_1^TX_2 = 0$ makes $(X^TX)$ block-diagonal
- Under orthogonality: $\hat{\beta}_1 = (X_1^TX_1)^{-1}X_1^TY$ independent of $X_2$
- Shifted intercept: $Y_i = \beta_0 + \beta_1 X + \beta_2 I(z_i = \text{DRG2}) + \beta_3 I(z_i = \text{DRG3}) + e_i$
- Interaction: $Y_i = \beta_0 + \beta_1 X + \beta_2 I(\text{DRG2}) + \beta_3 I(\text{DRG3}) + \beta_4 X \cdot I(\text{DRG2}) + \beta_5 X \cdot I(\text{DRG3}) + e_i$
  - Slope for DRG1: $\beta_1$; for DRG2: $\beta_1 + \beta_4$; for DRG3: $\beta_1 + \beta_5$

**Exercises / in-class activities:**

- Fit `odor ~ temp + gas + pack` and observe that orthogonal design gives zero covariance between coefficients; dropping a variable does not change other estimates
- Fit shifted-intercept and interaction models on WHD data; interpret coefficient changes
- Manually construct regression lines per group from coefficient estimates and plot with `geom_abline()`

**Important pedagogical notes:**

- Orthogonality is a property you engineer via experimental design; in observational data, columns of $X$ are almost always correlated
- When interpreting dummy variable coefficients: always relative to the omitted baseline group
- Interaction terms let you test whether the relationship between X and Y differs across groups

**Sugar Paradox connection:**

- Script 03 FE models are exactly the shifted-intercept model: country dummies shift the intercept while sugar supply has a common slope across countries (in the basic FE). This is the Wisconsin Hospital model with 36 "DRG" groups (countries) instead of 3.
- The interaction model maps to testing whether the sugar-diabetes slope varies by country or region -- a heterogeneity analysis
- Non-orthogonality is central to the Sugar Paradox: sugar supply correlates with GDP, urbanization, and dietary transition. Adding controls changes the sugar coefficient because $X_{\text{sugar}}^T X_{\text{GDP}} \neq 0$. This is why the cascade of controls in Script 03 is informative.

---

## M5: Hypothesis Testing in MLR

### M5-V1: Hypotheses Testing in Multiple Linear Regression

**Key concepts taught:**

- Geometry of the F-test: comparing projections of $y$ onto nested subspaces
- General F-test framework: compare any two nested models (full vs. reduced)
- Overall F-test: full model vs. intercept-only model ("do any variables matter?")
- Testing one predictor: drop one variable, compare SSE via F-test; this is equivalent to the t-test in the coefficient table ($F_{1,n-p} = t_{n-p}^2$)
- Testing a pair of predictors: drop multiple variables simultaneously
- Testing a subspace: $H_0: \beta_{\text{area}} = \beta_{\text{adjacent}}$ by fitting model with $I(\text{Area} + \text{Adjacent})$
- Testing a specific hypothesis: $H_0: \beta_{\text{Elevation}} = 0.5$ using `offset()`
- Equivalence of t-test, F-test, and likelihood ratio test in regression

**R functions introduced:**

- `anova(model_reduced, model_full)` -- F-test comparing nested models
- `deviance(model)` -- extract RSS (residual sum of squares)
- `df.residual(model)` -- residual degrees of freedom
- `pf(fstat, df1, df2)` -- F-distribution p-value
- `I(Area + Adjacent)` -- combine variables to test equal-coefficient hypothesis
- `offset(0.5 * Elevation)` -- fix a coefficient at a specific value to test $H_0: \beta = c$

**Datasets used:**

- `gala` (faraway) -- all F-test examples use this dataset

**Key formulas:**

- General F-statistic: $F = \frac{(SSE_{\text{reduced}} - SSE_{\text{full}})/(p - q)}{SSE_{\text{full}}/(n - p)} \sim F_{(p-q, n-p)}$
  - $p$ = number of parameters in full model (including intercept)
  - $q$ = number of parameters in reduced model
- When $p - q = 1$: $F_{1,n-p} = t_{n-p}^2$ (t-test and F-test are identical)
- Overall F-test: reduced model is intercept-only ($q = 1$)
- SSE difference = $\|X\hat{\beta}_1 - X\hat{\beta}_2\|^2 = \|r_1 - r_2\|^2$

**Exercises / in-class activities:**

- Compute overall F-test manually from `deviance()` and `df.residual()` and compare to `anova(nullmod, lmod)`
- Verify that dropping one variable via `anova()` gives the same p-value as the t-test in `summary()`
- Test whether Area and Adjacent can both be dropped
- Test $\beta_{\text{area}} = \beta_{\text{adjacent}}$ by fitting `I(Area + Adjacent)`
- Test $\beta_{\text{Elevation}} = 0.5$ using `offset()`

**Important pedagogical notes:**

- "We want to find the simplest model that explains most of the variation" -- the F-test formalizes this parsimony
- LR, F, and t-tests all produce the same conclusions in linear regression
- The F-test is inherently directional: it tests whether the more complex model explains significantly more variation

**Sugar Paradox connection:**

- Script 03 FE cascade is a sequence of nested F-tests: each step adds controls (GDP, urbanization, trade) and we can test whether the additional controls significantly improve fit
- The t-statistic on sugar supply in each FE model is equivalent to the F-test for dropping sugar from that specification: $t^2 = F_{1,n-p}$
- Testing $\beta_{\text{sugar,FE}} = 0$ across specifications is the core inference exercise of the Sugar Paradox
- Testing subspaces is relevant when asking: do all dietary variables (sugar, fat, protein) have a common coefficient? This tests whether dietary composition matters or just total calories.

---

### M5-V2: Permutation and Bootstrap Testing

**Key concepts taught:**

- Permutation tests as distribution-free alternatives when regression assumptions are questionable
- Permutation of response variable: scramble $y$ to break any relationship with $X$; compare real F-stat to distribution of permuted F-stats
- Permutation of a single predictor: scramble one $x$ column to test that specific variable
- Contrasts: testing linear combinations of coefficients, e.g., $H_0: \beta_{\text{area}} - \beta_{\text{elevation}} = 0$
- Contrast vector $c^T = (0, 1, -1, 0, 0, 0)$ so $c^T\hat{\beta} = \hat{\beta}_{\text{area}} - \hat{\beta}_{\text{elevation}}$
- Variance of a contrast: $\text{var}(c^T\hat{\beta}) = \sigma^2 c^T(X^TX)^{-1}c$
- Residual bootstrap: resample residuals, add to fitted values, refit
- Pairs bootstrap: resample entire rows (X, Y) with replacement, refit
- Residual bootstrap is better for designed experiments (fixed X); pairs bootstrap is better for observational data

**R functions introduced:**

- `sample(Species)` -- permute a vector (for permutation test of response)
- `sample(Scruz)` -- permute a single predictor (for permutation test of one variable)
- `summary(lmods)$fstat[1]` -- extract F-statistic
- `summary(lmods)$coef[3,3]` -- extract t-statistic for specific coefficient
- `residuals(lmod)` -- extract residuals for bootstrap
- `fitted(lmod)` -- extract fitted values for bootstrap
- `update(lmod, booty ~ .)` -- refit model with new response
- `quantile(x, c(0.025, 0.975))` -- bootstrap confidence interval percentiles
- `confint(lmod)` -- parametric confidence intervals for comparison

**R code patterns:**

Permutation test for overall F:
```r
nreps <- 4000
fstats <- numeric(nreps)
for(i in 1:nreps){
   lmods <- lm(sample(Species) ~ Nearest + Scruz, gala)
   fstats[i] <- summary(lmods)$fstat[1]
}
mean(fstats > lms$fstat[1])  # p-value
```

Permutation test for one predictor:
```r
tstats <- numeric(nreps)
for(i in 1:nreps){
   lmods <- lm(Species ~ Nearest + sample(Scruz), gala)
   tstats[i] <- summary(lmods)$coef[3,3]
}
mean(abs(tstats) > abs(lms$coef[3,3]))  # two-sided p-value
```

Residual bootstrap:
```r
nb <- 4000; coefmat <- matrix(NA, nb, 6)
resids <- residuals(lmod); preds <- fitted(lmod)
for(i in 1:nb){
  booty <- preds + sample(resids, rep=TRUE)
  bmod <- update(lmod, booty ~ .)
  coefmat[i,] <- coef(bmod)
}
apply(coefmat, 2, function(x) quantile(x, c(0.025, 0.975)))
```

Pairs bootstrap:
```r
for(i in 1:nb){
  gala.new = gala[sample(N[1], replace=TRUE), -2]
  lmod.new <- lm(Species ~ Area + Elevation + Nearest + Scruz + Adjacent, data=gala.new)
  coefmat[i,] = coef(lmod.new)
}
```

Contrast CI:
```r
c <- c(0, 1, -1, 0, 0, 0)
unscaled.cov <- summary(lmod)$cov.unscaled
sigma <- summary(lmod)$sigma
fit.diff <- coef(lmod)[2] - coef(lmod)[3]
ci <- fit.diff + c(1, -1) * 1.96 * sqrt(sigma^2 * t(c) %*% unscaled.cov %*% c)
```

**Datasets used:**

- `gala` (faraway) -- all permutation and bootstrap examples

**Key formulas:**

- Permutation p-value: proportion of permuted test statistics exceeding observed
- Contrast variance: $\text{var}(c^T\hat{\beta}) = \sigma^2_\varepsilon c^T(X^TX)^{-1}c$
- Contrast CI: $(c^T\hat{\beta}) \pm t^*_{0.975} \sqrt{\sigma^2 c^T(X^TX)^{-1}c}$
- Bootstrap CI: empirical 2.5th and 97.5th percentiles of bootstrap distribution

**Exercises / in-class activities:**

- Compare parametric F-test p-value to permutation p-value (should be similar when assumptions hold)
- Implement permutation test for a single predictor by scrambling that column
- Compare residual bootstrap and pairs bootstrap CIs to parametric `confint()`
- Compute contrast CI for difference of two coefficients using the covariance matrix

**Important pedagogical notes:**

- Permutation tests are useful when you question normality/homoscedasticity assumptions
- Two-sided permutation test uses `abs(tstats) > abs(observed)` -- must take absolute values
- Residual bootstrap assumes X is fixed (designed experiment); pairs bootstrap treats X as random (observational data)
- Bootstrap CIs should be similar to parametric CIs when assumptions hold; divergence signals assumption violations

**Sugar Paradox connection:**

- Script 12 implements both permutation tests and bootstrap inference on the actual sugar-diabetes panel data -- this is the direct application
- Pairs bootstrap is the right choice for Sugar Paradox data because it is observational (countries are sampled, not designed)
- Permutation tests provide a distribution-free check on whether the sugar-diabetes association is robust to assumption violations
- Contrast testing applies to comparing sugar coefficients across model specifications (e.g., is the FE sugar coefficient significantly different from the between-countries coefficient?)
- The bootstrap is used to construct confidence intervals that do not rely on normality -- important for panel data where residuals may be clustered/heteroscedastic

---

## M6: Predictions and Causal Inference

### M6-V1: Predictions with Regression

**Key concepts taught:**

- Predictive modeling: develop a model and predict outcomes for new observations
- Prediction vs. explanation: predictive modeling is not interested in cause and effect
- Fitted values at observed $x$ vs. predictions at new $x_{\text{new}}$: both are $x\hat{\beta}$, but uncertainty differs
- Two types of prediction uncertainty:
  1. **Confidence interval** (mean response): uncertainty in $E(Y|x_0) = x_0^T\hat{\beta}$ -- how precisely do we know the average?
  2. **Prediction interval** (individual observation): uncertainty in $Y_{\text{new}} = x_0^T\hat{\beta} + \varepsilon$ -- where will a single new observation fall?
- Prediction intervals are always wider than confidence intervals (include irreducible error $\sigma^2$)
- Intervals widen as $x_0$ moves away from the center of the data (leverage effect via $(x_0^T(X^TX)^{-1}x_0)$ term)
- Autoregression: using lagged values of $y$ as predictors for time series
- Extrapolation dangers: quantitative (outside range), qualitative (wrong population), overfitting, black swans, boss bias
- Model uncertainty is never accounted for in standard intervals

**R functions introduced:**

- `predict(lmod, new=data.frame(t(x0)), interval="prediction")` -- prediction interval
- `predict(lmod, new=data.frame(t(x0)), interval="confidence")` -- confidence interval
- `apply(x, 2, median)` / `apply(x, 2, function(x) quantile(x, 0.95))` -- compute predictor profiles
- `embed(log(airpass$pass), 14)` -- create lagged variable matrix for autoregression

**Datasets used:**

- `fat` (faraway) -- body composition: brozek (body fat %) predicted by 13 body measurements (age, weight, height, neck, chest, abdom, hip, thigh, knee, ankle, biceps, forearm, wrist). Pure prediction problem, no causal intent.
- `airpass` (faraway) -- airline passengers time series: autoregressive model with lag1, lag12, lag13
- `ozone` (faraway) -- exercise dataset for building predictive model

**Key formulas:**

- Variance of mean prediction: $\text{var}(x_0^T\hat{\beta}) = x_0^T(X^TX)^{-1}x_0 \cdot \sigma^2$
- Variance of individual prediction: $\text{var}(x_0^T\hat{\beta} + \varepsilon) = (x_0^T(X^TX)^{-1}x_0 + 1) \cdot \sigma^2$
- Confidence interval: $\hat{y}_0 \pm t_{n-p}^{(\alpha/2)} \hat{\sigma} \sqrt{x_0^T(X^TX)^{-1}x_0}$
- Prediction interval: $\hat{y}_0 \pm t_{n-p}^{(\alpha/2)} \hat{\sigma} \sqrt{1 + x_0^T(X^TX)^{-1}x_0}$
- Autoregressive model: $y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t-12} + \beta_{13} y_{t-13} + \varepsilon_t$

**Exercises / in-class activities:**

- Predict body fat at median predictor values and at 95th percentile; compare CI widths (1% vs 4%)
- Note that prediction intervals are similar width at both profiles because they are dominated by $\sigma^2$
- Build autoregressive model for airline data; extrapolate one step ahead; discuss iterative extrapolation
- Build predictive model for `ozone` data and produce prediction + confidence intervals

**Important pedagogical notes:**

- "Most importantly, model uncertainty is not accounted for. We don't know the correct form of the model." -- fundamental limitation of all parametric prediction
- Confidence intervals narrow as sample size grows; prediction intervals do not (bounded below by $\sigma$)
- Extrapolation is dangerous: the Big Short as an example of over-reliance on models that could not account for regime change
- Prediction intervals at median: (9.6%, 25.4%); at 95th percentile: (21.9%, 38.1%) -- similar widths because dominated by $\varepsilon$

**Sugar Paradox connection:**

- The Sugar Paradox paper uses regression for explanation, not prediction -- but understanding prediction intervals is essential for knowing how precisely the sugar-diabetes relationship is estimated
- The distinction between CI and PI matters when we ask: "What is the expected diabetes prevalence for a country with X sugar supply?" (CI) vs. "What will a specific country's diabetes prevalence be?" (PI)
- Extrapolation warnings apply directly: the model is estimated on 1990-2015 data for 36 SSA countries. Extrapolating to different time periods or non-SSA countries is qualitative extrapolation.
- Model uncertainty is the deepest issue in the Sugar Paradox: we do not know the correct structural model linking sugar to diabetes. All our estimates are conditional on model form.

---

### M6-V2: Causal Inference (Module slides) and Day6-Causal (Day lecture)

These two files cover nearly identical material (the module slides and the in-class lecture version). Combined below.

**Key concepts taught:**

- Three levels of regression use: prediction, association, causation (in increasing order of assumption strength)
- Coefficient interpretation depends on what else is in the model:
  - Elevation coefficient = 0.319 with full controls vs. 0.201 with Elevation alone
  - "You must only interpret a coefficient with respect to other predictors included in the model"
- Two interpretations of $\hat{\beta}_1$:
  1. "A 1-unit increase in $x_1$ produces a change of $\hat{\beta}_1$ in $y$" (incomplete)
  2. "A 1-unit increase in $x_1$ with other predictors held constant produces a change of $\hat{\beta}_1$ in $y$" (better but still problematic in observational data because you cannot actually hold other variables constant)
- Effect plots: predicted $y$ as function of one $x$ with all others at their means vs. raw scatterplot
- Potential outcomes framework (Rubin causal model):
  - Treatment indicator $T \in \{0, 1\}$
  - Individual causal effect: $\delta_i = y_i^1 - y_i^0$
  - Fundamental problem: only one outcome observed; the other is the counterfactual
  - Average treatment effect: $E(\delta_i) = E(y_i^1) - E(y_i^0)$
- Randomization solves confounding by making $T$ independent of all other factors
- Some variables are not like treatments (gender, geography) -- cannot meaningfully be "assigned"
- Confounding variable theory:
  - True model: $y_i = \beta_0^* + \beta_1^* T_i + \beta_2^* Z_i + \varepsilon_i$
  - $Z_i = \gamma_0 + \gamma_1 T_i + \varepsilon'_i$
  - Omitting $Z$: estimated treatment effect becomes $\beta_1^* + \beta_2^* \gamma_1$ (biased)
  - No bias only if $\beta_2^* = 0$ (Z has no effect on Y) OR $\gamma_1 = 0$ (T and Z are uncorrelated)
- Matching as alternative to regression for causal inference:
  - Form matched pairs with similar covariate profiles
  - Compare treated vs. control within pairs
  - Estimates local vertical distance between treatment groups, controlling for confounders
  - Matching and regression give similar but not identical answers
- Qualitative support for causation (Bradford Hill criteria, Faraway pg 69-70):
  1. Strength -- large effects are harder to explain away by confounders
  2. Consistency -- similar effects across separate studies/populations
  3. Specificity -- causal factor is rare and associated with particular response
  4. Temporality -- cause precedes effect
  5. Gradient -- dose-response relationship
  6. Plausibility -- theoretical mechanism
  7. Experiment -- a natural or related experiment exists

**R functions introduced:**

- `predict(lmod, data.frame(...))` -- effect plots (vary one predictor, hold others at means)
- `ifelse(newhamp$votesys == 'H', 1, 0)` -- create treatment indicator
- `chisq.test(nh)$p.value` -- chi-squared test of independence (2x2 table)
- `library(Matching); GenMatch()` -- genetic matching algorithm
  - `GenMatch(trt, covariate, ties=FALSE, caliper=0.05, pop.size=2000)` -- match treated to control within caliper
- `segments()` -- draw matching lines between treated/control pairs on scatterplot
- `t.test(pdiff)` -- test matched-pair differences

**R code patterns:**

Confounding demonstration:
```r
# Naive model (biased)
lmodu <- lm(pObama ~ trt, newhamp)
# With confounder controlled
lmodz <- lm(pObama ~ trt + Dean, newhamp)
# Show T-Z correlation
lm(Dean ~ trt, newhamp)
```

Matching workflow:
```r
mm <- GenMatch(newhamp$trt, newhamp$Dean, ties=FALSE, caliper=0.05, pop.size=2000)
pdiff <- newhamp$pObama[mm$matches[,1]] - newhamp$pObama[mm$matches[,2]]
t.test(pdiff)
```

**Datasets used:**

- `gala` (faraway) -- coefficient interpretation example (Elevation coefficient changes with controls)
- `newhamp` (faraway) -- New Hampshire 2008 primary: Clinton vs. Obama, hand-counted vs. digital ballot districts. Used to demonstrate confounding (Dean vote share confounds apparent trt effect) and matching.

**Key formulas:**

- Causal effect: $\delta_i = y_i^1 - y_i^0$
- ATE: $E(\delta_i) = E(y_i^1) - E(y_i^0)$
- Omitted variable bias: estimated coefficient = $\beta_1^* + \beta_2^* \gamma_1$ where $\gamma_1$ captures $T$-$Z$ correlation
- No bias condition: $\beta_2^* = 0$ or $\gamma_1 = 0$

**Exercises / in-class activities:**

- Compare coefficient on Elevation in full model (0.319) vs. bivariate model (0.201) -- demonstrates omitted variable bias in reverse (adding controls changes estimate)
- Fit naive model `pObama ~ trt` (significant, $\hat{\beta} = 0.04$); add Dean as control `pObama ~ trt + Dean` ($\hat{\beta}_{\text{trt}}$ becomes nonsignificant) -- classic confounding demonstration
- Verify that Dean correlates with treatment: `Dean ~ trt` is significant
- Matching exercise: match hand-counted to digital districts on Dean%, compare Obama vote share within pairs
- Compare regression estimate ($\hat{\beta}_{\text{trt}} = -0.005$, not significant) to matching estimate ($\bar{d} = -0.025$, CI covers zero in one version, significant in another) -- illustrates that regression and matching can give slightly different answers

**Important pedagogical notes:**

- The New Hampshire example is a powerful teaching case: the naive analysis finds a "significant" treatment effect that is entirely driven by confounding (Dean vote share). Controlling for the confounder eliminates the effect.
- "We predict that taller islands have more species. We can't say that altitude causes more species, only that it is associated."
- Matching and regression are conceptually similar (both estimate vertical distance between groups controlling for covariates) but can give different numerical answers
- Bradford Hill criteria provide qualitative support for causation when randomization is impossible

**Sugar Paradox connection:**

- The entire paper is an exercise in causal inference from observational data. The Sugar Paradox is precisely a confounding problem: does sugar cause diabetes, or are sugar supply and diabetes both consequences of economic development?
- Script 03 Mundlak/CRE decomposition separates "between" and "within" variation -- this maps directly to the between-country confounding vs. within-country temporal variation logic. The between-countries association is confounded by development level; the within-country association isolates temporal changes.
- Omitted variable bias formula: if $Z$ = economic development, $\gamma_1$ (correlation of sugar with development) is positive, $\beta_2^*$ (effect of development on diabetes) is positive, so the sugar coefficient is biased upward in cross-sectional regressions that omit development controls. This is the entire motivation for FE models.
- DAG thinking: Sugar <-- Development --> Diabetes creates a backdoor path. FE closes it by conditioning on country (absorbing all time-invariant confounders). But time-varying confounders (urbanization, dietary transition) may still bias FE estimates.
- Bradford Hill criteria map to the paper's identification argument:
  - Temporality: sugar supply changes precede diabetes changes (lagged models in Script 03)
  - Gradient: dose-response tested via continuous sugar variable
  - Consistency: tested across 36 countries and multiple model specifications
  - Plausibility: biological mechanism (fructose-insulin pathway)
  - Strength: the FE sugar coefficient, while statistically significant, is moderate in magnitude -- this is a weakness for the causal claim
- The matching framework suggests an alternative approach: match countries on development indicators and compare sugar-diabetes relationships within matched pairs. This is not done in the current paper but is a natural extension.

---

### Day6-Predictions (Day lecture on Predictions)

This file is the in-class lecture version of M6-V1. Content is nearly identical. Key differences/additions:

- Same body fat prediction example with identical code
- Same autoregressive model for airline data
- Same extrapolation warnings and Big Short discussion
- Same ozone exercise
- Slightly different formatting but no new conceptual material beyond M6-V1

No additional concepts, formulas, or Sugar Paradox connections beyond what is already documented in M6-V1 above.

---

## Cross-Module Summary: Sugar Paradox Connections

| Module | Sugar Paradox Element | Script(s) |
|--------|----------------------|-----------|
| M4 (MLR) | Cross-sectional regressions with controls as MLR; FE = MLR with country dummies; coefficient instability under correlated predictors | 02, 03 |
| M4 (Orthogonality) | Non-orthogonality of sugar/GDP/urbanization drives coefficient sensitivity; FE as shifted-intercept model | 02, 03 |
| M5 (F-tests) | FE cascade as nested F-tests; t-stat on sugar = F-test for dropping sugar; testing equal dietary coefficients | 03 |
| M5 (Permutation/Bootstrap) | Distribution-free inference on panel data; pairs bootstrap for observational data; contrast tests across specifications | 12 |
| M6 (Prediction) | CI vs PI for country-level diabetes projections; extrapolation warnings for out-of-sample countries/periods; model uncertainty | General |
| M6 (Causal) | Entire identification strategy; omitted variable bias formula; confounding by development; Mundlak within/between decomposition; Bradford Hill criteria | 03, paper argument |
