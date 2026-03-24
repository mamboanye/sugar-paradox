# Quizzes and Textbook Reference

This reference captures the topics, question types, skills tested, and key content from STAT 538 quizzes and the Sheather/Faraway textbooks. It is intended for a future agent building quiz-style interactive exercises into an HTML textbook using Sugar Paradox data.

---

## Quiz 3: Problem Solving Quiz -- Cathedral Data Analysis

**Dataset:** `cathedral` from the `faraway` package (25 English cathedrals; x = nave height, y = nave length)

**Topics covered:**
- Inverse response plots and optimal power transformations (`inverseResponsePlot`)
- Multivariate Box-Cox transformations (`powerTransform`, `testTransform`)
- Comparing diagnostic plots before and after transformation
- Cook's distance and identifying influential points
- Refitting models after removing influential observations
- Interpreting all four standard R diagnostic plots (Residuals vs Fitted, Normal QQ, Scale-Location, Residuals vs Leverage)

**Question types:**
1. **Computational (Q1):** Run `inverseResponsePlot` and report optimal lambda to two decimal places
2. **Interpretation/comparison (Q2):** Compare before/after diagnostic plots, identify which improved most
3. **Conceptual + computational (Q3):** Multivariate power transformation -- test whether (0,0) or (1,1) lambdas are adequate using `testTransform` and interpret p-values
4. **Computational + interpretation (Q4):** Identify points with largest Cook's distance from a Cook's distance plot and tabular output
5. **Interpretation (Q5):** Refit model without influential points, compare all four diagnostic plots before/after, assess improvement

**Key skills tested:**
- Running R code for power transformations and reading numerical output
- Understanding what lambda values mean (e.g., lambda = -2 means $x \mapsto 1/x^2$)
- Reading and comparing diagnostic plots side-by-side
- Understanding Cook's distance as a measure of influence (threshold: D > 1 is clearly influential)
- Understanding the effect of removing influential points on model diagnostics
- Interpreting hypothesis tests for transformation adequacy (if p > 0.05, the proposed lambda is adequate)

**Sugar Paradox exercises that would prepare for this quiz:**
- Apply `inverseResponsePlot` to sugar consumption vs. health outcome and find optimal transformation
- Compute Cook's distance on a sugar-health regression; identify which countries are most influential
- Compare diagnostic plots before/after log-transforming sugar consumption or GDP
- Use `powerTransform` to test whether log or no transformation is adequate for cross-country health data
- Remove 2-3 most influential countries and show how regression coefficients and diagnostics change

---

## Quiz 5: Reading Quiz -- Hypothesis Testing, Permutation, Bootstrap

**Source:** Faraway textbook (Chapter 3) supplemented by course slides on the Galapagos (`gala`) dataset

**Topics covered:**
- The overall F-test: what it tests (all coefficients = 0, NOT that coefficients are equal to each other)
- Definition of p in the F-test (number of parameters INCLUDING the intercept)
- Whether the ANOVA table is "vitally important" (Faraway's pragmatic view: it's one tool among several)
- Conditional interpretation of coefficients in multiple vs. simple regression (partial effects)
- Permutation tests: logic, assumptions, what statistic they use, what distribution they simulate
- Bootstrap: few assumptions, widely applicable (not restrictive)
- Residual bootstrap vs. pairs bootstrap (designed experiment vs. observational data)
- How bootstrap confidence interval endpoints are determined (quantiles of bootstrap coefficient distribution)

**Question types:**
1. **True/False conceptual (Q1):** Overall F-test tests coefficients are "the same" -- distinguishing "all = 0" from "all equal each other"
2. **Multiple choice definitional (Q2):** What is p? (includes vs. excludes intercept)
3. **True/False reading comprehension (Q3):** Is the ANOVA table "vitally important" per Faraway?
4. **Multiple choice conceptual (Q4):** Does $H_0: \beta_{area} = 0$ mean the same in simple vs. multiple regression?
5. **True/False conceptual (Q5):** Permutation test avoids normality assumptions
6. **True/False conceptual (Q6):** Permutation test uses the F-statistic as measure of association
7. **True/False conceptual (Q7):** Permutation test simulates null (not alternative) distribution
8. **True/False conceptual (Q8):** Bootstrap requires many restrictive assumptions (false -- it's flexible)
9. **True/False conceptual (Q9):** Residual bootstrap is for non-designed-experiment data (false -- residual bootstrap is for designed experiments; pairs bootstrap is for observational)
10. **Multiple choice (Q10):** How are bootstrap CI endpoints determined? (2.5th and 97.5th percentiles of bootstrap coefficient distribution)

**Key skills tested:**
- Distinguishing "all coefficients = 0" from "all coefficients are equal"
- Understanding degrees of freedom and what p counts
- Understanding partial/conditional effects in multiple regression vs. total effects in simple regression
- Knowing the logic of permutation tests: shuffle Y to simulate the null, compute F each time, compare
- Understanding that permutation tests are nonparametric (no normality assumption needed)
- Distinguishing residual bootstrap (fixed X, designed experiments) from pairs bootstrap (random X, observational data)
- Knowing bootstrap CIs come from quantiles of the resampled coefficient distribution

**Sugar Paradox exercises that would prepare for this quiz:**
- Interpret the overall F-test on a model predicting diabetes from sugar, GDP, urbanization, region
- Compare the coefficient on sugar consumption in a simple model vs. a multiple regression with confounders -- show how partial effects differ from total effects
- Run a permutation test (shuffle health outcome, recompute F 4000 times) and compare p-value to parametric F-test
- Run pairs bootstrap on the sugar-health observational data, compute 95% CIs, compare to standard CIs
- Identify which coefficient's bootstrap CI differs most from the standard CI

---

## Quiz 5: Problem Solving Quiz -- Prestige Data (F-tests, Permutation, Bootstrap)

**Dataset:** `Prestige` from `carData` (occupational prestige ~ education + income + women + type)

**Topics covered:**
- Extracting the overall F-statistic from `summary(lm(...))`
- Partial F-test for a categorical variable (`anova(small, big)`)
- Why NA filtering is necessary for valid model comparison
- Fixing permutation test code (shuffle Y, not individual X's)
- Pairs bootstrap for confidence intervals
- Comparing bootstrap vs. standard CI upper endpoints

**Question types:**
1. **Computational (Q1):** Run code, report the F-statistic (note: `type` has 3 levels = 2 dummies = 5 predictors total)
2. **Multiple choice interpretation (Q2):** Given `anova()` output with p = 0.0045, choose correct interpretation (reject null that type coefficients = 0)
3. **Multiple choice conceptual (Q3):** Why use `!is.na(Prestige$type)` -- to ensure both nested models use identical rows
4. **Multiple choice code comprehension (Q4):** How to fix a permutation test loop -- add `sample()` around the response (not predictors)
5. **Computational + interpretation (Q5):** Run bootstrap code, compare upper CI endpoints, identify which changed most

**Key skills tested:**
- Reading R output (finding F-statistic on the last line of `summary()`)
- Understanding how R handles factor variables (3-level factor = 2 dummy variables)
- Understanding nested model comparison and the NA consistency requirement
- Knowing to shuffle Y (not X's) for a permutation test of overall significance
- Understanding pairs bootstrap mechanics (resample entire rows with replacement)
- Computing absolute differences between bootstrap and standard CI endpoints

**Sugar Paradox exercises that would prepare for this quiz:**
- Fit a model with a categorical variable (e.g., WHO region with 6 levels), report overall F-stat, identify how many dummy variables
- Use `anova()` to test whether region is significant after controlling for sugar and GDP
- Demonstrate the NA problem: create a dataset with missing region values, show how model comparison breaks without filtering
- Write a permutation test loop for sugar-health data; show what happens if you shuffle X vs. Y
- Bootstrap the sugar coefficient, compare bootstrap CI to standard CI

---

## Textbook Chapter Summaries

### Ch 2: Simple Linear Regression (Sheather)

**Key formulas:**
- Model: $Y_i = \beta_0 + \beta_1 x_i + e_i$, where $E(e|X) = 0$, $\text{Var}(Y|X=x) = \sigma^2$
- Least squares slope: $\hat{\beta}_1 = \frac{SXY}{SXX} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$
- Least squares intercept: $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- Residuals: $\hat{e}_i = y_i - \hat{y}_i$
- $RSS = \sum \hat{e}_i^2$; $S^2 = \frac{RSS}{n-2}$ (unbiased estimate of $\sigma^2$)
- $\text{Var}(\hat{\beta}_1 | X) = \frac{\sigma^2}{SXX}$; $\hat{\beta}_1 | X \sim N(\beta_1, \sigma^2/SXX)$
- t-statistic for slope: $t = \frac{\hat{\beta}_1 - \beta_1}{se(\hat{\beta}_1)} \sim t_{n-2}$
- CI for slope: $\hat{\beta}_1 \pm t_{\alpha/2, n-2} \cdot se(\hat{\beta}_1)$
- $R^2 = 1 - RSS/SST$ (proportion of variance explained)
- Prediction interval: $\hat{y}_0 \pm t_{\alpha/2, n-2} \cdot S\sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SXX}}$
- Confidence interval for mean response: same but without the "1 +" under the square root

**Key theorems/results:**
- Normal equations yield unique OLS estimates when SXX > 0
- $\hat{\beta}_1$ is unbiased: $E(\hat{\beta}_1|X) = \beta_1$
- Variance of slope decreases as variability in X increases (important for experimental design)
- $\hat{\beta}_1$ is a linear combination of the $Y_i$'s (equation 2.5: $\hat{\beta}_1 = \sum c_i Y_i$)
- Prediction intervals are always wider than confidence intervals for mean response

**Example datasets and what they teach:**
- Production data (run time vs. run size): basic SLR fitting, interpretation of slope (0.26 min per additional item) and intercept (149.7 min setup time)

**What the Sugar Paradox data would illustrate here:**
- Simple regression of diabetes prevalence on per-capita sugar consumption across countries
- Interpret slope: "each additional gram of sugar per day is associated with X% change in diabetes prevalence"
- Compute prediction interval for a country with a specific sugar consumption level
- Show that $R^2$ is modest because sugar alone doesn't explain all variation -- motivating multiple regression

### Ch 3: Diagnostics and Transformations for SLR (Sheather)

**Key formulas:**
- Leverage: $h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{SXX}$; leverage point if $h_{ii} > 2 \times \text{average}(h_{ii}) = 4/n$
- Standardized residual: $r_i = \frac{\hat{e}_i}{S\sqrt{1 - h_{ii}}}$
- Cook's distance: $D_i = \frac{r_i^2}{2} \cdot \frac{h_{ii}}{1 - h_{ii}}$

**Key theorems/results:**
- Anscombe's quartet: four datasets with identical regression output but very different structures -- always plot your data
- Residual plots reveal model misspecification: curvature = missing nonlinear term; funnel = nonconstant variance
- Good leverage point: distant x-value, Y follows the pattern; Bad leverage point: distant x-value, Y does NOT follow pattern (= leverage point + outlier)
- Outlier identification: standardized residual outside [-2, 2] (small/moderate samples) or [-4, 4] (large samples)
- Box-Cox power transformation: find lambda that best normalizes residuals
- `inverseResponsePlot`: estimates optimal power transformation of the response
- `powerTransform`: multivariate Box-Cox for simultaneous transformation of multiple variables

**Example datasets and what they teach:**
- Anscombe's four data sets: identical numerical output, vastly different data structures
- Huber's leverage point example: demonstrates good vs. bad leverage points and their effect on $R^2$
- Cleaning data (contract cleaning): introduces nonconstant variance, motivates transformations

**What the Sugar Paradox data would illustrate here:**
- Residual plot of sugar vs. diabetes showing potential curvature (quadratic term needed?) or heteroscedasticity (high-sugar countries have more variance?)
- Identify leverage points: countries with extreme sugar consumption (e.g., very high or very low)
- Apply Box-Cox to find optimal transformation of health outcome
- Show that Cook's distance flags specific influential countries

### Ch 4: Weighted Least Squares (Sheather)

**Key formulas:**
- Model: $Y_i = \beta_0 + \beta_1 x_i + e_i$ where $\text{Var}(e_i) = \sigma^2 / w_i$
- WRSS: $\sum w_i(y_i - \beta_0 - \beta_1 x_i)^2$
- Weighted slope: $\hat{\beta}_{1W} = \frac{\sum w_i(x_i - \bar{x}_W)(y_i - \bar{y}_W)}{\sum w_i(x_i - \bar{x}_W)^2}$
- Weighted intercept: $\hat{\beta}_{0W} = \bar{y}_W - \hat{\beta}_{1W}\bar{x}_W$
- Weighted leverage: $h_{Wii} = w_i^S\left[\frac{1}{w_i} + \frac{(x_i - \bar{x}_W)^2}{WSXX}\right]$
- Weighted residuals: $\hat{e}_{Wi} = \sqrt{w_i}(y_i - \hat{y}_{Wi})$
- WLS via OLS trick: multiply both sides by $\sqrt{w_i}$, fit the transformed model with no intercept

**Key theorems/results:**
- When $w_i = 1/n$ for all i, WLS reduces to OLS (reality check)
- WLS is appropriate when $Y_i$ is an average of $n_i$ observations (use $w_i = n_i$)
- R's `predict()` may not give correct prediction intervals for WLS (Weisberg 2006 warning)
- WLS can be computed as OLS on transformed variables: $Y^{new}_i = \sqrt{w_i}Y_i$, $x^{new}_{1i} = \sqrt{w_i}$, $x^{new}_{2i} = \sqrt{w_i}x_i$ (no intercept model)

**Example datasets and what they teach:**
- Contract cleaning data: variance increases with number of crews, weights = $1/\text{StdDev}(Y_i)^2$
- Salary of statisticians: $Y_i$ = third quartile salary, $w_i = n_i$ (sample sizes differ by experience group)
- Houston real estate: WLS with $w_i = n_i$ for subdivision median prices

**What the Sugar Paradox data would illustrate here:**
- Country-level health data often aggregates over different population sizes -- use population as weight
- Show that high-population countries (India, China) should receive more weight than small island nations
- Compare OLS vs. WLS results; show how the regression line shifts when properly weighted

### Ch 5: Multiple Linear Regression (Sheather)

**Key formulas:**
- Model: $Y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \cdots + \beta_p x_{pi} + e_i$
- Matrix form: $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{e}$
- OLS: $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$
- Fitted values: $\hat{\mathbf{Y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$
- Residuals: $\hat{\mathbf{e}} = \mathbf{Y} - \hat{\mathbf{Y}}$
- RSS: $(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})'(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$
- Normal equations: $(\mathbf{X}'\mathbf{X})\boldsymbol{\beta} = \mathbf{X}'\mathbf{Y}$
- $R^2 = 1 - RSS/SST$; $R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{SST/(n-1)}$
- F-statistic: $F = \frac{(SST - RSS)/p}{RSS/(n-p-1)}$
- High leverage cutoff: $h_{ii} > 2(p+1)/n$

**Key theorems/results:**
- Polynomial regression is a special case of MLR (powers of x are the predictors)
- Each coefficient is a partial effect: the effect of $x_j$ holding all other predictors constant
- Adding irrelevant predictors increases $R^2$ but may decrease $R^2_{adj}$
- F-test and individual t-tests can disagree (significant F but no significant t's, or vice versa)

**Example datasets and what they teach:**
- Professional salary data: polynomial regression (salary ~ experience + experience^2); demonstrates nonlinear fit within linear model framework
- NYC restaurant pricing: multiple predictors (Food, Decor, Service, East)

**What the Sugar Paradox data would illustrate here:**
- Multiple regression: diabetes ~ sugar + GDP + urbanization + region
- Polynomial: diabetes ~ sugar + sugar^2 to capture diminishing/saturating effects
- Show that the coefficient on sugar changes when confounders are added (partial vs. total effect)
- Compute and interpret $R^2_{adj}$ as predictors are added one by one

### Ch 6: Diagnostics for Multiple Linear Regression (Sheather)

**Key formulas:**
- Hat matrix: $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ (maps $\mathbf{Y}$ to $\hat{\mathbf{Y}}$)
- Leverage: $h_{ii}$ = diagonal of $\mathbf{H}$; high leverage if $h_{ii} > 2(p+1)/n$
- Standardized residual: $r_i = \frac{\hat{e}_i}{S\sqrt{1 - h_{ii}}}$; outlier if $|r_i| > 2$
- Variance of residuals: $\text{Var}(\hat{\mathbf{e}}|X) = \sigma^2(\mathbf{I} - \mathbf{H})$
- Conditions for residual plots to be directly interpretable: (6.6) single-index model and (6.7) linear conditional expectation of predictors (elliptical symmetry)

**Key theorems/results:**
- Marginal model plots have wider applicability than residual plots when condition (6.7) fails
- Added variable plots assess each predictor's contribution after adjusting for others
- Variance inflation factors (VIF) measure collinearity; VIF > 5 or 10 is problematic
- When conditions (6.6) and (6.7) do not hold, residual patterns can be misleading about the nature of misspecification
- Li and Duan (1989) result: residual plots provide direct information about misspecification only under elliptical symmetry of predictors

**Example datasets and what they teach:**
- NYC restaurant data (continued): full diagnostic workup with 4 predictors, demonstrating leverage, outlier detection, residual patterns, collinearity assessment

**What the Sugar Paradox data would illustrate here:**
- Compute VIF for sugar, GDP, urbanization -- show which are collinear
- Create added variable plot for sugar after adjusting for GDP and urbanization
- Identify outlier countries using standardized residuals
- Check if predictor distributions satisfy the elliptical symmetry condition

### Ch 7: Variable Selection (Sheather)

**Key formulas:**
- $R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{SST/(n-1)}$ (tends to overfit; choose highest)
- AIC $= n\log(RSS/n) + 2p$ (as computed by R)
- AIC_C $= AIC + \frac{2(p+2)(p+3)}{n-p-1}$ (bias-corrected; use when $n/K < 40$)
- BIC $= -2\log L(\hat{\theta}|Y) + K\log(n)$ (heavier penalty than AIC when $n \geq 8$)
- Maximum likelihood estimate: $\hat{\sigma}^2_{MLE} = RSS/n$ (biased, vs. $S^2 = RSS/(n-p-1)$)
- Kullback-Leibler information: $I(f,g) = \int f(y)\log\frac{f(y)}{g(y|\theta)}dy$

**Key theorems/results:**
- $R^2_{adj}$ tends toward overfitting; AIC/AIC_C are based on information theory; BIC has Bayesian justification
- BIC penalizes complexity more heavily than AIC, favoring simpler models
- AIC is efficient (minimizes prediction error asymptotically); BIC is consistent (selects the true model if it's among candidates, given enough data)
- No universal agreement on which criterion is "best" -- depends on goals (prediction vs. model identification)
- Stepwise procedures (forward, backward, both) are heuristics; exhaustive search (e.g., `leaps`) is preferred when computationally feasible

**What the Sugar Paradox data would illustrate here:**
- Compare AIC, BIC, $R^2_{adj}$ for nested models: sugar only, sugar + GDP, sugar + GDP + urbanization, etc.
- Show how BIC selects a simpler model than AIC
- Demonstrate forward/backward stepwise selection
- Discuss the tension between "sugar is significant" and "the best model might not include sugar once confounders are added"

### Ch 8: Logistic Regression (Sheather)

**Key formulas:**
- Model: $\theta(x) = \frac{1}{1 + \exp(-(\beta_0 + \beta_1 x))}$
- Logit (log-odds): $\log\frac{\theta(x)}{1-\theta(x)} = \beta_0 + \beta_1 x$
- Odds ratio for 1-unit increase in x: $\exp(\beta_1)$
- Odds ratio for c-unit increase: $\exp(c \cdot \beta_1)$
- Likelihood: $L = \prod_{i=1}^{n} \binom{m_i}{y_i}\theta(x_i)^{y_i}(1-\theta(x_i))^{m_i - y_i}$
- Deviance: $-2\log L$ (replaces RSS as the measure of fit)
- Null deviance vs. residual deviance (analogous to SST vs. RSS)
- AIC for logistic regression

**Key theorems/results:**
- Least squares is inappropriate for binomial responses (variance depends on $\theta$, is not constant)
- Maximum likelihood estimation replaces OLS
- The logistic function is S-shaped, mapping any real number to (0, 1)
- Coefficient interpretation: $\hat{\beta}_1 = 0.501$ means each unit increase in food rating multiplies odds by $\exp(0.501) = 1.7$
- Binomial: $Y|x \sim \text{Bin}(m, \theta(x))$; sample proportion $y/m$ is unbiased for $\theta(x)$

**Example datasets and what they teach:**
- Michelin/Zagat restaurant data: probability of inclusion in Michelin guide based on Zagat food ratings
- Demonstrates S-shaped relationship, odds interpretation, deviance as goodness of fit

**What the Sugar Paradox data would illustrate here:**
- Convert health outcome to binary: country has "high diabetes prevalence" (above median) yes/no
- Logistic regression: $P(\text{high diabetes}) = f(\text{sugar}, \text{GDP}, \text{urbanization})$
- Interpret odds ratio: "each additional 10g sugar/day multiplies odds of high diabetes by X"
- Plot the fitted logistic curve for sugar consumption vs. probability of high diabetes
- Compare logistic regression to linear regression on the binary outcome (show why linear is inappropriate)

### Faraway Textbook: Linear Models with R

**Chapters/topics covered:**
1. Introduction (initial data analysis, data cleaning, when to use regression)
2. Estimation (matrix formulation, OLS, Gauss-Markov theorem, $R^2$, identifiability)
3. Inference (F-tests for model comparison, testing subsets of predictors, testing subspaces/contrasts, permutation tests, confidence intervals, prediction intervals)
4. Diagnostics (checking error assumptions, unusual observations, model structure)
5. Problems with Predictors (measurement error, scale changes, collinearity)
6. Problems with Errors (GLS, WLS, lack of fit, robust regression)
7. Transformation (response and predictor transformations)
8. Variable Selection (hierarchical models, testing-based procedures, criterion-based: AIC/BIC)
9. Shrinkage Methods (PCA, PLS, ridge regression)
10. Statistical Strategy and Model Uncertainty
11. Insurance Redlining (complete example)
12. Missing Data
13. Analysis of Covariance
14-16. ANOVA, Factorial Designs, Block Designs

**Key concepts not in Sheather (supplementary material):**
- **Permutation tests (Ch 3.3):** Detailed code and explanation. Shuffle Y (or the specific predictor being tested), recompute F (or t) 4000 times, compare observed statistic to permutation distribution. Key code pattern: `lm(sample(sr) ~ pop75+dpi, data=savings)`. P-value from permutation closely matches normal-theory p-value but does not require normality assumption.
- **Testing subspaces/contrasts (Ch 3.2):** Test $H_0: \beta_j = \beta_k$ by fitting reduced model with `I(Xj + Xk)`. Test $H_0: \beta_j = c$ using an offset: `lm(y ~ ... + offset(c*Xj))`. These generalize the standard F-test beyond just "is this coefficient zero?"
- **Gauss-Markov theorem (Ch 2.6):** OLS is BLUE (best linear unbiased estimator) when errors are uncorrelated with equal variance. Does NOT require normality. Proof via showing any other linear unbiased estimator has variance $\geq$ OLS variance.
- **Identifiability (Ch 2.9):** When $X^TX$ is singular, parameters are unidentifiable. Examples: redundant predictors, more predictors than observations. R drops variables silently. Near-unidentifiability (tiny perturbation) causes enormous standard errors.
- **Geometric interpretation (Ch 2.3-2.4):** $\hat{Y} = HY$ is the orthogonal projection of Y onto the column space of X. Residuals are perpendicular to the model space. This gives RSS its Pythagorean decomposition.
- **Confidence ellipses (Ch 3.4):** Joint confidence regions for pairs of coefficients are ellipses, not rectangles. The joint test can reject when both individual tests fail to reject, and vice versa. Correlation between predictors and correlation between their coefficients often differ in sign.
- **Shrinkage methods (Ch 9):** Ridge regression, PCA regression, PLS -- alternatives when predictors are highly collinear. Not covered in Sheather.
- **Robust regression (Ch 6.4):** M-estimators, least trimmed squares -- alternatives when errors are non-normal or there are outliers. Not covered in Sheather.

**Sugar Paradox connections:**
- Permutation test on the sugar coefficient: shuffle sugar values across countries, recompute t-stat 4000 times, show that even without normality assumptions the sugar effect is (or isn't) significant
- Test the contrast $H_0: \beta_{sugar} = \beta_{GDP}$ to see if sugar and GDP have equal effects on health
- Demonstrate near-collinearity between sugar consumption and GDP (both increase with development); show how ridge regression or variable selection handles this
- Use the complete-example strategy from Ch 11 (Insurance Redlining) as a template for a full Sugar Paradox analysis: initial data analysis, model building, diagnostics, transformation, variable selection, conclusions

---

## Cross-Cutting Skill Map

The following skills appear across multiple quizzes and chapters. These are the highest-priority targets for interactive exercises:

| Skill | Quiz 3 | Quiz 5 (Reading) | Quiz 5 (PS) | Chapters |
|-------|--------|-------------------|-------------|----------|
| Read/interpret R output | Q1, Q4 | | Q1, Q5 | All |
| Diagnostic plots (4-panel) | Q2, Q5 | | | Ch 3, 6 |
| Cook's distance / influence | Q4, Q5 | | | Ch 3, 6 |
| Power transformations | Q1, Q3 | | | Ch 3, 7 |
| F-test (overall) | | Q1, Q2, Q3 | Q1, Q2 | Ch 5 (Sheather), Ch 3 (Faraway) |
| Partial vs. total effects | | Q4 | | Ch 5, 6 |
| Permutation tests | | Q5, Q6, Q7 | Q4 | Ch 3 (Faraway) |
| Bootstrap (pairs vs. residual) | | Q8, Q9, Q10 | Q5 | Ch 3 (Faraway) |
| NA handling in model comparison | | | Q3 | Ch 6 |
| Categorical predictors (dummies) | | | Q1, Q2 | Ch 5 |
| Confidence/prediction intervals | | | | Ch 2, 5 (Sheather), Ch 3 (Faraway) |
| Variable selection (AIC/BIC) | | | | Ch 7 |
| Logistic regression | | | | Ch 8 |
| WLS | | | | Ch 4 |
