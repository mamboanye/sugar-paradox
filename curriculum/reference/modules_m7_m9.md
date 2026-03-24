# Modules M7-M9 Reference

Structured reference for building interactive textbook chapters. Source: STAT 538 Regression (Zaretzki), Faraway & Sheather textbooks.

---

## M7: Diagnostics for Multiple Linear Regression

### M7-V1: Diagnostics 2 (Faraway Ch 6)

**Key concepts taught:**

1. **Hierarchy of regression problems** (most to least serious):
   - Systematic form wrong -- biased predictions, misleading explanation
   - Bad leverage points -- can fundamentally change fit
   - Nonconstant variance -- inaccurate tests, CIs, prediction uncertainty
   - Normality -- matters only with small samples and very abnormal errors
   - Correlated errors -- less information than sample size suggests, may lead to wrong systematic components
2. **Checking model structure** -- the central new topic. Bivariate plots of $y \sim x_i$ are misleading because they ignore other variables.
3. **Partial regression (added variable) plots** -- isolate the effect of each $x_i$ on $y$ by removing other predictors' effects from both $y$ and $x_i$. Good for detecting bad leverage, outliers, influential points.
4. **Partial residual plots** -- better for detecting nonlinearity. Plot $x_i \hat{\beta}_i + \hat{\varepsilon}$ vs. $x_i$.
5. **Marginal model plots (MMPs)** -- compare parametric regression fit (red) with nonparametric smooth of data (blue). If lines overlap, model fits well. Discrepancy on one variable suggests transforming that variable or $Y$.
6. **Influence and outlier diagnostics:**
   - Studentized residuals: outliers if $|t_i| > 3.5$
   - Leverages (hat values): influential if $> 2(k+1)/n$
   - Cook's D: influential if $> 4/n$
   - DFBETAS: influential if $> 2/\sqrt{n}$
   - DFFITS: influential if $> 2\sqrt{(k+1)/n}$

**R functions introduced:**

- `avPlots()` -- added variable plots (library: car)
- `crPlots()` -- partial residual plots (library: car)
- `marginalModelPlots()` -- marginal model plots (library: car)
- `influenceIndexPlot()` -- leverage, Cook's D, studentized residuals (library: car)
- `outlierTest()` -- Bonferroni-adjusted outlier test (library: car)
- `dfbetaPlots()` -- DFBETAS visualization (library: car)
- `residualPlots()` -- residual plots (library: car)
- `compareCoefs()` -- tabulate fits from several models (library: car)
- `termplot()` -- partial residuals (library: stats)

**Datasets used:**

- `savings` (faraway) -- life-cycle savings across 50 countries. Variables: sr, pop15, pop75, dpi, ddpi
- `defective` (Sheather) -- manufacturing defects. Variables: Defective, Temperature, Density, Rate. URL: `https://gattonweb.uky.edu/sheather/book/docs/datasets/defects.txt`

**Key formulas:**

- Added variable plot construction:
  1. $d = Y - \hat{Y}_{(-i)}$ (residuals of $y$ regressed on all $x$ except $x_i$)
  2. $m = x_i - \hat{x}_{i(-i)}$ (residuals of $x_i$ regressed on all other $x$)
  3. Plot $d$ vs. $m$; slope equals $\hat{\beta}_i$ from full model
- Partial residuals: $y - \sum_{j \ne i} x_j \hat{\beta}_j = x_i \hat{\beta}_i + \hat{\varepsilon}$
- Centered partial residuals (termplot): $\hat{\varepsilon} + \hat{\beta}_i(x_i - \bar{x}_i)$
- Leverage cutoff: $h_{ii} > 2(k+1)/n$
- Cook's D cutoff: $D_i > 4/n$

**Exercises / in-class activities:**

- Interpret avPlots for savings data (Japan, South Rhodesia as problems for pop15; Libya for ddpi)
- Construct avPlot by hand from two auxiliary regressions and verify slope matches full model coefficient
- Compare crPlots to avPlots -- pop15 crPlot reveals data separation into two groups (pop15 > 35 vs. < 35)
- Fit separate models for subsets based on pop15 cutoff
- MMP comparison: defectives data with $Y$ vs. $\sqrt{Y}$ transformation -- see improvement in marginal model alignment
- influenceIndexPlot interpretation for savings data

**Sugar Paradox connection:**

- Script 04 (trend R2, AR(1), SE inflation): Correlated errors are listed as a serious regression problem. The Sugar Paradox time-series analysis deals directly with temporal autocorrelation inflating standard errors -- exactly the "less information than sample size suggests" issue taught here.
- Script 05 (soybean oil leverage/influence at country level): The influence diagnostics (Cook's D, DFBETAS, leverages) map directly to identifying which countries disproportionately drive the soybean oil-obesity association. A country that is both high-leverage and an outlier (bad leverage point) can flip the sign or significance of the relationship.
- LOO analysis in script 02: Leave-one-out is the operational version of the influence diagnostics -- dropping one observation and checking how much the fit changes is exactly what Cook's D and DFFITS measure algebraically.

---

### M7-V2: Multicollinearity (Faraway 7.3)

**Key concepts taught:**

1. **What multicollinearity is:** One or more predictors can be approximated by a linear combination of others. $X^t X$ is nearly singular.
2. **Consequences:**
   - High variance in coefficient estimates
   - Coefficient signs may oppose intuition
   - Inflated standard errors -- t-tests fail to reveal significant factors
   - Fit becomes sensitive to measurement error (small changes in $y$ cause large changes in $\hat{\beta}$)
3. **Detection methods:**
   - Correlation matrix: look for $|r| \approx 1$
   - Partial $R^2$: regress each $x_i$ on all other predictors; if $R^2_i \approx 1$, collinearity present
   - Eigenvalues of $X^t X$: small/zero eigenvalues indicate collinearity
   - Condition number: $\kappa = \sqrt{\lambda_1 / \lambda_p}$; $\kappa > 30$ indicates a problem
   - VIF: $\text{VIF}_j = 1/(1 - R^2_j)$; values exceeding 5 indicate poor estimation
4. **Remediation:** Remove variables that are less significant but highly correlated with others. Iteratively remove and recheck VIFs.

**R functions introduced:**

- `vif()` -- variance inflation factors (library: car)
- `eigen()` -- eigenvalue decomposition (base R)
- `cor()` -- correlation matrix (base R)
- `powerTransform()` -- multivariate Box-Cox estimation (library: car)
- `bcPower()` -- apply Box-Cox power transformations (library: car)
- `pairs()` -- scatterplot matrix (base R)
- `update()` -- modify model formula incrementally (base R)

**Datasets used:**

- `bridge` (Sheather) -- bridge design time prediction. Variables: Time, DArea, CCost, Dwgs, Length, Spans. URL: `https://gattonweb.uky.edu/sheather/book/docs/datasets/bridge.txt`
- `divusa` (faraway) -- US divorce rates. Variables: divorce, unemployed, femlab, marriage, birth, military

**Key formulas:**

- Two-variable case: $\text{Var}(\hat{\beta}_j) = \frac{1}{1 - r^2_{12}} \times \frac{\sigma^2}{(n-1)S^2_{x_j}}$
- General case: $\text{Var}(\hat{\beta}_j) = \frac{1}{1 - R^2_j} \times \frac{\sigma^2}{(n-1)S^2_{x_j}}$, where $\text{VIF}_j = \frac{1}{1 - R^2_j}$
- Condition number: $\kappa = \sqrt{\lambda_1 / \lambda_p}$
- Leverage cutoff for bridge data: $2 \times (p+1)/n = 0.267$

**Exercises / in-class activities:**

- Bridge data: scatterplot matrix before/after log transformation. Box-Cox suggests all logs ($\lambda = 0$).
- Log-transformed model has $R^2 = 0.78$ vs. 0.71 untransformed, but only one significant coefficient due to multicollinearity.
- Diagnose: correlations all very high after log transform; VIFs extremely large.
- Fix iteratively: remove log(Length), then log(DArea), then log(CCost) -- check VIFs after each removal.
- Final simplified model: log(Time) ~ log(Dwgs) + log(Spans), all VIFs acceptable.
- Faraway 7.3 exercise: divusa data -- compute VIFs, remove unemployed and military, check condition numbers.
- Key insight: log transforms compress scales and increase correlation, potentially worsening multicollinearity.

**Sugar Paradox connection:**

- The Sugar Paradox faces this directly when choosing between FAOSTAT and WDI indicators. Multiple food supply variables (total calories, sugar supply, fat supply, protein supply) are inherently correlated since they all derive from the same underlying dietary patterns. VIF analysis would reveal which combinations of predictors are estimable and which are redundant.
- Script 05 country-level analysis: including multiple correlated food system variables simultaneously would inflate standard errors. The decision to analyze soybean oil separately vs. within a multi-predictor model is fundamentally a multicollinearity management decision.
- The "wrong sign" phenomenon directly parallels cases where adding a correlated food variable flips the sign of another -- a pattern that could generate spurious "paradox" claims.

---

## M8: Model Selection

### M8-V1: Model Selection (Faraway Ch 10)

**Key concepts taught:**

1. **Guiding principle:** Occam's Razor -- use the simplest explanation that works well. Goals: improve predictions, ease interpretation, reduce coefficient uncertainty.
2. **Data analysis cycle:** Model development -> Diagnostics -> Model/variable selection (iterative).
3. **Hierarchical modeling principle:** When higher-order terms (interactions, quadratics) are in the model, always include the lower-order main effects, even if not significant. Dropping main effects makes the model non-invariant to scale changes.
   - Example: $y = \beta_0 + \beta_1 x + \beta_2 x^2$. If you drop $\beta_1 x$ and shift $x \to x + a$ (e.g., Kelvin to Fahrenheit), the $x$ term reappears. Dropping it assumes symmetry around $x = 0$.
   - Response surface: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11} x_1^2 + \beta_{22} x_2^2 + \beta_{12} x_1 x_2$. Do not remove $x_1 x_2$ without also removing $x_1^2, x_2^2$.
4. **Search techniques:**
   - **Backward elimination:** Start with all predictors, remove highest p-value above $\alpha_\text{crit}$, refit, repeat. $\alpha_\text{crit}$ can be 10-15% for prediction.
   - **Forward selection:** Start with no predictors, add one with lowest p-value below $\alpha_\text{crit}$.
   - **Stepwise:** Combination -- at each stage, variable may be added or removed.
   - Drawbacks: can miss optimal model, p-values are wrong due to multiple testing, selected models may be too small for optimal prediction.
5. **Model selection criteria:**
   - **AIC**: $-2L(\hat{\theta}) + 2p$. For linear regression: $n \log(\text{SSE}/n) + 2p$. Minimize. Penalty $2p$ blocks adding variables that don't significantly reduce SSE.
   - **BIC**: $-2L(\hat{\theta}) + p \cdot \log(n)$. Minimize. Heavier penalty, prefers smaller models. AIC is better for prediction.
   - **Adjusted $R^2$**: $R^2_a = 1 - \frac{\text{RSS}/(n-p)}{\text{TSS}/(n-1)}$. Maximize. Only increases if predictor has predictive value.
   - **$C_p$ statistic**: $C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} + 2p - n$. Choose small $p$ where $C_p \le p$.
6. **All subsets via leaps:** Enumerate all models (feasible for 15-20 variables). Automatically computes $R^2_\text{adj}$, BIC, $C_p$.
7. **Effect of influential points on model selection:** Alaska is high-leverage for the statedata; after removal, Area enters the best model.
8. **Effect of transformations:** Log-transforming skewed predictors (Population, Area) changes the best model.

**R functions introduced:**

- `regsubsets()` -- all possible subsets regression (library: leaps)
- `step()` -- stepwise selection using AIC (base R). Supports forward, backward, both.
- `lm.influence()$hat` -- hat values / leverages (base R)
- `stripchart()` -- jitter plot to detect skewness (base R)

**Datasets used:**

- `state.x77` / `statedata` (base R) -- US state statistics. Variables: Life.Exp, Population, Income, Illiteracy, Murder, HS.Grad, Frost, Area
- `corrosion` (faraway) -- iron content vs. weight loss. Variables: loss, Fe

**Key formulas:**

- $\text{AIC} = -2L(\hat{\theta}) + 2p$ (general); $\text{AIC} = n \log(\text{SSE}/n) + 2p$ (linear regression)
- $\text{BIC} = -2L(\hat{\theta}) + p \log(n)$
- $R^2_a = 1 - \frac{\hat{\sigma}^2_\text{model}}{\hat{\sigma}^2_\text{null}}$
- $C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} + 2p - n$
- Kullback-Leibler divergence motivation for AIC: $\hat{I}(f,g) = \int f(x) \log f(x) \, dx - \int f(x) \log[g(x|\hat{\theta})] \, dx$

**Exercises / in-class activities:**

- Backward elimination on statedata: remove Area, Illiteracy, Income, Population in order; population removal is a close call
- regsubsets on statedata: compute and plot $R^2_\text{adj}$, $C_p$, BIC for all subsets
- step() with forward, backward, and both directions
- Re-run regsubsets after removing Alaska -- best model changes (Area now enters)
- Re-run after log-transforming skewed variables -- best model changes again
- Polynomial overfitting example (corrosion data): $R^2$ from degree-6 polynomial is misleadingly high

**Sugar Paradox connection:**

- The Sugar Paradox upgrade-path audit is fundamentally a model selection exercise: choosing between FAOSTAT vs. WDI indicators, deciding which covariates belong in the specification. The AIC/BIC reasoning ("does adding this variable reduce SSE enough to justify the complexity?") directly applies to deciding whether to include GDP per capita, urbanization rate, etc. alongside sugar supply.
- The hierarchical principle matters when the Sugar Paradox considers interaction terms (e.g., sugar x income level) -- the main effects must stay in even if individually insignificant.
- Effect of influential observations on model selection: if removing one high-leverage country changes which predictors are selected, that undermines the robustness of the finding. This directly connects to LOO sensitivity analysis.

---

### M8-V2: Genetic Algorithms for Model Selection (No textbook)

**Key concepts taught:**

1. **Motivation:** All-subsets is infeasible for many variables. Stepwise produces suboptimal models due to limited search. GA searches model space more thoroughly via randomized, evolution-inspired search.
2. **GA fundamentals:**
   - Fitness function: objective to minimize/maximize
   - Chromosome: current state of variables (binary vector of length $p$ for subset selection)
   - Population: group of candidate solutions at each generation
   - Selection: produce new generation proportional to fitness
   - Crossover: mix chromosomes between models ($p_\text{crossover} = 0.8$)
   - Mutation: randomly modify chromosomes ($p_\text{mutation} = 0.1$)
   - Elitism: strong individuals pass through unchanged (default 5% of popSize)
   - Convergence: stop when population members converge to same fitness
3. **For regression:** Binary vector $[1,0,1,...,1_p]$ indicates which of $p$ variables are included. Fitness function computes $-\text{BIC}$ (negated because GA maximizes). Each generation evaluates a population of such models.
4. **Key parameters:** popSize (exploration vs. speed), pcrossover, pmutation, maxiter.
5. **Advantages:** Total flexibility in criteria, useful for large model spaces, easy to parallelize.

**R functions introduced:**

- `ga()` -- genetic algorithm optimization (library: GA). Key args: type="binary", fitness, nBits, popSize, pcrossover, pmutation, elitism, maxiter
- `lm.fit()` -- fast least-squares fitting without overhead of `lm()` (base R)
- `model.matrix()` -- design matrix extraction (base R)
- `model.response()` -- response extraction (base R)
- `BIC()` -- Bayesian information criterion (base R)
- `plot(GA)` -- convergence plot showing fitness over generations
- `summary(GA)` -- solution vector and fitness

**Datasets used:**

- `state.x77` / `statedata` (base R) -- same as M8-V1
- `fat` (faraway) -- body fat prediction. Response: brozek. Many predictors (columns 2,3 removed).

**Key formulas:**

- Fitness function: $f(\text{string}) = -\text{BIC}(\text{lm.fit}(X_\text{included}, y))$
- GA maximizes fitness, so negating BIC minimizes it

**Exercises / in-class activities:**

- Write fitness function using lm.fit + BIC for statedata
- Run GA with default parameters, plot convergence, interpret solution vector
- Apply to fat data with more variables (popSize=100, elitism=7): demonstrates scaling to larger model spaces
- Compare GA solution with regsubsets result from M8-V1

**Sugar Paradox connection:**

- If the Sugar Paradox analysis ever expands to consider a large pool of candidate variables (dozens of FAOSTAT food categories, WDI development indicators, trade metrics), GA-based model selection would be more practical than all-subsets. The fitness function could incorporate domain-specific criteria beyond AIC/BIC -- for instance, penalizing specifications that produce paradoxical signs or that are not robust to LOO.

---

## M9: Logistic Regression

### M9-V1: Introduction to Logistic Regression

**Key concepts taught:**

1. **What logistic regression is:** Regression for binary/binomial responses. $y \in \{0,1\}$ or binomial $(m, p)$.
2. **Competing methods:** LDA, CART, Random Forests, boosted trees, SVM, neural nets.
3. **Core problem:** Need to constrain $\theta(x) = P(Y=1|x) \in (0,1)$ while connecting it linearly to predictors.
4. **The logistic function (link):**
   $$\theta(x) = \frac{\exp(\beta_0 + \beta_1 x)}{1 + \exp(\beta_0 + \beta_1 x)}$$
5. **The logit (inverse):**
   $$\log\left(\frac{\theta}{1-\theta}\right) = \beta_0 + \beta_1 x$$
6. **Odds interpretation:**
   - Odds in favor: $\theta / (1-\theta)$
   - A one-unit increase in $x$ changes log-odds by $\beta_1$
   - Odds ratio: $\text{odds}(x+1) / \text{odds}(x) = \exp(\beta_1)$
   - Effect is multiplicative: $\exp(\beta_1)$ is the factor by which odds change per unit increase in $x$
7. **glm() syntax:** Two forms depending on response type:
   - Binary 0/1: `glm(y ~ x, family=binomial(link="logit"))`
   - Binomial counts: `glm(cbind(success, failure) ~ x, family=binomial)`

**R functions introduced:**

- `glm()` -- generalized linear model (base R). Key: `family=binomial(link="logit")`
- `summary(glm.fit)$coef` -- coefficient table
- `summary(glm.fit)$deviance` -- residual deviance
- `summary(glm.fit)$null.deviance` -- null deviance
- `predict()` -- predicted probabilities (with `type="response"`)

**Datasets used:**

- Michelin Guide data (Sheather) -- proportion of restaurants in Michelin Guide by Zagat food rating. Variables: Food, InMichelin, NotInMichelin, proportion. URL: `https://gattonweb.uky.edu/sheather/book/docs/datasets/MichelinFood.txt`
- `orings` (faraway) -- Challenger O-ring damage vs. temperature. Variables: damage, temp.
- Baseball playoffs (Sheather) -- playoff appearances vs. city population. Variables: PlayoffAppearances, Population, n. URL: `https://gattonweb.uky.edu/sheather/book/docs/datasets/playoffs.txt`
- MichelinNY (Sheather) -- individual NY restaurants. URL: `https://gattonweb.uky.edu/sheather/book/docs/datasets/MichelinNY.csv`

**Key formulas:**

- Logistic function: $\theta(x) = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}$
- Logit: $\log\left(\frac{\theta}{1-\theta}\right) = \beta_0 + \beta_1 x$
- Odds ratio for 1-unit change: $\exp(\beta_1)$
- Odds ratio for $c$-unit change: $\exp(c \cdot \beta_1)$
- Michelin example: $\log\left(\frac{\hat{\theta}}{1-\hat{\theta}}\right) = -10.842 + 0.501 \cdot \text{Food}$; one-unit increase in Food rating increases odds by $\exp(0.501) = 1.7$ times (70% increase)
- Baseball example: $\beta_\text{Pop} = 0.078$; odds of making playoffs increase by $\exp(0.078) = 1.08$ per million additional people (8% increase in odds)

**Exercises / in-class activities:**

- Challenger O-ring: linear regression on proportion shows problems (predictions can go below 0)
- Michelin data: fit binomial logistic regression, interpret coefficient as odds ratio
- Plot fitted logistic curve over data
- Baseball HW 8.1: fit logistic model for playoff appearances vs. population, interpret odds ratio, plot log-odds with fitted line
- Discuss why probability change from a 1-unit shift in $x$ depends on initial $x$ value (nonlinear effect on probability scale)

**Sugar Paradox connection:**

- Could frame as: "What if obesity were binary (above/below a prevalence threshold)?" This converts the continuous obesity rate into a classification problem. For example, does a country cross the 20% obesity prevalence threshold? Logistic regression would model $P(\text{obese rate} > 20\% | \text{sugar supply, GDP, ...})$.
- The odds-ratio interpretation is more intuitive for policy audiences than regression coefficients: "a 10 kcal/capita/day increase in sugar supply increases the odds of crossing the obesity threshold by X%."
- The Michelin Guide example is structurally analogous: expert vs. data-driven ratings of restaurant quality, just as WHO guidelines vs. observational food supply data can diverge.

---

### M9-V2: Logistic Regression Models

**Key concepts taught:**

1. **Maximum likelihood estimation (MLE) for logistic regression:**
   - Data model: $y_i | x_i \sim \text{Bin}(m_i, \theta(x_i))$
   - Likelihood: $L = \prod_{i=1}^n \binom{m_i}{y_i} \theta(x_i)^{y_i} (1 - \theta(x_i))^{m_i - y_i}$
   - Log-likelihood: $\ell = \sum_{i=1}^n [y_i(\beta_0 + \beta_1 x_i) - m_i \log(1 + e^{\beta_0 + \beta_1 x_i}) + \log \binom{m_i}{y_i}]$
   - Maximized via Newton-Raphson (iterative steps in coefficient space)
2. **Wald statistic:** $Z = \hat{\beta}_1 / \text{se}(\hat{\beta}_1) \sim N(0,1)$ under $H_0: \beta_1 = 0$. Reject if $|Z| > 1.96$. CI: $\hat{\beta}_1 \pm z_{1-\alpha/2} \cdot \text{se}(\hat{\beta}_1)$.
3. **Likelihood ratio (LR) tests:**
   - General: $LR(M_1, M_2) = 2(\ell(\hat{\theta}_{M_1}) - \ell(\hat{\theta}_{M_2})) \sim \chi^2$ under $H_0$
   - Approach 1: Compare nested models (analogous to F-test). Difference in deviances: $\text{dev}(M_2) - \text{dev}(M_1) \sim \chi^2_{p_1 - p_2}$
   - Approach 2: Goodness-of-fit (only works for binomial data with $m_i > 1$)
4. **Saturated model:** One parameter per observation, achieves maximum possible likelihood. $\hat{\theta}_S(x_i) = y_i / m_i$.
5. **Deviance (residual deviance):** $G^2 = 2(\ell(\hat{\theta}_\text{sat}) - \ell(\hat{\theta}_\text{fit}))$. Analogous to SSE in linear regression.
   - Deviance formula: $G^2 = 2 \sum_{i=1}^n [y_i \log(y_i / \hat{y}_i) + (m_i - y_i) \log((m_i - y_i)/(m_i - \hat{y}_i))]$
   - Degrees of freedom: number of unique rows $-$ number of $\beta$'s estimated
6. **Goodness-of-fit test:** Compare $G^2$ to $\chi^2_{n-p-1}$. Only valid for binomial data ($m_i \ge 2$). **Does NOT work for binary (0/1) data.**
7. **Model comparison via deviance:** $\text{dev}(M_2) - \text{dev}(M_1) = LR(M_1, M_2)$. Use `anova(model1, model2, test="LRT")`.
8. **$R^2$ for logistic regression:** $R^2_\text{dev} = 1 - \text{residual deviance} / \text{null deviance}$.
9. **Marginal model plots for logistic regression:** Same Cook-Weisberg idea -- compare nonparametric smooth of raw data to smooth of parametric fits. Bad alignment suggests missing terms.

**R functions introduced:**

- `anova(fit1, fit2, test="LRT")` -- likelihood ratio test comparing nested GLMs
- `pchisq()` -- chi-squared p-value computation (base R)
- `sumary()` -- Faraway's compact summary (library: faraway)
- `marginalModelPlots()` applied to glm objects (library: car)

**Datasets used:**

- Michelin Guide data (same as M9-V1)
- MichelinNY (individual restaurants, binary InMichelin response). Variables: Food, Decor, Service, Price, InMichelin

**Key formulas:**

- Log-likelihood: $\ell = \sum_{i=1}^n [y_i(\beta_0 + \beta_1 x_i) - m_i \log(1 + e^{\beta_0 + \beta_1 x_i}) + \log \binom{m_i}{y_i}]$
- Wald statistic: $Z = \hat{\beta}_j / \text{se}(\hat{\beta}_j)$
- Deviance: $G^2 = 2 \sum_{i=1}^n [y_i \log(y_i/\hat{y}_i) + (m_i - y_i) \log((m_i - y_i)/(m_i - \hat{y}_i))]$
- LR test: $\text{dev}(M_2) - \text{dev}(M_1) \sim \chi^2_{p_1 - p_2}$
- Deviance $R^2$: $R^2_\text{dev} = 1 - G^2_{H_A} / G^2_{H_0}$
- Michelin GOF: deviance = 11.36, df = 12, p-value $\approx 0.5$ (good fit)
- Michelin $R^2_\text{dev}$: computed from `1 - food.fit$deviance / food.fit$null.deviance`

**Exercises / in-class activities:**

- Fit null, simple (Food only), and saturated models for Michelin data; compare deviances
- Perform LR test by hand: `food.fit$null - food.fit$deviance` with `pchisq()`; verify matches `anova()` output
- GOF test on Michelin data: deviance 11.36 on 12 df, p-value near 0.5 -- model fits well
- MichelinNY: fit model with Food + Decor + Service + Price; use MMPs to identify misfit
- Adding log(Price) improves all marginal model plots

**Sugar Paradox connection:**

- Deviance-based model comparison (LR tests) is the logistic regression analog of the F-test comparisons used in the Sugar Paradox linear models. If the analysis were reframed as binary (obese/not obese country), the same nested model comparison logic would determine whether sugar supply adds explanatory power beyond GDP and urbanization.
- The GOF test is relevant: if a logistic specification for a binary obesity threshold does not fit well (large deviance relative to df), that signals the model is misspecified -- parallel to the Sugar Paradox's checks on whether the linear trend model captures the data adequately.

---

### M9-V3: Logistic Regression Model Building (Sheather 8.2.3)

**Key concepts taught:**

1. **Theory-guided model selection via conditional distributions:** Using $f(x|Y)$ to determine what belongs in the logistic model.
   - Start from Bayes' theorem: $\frac{\theta(x)}{1-\theta(x)} = \frac{P(Y=1)}{P(Y=0)} \cdot \frac{P(X=x|Y=1)}{P(X=x|Y=0)}$
   - Take logs: $\log\left(\frac{\theta}{1-\theta}\right) = \text{const} + \log\left(\frac{f(x|Y=1)}{f(x|Y=0)}\right)$
2. **Normal conditional distributions:** If $X|Y=j \sim N(\mu_j, \sigma_j^2)$:
   - Equal variances ($\sigma_0 = \sigma_1$): log-odds is linear in $x$ -- simple logistic regression is correct
   - Unequal variances ($\sigma_0 \ne \sigma_1$): log-odds has a quadratic term in $x^2$ -- add $x^2$ to the model
   - Formula: $\log\left(\frac{\theta}{1-\theta}\right) = \beta_0 + \beta_1 x + \beta_2 x^2$
3. **Multivariate extension:** $X|Y=j \sim N(\mu_j, \Sigma_j)$:
   - Log-odds: $\beta_0 + \beta' x + x' B x$ where $B = -\frac{1}{2}\Sigma_1^{-1} + \frac{1}{2}\Sigma_0^{-1}$
   - If $\Sigma_1 \ne \Sigma_0$: need quadratic terms
   - If $\text{cov}(X|Y=1) \ne \text{cov}(X|Y=0)$: need cross-product interactions
4. **Non-normal conditional distributions:** If $f(x|Y)$ is skewed, include both $x$ and $\log(x)$: $\log(\theta/(1-\theta)) = \beta_0 + \beta_1 x + \beta_2 \log(x)$
5. **Practical diagnostics for model building:**
   - Compare conditional variances of $x$ given $Y=0$ vs. $Y=1$ -- if different, add $x^2$
   - Compare conditional covariances -- if different across groups, add interactions
   - Check conditional distributions for skewness -- if skewed, add log terms
   - Use marginal model plots to visually assess fit and identify missing terms
6. **Full model building example (MichelinNY):**
   - Start with Food + Decor + Service + Price + log(Price)
   - Scatterplot of Decor vs. Service colored by InMichelin shows differing slopes (different covariances across groups) -- suggests Decor*Service interaction
   - Adding Service*Decor interaction improves MMPs and is significant by LR test

**R functions introduced:**

- Same as M9-V2, plus:
- `lsfit()` -- quick least-squares line for visualization (base R)
- `anova(m2, m3, test="LRT")` -- testing interaction significance

**Datasets used:**

- MichelinNY (Sheather) -- NY restaurant data with Food, Decor, Service, Price, InMichelin

**Key formulas:**

- Bayes' theorem for odds: $\frac{\theta(x)}{1-\theta(x)} = \frac{P(Y=1)}{P(Y=0)} \cdot \frac{f(x|Y=1)}{f(x|Y=0)}$
- Log ratio of normals: $\log\frac{\sigma_0}{\sigma_1} + \left(\frac{\mu_0^2}{2\sigma_0^2} - \frac{\mu_1^2}{2\sigma_1^2}\right) + \left(\frac{\mu_1}{\sigma_1^2} - \frac{\mu_0}{\sigma_0^2}\right)x + \frac{1}{2}\left(\frac{1}{\sigma_0^2} - \frac{1}{\sigma_1^2}\right)x^2$
- Multivariate quadratic form: $B = -\frac{1}{2}\Sigma_1^{-1} + \frac{1}{2}\Sigma_0^{-1}$
- Full model: $\log(\theta/(1-\theta)) = \beta_0 + \beta_1 \text{Food} + \beta_2 \text{Decor} + \beta_3 \text{Service} + \beta_4 \text{Price} + \beta_5 \log(\text{Price}) + \beta_6 \text{Service} \times \text{Decor}$

**Exercises / in-class activities:**

- Derive the log-odds formula from normal conditional distributions step by step
- Identify from the normal case when quadratic terms are needed (different variances) vs. not needed (equal variances)
- MichelinNY: scatterplot of Decor vs. Service with separate regression lines for InMichelin = 0 vs. 1 -- visually confirm differing covariances
- Fit model without interaction, examine MMPs -- see misalignment
- Add Decor*Service interaction, confirm improvement via LR test and improved MMPs
- Residual analysis: high leverage points disappear when using log(Price) instead of Price

**Sugar Paradox connection:**

- The theory-guided model building approach is directly applicable: if the conditional distributions of sugar supply given "obese vs. not obese" countries have different variances, a quadratic term for sugar supply would be theoretically justified. This provides a principled reason (beyond curve-fitting) for including nonlinear terms.
- The interaction detection method (comparing covariance structures across outcome groups) could reveal whether the sugar-obesity relationship is moderated by income level -- a key question in the Sugar Paradox. If $\text{cov}(\text{sugar}, \text{GDP} | \text{obese})$ differs from $\text{cov}(\text{sugar}, \text{GDP} | \text{not obese})$, including a sugar $\times$ GDP interaction is theoretically motivated.
- Marginal model plots on logistic specifications would reveal whether the assumed functional form for sugar supply (linear in log-odds) is adequate or whether transformations/interactions are needed.

---

## Day Lectures (Supplementary)

The Day7 lectures are near-duplicates of the module lectures with minor variations:

### Day7-Diagnostics

Identical structure to M7-V1 with two notable differences:
- **Problem severity ordering differs:** Correlated errors is ranked second (above nonconstant variance and bad leverage points), versus M7-V1 which ranks bad leverage points second. The Day lecture framing emphasizes that correlated errors "may lead to wrong systematic components."
- Influence statistics section is slightly condensed.

### Day7-ModelSelection

Identical to M8-V1.

### Day7-Multicollinearity

Nearly identical to M7-V2. One difference: bridge data is loaded from a local Windows path (`C:\Users\rzaretzk\...`) rather than a URL. The slide on eigenvalue-based condition numbers omits the code for computing them (just shows VIFs). The "Fix" section is truncated -- shows the concept but does not walk through the full iterative removal.

### Day7-GAModSelection

Identical to M8-V2. Missing the fat data application and the JStatSoftware link. Video references are embedded as images rather than links.
