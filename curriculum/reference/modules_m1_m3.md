# Modules M1-M3 Reference

Textbook: Sheather, "A Modern Approach to Regression with R"
Supplement: Faraway, "Linear Models with R"
Instructor: Zaretzki (STAT 538, University of Tennessee)

---

## M1: Simple Linear Regression

### M1-V1: Simple Linear Regression (SLR Model and Least Squares)

**Key concepts taught:**
- Bivariate data: paired observations $(x_i, y_i)$ for $i = 1, \ldots, n$
- Response (dependent) variable $y$ vs. explanatory (independent/predictor) variable $x$
- The SLR model: $Y_i = \beta_0 + \beta_1 x_i + e_i$
- Conditional expectation: $E(Y|X=x) = \beta_0 + \beta_1 x$ (systematic component)
- Random component: errors $e_i \sim N(0, \sigma^2)$
- Fitted values: $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$
- Residuals: $\hat{e}_i = y_i - \hat{y}_i$ (estimates of errors, but with limitations)
- Least squares estimation: minimize $RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- Normal equations (named for geometry, not normal distribution)
- LS estimates: $\hat{\beta}_1 = SXY/SXX$, $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- Error variance estimate: $S^2 = RSS/(n-2)$
- The LS line passes through $(\bar{x}, \bar{y})$
- Residuals sum to zero: $\sum \hat{e}_i = 0$
- Degrees of freedom: $df = n - 2$ (lose 2 for 2 estimated parameters)
- All LS estimates are unbiased
- Model assumptions: linearity, independence of errors, homoskedasticity, normality of errors

**R functions introduced:**
- `lm(y ~ x, data)` -- fit a linear regression model
- `summary(lm_object)` -- full regression output (two m's)
- `sumary(lm_object)` -- abbreviated output from `faraway` package (one m)
- `ggplot() + geom_point() + geom_smooth(method=lm, se=FALSE)` -- scatterplot with regression line
- `head(data)` -- preview first rows of data

**Datasets used:**
- `production.txt` (Sheather): RunTime vs. RunSize -- does production batch size predict running time? 20 observations.
- `changeover_times.txt` (Sheather): factory changeover times comparing Existing vs. New methods
- `invoices.txt` (Sheather): processing time vs. number of invoices over 30 days

**Key formulas:**
- $Y_i = \beta_0 + \beta_1 x_i + e_i$
- $E(Y|X=x) = \beta_0 + \beta_1 x$
- $RSS = \sum_{i=1}^{n}(y_i - (\hat{\beta}_0 + \hat{\beta}_1 x))^2$
- $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- $\hat{\beta}_1 = \frac{\sum_i^n(x_i-\bar{x})(y_i-\bar{y})}{\sum_i^n(x_i-\bar{x})^2} = \frac{SXY}{SXX}$
- $S^2 = \frac{RSS}{n-2}$

**Exercises / in-class activities:**
- Examine production scatterplot and discuss whether a relationship exists
- Interpret `lm()` output: Estimate, Std.Error, t value, Pr(>|t|), Residual SE, R-squared
- Understand the three tasks: (1) understand output, (2) check model fit/assumptions, (3) make predictions with intervals

**Important pedagogical notes:**
- The lecture emphasizes working backwards from fitted values to motivate least squares
- Distinction between $Y_i$ (random variable, capital letter) and observed $y_i$ is stressed
- Error term $e_i$ captures unobserved factors -- it "allows the left side to equal the right"
- Historical note: LS dates to 1800, was preferred pre-computers over alternatives like minimizing absolute deviations

**Sugar Paradox connection:**
- Script 02 (`02_cross_sectional.py`): Cross-sectional regression of obesity on sugar supply is a direct SLR application. The partial correlation r=0.496 is computed controlling for confounders, but the basic sugar-obesity bivariate relationship is a textbook SLR example
- Script 01 (`01_build_panel.py`): The 37-country panel with paired (sugar_supply, obesity_prevalence) observations is exactly the bivariate data structure $(x_i, y_i)$ taught here
- The fundamental question "does the mean of obesity change linearly as sugar supply changes?" maps directly to the SLR framework

---

### M1-V2: SLR Basics (Inference -- Tests, CIs, F-statistics)

**Key concepts taught:**
- Sampling distributions of parameter estimates:
  - $\hat{\beta}_1|X \sim N(\beta_1, \sigma^2/SXX)$
  - $\hat{\beta}_0|X \sim N(\beta_0, \sigma^2(1/n + \bar{x}^2/SXX))$
- T-statistic for slope: $T = \frac{\hat{\beta}_1 - \beta_1^0}{se(\hat{\beta}_1)} \sim t_{n-2}$
- Standard hypothesis test: $H_0: \beta_1 = 0$ vs. $H_A: \beta_1 \neq 0$ ("Does the mean of Y change as a function of x?")
- Two-sided p-values: computed as $2 \times P(t > |T_{obs}|)$
- Custom hypothesis tests (e.g., $H_0: \beta_1 = 0.01$) using estimate and s.e.
- Confidence intervals for parameters: $\hat{\beta}_1 \pm t(\alpha/2, n-2) \cdot se(\hat{\beta}_1)$
- Confidence intervals for the mean response $E(Y|X=x^*)$
- Prediction intervals for a new observation $Y_{new}|X=x^*$
- Prediction intervals are MUCH wider than confidence intervals (extra $\sigma^2$ term)
- Hourglass shape of CI bands (narrowest at $\bar{x}$, widest at extremes)
- Reading `summary(lm())` output: Residual SE, df, R-squared, Adjusted R-squared, F-statistic
- Residual SE estimates $\sigma$
- R-squared = proportion of variance in Y explained by X (between 0 and 1)
- Adjusted R-squared corrects for multiple regression
- Sums of squares decomposition: $SST = RSS + SSreg$ (Pythagorean theorem analogy)
- $R^2 = SSreg/SST$
- F-statistic: $F = \frac{SSreg/1}{RSS/(n-2)} \sim F_{1,n-2}$ (overall model test)
- F-test tests whether the model is better than intercept-only
- ANOVA table: Sum Sq, df, Mean Sq, F value

**R functions introduced:**
- `confint(lm_object)` -- confidence intervals for parameters
- `predict(lm_object, interval="confidence")` -- CI for mean response
- `predict(lm_object, interval="predict")` -- prediction interval for new observations
- `summary(lm_object)$f` -- extract F-statistic
- `anova(lm_object)` -- ANOVA table with sums of squares
- `drop1(lm_object, test="F")` -- F-test for dropping terms
- `pt(t_value, df, lower.tail=FALSE)` -- compute p-values from t-distribution
- `pf(f_value, df1, df2, lower.tail=FALSE)` -- compute p-values from F-distribution
- `var()` -- variance computation
- `geom_smooth(method=lm, se=TRUE)` -- CI band in ggplot
- `geom_line()` -- overlay prediction interval lines

**R code patterns (pedagogically important):**
```r
# Manual t-test computation
T = (0.259 - 0)/0.037  # (estimate - H0 value) / SE
2*pt(6.9798, df=18, lower.tail=FALSE)  # two-sided p-value

# Verify SST = RSS + SSreg
SST = var(new.data$RunTime)*(n-1)
SSReg = var(fit0$fitted.values)*(n-1)
RSS = sum((fit0$residuals)^2)
SST - (SSReg + RSS)  # should be ~0

# Manual F-statistic
(SSReg/1)/(RSS/18)
```

**Datasets used:**
- `production.txt` (continued): production example used for all inference demonstrations

**Key formulas:**
- $\hat{\beta}_1|X \sim N(\beta_1, \sigma^2/SXX)$
- $\hat{\beta}_0|X \sim N(\beta_0, \sigma^2(1/n + \bar{x}^2/SXX))$
- $T = \frac{\hat{\beta}_1 - \beta_1}{se(\hat{\beta}_1)} \sim t_{n-2}$
- $se(\hat{y}^*) = s\sqrt{1/n + (x^* - \bar{x})^2/SXX}$ (CI for mean)
- $se(Y^*_{new} - \hat{y}^*) = s\sqrt{1 + 1/n + (x^* - \bar{x})^2/SXX}$ (prediction)
- $SST = RSS + SSreg$
- $R^2 = SSreg/SST$
- $F = \frac{SSreg/1}{RSS/(n-2)} \sim F_{1,n-2}$

**Exercises / in-class activities:**
- Manually compute t-statistic and p-value from output for production data
- Verify SST = RSS + SSreg numerically
- Compare confidence vs. prediction interval widths visually
- Observe the hourglass shape of CI bands

**Important pedagogical notes:**
- The intercept test is "not a standard part of regression analysis" -- intercept is often arbitrary (changes with units/centering)
- Emphasizes that parameter estimates are themselves random variables with distributions
- Scientific notation explanation: $1.615 \times 10^{-6} = 0.000001615$
- "Still need to check assumptions" is stressed even after rejecting H0

**Sugar Paradox connection:**
- Script 02 (`02_cross_sectional.py`): The cross-sectional partial r=0.496 with p=0.0018 is a direct t-test on the slope. The Sugar Paradox narrative hinges on the fact that this significant cross-sectional test becomes null under fixed effects -- illustrating that passing a t-test alone is not sufficient
- Script 12 (`12_inference_robustness.py`): Wild cluster bootstrap p-values and permutation tests are advanced versions of the inference concepts taught here. The bootstrap p=0.80 for within-country sugar confirms the null
- Script 10 (`10_tables.py`): Table 2 reports t-statistics across specifications -- each is a direct application of the t-test formula taught in this video

---

### M1-V3: SLR Basics II (Dummy Variables and Practice Problem Setup)

**Key concepts taught:**
- Dummy variable regression: categorical predictor coded as 0/1
- ANOVA as regression: continuous response on categorical factor
- Only $j-1$ dummy variables needed for a factor with $j$ levels
- Baseline level coded as 0, treatment coded as 1
- `factor()` converts character/numeric to categorical in R
- `model.matrix()` shows how R internally represents dummies
- Interpretation: $\beta_0$ = mean of baseline group, $\beta_1$ = difference in means
- $E[Y|New] = \beta_0 + \beta_1$, $E[Y|Existing] = \beta_0$
- Connection to Welch's two-sample t-test (same difference in means, similar p-value)
- Boxplots and jitter plots for visualizing group comparisons

**R functions introduced:**
- `factor(variable)` -- convert to categorical
- `model.matrix(lm_object)` -- view the design matrix (dummy coding)
- `t.test(y ~ group, data)` -- Welch's two-sample t-test
- `geom_boxplot()` -- boxplots in ggplot
- `geom_jitter()` -- jittered dot plots

**Datasets used:**
- `changeover_times.txt` (Sheather): Does a New factory method reduce changeover time compared to Existing method?
  - New method averages 3.17 minutes faster, statistically significant
- `invoices.txt` (Sheather): processing time vs. invoices (practice problem setup)

**Key formulas:**
- $E[Y|Method=Existing] = \beta_0$
- $E[Y|Method=New] = \beta_0 + \beta_1$
- $\beta_1 = E[Y|New] - E[Y|Existing]$ (difference in group means)

**Exercises / in-class activities:**
- Sheather Problem 2.3 introduced (invoices data):
  - (a) Find 95% CI for start-up time ($\beta_0$)
  - (b) Test $H_0: \beta_1 = 0.01$ (benchmark processing time per invoice)
  - (c) Find prediction interval for processing 130 invoices
- Compare regression t-test result with Welch's two-sample t-test
- Examine boxplots and jitter plots before running formal tests

**Important pedagogical notes:**
- "The plots are hard to interpret. A statistical test may be more definitive." -- visual inspection alone is insufficient
- Intercept in dummy regression has a concrete meaning (mean of baseline group) -- but testing it may not be standard
- CI from t-test (0.32, 6.03) not containing 0 confirms significance
- Welch's t-test and dummy regression give the same difference in means but slightly different p-values

**Sugar Paradox connection:**
- Script 03b (`03b_food_groups_twfe.py`): The food groups analysis uses categorical variables (food group type) in a regression framework. The cross-sectional significant/not-significant pattern across 10 food groups is conceptually similar to dummy variable regression
- The changeover example (comparing methods) parallels comparing cross-sectional vs. within-country regression results -- two different "methods" of estimating the sugar-obesity relationship yield different conclusions

---

### M1-V4: SLR Practice (Sheather 2.3 Worked Example)

**Key concepts taught:**
- Complete worked example of hypothesis testing and prediction
- Custom hypothesis test: $H_0: \beta_1 = 0.01$ (benchmark value, not zero)
- Manual T-statistic computation: $T = (0.0112916 - 0.01)/0.0008184 = 1.578$
- Two-sided p-value = 0.1257 -- do not reject null
- Prediction at new x-value using `predict()` with `data.frame()`

**R functions introduced:**
- `dim(data)` -- check dimensions (n=30 gives df=28)
- `predict(model, data.frame(Invoices=130), interval="prediction")` -- prediction at new point
- `mean()`, `median()` -- basic summary statistics

**R code patterns (pedagogically important):**
```r
# Custom hypothesis test (H0: beta1 = 0.01)
T = (0.0112916 - 0.01) / (0.0008184)
2*pt(T, df=28, lower.tail=FALSE)  # p = 0.1257

# Prediction at new x value
predict(m1, data.frame(Invoices=130), interval="prediction")
```

**Datasets used:**
- `invoices.txt` (Sheather): 30 days of invoice processing data
  - Processing time (hours) vs. number of invoices
  - Slope estimate = 0.0112916 hours per additional invoice

**Key formulas:**
- $T = \frac{\hat{\beta}_1 - \beta_1^0}{se(\hat{\beta}_1)}$ with custom $\beta_1^0 = 0.01$

**Exercises / in-class activities:**
- Part (a): 95% CI for start-up time ($\beta_0$) using `confint()`
- Part (b): Test $H_0: \beta_1 = 0.01$ -- result: do not reject (p=0.1257), so the benchmark of 0.01 hours per invoice is plausible
- Part (c): Point estimate and 95% prediction interval for processing 130 invoices

**Important pedagogical notes:**
- Non-rejection of H0 does NOT mean the null is true -- it means the data are compatible with the null value
- This is the first time students test a non-zero null hypothesis manually
- Emphasizes the practical interpretation: 0.01 hours/invoice = 0.6 minutes/invoice benchmark is plausible

**Sugar Paradox connection:**
- Script 02 (`02_cross_sectional.py`): The cross-sectional partial correlation test asks "is the sugar coefficient zero?" The concept of testing non-zero hypotheses (as done here with benchmark 0.01) maps to asking "is the sugar effect at least X magnitude?" -- a more nuanced question than simple significance
- Script 05 (`05_robustness_failure.py`): The soybean oil case study tests whether an association is real vs. artifactual. This parallels testing whether the observed slope matches a benchmark value

---

## M2: Assumptions and Diagnostics

### M2-V1: Assumptions of Linear Regression

**Key concepts taught:**
- Goals of regression modeling: good predictions, interpretable factors, practical vs. statistical significance, stability, uncertainty quantification
- "Big picture" model appropriateness (Faraway Ch 4-5): representative samples, misspecification, publication bias, practical vs. statistical significance
- "Small picture" diagnostics (Sheather Ch 3, Faraway Ch 6): error assumptions, linear structure, outliers
- Anscombe's quartet: 4 datasets with identical slopes, p-values, and R-squared, but ONLY one appropriate linear fit
  - Lesson: summary statistics alone are misleading; always plot!
  - Identifies: outliers, non-linearity, bad leverage points
- Four model assumptions (redux):
  1. Constant variance ($\sigma^2$ constant across observations) -- homoskedasticity
  2. Independence ($cor(e_i, e_j) = 0$)
  3. Error uncorrelated with all $x$ variables
  4. Linear model: $E(Y) = \beta_0 + \beta_1 x$
- Unusual observations: small percentage of cases with strong effect on fit
- Residual plots diagnose assumption violations:
  - Linearity of fit
  - Leverage (extreme x-values)
  - Constant variance
  - Outliers (deviation from trend)
  - Serial correlation (time-ordered data)
  - Normality (prediction interval accuracy)

**R functions introduced:**
- `data(anscombe)` -- load built-in Anscombe's quartet
- `ggarrange()` from `ggpubr` -- arrange multiple plots
- `annotate_figure()` -- add titles to multi-panel figures
- `geom_abline()` -- add regression line with known intercept/slope

**Datasets used:**
- **Anscombe's quartet** (built-in R dataset): 4 synthetic datasets demonstrating the danger of relying on summary statistics alone. All have slope=0.5001, R-squared=0.6665.
  - Dataset 1: appropriate linear fit
  - Dataset 2: non-linear (quadratic) relationship
  - Dataset 3: outlier inflating fit
  - Dataset 4: bad leverage point (single extreme x)

**Key formulas:**
- $e \sim N(0, \sigma^2)$ (error assumptions)
- $E(Y) = \beta_0 + \beta_1 x$ (linear model assumption)

**Exercises / in-class activities:**
- Examine Anscombe's 4 datasets -- identify which fit is appropriate and why the others fail
- List the model assumptions and understand what each one means
- Understand the diagnostic checklist for residual plots

**Important pedagogical notes:**
- "Just because we can fit a straight line doesn't mean it is a good model"
- Anscombe's quartet is the central motivating example -- always visualize before trusting numbers
- The ordering of diagnostics matters: check linearity first, then other assumptions

**Sugar Paradox connection:**
- Script 04 (`04_trend_decomposition.py`): Trend R-squared of 99.68% for obesity in SSA is exactly the kind of near-perfect-fit statistic that Anscombe's quartet warns against. The high R-squared from trending data is misleading -- the "fit" comes from shared time trends, not causal relationships
- Script 05 (`05_robustness_failure.py`): The soybean oil case study is a real-world Anscombe lesson -- significant cross-sectional r, significant one-way FE, high Oster delta, yet the association is pure co-trending artifact. Summary statistics are misleading without diagnostic checks
- The "practical vs. statistical significance" distinction is central to the paper: sugar-obesity is statistically significant cross-sectionally but practically meaningless once confounding by shared trends is addressed

---

### M2-V2: Diagnostics Part 1 (Linearity, Variance, Normality, Serial Correlation)

**Key concepts taught:**
- Four standard diagnostic plots from `plot(lm_object)`:
  1. **Residuals vs. Fitted Values** -- checks linearity; no pattern expected. Curved pattern suggests quadratic or transformation needed
  2. **QQ-plot** -- checks normality of residuals. Straight line = normal. Deviations indicate heavy tails, skewness, or light tails
  3. **Scale-Location plot** -- checks constant variance. Plots $\sqrt{|\hat{r}_i|}$ vs. $\hat{y}_i$. Increasing/decreasing trend = heteroskedasticity
  4. **Residuals vs. Leverage** -- identifies influential points via Cook's distance contours
- Check linearity FIRST -- other diagnostics are meaningless if the model is misspecified
- Standardized residuals: $r_i = \hat{e}_i / (s\sqrt{1-h_{ii}})$
  - Accounts for fact that $Var(\hat{e}_i) = \sigma^2(1 - h_{ii})$
  - Residuals near high-leverage points have compressed variance
- Constant variance (homoskedasticity):
  - Critical assumption -- when violated, ALL inferential tools (p-values, CIs, PIs) fail unpredictably
  - "Duct tape fixes": transformations (Ch 3) or weighted least squares (Ch 4)
  - Fan shape in residual plots indicates heteroskedasticity
- Normality of residuals:
  - Residuals are combinations of errors: $\hat{e}_i = e_i - \sum_{j=1}^n h_{ij} e_j$
  - Kurtotic (heavy-tailed) and skewed distributions are most problematic
  - Most profound effect is on prediction intervals (use normal quantiles)
  - Remedies: transformations, additional variables, robust estimators, bootstrap
- QQ-plot interpretation:
  - Theoretical quantiles: $\Phi^{-1}(i/(n+1))$
  - Normal: range of sample quantiles matches theoretical
  - Heavy tails (Cauchy): extreme sample quantiles (-5, -20, -80)
  - Light tails (Uniform): smaller range than theoretical -- "not a serious issue"
  - Skewed (log-normal): asymmetric deviation from line
- Shapiro-Wilk test for normality: `shapiro.test()`
- Serial correlation (time-ordered data):
  - Leads to underestimation of standard errors
  - Durbin-Watson test: $DW = \sum_{i=2}^n (\hat{e}_i - \hat{e}_{i-1})^2 / \sum_{i=1}^n \hat{e}_i^2$
  - Lag-1 autocorrelation plot: $(\hat{e}_{i+1}, \hat{e}_i)$ scatterplot
  - Can also test with regression through origin

**R functions introduced:**
- `plot(lm_object)` -- standard 4-panel diagnostic plots
- `autoplot(lm_object)` from `ggfortify` -- ggplot-style diagnostic plots
- `rstandard(lm_object)` -- standardized residuals
- `shapiro.test()` -- formal test for normality
- `durbinWatsonTest(lm_object)` from `car` -- Durbin-Watson serial correlation test
- `geom_smooth(formula=y~x, method=loess)` -- loess smoother on residuals

**R code patterns (pedagogically important):**
```r
# Simulation: compare QQ plots for different error distributions
n <- 50; x = rnorm(n); fit = 5 + .5*x
e1 <- rnorm(n); e2 <- rcauchy(n); e3 <- runif(n); e4 <- exp(rnorm(n))
r1 = rstandard(lm(I(fit+e1)~x))  # I() protects arithmetic in formula
qqnorm(r1); qqline(r1)

# Serial correlation testing
fit.air = lm(nhtemp~wusa, data=globwarm)
sres.air = residuals(fit.air)
fit.air.res = lm(sres.air[2:145]~sres.air[1:144]-1)  # regression through origin
durbinWatsonTest(fit.air)
```

**Datasets used:**
- **Anscombe's quartet** (continued): m1 vs. m2 residual comparison
- **mpg** (built-in R dataset): cty vs. hwy mileage -- non-linear behavior check
- **globwarm** (faraway package): global warming from tree ring measurements over ~1000 years. Used for serial correlation example. nhtemp ~ wusa. DW test does not reject null.
- **airquality** (built-in, in Day2 version): Ozone vs. Temperature in LA, 4-month period

**Key formulas:**
- $Var(\hat{e}_i) = \sigma^2(1 - h_{ii})$
- $r_i = \frac{\hat{e}_i}{s\sqrt{1-h_{ii}}}$ (standardized residuals)
- $\hat{e}_i = e_i - \sum_{j=1}^n h_{ij} e_j$ (residual as combination of errors)
- $DW = \sum_{i=2}^n (\hat{e}_i - \hat{e}_{i-1})^2 / \sum_{i=1}^n \hat{e}_i^2$
- QQ theoretical quantiles: $\Phi^{-1}(i/(n+1))$

**Exercises / in-class activities:**
- Run the simulation code to generate QQ plots for Normal, Cauchy, Uniform, and Log-Normal errors
- Interpret diagnostic plots for Anscombe m1 (good) vs. m2 (non-linear)
- Run Shapiro-Wilk on normal vs. log-normal samples
- Examine lag-1 autocorrelation plots for global warming data

**Important pedagogical notes:**
- "Checking other assumptions won't make sense if a linear model is not the right choice" -- linearity FIRST
- Heteroskedasticity is described as "Serious!!!" -- all inferential tools fail unpredictably
- Uniform (light-tailed) QQ deviations "not a serious issue and can be ignored" -- asymmetry about how bad different violations are
- Normal QQ plots with n=50 "don't look perfectly straight" even for truly normal data -- sets calibration expectations
- Serial correlation "leads to underestimation of standard errors" -- inflates apparent significance

**Sugar Paradox connection:**
- Script 04 (`04_trend_decomposition.py`): The SE inflation analysis (1.9x to 3.8x) is a direct consequence of serial correlation in panel data. Obesity trends within countries are autocorrelated, inflating apparent significance of OLS estimates -- exactly the serial correlation problem taught here
- Script 04: Trend R-squared = 99.68% means residuals vs. fitted would show almost no scatter -- the model "fits" because both variables are trending, not because one causes the other
- Script 12 (`12_inference_robustness.py`): Wild cluster bootstrap and two-way clustering address the serial correlation and heteroskedasticity problems directly. The fact that bootstrap p=0.80 vs. naive p confirms how much inference depends on correct variance estimation
- The normality simulation (Cauchy vs. Normal QQ plots) parallels the concern about whether the sugar-obesity residuals meet distributional assumptions -- the paper checks this via permutation tests as a nonparametric alternative

---

### M2-V3: Diagnostics Part 2 (Leverage, Outliers, Cook's Distance)

**Key concepts taught:**
- **Leverage** = property of a point's x-coordinate location relative to others
  - $h_{ii} = 1/n + (x_i - \bar{x})^2/SXX$ (for SLR)
  - Leverages act as weights in fitted values: $\hat{y}_i = h_{ii} y_i + \sum_{j \neq i} h_{ij} y_j$
  - If $h_{ii} \approx 1$: fitted value determined by single observation (bad)
  - If $h_{ii}$ small: fitted value is an average of multiple observations (good)
  - Average leverage = $p/n$ (=2/n for SLR)
  - High leverage cutoff: $h_{ii} > 2p/n = 4/n$ for SLR
  - Larger $n$ leads to smaller leverage
  - High leverage points have SMALL residuals but HIGH fitted value variance
- **Bad leverage points**: extreme in x AND far from trend in y. Dramatically change the slope.
- **Outliers**: points that fall outside the trend but are NOT extreme in x. Affect residual SE ($s^2$), reducing significance of tests and widening intervals.
- **Clustering of points**: groups of points from different populations (e.g., "flower" bonds) -- really a mixture model problem
- Standardized residuals identify outliers: $r_i = \hat{e}_i / (s\sqrt{1-h_{ii}})$
  - Small datasets: outliers outside (-2, 2)
  - Large datasets: outliers outside (-4, 4)
- **Cook's Distance** combines leverage and outlier status:
  - $D_i = \frac{\sum_{j=1}^n (\hat{y}_{j(i)} - \hat{y}_j)^2}{2s^2}$
  - Shortcut: $D_i = \frac{r_i^2}{2} \cdot \frac{h_{ii}}{1-h_{ii}}$
  - Left term measures outlier status, right term measures leverage
  - If max $D_i \ll 1$: deleting the case does not change estimates much
  - Fox's cutoff: $4/(n-2)$; also look for gaps in half-normal plot
- Variance relationships:
  - $Var(\hat{e}_i) = \sigma^2(1-h_{ii})$ -- residual variance DECREASES with leverage
  - $Var(\hat{y}_i) = \sigma^2 h_{ii}$ -- fitted value variance INCREASES with leverage
  - This produces the hourglass shape of CI/PI bands
- What to do about outliers/leverage:
  - Do NOT routinely delete points just because they don't fit
  - Outliers are SIGNALS, not noise to remove
  - May indicate alternative model, additional predictor, or transformation
  - Removing without clear rationale overpromises model accuracy
  - "Be careful. You may be held to your model's exaggerated claims."

**R functions introduced:**
- `lm.influence(model)$hat` -- extract leverage values
- `rstandard(model)` -- standardized residuals
- `cooks.distance(model)` -- Cook's distance values
- `halfnorm(cook_dist, nlab=4)` from `faraway` -- half-normal plot with labeled points
- `influenceIndexPlot(model)` from `car` -- index plots of hat values, studentized residuals, Cook's distance, Bonferroni p-values
- `plot(model, 5, cook.levels=c(.12,.5,1))` -- residuals vs. leverage with Cook's contours
- `update(model, subset=...)` -- refit model excluding observations
- `flextable()` -- formatted tables

**R code patterns (pedagogically important):**
```r
# Add leverage and flag high-leverage points
huber$hii = lm.influence(fit1)$hat
huber$HighLev = (huber$hii > 2/3)  # cutoff = 4/n for n=6

# Handle bad leverage with quadratic term
fit3 = update(fit1, YBad ~ x + I(x^2))

# Compare CIs before and after removing influential points
bonds.fit2 = update(bonds.fit, subset=(1:35)[-c(4,13,35)])
cbind(confint(bonds.fit), confint(bonds.fit2))

# Half-normal plot with cutoff line
cd1 = cooks.distance(bonds.fit)
halfnorm(cd1, nlab=4)
abline(h=4/(35-2), lty=2)
```

**Datasets used:**
- **Huber's data** (Sheather): Synthetic data with 6 points illustrating bad leverage. YGood follows a clean linear trend; YBad has one point (x=10) moved away from trend. Shows how one bad leverage point rotates the entire regression line.
- **Anscombe datasets 3 and 4**: Dataset 3 has an outlier; Dataset 4 has a bad leverage point (single extreme x-value)
- **star** (faraway package): Stellar temperature vs. light intensity. Points 11, 20, 30, 34 form a cluster from a different population (giant stars vs. main sequence). Standardized residuals within (-1.98, 1.85) so no individual outliers, but huge group effect.
- **bonds** (Sheather): US Treasury bond prices. Bid Price vs. Coupon Rate for bonds maturing 1994-1998. Cases 4, 13, 35 are "flower" bonds with different tax consequences -- a mixture of two populations.
  - Removing flower bonds: slope increases, CI width decreases dramatically

**Key formulas:**
- $h_{ii} = 1/n + (x_i - \bar{x})^2/SXX$
- $\hat{y}_i = h_{ii} y_i + \sum_{j \neq i} h_{ij} y_j$
- $Var(\hat{e}_i) = \sigma^2(1-h_{ii})$
- $Var(\hat{y}_i) = \sigma^2 h_{ii}$
- $D_i = \frac{r_i^2}{2} \cdot \frac{h_{ii}}{1-h_{ii}}$ (Cook's distance)
- High leverage cutoff: $h_{ii} > 4/n$

**Exercises / in-class activities:**
- Compute and flag high-leverage points in Huber's data
- Compare fits with and without the bad leverage point
- Fit quadratic to Huber's bad data as alternative to deletion
- Identify flower bonds in treasury data using influenceIndexPlot
- Compare CIs before/after removing influential points
- Interpret Cook's distance half-normal plots and contour plots

**Important pedagogical notes:**
- "Points should not be routinely deleted from an analysis just because they do not fit the model"
- "Outliers and bad leverage points are signals"
- "Removing outliers without a clear rationale will paint an overly optimistic view of the accuracy of predicted outcomes"
- "Don't overpromise" -- being held to exaggerated model claims
- The bond example teaches that domain knowledge (flower bonds have different tax treatment) justifies separation, not just statistical diagnostics
- Clustering (star data, bond data) really indicates a mixture of populations that should be modeled separately

**Sugar Paradox connection:**
- Script 02 (`02_cross_sectional.py`): The LOO (Leave-One-Out) analysis (min partial r=0.417 dropping Mauritania) is a direct application of the influence/leverage concepts taught here. Each country is tested for its effect on the cross-sectional result.
- Script 08 (`08_cross_country_change.py`): The cross-country change analysis where initial obesity is the only significant predictor (t=3.45) could involve influential countries driving the result -- leverage diagnostics would be the natural check
- The bond "flower" example (mixture of populations) parallels the Sugar Paradox's subregion analysis: countries from different SSA subregions may behave differently (like flower vs. regular bonds), motivating subregion-stratified analysis
- Script 02: Subregion partial r=0.632 vs. full-sample r=0.496 -- controlling for subregion changes the result, similar to how removing flower bonds changed the bond regression

---

## M3: Transformations

### M3-V1: Transforms Part 1 (Variance Stabilization, Log-Log, Elasticity)

**Key concepts taught:**
- Process flow for regression modeling: Fit model -> Check diagnostics -> Transform if needed -> Refit -> Recheck
- Transformations improve models to better meet assumptions (linearity, constant variance)
- **Delta Method** for variance stabilization:
  - $V(f(Y)) = f'(E(Y))^2 \cdot V(Y)$ (first-order Taylor approximation)
  - $V(f(Y))$ is constant if $f'(E(Y))^2 = 1/(const \cdot V(Y))$
  - For Poisson ($E(Y) = V(Y)$): use $f(y) = \sqrt{y}$ -- gives $V(f(Y)) = 1/4$
  - For $E(Y) = sd(Y)$: use $f(y) = \log(y)$
- When X and Y measured in same units, consider same transformation for both
- Always back-transform predictions to original scale
- **Log-log regression** and **elasticity**:
  - $\log(Y) = \beta_0 + \beta_1 \log(x) + e$
  - $\beta_1$ = percentage change in Y for a 1% change in x (elasticity)
  - Without log(x): $\beta_1$ = percentage change in Y for a 1-unit change in x
  - Elasticity > 1: elastic; < 1: inelastic
- Practical example: Price elasticity of demand
  - Coefficient = -5.14: 1% price increase leads to 5% decrease in quantity sold
  - "Product is called elastic" -- Revenue = P*Q implications

**R functions introduced:**
- `I()` in formulas -- protects arithmetic operations (e.g., `I(SRP^(-0.25))`)
- `lm(sqrt(Rooms) ~ sqrt(Crews), data)` -- transform inside formula
- `lm(log(Sales) ~ log(Price), data)` -- log-log regression
- `autoplot(model)` from `ggfortify` -- diagnostic plots in ggplot style

**R code patterns (pedagogically important):**
```r
# Transform inside lm() for easy predictions
mod2 = lm(sqrt(Rooms) ~ sqrt(Crews), data=cleaning)

# Log-log model
food.fit = lm(log(Sales) ~ log(Price), data=food)

# Compare before/after transformation
p1 = ggplot(cleaning, aes(x=Crews, y=Rooms)) + geom_point() + geom_smooth(method="lm")
p2 = ggplot(cleaning, aes(x=sqrt(Crews), y=sqrt(Rooms))) + geom_point() + geom_smooth(method="lm")
ggarrange(p1, p2, ncol=2, nrow=1)
```

**Datasets used:**
- **cars04.csv** (Sheather): SuggestedRetailPrice vs. DealerCost for 2004 cars. Shows clear heteroskedasticity (variance increases with price). Tried reciprocal transform $y^{-0.25}$ vs. $x^{-0.25}$.
- **cleaning.txt** (Sheather): Industrial room cleaning data. Rooms cleaned vs. Crews. Square root transformation improves fit.
- **confood1.txt** (Sheather): Consumer packaged goods. Sales vs. Price.
  - Log-log model reveals elasticity = -5.14 (highly elastic product)
  - 1% price increase -> 5% decrease in sales
- **responsetransformation.txt** (Sheather): Simulated data for demonstrating transformation techniques

**Key formulas:**
- $V(f(Y)) = f'(E(Y))^2 \cdot V(Y)$ (delta method)
- $\log(Y) = \beta_0 + \beta_1 \log(x) + e$ (log-log model)
- $\beta_1$ in log-log = elasticity = % change in Y per 1% change in x
- For Poisson: $V(\sqrt{Y}) = 1/4$ (constant!)

**Exercises / in-class activities:**
- Room Cleaning Challenge: fit model before/after sqrt transformation, compare residual plots, predict for crews of size 2 and 4
- Examine cars04 data for heteroskedasticity in diagnostic plots
- Fit log-log model to food data and interpret elasticity coefficient
- Check residuals of log-log food model for remaining problems

**Important pedagogical notes:**
- "Don't forget to back transform predictions to the original scale"
- Process flow diagram is emphasized as the iterative modeling workflow
- The delta method provides the theoretical justification for why transformations stabilize variance
- "Check residuals vs. fitted, other factors may play a role" -- even after transformation, more work may be needed
- "Revenue = P*Q so raising price may be bad" -- connects statistics to business decisions

**Sugar Paradox connection:**
- Script 03 (`03_within_country_fe.py`): The panel models use `log GDP` as a control variable, applying the log transformation taught here. The interpretation is that a 1% increase in GDP corresponds to a fixed change in obesity -- an elasticity-style relationship
- Script 02 (`02_cross_sectional.py`): Cross-sectional specification controls for `log_gdp_pc` (log GDP per capita). This is a direct application of log transformation for a right-skewed variable
- The cars04 heteroskedasticity example (variance increasing with price) parallels how cross-country obesity data has more variance among wealthier/more urbanized countries -- the sugar-paradox panel may benefit from similar variance-stabilizing transformations
- Script 07 (`07_gdp_positive_control.py`): GDP detrended beta = -1.10 -- the sign reversal between cross-sectional (positive) and detrended (negative) GDP coefficients is more interpretable after log transformation

---

### M3-V2: Transforms Part 2 (Box-Cox, Inverse Response Plot, Simultaneous Transforms)

**Key concepts taught:**
- Power transformation family: $\Psi(Y, \lambda) = (Y^\lambda - 1)/\lambda$ if $\lambda \neq 0$; $\log(Y)$ if $\lambda = 0$
- Two methods for finding optimal transformation:
  1. **Inverse Response Plot** (graphical):
     - Fit initial model to get $\hat{y}$
     - Plot $\hat{y}$ (y-axis) vs. $y$ (x-axis) -- NOTE: reversed from usual
     - Overlay power transformations $\Psi(y, \lambda)$ for grid of $\lambda$ values
     - Choose $\lambda$ where transformed curve is closest to observed
     - Automated: minimize RSS of $\hat{y}_i = \alpha_0 + \alpha_1 \Psi(y, \lambda)$
  2. **Box-Cox** (likelihood-based):
     - Maximize log-likelihood by fixing $\lambda$ and finding $\beta_0, \beta_1$ that minimize $RSS(\lambda)$
     - Normalized power function uses geometric mean for unit consistency
     - Can transform $Y$, $X$, or both
     - "Tries to transform the data to make it more normal"
- Interpreting Box-Cox output:
  - Estimate for $\lambda$, rounded estimate, and CI
  - LR test $\lambda = 0$: tests whether log transform is adequate (reject if p small)
  - LR test $\lambda = 1$: tests whether NO transform is needed (reject if p small)
  - CI for $\lambda$ provides same information as both tests
- Transforming explanatory variables to normality using `powerTransform()`
- Transforming both Y and X:
  - Method 1: Transform X first with Box-Cox, then use IRP or Box-Cox to transform Y
  - Method 2: Simultaneous transformation using `powerTransform(cbind(Y, X) ~ 1)`
- Round $\lambda$ values to keep interpretable: identity (1), log (0), sqrt (0.5), 1/sqrt (-0.5), 1/x (-1)

**R functions introduced:**
- `powerTransform(model)` from `car` -- Box-Cox transformation estimation
- `powerTransform(variable)` -- transform a single variable to normality
- `powerTransform(cbind(Y, X) ~ 1, data)` -- simultaneous transformation
- `bcPower(variable, lambda)` from `car` -- apply Box-Cox power transformation
- `bcnPower()` -- Box-Cox with negatives allowed
- `inverseResponsePlot(model, lambda=c(...))` from `car` -- graphical method
- `summary(powerTransform_object)` -- shows estimate, CI, and LR tests
- `density()` -- kernel density estimate for visualization

**R code patterns (pedagogically important):**
```r
# Transform X to normality
lamx = powerTransform(cars04$DealerCost)
cars04$DCTrans = bcPower(cars04$DealerCost, lamx$lambda)

# Transform Y after X
m2 = lm(SuggestedRetailPrice ~ DCTrans, data=cars04)
bc = powerTransform(m2)
lam.y = bc$roundlam
cars04$SRPTrans = bcPower(cars04$SuggestedRetailPrice, lam.y)
m3 = lm(SRPTrans ~ DCTrans, data=cars04)

# Simultaneous transform of both
z = powerTransform(cbind(MaxSalary, Score) ~ 1, data=salarygov)
summary(z)
mod2.gov = lm(bcPower(MaxSalary, 0) ~ bcPower(Score, 0.5), data=salarygov)

# Inverse response plot
inverseResponsePlot(sim.t0, lambda=c(-1, -1/2, -1/3, 0, 1/3, 1/2, 1))

# Simple IRP code loop
lambda = seq(-1, 1, 0.01)
RSS = lambda
for(i in 1:length(lambda)) {
  ginvy = bcPower(sim.t$y, lambda[i])
  RSS[i] = sum(residuals(lm(yhat ~ ginvy))^2)
}
lambda[which.min(RSS)]
```

**Datasets used:**
- **responsetransformation.txt** (Sheather): Simulated data for demonstrating IRP and Box-Cox. Inverse response plot and Box-Cox give similar optimal $\lambda$.
- **cars04.csv** (continued): DealerCost transformed to normality with powerTransform. SuggestedRetailPrice then transformed with Box-Cox. Dramatic improvement in diagnostic plots.
- **salarygov.txt** (Sheather): Government salaries. MaxSalary vs. Score. Simultaneous transformation: $\lambda_Y = 0$ (log), $\lambda_X = 0.5$ (sqrt). So final model: $\log(MaxSalary) \sim \sqrt{Score}$.

**Key formulas:**
- $\Psi(Y, \lambda) = (Y^\lambda - 1)/\lambda$ if $\lambda \neq 0$; $\log(Y)$ if $\lambda = 0$
- IRP minimizes: $RSS = \sum (\hat{y}_i - \alpha_0 - \alpha_1 \Psi(y_i, \lambda))^2$
- Box-Cox minimizes: $RSS(\lambda) = \sum (\Psi_M(y_i, \lambda) - \hat{\beta}_0 - \hat{\beta}_1 x)^2$

**Exercises / in-class activities:**
- Apply IRP to simulated data and identify optimal lambda
- Compare IRP and Box-Cox results (should be similar)
- Transform cars04 data and compare diagnostic plots before/after
- Apply simultaneous transformation to salary data
- Interpret Box-Cox LR tests for lambda=0 and lambda=1

**Important pedagogical notes:**
- "Wow, what a long lecture" -- instructor acknowledges density of material
- Key takeaways explicitly listed:
  - Transformations correct violations of assumptions (constant variance, linearity)
  - Transform X can improve bad leverage
  - Can do X first then Y, or simultaneously
  - "It won't always work"
  - Round to keep simple and interpretable: identity, log, sqrt, 1/sqrt, 1/x
  - Use bcPower() in formulas for easier predictions
- "Consider the change in interpretation of coefficients and model after you transform. Are the tradeoffs worth it?"

**Sugar Paradox connection:**
- Script 02 and 03: The use of `log_gdp_pc` (log GDP per capita) throughout the sugar-paradox pipeline is implicitly a Box-Cox transformation with $\lambda = 0$. GDP is right-skewed, and log is the standard normalizing transformation
- Script 04 (`04_trend_decomposition.py`): The trend decomposition uses untransformed variables. If the sugar-obesity relationship is non-linear, a power transformation could reveal a different pattern -- but the paper shows the null holds regardless of specification
- The cars04 example (SuggestedRetailPrice ~ DealerCost with heteroskedasticity fixed by transformation) maps to the sugar-paradox context: even if you transform sugar supply and obesity to stabilize variance, the within-country null result persists because the problem is confounding by trends, not a transformation issue
- Script 07 (`07_gdp_positive_control.py`): The GDP positive control shows a sign reversal (positive cross-sectionally, negative detrended). This could be partly driven by non-linear GDP effects that log transformation addresses

---

## Day Lectures (In-Class Sessions)

### Day 1: Reviewing Simple Linear Regression

**Key concepts taught:**
This is a comprehensive in-class review that covers ALL of M1-V1 through M1-V4 in a single session. Content is essentially identical to the module videos with the following structure:

1. SLR model definition and notation
2. Conditional expectation and systematic/random components
3. Fitted values and residuals
4. Least squares estimation
5. Model assumptions
6. Sampling distributions of parameter estimates
7. T-tests and confidence intervals for slope
8. Confidence vs. prediction intervals (with hourglass shape)
9. Reading summary(lm()) output fully
10. Sums of squares decomposition (SST = RSS + SSreg)
11. F-statistics and ANOVA
12. Dummy variable regression (changeover example)
13. Practice problem (Sheather 2.3 -- invoices)

**Additional R code provided (not in videos):**
```r
# Practice problem solution hints
fit2 = lm(Time ~ Invoices, data=invoice)
sumary(fit2)
confint(fit2)
new.data = data.frame(Day=31, Invoices=130, Time=80)
predict(fit2, new.data, interval="confidence")
predict(fit2, new.data, interval="prediction")
```

**Datasets used:** production.txt, changeover_times.txt, invoices.txt (same as M1 videos)

**Sugar Paradox connection:** Same as M1 videos -- the full SLR toolkit is what Script 02 uses for cross-sectional analysis.

---

### Day 2: Model Assumptions and Regression Diagnostics

**Key concepts taught:**
This is a comprehensive in-class session covering ALL of M2-V1 through M2-V3 in a single lecture. Content is essentially identical to the module videos with the following structure:

1. Goals and assumptions of regression modeling
2. Anscombe's quartet
3. Basic model assumptions (error, linear model, unusual observations)
4. Diagnostic plots (Residuals vs. Fitted, QQ, Scale-Location, Residuals vs. Leverage)
5. Checking linearity first
6. Constant variance (homoskedasticity)
7. Normality of residuals (QQ-plot interpretation, Shapiro-Wilk)
8. Independence / Serial correlation (DW test)
9. Leverage and outliers (Huber's data, star data, bond data)
10. Cook's Distance
11. What to do about outliers -- ethical guidance

**Differences from module videos:**
- Uses `airquality` data (Ozone ~ Temp, LA, 4 months) instead of `globwarm` for serial correlation
- Uses `autoplot()` more extensively for Huber's data
- Slightly different code for some examples but identical concepts

**Datasets used:** Anscombe's quartet, mpg, airquality (or globwarm), Huber's data, star, bonds (same as M2 videos)

**Sugar Paradox connection:** Same as M2 videos -- diagnostic thinking is the foundation for recognizing that significant cross-sectional associations can be artifacts of shared trends.

---

## Cross-Module Summary: Sugar Paradox Script Mapping

| Course Concept | Sugar Paradox Script | Specific Connection |
|---|---|---|
| SLR: fitting, slope, R-squared | 02_cross_sectional.py | Cross-sectional sugar-obesity regression, partial r=0.496 |
| t-tests and p-values | 02, 03, 10 | t-statistics across specifications; cascade from significant to null |
| Confidence/prediction intervals | 02 | CI for partial correlation; LOO sensitivity |
| F-tests, ANOVA | 03_within_country_fe.py | F-tests for entity and year fixed effects |
| Dummy variable regression | 03b_food_groups_twfe.py | Food group categorical variables; subregion dummies in 02 |
| Custom hypothesis tests | 05_robustness_failure.py | Testing whether soybean oil coefficient equals zero under various specs |
| Residuals vs. fitted | 04_trend_decomposition.py | Trend R-squared = 99.68% leaves almost no residual variation |
| Heteroskedasticity | 12_inference_robustness.py | Two-way clustering addresses potential heteroskedasticity |
| Serial correlation, DW test | 04_trend_decomposition.py | SE inflation (1.9x-3.8x) from autocorrelated obesity trends |
| Normality of residuals | 12_inference_robustness.py | Permutation test as nonparametric alternative to normal-theory tests |
| Leverage and influence | 02_cross_sectional.py | LOO analysis, min partial r=0.417 dropping Mauritania |
| Cook's Distance / outlier ethics | 02, 08 | "Do not routinely delete" parallels: countries should not be dropped without rationale |
| Log transformation | 02, 03, 07 | log_gdp_pc used throughout as control variable |
| Variance stabilization | 04 | Entity FE absorb 95.73% of variance; year FE absorb more |
| Elasticity (log-log) | 07_gdp_positive_control.py | GDP elasticity with respect to obesity; sign reversal interpretation |
| Box-Cox / power transforms | Not directly used | Panel models use level variables; transformations could be explored |
| Anscombe's lesson (don't trust summaries alone) | 05_robustness_failure.py | Soybean oil passes all summary-statistic tests yet is a co-trending artifact |
