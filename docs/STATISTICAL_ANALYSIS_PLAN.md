# Statistical Analysis Plan — Re-Run with max_tokens=16384

> **Purpose:** Pre-registered analysis plan for the experiment re-run. Written before data collection so that analytical choices cannot be influenced by observed results. Reference this document during analysis; deviations must be documented and justified.
>
> **Context:** The v3 experiment (1,200 episodes) revealed extreme variance heterogeneity in C1 (SD=110,546 vs C2 SD=5,419) caused by truncation cascades at max_tokens=4096. When the LLM hits the token cap (`stop_reason != "end_turn"`), the runner initiates a continuation step, which in turn may also truncate, creating a multiplicative blowup. The re-run sets max_tokens=16384, which should largely eliminate truncation cascades and bring C1's variance into a range where parametric assumptions are more defensible.
>
> **Design recap:** 4 conditions x 300 tasks = 1,200 episodes. The same 300 tasks are presented in the same order to all 4 conditions. This creates a paired/repeated-measures structure (task as a blocking factor).

---

## 1. Pre-Registered Hypotheses

### Primary Hypotheses (confirmatory)

**H1 (Token Efficiency):** Conditions with feedback (C2, C3, C4) produce lower total token consumption than the control condition (C1).
- **H1a:** C2 (Likert feedback) < C1 (control)
- **H1b:** C3 (Likert + explanation embedding) < C1 (control)
- **H1c:** C4 (qualitative feedback) < C1 (control)
- **Direction:** One-sided (treatment < control). We have strong prior from v3 showing 73-77% reductions.

**H2 (Multi-Step Prevention):** Conditions with feedback have a lower rate of multi-step episodes (step_count > 1) than C1.
- Same three sub-hypotheses as H1.
- This tests the mechanism discovered in v2: structured feedback prompts prevent LLM exploration spirals.

**H3 (Retrieval Differentiation):** Mean pairwise Jaccard similarity of retrieved skill sets is lower between treatment conditions (C2 vs C4, C3 vs C4) than between conditions sharing the same feedback modality (C2 vs C3).
- Tests whether different feedback mechanisms drive different retrieval patterns.

### Secondary Hypotheses (exploratory)

**H4 (Bandit Convergence):** Treatment conditions (C2, C3) converge to a stable best arm within 300 tasks, while C4 does not.

**H5 (Ground Truth Hit Rate):** Treatment conditions have higher ground truth hit rates than C1.

**H6 (Feedback Modality):** Structured feedback (C2+C3 pooled) produces different mean reward scores than qualitative feedback (C4).

**H7 (Difficulty Interaction):** The token efficiency advantage of treatment conditions over C1 is larger for hard tasks than for easy tasks. (Treatment x Difficulty interaction.)

---

## 2. Statistical Tests by Metric

### 2.1 Token Consumption (Primary Outcome)

Token counts are non-negative, right-skewed, and potentially heavy-tailed. Even with max_tokens=16384, we expect right skew from multi-step episodes.

**Decision tree for test selection (evaluate after data collection, before running tests):**

1. **Check distributional assumptions:**
   - Shapiro-Wilk test on per-condition residuals (n=300 each; W > 0.95 suggests approximate normality)
   - Compute skewness and kurtosis. If |skewness| < 2 and |kurtosis| < 7, parametric tests are reasonable.
   - Compute variance ratio: max(SD) / min(SD). If ratio < 4, variance heterogeneity is manageable.

2. **If assumptions hold (skewness/kurtosis within bounds, variance ratio < 4):**
   - **Primary test:** Welch's t-test (unequal variance t-test) for each pairwise comparison.
   - Welch's t is preferred over Student's t because it does not assume equal variance and is robust to moderate non-normality at n=300 (Central Limit Theorem).
   - Report: t-statistic, degrees of freedom (Welch-Satterthwaite), p-value.

3. **If assumptions fail (skewness > 2 OR variance ratio > 4):**
   - **Primary test:** Mann-Whitney U test (rank-based, distribution-free).
   - **Sensitivity analysis:** Permutation test (10,000 permutations) on the mean difference as a robustness check.
   - Report: U-statistic, p-value, permutation p-value.

4. **Regardless of which test is primary, always also report:**
   - Bootstrap 95% CI on the mean difference (BCa method, 10,000 resamples)
   - Both Welch's t and Mann-Whitney p-values in a supplementary table so readers can compare

**Why Welch's t over Mann-Whitney (when assumptions hold):**
Mann-Whitney tests for stochastic dominance (P(X > Y) != 0.5), not for a difference in means. When the research question is "does feedback reduce average token consumption," Welch's t directly tests the quantity of interest. Mann-Whitney can reject with identical means if distributions differ in shape.

**Omnibus test (before pairwise):**
- If assumptions hold: Welch's ANOVA (one-way ANOVA with no equal-variance assumption, i.e., `scipy.stats.f_oneway` is inappropriate; use the Welch version via `pingouin.welch_anova` or manual implementation).
- If assumptions fail: Kruskal-Wallis H-test.
- Only proceed to pairwise comparisons if the omnibus test is significant at alpha=0.05.

### 2.2 Multi-Step Rate (Binary Outcome)

Multi-step rate = proportion of episodes with step_count > 1.

**Test:** Chi-square test of independence (4x2 contingency table: condition x {single-step, multi-step}).
- If any expected cell count < 5: Fisher's exact test via Monte Carlo simulation (scipy does not support 4x2 exact tests natively; use `scipy.stats.fisher_exact` for 2x2 follow-ups, or `scipy.stats.chi2_contingency` with simulated p-value).
- Follow-up pairwise comparisons: 2x2 Fisher's exact test for each C_i vs C1.

**Effect size:** Risk ratio (RR) and risk difference (RD) for each treatment vs C1.
- RR = P(multi-step | treatment) / P(multi-step | C1).
- 95% CI for RR via log-transform method.

### 2.3 Retrieval Quality Metrics

#### Jaccard Similarity (Continuous, [0, 1])

Per-task Jaccard similarity between pairs of conditions. Each task yields one Jaccard value per condition pair.

**Test:** Paired Wilcoxon signed-rank test (or paired t-test if approximately normal).
- Jaccard values are bounded [0, 1] and potentially non-normal (mass at 0 and 1).
- Compare: mean Jaccard(C2 vs C4) vs mean Jaccard(C2 vs C3).
- This is a paired comparison because both Jaccard values come from the same task.

#### Ground Truth Hit Rate (Binary per Task)

Whether any ground-truth skill appears in the top-5 retrieved skills.

**Test:** McNemar's test for paired binary outcomes.
- For each task, we have a hit/miss indicator for each condition.
- McNemar's tests whether the marginal proportions differ between paired conditions.
- For C_i vs C1: 2x2 table of (hit_i, hit_1) per task. McNemar's chi-square on the off-diagonal cells.

#### NDCG@5 and MRR (Continuous, Bounded)

Per-task NDCG@5 and MRR values, bounded [0, 1].

**Test:** Paired Wilcoxon signed-rank test.
- These metrics have point masses (many tasks may have NDCG=0 or NDCG=1) making normality unlikely.
- Alternative: Bootstrap paired difference of means (BCa, 10,000 resamples).

### 2.4 Reward Scores (C2, C3, C4 Only)

Per-episode mean reward from Likert/qualitative feedback. C1 has no reward (no feedback).

**Test:** Kruskal-Wallis across C2, C3, C4; followed by pairwise Mann-Whitney (C2 vs C3, C2 vs C4, C3 vs C4).
- Reward scores are ordinal/bounded [0, 1], not normally distributed.

### 2.5 Bandit Convergence

Convergence is not directly testable with a single p-value. Report descriptively:
- Episode at which the best arm stabilizes (defined as: the argmax of posterior means does not change for the last 20 consecutive episodes).
- Final posterior mean of the best arm.
- Entropy of the posterior distribution over arms (lower = more converged).

---

## 3. Multiple Comparison Correction

**Family-wise error rate (FWER) approach for confirmatory hypotheses:**

The primary family consists of 6 pairwise comparisons for token consumption (C1 vs C2, C1 vs C3, C1 vs C4, C2 vs C3, C2 vs C4, C3 vs C4).

**Correction method: Holm-Bonferroni (step-down)**

Holm-Bonferroni is uniformly more powerful than Bonferroni while still controlling FWER at alpha. It rejects more hypotheses when there is a mix of strong and weak effects (which we expect: C1 vs treatment will be strong, treatment vs treatment will be weak).

**Procedure:**
1. Order the 6 p-values from smallest to largest: p_(1) <= p_(2) <= ... <= p_(6).
2. For rank k, reject H_(k) if p_(k) < alpha / (6 - k + 1).
3. Stop at the first non-rejection; all remaining hypotheses are not rejected.

**For secondary/exploratory analyses:** Benjamini-Hochberg (FDR control at q=0.05). FDR is appropriate for exploratory work where we want to flag interesting patterns without being overly conservative.

**Report all three corrections** (raw, Holm-Bonferroni, BH) in a supplementary table so readers with different preferences can evaluate.

---

## 4. Effect Size Measures

Report ALL of the following for each pairwise comparison. Different readers/reviewers expect different measures, and with the variance heterogeneity in our data, no single measure tells the whole story.

### 4.1 For Token Consumption

| Measure | Formula | When it's most informative |
|---------|---------|----------------------------|
| **Cohen's d** | (M1 - M2) / s_pooled | Standard; assumes roughly equal variance. Will be misleading if C1 SD is still much larger. |
| **Glass's delta** | (M1 - M2) / s_control | Uses only the control group's SD as the denominator. Appropriate when treatment is expected to reduce variance (our case). Report this as the primary effect size if variance ratio > 2. |
| **Cliff's delta** | 2 * (U / (n1 * n2)) - 1 | Non-parametric ordinal effect size. Ranges [-1, 1]. Interpretable as: "probability that a random treatment observation is lower than a random control observation, minus the reverse." Robust to outliers and skew. |
| **Percent reduction** | (M_C1 - M_Cx) / M_C1 * 100 | Most practically interpretable for stakeholders. "Feedback reduces token consumption by X%." |
| **Common Language Effect Size (CLES)** | P(X_treatment < X_control) | = U / (n1 * n2). "There is a Z% probability that a random feedback episode uses fewer tokens than a random control episode." Directly interpretable. |

**Primary effect size for the paper:** Glass's delta (if variance heterogeneity persists) or Cohen's d (if variances equalize). Report Cliff's delta as a robustness check in all cases.

**Interpretation thresholds:**
- Cohen's d / Glass's delta: |0.2| small, |0.5| medium, |0.8| large
- Cliff's delta: |0.147| small, |0.33| medium, |0.474| large

### 4.2 For Binary Outcomes (Multi-Step Rate)

| Measure | What it means |
|---------|---------------|
| **Risk Ratio (RR)** | How many times more/less likely is multi-step in treatment vs control? |
| **Risk Difference (RD)** | Absolute percentage point difference. |
| **Odds Ratio (OR)** | Traditional epidemiological measure. Report for completeness. |

### 4.3 For Retrieval Quality (NDCG, MRR, Jaccard)

| Measure | Rationale |
|---------|-----------|
| **Cliff's delta** | These are bounded [0,1] and non-normal. Cliff's delta is appropriate. |
| **Mean difference + Bootstrap CI** | Directly interpretable magnitude. |

---

## 5. Power Analysis and Minimum Detectable Effect Size

### 5.1 For Welch's t-test (Token Consumption)

With n=300 per condition, alpha=0.05 (two-sided), power=0.80:

**Minimum detectable Cohen's d = 0.229**

Derivation: For a two-sample t-test, the minimum detectable effect size is:
```
d = (z_{alpha/2} + z_{beta}) * sqrt(1/n1 + 1/n2)
d = (1.96 + 0.84) * sqrt(1/300 + 1/300)
d = 2.80 * sqrt(2/300)
d = 2.80 * 0.0816
d = 0.229
```

**What this means in practice:**

If the re-run achieves similar SD to C2/C3/C4 in v3 (SD ~ 6,000-8,000 tokens), then:
- Pooled SD ~ 7,000
- MDE in raw units: 0.229 * 7,000 = **1,603 tokens**
- As percentage of C1 mean (~6,000 if truncation cascades are fixed): **~27%**

If C1's variance drops substantially with max_tokens=16384 (expected):
- Pooled SD might be ~ 4,000
- MDE in raw units: 0.229 * 4,000 = **916 tokens**
- As percentage of C1 mean: **~15%**

We have ample power to detect the effects seen in v3 (73-77% reductions). Even if the effect shrinks substantially post-fix, we can detect differences down to ~15-27%.

### 5.2 For Mann-Whitney U (if needed)

The asymptotic relative efficiency (ARE) of Mann-Whitney vs t-test is >= 0.864 for any distribution (and = pi/3 ~ 1.047 for normal). With n=300 per group, power is essentially equivalent.

### 5.3 For Chi-Square (Multi-Step Rate)

With n=300 per condition and v3 rates (C1=25.3%, C2=15.0%):
- Detectable difference: ~8 percentage points at power=0.80 (using normal approximation for proportion test).
- This is well within the observed 10-14 point gaps.

### 5.4 For Paired Tests (Retrieval Quality)

The paired structure increases power because within-task variance is eliminated. With 300 tasks:
- For paired Wilcoxon: MDE ~ 0.20 in Cliff's delta
- For McNemar's: MDE ~ 8 percentage points in hit rate difference

---

## 6. Handling the Paired Structure

### 6.1 The Problem

The same 300 tasks run across all 4 conditions in the same order. Observations are NOT independent across conditions for the same task. A hard task will tend to produce more tokens in ALL conditions. Ignoring this pairing wastes statistical power and can bias standard errors.

### 6.2 Recommended Approach: Paired Analysis as Primary

**For token consumption:**
- Compute per-task difference: delta_i = tokens(C1, task_i) - tokens(Cx, task_i)
- Test whether mean(delta) != 0 using a one-sample t-test on the differences (or Wilcoxon signed-rank if non-normal).
- This is equivalent to a paired t-test and is strictly more powerful than an unpaired test when task-level correlation is positive (which it will be).
- Report: mean difference, paired 95% CI, paired t-statistic.

**For retrieval quality (Jaccard, NDCG, MRR, hit rate):**
- Already inherently paired (one value per task per condition).
- Use paired Wilcoxon signed-rank or McNemar's as described in Section 2.

### 6.3 Mixed-Effects Models (Supplementary Analysis)

A linear mixed-effects model accounts for both the paired structure and potential task-level heterogeneity:

```
tokens_ij = beta_0 + beta_1 * condition_j + (1 | task_i) + epsilon_ij
```

Where:
- `tokens_ij` = total tokens for task i under condition j
- `condition_j` = fixed effect (dummy-coded, C1 as reference)
- `(1 | task_i)` = random intercept for task (captures task difficulty)
- `epsilon_ij` = residual

**Extensions:**
- `(1 | task_i) + (1 | theme_k)` — crossed random effects for task and theme
- `beta_2 * difficulty + beta_3 * condition * difficulty` — interaction with difficulty level
- Log-transform the outcome if residuals are heavily right-skewed: `log(tokens_ij)`

**Implementation:** `statsmodels.formula.api.mixedlm` or `pymer4` (Python wrapper for R's lme4).

**When to use:** The mixed-effects model is the **supplementary** analysis, not the primary. Primary results should use the simpler paired tests because:
1. They are easier for reviewers to verify.
2. They make fewer distributional assumptions.
3. The mixed-effects model adds complexity without changing the conclusion when there are only 4 conditions and 300 tasks.

**But DO run the mixed model** and report it as a robustness check. If the mixed model and paired tests agree (they should), this strengthens the finding. If they disagree, investigate why (likely driven by the random-effects structure absorbing variance).

### 6.4 Task Order Effects

Tasks are presented in the same shuffled order to all conditions. The bandit's behavior on task 250 depends on feedback from tasks 1-249. This creates temporal dependence.

**Check:** Plot per-condition mean tokens in sliding windows (e.g., 30-task windows). If there is a trend (increasing or decreasing over time), this suggests the bandit is learning or drifting.

**If trend is present:** Include `task_order` as a covariate in the mixed model:
```
tokens_ij = beta_0 + beta_1 * condition + beta_2 * order + (1 | task_i) + epsilon
```

---

## 7. Bootstrap Confidence Intervals

### 7.1 Method: Bias-Corrected and Accelerated (BCa)

BCa bootstrap CIs correct for both bias (the bootstrap distribution may not be centered on the sample statistic) and skewness (the sampling distribution may be asymmetric). This is critical for token data which is right-skewed.

**Standard percentile bootstrap CIs are NOT sufficient** because they assume the sampling distribution is symmetric around the estimate. BCa adjusts the percentile cutpoints.

### 7.2 Number of Resamples: 10,000

- 10,000 is the standard recommendation for 95% CIs (Efron & Tibshirani, 1993).
- For 99% CIs: use 50,000.
- The current codebase uses 10,000 (Bayesian bootstrap via Dirichlet weights). Switch to BCa for the re-run.
- Set seed=42 for reproducibility.

### 7.3 Implementation

```python
from scipy.stats import bootstrap

# For a single mean
result = bootstrap(
    (data,), np.mean,
    n_resamples=10000,
    method='BCa',
    random_state=42,
    confidence_level=0.95
)
ci = result.confidence_interval

# For a difference in means
def mean_diff(x, y):
    return np.mean(x) - np.mean(y)

# Or use paired difference:
diffs = tokens_c1 - tokens_cx  # per-task differences
result = bootstrap(
    (diffs,), np.mean,
    n_resamples=10000,
    method='BCa',
    random_state=42,
    confidence_level=0.95
)
```

### 7.4 What to Bootstrap

| Quantity | Bootstrap needed? | Notes |
|----------|-------------------|-------|
| Mean token consumption per condition | Yes (BCa) | Report alongside parametric CI |
| Mean difference (paired) | Yes (BCa) | Primary CI for the effect |
| Cohen's d / Glass's delta | Yes (BCa) | Effect size CI is often more informative than the p-value |
| Cliff's delta | Yes (BCa) | |
| NDCG@5 mean per condition | Yes (BCa) | Bounded metric, normality unlikely |
| Jaccard mean per condition pair | Yes (BCa) | Bounded metric |

---

## 8. Visualization Plan

### 8.1 Main Paper Figures (4-5 figures)

**Figure 1: Token Consumption by Condition**
- Violin plot with embedded box plot (shows distribution shape, median, IQR).
- Overlay individual data points (jittered) with alpha=0.15 for n=300.
- Mark the mean with a diamond.
- Annotate with paired significance brackets (Holm-Bonferroni adjusted p-values).
- Panel (a): all episodes. Panel (b): single-step only. Panel (c): multi-step only.

**Figure 2: Multi-Step Rate by Condition**
- Stacked bar chart: proportion single-step vs multi-step per condition.
- Error bars: 95% Wilson score CI for each proportion.
- Annotate with risk ratios relative to C1.

**Figure 3: Retrieval Differentiation**
- Heatmap of mean Jaccard similarity between all condition pairs.
- Color scale from blue (low similarity = more differentiation) to red (high similarity = identical retrieval).
- Annotate cells with values.

**Figure 4: Bandit Convergence**
- Line plot: posterior mean of top arm over episodes (x-axis: episode number, y-axis: posterior mean).
- One panel per treatment condition (C2, C3, C4).
- Shade the 95% credible interval.
- Vertical dashed line at convergence point.

**Figure 5: Effect Sizes Forest Plot**
- Forest plot showing Glass's delta (or Cohen's d) with 95% BCa CI for all pairwise comparisons.
- Vertical reference line at 0 (no effect) and at 0.2/0.5/0.8 (small/medium/large).
- Color-coded by comparison type (control vs treatment = blue, treatment vs treatment = gray).

### 8.2 Appendix Figures (5-7 figures)

**Figure A1: Per-Difficulty Token Consumption**
- Faceted violin plots: Easy / Medium / Hard (columns) x Condition (x-axis).

**Figure A2: Per-Theme Reward Heatmap**
- Heatmap: 5 themes (rows) x 3 treatment conditions (columns). Cell color = mean reward.

**Figure A3: NDCG@5 Distribution**
- Histogram or ECDF per condition, overlaid.

**Figure A4: Token Consumption Over Time**
- Line plot: 30-task sliding window mean per condition. Shows temporal trends and bandit learning.

**Figure A5: Residual Diagnostics**
- Q-Q plot of paired differences (for each C_i vs C1).
- Histogram of paired differences with normal overlay.
- This justifies whether Welch's t or Wilcoxon is more appropriate.

**Figure A6: Outlier Sensitivity**
- Plot the C1 vs C_x mean difference as a function of the outlier exclusion threshold (e.g., remove episodes > mean + k*SD for k = 1, 2, 3, 4, none). Shows robustness.

**Figure A7: Bootstrap Distribution**
- Histogram of 10,000 bootstrap replications of the mean difference (C1 - C2). Mark the observed statistic, BCa CI bounds.

---

## 9. Analysis Pipeline Execution Order

Run in this order after experiment completion:

1. **Data extraction:** Export episodes, step_log, retrieval_results, bandit_state from SQLite.
2. **Distributional diagnostics:** Shapiro-Wilk, skewness, kurtosis, variance ratios. Decision: Welch's t vs Mann-Whitney.
3. **Primary analysis (token consumption):**
   a. Omnibus test (Welch's ANOVA or Kruskal-Wallis)
   b. Pairwise paired tests (paired t or Wilcoxon signed-rank)
   c. BCa bootstrap CIs on paired differences
   d. Effect sizes (Glass's delta, Cliff's delta, percent reduction, CLES)
   e. Holm-Bonferroni and BH correction
4. **Multi-step analysis:** Chi-square / Fisher's exact, risk ratios.
5. **Retrieval quality:** Paired Wilcoxon on NDCG, MRR. McNemar's on GT hit rate. Paired Wilcoxon on Jaccard.
6. **Reward analysis:** Kruskal-Wallis across C2/C3/C4.
7. **Mixed-effects model:** Run as supplementary robustness check.
8. **Subgroup analyses:** Per-difficulty, per-theme.
9. **Temporal analysis:** Sliding window plots, order effects.
10. **Generate all figures and tables.**

---

## 10. Deviations from v3 Analysis

| v3 Analysis | Re-Run Change | Rationale |
|-------------|---------------|-----------|
| Mann-Whitney U as primary test | Welch's t-test (if assumptions hold) | MW tests stochastic dominance, not mean difference. With reduced variance heterogeneity, Welch's t directly tests the hypothesis. |
| Bayesian bootstrap (Dirichlet) | BCa bootstrap | BCa provides better coverage for skewed distributions. Bayesian bootstrap with flat Dirichlet prior is equivalent to the standard nonparametric bootstrap asymptotically, but BCa adds bias/acceleration corrections. |
| Bonferroni correction only | Holm-Bonferroni (primary) + BH (secondary) | Holm-Bonferroni is uniformly more powerful than Bonferroni with the same FWER guarantee. |
| Cohen's d only | Glass's delta (primary) + Cliff's delta + CLES | Cohen's d uses pooled SD, which is misleading when one group has much larger variance. Glass's delta uses only the control SD. |
| Unpaired comparisons | Paired tests (primary) | The design is paired (same 300 tasks). Paired tests are strictly more powerful. |
| No mixed-effects model | Mixed model as supplementary | Accounts for task-level heterogeneity in a principled way. |
| 10,000 bootstrap resamples | 10,000 (unchanged, switch to BCa method) | BCa method is more important than resample count. |

---

## 11. Sensitivity Analyses

Run these regardless of primary results:

1. **Outlier exclusion:** Re-run primary analysis after excluding episodes with tokens > mean + 3*SD. Report whether conclusions change.
2. **Log-transform:** Re-run Welch's t on log(tokens). This handles right-skew and makes multiplicative effects additive. Report back-transformed CIs.
3. **Trimmed means:** 10% trimmed mean comparison (robust to extreme values). Use Yuen's test (the trimmed-mean analog of Welch's t).
4. **Per-difficulty stratification:** Run the full primary analysis separately for Easy, Medium, and Hard tasks. Does the effect replicate within each stratum?
5. **First 150 vs last 150 tasks:** Split-half temporal analysis. If the effect is present in both halves, it is not driven by bandit learning trajectory.

---

## 12. What Constitutes a "Significant" Result

**For the paper:**
- Primary hypotheses (H1-H3): Holm-Bonferroni adjusted p < 0.05, with effect size CI excluding 0.
- Secondary hypotheses (H4-H7): BH-adjusted p < 0.05, flagged as exploratory.
- Report exact p-values, not just significance stars.
- Report effect sizes and CIs for ALL comparisons, regardless of significance.

**Do NOT report a result as "significant" based solely on p < 0.05.** The paper should emphasize effect sizes and practical significance (e.g., "27% token reduction, Glass's delta = 0.85 [0.62, 1.08]") over binary significance testing.

---

## 13. Code Changes Required

The current `analyze_experiment.py` needs these modifications for the re-run:

1. **Add paired tests:** Compute per-task differences and run one-sample t-test / Wilcoxon signed-rank on those differences.
2. **Add distributional diagnostics:** Shapiro-Wilk, skewness, kurtosis, variance ratio computation before choosing test.
3. **Switch bootstrap to BCa:** Replace `bayesian_bootstrap_mean`/`bayesian_bootstrap_diff` with `scipy.stats.bootstrap` using `method='BCa'`.
4. **Add Glass's delta and Cliff's delta:** New functions alongside existing `cohens_d`.
5. **Add Holm-Bonferroni:** New correction method alongside existing Bonferroni and BH.
6. **Add McNemar's test:** For paired binary outcomes (GT hit rate).
7. **Add mixed-effects model:** Using `statsmodels.formula.api.mixedlm`.
8. **Add CLES computation:** `U / (n1 * n2)` from Mann-Whitney U statistic.
9. **Add chi-square / Fisher's exact:** For multi-step rate contingency tables.
10. **Update max_tokens in config:** `experiment.yaml` line 61: `max_tokens: 16384`.
