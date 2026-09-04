# Final Research Report

## 1. Research question

This project asks:

> When does uncertainty-aware mean-variance optimization improve portfolio decisions after forecasting, risk, universe redundancy, accounting, and trading frictions are separated?

The complete analysis specification is in `configs/final_analysis.json`, and the final tables and figures are in `results/final/`.

## 2. Mathematical formulation

For 147 ETFs, the feasible set at decision date $t$ is

```math
\mathcal{W}_t = \{\, w : \mathbf{1}^\top w = 1,\; 0 \le w_i \le 0.10 \,\}.
```

The nominal problem under a common risk ceiling is

```math
\max_{w \in \mathcal{W}_t} \hat{\mu}_t^\top w
\quad \text{s.t.} \quad
w^\top \hat{\Sigma}_t w \le \sigma_\star^2.
```

The expected-return estimate is the annualized arithmetic mean of the preceding 504 daily returns. The covariance estimate is corrected IEWMA with 21-day volatility and 63-day correlation half-lives, standardized using the prior volatility state, winsorized at 4.2, annualized by 252, and eigenvalue-floored as specified in `configs/core_experiment.json`. We distinguish between the requested ceiling $\sigma_\star$, attained model-predicted volatility, common-base predicted volatility, and realized volatility.

## 3. Robust formulations

Let $S = \mathrm{diag}(s)$. Under box uncertainty,

```math
\mathcal{U}_\mu^{\mathrm{box}} = \{\, \hat{\mu} + Su : \|u\|_\infty \le \rho \,\}.
```

Norm duality gives

```math
\inf_{\mu \in \mathcal{U}_\mu^{\mathrm{box}}} \mu^\top w
= \hat{\mu}^\top w - \rho \|Sw\|_1.
```

Because the portfolios are long-only,

```math
\|Sw\|_1 = s^\top w,
```

so the resulting objective is

```math
\hat{\mu}^\top w - \rho s^\top w.
```

For ellipsoidal mean uncertainty,

```math
\mathcal{U}_\mu^{\mathrm{ell}} = \{\, \hat{\mu} + C_\mu^{1/2} u : \|u\|_2 \le \rho \,\},
```

which gives

```math
\inf_{\mu \in \mathcal{U}_\mu^{\mathrm{ell}}} \mu^\top w
= \hat{\mu}^\top w - \rho \sqrt{w^\top C_\mu w}.
```

For diagonal variance uncertainty, the worst-case covariance contribution is

```math
\sup_{\Sigma} w^\top \Sigma w
=
w^\top
\left[
\hat{\Sigma}
+
\kappa \,\mathrm{diag}(\mathrm{diag}(\hat{\Sigma}))
\right]
w.
```

Bootstrap standard errors, $C_\mu$, $\rho$, and $\kappa$ are estimated from each date's 504-row training window using 500 circular moving-block draws, 21-observation blocks, deterministic seeds, and 95% empirical coverage.

## 4. Data

The stored returns CSV contains a balanced panel of 147 ETFs. The universe was built from a current, manually chosen ETF list with sufficient historical coverage, so it is survivor-conditioned rather than a point-in-time historical ETF universe. The repository does not contain an inactive-fund master, dead-fund total returns, terminal proceeds, stable identifier history, merger and liquidation records, or point-in-time classification flags, so we cannot reconstruct a survivorship-bias-free historical universe from the available data. Static asset-class labels are used throughout the experiment.

## 5. Execution and backtesting

At rebalance date $t$, all forecasts use rows strictly before $t$. Existing holdings earn the close-$(t-1)$-to-close-$t$ return, the target portfolio executes at the close of $t$, and the new holdings first earn the following return. Holdings then drift until the next quarterly rebalance.

Trades are measured from drifted pre-trade weights:

```math
z_t = w_t - w_{t^-}, \qquad
TO_t^{\mathrm{gross}} = \|z_t\|_1, \qquad
TO_t^{\mathrm{oneway}} = \tfrac{1}{2}\|z_t\|_1.
```

Initial formation is reported separately. Linear transaction-cost scenarios of 0, 1, 5, 10, and 25 bp are charged to risky-asset dollar trades through the self-financing NAV equation. These cost levels are scenario assumptions rather than reconstructed historical ETF-specific costs.

## 6. Forecasting models

We compare sample covariance, EWMA, corrected IEWMA, and Ledoit-Wolf linear shrinkage as sequential covariance forecasts. Table 1 reports out-of-sample Gaussian covariance loss, equal-weight volatility bias and RMSE, condition number, effective rank, and downstream GMV volatility.

Ledoit-Wolf had the lowest mean covariance loss at -10.2203 and the lowest equal-weight volatility RMSE at 0.0828. Its median condition number was 3,935.6, compared with 293,767.7 for IEWMA, 889,784.7 for sample covariance, and 1,477,956.9 for EWMA. Corrected IEWMA is used as the main covariance estimator in the portfolio experiments.

## 7. Risk calibration

Under the common ex-ante risk ceiling, the 10% ceiling was binding on 100% of nominal decisions, about 19.4% of box decisions, about 45.2% of box-plus-diagonal decisions, and 0% of ellipsoidal decisions. Average attained model-predicted risks were 10.00%, 5.01%, 7.44%, and 3.05% respectively. The methods therefore operate at materially different attained risks even when they share the same ceiling.

To separate robustness from differences in risk-taking, we also solve

```math
\max_{w \in \mathcal{W}_t}
\left[
R_{\mathrm{rob},t}(w)
-
\frac{\gamma_t}{2} w^\top \Sigma_{\mathrm{decision},t} w
\right],
\qquad
\gamma_t \ge 0,
```

using deterministic bracket expansion and bisection to seek 10% predicted volatility. The attainable interval is bounded below by the constrained GMV solution and above by the zero-$\gamma$ robust optimum.

| Model | Dates attained | Dates target unattainable |
|---|---:|---:|
| Nominal | 31 | 0 |
| Box | 6 | 25 |
| Box + diagonal covariance | 14 | 17 |
| Ellipsoidal | 0 | 31 |

All nonattainment at the 10% target occurred because the zero-$\gamma$ robust optimum was already below the requested risk; negative risk aversion was not used. Only nominal supplied a complete 10% target-attainment path, and its 5-bp results are nearly identical to the nominal common-ceiling path.

Across the complete common-ceiling grids, there is no predicted-risk level shared by all four optimized model families: the all-model overlap lower bound is 6.00%, while the upper bound is 3.05%. Post-hoc realized-risk ranges are also disjoint. Because there is no common attainable risk level across all four methods, we do not report an equal-risk ranking across the full model set.

## 8. Benchmarks and realized performance

At the fixed 5-bp transaction-cost scenario:

| Strategy | Net annualized return | Realized volatility | Zero-RF/provisional Sharpe | Recurring one-way turnover |
|---|---:|---:|---:|---:|
| ETF equal weight | 9.16% | 14.32% | 0.684 | 2.69% |
| Asset-class equal weight | 8.72% | 14.00% | 0.668 | 2.73% |
| Inverse volatility | 5.77% | 9.37% | 0.646 | 7.78% |
| IEWMA GMV | 1.88% | 3.40% | 0.564 | 14.46% |
| Nominal MVO, 10% ceiling | 1.34% | 16.00% | 0.164 | 47.48% |
| Box robust, 10% ceiling | 0.26% | 7.61% | 0.072 | 24.89% |
| Box + diagonal, 10% ceiling | 0.99% | 6.21% | 0.190 | 22.90% |
| Ellipsoidal, 10% ceiling | 2.83% | 3.58% | 0.798 | 13.81% |

The robust portfolios attain different risk levels under the same ceiling, so these point estimates should not be interpreted as an equal-risk comparison.

## 9. Perturbation experiments

To measure input-to-weight sensitivity directly, we selected four dates and gave all four principal optimizers the same 24 shared 21-day circular moving-block training draws. For each draw, expected returns and IEWMA covariances were re-estimated while the baseline uncertainty-set geometry for that date was held fixed, isolating the effect of forecast perturbation on the resulting allocation.

We also applied deterministic mean shocks of

```math
\hat{\mu} \pm s
```

and three covariance shocks: multiplying total variance by 1.10, blending correlation 10% toward the identity matrix, and multiplying the leading eigenvalue by 1.10. Each stressed covariance matrix is projected back to the positive semidefinite cone using the same eigenvalue floor as the main estimator.

Mean bootstrap L1 weight sensitivity fell from 1.156 for nominal MVO to 0.649 for box, 0.552 for box plus diagonal covariance, and 0.174 for ellipsoidal, corresponding to reductions of roughly 44%, 52%, and 85% relative to nominal. Mean cosine similarity rose from 0.455 for nominal to 0.686, 0.767, and 0.965 respectively. Across these perturbations, the robust formulations produced substantially more stable portfolio weights than nominal MVO. None of the 560 direct or clone perturbation solves failed.

## 10. ETF redundancy and synthetic clones

At every rebalance, correlations are estimated using only the preceding training window. Pairwise distance is

```math
d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}.
```

Average-linkage clustering is applied at correlation-equivalent cutoffs of 0.80, 0.90, 0.95, and 0.975. Each cluster medoid minimizes average within-cluster distance, with a lexicographic tie-break. The full-universe calibrated uncertainty radius is retained and the corresponding uncertainty geometry is subset to the selected medoids.

Median IEWMA condition number fell from about 293,768 in the full universe to 9,412, 56,191, 137,145, and 161,400 across the four cutoffs. Average medoid counts were 40.5, 62.9, 82.7, and 97.8 respectively. The reduced universes were better conditioned, but turnover did not consistently decrease: GMV recurring turnover was 14.46% in the full universe versus 21.85%, 24.55%, 19.78%, and 17.53% across the four medoid universes.

Fourteen medoid model-date solves were infeasible under the unchanged 10% volatility ceiling, concentrated at the more aggressive clustering thresholds and under diagonal covariance inflation. These cases remain marked as infeasible, and affected full-path performance rows are incomplete.

The clone experiment adds synthetic near-duplicates of SPY or AGG to the training universe on the same four dates, with relative noise standard deviations of 0%, 1%, and 5%. For the box model, security-weight L1 distortion averaged about 0.025 while asset-class distortion was essentially zero. For box plus diagonal covariance, security-weight distortion averaged about 0.011 and economic-exposure distortion about 0.007. This suggests that much of the apparent instability under near-duplicate securities reflects substitution among similar instruments rather than large changes in economic exposure. The clones are synthetic perturbation devices rather than historical securities.

## 11. Statistical inference

The inference panel contains joint daily 5-bp net returns for the eight benchmark and optimized strategies. We use a stationary bootstrap with 2,000 replications, expected block length 10, seed 36129, and shared resampled time indices across strategies. Intervals are paired 95% percentile intervals. Because the repository does not contain a validated full-period risk-free series, Sharpe ratios are reported under a zero risk-free-rate assumption.

Relative to nominal MVO, estimated Sharpe differences and intervals were:

| Strategy | Delta Sharpe | 95% interval |
|---|---:|---:|
| ETF equal weight | 0.521 | [0.091, 0.955] |
| Asset-class equal weight | 0.504 | [0.075, 0.948] |
| Inverse volatility | 0.482 | [0.017, 0.969] |
| GMV | 0.400 | [-0.216, 1.114] |
| Box | -0.092 | [-0.668, 0.451] |
| Box + diagonal | 0.027 | [-0.552, 0.575] |
| Ellipsoidal | 0.634 | [0.034, 1.204] |

These intervals reflect sampling uncertainty in realized returns but do not remove the differences in attained risk across the robust models. We also compute the Deflated Sharpe Ratio across 27 complete core-experiment strategy specifications at the fixed 5-bp cost level. No DSR probability exceeded 0.762. Because the candidate return series and model specifications are strongly correlated, and because the finite candidate set does not represent every modeling decision, we treat DSR as a secondary diagnostic rather than a primary ranking criterion.

## 12. Regime analysis

At each rebalance, trend is measured by trailing 252-day SPY return. Volatility is the trailing 63-day annualized SPY standard deviation and is classified as high when it exceeds the expanding median of completed prior 63-day estimates. This produced 11 calm risk-on decisions, 14 volatile risk-on decisions, six stress/risk-off decisions, and no weak/cooling decisions. Regime results are descriptive because the individual state samples are small and the historical period is limited.

## 13. Sensitivity analysis

The sensitivity analysis varies calibrated $\rho$ and $\kappa$ using multipliers of 0, 0.5, 1.0, and 1.5, and varies the maximum asset weight across 5%, 10%, and 20%, using the same four evaluation dates. Box-plus-diagonal allocations move materially away from the baseline outside the calibrated parameter combination. Mean L1 change is about 1.09 at

```math
(\rho,\kappa) = (0,0).
```

Changing the weight cap from 10% produces average L1 allocation changes commonly near 0.8 to 1.0. The optimized portfolios are therefore sensitive to concentration limits and robust-radius choices even when the robust formulations are less sensitive to resampled return and covariance inputs. The full transaction-cost grid is reported separately. We did not vary the estimation-window length because doing so would require redesigning the nested-history comparison.

## 14. Limitations

The main limitations are:

- The ETF universe is survivor-conditioned rather than point-in-time.
- Asset classifications are static.
- The evaluation is nested pseudo-out-of-sample rather than a completely untouched holdout.
- The close-$t$ execution convention is an approximation imposed by the available total-return data.
- Cash earns zero because a validated historical risk-free series is not included.
- Transaction costs are linear scenarios rather than ETF- and date-specific historical estimates.
- Direct sensitivity and clone experiments use four selected dates.
- Medoid bootstrap sensitivity was not estimated.
- No weak/cooling regime occurred in the outer sample.
- DSR assumptions are imperfect, and the candidate count cannot represent every modeling choice.

## 15. Conclusion

Robust optimization materially reduced the sensitivity of portfolio weights to perturbations in expected returns and covariance estimates. The largest stability improvement came from the ellipsoidal formulation, while the box and box-plus-diagonal formulations also moved substantially less than nominal MVO. That added stability did not translate into consistent realized-return outperformance: simple diversified portfolios remained competitive, and the robust portfolios often operated at materially lower predicted risk than nominal MVO under the same volatility ceiling.

Redundancy reduction improved covariance conditioning but did not consistently reduce turnover. The synthetic-clone experiments also suggest that some security-level instability reflects substitution among economically similar ETFs rather than large changes in underlying exposure. Overall, in this survivor-conditioned ETF panel, robust optimization made allocations substantially less sensitive to estimation error, but did not consistently improve realized returns.
