# Credit Scoring Business Understanding

This document fulfills Task 1 requirements: it explains the business context and how Basel frameworks influence an interpretable, well-documented modeling approach; it justifies a proxy target; and it compares interpretable vs complex models within regulated environments.

## Basel Accords Summary and Impact

The Basel Accords (I, II, III) are international banking standards from BCBS to strengthen capital adequacy, risk management, and transparency.
- Basel I: Introduced risk-weighted assets (RWA) and 8% capital adequacy ratio.
- Basel II: Added risk-sensitive capital requirements and the three pillars:
  - Pillar 1: Quantitative rules for credit/market/operational risk; IRB allows internal models for PD, LGD, EAD.
  - Pillar 2: Supervisory review of internal processes and controls.
  - Pillar 3: Market discipline via disclosures.
- Basel III: Strengthened capital quality (CET1), buffers, liquidity ratios (LCR, NSFR), leverage ratio, and G-SIB add-ons.

Implications for credit risk:
- Positive: Better measurement, stronger capital/liquidity, improved governance and transparency, reduced systemic risk.
- Negative: Higher compliance costs, complexity, possible credit rationing for SMEs, incentives for regulatory arbitrage and shadow banking growth.

## Why Interpretability and Documentation Matter

Basel II’s IRB approach requires banks to estimate PD, LGD, and EAD with rigorous governance, validation, and disclosures. Regulators expect:
- Transparent modeling decisions and traceable data lineage.
- Stability, monotonicity, and explainability of risk estimates.
- Documented assumptions, performance monitoring, and change controls.

Thus, models should be interpretable enough to support auditability, reasoned overrides, and consistent policy application—especially when decisions affect access to credit.

## Proxy Target Justification and Risks

We lack a direct default label; we create a proxy target via RFM-based disengagement and label high-risk behavior (`is_high_risk` = 1) for least engaged customer clusters. This provides a practical training signal to estimate risk probability.

Business risks of proxy-based modeling:
- Label risk: Proxy may misclassify true repayment behavior; decisions could unfairly deny credit or misprice risk.
- Drift risk: Behavior patterns can change with seasonality or promotions; proxy may degrade.
- Governance risk: Proxies need clear documentation, thresholds, and periodic recalibration.
- Compliance risk: Ensure that features and proxy construction avoid discriminatory bias and align with fair lending laws.

Mitigations:
- Validate proxy with backtests and business review.
- Monitor model with MLflow/CI, retrain periodically.
- Use reject inference and incremental ground-truth collection when available.

## Trade-offs: Interpretable vs Complex Models

- Interpretable (e.g., Logistic Regression + WoE):
  - Pros: Transparency, stable monotonic relationships, easier validation and governance, faster deployment.
  - Cons: May underfit complex patterns, lower raw performance.

- Complex (e.g., Gradient Boosting/Random Forest):
  - Pros: Higher predictive accuracy, handles nonlinear interactions, robust to diverse features.
  - Cons: Harder to explain, requires stronger monitoring, potential overfitting, heavier MLOps.

Practical guidance:
- Start with interpretable baselines to establish governance and disclosures.
- Compare against complex models; prefer interpretable if performance is adequate and regulatory constraints are tight.
- If a complex model is chosen, add explainability tooling (SHAP), monotonic constraints where possible, and thorough documentation.

