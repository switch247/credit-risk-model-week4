# Interim Report — Week 3 (Tasks 1 & 2)

## Scope
- Focus: Task 1 (EDA & Stats) and Task 2 (Data versioning with DVC) on the 2014–2015 South African auto insurance portfolio.
- Goals: understand loss ratios and risk variation (province/zipcode/vehicle/gender/time), surface data quality issues, and version raw/processed assets for reproducibility.

## Data Snapshot
- Source: historical policy, vehicle, client, plan, and claims records (Feb 2014–Aug 2015).
- Expected schema highlights: IDs (`UnderwrittenCoverID`, `PolicyID`), time (`TransactionMonth`), client (`Gender`, `MaritalStatus`, `Citizenship`, `IsVATRegistered`), location (`Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`), vehicle (`VehicleType`, `Make`, `Model`, `RegistrationYear`, `Kilowatts`, `Bodytype`), plan (`SumInsured`, `CalculatedPremiumPerTerm`, `CoverType`, `CoverGroup`, `Product`), payments/claims (`TotalPremium`, `TotalClaims`).
- Data volumes and row counts: _[to be filled after ingestion run]_.

## Data Readiness Checks (Task 1)
- Dtypes: confirm dates parsed (`TransactionMonth`), categoricals encoded, numeric fields (`TotalPremium`, `TotalClaims`, `Kilowatts`, `SumInsured`) as floats/ints.
- Missingness: summarize % missing per column; flag high-missing categorical buckets for potential “Other/Unknown” grouping.
- Duplicates: check duplicate `PolicyID`/`UnderwrittenCoverID` rows post-sorting by `TransactionMonth`.
- Outlier flags: boxplot/IQR for `TotalClaims`, `TotalPremium`, `CustomValueEstimate`, `SumInsured`; consider winsorization caps for modeling, but preserve raw for audit.

## EDA Highlights (preliminary)
- Portfolio loss ratio: `TotalClaims / TotalPremium` overall and by `Province`, `VehicleType`, `Gender`. _[Insert computed values after run]._
- Geography: provinces/zipcodes with elevated loss ratio and premium growth/decline. _[Insert top/bottom 5 summary]._
- Vehicle: makes/models and body types with highest/lowest severity; check `Kilowatts` vs `TotalClaims` correlation.
- Gender & marital status: compare loss ratios; note sample sizes to avoid over-interpretation.
- Temporal: month-over-month trends in premium volume and claim severity over 18 months; look for seasonality.
- Correlations: Pearson/Spearman between `TotalPremium`, `TotalClaims`, `SumInsured`, `CalculatedPremiumPerTerm`, `Kilowatts`, `RegistrationYear`.

### Visual Placeholders (attach generated plots)
- ![Loss ratio by province placeholder](../figures/loss_ratio_by_province.png)
- ![Premium vs claims by zipcode placeholder](../figures/premium_claims_zipcode.png)
- ![Outlier boxplots placeholder](../figures/outliers_boxplot.png)

## Hypothesis Testing Plan (Task 1)
- H1: No risk differences across provinces — test loss ratio by province (ANOVA/Kruskal depending on normality), post-hoc pairwise with p-value correction.
- H2: No risk differences between zipcodes — focus on top-N volume zipcodes; apply non-parametric tests and effect sizes.
- H3: No significant margin (profit) difference between zipcodes — compute margin = `TotalPremium - TotalClaims`; compare distributions.
- H4: No significant risk difference between women and men — two-sample test on loss ratio or margin; report CI and effect size.
- Assumptions: check normality (Shapiro), variance (Levene), sample-size thresholds; prefer bootstrap CIs when distributional assumptions fail.

## DVC Setup (Task 2)
- Initialized DVC in repo; added default remote `localstorage` at `.dvc/storage`.
- Tracked raw data: `dvc add data/raw/insurance.csv`; committed `.dvc` + `.gitignore` entries.
- Planned next: add processed artifacts (`data/processed/insurance_clean.csv`) and model outputs as pipeline stages in `dvc.yaml` once preprocessing is finalized.

## Risks / Gaps
- Data availability: confirm access to raw file and encoding (UTF-8 vs others) to avoid parsing issues.
- Class imbalance: severity modeling may be impacted by many zero-claim records; consider zero-inflated approach or two-stage (freq × severity).
- High-cardinality zipcodes: may need target/WOE encoding; avoid naive one-hot blowup.
- Potential leakage: ensure plan features set post-underwriting do not leak target information.

## Next Steps
- Finalize ingestion and dtype fixes; persist `data/processed/insurance_clean.csv`.
- Generate EDA plots and replace placeholders with actual paths.
- Run hypothesis tests per plan and summarize p-values/effect sizes.
- Extend DVC pipeline to processed data and EDA outputs; push artifacts to remote.
