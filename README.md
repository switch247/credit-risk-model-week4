# Credit Risk Probability Model — Week 4

Build, deploy, and automate a credit risk probability model using alternative data (Xente transactions). This README stays project-specific. Reusable guides live in `docs/`.

## Project Overview
- Goal: Estimate risk probability per customer; derive a credit score and inform loan amount/duration.
- Proxy Target: RFM-based disengagement to label `is_high_risk` in absence of explicit defaults.
- Key Components: Data processing pipeline, proxy engineering, model training and tracking, API serving, CI.

## Quickstart (Windows PowerShell)
> Activate the project virtual environment for all commands: .\.venv\Scripts\Activate.ps1
```powershell
# create/activate venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# reproduce data pipeline (if DVC stages are configured)
dvc repro

# run unit tests
pytest -q
```

## Data & Features (project-specific)
- Source: Xente Challenge transactions (see links in `experiments/todo.md`).
- Entity: `CustomerId` with transaction-level features.
- Engineered: RFM metrics, aggregates (sum, mean, std, counts), temporal extracts (hour, day, month, year), encodings for categoricals.
- Target: `is_high_risk` derived via RFM clustering (see docs below).

## Documentation
- Business Understanding (Task 1): [docs/business_understanding.md](docs/business_understanding.md)
- Reusable Guides:
  - Folder structure and conventions: [docs/README.md](docs/README.md)
  - Notebooks workflow: [docs/notebooks.md](docs/notebooks.md)
  - Dependencies overview: [docs/dependencies.md](docs/dependencies.md)

Inline previews (for platforms that support HTML rendering):
<details>
<summary>View Folder Structure</summary>
<iframe src="docs/README.md#folder-structure" title="Folder Structure" style="width:100%;height:360px;border:1px solid #ddd;"></iframe>
</details>

<details>
<summary>View How to Run (High Level)</summary>
<iframe src="docs/README.md#how-to-run-high-level" title="How to Run" style="width:100%;height:260px;border:1px solid #ddd;"></iframe>
</details>

## Deliverables
- Interim: EDA notebook with 3–5 insights; summary report.
- Final: Medium-style report including proxy reasoning, RFM clustering, model comparison, API demo, limitations, and screenshots (MLflow, CI, Docker).

## Tasks (project-specific)
- Task 1 — Business Understanding: See [docs/business_understanding.md](docs/business_understanding.md).
- Task 2 — EDA: Explore data distributions, missingness, correlations; capture insights in `notebooks/`.
- Task 3 — Feature Engineering: Aggregates, temporal extracts, encoding, scaling/standardization, WoE/IV when appropriate.
- Task 4 — Proxy Target: RFM clustering (K-Means, scaled features, fixed `random_state`); assign `is_high_risk` and merge.
- Task 5 — Model Training & Tracking: Split data, train baseline (Logistic+WoE) and advanced (GB/RandomForest), track with MLflow, register best.
- Task 6 — Deployment & CI: FastAPI service loading MLflow model; Dockerized; GitHub Actions for lint+tests.

## Outputs
- models, metrics, predictions under `outputs/`.
- reports: interim and final under `reports/`.

## Notes
- Prefer interpretable baselines for governance; add explainability to complex models (SHAP) if chosen.
- Monitor proxy validity; document thresholds and refresh cadence.