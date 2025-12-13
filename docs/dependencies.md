# Dependencies & Environment

This project uses standard Python data tooling and selected ML libraries. The exact versions are managed via `requirements.txt` and `pyproject.toml`.

## Installation (High Level)

- Create and activate a Python virtual environment.
- Install dependencies from `requirements.txt`.
- Optional: install extras for explainability and advanced modeling.

## Typical Libraries (Indicative)

- Data: pandas, numpy
- Modeling: scikit-learn
- Tracking/Serving (optional): mlflow, fastapi, uvicorn
- Visualization: matplotlib, seaborn
- Explainability (optional): shap

Refer to the repository’s `requirements.txt` for the definitive list.

## Notes

- Keep environments isolated per project.
- Prefer pinned versions for reproducibility.
- Use CI to validate installation and tests on push.
