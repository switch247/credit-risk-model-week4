"""Insurance-specific modeling configuration (feature sets and actions)."""
from __future__ import annotations

from typing import Mapping, Sequence

NUMERIC_FEATURE_CANDIDATES: Sequence[str] = (
    "TotalPremium",
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "kilowatts",
    "RegistrationYear",
    "CustomValueEstimate",
)

CATEGORICAL_FEATURE_CANDIDATES: Sequence[str] = (
    "Province",
    "PostalCode",
    "VehicleType",
    "CoverType",
    "CoverCategory",
    "Gender",
)

DRIVER_ACTIONS: Mapping[str, str] = {
    "TotalPremium": "Check for underpriced risks; uplift base rate when SHAP signal is high.",
    "SumInsured": "Set minimum deductibles or apply loadings for large exposures.",
    "CalculatedPremiumPerTerm": "Enforce floor rates per term and flag anomalies for review.",
    "kilowatts": "Horsepower surcharge; restrict high-kW vehicles or require telematics.",
    "RegistrationYear": "Age-based pricing: add wear-and-tear factor for older vehicles.",
    "CustomValueEstimate": "Verify valuations; request appraisal or cap payout ratios when high.",
    "Province": "Geo-pricing: adjust base rate or theft coverage by provincial loss trends.",
    "PostalCode": "Micro-geo rules: anti-theft devices or garage requirements in hot spots.",
    "VehicleType": "Eligibility and caps: tighten for high-risk body types (e.g., sports cars, taxis).",
    "CoverType": "Move collision/own-damage to tiered deductibles where SHAP is large.",
    "Gender": "Compliance check: drop from deployment if legally restricted.",
}
