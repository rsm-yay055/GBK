from __future__ import annotations

from dataclasses import dataclass


ALL_METHODS = [
    "correlation",
    "regression",
    "drop_one",
    "shapley_lmg",
    "johnson",
    "random_forest",
    "xgboost",
    "shap",
]


@dataclass(frozen=True)
class Applicability:
    applicable: bool
    model_hint: str
    warning: str | None = None


def method_applicability(method: str, y_type: str) -> Applicability:
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown method: {method}")

    if method == "correlation":
        return Applicability(True, "pearson/spearman based on variable types")

    if method == "regression":
        hints = {
            "continuous": "statsmodels OLS",
            "binary": "statsmodels Logit",
            "ordered": "statsmodels OrderedModel",
        }
        return Applicability(True, hints.get(y_type, "regression"))

    if method == "drop_one":
        hints = {
            "continuous": "OLS full vs reduced R2",
            "binary": "Logit full vs reduced McFadden pseudo-R2",
            "ordered": "OrderedModel full vs reduced McFadden pseudo-R2",
        }
        return Applicability(True, hints.get(y_type, "drop-one model comparison"))

    if method in {"shapley_lmg", "johnson"} and y_type != "continuous":
        return Applicability(
            False,
            "not applicable",
            f"{method} is only implemented for continuous outcomes in Python v1.",
        )

    if method == "shapley_lmg":
        return Applicability(True, "custom LMG/Shapley over OLS R2")

    if method == "johnson":
        return Applicability(True, "custom Johnson relative weights")

    if method == "random_forest":
        return Applicability(
            True,
            "RandomForestRegressor" if y_type == "continuous" else "RandomForestClassifier",
        )

    if method == "xgboost":
        return Applicability(
            True,
            "XGBRegressor" if y_type == "continuous" else "XGBClassifier",
        )

    if method == "shap":
        return Applicability(
            True,
            "mean absolute SHAP values after fitted XGBoost model",
        )

    return Applicability(False, "not applicable", f"{method} is not configured.")

