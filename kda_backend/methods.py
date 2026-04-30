from __future__ import annotations

import itertools
import math
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.miscmodels.ordinal_model import OrderedModel
from xgboost import XGBClassifier, XGBRegressor

from .preprocessing import (
    aggregate_encoded_scores,
    encode_outcome,
    encode_predictors,
)
from .schemas import MethodResult


RANDOM_STATE = 454


def normalize_scores(values: pd.Series) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.empty:
        return out
    min_v = finite.min()
    max_v = finite.max()
    if math.isclose(float(min_v), float(max_v)):
        out.loc[finite.index] = 100.0 / len(finite)
    else:
        out.loc[finite.index] = (finite - min_v) / (max_v - min_v) * 100.0
    return out


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[1] == 0:
        return 0.0
    model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    return float(model.rsquared)


def _logit_pseudo_r2(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[1] == 0:
        return 0.0
    try:
        model = sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(disp=False, maxiter=200)
        null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=False, maxiter=200)
        return float(1 - model.llf / null.llf)
    except Exception:
        return float("nan")


def _ordered_pseudo_r2(y: np.ndarray, x: pd.DataFrame) -> float:
    if x.shape[1] == 0:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HessianInversionWarning)
            model = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False, maxiter=200)
            null_x = pd.DataFrame({"intercept_like": np.zeros(len(y))}, index=x.index)
            null = OrderedModel(y, null_x, distr="logit").fit(method="bfgs", disp=False, maxiter=200)
        return float(1 - model.llf / null.llf)
    except Exception:
        return float("nan")


def compute_lmg(x_df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    predictors = list(x_df.columns)
    p = len(predictors)

    @lru_cache(maxsize=None)
    def subset_r2(subset: tuple[int, ...]) -> float:
        if len(subset) == 0:
            return 0.0
        cols = [predictors[i] for i in subset]
        return _ols_r2(y, x_df[cols].to_numpy(dtype=float))

    lmg = {name: 0.0 for name in predictors}
    factorial_p = math.factorial(p)
    for j, name in enumerate(predictors):
        others = [i for i in range(p) if i != j]
        for k in range(p):
            weight = math.factorial(k) * math.factorial(p - k - 1) / factorial_p
            for subset in itertools.combinations(others, k):
                subset = tuple(sorted(subset))
                with_j = tuple(sorted(subset + (j,)))
                lmg[name] += weight * (subset_r2(with_j) - subset_r2(subset))
    return pd.Series(lmg, dtype=float)


def compute_johnson(x_df: pd.DataFrame, y: np.ndarray) -> pd.Series:
    x = StandardScaler().fit_transform(x_df.to_numpy(dtype=float))
    y_std = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    rxx = np.corrcoef(x, rowvar=False)
    rxy = np.corrcoef(np.column_stack([x, y_std]), rowvar=False)[:-1, -1]
    eigvals, eigvecs = np.linalg.eigh(rxx)
    eigvals = np.clip(eigvals, 0, None)
    delta = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    beta = np.linalg.pinv(delta) @ rxy
    raw_weights = (delta**2) @ (beta**2)
    return pd.Series(raw_weights, index=x_df.columns, dtype=float)


def correlation(data: pd.DataFrame, y_var: str, x_vars: list[str], y_type: str, **_) -> MethodResult:
    y = data[y_var]
    scores = {}
    methods = {}
    for x_var in x_vars:
        x = data[x_var]
        joined = pd.concat([y, x], axis=1).dropna()
        if joined.empty:
            scores[x_var] = np.nan
            methods[x_var] = "not enough data"
            continue
        if y_type == "continuous" and pd.api.types.is_numeric_dtype(x):
            method = "pearson"
        else:
            method = "spearman"
        scores[x_var] = abs(joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method))
        methods[x_var] = method
    return MethodResult(pd.Series(scores, dtype=float), {"correlation_methods": methods})


def regression(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    **_,
) -> MethodResult:
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, y_meta = encode_outcome(data[y_var], y_type)
    warnings: list[str] = []

    if y_type == "continuous":
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(x_encoded),
            index=x_encoded.index,
            columns=x_encoded.columns,
        )
        model = sm.OLS(y, sm.add_constant(x_scaled, has_constant="add")).fit()
        coeffs = pd.Series(model.params[1:], index=x_encoded.columns)
        scores = aggregate_encoded_scores(coeffs.abs(), x_vars, mapping)
        metadata = {"model_type": "OLS", "r2": float(model.rsquared)}
    elif y_type == "binary":
        x_scaled = pd.DataFrame(
            StandardScaler().fit_transform(x_encoded),
            index=x_encoded.index,
            columns=x_encoded.columns,
        )
        try:
            model = sm.Logit(y, sm.add_constant(x_scaled, has_constant="add")).fit(disp=False, maxiter=200)
            coeffs = pd.Series(model.params[1:], index=x_encoded.columns)
            scores = aggregate_encoded_scores(coeffs.abs(), x_vars, mapping)
            metadata = {"model_type": "Logit", "classes": y_meta.get("classes")}
        except Exception as exc:
            warnings.append(f"regression Logit failed; used absolute point-biserial fallback: {exc}")
            scores = correlation(data, y_var, x_vars, y_type).scores
            metadata = {"model_type": "Logit fallback"}
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", HessianInversionWarning)
                model = OrderedModel(y, x_encoded, distr="logit").fit(method="bfgs", disp=False, maxiter=200)
            coeffs = pd.Series(model.params[: x_encoded.shape[1]], index=x_encoded.columns)
            scores = aggregate_encoded_scores(coeffs.abs(), x_vars, mapping)
            metadata = {"model_type": "OrderedModel", "classes": y_meta.get("classes")}
        except Exception as exc:
            warnings.append(f"regression OrderedModel failed; used Spearman fallback: {exc}")
            scores = correlation(data, y_var, x_vars, y_type).scores
            metadata = {"model_type": "OrderedModel fallback"}

    return MethodResult(scores.reindex(x_vars), metadata, warnings)


def drop_one(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    **_,
) -> MethodResult:
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], y_type)

    if y_type == "continuous":
        score_fn = lambda frame: _ols_r2(y, frame.to_numpy(dtype=float))
    elif y_type == "binary":
        score_fn = lambda frame: _logit_pseudo_r2(y, frame.to_numpy(dtype=float))
    else:
        score_fn = lambda frame: _ordered_pseudo_r2(y, frame)

    full_score = score_fn(x_encoded)
    scores = {}
    warnings: list[str] = []
    for var in x_vars:
        reduced_cols = [col for col in x_encoded.columns if col not in mapping.get(var, [var])]
        reduced_score = score_fn(x_encoded[reduced_cols]) if reduced_cols else 0.0
        value = full_score - reduced_score
        if not np.isfinite(value):
            warnings.append(f"drop_one could not compute a stable score for {var}.")
            value = np.nan
        scores[var] = value
    return MethodResult(pd.Series(scores, dtype=float), {"full_score": full_score}, warnings)


def shapley_lmg(data: pd.DataFrame, y_var: str, x_vars: list[str], controls: list[str], **_) -> MethodResult:
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], "continuous")
    encoded_scores = compute_lmg(x_encoded, y)
    return MethodResult(aggregate_encoded_scores(encoded_scores, x_vars, mapping), {"model_type": "LMG"})


def johnson(data: pd.DataFrame, y_var: str, x_vars: list[str], controls: list[str], **_) -> MethodResult:
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], "continuous")
    encoded_scores = compute_johnson(x_encoded, y)
    return MethodResult(aggregate_encoded_scores(encoded_scores, x_vars, mapping), {"model_type": "Johnson"})


def _tree_model(method: str, y_type: str, params: dict):
    if y_type == "continuous":
        if method == "random_forest":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 300),
                max_features=params.get("max_features", "sqrt"),
                random_state=params.get("random_state", RANDOM_STATE),
                n_jobs=params.get("n_jobs", 1),
            )
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params.get("n_estimators", 150),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            min_child_weight=params.get("min_child_weight", 3),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            random_state=params.get("random_state", RANDOM_STATE),
            n_jobs=params.get("n_jobs", 1),
        )

    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_features=params.get("max_features", "sqrt"),
            random_state=params.get("random_state", RANDOM_STATE),
            n_jobs=params.get("n_jobs", 1),
        )

    return XGBClassifier(
        n_estimators=params.get("n_estimators", 150),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 3),
        min_child_weight=params.get("min_child_weight", 3),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        random_state=params.get("random_state", RANDOM_STATE),
        n_jobs=params.get("n_jobs", 1),
        eval_metric="logloss" if y_type == "binary" else "mlogloss",
    )


def random_forest(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    params: dict | None = None,
    **_,
) -> MethodResult:
    params = params or {}
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], y_type)
    model = _tree_model("random_forest", y_type, params)
    model.fit(x_encoded, y)
    scoring = "r2" if y_type == "continuous" else "accuracy"
    perm = permutation_importance(
        model,
        x_encoded,
        y,
        n_repeats=params.get("n_repeats", 8),
        random_state=params.get("random_state", RANDOM_STATE),
        n_jobs=params.get("n_jobs", 1),
        scoring=scoring,
    )
    encoded_scores = pd.Series(perm.importances_mean, index=x_encoded.columns)
    scores = aggregate_encoded_scores(encoded_scores, x_vars, mapping)
    return MethodResult(scores, {"model_type": type(model).__name__, "train_score": float(model.score(x_encoded, y))})


def xgboost(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    params: dict | None = None,
    **_,
) -> MethodResult:
    params = params or {}
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], y_type)
    model = _tree_model("xgboost", y_type, params)
    model.fit(x_encoded, y)
    gain = model.get_booster().get_score(importance_type="gain")
    encoded_scores = pd.Series({col: gain.get(col, 0.0) for col in x_encoded.columns}, dtype=float)
    scores = aggregate_encoded_scores(encoded_scores, x_vars, mapping)
    return MethodResult(scores, {"model_type": type(model).__name__, "train_score": float(model.score(x_encoded, y))})


def shap_values(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    params: dict | None = None,
    **_,
) -> MethodResult:
    params = params or {}
    all_predictors = [*x_vars, *controls]
    x_encoded, mapping = encode_predictors(data, all_predictors)
    y, _ = encode_outcome(data[y_var], y_type)
    model = _tree_model("xgboost", y_type, params)
    model.fit(x_encoded, y)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_encoded)
    if isinstance(values, list):
        arr = np.mean([np.abs(v) for v in values], axis=0)
    else:
        arr = np.abs(values)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    encoded_scores = pd.Series(arr.mean(axis=0), index=x_encoded.columns)
    scores = aggregate_encoded_scores(encoded_scores, x_vars, mapping)
    return MethodResult(scores, {"model_type": "TreeExplainer"})


METHOD_REGISTRY = {
    "correlation": correlation,
    "regression": regression,
    "drop_one": drop_one,
    "shapley_lmg": shapley_lmg,
    "johnson": johnson,
    "random_forest": random_forest,
    "xgboost": xgboost,
    "shap": shap_values,
}
