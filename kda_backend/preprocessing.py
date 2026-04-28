from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def detect_var_type(series: pd.Series, *, role: str = "x") -> str:
    non_missing = series.dropna()
    if non_missing.empty:
        return "unknown"

    dtype = non_missing.dtype
    if isinstance(dtype, CategoricalDtype) and dtype.ordered:
        return "ordered"

    unique_count = non_missing.nunique()
    if unique_count == 2:
        return "binary"

    if pd.api.types.is_bool_dtype(non_missing):
        return "binary"

    if pd.api.types.is_numeric_dtype(non_missing):
        values = non_missing.to_numpy(dtype=float)
        is_integer_like = np.all(np.isclose(values, np.round(values)))
        if role == "y" and is_integer_like and 2 < unique_count <= 7:
            return "ordered"
        return "continuous"

    if isinstance(dtype, CategoricalDtype):
        return "ordered" if dtype.ordered else "nominal"

    return "nominal"


def complete_cases(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str] | None = None,
) -> pd.DataFrame:
    controls = controls or []
    cols = [y_var, *x_vars, *controls]
    return data.loc[:, cols].dropna().copy()


def encode_predictors(data: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frames: list[pd.DataFrame] = []
    mapping: dict[str, list[str]] = {}

    for var in variables:
        series = data[var]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            frame = pd.DataFrame({var: pd.to_numeric(series, errors="coerce").astype(float)})
        else:
            frame = pd.get_dummies(series.astype("category"), prefix=var, drop_first=False, dtype=float)
        frames.append(frame)
        mapping[var] = list(frame.columns)

    if not frames:
        return pd.DataFrame(index=data.index), mapping

    encoded = pd.concat(frames, axis=1)
    return encoded.astype(float), mapping


def encode_outcome(series: pd.Series, y_type: str) -> tuple[np.ndarray, dict[str, object]]:
    if y_type == "continuous":
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float), {}

    if isinstance(series.dtype, CategoricalDtype):
        categories = list(series.cat.categories)
        codes = series.cat.codes.to_numpy()
    else:
        categories = sorted(series.dropna().unique().tolist())
        cat = pd.Categorical(series, categories=categories, ordered=y_type == "ordered")
        codes = cat.codes

    if y_type == "binary":
        return codes.astype(int), {"classes": categories}

    if y_type == "ordered":
        return codes.astype(int), {"classes": categories}

    return codes.astype(int), {"classes": categories}


def aggregate_encoded_scores(
    encoded_scores: pd.Series,
    variables: list[str],
    mapping: dict[str, list[str]],
) -> pd.Series:
    out = {}
    for var in variables:
        cols = mapping.get(var, [var])
        existing = [col for col in cols if col in encoded_scores.index]
        out[var] = float(encoded_scores.loc[existing].abs().sum()) if existing else np.nan
    return pd.Series(out, dtype=float)

