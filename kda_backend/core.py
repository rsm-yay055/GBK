from __future__ import annotations

import numpy as np
import pandas as pd

from .applicability import ALL_METHODS, method_applicability
from .methods import METHOD_REGISTRY, normalize_scores, rank_desc
from .plotting import driver_bar_chart
from .preprocessing import complete_cases, detect_var_type
from .schemas import KDAResult, MethodResult


MIN_ROWS = 10


def _validate_inputs(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None,
    subgroup: str | None,
) -> None:
    controls = controls or []
    needed = [y_var, *x_vars, *controls]
    if subgroup:
        needed.append(subgroup)
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise ValueError(f"Variables not found in data: {', '.join(missing)}")

    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")

    if len(set(x_vars)) != len(x_vars):
        raise ValueError("x_vars contains duplicate variables.")
    overlap = sorted(set(x_vars).intersection(controls))
    if overlap:
        raise ValueError(f"Variables cannot be both drivers and controls: {', '.join(overlap)}")


def _nan_method(method: str, x_vars: list[str], warning: str) -> MethodResult:
    return MethodResult(
        pd.Series(np.nan, index=x_vars, dtype=float),
        {"applicable": False},
        [warning],
    )


def _assemble_tables(
    x_vars: list[str],
    methods: list[str],
    method_scores: dict[str, pd.Series],
    method_warnings: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    importance = pd.DataFrame({"driver": x_vars})
    for method in methods:
        scores = method_scores[method].reindex(x_vars)
        importance[method] = scores.to_numpy()
        importance[f"{method}_index"] = normalize_scores(scores).to_numpy()
        importance[f"{method}_rank"] = rank_desc(scores).to_numpy()
        warnings = method_warnings.get(method, [])
        importance[f"{method}_warning"] = "; ".join(warnings) if warnings else ""

    index_cols = [f"{method}_index" for method in methods]
    rank_cols = [f"{method}_rank" for method in methods]
    importance["mean_method_index"] = importance[index_cols].mean(axis=1, skipna=True)
    importance["average_rank"] = importance[rank_cols].mean(axis=1, skipna=True)
    importance["median_rank"] = importance[rank_cols].median(axis=1, skipna=True)
    importance["top3_appearances"] = (importance[rank_cols] <= 3).sum(axis=1)

    ranking = importance[
        [
            "driver",
            "average_rank",
            "median_rank",
            "mean_method_index",
            "top3_appearances",
            *rank_cols,
        ]
    ].copy()
    ranking = ranking.sort_values(
        ["average_rank", "mean_method_index", "driver"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "overall_rank", np.arange(1, len(ranking) + 1))
    return importance, ranking


def _diagnostics(
    *,
    rows_input: int,
    rows_used: int,
    y_var: str,
    y_type: str,
    x_vars: list[str],
    controls: list[str],
    methods: list[str],
    subgroup: str | None,
    method_metadata: dict[str, dict],
) -> pd.DataFrame:
    rows = [
        ("rows_input", rows_input),
        ("rows_used", rows_used),
        ("rows_dropped_missing", rows_input - rows_used),
        ("y_var", y_var),
        ("y_type", y_type),
        ("x_vars", ", ".join(x_vars)),
        ("controls", ", ".join(controls)),
        ("methods", ", ".join(methods)),
        ("subgroup", subgroup or ""),
    ]
    for method, metadata in method_metadata.items():
        if "model_type" in metadata:
            rows.append((f"{method}_model_type", metadata["model_type"]))
        if "train_score" in metadata:
            rows.append((f"{method}_train_score", metadata["train_score"]))
    return pd.DataFrame(rows, columns=["metric", "value"])


def run_kda(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None = None,
    subgroup: str | None = None,
    method_params: dict | None = None,
    _run_subgroups: bool = True,
) -> KDAResult:
    controls = controls or []
    method_params = method_params or {}
    _validate_inputs(data, y_var, x_vars, methods, controls, subgroup)

    subgroup_results: dict[str, KDAResult] | None = None
    warnings: list[str] = []
    if subgroup and _run_subgroups:
        subgroup_results = {}
        for level, group in data.dropna(subset=[subgroup]).groupby(subgroup, sort=True):
            complete = complete_cases(group, y_var, x_vars, controls)
            if len(complete) < MIN_ROWS:
                warnings.append(f"Skipped subgroup {level}: only {len(complete)} complete rows.")
                continue
            subgroup_results[str(level)] = run_kda(
                group,
                y_var,
                x_vars,
                methods,
                controls=controls,
                subgroup=None,
                method_params=method_params,
                _run_subgroups=False,
            )

    model_data = complete_cases(data, y_var, x_vars, controls)
    if len(model_data) < max(MIN_ROWS, len(x_vars) + len(controls) + 2):
        raise ValueError(
            "Insufficient complete rows for KDA analysis: "
            f"{len(model_data)} rows after dropping missing values."
        )

    y_type = detect_var_type(model_data[y_var], role="y")
    method_scores: dict[str, pd.Series] = {}
    method_metadata: dict[str, dict] = {}
    method_warnings: dict[str, list[str]] = {}

    for method in methods:
        applicability = method_applicability(method, y_type)
        if not applicability.applicable:
            result = _nan_method(method, x_vars, applicability.warning or f"{method} is not applicable.")
        else:
            params = method_params.get(method, {})
            try:
                result = METHOD_REGISTRY[method](
                    model_data,
                    y_var=y_var,
                    x_vars=x_vars,
                    controls=controls,
                    y_type=y_type,
                    params=params,
                )
            except Exception as exc:
                result = _nan_method(method, x_vars, f"{method} failed: {exc}")

        metadata = {"applicable": applicability.applicable, "model_hint": applicability.model_hint}
        metadata.update(result.metadata)
        method_scores[method] = result.scores.reindex(x_vars)
        method_metadata[method] = metadata
        method_warnings[method] = result.warnings
        warnings.extend([f"{method}: {warning}" for warning in result.warnings])

    importance, ranking = _assemble_tables(x_vars, methods, method_scores, method_warnings)
    diagnostics = _diagnostics(
        rows_input=len(data),
        rows_used=len(model_data),
        y_var=y_var,
        y_type=y_type,
        x_vars=x_vars,
        controls=controls,
        methods=methods,
        subgroup=subgroup,
        method_metadata=method_metadata,
    )
    chart = driver_bar_chart(ranking)
    return KDAResult(
        importance_table=importance,
        ranking_table=ranking,
        diagnostics=diagnostics,
        bar_chart=chart,
        subgroup_results=subgroup_results,
        method_metadata=method_metadata,
        warnings=warnings,
    )

