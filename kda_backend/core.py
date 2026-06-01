from __future__ import annotations

from collections.abc import Callable
import warnings as py_warnings

import numpy as np
import pandas as pd

from .applicability import ALL_METHODS, method_applicability
from .methods import METHOD_REGISTRY, normalize_scores, rank_desc, share100_scores
from .plotting import driver_bar_chart
from .preprocessing import complete_cases, detect_var_type
from .schemas import KDAResult, MethodResult


MIN_ROWS = 10
VALID_Y_TYPES = {"continuous", "binary", "ordered", "nominal", "unknown"}
ProgressCallback = Callable[[float, str], None]

METHOD_PROGRESS_LABELS = {
    "correlation": "Correlation",
    "regression": "Regression",
    "drop_one": "Drop-one",
    "shapley_lmg": "Shapley / LMG",
    "johnson": "Johnson Relative Weights",
    "coa": "COA",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "shap": "SHAP",
}
SLOW_PROGRESS_METHODS = {"shapley_lmg", "random_forest", "xgboost", "shap"}


def _method_label(method: str) -> str:
    return METHOD_PROGRESS_LABELS.get(method, method.replace("_", " ").title())


def _emit_progress(
    progress_callback: ProgressCallback | None, progress: float, message: str
) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0.0, min(1.0, float(progress))), message)


def _scale_progress(start: float, end: float, fraction: float) -> float:
    return start + ((end - start) * max(0.0, min(1.0, float(fraction))))


def _minimum_required_rows(x_vars: list[str], controls: list[str]) -> int:
    return max(MIN_ROWS, len(x_vars) + len(controls) + 2)


def _validate_inputs(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None,
    subgroup: str | None,
    y_type_override: str | None = None,
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
    if y_type_override is not None and y_type_override not in VALID_Y_TYPES:
        valid = ", ".join(sorted(VALID_Y_TYPES))
        raise ValueError(f"Invalid y_type_override '{y_type_override}'. Expected one of: {valid}.")


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
    method_intervals: dict[str, tuple[pd.Series, pd.Series]] | None = None,
    method_share_intervals: dict[str, tuple[pd.Series, pd.Series]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_intervals = method_intervals or {}
    method_share_intervals = method_share_intervals or {}
    importance = pd.DataFrame({"driver": x_vars})
    for method in methods:
        scores = method_scores[method].reindex(x_vars)
        importance[method] = scores.to_numpy()
        if method in method_intervals:
            lower, upper = method_intervals[method]
            importance[f"{method}_ci_lower"] = lower.reindex(x_vars).to_numpy()
            importance[f"{method}_ci_upper"] = upper.reindex(x_vars).to_numpy()
        if method in method_share_intervals:
            lower, upper = method_share_intervals[method]
            importance[f"{method}_share_ci_lower"] = lower.reindex(x_vars).to_numpy()
            importance[f"{method}_share_ci_upper"] = upper.reindex(x_vars).to_numpy()
        average100_scores = normalize_scores(scores)
        share_scores = share100_scores(scores)
        importance[f"{method}_average100"] = average100_scores.to_numpy()
        importance[f"{method}_index"] = share_scores.to_numpy()
        importance[f"{method}_share"] = share_scores.to_numpy()
        importance[f"{method}_rank"] = rank_desc(scores).to_numpy()
        warnings = method_warnings.get(method, [])
        importance[f"{method}_warning"] = "; ".join(warnings) if warnings else ""

    index_cols = [f"{method}_index" for method in methods]
    share_cols = [f"{method}_share" for method in methods]
    average100_cols = [f"{method}_average100" for method in methods]
    rank_cols = [f"{method}_rank" for method in methods]
    importance["mean_method_index"] = importance[index_cols].mean(axis=1, skipna=True)
    importance["mean_method_share"] = importance[share_cols].mean(axis=1, skipna=True)
    importance["mean_method_average100"] = importance[average100_cols].mean(
        axis=1, skipna=True
    )
    importance["average_rank"] = importance[rank_cols].mean(axis=1, skipna=True)
    importance["median_rank"] = importance[rank_cols].median(axis=1, skipna=True)
    importance["top3_appearances"] = (importance[rank_cols] <= 3).sum(axis=1)

    ranking = importance[
        [
            "driver",
            "average_rank",
            "median_rank",
            "mean_method_share",
            "mean_method_index",
            "mean_method_average100",
            "top3_appearances",
            *rank_cols,
        ]
    ].copy()
    ranking = ranking.sort_values(
        ["average_rank", "mean_method_share", "driver"],
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


def _bootstrap_intervals(
    model_data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    methods: list[str],
    method_scores: dict[str, pd.Series],
    method_params: dict,
    bootstrap_methods: list[str] | None,
    bootstrap_params: dict | None,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> tuple[
    dict[str, tuple[pd.Series, pd.Series]],
    dict[str, tuple[pd.Series, pd.Series]],
    list[str],
]:
    if not bootstrap_methods:
        return {}, {}, []

    params = {
        "n_resamples": 200,
        "ci": 0.95,
        "random_state": 454,
        "min_rows": MIN_ROWS,
    }
    params.update(bootstrap_params or {})
    n_resamples = int(params["n_resamples"])
    ci = float(params["ci"])
    min_rows = int(params["min_rows"])
    alpha = (1 - ci) / 2
    rng = np.random.default_rng(int(params["random_state"]))

    intervals: dict[str, tuple[pd.Series, pd.Series]] = {}
    share_intervals: dict[str, tuple[pd.Series, pd.Series]] = {}
    warnings: list[str] = []
    selected = set(methods)
    total_bootstrap_methods = len(bootstrap_methods)
    for method_index, method in enumerate(bootstrap_methods, start=1):
        label = _method_label(method)
        method_start = _scale_progress(
            progress_start, progress_end, (method_index - 1) / total_bootstrap_methods
        )
        method_end = _scale_progress(
            progress_start, progress_end, method_index / total_bootstrap_methods
        )
        _emit_progress(
            progress_callback,
            method_start,
            f"Building bootstrap bands for {label} ({method_index}/{total_bootstrap_methods})...",
        )
        if method not in selected:
            warnings.append(f"{method} bootstrap skipped: method was not selected for the main analysis.")
            _emit_progress(progress_callback, method_end, f"Skipped bootstrap bands for {label}.")
            continue

        applicability = method_applicability(method, y_type)
        if not applicability.applicable:
            warnings.append(f"{method} bootstrap skipped: {applicability.warning or 'method is not applicable.'}")
            _emit_progress(progress_callback, method_end, f"Skipped bootstrap bands for {label}.")
            continue

        scores_by_driver = {driver: [] for driver in x_vars}
        share_scores_by_driver = {driver: [] for driver in x_vars}
        update_every = max(1, n_resamples // 20)
        for sample_num in range(1, n_resamples + 1):
            sample_idx = rng.integers(0, len(model_data), size=len(model_data))
            sample = model_data.iloc[sample_idx].reset_index(drop=True)
            sample_complete = complete_cases(sample, y_var, x_vars, controls)
            if len(sample_complete) < min_rows:
                if sample_num % update_every == 0 or sample_num == n_resamples:
                    _emit_progress(
                        progress_callback,
                        _scale_progress(method_start, method_end, sample_num / n_resamples),
                        f"Bootstrap {label}: {sample_num:,}/{n_resamples:,} resamples...",
                    )
                continue
            try:
                with py_warnings.catch_warnings():
                    py_warnings.simplefilter("ignore")
                    result = METHOD_REGISTRY[method](
                        sample_complete,
                        y_var=y_var,
                        x_vars=x_vars,
                        controls=controls,
                        y_type=y_type,
                        params=method_params.get(method, {}),
                    )
            except Exception:
                if sample_num % update_every == 0 or sample_num == n_resamples:
                    _emit_progress(
                        progress_callback,
                        _scale_progress(method_start, method_end, sample_num / n_resamples),
                        f"Bootstrap {label}: {sample_num:,}/{n_resamples:,} resamples...",
                    )
                continue

            sample_scores = result.scores.reindex(x_vars)
            sample_share_scores = share100_scores(sample_scores)
            for driver, value in sample_scores.items():
                if np.isfinite(value):
                    scores_by_driver[driver].append(float(value))
            for driver, value in sample_share_scores.items():
                if np.isfinite(value):
                    share_scores_by_driver[driver].append(float(value))
            if sample_num % update_every == 0 or sample_num == n_resamples:
                _emit_progress(
                    progress_callback,
                    _scale_progress(method_start, method_end, sample_num / n_resamples),
                    f"Bootstrap {label}: {sample_num:,}/{n_resamples:,} resamples...",
                )

        lower = pd.Series(np.nan, index=x_vars, dtype=float)
        upper = pd.Series(np.nan, index=x_vars, dtype=float)
        share_lower = pd.Series(np.nan, index=x_vars, dtype=float)
        share_upper = pd.Series(np.nan, index=x_vars, dtype=float)
        min_valid = int(params.get("min_valid_resamples", max(10, int(n_resamples * 0.2))))
        for driver, values in scores_by_driver.items():
            if len(values) < min_valid:
                continue
            lo = float(np.quantile(values, alpha))
            hi = float(np.quantile(values, 1 - alpha))
            score = method_scores[method].reindex(x_vars).loc[driver]
            if np.isfinite(score):
                lo = min(lo, float(score))
                hi = max(hi, float(score))
            lower.loc[driver] = lo
            upper.loc[driver] = hi
        main_share_scores = share100_scores(method_scores[method].reindex(x_vars))
        for driver, values in share_scores_by_driver.items():
            if len(values) < min_valid:
                continue
            lo = float(np.quantile(values, alpha))
            hi = float(np.quantile(values, 1 - alpha))
            score = main_share_scores.loc[driver]
            if np.isfinite(score):
                lo = min(lo, float(score))
                hi = max(hi, float(score))
            share_lower.loc[driver] = lo
            share_upper.loc[driver] = hi

        if lower.notna().any() and upper.notna().any():
            intervals[method] = (lower, upper)
            if share_lower.notna().any() and share_upper.notna().any():
                share_intervals[method] = (share_lower, share_upper)
        else:
            warnings.append(
                f"{method} bootstrap skipped: insufficient valid bootstrap samples after {n_resamples} resamples."
            )
        _emit_progress(progress_callback, method_end, f"Finished bootstrap bands for {label}.")

    return intervals, share_intervals, warnings


def run_kda(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None = None,
    subgroup: str | None = None,
    method_params: dict | None = None,
    bootstrap_methods: list[str] | None = None,
    bootstrap_params: dict | None = None,
    y_type_override: str | None = None,
    progress_callback: ProgressCallback | None = None,
    _run_subgroups: bool = True,
) -> KDAResult:
    controls = controls or []
    method_params = method_params or {}
    _validate_inputs(data, y_var, x_vars, methods, controls, subgroup, y_type_override)
    _emit_progress(progress_callback, 0.02, "Preparing data and checking selected methods...")

    subgroup_results: dict[str, KDAResult] | None = None
    subgroup_summary: pd.DataFrame | None = None
    warnings: list[str] = []
    if subgroup and _run_subgroups:
        subgroup_results = {}
        subgroup_rows = []
        min_required_rows = _minimum_required_rows(x_vars, controls)
        subgroup_groups = list(data.dropna(subset=[subgroup]).groupby(subgroup, sort=True))
        total_groups = max(1, len(subgroup_groups))
        _emit_progress(
            progress_callback,
            0.04,
            f"Preparing {len(subgroup_groups)} subgroup run{'s' if len(subgroup_groups) != 1 else ''}...",
        )
        for group_index, (level, group) in enumerate(subgroup_groups, start=1):
            group_start = _scale_progress(0.05, 0.40, (group_index - 1) / total_groups)
            group_end = _scale_progress(0.05, 0.40, group_index / total_groups)
            _emit_progress(
                progress_callback,
                group_start,
                f"Running subgroup {level} ({group_index}/{len(subgroup_groups)})...",
            )
            complete = complete_cases(group, y_var, x_vars, controls)
            if len(complete) < min_required_rows:
                reason = f"only {len(complete)} complete rows; at least {min_required_rows} are required."
                warnings.append(f"Skipped subgroup {level}: {reason}")
                subgroup_rows.append(
                    {
                        "subgroup_variable": subgroup,
                        "subgroup_level": str(level),
                        "rows_used": len(complete),
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                _emit_progress(
                    progress_callback,
                    group_end,
                    f"Skipped subgroup {level}: insufficient complete rows.",
                )
                continue
            try:
                def subgroup_progress(progress: float, message: str) -> None:
                    _emit_progress(
                        progress_callback,
                        _scale_progress(group_start, group_end, progress),
                        f"Subgroup {level}: {message}",
                    )

                subgroup_results[str(level)] = run_kda(
                    group,
                    y_var,
                    x_vars,
                    methods,
                    controls=controls,
                    subgroup=None,
                    method_params=method_params,
                    bootstrap_methods=bootstrap_methods,
                    bootstrap_params=bootstrap_params,
                    y_type_override=y_type_override,
                    progress_callback=subgroup_progress,
                    _run_subgroups=False,
                )
            except Exception as exc:
                reason = str(exc)
                warnings.append(f"Skipped subgroup {level}: {reason}")
                subgroup_rows.append(
                    {
                        "subgroup_variable": subgroup,
                        "subgroup_level": str(level),
                        "rows_used": len(complete),
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                _emit_progress(progress_callback, group_end, f"Skipped subgroup {level}: {reason}")
                continue
            subgroup_rows.append(
                {
                    "subgroup_variable": subgroup,
                    "subgroup_level": str(level),
                    "rows_used": len(complete),
                    "status": "included",
                    "reason": "",
                }
            )
            _emit_progress(progress_callback, group_end, f"Finished subgroup {level}.")
        subgroup_summary = pd.DataFrame(
            subgroup_rows,
            columns=["subgroup_variable", "subgroup_level", "rows_used", "status", "reason"],
        )

    has_subgroup_progress = bool(subgroup and _run_subgroups)
    data_progress = 0.42 if has_subgroup_progress else 0.06
    method_start = 0.45 if has_subgroup_progress else 0.10
    method_end = 0.72 if bootstrap_methods else 0.90
    bootstrap_end = 0.95

    _emit_progress(progress_callback, data_progress, "Preparing complete cases for the main analysis...")
    model_data = complete_cases(data, y_var, x_vars, controls)
    min_required_rows = _minimum_required_rows(x_vars, controls)
    if len(model_data) < min_required_rows:
        raise ValueError(
            "Insufficient complete rows for KDA analysis: "
            f"{len(model_data)} rows after dropping missing values; at least {min_required_rows} are required."
        )

    y_type = y_type_override or detect_var_type(model_data[y_var], role="y")
    method_scores: dict[str, pd.Series] = {}
    method_metadata: dict[str, dict] = {}
    method_warnings: dict[str, list[str]] = {}

    total_methods = max(1, len(methods))
    for method_index, method in enumerate(methods, start=1):
        label = _method_label(method)
        step_start = _scale_progress(method_start, method_end, (method_index - 1) / total_methods)
        step_end = _scale_progress(method_start, method_end, method_index / total_methods)
        slow_note = " This can take longer." if method in SLOW_PROGRESS_METHODS else ""
        _emit_progress(
            progress_callback,
            step_start,
            f"Running {label} ({method_index}/{len(methods)})...{slow_note}",
        )
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
        _emit_progress(progress_callback, step_end, f"Finished {label}.")

    method_intervals, method_share_intervals, bootstrap_warnings = _bootstrap_intervals(
        model_data=model_data,
        y_var=y_var,
        x_vars=x_vars,
        controls=controls,
        y_type=y_type,
        methods=methods,
        method_scores=method_scores,
        method_params=method_params,
        bootstrap_methods=bootstrap_methods,
        bootstrap_params=bootstrap_params,
        progress_callback=progress_callback,
        progress_start=method_end,
        progress_end=bootstrap_end,
    )
    warnings.extend(bootstrap_warnings)

    _emit_progress(progress_callback, 0.97, "Assembling rankings, diagnostics, and charts...")
    importance, ranking = _assemble_tables(
        x_vars,
        methods,
        method_scores,
        method_warnings,
        method_intervals,
        method_share_intervals,
    )
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
    _emit_progress(progress_callback, 1.0, "Analysis complete.")
    return KDAResult(
        importance_table=importance,
        ranking_table=ranking,
        diagnostics=diagnostics,
        bar_chart=chart,
        subgroup_results=subgroup_results,
        subgroup_summary=subgroup_summary,
        method_metadata=method_metadata,
        warnings=warnings,
    )
