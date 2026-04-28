from __future__ import annotations

import pandas as pd

from .core import run_kda
from .schemas import KDAResult


def run_from_streamlit_selection(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    selected_methods: list[str],
    controls: list[str] | None = None,
    subgroup: str | None = None,
    method_params: dict | None = None,
) -> KDAResult:
    return run_kda(
        data=data,
        y_var=y_var,
        x_vars=x_vars,
        methods=selected_methods,
        controls=controls,
        subgroup=subgroup,
        method_params=method_params,
    )

