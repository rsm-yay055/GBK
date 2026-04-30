from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MethodResult:
    scores: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class KDAResult:
    importance_table: pd.DataFrame
    ranking_table: pd.DataFrame
    diagnostics: pd.DataFrame
    bar_chart: Any
    subgroup_results: dict[str, "KDAResult"] | None = None
    method_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

