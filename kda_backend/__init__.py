from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "kda_matplotlib"))

from .applicability import ALL_METHODS
from .core import run_kda
from .schemas import KDAResult, MethodResult

__all__ = ["ALL_METHODS", "KDAResult", "MethodResult", "run_kda"]
