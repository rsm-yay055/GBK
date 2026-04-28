from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "kda_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def driver_bar_chart(ranking_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(ranking_table))))
    plot_df = ranking_table.sort_values("mean_method_index", ascending=True)
    ax.barh(plot_df["driver"], plot_df["mean_method_index"], color="#1f4e79")
    ax.set_xlabel("Mean method index (0-100)")
    ax.set_ylabel("Driver")
    ax.set_title("Key Driver Ranking")
    fig.tight_layout()
    return fig
