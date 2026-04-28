import unittest

import numpy as np
import pandas as pd

from kda_backend import run_kda
from GBK_app import run_analysis


class KDAFrontendIntegrationTests(unittest.TestCase):
    def test_frontend_repo_can_call_backend_and_get_streamlit_outputs(self):
        rng = np.random.default_rng(454)
        n = 60
        df = pd.DataFrame(
            {
                "satisfaction": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
                "brand": np.where(np.arange(n) < n / 2, "A", "B"),
            }
        )
        df["satisfaction"] = 2 * df["trust"] + 0.3 * df["value"] + rng.normal(scale=0.2, size=n)

        result = run_kda(
            df,
            y_var="satisfaction",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            subgroup="brand",
        )

        self.assertEqual(result.ranking_table.iloc[0]["driver"], "trust")
        self.assertEqual(set(result.subgroup_results), {"A", "B"})
        self.assertTrue(hasattr(result.bar_chart, "savefig"))

    def test_streamlit_run_analysis_uses_backend_multi_method_contract(self):
        rng = np.random.default_rng(455)
        n = 50
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.5 * df_num["trust"] + rng.normal(scale=0.2, size=n)
        df_raw = df_num.copy()
        df_raw["brand"] = np.where(np.arange(n) < n / 2, "A", "B")

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=["correlation", "regression"],
        )

        self.assertEqual(result["mode"], "single")
        self.assertEqual(result["top5"].index[0], "trust")
        self.assertIn("kda_result", result)


if __name__ == "__main__":
    unittest.main()
