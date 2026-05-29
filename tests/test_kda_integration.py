import unittest
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kda_backend import ALL_METHODS, run_kda
from GBK_app import (
    DEFAULT_BOOTSTRAP_METHODS,
    DEFAULT_METHODS,
    HEAVY_BOOTSTRAP_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    _driver_axis_sort,
    _importance_export_table,
    build_driver_interval_chart,
    build_interactive_chart_data,
    _client_style_shapley_table,
    build_interactive_driver_chart,
    prepare_model_data,
    read_uploaded_dataset,
    run_analysis,
)


class KDAFrontendIntegrationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

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

    def test_subgroup_summary_includes_skipped_levels(self):
        rng = np.random.default_rng(460)
        n = 55
        group = np.array(["A"] * 25 + ["B"] * 25 + ["Tiny"] * 5)
        df = pd.DataFrame(
            {
                "satisfaction": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
                "brand": group,
            }
        )
        df["satisfaction"] = 2 * df["trust"] + rng.normal(scale=0.2, size=n)

        result = run_kda(
            df,
            y_var="satisfaction",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            subgroup="brand",
        )

        summary = result.subgroup_summary
        self.assertIsNotNone(summary)
        self.assertEqual(set(summary["subgroup_level"]), {"A", "B", "Tiny"})
        tiny = summary.loc[summary["subgroup_level"] == "Tiny"].iloc[0]
        self.assertEqual(tiny["status"], "skipped")
        self.assertIn("only 5 complete rows", tiny["reason"])

    def test_subgroup_with_too_few_rows_for_number_of_predictors_is_skipped(self):
        rng = np.random.default_rng(468)
        x_vars = [f"x{i}" for i in range(12)]
        n = 52
        df = pd.DataFrame(rng.normal(size=(n, len(x_vars))), columns=x_vars)
        df["y"] = 1.2 * df["x0"] + rng.normal(scale=0.2, size=n)
        df["Gender"] = ["A"] * 20 + ["B"] * 20 + ["C"] * 12

        result = run_kda(
            df,
            y_var="y",
            x_vars=x_vars,
            methods=["correlation", "regression"],
            subgroup="Gender",
        )

        self.assertEqual(set(result.subgroup_results), {"A", "B"})
        skipped = result.subgroup_summary.loc[result.subgroup_summary["subgroup_level"] == "C"].iloc[0]
        self.assertEqual(skipped["status"], "skipped")
        self.assertIn("at least 14", skipped["reason"])

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
        self.assertFalse(result["include_bootstrap"])
        self.assertEqual(result["driver_scores"].index[0], "trust")
        self.assertEqual(len(result["driver_scores"]), 3)
        self.assertEqual(set(result["export_table"]["driver"]), {"trust", "value", "style"})
        self.assertIn("correlation", result["export_table"].columns)
        self.assertIn("regression", result["export_table"].columns)
        self.assertNotIn("correlation_ci_lower", result["export_table"].columns)
        self.assertIn("kda_result", result)

    def test_run_analysis_only_bootstraps_when_requested(self):
        rng = np.random.default_rng(466)
        n = 60
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

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=["correlation", "regression"],
            include_bootstrap=True,
        )

        self.assertTrue(result["include_bootstrap"])
        self.assertEqual(result["bootstrap_methods"], ["correlation", "regression"])
        self.assertIn("correlation_ci_lower", result["export_table"].columns)
        self.assertIn("regression_ci_lower", result["export_table"].columns)

    def test_run_analysis_does_not_bootstrap_heavy_methods_by_default(self):
        rng = np.random.default_rng(469)
        n = 72
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

        methods = ["correlation", "regression", "random_forest", "xgboost", "shap"]
        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=methods,
            include_bootstrap=True,
        )

        self.assertTrue(result["include_bootstrap"])
        self.assertEqual(result["bootstrap_methods"], ["correlation", "regression"])
        for method in ["correlation", "regression"]:
            self.assertIn(f"{method}_ci_lower", result["export_table"].columns)
            self.assertIn(f"{method}_ci_upper", result["export_table"].columns)
        for method in HEAVY_BOOTSTRAP_METHODS:
            self.assertIn(method, result["export_table"].columns)
            self.assertNotIn(f"{method}_ci_lower", result["export_table"].columns)
            self.assertNotIn(f"{method}_ci_upper", result["export_table"].columns)

    def test_prepare_model_data_excludes_brand_lookup_columns_from_subgroup_candidates(self):
        df = pd.DataFrame(
            {
                "record": [1, 2, 3, 4],
                "brand": ["A", "A", "B", "B"],
                "top_brand_name": ["A", "B", "A", "B"],
                "ownership_brand_name": ["A", "None", "B", "None"],
                "Gender": [1, 2, 1, 2],
                "Age Range": [1, 2, 3, 4],
                "n2_consideration": [1, 0, 1, 0],
                "Has the best designs in the category": [1, 0, 1, 0],
                "Is a brand I trust": [1, 1, 0, 0],
            }
        )

        _, _, meta = prepare_model_data(df)

        self.assertIn("brand", meta["subgroup_candidates"])
        self.assertIn("Gender", meta["subgroup_candidates"])
        self.assertIn("Age Range", meta["subgroup_candidates"])
        self.assertNotIn("top_brand_name", meta["subgroup_candidates"])
        self.assertNotIn("ownership_brand_name", meta["subgroup_candidates"])
        self.assertIn("brand", meta["control_candidates"])
        self.assertIn("Gender", meta["control_candidates"])
        self.assertIn("Age Range", meta["control_candidates"])
        self.assertIn("top_brand_name", meta["control_candidates"])
        self.assertIn("ownership_brand_name", meta["control_candidates"])
        self.assertEqual(meta["outcome_candidates"], ["n2_consideration"])
        self.assertEqual(
            set(meta["driver_candidates"]),
            {"Has the best designs in the category", "Is a brand I trust"},
        )

    def test_upload_reader_prefers_consideration_long_sheet(self):
        workbook = BytesIO()
        summary = pd.DataFrame({"summary_value": [1, 2]})
        consideration_long = pd.DataFrame(
            {
                "record": [1, 1],
                "model": ["brands", "brands"],
                "brand_code": [1, 2],
                "consideration": [5, 4],
                "statement_1": [4, 3],
            }
        )
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="client_key_drivers", index=False)
            consideration_long.to_excel(writer, sheet_name="consideration_long", index=False)
        workbook.seek(0)

        loaded = read_uploaded_dataset(workbook)

        self.assertEqual(list(loaded.columns), list(consideration_long.columns))
        self.assertEqual(len(loaded), 2)

    def test_upload_reader_respects_explicit_sheet_selection(self):
        workbook = BytesIO()
        first = pd.DataFrame({"wrong": [1]})
        second = pd.DataFrame({"right": [2, 3]})
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            first.to_excel(writer, sheet_name="first", index=False)
            second.to_excel(writer, sheet_name="source", index=False)
        workbook.seek(0)

        loaded = read_uploaded_dataset(workbook, sheet_name="source")

        self.assertEqual(list(loaded.columns), ["right"])
        self.assertEqual(len(loaded), 2)

    def test_generic_dataset_without_zagg_names_can_run(self):
        rng = np.random.default_rng(467)
        n = 72
        df = pd.DataFrame(
            {
                "renewal_score": rng.normal(size=n),
                "service_speed": rng.normal(size=n),
                "price_fairness": rng.normal(size=n),
                "support_quality": rng.normal(size=n),
                "customer_type": np.where(np.arange(n) < n / 2, "Consumer", "Business"),
            }
        )
        df["renewal_score"] = (
            1.2 * df["service_speed"]
            + 0.7 * df["support_quality"]
            + rng.normal(scale=0.25, size=n)
        )

        df_raw, df_num, meta = prepare_model_data(df)
        result = run_analysis(
            df_num,
            df_raw,
            target="renewal_score",
            x_vars=["service_speed", "price_fairness", "support_quality"],
            sg_var="customer_type",
            methods=["correlation", "regression"],
        )

        self.assertEqual(meta["outcome_candidates"], [])
        self.assertIn("customer_type", meta["subgroup_candidates"])
        self.assertEqual(result["mode"], "subgroup")
        self.assertEqual({item["group"] for item in result["results"]}, {"Business", "Consumer"})
        self.assertEqual(set(result["subgroup_export_table"]["driver"]), {
            "service_speed",
            "price_fairness",
            "support_quality",
        })

    def test_subgroup_run_analysis_returns_combined_export_table(self):
        rng = np.random.default_rng(456)
        n = 64
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
            sg_var="brand",
            methods=["correlation", "regression"],
        )

        self.assertEqual(result["mode"], "subgroup")
        self.assertEqual({item["group"] for item in result["results"]}, {"A", "B"})
        for item in result["results"]:
            self.assertEqual(len(item["driver_scores"]), 3)
            self.assertEqual(set(item["export_table"]["driver"]), {"trust", "value", "style"})
        self.assertEqual(set(result["subgroup_export_table"]["subgroup_level"]), {"A", "B"})
        self.assertEqual(set(result["subgroup_export_table"]["driver"]), {"trust", "value", "style"})
        self.assertIn("subgroup_variable", result["subgroup_export_table"].columns)
        self.assertIn("correlation", result["subgroup_export_table"].columns)

    def test_run_analysis_reports_skipped_subgroup_levels(self):
        rng = np.random.default_rng(461)
        n = 55
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
        df_raw["brand"] = ["A"] * 25 + ["B"] * 25 + ["Tiny"] * 5

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var="brand",
            methods=["correlation", "regression"],
        )

        tiny = [item for item in result["results"] if item["group"] == "Tiny"][0]
        self.assertTrue(tiny["skipped"])
        self.assertIn("only 5 complete rows", tiny["reason"])

    def test_bootstrap_correlation_adds_confidence_interval_columns(self):
        rng = np.random.default_rng(457)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["correlation"],
            bootstrap_params={"n_resamples": 40, "random_state": 454},
        )

        table = result.importance_table.set_index("driver")
        self.assertIn("correlation_ci_lower", table.columns)
        self.assertIn("correlation_ci_upper", table.columns)
        for driver in ["trust", "value", "style"]:
            score = table.loc[driver, "correlation"]
            lower = table.loc[driver, "correlation_ci_lower"]
            upper = table.loc[driver, "correlation_ci_upper"]
            self.assertLessEqual(lower, upper)
            self.assertLessEqual(lower, score)
            self.assertLessEqual(score, upper)

    def test_bootstrap_selected_methods_add_confidence_interval_columns(self):
        rng = np.random.default_rng(462)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression", "random_forest"],
            bootstrap_methods=["correlation", "regression", "random_forest"],
            bootstrap_params={"n_resamples": 20, "random_state": 454, "min_valid_resamples": 8},
            method_params={"random_forest": {"n_estimators": 10, "n_repeats": 2}},
        )

        for method in ["correlation", "regression", "random_forest"]:
            self.assertIn(f"{method}_ci_lower", result.importance_table.columns)
            self.assertIn(f"{method}_ci_upper", result.importance_table.columns)

    def test_bootstrap_johnson_skips_non_continuous_outcomes_with_warning(self):
        rng = np.random.default_rng(458)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.integers(0, 2, size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["johnson"],
            bootstrap_params={"n_resamples": 20, "random_state": 454},
        )

        self.assertNotIn("johnson_ci_lower", result.importance_table.columns)
        self.assertTrue(any("johnson bootstrap skipped" in warning for warning in result.warnings))

    def test_driver_interval_chart_uses_ci_columns_when_available(self):
        rng = np.random.default_rng(459)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)
        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["correlation"],
            bootstrap_params={"n_resamples": 40, "random_state": 454},
        )

        fig = build_driver_interval_chart(result.importance_table, ["correlation"])

        self.assertTrue(hasattr(fig, "savefig"))
        self.assertGreaterEqual(len(fig.axes[0].collections), 1)

    def test_interactive_chart_data_uses_sum100_scores_and_ci(self):
        rng = np.random.default_rng(463)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)
        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            bootstrap_methods=["correlation", "regression"],
            bootstrap_params={"n_resamples": 20, "random_state": 454, "min_valid_resamples": 8},
        )

        chart_df = build_interactive_chart_data(result.importance_table, ["correlation", "regression"])

        self.assertEqual(set(chart_df["method"]), {"Correlation", "Regression"})
        for _, group in chart_df.groupby("method"):
            self.assertAlmostEqual(group["score"].sum(), 100.0, places=6)
        self.assertTrue(chart_df["ci_lower"].notna().any())
        self.assertTrue(chart_df["ci_upper"].notna().any())
        self.assertLessEqual(chart_df["score"].max(), 100)

    def test_run_kda_adds_sum100_share_columns_for_each_method(self):
        rng = np.random.default_rng(475)
        n = 80
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 1.4 * df["trust"] + 0.5 * df["value"] + rng.normal(scale=0.25, size=n)

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression", "shapley_lmg"],
        )

        table = result.importance_table
        for method in ["correlation", "regression", "shapley_lmg"]:
            share_col = f"{method}_share"
            index_col = f"{method}_index"
            average100_col = f"{method}_average100"
            self.assertIn(share_col, table.columns)
            self.assertIn(index_col, table.columns)
            self.assertIn(average100_col, table.columns)
            self.assertAlmostEqual(table[share_col].sum(), 100.0, places=8)
            self.assertAlmostEqual(table[index_col].sum(), 100.0, places=8)
            self.assertAlmostEqual(table[average100_col].mean(), 100.0, places=8)
        self.assertIn("mean_method_share", table.columns)
        self.assertIn("mean_method_index", table.columns)
        self.assertIn("mean_method_average100", table.columns)
        self.assertAlmostEqual(table["mean_method_share"].sum(), 100.0, places=8)
        self.assertAlmostEqual(table["mean_method_index"].sum(), 100.0, places=8)
        self.assertAlmostEqual(table["mean_method_average100"].mean(), 100.0, places=8)

        export_table = _importance_export_table(result)
        for method in ["correlation", "regression", "shapley_lmg"]:
            self.assertIn(f"{method}_sum100", export_table.columns)
            self.assertIn(f"{method}_average100", export_table.columns)
            pd.testing.assert_series_equal(
                export_table[f"{method}_sum100"],
                export_table[f"{method}_share"],
                check_names=False,
            )
            pd.testing.assert_series_equal(
                export_table[f"{method}_index"],
                export_table[f"{method}_share"],
                check_names=False,
            )

    def test_interactive_chart_data_uses_sum100_share_scores(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["top_driver", "middle_driver", "bottom_driver"],
                "mean_method_index": [50.0, 33.3333333333, 16.6666666667],
                "mean_method_share": [50.0, 33.3333333333, 16.6666666667],
                "correlation": [0.9, 0.6, 0.3],
                "correlation_index": [50.0, 33.3333333333, 16.6666666667],
                "correlation_share": [50.0, 33.3333333333, 16.6666666667],
                "correlation_average100": [150.0, 100.0, 50.0],
            }
        )

        chart_df = build_interactive_chart_data(importance_table, ["correlation"])

        self.assertAlmostEqual(chart_df["score"].sum(), 100.0, places=8)
        self.assertEqual(
            chart_df.sort_values("driver_order")["score"].round(1).tolist(),
            [50.0, 33.3, 16.7],
        )
        self.assertEqual(
            chart_df.sort_values("driver_order")["legacy_index"].round(1).tolist(),
            [150.0, 100.0, 50.0],
        )

    def test_interactive_chart_uses_bounded_data_driven_x_axis_window(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["trust", "value", "service"],
                "mean_method_share": [42.8571428571, 32.1428571429, 25.0],
                "correlation": [0.24, 0.18, 0.14],
                "correlation_share": [42.8571428571, 32.1428571429, 25.0],
                "correlation_ci_lower": [0.21, 0.16, 0.12],
                "correlation_ci_upper": [0.27, 0.20, 0.16],
            }
        )

        chart = build_interactive_driver_chart(importance_table, ["correlation"])
        spec = chart.to_dict()
        x_domain = spec["layer"][0]["encoding"]["x"]["scale"]["domain"]
        unbounded_scale_params = [
            param
            for param in spec.get("params", [])
            if param.get("bind") == "scales"
        ]

        self.assertEqual(x_domain, [0.0, 55.0])
        self.assertFalse(unbounded_scale_params)

    def test_interactive_chart_axis_order_puts_top_driver_first(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["top_driver", "middle_driver", "bottom_driver"],
                "mean_method_index": [150.0, 100.0, 50.0],
                "correlation": [0.9, 0.6, 0.3],
                "correlation_index": [150.0, 100.0, 50.0],
            }
        )

        chart_df = build_interactive_chart_data(importance_table, ["correlation"])

        self.assertEqual(
            _driver_axis_sort(chart_df),
            ["Top Driver", "Middle Driver", "Bottom Driver"],
        )

    def test_interactive_chart_uses_gbk_dark_background(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["trust", "value", "service"],
                "mean_method_index": [150.0, 100.0, 50.0],
                "correlation": [0.9, 0.6, 0.3],
                "correlation_index": [150.0, 100.0, 50.0],
            }
        )

        chart = build_interactive_driver_chart(importance_table, ["correlation"])
        spec = chart.to_dict()

        self.assertEqual(spec["background"], "#334651")
        self.assertEqual(spec["config"]["view"]["fill"], "#334651")
        self.assertEqual(spec["config"]["axis"]["labelColor"], "#C7D8E4")
        self.assertEqual(spec["config"]["axis"]["gridColor"], "#5E7486")
        self.assertNotIn(spec["background"].lower(), {"#000", "#000000", "black"})
        self.assertNotIn(spec["background"], set(METHOD_COLORS.values()))

    def test_interactive_chart_hides_average100_tooltip(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["trust", "value", "service"],
                "mean_method_index": [50.0, 33.3333333333, 16.6666666667],
                "correlation": [0.9, 0.6, 0.3],
                "correlation_index": [50.0, 33.3333333333, 16.6666666667],
                "correlation_average100": [150.0, 100.0, 50.0],
                "correlation_ci_lower": [0.81, 0.54, 0.27],
                "correlation_ci_upper": [0.99, 0.648, 0.324],
            }
        )

        chart = build_interactive_driver_chart(importance_table, ["correlation"])
        spec_text = str(chart.to_dict())

        self.assertIn("Sum-to-100 index", spec_text)
        self.assertIn("Uncertainty interval", spec_text)
        self.assertNotIn("Average-100 index", spec_text)
        self.assertNotIn("Lower uncertainty band", spec_text)
        self.assertNotIn("Upper uncertainty band", spec_text)

    def test_default_methods_are_correlation_and_regression(self):
        self.assertEqual(DEFAULT_METHODS, ("correlation", "regression"))

    def test_default_bootstrap_methods_exclude_tree_and_shap_methods(self):
        self.assertEqual(DEFAULT_BOOTSTRAP_METHODS, ("correlation", "regression", "shapley_lmg", "johnson", "coa"))
        self.assertEqual(HEAVY_BOOTSTRAP_METHODS, ("random_forest", "xgboost", "shap"))
        self.assertTrue(set(DEFAULT_BOOTSTRAP_METHODS).isdisjoint(HEAVY_BOOTSTRAP_METHODS))

    def test_each_method_has_distinct_visible_chart_color(self):
        colors = list(METHOD_COLORS.values())

        self.assertEqual(len(colors), len(set(colors)))
        self.assertNotIn("#FFFFFF", {color.upper() for color in colors})

    def test_coa_is_available_in_method_ui_metadata(self):
        self.assertIn("coa", ALL_METHODS)
        self.assertLess(ALL_METHODS.index("coa"), ALL_METHODS.index("random_forest"))
        self.assertEqual(METHOD_LABELS["coa"], "COA")
        self.assertIn("coa", METHOD_COLORS)
        self.assertNotIn(METHOD_COLORS["coa"].upper(), {"#FFFFFF", "#000000"})

    def test_binary_regression_uses_gbk_standardized_ols_coefficients(self):
        rng = np.random.default_rng(464)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        logits = 1.4 * x1 - 0.4 * x2
        y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = run_kda(df, "y", ["x1", "x2"], ["regression", "shapley_lmg", "johnson"])

        self.assertFalse(result.importance_table["shapley_lmg"].isna().all())
        self.assertFalse(result.importance_table["johnson"].isna().all())
        x_scaled = (df[["x1", "x2"]] - df[["x1", "x2"]].mean()) / df[["x1", "x2"]].std(ddof=0)
        y_scaled = (df["y"] - df["y"].mean()) / df["y"].std(ddof=0)
        expected = np.linalg.lstsq(
            np.column_stack([np.ones(len(df)), x_scaled.to_numpy()]),
            y_scaled.to_numpy(),
            rcond=None,
        )[0][1:]
        observed = result.importance_table.set_index("driver").loc[["x1", "x2"], "regression"].to_numpy()
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)

    def test_lmg_and_johnson_scores_are_share_scaled(self):
        rng = np.random.default_rng(465)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        y = 2 * x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        result = run_kda(df, "y", ["x1", "x2", "x3"], ["shapley_lmg", "johnson"])

        self.assertAlmostEqual(result.importance_table["shapley_lmg"].sum(), 1.0, places=10)
        self.assertAlmostEqual(result.importance_table["johnson"].sum(), 1.0, places=10)

    def test_coa_scores_are_share_scaled_and_rank_true_driver_first(self):
        rng = np.random.default_rng(473)
        n = 90
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        y = 1.8 * x1 + 0.4 * x2 + rng.normal(scale=0.25, size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        result = run_kda(df, "y", ["x1", "x2", "x3"], ["coa"])

        table = result.importance_table.set_index("driver")
        self.assertAlmostEqual(table["coa"].sum(), 1.0, places=10)
        self.assertEqual(table["coa"].idxmax(), "x1")
        self.assertEqual(result.method_metadata["coa"]["model_type"], "COA")

    def test_run_analysis_bootstraps_coa_as_lightweight_method(self):
        rng = np.random.default_rng(474)
        n = 70
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.4 * df_num["trust"] + rng.normal(scale=0.3, size=n)
        df_raw = df_num.copy()

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=["coa"],
            include_bootstrap=True,
            bootstrap_resamples=20,
        )

        self.assertEqual(result["bootstrap_methods"], ["coa"])
        self.assertIn("coa_ci_lower", result["export_table"].columns)
        self.assertIn("coa_ci_upper", result["export_table"].columns)

    def test_run_kda_can_force_likert_outcome_to_continuous(self):
        rng = np.random.default_rng(470)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = pd.cut(1.5 * x1 + 0.3 * x2, bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df = pd.DataFrame({"consideration": y, "statement_1": x1, "statement_2": x2})

        result = run_kda(
            df,
            "consideration",
            ["statement_1", "statement_2"],
            ["shapley_lmg"],
            y_type_override="continuous",
        )

        self.assertEqual(
            result.diagnostics.loc[result.diagnostics["metric"] == "y_type", "value"].iloc[0],
            "continuous",
        )
        self.assertFalse(result.importance_table["shapley_lmg"].isna().all())

    def test_shapley_lmg_can_hold_controls_always_in_model(self):
        rng = np.random.default_rng(471)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        brand_b = np.array([0, 1] * (n // 2))
        y = 1.5 * x1 + 0.4 * x2 + 2.0 * brand_b + rng.normal(scale=0.2, size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "brand_b": brand_b})

        plain = run_kda(df, "y", ["x1", "x2"], ["shapley_lmg"])
        controlled = run_kda(
            df,
            "y",
            ["x1", "x2"],
            ["shapley_lmg"],
            controls=["brand_b"],
            method_params={"shapley_lmg": {"always_controls": True}},
        )

        self.assertAlmostEqual(controlled.importance_table["shapley_lmg"].sum(), 1.0, places=10)
        self.assertNotEqual(
            plain.importance_table["shapley_lmg"].round(8).tolist(),
            controlled.importance_table["shapley_lmg"].round(8).tolist(),
        )

    def test_run_analysis_passes_generic_control_variables_to_backend(self):
        rng = np.random.default_rng(472)
        n = 80
        segment = np.array(["A", "B"] * (n // 2))
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1.3 * x1 + 0.2 * x2 + np.where(segment == "B", 2.0, 0.0) + rng.normal(scale=0.2, size=n)
        df_raw = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "segment": segment})
        df_num = df_raw[["y", "x1", "x2"]].copy()

        result = run_analysis(
            df_num,
            df_raw,
            target="y",
            x_vars=["x1", "x2"],
            sg_var=None,
            methods=["shapley_lmg"],
            control_vars=["segment"],
        )

        self.assertEqual(result["controls"], ["segment"])
        self.assertEqual(
            result["kda_result"].method_metadata["shapley_lmg"]["control_mode"],
            "always",
        )
        self.assertIn("segment", result["kda_result"].diagnostics["value"].tolist())

    def test_amazon_autos_tool_output_matches_client_sum100_and_average100_when_using_lmg(self):
        project_dir = Path(__file__).resolve().parents[2]
        data_path = project_dir / "Amazon_autos_dataset" / "amazon_autos_outputs" / "consideration_long.csv"
        if not data_path.exists():
            self.skipTest("Amazon Autos prepared data is not available in this checkout.")
        df = pd.read_csv(data_path)
        df_raw, df_num, _ = prepare_model_data(df)
        x_vars = [f"statement_{i}" for i in range(1, 8)]

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=x_vars,
            sg_var="model",
            methods=["shapley_lmg"],
            control_vars=["brand"],
        )

        self.assertEqual(result["mode"], "subgroup")
        self.assertEqual(result["controls"], ["brand"])
        sum100_by_group = {}
        index_by_group = {}
        average100_by_group = {}
        for item in result["results"]:
            if item["skipped"]:
                continue
            self.assertEqual(item["export_table"]["driver"].tolist(), x_vars)
            table = item["export_table"].set_index("driver").loc[x_vars]
            sum100_by_group[item["group"]] = table["shapley_lmg_sum100"].round(1).tolist()
            index_by_group[item["group"]] = table["shapley_lmg_index"].round(1).tolist()
            average100_by_group[item["group"]] = table["shapley_lmg_average100"].round(1).tolist()

        self.assertEqual(
            sum100_by_group["brands"],
            [16.8, 16.1, 12.7, 16.7, 12.4, 11.7, 13.6],
        )
        self.assertEqual(
            sum100_by_group["dealerships"],
            [20.5, 13.5, 19.9, 11.8, 13.5, 12.2, 8.6],
        )
        self.assertEqual(
            index_by_group["brands"],
            [16.8, 16.1, 12.7, 16.7, 12.4, 11.7, 13.6],
        )
        self.assertEqual(
            index_by_group["dealerships"],
            [20.5, 13.5, 19.9, 11.8, 13.5, 12.2, 8.6],
        )
        self.assertEqual(
            average100_by_group["brands"],
            [117.5, 113.0, 88.6, 117.0, 87.1, 81.7, 95.0],
        )
        self.assertEqual(
            average100_by_group["dealerships"],
            [143.7, 94.3, 139.2, 82.8, 94.6, 85.4, 60.0],
        )

        client_style = _client_style_shapley_table(result["subgroup_export_table"])
        self.assertIsNotNone(client_style)
        self.assertEqual(client_style["driver"].tolist(), x_vars)
        self.assertEqual(client_style["brands_sum100"].round(1).tolist(), [16.8, 16.1, 12.7, 16.7, 12.4, 11.7, 13.6])
        self.assertEqual(client_style["dealerships_sum100"].round(1).tolist(), [20.5, 13.5, 19.9, 11.8, 13.5, 12.2, 8.6])
        self.assertEqual(client_style["brands_average100"].round(1).tolist(), [117.5, 113.0, 88.6, 117.0, 87.1, 81.7, 95.0])
        self.assertEqual(client_style["dealerships_average100"].round(1).tolist(), [143.7, 94.3, 139.2, 82.8, 94.6, 85.4, 60.0])


if __name__ == "__main__":
    unittest.main()
