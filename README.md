# GBK Marketing Insights Suite

A Streamlit application for running key driver analysis on survey data. The
tool is designed for consultants and insight teams who need a clear, repeatable
way to identify which survey measures are most strongly associated with an
outcome such as consideration, preference, trust, or satisfaction.

Live app: https://gbkkeydrivers.streamlit.app/

## Overview

GBK Marketing Insights Suite turns a clean Excel workbook into a driver ranking
workflow. Users upload a model-ready survey dataset, select an outcome variable,
choose candidate drivers, optionally compare subgroups, and export detailed
method-level scores for QA, appendices, and client delivery.

The application combines a consultant-friendly Streamlit interface with a
backend key driver analysis engine that supports traditional statistical
methods and machine-learning based importance checks.

## Core Capabilities

- Excel upload with worksheet selection for `.xlsx` workbooks.
- Guided analysis setup for outcome, driver, control, and subgroup variables.
- Automatic data preparation for numeric modeling, missing values, constant
  columns, and obvious ID/date/meta fields.
- Multiple key driver methods in a single run.
- Bootstrap confidence intervals for supported methods.
- Subgroup analysis that reruns the same setup within each segment level.
- Interactive ranking charts with indexed scores where 100 is the average
  shown driver.
- Plain-language result guidance for consultant review.
- Detailed score tables and downloadable CSV exports.

## Analysis Methods

The backend supports nine driver analysis methods:

| Method | Purpose |
| --- | --- |
| Correlation | Simple first read of one-to-one association with the outcome. |
| Regression | Standardized coefficient comparison while controlling for other drivers. |
| Drop-one | Measures how much model fit weakens when each driver is removed. |
| Shapley / LMG | Allocates model explanatory power across overlapping predictors. |
| Johnson Relative Weights | Estimates relative contribution when predictors are correlated. |
| COA | Fast association-based driver read using share-scaled squared one-way association. |
| Random Forest | Tree-based importance check for nonlinear relationships. |
| XGBoost | Gradient-boosted model importance for predictive driver review. |
| SHAP | Mean absolute SHAP values from an XGBoost model for advanced importance ranking. |

## Recommended Workflow

1. Upload a clean Excel workbook with one row per respondent and one column per
   survey question, metric, or segment.
2. Select the worksheet to analyze.
3. Choose the outcome variable.
4. Select candidate driver variables and optional controls.
5. Choose whether to run one overall model or repeat the model by subgroup.
6. Select one or more analysis methods.
7. Run the analysis, review the ranking chart, and export detailed scores.

## Outputs

The app produces:

- Ranked driver chart with method-level indexed scores.
- Plain-language readout of lead and supporting drivers.
- Detailed score export for QA and appendix review.
- Final ranking summary.
- Technical diagnostics.
- Optional subgroup status and subgroup-level score tables.
- CSV download for combined subgroup results.

## Data Expectations

Use a clean, model-ready `.xlsx` file. The app expects one record per respondent
and does not reshape raw survey exports. Outcome and driver columns should be
numeric or convertible to numeric modeling inputs. Segment variables can be used
for subgroup analysis when each level has enough usable rows.

Uploaded data is processed by the running Streamlit app session. For sensitive
or restricted datasets, use only an approved deployment environment.

## Local Development

This project targets Python 3.12.

Recommended setup with `uv`:

```bash
uv sync
uv run streamlit run GBK_app.py
```

Fallback setup with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run GBK_app.py
```

## Testing

Run the integration test suite:

```bash
python -m unittest discover -s tests -v
```

If you are using `uv`:

```bash
uv run python -m unittest discover -s tests -v
```

## Deployment

The app is intended to run on Streamlit Community Cloud or any Python hosting
environment that can install the packages in `requirements.txt`.

For Streamlit Community Cloud:

1. Connect this GitHub repository.
2. Set the app entry point to `GBK_app.py`.
3. Use Python 3.12.
4. Deploy from the `main` branch.

## Project Structure

```text
.
|-- GBK_app.py                  # Streamlit frontend and dashboard workflow
|-- kda_backend/                # Key driver analysis backend
|   |-- applicability.py        # Method availability and model hints
|   |-- core.py                 # Main KDA orchestration
|   |-- methods.py              # Statistical and ML importance methods
|   |-- plotting.py             # Plotting helpers
|   |-- preprocessing.py        # Data preparation utilities
|   |-- schemas.py              # Result dataclasses and shared schemas
|   `-- streamlit_adapter.py    # Streamlit-facing backend adapter
|-- tests/                      # Integration tests
|-- requirements.txt            # Runtime dependencies
|-- pyproject.toml              # Project metadata and uv configuration
`-- uv.lock                     # Locked dependency graph for uv
```

## Tech Stack

- Streamlit
- pandas and NumPy
- statsmodels
- scikit-learn
- XGBoost
- SHAP
- Matplotlib and Altair
