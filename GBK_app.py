import re as _re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="GBK Marketing Insights Suite", layout="wide")

for k, v in {
    "uploaded_df_raw": None,
    "uploaded_df_num": None,
    "uploaded_meta": None,
    "uploaded_filename": None,
    "analysis_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #2d3748 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem !important; max-width: 100% !important; }
.gbk-hero { background: #1a202c; padding: 2rem 2rem 1.75rem; border-radius: 10px; margin-bottom: 1.5rem; }
.gbk-eyebrow { font-size: 11px; color: #E8503A; letter-spacing: 3px; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem; }
.gbk-hero h1 { font-size: 36px; font-weight: 900; color: white; line-height: 1.05; text-transform: uppercase; letter-spacing: -1px; margin: 0 0 0.5rem; }
.gbk-hero p { font-size: 13px; color: rgba(255,255,255,0.45); line-height: 1.7; margin: 0; }
.gbk-label { font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 2.5px; font-weight: 700; margin-bottom: 0.4rem; }
.gbk-panel { background: #1a202c; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.gbk-panel-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px; color: rgba(255,255,255,0.3); margin-bottom: 0.75rem; }
.gbk-note { font-size: 13px; color: rgba(255,255,255,0.55); line-height: 1.7; }
.gbk-stat { font-size: 30px; font-weight: 800; color: white; line-height: 1; }
.gbk-card { background: #1a202c; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); padding: 1.25rem 1.5rem; }
.gbk-card-kicker { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px; color: rgba(255,255,255,0.3); margin-bottom: 0.6rem; }
.gbk-card-text { font-size: 18px; font-weight: 800; color: white; line-height: 1.2; }
.gbk-bar-wrap { margin-bottom: 10px; }
.gbk-bar-label { font-size: 12px; color: rgba(255,255,255,0.65); margin-bottom: 3px; }
.gbk-bar-row { display: flex; align-items: center; gap: 10px; }
.gbk-bar-track { flex: 1; background: rgba(255,255,255,0.07); border-radius: 4px; height: 8px; overflow: hidden; }
.gbk-bar-fill { height: 100%; border-radius: 4px; }
.gbk-bar-val { font-size: 11px; color: rgba(255,255,255,0.3); width: 48px; text-align: right; }
.gbk-disclaimer { font-size: 11px; color: rgba(255,255,255,0.2); margin-top: 0.75rem; font-style: italic; }
.gbk-insight { background: rgba(255,255,255,0.03); border-radius: 6px; padding: 0.75rem 1rem; border-left: 3px solid rgba(255,255,255,0.12); font-size: 13px; color: rgba(255,255,255,0.55); line-height: 1.6; margin-bottom: 8px; }
.gbk-insight b { color: white; }
.gbk-insight-red { border-left-color: #E8503A; }
.gbk-insight-blue { border-left-color: #7a9db8; }
.gbk-step-item { display: flex; gap: 10px; align-items: flex-start; font-size: 13px; color: rgba(255,255,255,0.55); line-height: 1.6; margin-bottom: 10px; }
.gbk-step-num { background: #E8503A; color: white; font-size: 10px; font-weight: 700; min-width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-top: 2px; flex-shrink: 0; }
.gbk-step-item b { color: rgba(255,255,255,0.85); }
.gbk-summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.gbk-summary-key { font-size: 10px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px; }
.gbk-summary-val { font-size: 13px; color: white; font-weight: 600; }
.gbk-tag { display: inline-block; background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.7); font-size: 12px; padding: 2px 9px; border-radius: 4px; margin: 2px; }
.gbk-method-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; margin-top: 8px; }
.gbk-method-title { font-size: 13px; color: white; font-weight: 700; margin-bottom: 4px; }
.gbk-method-desc { font-size: 12px; color: rgba(255,255,255,0.5); line-height: 1.55; }
.gbk-shap-badge { display: inline-block; background: rgba(232,80,58,0.15); border: 1px solid rgba(232,80,58,0.35); color: #E8503A; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; padding: 1px 7px; border-radius: 4px; margin-left: 8px; vertical-align: middle; }
.gbk-input-warning { font-size: 11px; color: #E8503A; margin-top: 6px; font-weight: 600; }
div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button { background: #E8503A !important; color: white !important; border: none !important; border-radius: 8px !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 1.2px !important; text-transform: uppercase !important; padding: 0.6rem 1rem !important; }
div[data-testid="stButton"] > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background: #d4432e !important; }
div[data-baseweb="select"] > div, div[data-testid="stSelectbox"] > div > div { background: #1a202c !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; color: white !important; }
div[data-testid="stMultiSelect"] > div { background: #1a202c !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; }
div[data-testid="stFileUploader"] section { background: #1a202c !important; border: 1px dashed rgba(255,255,255,0.18) !important; border-radius: 10px !important; }
div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] label, div[data-testid="stFileUploader"] p { color: rgba(255,255,255,0.6) !important; }
details { background: #1a202c !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; margin-bottom: 1rem !important; }
details summary { font-size: 12px !important; color: rgba(255,255,255,0.4) !important; padding: 0.75rem 1rem !important; }
div[data-testid="stDataFrame"] { background: #1a202c !important; border-radius: 8px !important; }
.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.75) !important; }
.stMarkdown strong, .stMarkdown b { color: white !important; }
div[data-testid="stCheckbox"] label span { color: rgba(255,255,255,0.75) !important; }
</style>
""", unsafe_allow_html=True)

NAME_MAP = {
    "C5_FINALr6": "Overall Satisfaction",
    "C6": "Product Quality",
    "C7": "Ease of Use",
    "C8": "Price / Value",
    "C9": "Customer Support",
    "C10": "Purchase Experience",
    "C11": "Brand Trust",
    "C12": "Retailer",
}
BAR_COLORS = ["#E8503A", "#9ab8d0", "#7a9db8", "#5a82a0", "#3a6788"]

METHOD_INFO = {
    "SHAP (Recommended)": {
        "title": "SHAP — Shapley Values",
        "recommended": True,
        "desc": (
            "Best for most datasets. Uses Shapley values to fairly distribute each variable's contribution "
            "to the predicted outcome. Works with any model, captures non-linear effects, and is the "
            "industry-preferred method for interpretable driver analysis. Start here."
        ),
        "note": "Mean |SHAP value| — average absolute impact on outcome.",
    },
    "Random Forest": {
        "title": "Random Forest Importance",
        "recommended": False,
        "desc": (
            "Good for non-linear patterns. Trains an ensemble of decision trees and ranks variables by how "
            "much they reduce prediction error. Reliable and fast, but scores can be biased toward "
            "high-cardinality variables. Good as a comparison check against SHAP."
        ),
        "note": "Random Forest feature importance — proportion of variance explained.",
    },
    "Correlation": {
        "title": "Correlation (Pearson |r|)",
        "recommended": False,
        "desc": (
            "Simple and transparent. Ranks variables by absolute linear correlation with the outcome. "
            "Easy to explain to non-technical clients, but misses non-linear effects and ignores "
            "interactions between variables."
        ),
        "note": "Pearson |r| — absolute linear correlation with outcome.",
    },
    "Regression": {
        "title": "Standardized Regression Coefficients",
        "recommended": False,
        "desc": (
            "Good for linear relationships. Fits a linear model and ranks by standardized beta weights. "
            "Interpretable but assumes linearity and can be unstable when predictors are highly correlated "
            "(multicollinearity)."
        ),
        "note": "Standardized coefficient |β| — linear importance.",
    },
}


def _auto_label(col):
    s = str(col).replace("_", " ")
    s = _re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = _re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return _re.sub(r"\s+", " ", s).strip().title()

def display_name(col):
    return NAME_MAP.get(col, _auto_label(col))

def is_excluded_column(col):
    return any(k in str(col).lower() for k in [
        "uuid", "record", "date", "start_date", "psid", "pid",
        "marker", "status", "qualityscore", "linercheck", "loi"
    ])

def prepare_model_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    excluded_cols = [c for c in df.columns if is_excluded_column(c)]
    df_model = df.drop(columns=excluded_cols, errors="ignore")
    df_num = df_model.select_dtypes(include=["number"]).copy()
    missing_ratio = df_num.isna().mean()
    high_missing = missing_ratio[missing_ratio > 0.4].index.tolist()
    subgroup_candidates, drop_missing_cols = [], []
    for col in high_missing:
        (subgroup_candidates if df_num[col].notna().sum() > 30 else drop_missing_cols).append(col)
    df_num = df_num.drop(columns=drop_missing_cols, errors="ignore")
    constant_cols = df_num.columns[df_num.nunique(dropna=True) <= 1].tolist()
    df_num = df_num.drop(columns=constant_cols, errors="ignore")
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())
    return df, df_num, {
        "excluded_cols": excluded_cols,
        "drop_missing_cols": drop_missing_cols,
        "subgroup_candidates": subgroup_candidates,
        "constant_cols": constant_cols,
    }

def compute_importance(X, y, method):
    if method == "SHAP (Recommended)":
        if not SHAP_AVAILABLE:
            st.warning("shap package not installed — falling back to Random Forest. Run: pip install shap")
            method = "Random Forest"
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
            return imp.sort_values(ascending=False)
    if method == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
    elif method == "Correlation":
        imp = pd.Series({c: abs(X[c].corr(y)) for c in X.columns})
    else:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LinearRegression().fit(Xs, y)
        imp = pd.Series(np.abs(model.coef_), index=X.columns)
    return imp.sort_values(ascending=False)

def pill_tags(items):
    if not items:
        return '<span style="color:rgba(255,255,255,0.3);font-size:13px;">None</span>'
    return "".join(f'<span class="gbk-tag">{x}</span>' for x in items)

def render_method_info_box(method_key):
    info = METHOD_INFO.get(method_key)
    if not info:
        return
    badge = '<span class="gbk-shap-badge">Recommended</span>' if info["recommended"] else ""
    st.markdown(
        f'<div class="gbk-method-box">'
        f'<div class="gbk-method-title">{info["title"]}{badge}</div>'
        f'<div class="gbk-method-desc">{info["desc"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_bar_chart(top5, title="Top Associated Drivers", method_key=""):
    max_val = top5.iloc[0] if len(top5) else 1
    bars_html = ""
    for i, (col, val) in enumerate(top5.items()):
        color = BAR_COLORS[i] if i < len(BAR_COLORS) else BAR_COLORS[-1]
        pct = round(val / max_val * 100) if max_val else 0
        bars_html += (
            f'<div class="gbk-bar-wrap">'
            f'<div class="gbk-bar-label">{display_name(col)}</div>'
            f'<div class="gbk-bar-row">'
            f'<div class="gbk-bar-track"><div class="gbk-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<div class="gbk-bar-val">{val:.3f}</div>'
            f'</div></div>'
        )
    note = METHOD_INFO.get(method_key, {}).get("note", "Longer bar = stronger importance.")
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">{title}</div>{bars_html}'
        f'<div class="gbk-disclaimer">{note} Directional, not causal.</div></div>',
        unsafe_allow_html=True
    )

def render_insights(target, top5):
    names = [display_name(x) for x in top5.index]
    if not names:
        return
    t = display_name(target)
    n2 = names[1] if len(names) > 1 else names[0]
    n3 = names[2] if len(names) > 2 else n2
    n4 = names[3] if len(names) > 3 else n3
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">What the data suggests</div>'
        f'<div class="gbk-insight gbk-insight-red"><b>Primary signal</b><br>'
        f'<b>{names[0]}</b> appears to be the strongest variable associated with <b>{t}</b>.</div>'
        f'<div class="gbk-insight gbk-insight-blue"><b>Secondary signal</b><br>'
        f'<b>{n2}</b> also shows a meaningful association with this outcome.</div>'
        f'<div class="gbk-insight"><b>Broader context</b><br>'
        f'<b>{n3}</b> and <b>{n4}</b> are also worth reviewing alongside business context and prior knowledge.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_next_steps(target, top5):
    names = [display_name(x) for x in top5.index]
    if not names:
        return
    t = display_name(target)
    n2 = names[1] if len(names) > 1 else names[0]
    n3 = names[2] if len(names) > 2 else n2
    n4 = names[3] if len(names) > 3 else n3
    steps = [
        f"<b>Start with {names[0]}.</b> It shows the strongest association with {t}.",
        f"<b>Review {n2} next.</b> It appears to be another meaningful driver candidate.",
        f"<b>Keep {n3} and {n4} in the discussion.</b> They may matter depending on audience and context.",
        "<b>Validate with business judgment.</b> Use these results as decision support, not a standalone answer.",
    ]
    items = "".join(
        f'<div class="gbk-step-item"><div class="gbk-step-num">{i+1}</div><div>{s}</div></div>'
        for i, s in enumerate(steps)
    )
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">Suggested next steps</div>{items}</div>',
        unsafe_allow_html=True
    )

def render_detail_table(ranked):
    rows = "".join(
        f"<tr>"
        f"<td style='color:rgba(255,255,255,0.25);padding:6px 8px;'>{i+1}</td>"
        f"<td style='color:rgba(255,255,255,0.7);padding:6px 8px;'>{display_name(col)}</td>"
        f"<td style='color:rgba(255,255,255,0.35);padding:6px 8px;'>{val:.3f}</td>"
        f"</tr>"
        for i, (col, val) in enumerate(ranked.items())
    )
    st.markdown(
        f'<div class="gbk-panel"><div class="gbk-panel-title">Full Driver Ranking</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;"><thead><tr>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">#</th>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">Driver</th>'
        f'<th style="text-align:left;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">Score</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True
    )

def run_analysis(df_num, df_raw, target, x_vars, sg_var, method):
    predictors = [c for c in (x_vars if x_vars else df_num.columns) if c != target and c in df_num.columns]
    if not predictors:
        return {"error": "No valid predictor columns available."}
    if not sg_var:
        X, y = df_num[predictors], df_num[target]
        ranked = compute_importance(X, y, method)
        return {"mode": "single", "target": target, "method": method, "ranked": ranked, "top5": ranked.head(5)}
    if sg_var not in df_raw.columns:
        return {"error": f"Subgroup variable '{sg_var}' not found."}
    subgroup_results = []
    for gval in sorted(df_raw[sg_var].dropna().unique()):
        mask = df_raw[sg_var] == gval
        grp = df_num.loc[df_num.index.isin(df_raw[mask].index)]
        if grp.shape[0] < 30:
            subgroup_results.append({"group": gval, "n": grp.shape[0], "skipped": True, "reason": "Too few respondents"})
            continue
        ranked = compute_importance(grp[predictors], grp[target], method)
        subgroup_results.append({"group": gval, "n": grp.shape[0], "skipped": False, "ranked": ranked, "top5": ranked.head(5)})
    return {"mode": "subgroup", "target": target, "method": method, "sg_var": sg_var, "results": subgroup_results}

def render_dashboard():
    st.markdown("""
    <div class="gbk-hero">
      <div class="gbk-eyebrow">GBK Toolbox</div>
      <h1>Marketing<br>Insights Suite</h1>
      <p>Upload a clean dataset, select your variables, choose a method, and run.<br>
      Pre-filter to relevant variables before uploading (1 Y &nbsp;·&nbsp; ~12 X &nbsp;·&nbsp; a few subgroup columns).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gbk-label">Upload dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"], key="dashboard_upload", label_visibility="collapsed")

    if uploaded_file:
        try:
            df_raw, df_num, meta = prepare_model_data(pd.read_excel(uploaded_file))
            st.session_state.uploaded_df_raw = df_raw
            st.session_state.uploaded_df_num = df_num
            st.session_state.uploaded_meta = meta
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.analysis_result = None
        except Exception as e:
            st.error(f"Error loading file: {e}")

    df_raw = st.session_state.uploaded_df_raw
    df_num = st.session_state.uploaded_df_num
    meta = st.session_state.uploaded_meta

    if df_raw is None or df_num is None:
        st.markdown('<div class="gbk-note" style="color:rgba(255,255,255,0.25);">Upload a clean .xlsx dataset to begin.</div>', unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">File</div><div class="gbk-card-text" style="font-size:13px;">{st.session_state.uploaded_filename}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Respondents</div><div class="gbk-stat">{df_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Total Columns</div><div class="gbk-stat">{df_raw.shape[1]:,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="gbk-card"><div class="gbk-card-kicker">Model-Ready</div><div class="gbk-stat">{df_num.shape[1]:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    y_options = df_num.columns.tolist()
    raw_cats = [c for c in df_raw.columns if df_raw[c].dtype == object or (df_raw[c].nunique() <= 10 and c not in y_options)]

    with st.form("analysis_form"):

        # Step 1
        st.markdown('<div class="gbk-panel"><div class="gbk-panel-title">Step 1 · Outcome variable (Y)</div><div class="gbk-note">The metric you want to explain — e.g. overall satisfaction, consideration.</div></div>', unsafe_allow_html=True)
        y_var = st.selectbox("Y variable", ["(select)"] + y_options, format_func=lambda c: display_name(c) if c != "(select)" else "— select outcome —", label_visibility="collapsed", key="dash_y")
        y_selected = y_var if y_var != "(select)" else None

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 2
        x_options = [c for c in y_options if c != y_selected]
        n_x = len(x_options)
        if n_x > 30:
            x_hint = f'<div class="gbk-input-warning">⚠ {n_x} numeric columns detected. Dan recommends selecting ~12 key brand image X variables rather than running all columns through the model.</div>'
        else:
            x_hint = f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:5px;">{n_x} numeric variable{"s" if n_x!=1 else ""} available.</div>'

        st.markdown(
            f'<div class="gbk-panel"><div class="gbk-panel-title">Step 2 · Predictor variables (X)</div>'
            f'<div class="gbk-note">Select specific drivers — recommended: ~10–12 brand image variables. Leave empty to use all available.</div>'
            f'{x_hint}</div>',
            unsafe_allow_html=True
        )
        x_vars = st.multiselect("X variables", x_options, format_func=display_name, label_visibility="collapsed", key="dash_x", placeholder="All variables (default) — selecting ~12 key variables is recommended")

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3
        st.markdown('<div class="gbk-panel"><div class="gbk-panel-title">Step 3 · Subgroup loop (optional)</div><div class="gbk-note">Run driver analysis separately for each level of one categorical variable (e.g. brand, region). One grouping variable at a time.</div></div>', unsafe_allow_html=True)
        use_sg = st.checkbox("Enable subgroup loop", key="dash_use_sg")
        sg_var = None
        if use_sg:
            if raw_cats:
                sg_raw = st.selectbox("Grouping variable", ["(select)"] + raw_cats, label_visibility="collapsed", key="dash_sg")
                sg_var = sg_raw if sg_raw != "(select)" else None
            else:
                st.markdown('<div class="gbk-note">No suitable categorical columns detected.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 4 — Method with info box
        method_options = list(METHOD_INFO.keys())
        if not SHAP_AVAILABLE:
            method_options = [m for m in method_options if m != "SHAP (Recommended)"]

        st.markdown(
            '<div class="gbk-panel"><div class="gbk-panel-title">Step 4 · Method</div>'
            '<div class="gbk-note">Choose how driver importance is calculated. '
            'An explanation of the selected method appears below the dropdown.</div></div>',
            unsafe_allow_html=True
        )
        method = st.selectbox("Method", method_options, label_visibility="collapsed", key="dash_method")
        render_method_info_box(method)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pipeline summary
        x_label = ", ".join(display_name(c) for c in x_vars) if x_vars else "All variables"
        sg_label = f"Loop by {sg_var}" if use_sg and sg_var else "None"
        y_label = display_name(y_selected) if y_selected else "Not selected"
        st.markdown(f"""
        <div class="gbk-panel" style="border-color:rgba(232,80,58,0.3);">
          <div class="gbk-panel-title">Pipeline summary</div>
          <div class="gbk-summary-grid">
            <div><div class="gbk-summary-key">Y</div><div class="gbk-summary-val">{y_label}</div></div>
            <div><div class="gbk-summary-key">X vars</div><div class="gbk-summary-val">{x_label}</div></div>
            <div><div class="gbk-summary-key">Subgroup</div><div class="gbk-summary-val">{sg_label}</div></div>
            <div><div class="gbk-summary-key">Method</div><div class="gbk-summary-val">{method}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_clicked = st.form_submit_button("Run Analysis", use_container_width=True)
        with btn_col2:
            clear_clicked = st.form_submit_button("Clear Results", use_container_width=True)

    if clear_clicked:
        st.session_state.analysis_result = None

    if run_clicked:
        if not y_selected:
            st.error("Please select an outcome variable (Y).")
        else:
            with st.spinner(f"Running {method}..."):
                result = run_analysis(df_num, df_raw, y_selected, x_vars or None, sg_var, method)
            st.session_state.analysis_result = result

    result = st.session_state.analysis_result

    if result:
        if "error" in result:
            st.error(result["error"])
        elif result["mode"] == "single":
            render_bar_chart(result["top5"], method_key=result["method"])
            render_insights(result["target"], result["top5"])
            render_next_steps(result["target"], result["top5"])
            with st.expander("Full driver ranking"):
                render_detail_table(result["ranked"])
        elif result["mode"] == "subgroup":
            st.markdown(f'<div class="gbk-panel"><div class="gbk-panel-title">Subgroup loop · {_auto_label(result["sg_var"])}</div><div class="gbk-note">Running analysis separately for each level of <b>{_auto_label(result["sg_var"])}</b>.</div></div>', unsafe_allow_html=True)
            for item in result["results"]:
                if item["skipped"]:
                    st.markdown(f'<div class="gbk-note" style="margin:6px 0;color:rgba(255,255,255,0.3);">Skipping <b>{item["group"]}</b> — only {item["n"]} respondents.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size:13px;font-weight:700;color:#E8503A;margin:1rem 0 0.25rem;text-transform:uppercase;letter-spacing:1.5px;">{_auto_label(result["sg_var"])}: {item["group"]} · n={item["n"]:,}</div>', unsafe_allow_html=True)
                    render_bar_chart(item["top5"], title=f"Top Drivers — {item['group']}", method_key=result["method"])

    with st.expander("Raw data preview"):
        st.dataframe(df_raw.head(), use_container_width=True)

    with st.expander("Cleaning details"):
        st.markdown(f"""
        <div class="gbk-note">
          <b>Excluded (ID/date/meta):</b><br>{pill_tags(meta['excluded_cols'])}<br><br>
          <b>Dropped (high missing):</b><br>{pill_tags(meta['drop_missing_cols'])}<br><br>
          <b>Subgroup candidates:</b><br>{pill_tags(meta['subgroup_candidates'])}<br><br>
          <b>Dropped (constant):</b><br>{pill_tags(meta['constant_cols'])}
        </div>
        """, unsafe_allow_html=True)

render_dashboard()
