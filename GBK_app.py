import re as _re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="GBK Driver Analysis Tool", layout="wide")

# =========================================================
# GBK Brand Styling
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #2d3748 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

.gbk-nav {
    background: #1a202c;
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    position: sticky;
    top: 0;
    z-index: 999;
}
.gbk-nav-logo {
    display: flex;
    align-items: baseline;
    gap: 5px;
}
.gbk-logo-text {
    font-size: 20px;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1;
}
.gbk-logo-g { color: #E8503A; }
.gbk-logo-collective {
    font-size: 10px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding-bottom: 1px;
}
.gbk-nav-links {
    display: flex;
    gap: 2rem;
}
.gbk-nav-links a {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    text-decoration: none;
    transition: color 0.15s;
}
.gbk-nav-links a:hover { color: white; }

.gbk-hero {
    background: #1a202c;
    padding: 3rem 2.5rem 2.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0;
}
.gbk-eyebrow {
    font-size: 11px;
    color: #E8503A;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.gbk-hero h1 {
    font-size: 42px;
    font-weight: 900;
    color: white;
    line-height: 1.05;
    text-transform: uppercase;
    letter-spacing: -1px;
    margin-bottom: 0.75rem;
}
.gbk-hero p {
    font-size: 14px;
    color: rgba(255,255,255,0.45);
    max-width: 520px;
    line-height: 1.7;
    margin: 0;
}

.gbk-content {
    padding: 2rem 2.5rem;
}

.gbk-section-label {
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.gbk-upload-box {
    background: #1a202c;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
}

.gbk-note {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    line-height: 1.7;
}

.gbk-panel {
    background: #1a202c;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.gbk-panel-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: rgba(255,255,255,0.3);
    margin-bottom: 1.25rem;
}

.gbk-disclaimer {
    font-size: 11px;
    color: rgba(255,255,255,0.2);
    margin-top: 0.75rem;
    font-style: italic;
}

div[data-testid="stSelectbox"] > div > div {
    background: #1a202c !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: white !important;
}
div[data-testid="stSelectbox"] svg { fill: rgba(255,255,255,0.4) !important; }

div[data-testid="stButton"] > button {
    background: #E8503A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
}
div[data-testid="stButton"] > button:hover {
    background: #d4432e !important;
    color: white !important;
}

div[data-testid="stFileUploader"] section {
    background: #1a202c !important;
    border: 1px dashed rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p {
    color: rgba(255,255,255,0.6) !important;
}

details {
    background: #1a202c !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    margin-bottom: 1rem !important;
}
details summary {
    font-size: 12px !important;
    color: rgba(255,255,255,0.4) !important;
    padding: 0.75rem 1rem !important;
}

div[data-testid="stDataFrame"] {
    background: #1a202c !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

.stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stText"],
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] span {
    color: rgba(255,255,255,0.75) !important;
}
.stMarkdown strong, .stMarkdown b,
div[data-testid="stExpander"] strong,
div[data-testid="stExpander"] b {
    color: white !important;
}

pre, code {
    background: rgba(0,0,0,0.3) !important;
    color: rgba(255,255,255,0.75) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Nav Bar
# =========================================================
st.markdown("""
<div class="gbk-nav">
  <div class="gbk-nav-logo">
    <div class="gbk-logo-text">
      <span class="gbk-logo-g">GBK</span>
    </div>
    <div class="gbk-logo-collective">Collective</div>
  </div>
  <div class="gbk-nav-links">
    <a href="#">Our Approach</a>
    <a href="#">Team</a>
    <a href="#">GBK Toolbox</a>
    <a href="#">Thought Leadership</a>
    <a href="#">Contact</a>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Hero
# =========================================================
st.markdown("""
<div class="gbk-hero">
  <div class="gbk-eyebrow">GBK Toolbox</div>
  <h1>Driver<br>Analysis Tool</h1>
  <p>Identify the top factors behind key business outcomes. Built for marketing teams who need clarity, fast.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Config
# =========================================================
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

# =========================================================
# Helper Functions
# =========================================================
def _auto_label(col: str) -> str:
    s = str(col)
    s = s.replace("_", " ")
    s = _re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = _re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s.title()

def clean_column_name(col: str) -> str:
    return str(col).strip()

def display_name(col: str) -> str:
    return NAME_MAP.get(col, _auto_label(col))

def is_excluded_column(col: str) -> bool:
    col_lower = col.lower()
    exclude_keywords = [
        "uuid", "record", "date", "start_date", "psid", "pid",
        "marker", "status", "qualityscore", "linercheck", "loi"
    ]
    return any(k in col_lower for k in exclude_keywords)

def prepare_model_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # 1. 去掉明显不该进模型的列
    excluded_cols = [c for c in df.columns if is_excluded_column(c)]
    df_model = df.drop(columns=excluded_cols, errors="ignore")

    # 2. 只保留数值列
    df_num = df_model.select_dtypes(include=["number"]).copy()

    # 3. 检查缺失值比例
    missing_ratio = df_num.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.4].index.tolist()

    # 4. 高缺失列分成两类
    subgroup_candidates = []
    drop_missing_cols = []

    for col in high_missing_cols:
        non_null_count = df_num[col].notna().sum()
        if non_null_count > 30:
            subgroup_candidates.append(col)
        else:
            drop_missing_cols.append(col)

    # 5. baseline model 先删掉真正太空的列
    df_num = df_num.drop(columns=drop_missing_cols, errors="ignore")

    # 6. 去掉 constant columns
    nunique = df_num.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    df_num = df_num.drop(columns=constant_cols, errors="ignore")

    # 7. 用 median 补剩下的缺失值
    for col in df_num.columns:
        df_num[col] = df_num[col].fillna(df_num[col].median())

    meta = {
        "excluded_cols": excluded_cols,
        "drop_missing_cols": drop_missing_cols,
        "subgroup_candidates": subgroup_candidates,
        "constant_cols": constant_cols,
    }

    return df, df_num, meta

PANEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: transparent; }
.gbk-panel { background:#1a202c; border-radius:10px; border:1px solid rgba(255,255,255,0.06); padding:1.5rem 1.75rem; margin-bottom:4px; }
.gbk-panel-title { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; color:rgba(255,255,255,0.3); margin-bottom:1.25rem; }
.gbk-bar-row { display:flex; align-items:center; gap:14px; margin-bottom:12px; }
.gbk-bar-label { font-size:13px; color:rgba(255,255,255,0.65); width:175px; flex-shrink:0; }
.gbk-bar-track { flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px; overflow:hidden; }
.gbk-bar-fill { height:100%; border-radius:4px; }
.gbk-bar-value { font-size:12px; color:rgba(255,255,255,0.3); width:42px; text-align:right; flex-shrink:0; }
.gbk-disclaimer { font-size:11px; color:rgba(255,255,255,0.2); margin-top:0.75rem; font-style:italic; }
.gbk-insights { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.gbk-insight { background:rgba(255,255,255,0.03); border-radius:6px; padding:0.875rem 1rem; border-left:3px solid; font-size:13px; color:rgba(255,255,255,0.55); line-height:1.6; }
.gbk-insight b { color:white; font-weight:600; }
.gbk-insight-wide { grid-column:span 2; }
.il-red { border-color:#E8503A; }
.il-blue { border-color:#7a9db8; }
.il-gray { border-color:rgba(255,255,255,0.12); }
.gbk-steps { list-style:none; padding:0; display:flex; flex-direction:column; gap:12px; }
.gbk-step { display:flex; align-items:flex-start; gap:12px; font-size:13px; color:rgba(255,255,255,0.55); line-height:1.6; }
.gbk-step b { color:rgba(255,255,255,0.85); }
.gbk-step-num { background:#E8503A; color:white; font-size:10px; font-weight:700; width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; }
.gbk-table { width:100%; border-collapse:collapse; font-size:13px; }
.gbk-table th { text-align:left; font-size:10px; color:rgba(255,255,255,0.3); letter-spacing:2px; text-transform:uppercase; padding:0 0 0.75rem; border-bottom:1px solid rgba(255,255,255,0.08); font-weight:700; }
.gbk-table td { padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.04); color:rgba(255,255,255,0.6); }
.gbk-table td:first-child { color:rgba(255,255,255,0.25); width:32px; }
.gbk-table td:last-child { color:rgba(255,255,255,0.35); }
</style>
"""

def render_bar_chart(top5: pd.Series):
    max_val = top5.iloc[0]
    bars_html = ""
    for i, (col, val) in enumerate(top5.items()):
        color = BAR_COLORS[i] if i < len(BAR_COLORS) else BAR_COLORS[-1]
        pct = round(val / max_val * 100)
        label = display_name(col)
        bars_html += f"""
        <div class="gbk-bar-row">
          <div class="gbk-bar-label">{label}</div>
          <div class="gbk-bar-track">
            <div class="gbk-bar-fill" style="width:{pct}%; background:{color};"></div>
          </div>
          <div class="gbk-bar-value">{val:.3f}</div>
        </div>
        """
    components.html(PANEL_CSS + f"""
    <div class="gbk-panel">
      <div class="gbk-panel-title">What's driving the outcome? (Top 5 factors)</div>
      {bars_html}
      <div class="gbk-disclaimer">Longer bar = stronger influence on the outcome. These are patterns from the data, not absolute facts.</div>
    </div>
    """, height=60 + len(top5) * 44)

def render_insights(target: str, top5: pd.Series):
    names = [display_name(x) for x in top5.index.tolist()]
    t_label = display_name(target)

    second = names[1] if len(names) > 1 else names[0]
    third = names[2] if len(names) > 2 else second
    fourth = names[3] if len(names) > 3 else third

    components.html(PANEL_CSS + f"""
    <div class="gbk-panel">
      <div class="gbk-panel-title">What the data is telling us</div>
      <div class="gbk-insights">
        <div class="gbk-insight il-red">
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:rgba(255,80,58,0.8); font-weight:700; margin-bottom:6px;">Biggest lever</div>
          When customers rate <b>{t_label}</b> highly or poorly, <b>{names[0]}</b> is the strongest factor.
        </div>
        <div class="gbk-insight il-blue">
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:rgba(122,157,184,0.9); font-weight:700; margin-bottom:6px;">Supporting driver</div>
          <b>{second}</b> is also strongly associated with this outcome.
        </div>
        <div class="gbk-insight il-gray gbk-insight-wide">
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:rgba(255,255,255,0.35); font-weight:700; margin-bottom:6px;">Also worth watching</div>
          <b>{third}</b> and <b>{fourth}</b> round out the picture and may matter in future optimization.
        </div>
      </div>
    </div>
    """, height=260)

def render_next_steps(target: str, top5: pd.Series):
    names = [display_name(x) for x in top5.index.tolist()]
    t_label = display_name(target)

    second = names[1] if len(names) > 1 else names[0]
    third = names[2] if len(names) > 2 else second
    fourth = names[3] if len(names) > 3 else third

    steps = [
        f"<b>Start with {names[0]}.</b> This has the strongest relationship with {t_label}.",
        f"<b>Then review {second}.</b> This is the second strongest lever.",
        f"<b>Watch {third} and {fourth}.</b> They may still influence the outcome over time.",
        f"<b>Validate with business context.</b> Use these patterns alongside team judgment.",
    ]
    items = "".join(
        f'<li class="gbk-step"><span class="gbk-step-num">{i+1}</span><span>{s}</span></li>'
        for i, s in enumerate(steps)
    )
    components.html(PANEL_CSS + f"""
    <div class="gbk-panel">
      <div class="gbk-panel-title">What to do next</div>
      <ul class="gbk-steps">{items}</ul>
    </div>
    """, height=300)

def render_detail_table(top5: pd.Series):
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{display_name(col)}</td><td>{val:.3f}</td></tr>"
        for i, (col, val) in enumerate(top5.items())
    )
    components.html(PANEL_CSS + f"""
    <div class="gbk-panel">
      <div class="gbk-panel-title">Full Driver Ranking</div>
      <table class="gbk-table">
        <thead><tr><th>#</th><th>Driver</th><th>Importance Score</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, height=60 + len(top5) * 44)

def pill_list(items):
    if not items:
        return '<span style="color:rgba(255,255,255,0.3); font-size:13px;">None</span>'
    return " ".join(
        f'<span style="display:inline-block; background:rgba(255,255,255,0.07); color:rgba(255,255,255,0.7); font-size:12px; padding:3px 10px; border-radius:4px; margin:2px 2px;">{x}</span>'
        for x in items
    )

# =========================================================
# Main App
# =========================================================
st.markdown('<div class="gbk-content">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is None:
    st.markdown("""
    <div class="gbk-upload-box">
      <div class="gbk-section-label">Upload dataset</div>
      <div class="gbk-note">
        Upload an <b>.xlsx</b> survey dataset to begin. The tool will automatically clean the data,
        identify model-ready numeric fields, and allow you to run driver analysis on a selected outcome metric.
        Uploaded files are only used during the current session.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_raw, df_num, meta = prepare_model_data(df_raw)

        components.html(PANEL_CSS + f"""
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px;">
          <div class="gbk-panel" style="margin-bottom:0;">
            <div class="gbk-panel-title">Respondents</div>
            <div style="font-size:32px; font-weight:800; color:white; line-height:1;">{df_raw.shape[0]:,}</div>
          </div>
          <div class="gbk-panel" style="margin-bottom:0;">
            <div class="gbk-panel-title">Total Columns</div>
            <div style="font-size:32px; font-weight:800; color:white; line-height:1;">{df_raw.shape[1]:,}</div>
          </div>
          <div class="gbk-panel" style="margin-bottom:0;">
            <div class="gbk-panel-title">Model-Ready</div>
            <div style="font-size:32px; font-weight:800; color:white; line-height:1;">{df_num.shape[1]:,}</div>
          </div>
        </div>
        <div style="background:rgba(232,80,58,0.07); border:1px solid rgba(232,80,58,0.2); border-radius:8px; padding:0.875rem 1.25rem; font-size:13px; color:rgba(255,255,255,0.55); line-height:1.7;">
          <b style="color:#E8503A; font-weight:600;">Automated data prep:</b>
          ID / date / meta fields excluded &nbsp;·&nbsp;
          structurally empty columns dropped &nbsp;·&nbsp;
          subgroup-driven sparse variables flagged separately &nbsp;·&nbsp;
          remaining gaps filled with median &nbsp;·&nbsp;
          constant columns dropped
        </div>
        """, height=220)

        if df_num.shape[1] < 2:
            st.error("Not enough usable numeric columns available after cleaning.")
        else:
            st.markdown('<div class="gbk-section-label">Select outcome metric</div>', unsafe_allow_html=True)
            col_sel, col_btn = st.columns([4, 1])

            with col_sel:
                target = st.selectbox(
                    "",
                    options=df_num.columns.tolist(),
                    format_func=display_name,
                    label_visibility="collapsed"
                )

            with col_btn:
                run = st.button("Run Analysis")

            if run:
                X = df_num.drop(columns=[target], errors="ignore")
                y = df_num[target]

                if X.shape[1] == 0:
                    st.error("No predictor columns available after cleaning.")
                else:
                    with st.spinner("Running model..."):
                        model = RandomForestRegressor(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X, y)

                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    top5 = importances.sort_values(ascending=False).head(5)

                    render_bar_chart(top5)
                    render_insights(target, top5)
                    render_next_steps(target, top5)

                    with st.expander("View detailed driver ranking"):
                        render_detail_table(top5)

        with st.expander("View raw data preview"):
            st.dataframe(df_raw.head(), use_container_width=True)

        st.write("subgroup candidates:", meta["subgroup_candidates"])

        with st.expander("View cleaning details"):
            components.html(PANEL_CSS + f"""
            <div style="padding: 4px 0;">

              <div style="margin-bottom:1.25rem;">
                <div class="gbk-panel-title" style="margin-bottom:0.5rem;">
                  Excluded ID / date / meta columns
                </div>
                <div>{pill_list(meta['excluded_cols'])}</div>
              </div>

              <div style="margin-bottom:1.25rem;">
                <div class="gbk-panel-title" style="margin-bottom:0.5rem;">
                  Dropped high-missing columns (&gt;40% missing)
                </div>
                <div>{pill_list(meta['drop_missing_cols'])}</div>
              </div>

              <div style="margin-bottom:1.25rem;">
                <div class="gbk-panel-title" style="margin-bottom:0.5rem;">
                  Subgroup candidate variables
                </div>
                <div>{pill_list(meta['subgroup_candidates'])}</div>
              </div>

              <div>
                <div class="gbk-panel-title" style="margin-bottom:0.5rem;">
                  Dropped constant columns
                </div>
                <div>{pill_list(meta['constant_cols'])}</div>
              </div>

            </div>
            """, height=380)

    except Exception as e:
        st.error(f"Error loading data: {e}")

st.markdown('</div>', unsafe_allow_html=True)
