"""
NBA Bottom-10 Turnaround Dashboard
Reads data exported from the Databricks gold layer (06_export_for_dashboard.py)
and committed to this repo -- no live database connection, same pattern as
SwishScore and the HSR dashboard.

Run locally:
    pip install streamlit pandas plotly
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Points at the nba_turnaround repo's raw file path.
GITHUB_BASE_URL = "https://raw.githubusercontent.com/jngoh24/nba_turnaround/main/data"

st.set_page_config(page_title="NBA Turnaround Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Design system -- matches SwishScore/HSR: Athletic-inspired light theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp { background-color: #f7f7f5; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; }
p, div, span, label { font-family: 'Inter', sans-serif; }
.metric-mono { font-family: 'JetBrains Mono', monospace; }

[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

COLOR_JUMPED = "#1a5f3f"   # improved -- dark green
COLOR_STAYED = "#b0413e"   # stayed stuck -- dark red
COLOR_ACCENT = "#c9a961"   # gold accent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    case_df = pd.read_csv(f"{GITHUB_BASE_URL}/bottom10_case_table.csv")
    delta_summary_df = pd.read_csv(f"{GITHUB_BASE_URL}/delta_comparison_summary.csv")
    return case_df, delta_summary_df

try:
    case_df, delta_summary_df = load_data()
except Exception as e:
    st.error(
        f"Couldn't load data from GitHub -- check GITHUB_BASE_URL points at the "
        f"right repo and the CSVs have been pushed. ({e})"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("From Bottom-10 to the Playoffs")
st.markdown(
    "What actually changes for a bad NBA team that turns it around -- "
    "and by how much, realistically."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_changed, tab_explorer = st.tabs(["Overview", "What Changed", "Case Explorer"])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab_overview:
    n_total = len(case_df)
    n_jumped_a = int(case_df["NEXT_target_a_top10_conf"].sum())
    n_jumped_b = int(case_df["NEXT_target_b_made_bracket"].sum())
    n_star_added = int(case_df["star_added"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bottom-10 team-seasons studied", n_total)
    col2.metric("Reached top-10 next season", f"{n_jumped_a} ({n_jumped_a/n_total:.0%})")
    col3.metric("Reached the playoff bracket", f"{n_jumped_b} ({n_jumped_b/n_total:.0%})")
    col4.metric("Added a star-tier player", f"{n_star_added} ({n_star_added/n_total:.0%})")

    st.markdown("---")
    st.markdown(
        "**Method note:** \"Bottom-10\" = league-wide bottom 10 teams by win "
        "percentage. \"Reached the bracket\" means the team actually appeared "
        "in a playoff series the following season (play-in included). "
        "Covers 2016-17 through 2024-25."
    )

# ---------------------------------------------------------------------------
# Tab 2: What Changed
# ---------------------------------------------------------------------------

with tab_changed:
    st.subheader("Which stats actually separated the teams that turned around")
    st.markdown(
        "For each stat, the sign-corrected gap between teams that reached "
        "the bracket and teams that didn't -- positive always means "
        "'genuinely helped the turnaround,' accounting for stats where "
        "*lower* is actually the good direction (opponent shooting, "
        "turnovers, points allowed). PACE has no inherently good direction "
        "and is shown for reference only."
    )

    # NET/OFF/DEF/CLUTCH_NET rating are excluded here, not upstream --
    # net rating mechanically determines wins (it's essentially point
    # differential), and off/def rating are the two halves that combine to
    # produce it, so "net rating improved for teams that made the
    # playoffs" restates the outcome rather than explaining it. The
    # underlying gold table still has these for reference/other uses;
    # this view just isn't the place for them.
    TAUTOLOGICAL_STATS = {"NET_RATING", "OFF_RATING", "DEF_RATING", "CLUTCH_NET_RATING"}
    chart_df = delta_summary_df[~delta_summary_df["stat"].isin(TAUTOLOGICAL_STATS)]

    sorted_df = chart_df.sort_values("improvement", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df["stat"],
        x=sorted_df["improvement"],
        orientation="h",
        marker_color=[
            "#999999" if stat == "PACE" else (COLOR_JUMPED if v > 0 else COLOR_STAYED)
            for stat, v in zip(sorted_df["stat"], sorted_df["improvement"])
        ],
    ))
    fig.update_layout(
        plot_bgcolor="#f7f7f5",
        paper_bgcolor="#f7f7f5",
        font_family="Inter",
        height=500,
        xaxis_title="Improvement (positive = genuinely helped teams that turned around)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Realistic range of improvement, per stat")
    st.markdown(
        "The actual 10th-90th percentile of year-over-year change seen across "
        "all 90 bottom-10 team-seasons -- what a realistic swing looks like, "
        "not a hypothetical one."
    )

    selected_stat = st.selectbox("Stat", options=chart_df["stat"].tolist())
    stat_row = chart_df[chart_df["stat"] == selected_stat].iloc[0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[stat_row["p10"], stat_row["p90"]], y=[0, 0],
        mode="lines", line=dict(color=COLOR_ACCENT, width=6), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=[stat_row["median"]], y=[0], mode="markers",
        marker=dict(size=14, color=COLOR_JUMPED), name="Historical median",
    ))
    fig2.update_layout(
        plot_bgcolor="#f7f7f5", paper_bgcolor="#f7f7f5", font_family="Inter",
        height=150, yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"Year-over-year change in {selected_stat} (percentile points)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("10th percentile", f"{stat_row['p10']:.3f}")
    c2.metric("Median", f"{stat_row['median']:.3f}")
    c3.metric("90th percentile", f"{stat_row['p90']:.3f}")

# ---------------------------------------------------------------------------
# Tab 3: Case Explorer
# ---------------------------------------------------------------------------

with tab_explorer:
    st.subheader("Browse the 90 bottom-10 team-seasons")

    col1, col2 = st.columns(2)
    with col1:
        outcome_filter = st.multiselect(
            "Outcome", options=["Reached bracket", "Did not reach bracket"],
            default=["Reached bracket", "Did not reach bracket"],
        )
    with col2:
        star_filter = st.multiselect(
            "Star addition", options=["Added a star", "No star addition"],
            default=["Added a star", "No star addition"],
        )

    filtered_df = case_df.copy()
    outcome_mask = pd.Series(False, index=filtered_df.index)
    if "Reached bracket" in outcome_filter:
        outcome_mask |= filtered_df["NEXT_target_b_made_bracket"] == True
    if "Did not reach bracket" in outcome_filter:
        outcome_mask |= filtered_df["NEXT_target_b_made_bracket"] == False
    filtered_df = filtered_df[outcome_mask]

    star_mask = pd.Series(False, index=filtered_df.index)
    if "Added a star" in star_filter:
        star_mask |= filtered_df["star_added"] == True
    if "No star addition" in star_filter:
        star_mask |= filtered_df["star_added"] == False
    filtered_df = filtered_df[star_mask]

    display_cols = [
        "TEAM_NAME", "season", "WinPCT",
        "NEXT_target_a_top10_conf", "NEXT_target_b_made_bracket",
        "star_added", "ROOKIES_max_minutes",
    ]
    if "predicted_prob_target_b" in filtered_df.columns:
        display_cols.append("predicted_prob_target_b")

    st.dataframe(
        filtered_df[display_cols].sort_values("season", ascending=False),
        use_container_width=True, hide_index=True,
    )
