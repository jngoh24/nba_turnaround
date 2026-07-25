"""
NBA Bottom-10 Turnaround Dashboard
Reads data exported from the Databricks gold layer (06_export_for_dashboard.py)
and committed to this repo -- no live database connection, same pattern as
SwishScore and the HSR dashboard.

Run locally:
    pip install streamlit pandas plotly
    streamlit run app.py
"""

import io

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
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

.stApp { background-color: #f7f7f5; color: #1a1a1a; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; color: #1a1a1a !important; }
p, div, span, label { font-family: 'Inter', sans-serif; color: #1a1a1a; }
.metric-mono { font-family: 'JetBrains Mono', monospace; }

[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #1a1a1a !important; }
[data-testid="stMetricLabel"] { color: #1a1a1a !important; }
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
    full_df = pd.read_csv(f"{GITHUB_BASE_URL}/team_season_features.csv")
    roster_bounds_df = pd.read_csv(f"{GITHUB_BASE_URL}/roster_feature_bounds.csv")
    return case_df, delta_summary_df, full_df, roster_bounds_df

try:
    case_df, delta_summary_df, full_df, roster_bounds_df = load_data()
except Exception as e:
    st.error(
        f"Couldn't load data from GitHub -- check GITHUB_BASE_URL points at the "
        f"right repo and the CSVs have been pushed. ({e})"
    )
    st.stop()


@st.cache_resource(ttl=3600)
def load_model_and_features():
    """
    Loads the TRIMMED model (no tautological rating-delta features) --
    the actionable-features version, meant for exactly this kind of
    scenario tool. feature_cols_trimmed.json defines the exact column
    order the model expects; loaded from the same export rather than
    hardcoded here, so the two can't silently drift apart.
    """
    model_resp = requests.get(f"{GITHUB_BASE_URL}/model_target_b.joblib")
    model_resp.raise_for_status()
    model = joblib.load(io.BytesIO(model_resp.content))

    features_resp = requests.get(f"{GITHUB_BASE_URL}/feature_cols_trimmed.json")
    features_resp.raise_for_status()
    feature_cols = features_resp.json()

    return model, feature_cols

try:
    model, FEATURE_COLS_TRIMMED = load_model_and_features()
    LEVEL_FEATURES = [c for c in FEATURE_COLS_TRIMMED if c.endswith("_PCTILE") and not c.startswith("DELTA_")]
    DELTA_FEATURES = [c for c in FEATURE_COLS_TRIMMED if c.startswith("DELTA_")]
    ROOKIE_FEATURES = [c for c in FEATURE_COLS_TRIMMED if c.startswith("ROOKIES_")]
    model_load_error = None
except Exception as e:
    model, FEATURE_COLS_TRIMMED = None, []
    LEVEL_FEATURES, DELTA_FEATURES, ROOKIE_FEATURES = [], [], []
    model_load_error = str(e)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("From Bottom-10 to the Playoffs")
st.markdown(
    "What actually changes for a bad NBA team that turns it around -- "
    "and by how much, realistically."
)

# ---------------------------------------------------------------------------
# Shared team/season selector -- used by Team Diagnostic and What-If
# ---------------------------------------------------------------------------

full_df["team_season_label"] = full_df["TEAM_NAME"] + " -- " + full_df["season"]
label_options = full_df.sort_values(["season", "TEAM_NAME"], ascending=[False, True])["team_season_label"].tolist()

st.markdown("---")
selected_label = st.selectbox(
    "Team & season to analyze (used by Team Diagnostic and What-If below)",
    options=label_options,
)
selected_row = full_df[full_df["team_season_label"] == selected_label].iloc[0]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_changed, tab_diagnostic, tab_whatif, tab_explorer = st.tabs(
    ["Overview", "What Changed", "Team Diagnostic", "What-If", "Case Explorer"]
)

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
# Tab 3: Team Diagnostic
# ---------------------------------------------------------------------------

with tab_diagnostic:
    st.subheader(f"{selected_row['TEAM_NAME']} -- {selected_row['season']}")

    is_bottom10 = bool(selected_row.get("is_bottom_n", False))
    st.markdown(
        f"**{'Bottom-10 team' if is_bottom10 else 'Not a bottom-10 team'}** "
        f"that season -- WinPCT {selected_row['WinPCT']:.3f}"
    )

    st.markdown(
        "Percentile rank within the league that season, for every stat -- "
        "0 = worst in the league, 1 = best. This is where the team actually "
        "stood, not where it ended up."
    )

    diag_stats = [c.replace("_PCTILE", "") for c in LEVEL_FEATURES]
    diag_values = [selected_row[c] for c in LEVEL_FEATURES]
    diag_df = pd.DataFrame({"stat": diag_stats, "percentile": diag_values}).sort_values("percentile")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        y=diag_df["stat"], x=diag_df["percentile"], orientation="h",
        marker_color=[COLOR_STAYED if v < 0.5 else COLOR_JUMPED for v in diag_df["percentile"]],
    ))
    fig3.add_vline(x=0.5, line_dash="dash", line_color="#999999")
    fig3.update_layout(
        plot_bgcolor="#f7f7f5", paper_bgcolor="#f7f7f5", font_family="Inter",
        height=550, xaxis_title="League percentile that season (dashed line = league median)",
        margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 1],
    )
    st.plotly_chart(fig3, use_container_width=True)

    if is_bottom10:
        st.markdown("---")
        st.markdown(
            f"**What actually happened next season:** reached top-10 conference: "
            f"**{selected_row.get('NEXT_target_a_top10_conf', 'N/A')}** -- "
            f"reached the playoff bracket: **{selected_row.get('NEXT_target_b_made_bracket', 'N/A')}**"
        )

# ---------------------------------------------------------------------------
# Tab 4: What-If
# ---------------------------------------------------------------------------

with tab_whatif:
    if model is None:
        st.error(
            f"Couldn't load the model from GitHub -- check model_target_b.joblib "
            f"and feature_cols_trimmed.json have been pushed. ({model_load_error})"
        )
    else:
        st.subheader(f"What-if: {selected_row['TEAM_NAME']} -- {selected_row['season']}")
        st.markdown(
            "Starting from this team's ACTUAL current-season stats, assume no "
            "further improvement (all sliders at zero) as the baseline, then "
            "move sliders to see how realistic changes shift the probability "
            "of reaching the playoff bracket next season. Every slider is "
            "bounded by the actual 10th-90th percentile range of year-over-year "
            "change seen across 90 bottom-10 team-seasons -- you can't ask for "
            "an improvement that's never actually happened."
        )
        st.caption(
            "Uses the tautology-free model -- excludes NET/OFF/DEF/CLUTCH_NET "
            "rating deltas, since those mechanically restate the outcome "
            "rather than something a front office can act on."
        )

        def build_input_row(deltas: dict, rookies: dict, star_added_val: bool) -> pd.DataFrame:
            row = {}
            for col in LEVEL_FEATURES:
                row[col] = selected_row[col]
            for col in DELTA_FEATURES:
                row[col] = deltas.get(col, 0.0)
            for col in ROOKIE_FEATURES:
                row[col] = rookies.get(col, 0.0)
            if "star_added" in FEATURE_COLS_TRIMMED:
                row["star_added"] = float(star_added_val)
            if "is_bottom_n" in FEATURE_COLS_TRIMMED:
                row["is_bottom_n"] = float(selected_row.get("is_bottom_n", True))
            if "WinPCT" in FEATURE_COLS_TRIMMED:
                row["WinPCT"] = selected_row["WinPCT"]
            return pd.DataFrame([row])[FEATURE_COLS_TRIMMED]

        baseline_row = build_input_row(deltas={}, rookies={}, star_added_val=False)
        baseline_prob = model.predict_proba(baseline_row)[0, 1]

        st.markdown("---")
        col_sliders, col_result = st.columns([2, 1])

        with col_sliders:
            st.markdown("**Star addition**")
            star_added_input = st.checkbox("Assume a star-tier player is added this offseason")

            st.markdown("**Rookie contribution**")
            rookie_inputs = {}
            for col in ROOKIE_FEATURES:
                bounds_row = roster_bounds_df[roster_bounds_df["stat"] == col]
                if len(bounds_row) == 0:
                    continue
                lo, hi = float(bounds_row.iloc[0]["p10"]), float(bounds_row.iloc[0]["p90"])
                lo, hi = min(lo, 0.0), max(hi, 0.0)
                rookie_inputs[col] = st.slider(
                    col.replace("ROOKIES_", "").replace("_", " ").title(),
                    min_value=lo, max_value=hi, value=0.0,
                )

            st.markdown("**Component stat changes** (year-over-year, percentile points)")
            delta_inputs = {}
            for col in DELTA_FEATURES:
                stat_name = col.replace("DELTA_", "").replace("_PCTILE", "")
                bounds_row = delta_summary_df[delta_summary_df["stat"] == stat_name]
                if len(bounds_row) == 0:
                    continue
                lo, hi = float(bounds_row.iloc[0]["p10"]), float(bounds_row.iloc[0]["p90"])
                delta_inputs[col] = st.slider(stat_name, min_value=lo, max_value=hi, value=0.0)

        scenario_row = build_input_row(deltas=delta_inputs, rookies=rookie_inputs, star_added_val=star_added_input)
        scenario_prob = model.predict_proba(scenario_row)[0, 1]

        with col_result:
            st.metric(
                "Playoff bracket probability",
                f"{scenario_prob:.0%}",
                delta=f"{(scenario_prob - baseline_prob):+.0%} vs. no change",
            )
            st.caption(f"Baseline (no improvement): {baseline_prob:.0%}")

# ---------------------------------------------------------------------------
# Tab 5: Case Explorer
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
