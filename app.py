"""
NBA Bottom-10 Turnaround Dashboard
Reads data exported from the Databricks gold layer (06_export_for_dashboard.py)
and committed to this repo -- no live database connection, same pattern as
SwishScore and the HSR dashboard.

Run locally:
    pip install streamlit pandas plotly scikit-learn joblib requests
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

GITHUB_BASE_URL = "https://raw.githubusercontent.com/jngoh24/nba_turnaround/main/data"

st.set_page_config(page_title="NBA Turnaround Dashboard", layout="wide", initial_sidebar_state="expanded")

# Fixed data-encoding colors -- NEVER swapped for team colors, so "good/bad"
# always reads the same way regardless of which team is selected.
COLOR_JUMPED = "#1a7a4c"
COLOR_STAYED = "#b0413e"
COLOR_NEUTRAL = "#9a9a95"
BG = "#f7f7f5"
TEXT = "#1a1a1a"
TEXT_MUTED = "#6b6b66"

# 30 current NBA teams -- primary/secondary brand colors (stable across
# seasons) and the logocdn.com slug for the current logo.
TEAM_STYLE = {
    "Atlanta Hawks": {"primary": "#E03A3E", "secondary": "#C1D32F", "slug": "atlanta-hawks"},
    "Boston Celtics": {"primary": "#007A33", "secondary": "#BA9653", "slug": "boston-celtics"},
    "Brooklyn Nets": {"primary": "#000000", "secondary": "#707271", "slug": "brooklyn-nets"},
    "Charlotte Hornets": {"primary": "#1D1160", "secondary": "#00788C", "slug": "charlotte-hornets"},
    "Chicago Bulls": {"primary": "#CE1141", "secondary": "#000000", "slug": "chicago-bulls"},
    "Cleveland Cavaliers": {"primary": "#860038", "secondary": "#FDBB30", "slug": "cleveland-cavaliers"},
    "Dallas Mavericks": {"primary": "#00538C", "secondary": "#002B5E", "slug": "dallas-mavericks"},
    "Denver Nuggets": {"primary": "#0E2240", "secondary": "#FEC524", "slug": "denver-nuggets"},
    "Detroit Pistons": {"primary": "#C8102E", "secondary": "#1D42BA", "slug": "detroit-pistons"},
    "Golden State Warriors": {"primary": "#1D428A", "secondary": "#FFC72C", "slug": "golden-state-warriors"},
    "Houston Rockets": {"primary": "#CE1141", "secondary": "#000000", "slug": "houston-rockets"},
    "Indiana Pacers": {"primary": "#002D62", "secondary": "#FDBB30", "slug": "indiana-pacers"},
    "LA Clippers": {"primary": "#C8102E", "secondary": "#1D428A", "slug": "los-angeles-clippers"},
    "Los Angeles Clippers": {"primary": "#C8102E", "secondary": "#1D428A", "slug": "los-angeles-clippers"},
    "Los Angeles Lakers": {"primary": "#552583", "secondary": "#FDB927", "slug": "los-angeles-lakers"},
    "Memphis Grizzlies": {"primary": "#5D76A9", "secondary": "#12173F", "slug": "memphis-grizzlies"},
    "Miami Heat": {"primary": "#98002E", "secondary": "#F9A01B", "slug": "miami-heat"},
    "Milwaukee Bucks": {"primary": "#00471B", "secondary": "#EEE1C6", "slug": "milwaukee-bucks"},
    "Minnesota Timberwolves": {"primary": "#0C2340", "secondary": "#236192", "slug": "minnesota-timberwolves"},
    "New Orleans Pelicans": {"primary": "#0C2340", "secondary": "#C8102E", "slug": "new-orleans-pelicans"},
    "New York Knicks": {"primary": "#006BB6", "secondary": "#F58426", "slug": "new-york-knicks"},
    "Oklahoma City Thunder": {"primary": "#007AC1", "secondary": "#EF3B24", "slug": "oklahoma-city-thunder"},
    "Orlando Magic": {"primary": "#0077C0", "secondary": "#C4CED4", "slug": "orlando-magic"},
    "Philadelphia 76ers": {"primary": "#006BB6", "secondary": "#ED174C", "slug": "philadelphia-76ers"},
    "Phoenix Suns": {"primary": "#1D1160", "secondary": "#E56020", "slug": "phoenix-suns"},
    "Portland Trail Blazers": {"primary": "#E03A3E", "secondary": "#000000", "slug": "portland-trail-blazers"},
    "Sacramento Kings": {"primary": "#5A2D81", "secondary": "#63727A", "slug": "sacramento-kings"},
    "San Antonio Spurs": {"primary": "#8A8D8F", "secondary": "#000000", "slug": "san-antonio-spurs"},
    "Toronto Raptors": {"primary": "#CE1141", "secondary": "#000000", "slug": "toronto-raptors"},
    "Utah Jazz": {"primary": "#002B5C", "secondary": "#F9A01B", "slug": "utah-jazz"},
    "Washington Wizards": {"primary": "#002B5C", "secondary": "#E31837", "slug": "washington-wizards"},
}
DEFAULT_STYLE = {"primary": "#1a1a1a", "secondary": "#9a9a95", "slug": None}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    case_df = pd.read_csv(f"{GITHUB_BASE_URL}/bottom10_case_table.csv")
    delta_summary_df = pd.read_csv(f"{GITHUB_BASE_URL}/delta_comparison_summary.csv")
    full_df = pd.read_csv(f"{GITHUB_BASE_URL}/team_season_features.csv")
    roster_bounds_df = pd.read_csv(f"{GITHUB_BASE_URL}/roster_feature_bounds.csv")
    benchmark_df = pd.read_csv(f"{GITHUB_BASE_URL}/playoff_benchmark.csv")
    return case_df, delta_summary_df, full_df, roster_bounds_df, benchmark_df

try:
    case_df, delta_summary_df, full_df, roster_bounds_df, benchmark_df = load_data()
except Exception as e:
    st.error(
        f"Couldn't load data from GitHub -- check GITHUB_BASE_URL points at the "
        f"right repo and the CSVs have been pushed. ({e})"
    )
    st.stop()


@st.cache_resource(ttl=3600)
def load_model_and_features():
    model_resp = requests.get(f"{GITHUB_BASE_URL}/model_target_b.joblib")
    model_resp.raise_for_status()
    model = joblib.load(io.BytesIO(model_resp.content))
    features_resp = requests.get(f"{GITHUB_BASE_URL}/feature_cols_trimmed.json")
    features_resp.raise_for_status()
    return model, features_resp.json()

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
# Sidebar -- team selector, logo, headline metric, methodology note
# ---------------------------------------------------------------------------

team_options = sorted(full_df["TEAM_NAME"].unique())

with st.sidebar:
    st.markdown("**NBA TURNAROUND MODEL**")
    st.caption("2016-17 to 2024-25 -- bottom-10 team turnarounds")
    st.markdown("---")

    selected_team = st.selectbox("Team", options=team_options)

    season_options = sorted(
        full_df[full_df["TEAM_NAME"] == selected_team]["season"].unique(), reverse=True
    )
    selected_season = st.selectbox("Season", options=season_options)

    selected_row = full_df[
        (full_df["TEAM_NAME"] == selected_team) & (full_df["season"] == selected_season)
    ].iloc[0]
    team_name = selected_row["TEAM_NAME"]
    style = TEAM_STYLE.get(team_name, DEFAULT_STYLE)

    if style["slug"]:
        st.image(f"https://i.logocdn.com/nba/current/{style['slug']}.svg", width=90)

    is_bottom10 = bool(selected_row.get("is_bottom_n", False))
    st.markdown(
        f"<div style='border:1px solid #ddd; border-radius:6px; padding:12px; margin-top:8px;'>"
        f"<div style='font-size:11px; letter-spacing:0.05em; color:{TEXT_MUTED}; text-transform:uppercase;'>WinPCT that season</div>"
        f"<div style='font-size:28px; font-weight:700; color:{style['primary']};'>{selected_row['WinPCT']:.3f}</div>"
        f"<div style='font-size:12px; color:{TEXT_MUTED};'>{'Bottom-10 team' if is_bottom10 else 'Not bottom-10'}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "**Bottom-10** = league-wide bottom 10 teams by win %. "
        "**Reached the bracket** = actually appeared in a playoff series "
        "the following season (play-in included)."
    )

# ---------------------------------------------------------------------------
# Dynamic team-color theming -- accents/headers only, data-encoding
# colors (green=good/red=bad) stay fixed regardless of team.
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {{ background-color: {BG}; color: {TEXT}; }}
h1 {{ font-family: 'Source Serif 4', serif !important; color: {TEXT} !important; font-weight: 700 !important; }}
h2, h3 {{ font-family: 'Inter', sans-serif !important; color: {TEXT} !important; font-weight: 600 !important; }}
p, div, span, label {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}
.kicker {{ font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: {TEXT_MUTED}; font-family: 'Inter', sans-serif; }}
.badge {{ background-color: {TEXT}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
.kpi-label {{ font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: {TEXT_MUTED}; }}
.kpi-value {{ font-size: 32px; font-weight: 700; color: {style['primary']}; font-family: 'JetBrains Mono', monospace; }}
[data-testid="stMetricValue"] {{ color: {style['primary']} !important; }}
[data-testid="stSidebar"] {{ background-color: #ffffff; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

n_total = len(case_df)
n_jumped_b = int(case_df["NEXT_target_b_made_bracket"].sum())
jump_rate = n_jumped_b / n_total

st.markdown('<div class="kicker">NBA &middot; 2016-17 TO 2024-25 &middot; BOTTOM-10 TEAM TURNAROUNDS</div>', unsafe_allow_html=True)
st.title("From Bottom-10 to the Playoffs")
st.markdown(
    f'<p style="font-style: italic; color: {TEXT_MUTED};">Bottom-10 teams reach the playoff bracket '
    f'<span class="badge">{jump_rate:.0%}</span> of the time -- here\'s what separates the ones that do.</p>',
    unsafe_allow_html=True,
)

n_jumped_a = int(case_df["NEXT_target_a_top10_conf"].sum())
n_star_added = int(case_df["star_added"].sum())

kpi_cols = st.columns(4)
kpis = [
    ("Bottom-10 team-seasons studied", f"{n_total}"),
    ("Reached top-10 next season", f"{n_jumped_a} ({n_jumped_a/n_total:.0%})"),
    ("Reached the playoff bracket", f"{n_jumped_b} ({n_jumped_b/n_total:.0%})"),
    ("Added a star-tier player", f"{n_star_added} ({n_star_added/n_total:.0%})"),
]
for col, (label, value) in zip(kpi_cols, kpis):
    col.markdown(
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs -- Team Diagnostic first (the actual answer), then What-If, then
# supporting evidence (What Changed), then raw data browse (Case Explorer)
# ---------------------------------------------------------------------------

tab_diagnostic, tab_whatif, tab_changed, tab_explorer = st.tabs(
    ["Team Diagnostic", "What-If", "What Changed", "Case Explorer"]
)

# ---------------------------------------------------------------------------
# Tab 1: Team Diagnostic
# ---------------------------------------------------------------------------

with tab_diagnostic:
    st.markdown(f'<div class="kicker">{team_name.upper()} &middot; {selected_row["season"]}</div>', unsafe_allow_html=True)
    st.subheader("League percentile profile")
    st.caption(
        "0 = worst in the league that season, 1 = best -- sign-corrected so "
        "this is always true (e.g. a low turnover rate shows as a HIGH bar "
        "here, since fewer turnovers is the good outcome, even though the "
        "raw stat's percentile runs the other way). Dashed line = league "
        "median. PACE has no inherent good/bad direction, shown unadjusted."
    )

    diag_rows = []
    for col in LEVEL_FEATURES:
        stat_name = col.replace("_PCTILE", "")
        raw_val = selected_row[col]
        bound_row = delta_summary_df[delta_summary_df["stat"] == stat_name]
        higher_is_better = bool(bound_row.iloc[0]["higher_is_better"]) if len(bound_row) > 0 else True
        display_val = raw_val if higher_is_better else (1 - raw_val)
        diag_rows.append({"stat": stat_name, "percentile": display_val})
    diag_df = pd.DataFrame(diag_rows).sort_values("percentile")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        y=diag_df["stat"], x=diag_df["percentile"], orientation="h",
        marker_color=[COLOR_STAYED if v < 0.5 else COLOR_JUMPED for v in diag_df["percentile"]],
        text=[f"{v:.2f}" for v in diag_df["percentile"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    fig3.add_vline(x=0.5, line_dash="dash", line_color=TEXT_MUTED)
    fig3.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT,
        height=540, margin=dict(l=10, r=30, t=10, b=10), xaxis_range=[0, 1.08],
        xaxis=dict(gridcolor="#e5e5e2"), yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    if is_bottom10:
        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="kpi-label">Reached top-10 next season</div>'
            f'<div class="kpi-value">{selected_row.get("NEXT_target_a_top10_conf", "N/A")}</div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="kpi-label">Reached the playoff bracket</div>'
            f'<div class="kpi-value">{selected_row.get("NEXT_target_b_made_bracket", "N/A")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Gap to a typical playoff team")
    st.caption(
        "Current percentile vs. the average ACTUAL playoff team over the last 3 seasons "
        "(2022-23 through 2024-25). Green = realistically closeable in one season "
        "(within the biggest single-season swing any bottom-10 team has actually made "
        "on that stat); red = beyond that historical ceiling."
    )

    gap_rows = []
    for _, brow in benchmark_df.iterrows():
        stat = brow["stat"]
        if stat == "PACE":
            continue
        level_col = f"{stat}_PCTILE"
        if level_col not in selected_row.index:
            continue
        bound_row = delta_summary_df[delta_summary_df["stat"] == stat]
        if len(bound_row) == 0:
            continue
        higher_is_better = bool(bound_row.iloc[0]["higher_is_better"])
        current_val = selected_row[level_col]
        target_val = brow["playoff_avg_pctile"]
        raw_diff = target_val - current_val
        gap = raw_diff if higher_is_better else -raw_diff
        raw_p90 = float(bound_row.iloc[0]["p90"])
        raw_p10 = float(bound_row.iloc[0]["p10"])
        max_realistic = raw_p90 if higher_is_better else -raw_p10
        gap_rows.append({
            "stat": stat, "current": current_val, "playoff_avg": target_val, "gap": gap,
            "max_realistic_1yr_gain": max_realistic, "closeable_in_1_season": gap <= max_realistic,
        })

    gap_df = pd.DataFrame(gap_rows).sort_values("gap", ascending=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=gap_df["stat"], x=gap_df["gap"], orientation="h",
        marker_color=[COLOR_JUMPED if c else COLOR_STAYED for c in gap_df["closeable_in_1_season"]],
        text=[f"{v:+.2f}" for v in gap_df["gap"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    fig4.add_vline(x=0, line_color=TEXT_MUTED)
    fig4.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT,
        height=540, margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(gridcolor="#e5e5e2", title="Percentile points behind playoff average"),
        yaxis=dict(showgrid=False), showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Full numbers"):
        st.dataframe(
            gap_df.rename(columns={
                "stat": "Stat", "current": "Current pctile", "playoff_avg": "Playoff avg pctile",
                "gap": "Gap", "max_realistic_1yr_gain": "Biggest 1-season gain seen historically",
                "closeable_in_1_season": "Realistic in 1 season",
            }),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# Tab 2: What-If
# ---------------------------------------------------------------------------

with tab_whatif:
    if model is None:
        st.error(f"Couldn't load the model from GitHub. ({model_load_error})")
    else:
        st.markdown(f'<div class="kicker">{team_name.upper()} &middot; {selected_row["season"]}</div>', unsafe_allow_html=True)
        st.subheader("Scenario builder")
        st.caption(
            "Baseline assumes no further improvement (all sliders at zero). Every slider "
            "is bounded by the actual 10th-90th percentile of year-over-year change seen "
            "across 90 bottom-10 team-seasons -- no impossible improvements. Uses the "
            "tautology-free model (excludes NET/OFF/DEF/CLUTCH_NET rating deltas, which "
            "mechanically restate the outcome rather than something a front office acts on)."
        )

        def build_input_row(deltas, rookies, star_added_val):
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

        baseline_row = build_input_row({}, {}, False)
        baseline_prob = model.predict_proba(baseline_row)[0, 1]

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
                    col.replace("ROOKIES_", "").replace("_", " ").title(), min_value=lo, max_value=hi, value=0.0,
                )

            st.markdown("**Component stat changes** (year-over-year, percentile points)")
            st.caption("For a few stats, moving the slider NEGATIVE is the improvement direction (opponent shooting, turnovers, points allowed) -- marked below.")
            delta_inputs = {}
            for col in DELTA_FEATURES:
                stat_name = col.replace("DELTA_", "").replace("_PCTILE", "")
                bounds_row = delta_summary_df[delta_summary_df["stat"] == stat_name]
                if len(bounds_row) == 0:
                    continue
                lo, hi = float(bounds_row.iloc[0]["p10"]), float(bounds_row.iloc[0]["p90"])
                higher_is_better = bool(bounds_row.iloc[0]["higher_is_better"])
                label = stat_name if higher_is_better else f"{stat_name} (lower = better)"
                delta_inputs[col] = st.slider(label, min_value=lo, max_value=hi, value=0.0)

        scenario_row = build_input_row(delta_inputs, rookie_inputs, star_added_input)
        scenario_prob = model.predict_proba(scenario_row)[0, 1]

        with col_result:
            st.markdown(
                f'<div class="kpi-label">Playoff bracket probability</div>'
                f'<div class="kpi-value" style="font-size:48px;">{scenario_prob:.0%}</div>'
                f'<div style="color:{COLOR_JUMPED if scenario_prob >= baseline_prob else COLOR_STAYED}; font-weight:600;">'
                f'{(scenario_prob - baseline_prob):+.0%} vs. no change</div>'
                f'<div class="kpi-label" style="margin-top:16px;">Baseline (no improvement)</div>'
                f'<div style="font-size:20px; font-family:JetBrains Mono, monospace;">{baseline_prob:.0%}</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Tab 3: What Changed
# ---------------------------------------------------------------------------

with tab_changed:
    st.subheader("Which stats actually separated the teams that turned around")
    st.caption(
        "Sign-corrected gap between teams that reached the bracket and teams that "
        "didn't -- positive always means 'genuinely helped,' accounting for stats "
        "where lower is the good direction. PACE excluded (no inherent good "
        "direction). NET/OFF/DEF/CLUTCH_NET rating excluded (mechanically restate "
        "the outcome rather than explain it)."
    )

    TAUTOLOGICAL_STATS = {"NET_RATING", "OFF_RATING", "DEF_RATING", "CLUTCH_NET_RATING"}
    chart_df = delta_summary_df[~delta_summary_df["stat"].isin(TAUTOLOGICAL_STATS)]
    sorted_df = chart_df.sort_values("improvement", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df["stat"], x=sorted_df["improvement"], orientation="h",
        marker_color=["#999999" if s == "PACE" else (COLOR_JUMPED if v > 0 else COLOR_STAYED)
                      for s, v in zip(sorted_df["stat"], sorted_df["improvement"])],
        text=[f"{v:+.2f}" for v in sorted_df["improvement"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11),
    ))
    fig.add_vline(x=0, line_color=TEXT_MUTED)
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT, height=500,
        xaxis=dict(gridcolor="#e5e5e2", title="Improvement (positive = genuinely helped)"),
        yaxis=dict(showgrid=False), margin=dict(l=10, r=40, t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Realistic range of improvement, per stat")
    st.caption("Actual 10th-90th percentile of year-over-year change across all 90 bottom-10 team-seasons.")

    selected_stat = st.selectbox("Stat", options=chart_df["stat"].tolist())
    stat_row = chart_df[chart_df["stat"] == selected_stat].iloc[0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[stat_row["p10"], stat_row["p90"]], y=[0, 0], mode="lines",
        line=dict(color=TEXT_MUTED, width=6), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=[stat_row["median"]], y=[0], mode="markers",
        marker=dict(size=16, color=style["primary"]), name="Historical median",
    ))
    fig2.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT, height=140,
        yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=f"Year-over-year change in {selected_stat}", gridcolor="#e5e5e2"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="kpi-label">10th percentile</div><div class="kpi-value" style="font-size:22px;">{stat_row["p10"]:.3f}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-label">Median</div><div class="kpi-value" style="font-size:22px;">{stat_row["median"]:.3f}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-label">90th percentile</div><div class="kpi-value" style="font-size:22px;">{stat_row["p90"]:.3f}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 4: Case Explorer
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
