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

# Plain-English labels -- nobody outside this project knows what
# "OPP_EFG_PCT" means. Used everywhere a stat name is displayed.
STAT_LABELS = {
    "NET_RATING": "Net Rating",
    "CLUTCH_NET_RATING": "Clutch-Time Net Rating",
    "OFF_RATING": "Offensive Rating",
    "DEF_RATING": "Defensive Rating (Points Allowed)",
    "PACE": "Pace of Play",
    "AST_RATIO": "Ball Movement (Assist Ratio)",
    "TM_TOV_PCT": "Turnover Rate",
    "FF_TM_TOV_PCT": "Turnover Rate",
    "EFG_PCT": "Shooting Efficiency",
    "FF_EFG_PCT": "Shooting Efficiency",
    "FTA_RATE": "Free Throw Rate",
    "OREB_PCT": "Offensive Rebounding",
    "OPP_EFG_PCT": "Opponent Shooting Efficiency Allowed",
    "OPP_FTA_RATE": "Opponent Free Throw Rate Allowed",
    "OPP_TOV_PCT": "Turnovers Forced on Defense",
    "OPP_OREB_PCT": "Offensive Rebounds Allowed",
}

def pretty(stat_code: str) -> str:
    return STAT_LABELS.get(stat_code, stat_code.replace("_", " ").title())

# EFG_PCT/FF_EFG_PCT and TM_TOV_PCT/FF_TM_TOV_PCT are literally the same
# underlying stat pulled from two different nba_api endpoints -- showing
# "Shooting Efficiency" twice in a dropdown looks like a bug, not a
# feature. Keep only one of each pair in anything user-facing.
DUPLICATE_STATS = {"FF_EFG_PCT", "FF_TM_TOV_PCT"}

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
    ROOKIE_FEATURES = [c for c in FEATURE_COLS_TRIMMED if c.startswith("ROOKIES_") or c.startswith("CURRENT_ROOKIES_")]
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
    league_rank = int(selected_row["LEAGUE_RANK"])
    if league_rank <= 12:
        rank_color = COLOR_JUMPED
    elif league_rank >= 21:
        rank_color = COLOR_STAYED
    else:
        rank_color = "#c9a227"  # yellow -- middle of the pack

    st.markdown(
        f"<div style='border:1px solid #ddd; border-radius:6px; padding:12px; margin-top:8px;'>"
        f"<div style='font-size:11px; letter-spacing:0.05em; color:{TEXT_MUTED}; text-transform:uppercase;'>WinPCT that season</div>"
        f"<div style='font-size:28px; font-weight:700; color:{style['primary']};'>{selected_row['WinPCT']:.3f}</div>"
        f"<div style='font-size:12px; color:{TEXT_MUTED}; margin-top:4px;'>League rank</div>"
        f"<div style='font-size:18px; font-weight:700; color:{rank_color};'>{league_rank} of 30</div>"
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
.kpi-value {{ font-size: 32px; font-weight: 700; color: {style['primary']}; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }}
.kpi-value-fixed {{ font-size: 32px; font-weight: 700; color: {TEXT}; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }}
[data-testid="stMetricValue"] {{ color: {style['primary']} !important; }}
[data-testid="stSidebar"] {{ background-color: #ffffff; }}
[data-testid="stSidebar"] [data-testid="stImage"] {{ filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1)); }}
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
        f'<div class="kpi-label">{label}</div><div class="kpi-value-fixed">{value}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs -- Team Diagnostic first (the actual answer), then What-If, then
# supporting evidence (What Changed), then raw data browse (Case Explorer)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tabs -- Turnaround Playbook first (the actual differentiator: which teams
# pulled this off and how), then What-If, then Team Lookup (any team/season,
# demoted since it's the generic view), then Case Explorer.
# ---------------------------------------------------------------------------

tab_playbook, tab_whatif, tab_diagnostic, tab_explorer = st.tabs(
    ["Turnaround Playbook", "What-If", "Team Lookup", "Case Explorer"]
)

# ---------------------------------------------------------------------------
# Tab 0: Turnaround Playbook -- THE differentiator. Not "browse any team's
# stats" (generic) -- specifically which of the 90 bottom-10 teams reached
# the playoffs, what changed for them vs. teams that stayed stuck, and the
# actual team-season examples behind each stat (biggest jump, smallest
# jump, average), not just an abstract aggregate number.
# ---------------------------------------------------------------------------

with tab_playbook:
    jumped_df = case_df[case_df["NEXT_target_b_made_bracket"] == True].copy()
    stayed_df = case_df[case_df["NEXT_target_b_made_bracket"] == False].copy()

    st.subheader(f"{len(jumped_df)} teams turned a bottom-10 season into a playoff bracket appearance")
    st.caption(
        "Out of 90 bottom-10 team-seasons (2016-17 to 2024-25). Everything "
        "below is built from these teams specifically -- not a generic "
        "league-wide stat browser."
    )

    TAUTOLOGICAL_STATS = {"NET_RATING", "OFF_RATING", "DEF_RATING", "CLUTCH_NET_RATING"}
    chart_df = delta_summary_df[
        ~delta_summary_df["stat"].isin(TAUTOLOGICAL_STATS | DUPLICATE_STATS)
    ].copy()
    chart_df["label"] = chart_df["stat"].apply(pretty)
    sorted_df = chart_df.sort_values("improvement", ascending=True)

    max_abs = sorted_df["improvement"].abs().max()
    fig_playbook = go.Figure()
    fig_playbook.add_trace(go.Bar(
        y=sorted_df["label"], x=sorted_df["improvement"], orientation="h",
        marker=dict(
            color=sorted_df["improvement"],
            colorscale=[[0, COLOR_STAYED], [0.5, "#f0ede6"], [1, COLOR_JUMPED]],
            cmin=-max_abs, cmax=max_abs,
            line=dict(width=0),
        ),
        text=[f"{v:+.2f}" for v in sorted_df["improvement"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=TEXT),
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}<extra></extra>",
    ))
    fig_playbook.add_vline(x=0, line_color=TEXT_MUTED, line_width=1.5)
    fig_playbook.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT, height=480,
        xaxis=dict(gridcolor="#e5e5e2", title="Improvement (positive = genuinely helped teams reach the bracket)",
                    zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        margin=dict(l=10, r=50, t=10, b=10), showlegend=False, bargap=0.35,
    )
    st.plotly_chart(fig_playbook, use_container_width=True)
    st.caption(
        "Pace of Play excluded (no inherent good direction). Net/Offensive/"
        "Defensive/Clutch Rating excluded -- they mechanically restate the "
        "outcome (better record = better rating) rather than explain what "
        "drove it."
    )

    st.markdown("---")
    st.subheader("Which teams actually drove each stat")

    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:15px; font-weight:600; margin-bottom:6px;">'
            f'&#128269; Pick a stat from the chart above to drill into the teams behind it</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "See who improved it the most, who barely moved it at all (and "
            "still made the bracket anyway), and where the typical successful "
            "team landed."
        )
        stat_options = chart_df["stat"].tolist()
        playbook_stat = st.selectbox(
            "Stat to explore", options=stat_options, format_func=pretty, key="playbook_stat",
            label_visibility="collapsed",
        )
    stat_bound_row = delta_summary_df[delta_summary_df["stat"] == playbook_stat].iloc[0]
    higher_is_better = bool(stat_bound_row["higher_is_better"])
    delta_col = f"DELTA_{playbook_stat}_PCTILE"

    jumped_df["_signed_delta"] = jumped_df[delta_col] if higher_is_better else -jumped_df[delta_col]
    jumped_sorted = jumped_df.sort_values("_signed_delta", ascending=False)

    biggest = jumped_sorted.iloc[0]
    smallest = jumped_sorted.iloc[-1]
    avg_jump = jumped_df["_signed_delta"].mean()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="kpi-label">&#9650; Biggest jump</div>'
        f'<div class="kpi-value-fixed" style="font-size:26px; color:{COLOR_JUMPED};">{biggest["_signed_delta"]:+.2f}</div>'
        f'<div style="font-size:13px; color:{TEXT_MUTED};">{biggest["TEAM_NAME"]} &middot; {biggest["season"]}</div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="kpi-label">&#9679; Average jump</div>'
        f'<div class="kpi-value-fixed" style="font-size:26px;">{avg_jump:+.2f}</div>'
        f'<div style="font-size:13px; color:{TEXT_MUTED};">teams that reached the bracket</div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="kpi-label">&#9660; Smallest jump</div>'
        f'<div class="kpi-value-fixed" style="font-size:26px; color:{COLOR_STAYED};">{smallest["_signed_delta"]:+.2f}</div>'
        f'<div style="font-size:13px; color:{TEXT_MUTED};">{smallest["TEAM_NAME"]} &middot; {smallest["season"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"**Where each team started vs. where they ended up** &mdash; {pretty(playbook_stat)}")
    st.caption(
        "League percentile before (hollow) and after (filled) the turnaround "
        "season, one line per team, sorted by improvement. Both ends already "
        "account for direction, so a line sloping UP always means genuine "
        "improvement, regardless of the underlying stat. Biggest/average/"
        "smallest jump above are always computed across all teams, "
        "regardless of how many are shown in the chart below."
    )

    next_col = f"NEXT_{playbook_stat}_PCTILE"
    level_col = f"{playbook_stat}_PCTILE"
    jumped_sorted["_before"] = jumped_sorted[level_col] if higher_is_better else (1 - jumped_sorted[level_col])
    jumped_sorted["_after"] = jumped_sorted[next_col] if higher_is_better else (1 - jumped_sorted[next_col])
    jumped_sorted["_team_label"] = jumped_sorted["TEAM_NAME"] + " (" + jumped_sorted["season"] + ")"

    show_n = st.slider(
        "Show top N teams (by improvement on this stat)",
        min_value=5, max_value=len(jumped_sorted), value=min(10, len(jumped_sorted)),
    )
    slope_df = jumped_sorted.head(show_n)
    slope_order = slope_df.sort_values("_signed_delta", ascending=True)["_team_label"].tolist()

    fig_slope = go.Figure()
    for _, row in slope_df.iterrows():
        line_color = COLOR_JUMPED if row["_after"] >= row["_before"] else COLOR_STAYED
        fig_slope.add_trace(go.Scatter(
            x=[row["_before"], row["_after"]], y=[row["_team_label"]] * 2,
            mode="lines", line=dict(color=line_color, width=2), showlegend=False,
            hoverinfo="skip",
        ))
    fig_slope.add_trace(go.Scatter(
        x=slope_df["_before"], y=slope_df["_team_label"], mode="markers",
        marker=dict(size=9, color=BG, line=dict(color=TEXT_MUTED, width=1.5)),
        name="Before", hovertemplate="Before: %{x:.2f}<extra></extra>",
    ))
    fig_slope.add_trace(go.Scatter(
        x=slope_df["_after"], y=slope_df["_team_label"], mode="markers",
        marker=dict(size=9, color=style["primary"]),
        name="After", hovertemplate="After: %{x:.2f}<extra></extra>",
    ))
    fig_slope.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT,
        height=max(380, 30 * len(slope_df)),
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(gridcolor="#e5e5e2", title="League percentile (0=worst, 1=best)", range=[0, 1]),
        yaxis=dict(showgrid=False, categoryarray=slope_order, categoryorder="array"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_slope, use_container_width=True)

    st.markdown("---")
    st.subheader("Turnaround stories")
    st.caption("Every team that went bottom-10 to the playoff bracket. Sortable, filterable.")

    story_cols = ["TEAM_NAME", "season", "WinPCT", "star_added", "ROOKIES_on_roster_count",
                  "ROOKIES_max_minutes_per_game", "ROOKIES_avg_minutes_per_game"]
    if "predicted_prob_target_b" in jumped_df.columns:
        story_cols.append("predicted_prob_target_b")
    story_display = jumped_df[story_cols].sort_values("season", ascending=False).copy()
    for c in ["ROOKIES_max_minutes_per_game", "ROOKIES_avg_minutes_per_game"]:
        if c in story_display.columns:
            story_display[c] = story_display[c].round(0)
    story_display = story_display.rename(columns={
        "TEAM_NAME": "Team", "season": "Season", "WinPCT": "Win %", "star_added": "Star Added",
        "ROOKIES_on_roster_count": "# Rookies", "ROOKIES_max_minutes_per_game": "Top Rookie Min/Game",
        "ROOKIES_avg_minutes_per_game": "Avg Rookie Min/Game",
        "predicted_prob_target_b": "Predicted Playoff Prob.",
    })
    st.dataframe(story_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Team Lookup (demoted -- generic any-team/any-season reference view)
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
        if stat_name in DUPLICATE_STATS:
            continue
        raw_val = selected_row[col]
        bound_row = delta_summary_df[delta_summary_df["stat"] == stat_name]
        higher_is_better = bool(bound_row.iloc[0]["higher_is_better"]) if len(bound_row) > 0 else True
        display_val = raw_val if higher_is_better else (1 - raw_val)
        diag_rows.append({"stat": stat_name, "label": pretty(stat_name), "percentile": display_val})
    diag_df = pd.DataFrame(diag_rows).sort_values("percentile")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        y=diag_df["label"], x=diag_df["percentile"], orientation="h",
        marker=dict(
            color=diag_df["percentile"],
            colorscale=[[0, COLOR_STAYED], [0.5, "#f0ede6"], [1, COLOR_JUMPED]],
            cmin=0, cmax=1, line=dict(width=0),
        ),
        text=[f"{v:.2f}" for v in diag_df["percentile"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=TEXT),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
    ))
    fig3.add_vline(x=0.5, line_dash="dash", line_color=TEXT_MUTED)
    fig3.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT,
        height=520, margin=dict(l=10, r=30, t=10, b=10), xaxis_range=[0, 1.08],
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
        if stat == "PACE" or stat in DUPLICATE_STATS:
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
            "stat": stat, "label": pretty(stat), "current": current_val, "playoff_avg": target_val, "gap": gap,
            "max_realistic_1yr_gain": max_realistic, "closeable_in_1_season": gap <= max_realistic,
        })

    gap_df = pd.DataFrame(gap_rows).sort_values("gap", ascending=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=gap_df["label"], x=gap_df["gap"], orientation="h",
        marker=dict(
            color=[COLOR_JUMPED if c else COLOR_STAYED for c in gap_df["closeable_in_1_season"]],
            line=dict(width=0),
        ),
        text=[f"{v:+.2f}" for v in gap_df["gap"]], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=TEXT),
        hovertemplate="<b>%{y}</b><br>Gap: %{x:+.2f}<extra></extra>",
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
            gap_df.drop(columns=["stat"]).rename(columns={
                "label": "Stat", "current": "Current pctile", "playoff_avg": "Playoff avg pctile",
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

        def build_input_row(deltas, star_added_val):
            row = {}
            for col in LEVEL_FEATURES:
                row[col] = selected_row[col]
            for col in DELTA_FEATURES:
                row[col] = deltas.get(col, 0.0)
            for col in ROOKIE_FEATURES:
                # Fixed, known fact about the bottom-10 season itself --
                # not a what-if assumption, so always pulled from the
                # team's actual data regardless of scenario.
                row[col] = selected_row.get(col, 0.0)
            if "star_added" in FEATURE_COLS_TRIMMED:
                row["star_added"] = float(star_added_val)
            if "is_bottom_n" in FEATURE_COLS_TRIMMED:
                row["is_bottom_n"] = float(selected_row.get("is_bottom_n", True))
            if "WinPCT" in FEATURE_COLS_TRIMMED:
                row["WinPCT"] = selected_row["WinPCT"]
            return pd.DataFrame([row])[FEATURE_COLS_TRIMMED]

        baseline_row = build_input_row({}, False)
        baseline_prob = model.predict_proba(baseline_row)[0, 1]

        col_sliders, col_result = st.columns([2, 1])

        with col_sliders:
            st.markdown("**Star addition**")
            star_added_input = st.checkbox("Assume a star-tier player is added this offseason")

            st.markdown("**Rookies already on the roster** (fixed -- known fact about this season, not adjustable)")
            rookie_facts = []
            for col in ROOKIE_FEATURES:
                label = col.replace("CURRENT_ROOKIES_", "").replace("_", " ").title()
                val = selected_row.get(col, 0)
                val_display = f"{val:.0f}" if pd.notna(val) else "N/A"
                rookie_facts.append(f"**{val_display}** {label}")
            st.caption(" &nbsp;&middot;&nbsp; ".join(rookie_facts))

            st.markdown("**Component stat changes** (year-over-year)")
            st.caption("For stats where lower is actually better (opponent shooting, turnovers, points allowed), moving the slider left is the improvement direction -- marked below.")
            delta_inputs = {}
            for col in DELTA_FEATURES:
                stat_name = col.replace("DELTA_", "").replace("_PCTILE", "")
                if stat_name in DUPLICATE_STATS:
                    continue
                bounds_row = delta_summary_df[delta_summary_df["stat"] == stat_name]
                if len(bounds_row) == 0:
                    continue
                lo, hi = float(bounds_row.iloc[0]["p10"]), float(bounds_row.iloc[0]["p90"])
                higher_is_better = bool(bounds_row.iloc[0]["higher_is_better"])
                label = pretty(stat_name) if higher_is_better else f"{pretty(stat_name)} (lower = better)"
                delta_inputs[col] = st.slider(label, min_value=lo, max_value=hi, value=0.0)

        scenario_row = build_input_row(delta_inputs, star_added_input)
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

    st.markdown("---")
    st.subheader(f"Realistic range for {playbook_stat}")
    st.caption("Actual 10th-90th percentile of year-over-year change across all 90 bottom-10 team-seasons (not just the ones that reached the bracket).")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[stat_bound_row["p10"], stat_bound_row["p90"]], y=[0, 0], mode="lines",
        line=dict(color=TEXT_MUTED, width=6), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=[stat_bound_row["median"]], y=[0], mode="markers",
        marker=dict(size=16, color=style["primary"]), name="Historical median",
    ))
    fig2.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_family="Inter", font_color=TEXT, height=140,
        yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=f"Year-over-year change in {playbook_stat}", gridcolor="#e5e5e2"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    rc1, rc2, rc3 = st.columns(3)
    rc1.markdown(f'<div class="kpi-label">10th percentile</div><div class="kpi-value-fixed" style="font-size:22px;">{stat_bound_row["p10"]:.3f}</div>', unsafe_allow_html=True)
    rc2.markdown(f'<div class="kpi-label">Median</div><div class="kpi-value-fixed" style="font-size:22px;">{stat_bound_row["median"]:.3f}</div>', unsafe_allow_html=True)
    rc3.markdown(f'<div class="kpi-label">90th percentile</div><div class="kpi-value-fixed" style="font-size:22px;">{stat_bound_row["p90"]:.3f}</div>', unsafe_allow_html=True)

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
        "star_added", "ROOKIES_on_roster_count",
        "ROOKIES_max_minutes_per_game", "ROOKIES_avg_minutes_per_game",
    ]
    if "predicted_prob_target_b" in filtered_df.columns:
        display_cols.append("predicted_prob_target_b")

    explorer_display = filtered_df[display_cols].sort_values("season", ascending=False).copy()
    for c in ["ROOKIES_max_minutes_per_game", "ROOKIES_avg_minutes_per_game"]:
        if c in explorer_display.columns:
            explorer_display[c] = explorer_display[c].round(0)
    explorer_display = explorer_display.rename(columns={
        "TEAM_NAME": "Team", "season": "Season", "WinPCT": "Win %",
        "NEXT_target_a_top10_conf": "Reached Top-10", "NEXT_target_b_made_bracket": "Reached Bracket",
        "star_added": "Star Added", "ROOKIES_on_roster_count": "# Rookies",
        "ROOKIES_max_minutes_per_game": "Top Rookie Min/Game",
        "ROOKIES_avg_minutes_per_game": "Avg Rookie Min/Game",
        "predicted_prob_target_b": "Predicted Playoff Prob.",
    })
    st.dataframe(explorer_display, use_container_width=True, hide_index=True)
