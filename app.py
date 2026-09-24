"""
MAR-RES — AIS-Driven Early Warning and Digital Twin for Resilient Port Operations
Streamlit decision-support prototype.

IMPORTANT: this file trains NOTHING. Every analytical model, threshold, preprocessing
step, and simulation parameter is loaded from artifacts produced by the Colab analysis
notebook (data/, config/). If those artifacts are missing, pages show an explicit
"artifact not found" message instead of fabricating output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "engine"))
import simulation_engine as engine  # noqa: E402

# ------------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

REQUIRED_ARTIFACTS = {
    "cleaned_singapore": DATA_DIR / "cleaned_singapore.csv",
    "disruption_periods": DATA_DIR / "disruption_periods.csv",
    "yearly_summary": DATA_DIR / "yearly_summary.csv",
    "correlation_results": DATA_DIR / "correlation_results.csv",
    "scenario_results": DATA_DIR / "scenario_results.csv",
    "normal_vs_disruption": DATA_DIR / "normal_vs_disruption.csv",
    "model_summary": CONFIG_DIR / "model_summary.json",
    "simulation_config": CONFIG_DIR / "simulation_config.json",
}
OPTIONAL_ARTIFACTS = {
    "recovery_analysis": DATA_DIR / "recovery_analysis.csv",
    "recovery_strategy_results": DATA_DIR / "recovery_strategy_results.csv",
    "descriptive_stats": DATA_DIR / "singapore_descriptive_stats.csv",
}

ARTIFACT_MISSING_MSG = "Required analytical artifact not found. Please run the Colab analytical engine first."

# ------------------------------------------------------------------------------------
# PAGE CONFIG + VISUAL THEME (clean white / navy / teal — "maritime control tower")
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="MAR-RES | Port Resilience Prototype",
    page_icon="\U0001F6A2",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1F3A"
NAVY_LIGHT = "#13315C"
TEAL = "#0E8388"
TEAL_LIGHT = "#2FBF9F"
GREY_BG = "#F5F7FA"
RED = "#C0392B"
AMBER = "#B9770E"

st.markdown(f"""
<style>
    .stApp {{ background-color: {GREY_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #E8ECF1 !important; }}
    h1, h2, h3 {{ color: {NAVY}; font-family: "Segoe UI", "Helvetica Neue", sans-serif; }}
    .kpi-card {{
        background: white; border-radius: 8px; padding: 18px 20px;
        border-left: 4px solid {TEAL}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .kpi-label {{ font-size: 0.8rem; color: #5B6B82; text-transform: uppercase; letter-spacing: 0.03em; }}
    .kpi-value {{ font-size: 1.7rem; color: {NAVY}; font-weight: 700; margin-top: 2px; }}
    .kpi-sub {{ font-size: 0.78rem; color: #7A8699; margin-top: 2px; }}
    .status-badge {{
        display: inline-block; padding: 4px 14px; border-radius: 14px; font-weight: 600;
        font-size: 0.85rem; letter-spacing: 0.02em;
    }}
    .badge-normal {{ background:#E4F5EE; color:#0E8388; }}
    .badge-alert {{ background:#FDF1DC; color:#B9770E; }}
    .badge-disruption {{ background:#FBE7E5; color:#C0392B; }}
    .flow-box {{
        background: {NAVY}; color: white; border-radius: 6px; padding: 10px 6px;
        text-align: center; font-size: 0.82rem; font-weight: 600; line-height: 1.25;
    }}
    .note-box {{
        background: #EEF4FB; border-left: 3px solid {NAVY_LIGHT}; padding: 10px 14px;
        font-size: 0.88rem; color: #2E3B4E; border-radius: 4px;
    }}
    .limit-box {{
        background: #FBE7E5; border-left: 3px solid {RED}; padding: 10px 14px;
        font-size: 0.85rem; color: #5B2422; border-radius: 4px;
    }}
    div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------
# CACHED LOADERS  (data: @st.cache_data | engine/config objects: @st.cache_resource)
# ------------------------------------------------------------------------------------
def artifact_exists(path: Path) -> bool:
    return path.exists()


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_resource(show_spinner=False)
def load_engine_config():
    """Load the fitted model + simulation config once per session (not retrained)."""
    model_summary, sim_config = engine.load_artifacts(str(CONFIG_DIR))
    return model_summary, sim_config


def check_required_artifacts():
    missing = [name for name, p in REQUIRED_ARTIFACTS.items() if not artifact_exists(p)]
    return missing


# ------------------------------------------------------------------------------------
# SMALL UI HELPERS
# ------------------------------------------------------------------------------------
def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text, kind):
    cls = {"normal": "badge-normal", "alert": "badge-alert", "disruption": "badge-disruption"}[kind]
    st.markdown(f'<span class="status-badge {cls}">{text}</span>', unsafe_allow_html=True)


def flow_diagram(steps):
    cols = st.columns(len(steps) * 2 - 1)
    for i, step in enumerate(steps):
        with cols[i * 2]:
            st.markdown(f'<div class="flow-box">{step}</div>', unsafe_allow_html=True)
        if i * 2 + 1 < len(cols):
            with cols[i * 2 + 1]:
                st.markdown('<div style="text-align:center; padding-top:10px; color:#7A8699;">&#8594;</div>',
                            unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def limitation(text):
    st.markdown(f'<div class="limit-box"><b>Limitation:</b> {text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# ------------------------------------------------------------------------------------
def page_overview(df, disruption_df, sim_config):
    st.title("MAR-RES")
    st.markdown("#### AIS-Driven Early Warning and Digital Twin for Resilient Port Operations")
    st.caption("Singapore Port Case | AIS-based Resilience Analytics")
    st.write("")

    n_obs = len(df)
    period = f"{int(df['Year'].min())}\u2013{int(df['Year'].max())}"
    avg_wait = df["avgwaiting_time"].mean()
    avg_service = df["avgservice_time"].mean()
    avg_activity = df["count_entrance"].mean()
    n_disruption_weeks = int(disruption_df["disruption_flag"].sum()) if "disruption_flag" in disruption_df else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Observation Period", period, f"{n_obs} weekly observations")
    with c2:
        kpi_card("Avg Waiting Time", f"{avg_wait:.1f} hrs", "Primary KPI, all weeks")
    with c3:
        kpi_card("Avg Service Time", f"{avg_service:.1f} hrs", "All weeks")

    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card("Avg Vessel Activity", f"{avg_activity:.0f}", "Entrances / week")
    with c5:
        kpi_card("Detected Disruption Weeks", f"{n_disruption_weeks}", "Robust z-score / MAD method (Colab)")
    with c6:
        method = sim_config.get("final_disruption_method", "n/a")
        kpi_card("Detection Method", method.replace("_", " ").title(), "Chosen & validated in Colab")

    st.write("")
    st.subheader("Analytical Flow")
    flow_diagram(["AIS DATA", "ANALYTICS", "EARLY WARNING", "SCENARIO SIMULATION", "RECOVERY DECISION"])

    st.write("")
    note("This prototype is a <b>decision-support system</b>. All models, thresholds, and "
         "simulation parameters were computed once in Google Colab and are only read here — "
         "Streamlit performs no training and derives no new statistical thresholds.")


# ------------------------------------------------------------------------------------
# PAGE 2 — PORT MONITOR
# ------------------------------------------------------------------------------------
def page_port_monitor(df, disruption_df):
    st.title("Port Monitor")
    st.caption("Actual weekly operational data — cleaned_singapore.csv")

    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_years = st.multiselect("Year", years, default=years)
    with c2:
        week_min, week_max = int(df["Week"].min()), int(df["Week"].max())
        week_range = st.slider("Week range (within selected years)", week_min, week_max, (week_min, week_max))

    dff = df[df["Year"].isin(selected_years) & df["Week"].between(*week_range)].copy()
    if dff.empty:
        st.warning("No observations in the selected filter range.")
        return

    merged = dff.merge(
        disruption_df[["period_label", "disruption_flag"]], on="period_label", how="left"
    )
    merged["disruption_flag"] = merged["disruption_flag"].fillna(False)

    tabs = st.tabs(["Waiting Time", "Service Time", "Vessel Entrance", "Port Calls"])
    series_map = {
        "Waiting Time": ("avgwaiting_time", TEAL),
        "Service Time": ("avgservice_time", NAVY_LIGHT),
        "Vessel Entrance": ("count_entrance", "#7C3AED"),
        "Port Calls": ("numofcall", "#B9770E"),
    }
    for tab, (label, (col, color)) in zip(tabs, series_map.items()):
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged["period_label"], y=merged[col], mode="lines",
                                      line=dict(color=color, width=2), name=label))
            disrupt_pts = merged[merged["disruption_flag"]]
            if not disrupt_pts.empty:
                fig.add_trace(go.Scatter(x=disrupt_pts["period_label"], y=disrupt_pts[col], mode="markers",
                                          marker=dict(color=RED, size=8), name="Detected disruption signal"))
            fig.update_layout(height=380, margin=dict(t=20, b=20), plot_bgcolor="white",
                               xaxis_title="Week", yaxis_title=label)
            st.plotly_chart(fig, width="stretch")

    st.write("")
    st.subheader("Selected Period Snapshot (latest week in filter)")
    latest = merged.sort_values("t").iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Waiting Time", f"{latest['avgwaiting_time']:.1f} hrs", latest["period_label"])
    with c2:
        kpi_card("Service Time", f"{latest['avgservice_time']:.1f} hrs", latest["period_label"])
    with c3:
        kpi_card("Vessel Entrance", f"{latest['count_entrance']:.0f}", latest["period_label"])
    with c4:
        if latest["disruption_flag"]:
            st.markdown("**Disruption Status**")
            status_badge("Detected disruption signal", "disruption")
        else:
            st.markdown("**Disruption Status**")
            status_badge("Normal", "normal")


# ------------------------------------------------------------------------------------
# PAGE 3 — DISRUPTION DETECTION
# ------------------------------------------------------------------------------------
def page_disruption_detection(df, disruption_df, recovery_df, sim_config):
    st.title("Disruption Detection")
    note(f"Disruption flag uses the method already finalized in Colab: "
         f"<b>{sim_config.get('final_disruption_method', 'n/a')}</b> "
         f"(threshold |z| &gt; {sim_config.get('disruption_threshold_robust_z', 'n/a')}). "
         f"No threshold is recomputed in this app.")

    merged = df.merge(disruption_df[["period_label", "disruption_flag", "deviation",
                                      "baseline_waiting_time_ma12"]], on="period_label", how="left")
    merged["disruption_flag"] = merged["disruption_flag"].fillna(False)

    fig = go.Figure()
    normal_pts = merged[~merged["disruption_flag"]]
    disrupt_pts = merged[merged["disruption_flag"]]
    fig.add_trace(go.Scatter(x=merged["period_label"], y=merged["avgwaiting_time"], mode="lines",
                              line=dict(color="#9AA5B5", width=1.5), name="avgwaiting_time"))
    fig.add_trace(go.Scatter(x=normal_pts["period_label"], y=normal_pts["avgwaiting_time"], mode="markers",
                              marker=dict(color=TEAL, size=5), name="Normal"))
    fig.add_trace(go.Scatter(x=disrupt_pts["period_label"], y=disrupt_pts["avgwaiting_time"], mode="markers",
                              marker=dict(color=RED, size=9, symbol="diamond"),
                              name="Detected disruption signal"))
    fig.update_layout(height=420, margin=dict(t=20, b=20), plot_bgcolor="white",
                       xaxis_title="Week", yaxis_title="Avg Waiting Time (hrs)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Detected Disruption Periods")
    if recovery_df is not None and not recovery_df.empty:
        show_cols = ["episode_start", "episode_end", "peak_week", "peak_value", "recovery_weeks_from_peak"]
        show_cols = [c for c in show_cols if c in recovery_df.columns]
        st.dataframe(recovery_df[show_cols].rename(columns={
            "episode_start": "Start", "episode_end": "End", "peak_week": "Peak Week",
            "peak_value": "Max Waiting Time (deviation peak)", "recovery_weeks_from_peak": "Recovery (weeks)"
        }), width="stretch", hide_index=True)
    else:
        st.info(ARTIFACT_MISSING_MSG + " (recovery_analysis.csv)")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_dev = disrupt_pts["avgwaiting_time"].max() if not disrupt_pts.empty else np.nan
        kpi_card("Maximum Waiting Time (flagged weeks)",
                  f"{max_dev:.1f} hrs" if pd.notna(max_dev) else "Not available")
    with c2:
        avg_disrupt = disrupt_pts["avgwaiting_time"].mean() if not disrupt_pts.empty else np.nan
        kpi_card("Avg Waiting Time During Disruption",
                  f"{avg_disrupt:.1f} hrs" if pd.notna(avg_disrupt) else "Not available")
    with c3:
        baseline_med = sim_config.get("median_waiting_time")
        kpi_card("Baseline (Median, Normal Weeks)",
                  f"{baseline_med:.1f} hrs" if baseline_med is not None else "Not available")

    st.write("")
    limitation("These are <b>detected disruption signals</b> from statistical thresholding on AIS-derived "
               "waiting time, not a confirmed operational incident log. No independent ground-truth "
               "disruption record exists in this dataset to fully confirm operational cause.")


# ------------------------------------------------------------------------------------
# PAGE 4 — SCENARIO SIMULATOR
# ------------------------------------------------------------------------------------
def page_scenario_simulator(model_summary, sim_config):
    st.title("Scenario Simulator")
    note("Scenario magnitudes (arrival surge level, service-capacity disruption level) come "
         "directly from <code>simulation_config.json</code>, computed in Colab from this port's "
         "own historical distribution (P90 arrivals, observed disruption-period service time). "
         "No arbitrary numbers are introduced here.")

    scenario_names = engine.list_available_scenarios(sim_config)
    recovery_names = engine.list_available_recovery_strategies(sim_config)

    c1, c2 = st.columns(2)
    with c1:
        scenario_choice = st.selectbox(
            "Scenario", scenario_names,
            format_func=lambda s: engine.SCENARIO_LABELS.get(s, s),
        )
    with c2:
        recovery_choice = st.selectbox(
            "Recovery Strategy", recovery_names,
            format_func=lambda s: engine.RECOVERY_LABELS.get(s, s),
        )

    with st.expander("Scenario input parameters (from simulation_config.json)"):
        presets = engine._presets(sim_config)
        st.json({
            "scenario_inputs": presets[scenario_choice],
            "recovery_inputs": presets[recovery_choice],
        })

    run = st.button("RUN SIMULATION", type="primary")

    if run:
        scenario_result = engine.run_scenario(scenario_choice, model_summary=model_summary, sim_config=sim_config)
        recovery_result = engine.run_scenario(recovery_choice, model_summary=model_summary, sim_config=sim_config)
        baseline_result = engine.run_scenario("S0_Baseline", model_summary=model_summary, sim_config=sim_config)
        st.session_state["last_simulation"] = {
            "scenario": scenario_result, "recovery": recovery_result, "baseline": baseline_result,
        }

    result = st.session_state.get("last_simulation")
    if not result:
        st.info("Select a scenario and recovery strategy, then click RUN SIMULATION.")
        return

    base, scen, rec = result["baseline"], result["scenario"], result["recovery"]

    st.write("")
    st.subheader("Scenario Output — Model-based Estimate")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Waiting Time — Baseline", f"{base['predicted_waiting_time']:.1f} hrs")
    with c2:
        kpi_card("Waiting Time — Scenario", f"{scen['predicted_waiting_time']:.1f} hrs",
                  f"Change vs Baseline: {scen['delta_vs_baseline_pct']:+.1f}%")
    with c3:
        kpi_card("Waiting Time — After Recovery", f"{rec['predicted_waiting_time']:.1f} hrs",
                  f"Change vs Disruption: {rec['delta_vs_disruption_pct']:+.1f}%")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Throughput (numofcall, input assumption) — Scenario",
                  f"{scen['throughput_numofcall_assumed']:.0f}")
    with c2:
        kpi_card("Throughput (numofcall, input assumption) — Recovery",
                  f"{rec['throughput_numofcall_assumed']:.0f}")

    st.write("")
    st.subheader("Baseline vs Disruption vs Recovery")
    labels = ["Baseline", engine.SCENARIO_LABELS.get(scenario_choice, scenario_choice), "Recovery"]
    wt_vals = [base["predicted_waiting_time"], scen["predicted_waiting_time"], rec["predicted_waiting_time"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=wt_vals, marker_color=[NAVY_LIGHT, RED, TEAL],
                          text=[f"{v:.1f}h" for v in wt_vals], textposition="outside"))
    fig.add_annotation(x=labels[1], y=wt_vals[1], text=f"{scen['delta_vs_baseline_pct']:+.1f}% vs baseline",
                        showarrow=False, yshift=25, font=dict(color=RED, size=12))
    fig.add_annotation(x=labels[2], y=wt_vals[2], text=f"{rec['delta_vs_disruption_pct']:+.1f}% vs disruption",
                        showarrow=False, yshift=25, font=dict(color=TEAL, size=12))
    fig.update_layout(height=380, margin=dict(t=40, b=20), plot_bgcolor="white",
                       yaxis_title="Waiting Time (hrs)", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.caption("Labeled as **Scenario Simulation / Model-based Estimate** — not an actual future prediction.")
    limitation("Recovery TIME is not shown here because the static regression model has no time "
               "dynamics. See the Recovery Analysis page for the historically OBSERVED recovery "
               "duration of real disruption episodes.")


# ------------------------------------------------------------------------------------
# PAGE 5 — RECOVERY ANALYSIS
# ------------------------------------------------------------------------------------
def page_recovery_analysis(model_summary, sim_config, recovery_df):
    st.title("Recovery Analysis")
    flow_diagram(["DISRUPTION MAGNITUDE", "RECOVERY STRATEGY", "POST-RECOVERY PERFORMANCE"])
    st.write("")

    recovery_names = engine.list_available_recovery_strategies(sim_config)
    rows = []
    for name in recovery_names:
        r = engine.run_scenario(name, model_summary=model_summary, sim_config=sim_config)
        rows.append({
            "Strategy": engine.RECOVERY_LABELS.get(name, name),
            "Waiting Time (hrs)": r["predicted_waiting_time"],
            "\u0394 vs Disruption": f"{r['delta_vs_disruption_pct']:+.1f}%",
            "Recovery Indicator": "Not available (static model — see historical reference below)",
        })
    st.subheader("Scenario Comparison")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    note("All strategies are shown together with no subjective ranking. This is a "
         "<b>scenario comparison</b>, not a 'best strategy' recommendation.")

    st.write("")
    st.subheader("Historical Reference — Observed Recovery (from Colab, real 2021 episodes)")
    if recovery_df is not None and not recovery_df.empty:
        st.dataframe(recovery_df, width="stretch", hide_index=True)
        note("These recovery durations are <b>observed historical facts</b> (2\u20133 weeks), not "
             "model-simulated. They contextualize the scenario table above but were not produced "
             "by the static regression engine.")
    else:
        st.info(ARTIFACT_MISSING_MSG + " (recovery_analysis.csv)")


# ------------------------------------------------------------------------------------
# PAGE 6 — DECISION DASHBOARD
# ------------------------------------------------------------------------------------
def page_decision_dashboard(df, sim_config, model_summary):
    st.title("Decision Dashboard")
    st.caption("Decision-support summary — not an autonomous decision system")

    latest = df.sort_values("t").iloc[-1]
    pressure, status = engine.classify_operational_pressure(latest["avgwaiting_time"], sim_config)

    st.subheader("Current State")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Waiting Time", f"{latest['avgwaiting_time']:.1f} hrs", latest["period_label"])
    with c2:
        st.markdown("**Disruption Status**")
        kind = {"NORMAL": "normal", "ALERT": "alert", "DISRUPTION SIGNAL": "disruption"}[status]
        status_badge(status, kind)
    with c3:
        st.markdown("**Operational Pressure**")
        kind = {"LOW": "normal", "MEDIUM": "alert", "HIGH": "disruption"}[pressure]
        status_badge(pressure, kind)

    st.write("")
    st.subheader("Selected Scenario")
    result = st.session_state.get("last_simulation")
    if result:
        base, scen, rec = result["baseline"], result["scenario"], result["recovery"]
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Scenario", scen["label"])
        with c2:
            kpi_card("Expected Impact", f"{scen['delta_vs_baseline_pct']:+.1f}% waiting time",
                      f"vs baseline ({base['predicted_waiting_time']:.1f} hrs)")
        with c3:
            kpi_card("Recovery", rec["label"], f"{rec['delta_vs_disruption_pct']:+.1f}% vs disruption")

        st.write("")
        st.markdown("**What changed?**")
        st.write(f"Moving from Baseline to **{scen['label']}** changes predicted waiting time by "
                 f"**{scen['delta_vs_baseline_pct']:+.1f}%** ({base['predicted_waiting_time']:.1f}h \u2192 "
                 f"{scen['predicted_waiting_time']:.1f}h), based on the fitted operational-response "
                 f"relationship from the Colab analysis.")
        st.markdown("**What is the operational implication?**")
        st.write(f"Applying **{rec['label']}** moves predicted waiting time to "
                 f"{rec['predicted_waiting_time']:.1f}h, a **{rec['delta_vs_disruption_pct']:+.1f}%** "
                 f"change relative to the disruption case. This is a model-based estimate to support "
                 f"discussion, not an automated operational decision.")
    else:
        st.info("No scenario has been run yet. Go to **Scenario Simulator**, run a simulation, and "
                "return here to see it reflected in the dashboard.")

    st.write("")
    limitation("This dashboard is a <b>decision-support</b> tool. It does not issue operational "
               "commands and does not replace the judgment of the port operator.")


# ------------------------------------------------------------------------------------
# PAGE 7 — METHODOLOGY
# ------------------------------------------------------------------------------------
def page_methodology(model_summary, sim_config):
    st.title("Methodology")

    st.subheader("Data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
- **Source:** AISData.csv (AIS-derived weekly port operational data)
- **Case:** Singapore
- **Period:** 2019\u20132022
- **Granularity:** Weekly
        """)
    with c2:
        st.markdown("""
- **Primary KPI:** Average Waiting Time
- **Supporting KPI:** Average Service Time, Vessel Entrance, Port Calls, Vessel-size composition
        """)

    st.subheader("Analytical Methods")
    st.markdown("""
- Descriptive statistics (mean, median, std, quartiles, CV)
- Correlation analysis (Pearson + Spearman; Spearman used as primary given right-skewed data)
- Disruption detection: 4 independent methods (percentile, IQR, moving-average deviation,
  robust z-score/MAD) cross-checked with PELT change-point detection
- Statistical comparison: Mann-Whitney U + Cliff's delta effect size (Normal vs Disruption)
- Scenario simulation: fitted OLS regression response surface, scenario magnitudes drawn
  from the port's own historical distribution
    """)

    st.subheader("Machine Learning Decision")
    ml_used = model_summary.get("ml_used", False)
    if ml_used:
        st.write("Machine learning was used. See model details below.")
    else:
        note("Machine learning (Random Forest / XGBoost / LSTM / Neural Network) was "
             "<b>evaluated and explicitly NOT used</b>: with n=210 weekly observations for a "
             "single port and predictors showing modest, largely linear relationships with the "
             "target, a transparent OLS regression (R\u00b2 = "
             f"{model_summary.get('r_squared', 'n/a')}) was judged more defensible than a "
             "non-linear/ensemble model that would risk overfitting at this sample size.")

    st.subheader("Fitted Model")
    st.json({
        "model_type": model_summary.get("model_type"),
        "target": model_summary.get("target"),
        "predictors": model_summary.get("predictors"),
        "r_squared": model_summary.get("r_squared"),
        "n_observations": model_summary.get("n_observations"),
        "validation": model_summary.get("validation"),
    })

    st.subheader("Disruption Definition")
    st.write(f"Final method: **{sim_config.get('final_disruption_method')}** "
             f"(|robust z-score| > {sim_config.get('disruption_threshold_robust_z')}), chosen because "
             "avgwaiting_time is right-skewed and MAD-based statistics are robust to that skew and to "
             "the outliers being detected. Cross-validated against 3 other methods and an independent "
             "change-point detector in Colab.")

    st.subheader("Limitations")
    limitation("Single-port case study (Singapore only). The fitted model explains ~"
               f"{round(model_summary.get('r_squared', 0)*100)}% of weekly waiting-time variance — "
               "other unobserved factors (weather, labor, berth allocation, global supply-chain "
               "shocks) are not in this dataset. Disruption detection is statistical, not validated "
               "against an independent incident log. The scenario simulation is static and cannot "
               "produce a genuine dynamic recovery-time forecast.")

    with st.expander("Figures from the Colab analysis"):
        fig_dir = BASE_DIR / "assets" / "figures"
        if fig_dir.exists():
            figs = sorted(fig_dir.glob("*.png"))
            cols = st.columns(2)
            for i, fp in enumerate(figs):
                with cols[i % 2]:
                    st.image(str(fp), caption=fp.stem, width="stretch")
        else:
            st.info("No figures directory found.")


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
def main():
    missing = check_required_artifacts()
    if missing:
        st.error(ARTIFACT_MISSING_MSG)
        st.write("Missing artifacts:")
        for m in missing:
            st.code(str(REQUIRED_ARTIFACTS[m]))
        st.stop()

    df = load_csv(str(REQUIRED_ARTIFACTS["cleaned_singapore"]))
    disruption_df = load_csv(str(REQUIRED_ARTIFACTS["disruption_periods"]))
    recovery_df = load_csv(str(OPTIONAL_ARTIFACTS["recovery_analysis"])) \
        if artifact_exists(OPTIONAL_ARTIFACTS["recovery_analysis"]) else None
    model_summary, sim_config = load_engine_config()

    st.sidebar.title("MAR-RES")
    st.sidebar.caption("Port Resilience Prototype")
    page = st.sidebar.radio("Navigation", [
        "1. Overview", "2. Port Monitor", "3. Disruption Detection",
        "4. Scenario Simulator", "5. Recovery Analysis",
        "6. Decision Dashboard", "7. Methodology",
    ], label_visibility="collapsed")

    st.sidebar.write("")
    st.sidebar.markdown("---")
    st.sidebar.caption("All models & thresholds sourced from the Colab analytical engine. "
                        "Streamlit performs no training.")

    if page.startswith("1"):
        page_overview(df, disruption_df, sim_config)
    elif page.startswith("2"):
        page_port_monitor(df, disruption_df)
    elif page.startswith("3"):
        page_disruption_detection(df, disruption_df, recovery_df, sim_config)
    elif page.startswith("4"):
        page_scenario_simulator(model_summary, sim_config)
    elif page.startswith("5"):
        page_recovery_analysis(model_summary, sim_config, recovery_df)
    elif page.startswith("6"):
        page_decision_dashboard(df, sim_config, model_summary)
    elif page.startswith("7"):
        page_methodology(model_summary, sim_config)


if __name__ == "__main__":
    main()
