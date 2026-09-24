"""
simulation_engine.py
---------------------
MAR-RES scenario-simulation engine.

This module is a THIN, DEPENDENCY-LIGHT wrapper around the OLS regression coefficients
already fitted in the Colab analysis notebook and saved to:
    config/model_summary.json
    config/simulation_config.json

Streamlit does NOT retrain anything at runtime. It only:
  1. loads cleaned_singapore.csv (historical data for charts)
  2. loads model_summary.json (fitted OLS coefficients)
  3. loads simulation_config.json (baseline inputs, thresholds, scenario magnitudes)
  4. calls run_scenario(...) / simulate_disruption(...) / simulate_recovery(...) below

IMPORTANT (per project rules): this is an analytical / scenario-based STATISTICAL
simulation, not a digital twin and not a real forecast. It has no physical representation
of berths, cranes, or discrete vessel events. Every magnitude used below (baseline means,
P90 arrival surge, disruption-period mean service time, disruption/recovery thresholds)
was computed ONCE in the Colab notebook and is only read here — never recomputed.
"""
import json
from pathlib import Path


# ----------------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------------
def load_artifacts(config_dir="config"):
    """Load the fitted model and simulation configuration produced by the Colab notebook."""
    config_dir = Path(config_dir)
    with open(config_dir / "model_summary.json") as f:
        model_summary = json.load(f)
    with open(config_dir / "simulation_config.json") as f:
        sim_config = json.load(f)
    return model_summary, sim_config


# ----------------------------------------------------------------------------------
# Core prediction (fitted OLS response surface — Section G/H of the Colab analysis)
# ----------------------------------------------------------------------------------
def predict_waiting_time(count_entrance, numofcall, avgservice_time, share_neopan, model_summary):
    """
    Predict avgwaiting_time from the fitted OLS coefficients.
    This is a linear, interpretable response surface — NOT a black-box / ML model
    (the Colab analysis concluded ML was not justified for this dataset: n=210,
    single port, modest/near-linear relationships).
    """
    c = model_summary["coefficients"]
    return (
        c["const"]
        + c["count_entrance"] * count_entrance
        + c["numofcall"] * numofcall
        + c["avgservice_time"] * avgservice_time
        + c["share_neopan"] * share_neopan
    )


def _presets(sim_config):
    """Build every named scenario/intervention input set from Colab-derived config only."""
    baseline = sim_config["normal_period_baseline_inputs"]
    ratio = sim_config["call_to_entrance_ratio"]
    surge_entrance = sim_config["surge_entrance_p90"]
    capacity_disrupt_service_time = sim_config["capacity_disrupt_service_time"]

    return {
        # --- Disruption scenarios (Scenario Simulator page) ---
        "S0_Baseline": dict(count_entrance=baseline["count_entrance"], numofcall=baseline["numofcall"],
                             avgservice_time=baseline["avgservice_time"], share_neopan=baseline["share_neopan"]),
        "S1_ArrivalSurge": dict(count_entrance=surge_entrance, numofcall=surge_entrance * ratio,
                                 avgservice_time=baseline["avgservice_time"], share_neopan=baseline["share_neopan"]),
        "S2_CapacityDisruption": dict(count_entrance=baseline["count_entrance"], numofcall=baseline["numofcall"],
                                       avgservice_time=capacity_disrupt_service_time, share_neopan=baseline["share_neopan"]),
        "S3_Combined": dict(count_entrance=surge_entrance, numofcall=surge_entrance * ratio,
                             avgservice_time=capacity_disrupt_service_time, share_neopan=baseline["share_neopan"]),
        # --- Recovery strategies (Recovery Analysis page). No_intervention == S3_Combined;
        #     Combined_recovery == S0_Baseline; the two single-lever strategies restore ONE
        #     input back to baseline while the other stays at its disrupted level. ---
        "No_intervention": dict(count_entrance=surge_entrance, numofcall=surge_entrance * ratio,
                                 avgservice_time=capacity_disrupt_service_time, share_neopan=baseline["share_neopan"]),
        "Arrival_smoothing": dict(count_entrance=baseline["count_entrance"], numofcall=baseline["numofcall"],
                                   avgservice_time=capacity_disrupt_service_time, share_neopan=baseline["share_neopan"]),
        "Service_capacity_recovery": dict(count_entrance=surge_entrance, numofcall=surge_entrance * ratio,
                                           avgservice_time=baseline["avgservice_time"], share_neopan=baseline["share_neopan"]),
        "Combined_recovery": dict(count_entrance=baseline["count_entrance"], numofcall=baseline["numofcall"],
                                   avgservice_time=baseline["avgservice_time"], share_neopan=baseline["share_neopan"]),
    }


SCENARIO_NAMES = ["S0_Baseline", "S1_ArrivalSurge", "S2_CapacityDisruption", "S3_Combined"]
RECOVERY_STRATEGY_NAMES = ["No_intervention", "Arrival_smoothing", "Service_capacity_recovery", "Combined_recovery"]

SCENARIO_LABELS = {
    "S0_Baseline": "Baseline",
    "S1_ArrivalSurge": "Arrival Surge",
    "S2_CapacityDisruption": "Service Capacity Disruption",
    "S3_Combined": "Combined Disruption",
}
RECOVERY_LABELS = {
    "No_intervention": "No Intervention",
    "Arrival_smoothing": "Arrival Smoothing",
    "Service_capacity_recovery": "Capacity Recovery",
    "Combined_recovery": "Combined Recovery",
}


def list_available_scenarios(sim_config):
    """Every scenario name Streamlit may offer — all backed by real Colab-derived config."""
    return list(SCENARIO_NAMES)


def list_available_recovery_strategies(sim_config):
    return list(RECOVERY_STRATEGY_NAMES)


def run_scenario(scenario_name, count_entrance=None, numofcall=None, avgservice_time=None,
                  share_neopan=None, model_summary=None, sim_config=None):
    """
    Run one named scenario/intervention (or "Custom") and return predicted waiting time
    plus deltas vs both the Colab-derived baseline and, where relevant, the disruption case.

    scenario_name: one of SCENARIO_NAMES + RECOVERY_STRATEGY_NAMES, or "Custom" (all four
                   input kwargs must then be supplied, e.g. from Streamlit sliders whose
                   min/max come from the historical data range — never invented).
    """
    presets = _presets(sim_config)

    if scenario_name == "Custom":
        inputs = dict(count_entrance=count_entrance, numofcall=numofcall,
                       avgservice_time=avgservice_time, share_neopan=share_neopan)
        if any(v is None for v in inputs.values()):
            raise ValueError("Custom scenario requires all four inputs: count_entrance, "
                              "numofcall, avgservice_time, share_neopan.")
    else:
        if scenario_name not in presets:
            raise ValueError(f"Unknown scenario_name '{scenario_name}'. "
                              f"Choose from {list(presets)} or 'Custom'.")
        inputs = presets[scenario_name]

    baseline_pred = predict_waiting_time(**presets["S0_Baseline"], model_summary=model_summary)
    disruption_pred = predict_waiting_time(**presets["S3_Combined"], model_summary=model_summary)
    pred = predict_waiting_time(**inputs, model_summary=model_summary)

    delta_vs_baseline_pct = (pred - baseline_pred) / baseline_pred * 100
    delta_vs_disruption_pct = (pred - disruption_pred) / disruption_pred * 100

    return {
        "scenario": scenario_name,
        "label": {**SCENARIO_LABELS, **RECOVERY_LABELS}.get(scenario_name, scenario_name),
        "inputs": inputs,
        "predicted_waiting_time": round(pred, 2),
        "baseline_waiting_time": round(baseline_pred, 2),
        "disruption_waiting_time": round(disruption_pred, 2),
        "delta_vs_baseline_pct": round(delta_vs_baseline_pct, 1),
        "delta_vs_disruption_pct": round(delta_vs_disruption_pct, 1),
        "throughput_numofcall_assumed": round(inputs["numofcall"], 1),
    }


# ----------------------------------------------------------------------------------
# Disruption flagging (reuses the SAME robust z-score/MAD rule chosen as final in Colab)
# ----------------------------------------------------------------------------------
def simulate_disruption(observed_series, sim_config):
    """
    Flag disruption weeks in an observed avgwaiting_time series using the robust
    z-score / MAD rule chosen as final in the Colab analysis. Thresholds come from
    sim_config, never re-derived here.
    """
    import numpy as np
    median_wt = sim_config["median_waiting_time"]
    mad = sim_config["mad_scaled"]
    threshold = sim_config["disruption_threshold_robust_z"]
    arr = np.asarray(observed_series, dtype=float)
    z = (arr - median_wt) / mad
    return (np.abs(z) > threshold).tolist()


def classify_operational_pressure(value, sim_config):
    """
    LOW / MEDIUM / HIGH banding using the SAME robust z-score used for disruption
    detection, plus the k=1.5 threshold already used in Colab (Method 3, MA-deviation)
    as the "elevated / alert" boundary. No new threshold is invented here.
    """
    import numpy as np
    median_wt = sim_config["median_waiting_time"]
    mad = sim_config["mad_scaled"]
    z = abs((value - median_wt) / mad)
    if z > sim_config["disruption_threshold_robust_z"]:
        return "HIGH", "DISRUPTION SIGNAL"
    elif z > 1.5:
        return "MEDIUM", "ALERT"
    else:
        return "LOW", "NORMAL"


def simulate_recovery(peak_waiting_time, sim_config):
    """
    LIMITATION (stated explicitly, per project rules against fabricating recovery claims):
    the static regression / scenario model has NO time dimension, so it CANNOT compute a
    valid recovery TIME for a hypothetical scenario. What the Colab analysis DOES provide
    (data/recovery_analysis.csv) is the OBSERVED recovery duration for the three real
    disruption episodes found in the historical Singapore data. This function returns that
    historical reference rather than fabricating a scenario-specific number.
    """
    return {
        "recovery_time_computable_for_hypothetical_scenario": False,
        "reason": "Static regression response surface has no time dynamics; a valid recovery "
                  "time requires a dynamic or discrete-event model, which this dataset does not "
                  "support.",
        "recovery_threshold_used_historically": sim_config["recovery_threshold"],
        "historical_reference": "See data/recovery_analysis.csv for observed recovery times "
                                 "of the real 2021 disruption episodes.",
    }


if __name__ == "__main__":
    # Quick smoke test
    model_summary, sim_config = load_artifacts("../config")
    for name in SCENARIO_NAMES + RECOVERY_STRATEGY_NAMES:
        print(run_scenario(name, model_summary=model_summary, sim_config=sim_config))
