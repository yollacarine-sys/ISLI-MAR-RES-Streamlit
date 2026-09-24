# MAR-RES — Port Resilience Decision-Support Prototype

AIS-Driven Early Warning and Digital Twin for Resilient Port Operations
(ISLI 2026 — Resilient Maritime Logistics)

A Streamlit **decision-support prototype** built entirely on top of analytical artifacts
produced in a Google Colab notebook. **This app trains nothing.** Every disruption
threshold, statistical model, and simulation parameter was computed once in Colab and is
only *read* here.

---

## 1. Relationship between Colab and Streamlit

```
 GOOGLE COLAB (offline, one-time)              STREAMLIT (this repo, runtime)
 ───────────────────────────────              ──────────────────────────────
 Data audit, cleaning                    ──▶   data/cleaned_singapore.csv
 Disruption detection (4 methods)        ──▶   data/disruption_periods.csv
 Normal vs disruption comparison         ──▶   data/normal_vs_disruption.csv
 Correlation / driver analysis           ──▶   data/correlation_results.csv
 OLS regression fitting                  ──▶   config/model_summary.json
 Scenario magnitude derivation           ──▶   config/simulation_config.json
 Scenario / recovery simulation logic    ──▶   engine/simulation_engine.py
 Historical recovery-time measurement    ──▶   data/recovery_analysis.csv
                                                       │
                                                       ▼
                                          app.py just LOADS + CALLS these,
                                          via @st.cache_data / @st.cache_resource.
                                          No retraining, no new thresholds.
```

If you change the case (e.g. a different port), or want to refresh the analysis,
**re-run the Colab notebook** and copy its `outputs/` contents into `data/` and `config/`
here — do not edit numbers by hand and do not recompute them inside Streamlit.

---

## 2. Folder structure

```
.
├── app.py                          # Main Streamlit app (7 pages)
├── engine/
│   └── simulation_engine.py        # Thin wrapper around Colab-fitted OLS model
├── data/
│   ├── cleaned_singapore.csv
│   ├── disruption_periods.csv
│   ├── yearly_summary.csv
│   ├── correlation_results.csv
│   ├── scenario_results.csv
│   ├── recovery_analysis.csv
│   ├── recovery_strategy_results.csv
│   ├── normal_vs_disruption.csv
│   └── singapore_descriptive_stats.csv
├── config/
│   ├── model_summary.json          # Fitted OLS coefficients + validation metrics
│   └── simulation_config.json      # Baseline inputs, scenario magnitudes, thresholds
├── assets/
│   └── figures/                    # PNG figures from the Colab notebook (Methodology page)
├── requirements.txt
└── README.md
```

**No `models/` folder** (`model.pkl` / `scaler.pkl` / `feature_columns.pkl`) is included.
The Colab analysis (Section H) evaluated machine learning and explicitly concluded it was
**not justified** for this dataset (n=210, single port, near-linear relationships) — an
OLS regression was used instead, and its coefficients live in `config/model_summary.json`.
Creating placeholder `.pkl` files would misrepresent that decision, so none exist.

**No `data/simulation_results.csv`** either — the brief listed it as "if available"; the
Colab notebook instead produced `scenario_results.csv`, `recovery_analysis.csv`, and
`recovery_strategy_results.csv`, which together cover the same content and are what the
app actually reads.

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. How to run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## 5. Required artifacts

The app checks for these files on startup and will **not** run with fabricated data if
any are missing — it shows:

> "Required analytical artifact not found. Please run the Colab analytical engine first."

Required:
`data/cleaned_singapore.csv`, `data/disruption_periods.csv`, `data/yearly_summary.csv`,
`data/correlation_results.csv`, `data/scenario_results.csv`, `data/normal_vs_disruption.csv`,
`config/model_summary.json`, `config/simulation_config.json`.

Optional (used when present, silently skipped with a note when absent):
`data/recovery_analysis.csv`, `data/recovery_strategy_results.csv`,
`data/singapore_descriptive_stats.csv`, `assets/figures/*.png`.

---

## 6. Application pages

1. **Overview** — KPI summary + AIS→Analytics→Early Warning→Scenario→Recovery flow diagram.
2. **Port Monitor** — actual weekly time series (waiting time, service time, entrance,
   calls) with Year/Week filters, from `cleaned_singapore.csv` only.
3. **Disruption Detection** — waiting-time series overlaid with the disruption flag
   **already determined in Colab** (no threshold is recomputed here). Uses the wording
   "Detected disruption signal", not "confirmed operational disruption".
4. **Scenario Simulator** — select a disruption scenario + recovery strategy (all
   Colab-derived presets) and click **RUN SIMULATION** to call
   `engine.simulation_engine.run_scenario(...)`. Results are explicitly labeled
   "Scenario Simulation / Model-based Estimate", not a real forecast.
5. **Recovery Analysis** — comparative table of all recovery strategies (no subjective
   "best strategy" ranking) plus the historically **observed** recovery duration from the
   three real 2021 disruption episodes, kept clearly separate from the model-based table.
6. **Decision Dashboard** — current operational state (waiting time, disruption status,
   operational pressure banding) plus a summary of the last scenario run on page 4.
   Positioned explicitly as a decision-support summary, not an autonomous decision system.
7. **Methodology** — full transparency on data, case, period, methods, the ML
   go/no-go decision, the fitted model, the disruption definition, and stated limitations.

---

## 7. Performance / caching

- `@st.cache_data` — all CSV/JSON loading (`load_csv`, artifact reads).
- `@st.cache_resource` — the model + simulation-config objects (`load_engine_config`).
- The simulation itself only runs when the user clicks **RUN SIMULATION** — nothing is
  computed automatically on every rerun.
- No training occurs anywhere in this app.

---

## 8. Limitations (see also the in-app Methodology page)

- Single-port case study (Singapore only, n=210 weekly observations).
- The fitted OLS model explains R² ≈ 0.40 of weekly waiting-time variance; other
  unobserved factors (weather, labor, berth allocation, global supply-chain shocks) are
  not in this dataset.
- Disruption detection is statistical (robust z-score / MAD), not validated against an
  independent port-authority incident log.
- The scenario-simulation engine is a **static regression response surface**, not a
  digital twin: it has no time dynamics, so it **cannot** compute a valid recovery time
  for a hypothetical scenario. Where the app shows "Not available," this is intentional —
  the alternative would be fabricating a number the model cannot support.
- This is a **decision-support prototype**. It does not issue operational commands and is
  not a substitute for the judgment of the port operator.
