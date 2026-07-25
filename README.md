# 🏢 Eco-Loop: Autonomous AI-Driven Building Energy Agent
> **Honeywell Automation Hackathon 2026 Submission**

[![Tests](https://img.shields.io/badge/Tests-36%2F36%20Passed-10B981?style=flat-square)](result.txt)
[![EnergyPlus](https://img.shields.io/badge/EnergyPlus-v26.1.0-blue?style=flat-square)](https://energyplus.net)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?style=flat-square)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Honeywell%20Forge-E5261F?style=flat-square)](https://www.honeywell.com/us/en/forge)

---

## 📌 Executive Summary

**Eco-Loop** transforms commercial building automation from static rule-based control into an **autonomous, self-correcting cognitive loop**. It connects a local open-source LLM (`qwen2.5:7b-instruct`) directly to **EnergyPlus 26.1.0** building physics telemetry, dynamically optimizing HVAC setpoints every 15 minutes.

- **Energy Reduction**: **~24% kWh savings** over standard fixed BMS setpoints.
- **Thermal Comfort**: **>90% compliance** with **ASHRAE Standard 55 / ISO 7730** Fanger PMV bounds ($\pm 0.5$).
- **Predictive Maintenance**: Z-score statistical & physics rule engine detecting equipment degradation and thermal deadband fighting.
- **Natural Language Interface**: Data-grounded assistant for facility managers to query building status in plain English.

---

## 🧑‍⚖️ Evaluation Guide for Judges / Reviewers

The repository is **100% self-contained**. Evaluators can test and inspect the project in **two ways**:

### Option A: Instant Zero-Setup Dashboard & Telemetry Review (Recommended — 30 Seconds)
Pre-generated simulation runs, active action logs, anomaly reports, and test outputs are included directly in the repository. You can launch and inspect the full interactive **Honeywell Forge Dashboard** immediately:

```bash
# 1. Clone the repository
git clone https://github.com/Raj1865/Honeywell-Hack-Eco-Loop-Building-Agent.git
cd Honeywell-Hack-Eco-Loop-Building-Agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Honeywell Forge Dashboard
python src/dashboard/app.py
```
👉 Open **http://127.0.0.1:8050** in your browser to view live KPI cards, interactive Plotly charts, the predictive anomaly table, and the facility manager chat panel.

---

### Option B: Run Automated Industry-Grade Test Suite (10 Seconds)
Verify system stability, physical safety clamping, parser correctness, and orchestrator integrity across 36 automated unit, integration, and system tests:

```bash
python tests/test_suite.py
```
*(All 36 tests execute and print a complete report to terminal and `result.txt`.)*

---

### Option C: Run Full Simulation & AI Optimization Loop from Scratch
To execute the complete closed-loop agent with live EnergyPlus simulation and local LLM inference:

#### Prerequisites:
1. Install [EnergyPlus 26.1.0](https://energyplus.net/downloads) (Default path: `C:\EnergyPlusV26-1-0`).
2. Install and launch [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

#### Execution:
```bash
# 1. Run baseline simulation (24h thermodynamic simulation)
python scripts/run_baseline.py

# 2. Run predictive anomaly analysis
python scripts/run_anomaly_analysis.py

# 3. Run AI-optimized closed-loop orchestrator (96 timesteps)
python scripts/run_loop.py
```

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   EnergyPlus 26.1.0 Engine    │
                                  │  (Thermodynamic Building IDF) │
                                  └──────────────┬────────────────┘
                                                 │ 15-min Timestep CSV
                                                 ▼
┌───────────────────────────────┐  Sensor Telemetry  ┌───────────────────────────────┐
│     Facility Manager Chat     ├───────────────────►│  PyIDF CSV & Telemetry Parser │
│  (Data-Grounded Query API)    │                    └──────────────┬────────────────┘
└───────────────────────────────┘                                   │ Cleaned State Vector
                                                                    ▼
                                                     ┌───────────────────────────────┐
                                                     │    Safety Clamping & Rules    │
                                                     │ (Hard Bounds: 18°C - 28°C)    │
                                                     └──────────────┬────────────────┘
                                                                    │ Validated State
                                                                    ▼
┌───────────────────────────────┐ LLM Tool Calls     ┌───────────────────────────────┐
│   Predictive Anomaly Engine   │◄───────────────────┤  FSM Loop Orchestrator Agent │
│   (Z-Score & Physics Rules)   │ (Actuations)       │   (READ -> REASON -> ACT)     │
└──────────────┬────────────────┘                    └──────────────┬────────────────┘
               │                                                    │ Action Log (.json)
               ▼                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│              Honeywell Forge Dashboard (Plotly Dash - 3s Poll @ :8050)             │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Comparison

| Metric | Baseline (Uncontrolled) | Eco-Loop AI Agent | Impact / Improvement |
| :--- | :---: | :---: | :---: |
| **Total Facility Energy (kWh)** | 15,970 kWh | 12,137 kWh | **▼ 24.0% Reduction** |
| **Thermal Comfort Compliance** | 49.7% | **94.0%** | **▲ +44.3% Comfort Improvement** |
| **Fanger PMV Range** | -1.88 to +0.95 | **-0.45 to +0.42** | **Strictly within ISO 7730 Bounds** |
| **Predictive Faults Detected** | 0 (Unmonitored) | **15 Active Alerts** | **Early Warning (Equipment & Deadbands)** |

---

## 📁 Repository Structure

```
Honeywell-Hack-Eco-Loop-Building-Agent/
├── config/
│   └── settings.yaml            # Building constraints, safety bounds, tariff schedules
├── data/
│   ├── baseline_results/        # Baseline EnergyPlus simulation output & CSV telemetry
│   ├── anomaly_report.json      # Generated predictive fault & degradation reports
│   └── eco_loop.json            # Step-by-step AI action log & telemetry history
├── models/
│   └── baseline.idf             # EnergyPlus 5-Zone Commercial Building Model
├── scripts/
│   ├── run_baseline.py          # Runs baseline EnergyPlus simulation
│   ├── run_anomaly_analysis.py  # Executes predictive anomaly detection
│   └── run_loop.py              # Executes 96-step closed-loop AI agent
├── src/
│   ├── agent/
│   │   ├── actuator.py          # Physical safety clamping & deadband enforcement
│   │   ├── anomaly_detector.py  # Statistical Z-score & physics rule fault detector
│   │   ├── facility_chat.py     # Data-grounded natural language Q&A interface
│   │   ├── llm_client.py        # Ollama API client with exponential backoff & tool parser
│   │   ├── orchestrator.py      # FSM orchestrator (READ -> REASON -> ACT)
│   │   ├── memory.py            # Sliding context memory window
│   │   └── prompts.py           # System prompts & structured tool definitions
│   ├── dashboard/
│   │   └── app.py               # Plotly Dash application (Honeywell Forge Industrial Theme)
│   └── energyplus/
│       ├── parser.py            # PyIDF output CSV & KPI parser
│       └── runner.py            # EnergyPlus process execution wrapper
├── tests/
│   └── test_suite.py            # 36-test automated industry-grade test runner
├── result.txt                   # Automated test report output
└── requirements.txt             # Python dependencies
```

---

## 📜 License & Compliance

Developed for the **Honeywell Automation Hackathon 2026**.  
Built under the **MIT License**.
