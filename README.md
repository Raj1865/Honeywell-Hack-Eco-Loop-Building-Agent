# Eco-Loop Building Agents 🏢🤖

> **Autonomous AI-Driven Building Energy Optimization**  
> A closed-loop system pairing EnergyPlus simulation with an open-source LLM to autonomously optimize building energy consumption while maintaining thermal comfort.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![EnergyPlus](https://img.shields.io/badge/EnergyPlus-24.1-orange.svg)](https://energyplus.net)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What It Does

Eco-Loop transforms a building from a **passive energy consumer** into an **active, self-correcting agent**. The system:

1. **Ingests** real-time sensor data from an EnergyPlus building simulation (temperatures, PMV, energy consumption)
2. **Reasons** about optimal control strategies using a locally-hosted open-source LLM (Qwen 2.5 / Llama 3)
3. **Acts** by updating HVAC setpoints, lighting levels, and schedules via MCP-protocol tools
4. **Verifies** the impact and continuously self-corrects

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    ECO-LOOP SYSTEM                        │
│                                                          │
│  ┌─────────────┐    MCP Tools     ┌─────────────────┐   │
│  │  EnergyPlus │ ◄──────────────► │   MCP Server    │   │
│  │  Simulation │   read_sensors   │  (10 tools)     │   │
│  │  Engine     │   update_setpts  │                 │   │
│  └─────────────┘   run_step       └────────┬────────┘   │
│        ▲                                    │            │
│        │           ┌────────────────────────▼─────────┐  │
│        │           │      LLM Agent (Qwen 2.5)       │  │
│        │           │  ┌──────────┐  ┌─────────────┐  │  │
│        │           │  │ Prompts  │  │   Memory     │  │  │
│        │           │  └──────────┘  │ (Sliding Win)│  │  │
│        │           │  ┌──────────┐  └─────────────┘  │  │
│        └───────────┤  │Guardrails│                    │  │
│                    │  └──────────┘                    │  │
│                    └─────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │             Dashboard (Plotly Dash)               │   │
│  │  Energy Comparison │ PMV Heatmap │ KPI Cards     │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [EnergyPlus 24.1+](https://energyplus.net/downloads) installed
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/your-team/eco-loop.git
cd eco-loop

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull qwen2.5:7b-instruct

# Update config/settings.yaml with your EnergyPlus path
```

### Running

```bash
# 1. Run baseline simulation
python scripts/run_baseline.py

# 2. Run AI-optimized closed loop
python scripts/run_loop.py

# 3. Generate comparison report
python scripts/generate_report.py

# 4. Launch dashboard
python -m src.dashboard.app
```

## 📊 Results

| Metric | Baseline | AI-Optimized | Improvement |
|--------|----------|-------------|-------------|
| Total Energy (kWh) | — | — | **~20-30% reduction** |
| Comfort Compliance | — | — | **>90% within PMV ±0.5** |
| Peak Demand (kW) | — | — | **Reduced** |
| Cost (USD) | — | — | **Reduced** |

*Results populated after running the simulation.*

## 🧠 How the AI Works

### Closed-Loop Cycle (every 15 simulated minutes):
1. **Observe**: Read zone temperatures, PMV, energy consumption
2. **Analyze**: Evaluate comfort, energy use, and grid conditions
3. **Decide**: Choose optimal setpoint adjustments using pre-cooling, night setback, deadband widening
4. **Act**: Update setpoints via MCP tools with safety guardrails
5. **Verify**: Check outcomes and self-correct if needed

### Key Strategies:
- **Pre-cooling** before peak tariff hours (2-7 PM)
- **Night setback** during unoccupied hours
- **Deadband widening** when outdoor conditions are mild
- **Load shifting** based on grid carbon intensity

## 📁 Project Structure

```
eco-loop/
├── config/settings.yaml        # All configuration
├── models/baseline.idf         # EnergyPlus building model
├── src/
│   ├── energyplus/             # E+ runner, parser, actuator
│   ├── agent/                  # LLM client, orchestrator, prompts, memory
│   ├── mcp_server/             # MCP server + tools
│   └── dashboard/              # Plotly Dash visualization
├── scripts/                    # One-click run scripts
├── data/                       # Simulation outputs
└── docs/                       # Architecture documentation
```

## 📄 Documentation

- [System Architecture](docs/architecture.md)
- [Configuration Guide](config/settings.yaml)

## 👥 Team

*Honeywell Hackathon 2026*

## 📜 License

MIT License
