# System Architecture — Eco-Loop Building Agents

## 1. Overview

Eco-Loop is a closed-loop AI system that autonomously optimizes building energy consumption using EnergyPlus as the physics engine and an open-source LLM as the cognitive engine. The system communicates through the Model Context Protocol (MCP), exposing EnergyPlus interactions as structured tools the LLM can invoke.

### Key Innovation
Traditional BMS uses rigid schedules. Eco-Loop uses an LLM that **reasons about** building physics, weather, occupancy, and grid conditions to make dynamic, context-aware control decisions — with self-correction when things go wrong.

---

## 2. System Architecture Diagram

```mermaid
graph TB
    subgraph Simulation["EnergyPlus Simulation Engine"]
        IDF[".idf Building Model"]
        EPW["Weather File (.epw)"]
        EMS["EMS Actuators"]
        OUT["Output: .eso / .csv"]
    end

    subgraph MCP["MCP Server (10 Tools)"]
        RS["read_sensors"]
        CS["get_comfort_status"]
        ES["get_energy_summary"]
        US["update_setpoints"]
        AL["adjust_lighting"]
        MS["modify_schedule"]
        WF["get_weather_forecast"]
        SS["run_simulation_step"]
        PI["parse_idf_section"]
        EL["get_error_log"]
    end

    subgraph Agent["LLM Cognitive Engine"]
        LLM["Qwen 2.5-7B-Instruct"]
        PM["Prompt Manager"]
        MEM["Context Memory<br/>(Sliding Window)"]
        GR["Safety Guardrails"]
    end

    subgraph Loop["Orchestrator (State Machine)"]
        READ["1. READ Sensors"]
        REASON["2. REASON with LLM"]
        ACT["3. ACT via Tools"]
        ADV["4. ADVANCE Sim"]
        LOG["5. LOG Data"]
    end

    subgraph Dash["Dashboard"]
        KPI["KPI Cards"]
        CHART["Charts"]
        TABLE["Summary Table"]
    end

    IDF --> OUT
    EPW --> IDF
    OUT --> RS
    RS --> READ
    READ --> REASON
    REASON --> LLM
    LLM --> ACT
    ACT --> US
    US --> EMS
    EMS --> IDF
    ACT --> AL
    ADV --> SS
    LOG --> Dash
```

---

## 3. Component Deep-Dive

### 3.1 EnergyPlus Simulation Engine
- **Model**: DOE Reference Small/Medium Office
- **Timestep**: 15-minute intervals (4 per hour)
- **Output variables**: Zone temperatures, humidity, PMV, HVAC energy, lighting energy
- **Control interface**: EMS actuators for runtime setpoint overrides

### 3.2 MCP Server & Tool Definitions
- **Transport**: stdio (primary) / SSE (for dashboard integration)
- **10 tools** covering observation, control, and diagnostics
- **Error handling**: Every tool returns structured JSON with success/error status

### 3.3 LLM Cognitive Engine
- **Model**: Qwen 2.5-7B-Instruct (best OSS tool-calling accuracy)
- **Runtime**: Ollama (local inference)
- **Temperature**: 0.2 (deterministic for control decisions)
- **Context window**: Managed by sliding-window memory

### 3.4 Closed-Loop Orchestrator
- **State machine**: INIT → READING → REASONING → ACTING → ADVANCING → LOGGING
- **Self-correction**: Failed tool calls are fed back to the LLM for recovery
- **Safe mode**: After 5 consecutive failures, falls back to default setpoints

---

## 4. Tool-Calling Architecture

### MCP Protocol Flow
```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant L as LLM (Qwen 2.5)
    participant M as MCP Server
    participant E as EnergyPlus

    O->>M: call read_sensors()
    M->>E: Parse latest CSV row
    E-->>M: Sensor JSON
    M-->>O: Tool result

    O->>L: System prompt + sensor data + tools
    L-->>O: Reasoning + tool_calls[]

    loop For each tool_call
        O->>M: call tool(args)
        M->>E: Modify IDF / EMS
        E-->>M: Confirmation
        M-->>O: Tool result
    end

    O->>M: call run_simulation_step()
    M->>E: Advance timestep
```

### Error Handling & Self-Correction
1. Tool call fails → structured error returned
2. Error fed back to LLM with recovery prompt
3. LLM suggests corrective action (up to 3 retries)
4. If all retries fail → safe defaults applied

---

## 5. Prompt Engineering Strategies

### System Prompt Design
- **Role definition**: Clear identity as building energy optimizer
- **Prioritized objectives**: Safety > Comfort > Energy > Cost > Carbon
- **Tool documentation**: Inline descriptions with parameter constraints
- **Strategy hints**: Pre-cooling, night setback, deadband widening
- **Output schema**: Enforced JSON structure for reasoning + tool calls

### Per-Timestep Message
- Structured markdown with all sensor readings
- Previous action outcomes for continuity
- Grid and weather context for forward planning

---

## 6. Prompt Latency Management

| Strategy | Implementation |
|----------|---------------|
| **Batched payloads** | All sensor data in one JSON blob per timestep |
| **Concise output schema** | Structured JSON reduces token count |
| **Low temperature** | 0.2 — fewer sampling iterations |
| **Model quantization** | Ollama uses GGUF Q4 by default (~4GB VRAM) |
| **Token budget** | max_tokens=2048 prevents runaway generation |
| **Timeout** | 30s hard limit per LLM call |

---

## 7. Handling Lengthy Simulation Logs

| Challenge | Solution |
|-----------|----------|
| `.eso` files can be 100MB+ | Parse only latest timestep via `parser.py` |
| Error logs grow unbounded | Extract last 50 lines + count warnings/errors |
| Context window overflow | Sliding window: last 20 timesteps in detail, older data summarized |
| Large IDF files | Parse only requested sections via `parse_idf_section` tool |

---

## 8. Results & Analysis

*[To be populated after simulation runs]*

| Metric | Baseline | AI-Optimized | Δ |
|--------|----------|-------------|---|
| Total Energy (kWh) | | | |
| Peak Demand (kW) | | | |
| Comfort Compliance (%) | | | |
| Cost (USD) | | | |
| CO₂ (kg) | | | |
