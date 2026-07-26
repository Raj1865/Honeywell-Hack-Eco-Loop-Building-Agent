"""
Natural Language Facility Manager Interface
=============================================
Allows building operators to ask questions about their building
and get answers grounded in real simulation data.

Examples:
  "Why is floor 3 so hot?"
  "What would happen if I raise the setpoint by 2°C?"
  "Show me the energy trend for the last 6 hours"
  "Is there anything wrong with the HVAC in the core zone?"

This is the feature that sells Honeywell products — making complex
building data accessible through plain English conversation.
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from loguru import logger


# ======================================================================
# Context Builder — Grounds the LLM in real building data
# ======================================================================

FACILITY_CHAT_PROMPT = """You are the **Eco-Loop Building Assistant**, an expert facility management advisor deployed in a Honeywell-managed commercial building.

## Your Role
A facility manager is asking you questions about their building. Answer using the REAL building data provided below. Be specific — cite actual temperatures, energy values, and zone names from the data.

## Current Building Data
{building_context}

## Anomaly Alerts
{anomaly_context}

## Rules
1. Always reference specific data points (e.g., "Core_ZN is currently at 27.3°C")
2. If asked about something not in the data, say so honestly
3. For "what-if" questions, use your knowledge of building physics to estimate
4. Suggest specific, actionable steps — not generic advice
5. If you detect a potential problem, flag it proactively
6. Keep responses concise and professional

## Response Format
Answer directly. Use bullet points for actionable items. Include relevant numbers."""


class FacilityChatInterface:
    """
    Natural language interface for facility managers to query building status.

    Grounds LLM responses in real EnergyPlus simulation data so answers
    are factual and specific, not generic.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._conversation_history: list[dict] = []
        self._max_history = 10

    def _build_building_context(self) -> str:
        """Build a context string from the latest simulation data."""
        context_parts = []

        # Load baseline KPIs
        baseline_dir = Path("data/baseline_results")
        kpi_files = sorted(baseline_dir.glob("*/baseline_kpis.json"))
        if kpi_files:
            kpis = json.loads(kpi_files[-1].read_text())
            context_parts.append(f"""### Baseline Performance
- Total Energy: {kpis.get('total_kwh', 'N/A'):.0f} kWh
- Peak Demand: {kpis.get('peak_kw', 'N/A'):.1f} kW
- Average PMV: {kpis.get('avg_pmv', 'N/A'):.3f}
- Comfort Compliance: {kpis.get('comfort_hours_pct', 'N/A'):.1f}%
- Avg Zone Temp: {kpis.get('avg_zone_temp_c', 'N/A'):.1f}°C
- Max Zone Temp: {kpis.get('max_zone_temp_c', 'N/A'):.1f}°C
- Min Zone Temp: {kpis.get('min_zone_temp_c', 'N/A'):.1f}°C""")

        # Load latest action log
        if log_path.exists():
            try:
                log_data = json.loads(log_path.read_text())
                if isinstance(log_data, list) and log_data:
                    latest = log_data[-1]
                    n_steps = len(log_data)
                    total_steps = max(n_steps, 96)
                    context_parts.append(f"""### AI Optimization Status
- Steps Completed: {n_steps} / {total_steps}
- Latest Step: {latest.get('step', '?')}
- Current State: {latest.get('state', '?')}""")

                    sensor = latest.get("sensor_data", {})
                    if isinstance(sensor, dict) and sensor:
                        context_parts.append(f"### Latest Sensor Readings")
                        for key, val in sensor.items():
                            if isinstance(val, (int, float)):
                                context_parts.append(f"- {key}: {val:.2f}")
                            elif isinstance(val, dict):
                                for k2, v2 in val.items():
                                    if isinstance(v2, (int, float)):
                                        context_parts.append(f"- {key}.{k2}: {v2:.2f}")

                    reasoning = latest.get("llm_reasoning", "")
                    if reasoning:
                        context_parts.append(f"### Latest AI Reasoning\n{reasoning[:500]}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not load action log: {e}")

        # Load latest baseline CSV for zone-level data
        csv_files = sorted(baseline_dir.glob("*/eplusout.csv"))
        if csv_files:
            import pandas as pd
            try:
                df = pd.read_csv(csv_files[-1])
                df.columns = [c.strip() for c in df.columns]
                last_row = df.iloc[-1]

                temp_cols = [c for c in df.columns if "zone mean air temp" in c.lower()]
                if temp_cols:
                    context_parts.append("### Current Zone Temperatures")
                    for tc in temp_cols:
                        zone = tc.split(":")[0]
                        context_parts.append(f"- {zone}: {last_row[tc]:.1f}°C")

                pmv_cols = [c for c in df.columns if "pmv" in c.lower()]
                if pmv_cols:
                    context_parts.append("### Current Zone Comfort (PMV)")
                    for pc in pmv_cols:
                        zone = pc.split(":")[0]
                        pmv_val = last_row[pc]
                        if pmv_val > 0.5:
                            status = "TOO WARM"
                        elif pmv_val < -0.5:
                            status = "TOO COLD"
                        else:
                            status = "COMFORTABLE"
                        context_parts.append(f"- {zone}: PMV={pmv_val:.2f} ({status})")
            except Exception as e:
                logger.warning(f"Could not parse CSV: {e}")

        return "\n\n".join(context_parts) if context_parts else "No building data available yet."

    def _build_anomaly_context(self) -> str:
        """Load anomaly report if available."""
        anomaly_path = Path("data/anomaly_report.json")
        if anomaly_path.exists():
            try:
                anomalies = json.loads(anomaly_path.read_text())
                if anomalies:
                    lines = []
                    for a in anomalies[:5]:  # Show top 5
                        lines.append(
                            f"- [{a['severity']}] {a['category']}: {a['description'][:120]}"
                        )
                    return "\n".join(lines)
            except Exception:
                pass
        return "No anomalies detected."

    def ask(self, question: str) -> str:
        """
        Ask a question about the building and get a data-grounded answer.

        Args:
            question: Natural language question from facility manager.

        Returns:
            LLM response grounded in real building data.
        """
        if self.llm is None:
            return "Error: LLM client not initialized. Start Ollama first."

        # Build context from real data
        building_context = self._build_building_context()
        anomaly_context = self._build_anomaly_context()

        system_msg = FACILITY_CHAT_PROMPT.format(
            building_context=building_context,
            anomaly_context=anomaly_context,
        )

        # Build message list
        messages = [{"role": "system", "content": system_msg}]

        # Add conversation history for continuity
        for entry in self._conversation_history[-self._max_history:]:
            messages.append(entry)

        # Add current question
        messages.append({"role": "user", "content": question})

        try:
            response = self.llm.chat(messages=messages, temperature=0.3)
            answer = self.llm.extract_content(response)

            # Save to history
            self._conversation_history.append({"role": "user", "content": question})
            self._conversation_history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Sorry, I encountered an error: {e}"

    def get_history(self) -> list[dict]:
        """Return conversation history."""
        return self._conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
