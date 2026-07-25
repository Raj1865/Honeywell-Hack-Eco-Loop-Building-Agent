"""
Context Window Memory Manager
================================
Manages the sliding window of conversation history to keep the LLM
within its context limits while retaining the most relevant information.
"""

import json
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class TimestepRecord:
    """A single timestep's data in the memory."""
    timestamp: str
    sensor_data: dict
    llm_reasoning: Optional[str] = None
    actions_taken: list = field(default_factory=list)
    action_results: list = field(default_factory=list)
    energy_kwh: float = 0.0
    pmv: float = 0.0


class ContextMemory:
    """
    Sliding-window memory manager for the LLM agent.
    
    Keeps the last N timesteps of detailed data in the context window.
    Older data is summarized into aggregate statistics to preserve
    long-term trends without exceeding token limits.
    
    Design:
    - Recent window (last N steps): Full detail for tactical decisions
    - Summary buffer: Aggregated stats for strategic context
    - Action history: Last M actions for continuity
    """

    def __init__(self, max_recent: int = 20, max_actions: int = 50):
        self.max_recent = max_recent
        self.max_actions = max_actions
        self.recent: deque[TimestepRecord] = deque(maxlen=max_recent)
        self.action_history: deque[dict] = deque(maxlen=max_actions)
        self._total_steps = 0
        self._cumulative_energy_kwh = 0.0
        self._pmv_values: list[float] = []
        self._summary_cache: Optional[str] = None

    def add_timestep(self, record: TimestepRecord):
        """Add a new timestep record to the memory."""
        self.recent.append(record)
        self._total_steps += 1
        self._cumulative_energy_kwh += record.energy_kwh
        self._pmv_values.append(record.pmv)
        self._summary_cache = None  # invalidate cache
        logger.debug(f"Memory: added timestep {record.timestamp} (total: {self._total_steps})")

    def add_action(self, action: dict):
        """Record an action taken by the agent."""
        self.action_history.append(action)

    def get_context_messages(self) -> list[dict]:
        """
        Build the conversation context for the LLM.
        
        Returns:
            List of message dicts ready to be sent to the LLM,
            containing a summary of history + recent detailed data.
        """
        messages = []

        # Add historical summary if we have older data
        if self._total_steps > self.max_recent:
            summary = self._build_summary()
            messages.append({
                "role": "user",
                "content": f"## Historical Summary\n{summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll factor in this historical context for my decisions.",
            })

        # Add recent timestep data as user messages
        for record in self.recent:
            messages.append({
                "role": "user",
                "content": self._format_record(record),
            })
            if record.llm_reasoning:
                messages.append({
                    "role": "assistant",
                    "content": record.llm_reasoning,
                })

        return messages

    def _build_summary(self) -> str:
        """Build a statistical summary of older (evicted) timesteps."""
        if self._summary_cache:
            return self._summary_cache

        n_summarized = self._total_steps - len(self.recent)
        if n_summarized <= 0:
            return "No historical data to summarize."

        avg_pmv = sum(self._pmv_values[:-len(self.recent)]) / n_summarized if n_summarized > 0 else 0
        comfort_violations = sum(
            1 for pmv in self._pmv_values[:-len(self.recent)]
            if abs(pmv) > 0.5
        )

        summary = f"""Over the past {n_summarized} timesteps:
- **Total energy consumed**: {self._cumulative_energy_kwh:.1f} kWh
- **Average PMV**: {avg_pmv:.2f}
- **Comfort violations** (|PMV| > 0.5): {comfort_violations} / {n_summarized} steps ({comfort_violations / n_summarized * 100:.1f}%)
- **Recent actions summary**: {len(self.action_history)} actions taken
"""
        self._summary_cache = summary
        return summary

    def _format_record(self, record: TimestepRecord) -> str:
        """Format a single timestep record for the LLM context."""
        parts = [f"**Timestep: {record.timestamp}**"]
        parts.append(f"Energy: {record.energy_kwh:.2f} kWh | PMV: {record.pmv:.2f}")

        if record.actions_taken:
            parts.append(f"Actions: {json.dumps(record.actions_taken, indent=None)}")

        if record.action_results:
            results_str = "; ".join(str(r) for r in record.action_results)
            parts.append(f"Results: {results_str}")

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get current memory statistics."""
        return {
            "total_steps": self._total_steps,
            "recent_steps": len(self.recent),
            "total_actions": len(self.action_history),
            "cumulative_energy_kwh": self._cumulative_energy_kwh,
            "avg_pmv": sum(self._pmv_values) / len(self._pmv_values) if self._pmv_values else 0,
        }

    def clear(self):
        """Clear all memory."""
        self.recent.clear()
        self.action_history.clear()
        self._total_steps = 0
        self._cumulative_energy_kwh = 0.0
        self._pmv_values.clear()
        self._summary_cache = None
        logger.info("Memory cleared")
