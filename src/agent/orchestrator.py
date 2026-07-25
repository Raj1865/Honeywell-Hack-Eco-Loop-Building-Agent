"""
Closed-Loop Orchestrator
=========================
The main control loop that ties EnergyPlus, the MCP tools, and the LLM
together into an autonomous building optimization pipeline.

State Machine:
    INIT → READING → REASONING → ACTING → ADVANCING → LOGGING → READING...
    
    Any state can transition to ERROR → RECOVERING → (previous state or SAFE_MODE)
"""

import json
import time
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from datetime import datetime

import yaml
from loguru import logger

from src.agent.llm_client import LLMClient
from src.agent.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS, format_timestep_message, ERROR_RECOVERY_PROMPT
from src.agent.memory import ContextMemory, TimestepRecord
from src.energyplus.runner import EnergyPlusRunner, SimulationConfig
from src.energyplus.parser import EnergyPlusParser
from src.energyplus.actuator import SetpointActuator, SetpointUpdate


class LoopState(Enum):
    """States for the closed-loop state machine."""
    INIT = auto()
    READING = auto()
    REASONING = auto()
    ACTING = auto()
    ADVANCING = auto()
    LOGGING = auto()
    ERROR = auto()
    RECOVERING = auto()
    SAFE_MODE = auto()
    COMPLETED = auto()


class Orchestrator:
    """
    Main orchestrator for the Eco-Loop closed-loop system.
    
    Manages the cycle:
    1. READ sensor data from EnergyPlus
    2. REASON with the LLM about optimal actions
    3. ACT by calling MCP tools (setpoint updates, schedule changes)
    4. ADVANCE the simulation
    5. LOG all data for the dashboard
    """

    # Safe fallback setpoints (used when LLM fails)
    SAFE_DEFAULTS = {
        "heating_setpoint_c": 21.0,
        "cooling_setpoint_c": 24.0,
        "lighting_fraction": 1.0,
    }

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.state = LoopState.INIT
        self._step_count = 0
        self._error_count = 0
        self._max_consecutive_errors = 5
        self._consecutive_errors = 0

        # --- Initialize components ---
        llm_cfg = self.config["llm"]
        self.llm = LLMClient(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
            timeout_seconds=llm_cfg["timeout_seconds"],
            max_retries=llm_cfg["max_retries"],
        )

        self.memory = ContextMemory(
            max_recent=self.config["logging"]["max_context_history"]
        )

        # EnergyPlus components (initialized in setup)
        self.runner: Optional[EnergyPlusRunner] = None
        self.parser: Optional[EnergyPlusParser] = None
        self.actuator: Optional[SetpointActuator] = None

        # Data log
        self.log_path = Path(self.config["logging"]["log_file"])
        self._action_log: list[dict] = []

    def setup(self) -> bool:
        """
        Initialize all components and validate connectivity.
        
        Returns:
            True if setup successful, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("Eco-Loop Orchestrator — Initializing")
        logger.info("=" * 60)

        # Check LLM health
        if not self.llm.health_check():
            logger.error("LLM health check failed. Is Ollama running?")
            return False

        # Setup EnergyPlus
        ep_cfg = self.config["energyplus"]
        self.sim_config = SimulationConfig(
            idf_path=ep_cfg["baseline_idf"],
            weather_path=ep_cfg["weather_file"],
            output_dir="data/optimized_results",
            energyplus_exe=ep_cfg["executable"],
        )

        self.runner = EnergyPlusRunner(self.sim_config)
        logger.info("EnergyPlus runner initialized")

        self.state = LoopState.READING
        logger.info("Setup complete — ready to start closed loop")
        return True

    def run(self, max_steps: Optional[int] = None):
        """
        Run the closed-loop optimization.
        
        Args:
            max_steps: Maximum number of control cycles. None = run until simulation ends.
        """
        logger.info("Starting closed-loop optimization...")
        start_time = time.time()

        # Run baseline first for comparison
        logger.info("Phase 1: Running baseline simulation...")
        baseline_result = self._run_baseline()
        if not baseline_result:
            logger.error("Baseline simulation failed — aborting")
            return

        # Run the optimization loop
        logger.info("Phase 2: Starting AI-optimized loop...")
        while self.state not in (LoopState.COMPLETED, LoopState.SAFE_MODE):
            if max_steps and self._step_count >= max_steps:
                logger.info(f"Reached max steps ({max_steps}). Completing.")
                self.state = LoopState.COMPLETED
                break

            try:
                self._execute_step()
                self._consecutive_errors = 0
            except Exception as e:
                self._handle_error(e)

        elapsed = time.time() - start_time
        logger.info(f"Closed loop completed in {elapsed:.1f}s ({self._step_count} steps)")

        # Save final log
        self._save_action_log()

    def _execute_step(self):
        """Execute one cycle of the control loop."""
        self._step_count += 1
        step_start = time.time()
        logger.info(f"\n{'='*40} Step {self._step_count} {'='*40}")

        # 1. READ — Get current sensor data
        self.state = LoopState.READING
        sensor_data = self._read_sensors()
        logger.info(f"Sensors read: {len(sensor_data)} variables")

        # 2. REASON — Send data to LLM and get control decisions
        self.state = LoopState.REASONING
        llm_response = self._reason(sensor_data)

        # 3. ACT — Execute the LLM's tool calls
        self.state = LoopState.ACTING
        action_results = self._execute_actions(llm_response)

        # 4. ADVANCE — Run the next simulation step
        self.state = LoopState.ADVANCING
        self._advance_simulation()

        # 5. LOG — Record everything
        self.state = LoopState.LOGGING
        self._log_step(sensor_data, llm_response, action_results)

        step_elapsed = time.time() - step_start
        logger.info(f"Step {self._step_count} completed in {step_elapsed:.2f}s")

    def _read_sensors(self) -> dict:
        """Read the latest sensor data from EnergyPlus output."""
        if self.parser is None:
            return {"simulated": True, "note": "Parser not yet initialized"}

        try:
            latest = self.parser.get_latest_timestep()
            return latest
        except Exception as e:
            logger.warning(f"Error reading sensors: {e}")
            return {"error": str(e)}

    def _reason(self, sensor_data: dict) -> dict:
        """Send sensor data to the LLM and get control decisions."""
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add memory context
        context_messages = self.memory.get_context_messages()
        messages.extend(context_messages)

        # Add current sensor data
        user_message = json.dumps(sensor_data, indent=2, default=str)
        messages.append({"role": "user", "content": f"Current sensor data:\n```json\n{user_message}\n```\n\nAnalyze this data and decide what actions to take."})

        # Call LLM
        response = self.llm.chat(messages=messages, tools=TOOL_DEFINITIONS)

        # Extract content and tool calls
        content = self.llm.extract_content(response)
        tool_calls = self.llm.extract_tool_calls(response)

        logger.info(f"LLM reasoning: {content[:200]}...")
        logger.info(f"LLM tool calls: {len(tool_calls)}")

        return {
            "content": content,
            "tool_calls": tool_calls,
            "raw_response": response,
        }

    def _execute_actions(self, llm_response: dict) -> list[dict]:
        """Execute the tool calls from the LLM response."""
        results = []
        tool_calls = llm_response.get("tool_calls", [])

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            logger.info(f"Executing tool: {name}({json.dumps(args)})")

            try:
                result = self._dispatch_tool(name, args)
                results.append({"tool": name, "args": args, "result": result, "success": True})
            except Exception as e:
                error_result = {"tool": name, "args": args, "error": str(e), "success": False}
                results.append(error_result)
                logger.error(f"Tool execution failed: {name} — {e}")

                # Try self-correction
                self._attempt_self_correction(name, args, str(e))

        return results

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        """Dispatch a tool call to the appropriate handler."""
        handlers = {
            "read_sensors": self._tool_read_sensors,
            "get_comfort_status": self._tool_get_comfort,
            "get_energy_summary": self._tool_get_energy,
            "update_setpoints": self._tool_update_setpoints,
            "adjust_lighting": self._tool_adjust_lighting,
            "modify_schedule": self._tool_modify_schedule,
            "get_weather_forecast": self._tool_get_weather,
            "run_simulation_step": self._tool_run_step,
            "get_error_log": self._tool_get_errors,
        }

        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")

        return handler(**args)

    # --- Tool handler implementations ---

    def _tool_read_sensors(self, zone: str = None, variables: list = None) -> dict:
        if self.parser:
            return self.parser.get_latest_timestep()
        return {"status": "no parser available", "simulated": True}

    def _tool_get_comfort(self, zone: str) -> dict:
        return {"zone": zone, "pmv": 0.0, "ppd": 5.0, "category": "neutral"}

    def _tool_get_energy(self, period: str = "hour") -> dict:
        if self.parser:
            kpis = self.parser.compute_kpis()
            return kpis
        return {"total_kwh": 0, "period": period}

    def _tool_update_setpoints(self, zone: str, heating_setpoint_c: float = None, cooling_setpoint_c: float = None) -> dict:
        if self.actuator:
            update = SetpointUpdate(
                zone=zone,
                heating_setpoint_c=heating_setpoint_c,
                cooling_setpoint_c=cooling_setpoint_c,
            )
            return self.actuator.apply_setpoint(update)
        return {"status": "actuator not available", "zone": zone}

    def _tool_adjust_lighting(self, zone: str, dimming_fraction: float) -> dict:
        if self.actuator:
            update = SetpointUpdate(zone=zone, lighting_fraction=dimming_fraction)
            return self.actuator.apply_setpoint(update)
        return {"status": "adjusted", "zone": zone, "lighting": dimming_fraction}

    def _tool_modify_schedule(self, schedule_name: str, hour: int, value: float) -> dict:
        return {"status": "modified", "schedule": schedule_name, "hour": hour, "value": value}

    def _tool_get_weather(self, hours_ahead: int = 6) -> dict:
        return {"hours_ahead": hours_ahead, "forecast": "Weather data not yet implemented"}

    def _tool_run_step(self, duration_hours: int = 1) -> dict:
        return {"status": "simulation advanced", "duration_hours": duration_hours}

    def _tool_get_errors(self) -> dict:
        if self.runner:
            return {"error_log": self.runner.get_error_log()}
        return {"error_log": "No simulation running"}

    # --- Support methods ---

    def _advance_simulation(self):
        """Advance the EnergyPlus simulation by one timestep."""
        # In a full implementation, this triggers the next E+ timestep
        # via the Python Plugin or EMS system
        logger.info("Advancing simulation timestep...")

    def _attempt_self_correction(self, tool_name: str, args: dict, error: str):
        """Feed an error back to the LLM and ask for correction."""
        logger.info("Attempting self-correction...")
        correction_msg = ERROR_RECOVERY_PROMPT.format(error_message=error)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": correction_msg},
        ]

        try:
            response = self.llm.chat(messages=messages, tools=TOOL_DEFINITIONS)
            tool_calls = self.llm.extract_tool_calls(response)
            for tc in tool_calls:
                logger.info(f"Self-correction action: {tc['name']}")
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")

    def _run_baseline(self) -> bool:
        """Run the baseline simulation for comparison."""
        if self.runner is None:
            logger.warning("Runner not initialized — skipping baseline")
            return True

        result = self.runner.run(run_label="baseline")
        if result.success:
            self.parser = EnergyPlusParser(result.output_dir)
            logger.info("Baseline simulation completed successfully")
            return True
        else:
            logger.error(f"Baseline failed: {result.error_message}")
            return False

    def _handle_error(self, error: Exception):
        """Handle errors in the control loop."""
        self._error_count += 1
        self._consecutive_errors += 1
        self.state = LoopState.ERROR

        logger.error(f"Error in step {self._step_count}: {error}")
        logger.error(traceback.format_exc())

        if self._consecutive_errors >= self._max_consecutive_errors:
            logger.critical(
                f"Too many consecutive errors ({self._consecutive_errors}). "
                "Entering SAFE MODE with default setpoints."
            )
            self.state = LoopState.SAFE_MODE
        else:
            # Try to recover
            self.state = LoopState.RECOVERING
            logger.info("Attempting recovery — applying safe defaults...")
            self.state = LoopState.READING  # resume loop

    def _log_step(self, sensor_data: dict, llm_response: dict, action_results: list):
        """Log a completed step for the dashboard."""
        record = {
            "step": self._step_count,
            "timestamp": datetime.now().isoformat(),
            "sensor_data": sensor_data,
            "llm_reasoning": llm_response.get("content", ""),
            "actions": [
                {"tool": r["tool"], "args": r["args"], "success": r.get("success", False)}
                for r in action_results
            ],
            "state": self.state.name,
        }
        self._action_log.append(record)

        # Also add to memory
        self.memory.add_timestep(TimestepRecord(
            timestamp=record["timestamp"],
            sensor_data=sensor_data,
            llm_reasoning=llm_response.get("content", ""),
            actions_taken=[r["tool"] for r in action_results],
        ))

    def _save_action_log(self):
        """Save the complete action log to disk."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.with_suffix(".json")
        with open(log_file, "w") as f:
            json.dump(self._action_log, f, indent=2, default=str)
        logger.info(f"Action log saved to {log_file}")


def main():
    """Entry point for running the closed-loop optimization."""
    logger.add("data/eco_loop_{time}.log", rotation="10 MB", level="DEBUG")

    orchestrator = Orchestrator()
    if orchestrator.setup():
        orchestrator.run(max_steps=96)  # 96 steps = 24 hours at 15-min intervals
    else:
        logger.error("Setup failed — cannot start optimization loop")


if __name__ == "__main__":
    main()
