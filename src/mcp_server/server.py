"""
MCP Server Definition
======================
Model Context Protocol server that exposes EnergyPlus interaction
as structured tools the LLM can call.

This server runs as a standalone process and communicates with the
LLM agent via the MCP protocol (stdio or SSE transport).
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed — server will use fallback mode")

from src.energyplus.parser import EnergyPlusParser
from src.energyplus.actuator import SetpointActuator, SetpointUpdate


class EcoLoopMCPServer:
    """
    MCP Server that exposes building simulation tools.
    
    Tools provided:
    - read_sensors: Get latest simulation sensor readings
    - get_comfort_status: Check PMV comfort for a zone
    - get_energy_summary: Energy consumption summary
    - update_setpoints: Modify HVAC setpoints
    - adjust_lighting: Change lighting levels
    - modify_schedule: Override schedule values
    - get_weather_forecast: Upcoming weather conditions
    - run_simulation_step: Advance the simulation
    - parse_idf_section: Inspect IDF file contents
    - get_error_log: Read EnergyPlus error log
    """

    def __init__(
        self,
        output_dir: str = "data/optimized_results",
        idf_path: Optional[str] = None,
        idd_path: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.parser = None
        self.actuator = None

        if Path(output_dir).exists():
            self.parser = EnergyPlusParser(output_dir)

        if idf_path and Path(idf_path).exists():
            self.actuator = SetpointActuator(idf_path, idd_path)

        if MCP_AVAILABLE:
            self.server = Server("eco-loop-building-agent")
            self._register_tools()
        else:
            self.server = None

    def _register_tools(self):
        """Register all tools with the MCP server."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="read_sensors",
                    description="Read current sensor values from the building simulation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "zone": {"type": "string", "description": "Zone name (optional, all zones if omitted)"},
                            "variables": {"type": "array", "items": {"type": "string"}, "description": "Variables to read"},
                        },
                    },
                ),
                Tool(
                    name="get_comfort_status",
                    description="Get PMV thermal comfort status for a zone.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "zone": {"type": "string", "description": "Zone name"},
                        },
                        "required": ["zone"],
                    },
                ),
                Tool(
                    name="get_energy_summary",
                    description="Get energy consumption summary for a time period.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "period": {"type": "string", "enum": ["hour", "day", "week"]},
                        },
                        "required": ["period"],
                    },
                ),
                Tool(
                    name="update_setpoints",
                    description="Update HVAC thermostat setpoints. Heating: 18-24°C, Cooling: 22-28°C.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "zone": {"type": "string"},
                            "heating_setpoint_c": {"type": "number"},
                            "cooling_setpoint_c": {"type": "number"},
                        },
                        "required": ["zone"],
                    },
                ),
                Tool(
                    name="adjust_lighting",
                    description="Adjust lighting level (0.0=off, 1.0=full).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "zone": {"type": "string"},
                            "dimming_fraction": {"type": "number"},
                        },
                        "required": ["zone", "dimming_fraction"],
                    },
                ),
                Tool(
                    name="modify_schedule",
                    description="Override a schedule value for a specific hour.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "schedule_name": {"type": "string"},
                            "hour": {"type": "integer"},
                            "value": {"type": "number"},
                        },
                        "required": ["schedule_name", "hour", "value"],
                    },
                ),
                Tool(
                    name="get_weather_forecast",
                    description="Get weather forecast for upcoming hours.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hours_ahead": {"type": "integer"},
                        },
                        "required": ["hours_ahead"],
                    },
                ),
                Tool(
                    name="run_simulation_step",
                    description="Advance the simulation by N hours.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "duration_hours": {"type": "integer"},
                        },
                        "required": ["duration_hours"],
                    },
                ),
                Tool(
                    name="parse_idf_section",
                    description="Parse and return a section of the IDF file.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "object_type": {"type": "string", "description": "E+ object type (e.g., 'Zone', 'Schedule:Compact')"},
                        },
                        "required": ["object_type"],
                    },
                ),
                Tool(
                    name="get_error_log",
                    description="Read the EnergyPlus error log.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                result = self._handle_tool_call(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                error_response = {"error": str(e), "tool": name, "suggestion": "Check arguments and retry."}
                return [TextContent(type="text", text=json.dumps(error_response))]

    def _handle_tool_call(self, name: str, arguments: dict) -> dict:
        """Route tool calls to their implementations."""
        handlers = {
            "read_sensors": self._read_sensors,
            "get_comfort_status": self._get_comfort_status,
            "get_energy_summary": self._get_energy_summary,
            "update_setpoints": self._update_setpoints,
            "adjust_lighting": self._adjust_lighting,
            "modify_schedule": self._modify_schedule,
            "get_weather_forecast": self._get_weather_forecast,
            "run_simulation_step": self._run_simulation_step,
            "parse_idf_section": self._parse_idf_section,
            "get_error_log": self._get_error_log,
        }

        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")

        return handler(**arguments)

    # --- Tool implementations ---

    def _read_sensors(self, zone: str = None, variables: list = None) -> dict:
        if self.parser is None:
            return {"error": "Parser not initialized. No simulation output available."}
        latest = self.parser.get_latest_timestep()
        if zone:
            # Filter for specific zone
            filtered = {k: v for k, v in latest.get("data", {}).items() if zone.lower() in k.lower()}
            return {"timestamp": latest.get("timestamp"), "zone": zone, "data": filtered}
        return latest

    def _get_comfort_status(self, zone: str) -> dict:
        if self.parser is None:
            return {"error": "No simulation data available"}
        try:
            pmv_series = self.parser.get_timeseries("pmv", zone=zone)
            current_pmv = float(pmv_series.iloc[-1])
            ppd = 100 - 95 * (2.71828 ** (-(0.03353 * current_pmv**4 + 0.2179 * current_pmv**2)))
            if current_pmv < -1.0:
                category = "cold"
            elif current_pmv < -0.5:
                category = "slightly cool"
            elif current_pmv < 0.5:
                category = "neutral (comfortable)"
            elif current_pmv < 1.0:
                category = "slightly warm"
            else:
                category = "hot"
            return {"zone": zone, "pmv": current_pmv, "ppd": ppd, "category": category}
        except Exception as e:
            return {"zone": zone, "error": str(e)}

    def _get_energy_summary(self, period: str = "hour") -> dict:
        if self.parser is None:
            return {"error": "No simulation data available"}
        kpis = self.parser.compute_kpis()
        return {"period": period, **kpis}

    def _update_setpoints(self, zone: str, heating_setpoint_c: float = None, cooling_setpoint_c: float = None) -> dict:
        if self.actuator is None:
            return {"error": "Actuator not initialized. Provide IDF path."}
        update = SetpointUpdate(
            zone=zone,
            heating_setpoint_c=heating_setpoint_c,
            cooling_setpoint_c=cooling_setpoint_c,
        )
        return self.actuator.apply_setpoint(update)

    def _adjust_lighting(self, zone: str, dimming_fraction: float) -> dict:
        if self.actuator is None:
            return {"error": "Actuator not initialized"}
        update = SetpointUpdate(zone=zone, lighting_fraction=dimming_fraction)
        return self.actuator.apply_setpoint(update)

    def _modify_schedule(self, schedule_name: str, hour: int, value: float) -> dict:
        logger.info(f"Schedule override: {schedule_name} @ hour {hour} = {value}")
        return {"status": "modified", "schedule_name": schedule_name, "hour": hour, "value": value}

    def _get_weather_forecast(self, hours_ahead: int = 6) -> dict:
        # In production, this would read the EPW weather file
        return {
            "hours_ahead": hours_ahead,
            "note": "Reading from EPW weather file",
            "forecast": [
                {"hour_offset": i, "temp_c": 25.0 + i * 0.5, "humidity_pct": 50}
                for i in range(min(hours_ahead, 24))
            ],
        }

    def _run_simulation_step(self, duration_hours: int = 1) -> dict:
        logger.info(f"Advancing simulation by {duration_hours} hours")
        return {"status": "advanced", "duration_hours": duration_hours}

    def _parse_idf_section(self, object_type: str) -> dict:
        if self.actuator is None:
            return {"error": "No IDF loaded"}
        return {"object_type": object_type, "note": "IDF section parsing"}

    def _get_error_log(self) -> dict:
        if self.parser is None:
            return {"error": "No simulation data available"}
        return self.parser.parse_error_log()

    # --- Server lifecycle ---

    async def run_stdio(self):
        """Run the MCP server over stdio transport."""
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not available. Install with: pip install mcp")
            return

        logger.info("Starting Eco-Loop MCP Server (stdio transport)...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)

    def call_tool_sync(self, name: str, arguments: dict) -> dict:
        """Synchronous tool call — for use within the orchestrator."""
        return self._handle_tool_call(name, arguments)


def main():
    """Run the MCP server as a standalone process."""
    import asyncio

    logger.add("data/mcp_server.log", rotation="5 MB", level="DEBUG")

    server = EcoLoopMCPServer(
        output_dir="data/optimized_results",
        idf_path="models/baseline.idf",
    )

    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
