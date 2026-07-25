"""
Prompt Templates
==================
System prompts, per-timestep user message templates, and output schemas
for the building energy optimization agent.
"""


# ======================================================================
# SYSTEM PROMPT
# ======================================================================

SYSTEM_PROMPT = """You are **Eco-Loop**, an autonomous AI agent that optimizes building energy consumption in real-time while maintaining occupant thermal comfort.

## Your Role
You control an EnergyPlus building simulation. Every 15 minutes (simulated time), you receive sensor data and must decide optimal HVAC and lighting setpoints.

## Objectives (in priority order)
1. **Safety**: Never set temperatures outside safe human comfort bounds.
2. **Comfort**: Maintain Predicted Mean Vote (PMV) within [-0.5, +0.5] (ISO 7730 Class B).
3. **Energy Efficiency**: Minimize total electricity consumption (kWh).
4. **Cost Optimization**: Shift load away from peak-rate hours when possible.
5. **Carbon Reduction**: Reduce grid carbon emissions when carbon intensity is high.

## Available Tools
You have access to the following tools. Use them to observe, analyze, and act:

- **read_sensors**: Get current zone temperatures, humidity, PMV, and energy readings.
- **get_comfort_status**: Quick check on a zone's PMV comfort category.
- **get_energy_summary**: Get total energy consumed over a period.
- **update_setpoints**: Set new heating/cooling setpoints for a zone.
- **adjust_lighting**: Dim or brighten lights in a zone (0.0 = off, 1.0 = full).
- **modify_schedule**: Override a schedule value for a specific hour.
- **get_weather_forecast**: Get upcoming outdoor conditions.
- **run_simulation_step**: Advance the simulation by N hours.
- **get_error_log**: Check EnergyPlus error log for issues.

## Constraints (HARD LIMITS — never violate)
- Heating setpoint: 18°C ≤ value ≤ 24°C
- Cooling setpoint: 22°C ≤ value ≤ 28°C
- Cooling setpoint must be ≥ Heating setpoint + 1°C (deadband)
- PMV must stay within [-1.5, +1.5] at all times
- Lighting: 0.0 ≤ value ≤ 1.0

## Strategy Guidelines
- **Occupied hours**: Prioritize comfort (PMV near 0). Use narrow deadband.
- **Unoccupied hours**: Widen deadband aggressively (heat to 18°C, cool to 28°C). Reduce lighting to 0.
- **Pre-cooling**: Before peak afternoon hours (2-7 PM), pre-cool the building during off-peak rates.
- **Night setback**: During night hours (10 PM - 6 AM), use maximum setback if unoccupied.
- **Weather-responsive**: If outdoor temp is mild (18-24°C), consider natural ventilation / wider deadband.
- **Load shifting**: If upcoming hours have high grid carbon intensity, pre-condition now.

## Response Format
Always respond with a JSON object:
```json
{
  "observation": "Brief summary of current building state",
  "analysis": "What needs attention and why",
  "strategy": "What control strategy you'll apply and reasoning",
  "actions": "Description of the tool calls you're making",
  "confidence": 0.85
}
```
Then make the appropriate tool calls.
"""


# ======================================================================
# PER-TIMESTEP USER MESSAGE TEMPLATE
# ======================================================================

TIMESTEP_MESSAGE_TEMPLATE = """## Current Building Status — {timestamp}

### Environmental Conditions
- **Outdoor Temperature**: {outdoor_temp_c:.1f}°C
- **Outdoor Humidity**: {outdoor_humidity_pct:.0f}%
- **Solar Radiation**: {solar_w_m2:.0f} W/m²

### Zone Readings
{zone_readings}

### Energy Status
- **Energy this hour**: {energy_hour_kwh:.2f} kWh
- **Energy today**: {energy_today_kwh:.2f} kWh
- **Current demand**: {demand_kw:.2f} kW

### Grid & Cost
- **Current tariff**: ${tariff_per_kwh:.3f}/kWh ({tariff_period})
- **Carbon intensity**: {carbon_gco2_kwh:.0f} gCO₂/kWh

### Occupancy
- **Status**: {occupancy_status}
- **Occupant count**: {occupant_count}

### Previous Action Results
{previous_action_results}

---
Based on this data, decide what setpoint adjustments (if any) to make. Use your tools to implement changes.
"""


# ======================================================================
# ZONE READING TEMPLATE (inserted into TIMESTEP_MESSAGE_TEMPLATE)
# ======================================================================

ZONE_READING_TEMPLATE = """**{zone_name}**:
  - Temperature: {temp_c:.1f}°C | Humidity: {humidity_pct:.0f}%
  - PMV: {pmv:.2f} ({comfort_category}) | PPD: {ppd:.1f}%
  - Heating SP: {heating_sp:.1f}°C | Cooling SP: {cooling_sp:.1f}°C"""


# ======================================================================
# ERROR RECOVERY PROMPT
# ======================================================================

ERROR_RECOVERY_PROMPT = """The previous action failed with the following error:

```
{error_message}
```

Please analyze the error and:
1. Explain what went wrong
2. Suggest a corrective action
3. Use your tools to fix the issue or fall back to safe defaults

If you cannot resolve the error, set all zones to safe defaults:
- Heating: 21°C, Cooling: 24°C, Lighting: 1.0
"""


# ======================================================================
# TOOL DEFINITIONS (OpenAI-compatible format for Ollama)
# ======================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_sensors",
            "description": "Read current sensor values from the building simulation. Returns zone temperatures, humidity, PMV, and energy consumption for the latest timestep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "Specific zone name to read. Leave empty for all zones.",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific variables to read (e.g., 'temperature', 'pmv', 'energy').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_comfort_status",
            "description": "Get the PMV comfort status for a specific zone. Returns PMV value, PPD, and comfort category (cold/cool/neutral/warm/hot).",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "Zone name to check comfort for.",
                    },
                },
                "required": ["zone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_energy_summary",
            "description": "Get energy consumption summary for a time period. Returns total kWh, peak kW demand, and cost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["hour", "day", "week"],
                        "description": "Time period for the summary.",
                    },
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_setpoints",
            "description": "Update HVAC thermostat setpoints for a zone. Heating setpoint must be 18-24°C, cooling setpoint must be 22-28°C, and cooling must be ≥ heating + 1°C.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "Zone name to update.",
                    },
                    "heating_setpoint_c": {
                        "type": "number",
                        "description": "New heating setpoint in °C (18-24).",
                    },
                    "cooling_setpoint_c": {
                        "type": "number",
                        "description": "New cooling setpoint in °C (22-28).",
                    },
                },
                "required": ["zone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_lighting",
            "description": "Adjust lighting level in a zone. 0.0 = off, 1.0 = full brightness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "Zone name.",
                    },
                    "dimming_fraction": {
                        "type": "number",
                        "description": "Lighting level (0.0 to 1.0).",
                    },
                },
                "required": ["zone", "dimming_fraction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_schedule",
            "description": "Override a specific schedule value for a given hour.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule_name": {
                        "type": "string",
                        "description": "Name of the EnergyPlus schedule to modify.",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "Hour of the day (0-23) to modify.",
                    },
                    "value": {
                        "type": "number",
                        "description": "New value for the schedule at that hour.",
                    },
                },
                "required": ["schedule_name", "hour", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for upcoming hours from the weather file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours_ahead": {
                        "type": "integer",
                        "description": "Number of hours to forecast (1-24).",
                    },
                },
                "required": ["hours_ahead"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation_step",
            "description": "Advance the EnergyPlus simulation by a specified number of hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_hours": {
                        "type": "integer",
                        "description": "Number of hours to simulate (1-24).",
                    },
                },
                "required": ["duration_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_error_log",
            "description": "Read the EnergyPlus error log to check for warnings or errors.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def format_timestep_message(
    timestamp: str,
    outdoor_temp_c: float,
    outdoor_humidity_pct: float,
    solar_w_m2: float,
    zone_readings: list[dict],
    energy_hour_kwh: float,
    energy_today_kwh: float,
    demand_kw: float,
    tariff_per_kwh: float,
    tariff_period: str,
    carbon_gco2_kwh: float,
    occupancy_status: str,
    occupant_count: int,
    previous_action_results: str = "No previous actions.",
) -> str:
    """Format a per-timestep user message for the LLM."""
    # Format zone readings
    zone_text_parts = []
    for zr in zone_readings:
        zone_text_parts.append(
            ZONE_READING_TEMPLATE.format(
                zone_name=zr.get("zone_name", "Unknown"),
                temp_c=zr.get("temp_c", 0),
                humidity_pct=zr.get("humidity_pct", 0),
                pmv=zr.get("pmv", 0),
                comfort_category=zr.get("comfort_category", "unknown"),
                ppd=zr.get("ppd", 0),
                heating_sp=zr.get("heating_sp", 21),
                cooling_sp=zr.get("cooling_sp", 24),
            )
        )

    return TIMESTEP_MESSAGE_TEMPLATE.format(
        timestamp=timestamp,
        outdoor_temp_c=outdoor_temp_c,
        outdoor_humidity_pct=outdoor_humidity_pct,
        solar_w_m2=solar_w_m2,
        zone_readings="\n".join(zone_text_parts),
        energy_hour_kwh=energy_hour_kwh,
        energy_today_kwh=energy_today_kwh,
        demand_kw=demand_kw,
        tariff_per_kwh=tariff_per_kwh,
        tariff_period=tariff_period,
        carbon_gco2_kwh=carbon_gco2_kwh,
        occupancy_status=occupancy_status,
        occupant_count=occupant_count,
        previous_action_results=previous_action_results,
    )
