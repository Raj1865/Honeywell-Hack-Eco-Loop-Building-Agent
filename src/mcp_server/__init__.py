"""
MCP Server Module
================================
Model Context Protocol (MCP) server that exposes EnergyPlus
simulation tools to the autonomous LLM agent.

The server registers tools such as read_sensors, update_setpoints,
adjust_lighting, and get_weather_forecast, allowing the LLM to
interact with the building simulation via standardized tool-calling.
"""

__all__: list[str] = []
