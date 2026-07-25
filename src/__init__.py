# Eco-Loop Building Agents
"""
Autonomous closed-loop building energy optimization using EnergyPlus + LLM.

Modules:
    energyplus  - EnergyPlus simulation runner, output parser, and setpoint actuator
    agent       - LLM client, orchestrator state machine, memory manager, and prompts
    mcp_server  - Model Context Protocol server exposing building tools to the LLM
    dashboard   - Real-time Plotly Dash analytics interface
"""

__version__ = "1.0.0"
__author__ = "Raj Kokate"
__project__ = "Eco-Loop Building Agent — Honeywell Automation Hackathon 2026"
