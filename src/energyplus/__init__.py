"""
EnergyPlus Integration Module
================================
Provides Python wrappers for executing, parsing, and actuating
EnergyPlus building energy simulations.

Exports:
    EnergyPlusRunner    - Manages subprocess lifecycle for E+ simulations
    SimulationConfig    - Dataclass holding simulation parameters
    SimulationResult    - Dataclass holding simulation output metadata
    EnergyPlusParser    - Parses .csv/.eso/.err output into DataFrames and KPIs
    SetpointActuator    - Injects HVAC/lighting setpoints into .idf files
    SetpointUpdate      - Dataclass describing a setpoint change request
"""

from src.energyplus.runner import EnergyPlusRunner, SimulationConfig, SimulationResult
from src.energyplus.parser import EnergyPlusParser
from src.energyplus.actuator import SetpointActuator, SetpointUpdate

__all__ = [
    "EnergyPlusRunner",
    "SimulationConfig",
    "SimulationResult",
    "EnergyPlusParser",
    "SetpointActuator",
    "SetpointUpdate",
]
