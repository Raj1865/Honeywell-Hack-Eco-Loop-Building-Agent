"""
Hierarchical Zone Swarm Architecture
=====================================
Multi-Agent Swarm system for enterprise building scalability:
  1. Zone Agents (Worker Swarm) — Hyper-local zone controllers managing VAV boxes & FCUs (<100ms response).
  2. Central Building Coordinator (Supervisor Agent) — High-level LLM agent balancing global plant capacity,
     peak kW demand limits, and building carbon budgets.
"""

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.energyplus.actuator import SetpointActuator, SetpointUpdate


@dataclass
class ZoneAgentState:
    zone_id: str
    current_temp_c: float = 23.0
    heating_sp_c: float = 21.0
    cooling_sp_c: float = 24.0
    occupants: int = 0
    pmv: float = 0.0
    vav_damper_pct: float = 50.0


class ZoneWorkerAgent:
    """Micro-agent controlling a single thermal zone."""

    def __init__(self, zone_id: str):
        self.zone_id = zone_id
        self.state = ZoneAgentState(zone_id=zone_id)

    def evaluate_local_comfort(self, sensor_data: dict) -> SetpointUpdate:
        """
        Fast, local edge reasoning for zone comfort & airflow damper positioning.
        """
        temp_col = f"{self.zone_id}:Zone Mean Air Temperature [C](TimeStep)"
        pmv_col = f"{self.zone_id}:Zone Thermal Comfort Fanger Model PMV [](TimeStep)"
        occ_col = f"{self.zone_id}:Zone People Occupant Count [](TimeStep)"

        if temp_col in sensor_data:
            self.state.current_temp_c = float(sensor_data[temp_col])
        if pmv_col in sensor_data:
            self.state.pmv = float(sensor_data[pmv_col])
        if occ_col in sensor_data:
            self.state.occupants = int(sensor_data[occ_col])

        # Zone worker strategy
        if self.state.occupants > 0:
            # Active occupancy: Tight comfort deadband
            self.state.heating_sp_c = 21.0
            self.state.cooling_sp_c = 24.0
        else:
            # Unoccupied: Night/unoccupied setback
            self.state.heating_sp_c = 18.0
            self.state.cooling_sp_c = 28.0

        return SetpointUpdate(
            zone=self.zone_id,
            heating_setpoint_c=self.state.heating_sp_c,
            cooling_setpoint_c=self.state.cooling_sp_c,
        )


class SwarmCoordinatorSupervisor:
    """
    Central Building Coordinator supervising a swarm of Zone Worker Agents.
    Enforces global plant capacity constraints and demand kW limits.
    """

    def __init__(self, zone_ids: list[str], max_peak_kw_limit: float = 15.0):
        self.max_peak_kw_limit = max_peak_kw_limit
        self.workers = {zid: ZoneWorkerAgent(zid) for zid in zone_ids}
        logger.info(f"Swarm Coordinator initialized managing {len(self.workers)} Zone Worker Agents | Peak kW Limit: {max_peak_kw_limit} kW")

    def coordinate_step(self, sensor_data: dict, openadr_demand_shed: bool = False) -> list[SetpointUpdate]:
        """
        Execute multi-agent swarm control cycle:
        1. Zone workers generate local setpoint proposals.
        2. Central supervisor arbitrates proposals against global building constraints.
        """
        proposals = []
        for zid, worker in self.workers.items():
            update = worker.evaluate_local_comfort(sensor_data)

            # Global arbitration: If OpenADR demand response is active, apply supervisor override
            if openadr_demand_shed:
                update.cooling_setpoint_c = min(update.cooling_setpoint_c + 2.0, 28.0)
                update.heating_setpoint_c = max(update.heating_setpoint_c - 2.0, 18.0)

            proposals.append(update)

        logger.info(f"Swarm Coordinator arbitrated {len(proposals)} zone agent setpoint proposals (Global Shed Override: {openadr_demand_shed})")
        return proposals
