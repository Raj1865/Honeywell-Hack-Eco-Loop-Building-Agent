"""
OpenADR 2.0b Dynamic Demand Response & Grid Carbon Client
===========================================================
Connects to Utility Virtual Top Node (VTN) to ingest real-time demand response
signals (OpenADR 2.0b EiEvent) and grid carbon intensity metrics.

Triggers pre-cooling thermal energy storage and load-shedding strategies
before peak pricing hours or grid stress events.
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


class DREventLevel(Enum):
    NORMAL = 0      # Normal operation
    MODERATE = 1    # Soft load shed / minor pre-cooling
    HIGH = 2        # Active demand response shed
    CRITICAL = 3    # Emergency load shed (max setback)


@dataclass
class OpenADREvent:
    event_id: str
    signal_type: str             # level, price, carbon
    level: DREventLevel
    start_time: str
    duration_minutes: int
    payload_value: float          # e.g., $0.45/kWh tariff or level 2
    recommended_strategy: str


class OpenADRClient:
    """
    OpenADR 2.0b VEN (Virtual End Node) client.
    Listens for dynamic grid signals and dictates energy-shedding strategies.
    """

    def __init__(self, vtn_url: str = "https://vtn.grid-utility.com/OpenADR2/Simple/2.0b", ven_id: str = "VEN_HONEYWELL_BLDG_01"):
        self.vtn_url = vtn_url
        self.ven_id = ven_id
        self._active_events: list[OpenADREvent] = []
        logger.info(f"OpenADR 2.0b VEN Client initialized [VEN ID: {ven_id}] -> VTN Endpoint: {vtn_url}")

    def poll_events(self, current_hour: int = 14, outdoor_temp_c: float = 28.0) -> list[OpenADREvent]:
        """
        Poll VTN for active or upcoming Demand Response signals.
        Simulates grid event triggers during peak afternoon pricing (14:00 - 18:00).
        """
        events = []

        # Peak Tariff & Grid Stress Event (14:00 - 17:00)
        if 14 <= current_hour <= 17:
            level = DREventLevel.HIGH if outdoor_temp_c > 27.0 else DREventLevel.MODERATE
            events.append(OpenADREvent(
                event_id=f"DR_EVT_{datetime.now().strftime('%Y%m%d')}_01",
                signal_type="price_and_carbon",
                level=level,
                start_time=f"{current_hour:02d}:00",
                duration_minutes=180,
                payload_value=0.45,  # $0.45/kWh peak rate
                recommended_strategy="PRE_COOLING_THEN_SETBACK" if level == DREventLevel.HIGH else "MODERATE_DEADBAND_EXPANSION"
            ))
            logger.warning(
                f"⚡ [OpenADR 2.0b SIGNAL] Active Demand Response Event detected! "
                f"Level: {level.name} | Rate: $0.45/kWh | Strategy: {events[-1].recommended_strategy}"
            )
        else:
            logger.info("⚡ [OpenADR 2.0b SIGNAL] Grid status normal — off-peak rate ($0.12/kWh).")

        self._active_events = events
        return events

    def get_grid_carbon_intensity(self) -> float:
        """Return live grid carbon intensity in gCO2 / kWh."""
        # Simulated grid carbon intensity curve (higher during peak fossil generation)
        hour = datetime.now().hour
        if 14 <= hour <= 19:
            return 480.0  # High fossil peaking plant intensity
        return 210.0      # Off-peak hydro/renewable intensity

    def get_setpoint_adjustment(self, event: OpenADREvent) -> dict:
        """
        Calculate recommended thermostat setpoint adjustments based on OpenADR event.
        """
        if event.level == DREventLevel.HIGH or event.level == DREventLevel.CRITICAL:
            return {
                "heating_delta_c": -2.0,  # Lower heating SP
                "cooling_delta_c": +2.5,  # Raise cooling SP (shed load)
                "lighting_factor": 0.6,   # Dim lights by 40%
            }
        elif event.level == DREventLevel.MODERATE:
            return {
                "heating_delta_c": -1.0,
                "cooling_delta_c": +1.5,
                "lighting_factor": 0.8,
            }
        return {"heating_delta_c": 0.0, "cooling_delta_c": 0.0, "lighting_factor": 1.0}
