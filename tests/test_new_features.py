"""
Automated Tests for Hardware HIL, OpenADR 2.0b & Swarm Orchestrator
====================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.bacnet_driver import BACnetHILDriver, DriverMode
from src.grid.openadr_client import OpenADRClient, DREventLevel
from src.agent.swarm_orchestrator import SwarmCoordinatorSupervisor, ZoneWorkerAgent


def test_bacnet_driver():
    print("\n--- Testing BACnet/IP HIL Driver ---")
    driver = BACnetHILDriver(mode=DriverMode.HARDWARE_IN_THE_LOOP, ip_address="192.168.1.105", device_id=2001)
    assert driver.connect() == True

    temp = driver.read_property("CORE_ZN_temp")
    assert temp == 22.5
    print(f"  [PASS] Read BACnet property CORE_ZN_temp = {temp}°C")

    wrote = driver.write_property("CORE_ZN_cool_sp", 24.5)
    assert wrote == True
    assert driver.read_property("CORE_ZN_cool_sp") == 24.5
    print(f"  [PASS] Write BACnet property CORE_ZN_cool_sp = 24.5°C")

    status = driver.get_status()
    assert status["mode"] == "HARDWARE_IN_THE_LOOP"
    print(f"  [PASS] BACnet status verified: {status}")


def test_openadr_client():
    print("\n--- Testing OpenADR 2.0b Client ---")
    client = OpenADRClient(ven_id="VEN_HONEYWELL_TEST")

    events = client.poll_events(current_hour=15, outdoor_temp_c=29.0)
    assert len(events) == 1
    assert events[0].level == DREventLevel.HIGH
    print(f"  [PASS] Active OpenADR event detected: Level={events[0].level.name}, Rate=${events[0].payload_value}/kWh")

    carbon = client.get_grid_carbon_intensity()
    assert carbon > 0
    print(f"  [PASS] Grid carbon intensity: {carbon} gCO2/kWh")

    adj = client.get_setpoint_adjustment(events[0])
    assert adj["cooling_delta_c"] == 2.5
    print(f"  [PASS] OpenADR setpoint adjustment calculated: {adj}")


def test_swarm_orchestrator():
    print("\n--- Testing Swarm Coordinator & Zone Worker Swarm ---")
    zones = ["CORE_ZN", "PERIMETER_ZN_1", "PERIMETER_ZN_2"]
    supervisor = SwarmCoordinatorSupervisor(zone_ids=zones, max_peak_kw_limit=15.0)

    sensor_data = {
        "CORE_ZN:Zone Mean Air Temperature [C](TimeStep)": 26.5,
        "CORE_ZN:Zone People Occupant Count [](TimeStep)": 5,
        "PERIMETER_ZN_1:Zone People Occupant Count [](TimeStep)": 0,
    }

    # Test local worker comfort evaluation & central supervisor arbitration
    proposals = supervisor.coordinate_step(sensor_data, openadr_demand_shed=True)
    assert len(proposals) == 3

    core_prop = [p for p in proposals if p.zone == "CORE_ZN"][0]
    # Occupied core zone: 24.0°C base + 2.0°C OpenADR shed override = 26.0°C
    assert core_prop.cooling_setpoint_c == 26.0
    print(f"  [PASS] Swarm Coordinator arbitrated CORE_ZN proposal: cooling_sp={core_prop.cooling_setpoint_c}°C")


if __name__ == "__main__":
    test_bacnet_driver()
    test_openadr_client()
    test_swarm_orchestrator()
    print("\n============================================================================")
    print("  VERDICT: ALL NEW ENTERPRISE FEATURES PASSED (100%)")
    print("============================================================================\n")
