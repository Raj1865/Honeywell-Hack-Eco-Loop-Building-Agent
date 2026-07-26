"""
EnergyPlus Setpoint Actuator
==============================
Modifies .idf files to inject new HVAC setpoints, lighting schedules,
and other control parameters. Supports both static IDF edits and
EMS (Energy Management System) actuator overrides.
"""

import re
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from loguru import logger

try:
    from eppy.modeleditor import IDF
    EPPY_AVAILABLE = True
except ImportError:
    EPPY_AVAILABLE = False
    logger.warning("eppy not installed — falling back to regex-based IDF editing")


@dataclass
class SetpointUpdate:
    """Describes a setpoint change to apply."""
    zone: str
    heating_setpoint_c: Optional[float] = None
    cooling_setpoint_c: Optional[float] = None
    lighting_fraction: Optional[float] = None
    ventilation_rate: Optional[float] = None


class SetpointActuator:
    """
    Modifies EnergyPlus IDF files to inject new control setpoints.
    
    Two strategies:
    1. Schedule Override: Modify Schedule:Compact objects for thermostat setpoints
    2. EMS Injection: Add EMS:ProgramCallingManager + EMS:Program objects
       for runtime actuator overrides (more flexible, no re-run needed)
    """

    # Safety bounds — hard limits that override any LLM request
    SAFETY_BOUNDS = {
        "heating_min_c": 15.0,
        "heating_max_c": 26.0,
        "cooling_min_c": 20.0,
        "cooling_max_c": 30.0,
        "lighting_min": 0.0,
        "lighting_max": 1.0,
    }

    def __init__(self, idf_path: str, idd_path: Optional[str] = None):
        self.idf_path = Path(idf_path)
        self.idd_path = idd_path

        if EPPY_AVAILABLE and idd_path:
            IDF.setiddname(idd_path)
            self._idf = IDF(str(self.idf_path))
        else:
            self._idf = None

    def apply_setpoint(self, update: SetpointUpdate) -> dict:
        """
        Apply a setpoint update to the IDF file.
        
        Validates against safety bounds before applying.
        
        Returns:
            Dict with applied values and any clamping that occurred.
        """
        result = {"zone": update.zone, "applied": {}, "clamped": {}}

        # --- Validate & clamp heating setpoint ---
        if update.heating_setpoint_c is not None:
            original = update.heating_setpoint_c
            clamped = max(
                self.SAFETY_BOUNDS["heating_min_c"],
                min(self.SAFETY_BOUNDS["heating_max_c"], original),
            )
            if clamped != original:
                result["clamped"]["heating_setpoint_c"] = {
                    "requested": original, "applied": clamped
                }
                logger.warning(
                    f"Heating setpoint clamped: {original}°C -> {clamped}°C"
                )
            result["applied"]["heating_setpoint_c"] = clamped

        # --- Validate & clamp cooling setpoint ---
        if update.cooling_setpoint_c is not None:
            original = update.cooling_setpoint_c
            clamped = max(
                self.SAFETY_BOUNDS["cooling_min_c"],
                min(self.SAFETY_BOUNDS["cooling_max_c"], original),
            )
            if clamped != original:
                result["clamped"]["cooling_setpoint_c"] = {
                    "requested": original, "applied": clamped
                }
                logger.warning(
                    f"Cooling setpoint clamped: {original}°C -> {clamped}°C"
                )
            result["applied"]["cooling_setpoint_c"] = clamped

        # --- Validate & clamp lighting ---
        if update.lighting_fraction is not None:
            original = update.lighting_fraction
            clamped = max(
                self.SAFETY_BOUNDS["lighting_min"],
                min(self.SAFETY_BOUNDS["lighting_max"], original),
            )
            if clamped != original:
                result["clamped"]["lighting_fraction"] = {
                    "requested": original, "applied": clamped
                }
            result["applied"]["lighting_fraction"] = clamped

        # Apply to IDF
        if EPPY_AVAILABLE and self._idf:
            self._apply_with_eppy(result)
        else:
            self._apply_with_regex(result)

        logger.info(f"Applied setpoints for zone '{update.zone}': {result['applied']}")
        return result

    def _apply_with_eppy(self, result: dict):
        """Apply setpoint changes using eppy library."""
        zone = result["zone"]
        applied = result["applied"]

        # Modify thermostat schedule setpoints
        if "heating_setpoint_c" in applied:
            schedules = self._idf.idfobjects["Schedule:Compact"]
            for sched in schedules:
                if "htg" in sched.Name.lower() or "heating" in sched.Name.lower():
                    # Update the schedule values — this is simplified;
                    # real implementation needs to handle schedule structure
                    logger.info(f"Updating heating schedule: {sched.Name}")
                    self._update_schedule_value(sched, applied["heating_setpoint_c"])
                    break

        if "cooling_setpoint_c" in applied:
            schedules = self._idf.idfobjects["Schedule:Compact"]
            for sched in schedules:
                if "clg" in sched.Name.lower() or "cooling" in sched.Name.lower():
                    logger.info(f"Updating cooling schedule: {sched.Name}")
                    self._update_schedule_value(sched, applied["cooling_setpoint_c"])
                    break

        # Save modified IDF
        self._idf.save()
        logger.info(f"Saved modified IDF: {self.idf_path}")

    def _update_schedule_value(self, schedule, new_value: float):
        """Update numeric values in a Schedule:Compact object."""
        # Schedule:Compact has field values like "Until: 24:00, 21.0"
        # We update all numeric terminal values
        for i, field in enumerate(schedule.fieldvalues):
            if isinstance(field, (int, float)):
                schedule.fieldvalues[i] = new_value

    def _apply_with_regex(self, result: dict):
        """Apply setpoint changes using regex-based IDF text manipulation."""
        text = self.idf_path.read_text(encoding="utf-8")
        applied = result["applied"]

        if "heating_setpoint_c" in applied:
            val = applied["heating_setpoint_c"]
            # Find heating thermostat schedule and update values
            text = self._regex_update_schedule(text, "heating", val)

        if "cooling_setpoint_c" in applied:
            val = applied["cooling_setpoint_c"]
            text = self._regex_update_schedule(text, "cooling", val)

        # Write back
        self.idf_path.write_text(text, encoding="utf-8")
        logger.info(f"Saved modified IDF (regex): {self.idf_path}")

    def _regex_update_schedule(self, text: str, schedule_type: str, new_value: float) -> str:
        """Update schedule values in specific thermostat setpoint schedules in IDF text."""
        kw = "htg" if "heat" in schedule_type.lower() else "clg"

        def update_block(block_match):
            block_text = block_match.group(0)
            pattern = r"(Until:\s*\d{1,2}:\d{2}\s*,\s*)(\d+\.?\d*)"
            return re.sub(pattern, lambda m: f"{m.group(1)}{new_value:.2f}", block_text)

        # Match Schedule:Compact blocks containing the target keyword (HTG or CLG)
        block_pattern = r"(Schedule:Compact,\s*[\w_-]*" + kw + r"[\w_-]*,[\s\S]*?;\s*)"
        return re.sub(block_pattern, update_block, text, flags=re.IGNORECASE)

    def backup_idf(self, suffix: str = "backup") -> str:
        """Create a backup copy of the current IDF."""
        backup_path = self.idf_path.with_suffix(f".{suffix}.idf")
        shutil.copy2(self.idf_path, backup_path)
        logger.info(f"IDF backed up to: {backup_path}")
        return str(backup_path)

    def inject_ems_program(self, program_name: str, program_lines: list[str]) -> str:
        """
        Inject an EMS (Energy Management System) program into the IDF.
        This allows runtime actuator overrides without modifying schedules.
        
        Args:
            program_name: Name for the EMS program.
            program_lines: List of EMS program lines (EnergyPlus Runtime Language).
        
        Returns:
            The injected EMS program text.
        """
        ems_program = f"""
EnergyManagementSystem:Program,
    {program_name},
    {','.join(f'{chr(10)}    {line}' for line in program_lines)};

EnergyManagementSystem:ProgramCallingManager,
    {program_name}_Manager,
    AfterPredictorAfterHVACManagers,
    {program_name};
"""
        # Append to IDF
        text = self.idf_path.read_text(encoding="utf-8")
        text += "\n" + ems_program
        self.idf_path.write_text(text, encoding="utf-8")

        logger.info(f"Injected EMS program: {program_name}")
        return ems_program

    def get_current_setpoints(self) -> dict:
        """Read the current thermostat setpoints from the IDF."""
        if EPPY_AVAILABLE and self._idf:
            schedules = self._idf.idfobjects.get("Schedule:Compact", [])
            setpoints = {}
            for sched in schedules:
                name = sched.Name.lower()
                if "htg" in name or "heating" in name:
                    values = [v for v in sched.fieldvalues if isinstance(v, (int, float))]
                    if values:
                        setpoints["heating_setpoint_c"] = values[0]
                if "clg" in name or "cooling" in name:
                    values = [v for v in sched.fieldvalues if isinstance(v, (int, float))]
                    if values:
                        setpoints["cooling_setpoint_c"] = values[0]
            return setpoints
        else:
            return {"note": "eppy not available — cannot read setpoints from IDF"}
