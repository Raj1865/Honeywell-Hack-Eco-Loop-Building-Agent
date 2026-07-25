"""
EnergyPlus Simulation Runner
=============================
Manages the lifecycle of EnergyPlus simulation processes.
Supports both full-run and segmented (step-by-step) execution modes.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SimulationConfig:
    """Configuration for an EnergyPlus simulation run."""
    idf_path: str
    weather_path: str
    output_dir: str
    energyplus_exe: str = "energyplus"
    design_day_only: bool = False
    annual: bool = False
    readvars: bool = True
    extra_args: list = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result from a completed EnergyPlus simulation."""
    success: bool
    return_code: int
    output_dir: str
    eso_path: Optional[str] = None
    csv_path: Optional[str] = None
    err_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None


class EnergyPlusRunner:
    """
    Manages EnergyPlus simulation execution.
    
    Handles:
    - Copying IDF to a working directory
    - Running EnergyPlus via subprocess
    - Monitoring process health
    - Collecting output file paths
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._working_dir: Optional[Path] = None

    def _prepare_working_dir(self, run_label: str = "run") -> Path:
        """Create a clean working directory for this simulation run."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        working_dir = Path(self.config.output_dir) / f"{run_label}_{timestamp}"
        working_dir.mkdir(parents=True, exist_ok=True)

        # Copy the IDF into the working directory
        src_idf = Path(self.config.idf_path)
        dst_idf = working_dir / src_idf.name
        shutil.copy2(src_idf, dst_idf)
        logger.info(f"Prepared working directory: {working_dir}")
        logger.info(f"Copied IDF: {src_idf.name}")

        self._working_dir = working_dir
        return working_dir

    def _build_command(self, idf_path: Path) -> list[str]:
        """Build the EnergyPlus command line."""
        cmd = [
            self.config.energyplus_exe,
            "--weather", str(Path(self.config.weather_path).resolve()),
            "--output-directory", str(self._working_dir.resolve()),
        ]

        if self.config.design_day_only:
            cmd.append("--design-day")

        if self.config.annual:
            cmd.append("--annual")

        if self.config.readvars:
            cmd.append("--readvars")

        cmd.extend(self.config.extra_args)
        cmd.append(str(idf_path.resolve()))
        return cmd

    def run(self, run_label: str = "sim", blocking: bool = True) -> SimulationResult:
        """
        Execute an EnergyPlus simulation.

        Args:
            run_label: Label for the output directory.
            blocking: If True, wait for simulation to complete.

        Returns:
            SimulationResult with output paths and status.
        """
        working_dir = self._prepare_working_dir(run_label)
        idf_path = working_dir / Path(self.config.idf_path).name
        cmd = self._build_command(idf_path)

        logger.info(f"Starting EnergyPlus: {' '.join(cmd)}")
        start_time = time.time()

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(working_dir),
            )

            if blocking:
                stdout, stderr = self._process.communicate()
                elapsed = time.time() - start_time
                return_code = self._process.returncode

                result = SimulationResult(
                    success=(return_code == 0),
                    return_code=return_code,
                    output_dir=str(working_dir),
                    elapsed_seconds=elapsed,
                )

                if return_code != 0:
                    result.error_message = stderr.decode("utf-8", errors="replace")
                    logger.error(f"EnergyPlus failed (code {return_code}): {result.error_message[:500]}")
                else:
                    logger.success(f"EnergyPlus completed in {elapsed:.1f}s")

                # Locate output files
                result.eso_path = self._find_file(working_dir, "eplusout.eso")
                result.csv_path = self._find_file(working_dir, "eplusout.csv")
                result.err_path = self._find_file(working_dir, "eplusout.err")

                return result
            else:
                logger.info("EnergyPlus started in background")
                return SimulationResult(
                    success=True,
                    return_code=-1,
                    output_dir=str(working_dir),
                )

        except FileNotFoundError:
            elapsed = time.time() - start_time
            msg = f"EnergyPlus executable not found: {self.config.energyplus_exe}"
            logger.error(msg)
            return SimulationResult(
                success=False,
                return_code=-1,
                output_dir=str(working_dir),
                elapsed_seconds=elapsed,
                error_message=msg,
            )

    def is_running(self) -> bool:
        """Check if the simulation process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def kill(self):
        """Kill the running simulation process."""
        if self._process and self.is_running():
            self._process.kill()
            logger.warning("EnergyPlus process killed")

    def get_error_log(self, max_lines: int = 50) -> str:
        """Read the last N lines of the EnergyPlus error log."""
        if self._working_dir is None:
            return "No simulation has been run yet."

        err_path = self._working_dir / "eplusout.err"
        if not err_path.exists():
            return "Error log file not found."

        lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    @staticmethod
    def _find_file(directory: Path, filename: str) -> Optional[str]:
        """Find a file in the output directory."""
        target = directory / filename
        if target.exists():
            return str(target)
        # Also check for ReadVarsESO-generated CSV
        if filename == "eplusout.csv":
            alt = directory / "eplusout.csv"
            if alt.exists():
                return str(alt)
        return None


def run_baseline(config_path: str = "config/settings.yaml") -> SimulationResult:
    """Convenience function to run a baseline simulation using settings.yaml."""
    import yaml

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    ep = settings["energyplus"]
    sim_config = SimulationConfig(
        idf_path=ep["baseline_idf"],
        weather_path=ep["weather_file"],
        output_dir="data/baseline_results",
        energyplus_exe=ep["executable"],
    )

    runner = EnergyPlusRunner(sim_config)
    return runner.run(run_label="baseline")
