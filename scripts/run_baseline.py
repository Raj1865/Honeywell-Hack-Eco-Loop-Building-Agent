"""
Run Baseline Simulation
========================
One-click script to run the EnergyPlus baseline simulation
and compute reference KPIs.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.energyplus.runner import run_baseline
from src.energyplus.parser import EnergyPlusParser
from loguru import logger


def main():
    logger.add("data/baseline_run.log", level="DEBUG")
    logger.info("=" * 60)
    logger.info("Running Baseline EnergyPlus Simulation")
    logger.info("=" * 60)

    # Run simulation
    result = run_baseline()

    if not result.success:
        logger.error(f"Baseline simulation failed: {result.error_message}")
        sys.exit(1)

    logger.info(f"Simulation completed in {result.elapsed_seconds:.1f}s")
    logger.info(f"Output directory: {result.output_dir}")

    # Parse results and compute KPIs
    parser = EnergyPlusParser(result.output_dir)
    kpis = parser.compute_kpis()

    # Save KPIs
    kpi_file = Path(result.output_dir) / "baseline_kpis.json"
    with open(kpi_file, "w") as f:
        json.dump(kpis, f, indent=2, default=str)

    logger.info(f"Baseline KPIs saved to {kpi_file}")
    logger.info(f"KPIs: {json.dumps(kpis, indent=2, default=str)}")

    # Check error log
    err_info = parser.parse_error_log()
    if err_info.get("fatal_errors", 0) > 0:
        logger.error("Baseline had fatal errors — check eplusout.err")
    elif err_info.get("severe_errors", 0) > 0:
        logger.warning(f"Baseline had {err_info['severe_errors']} severe errors")
    else:
        logger.success("Baseline completed with no severe errors")

    return kpis


if __name__ == "__main__":
    main()
