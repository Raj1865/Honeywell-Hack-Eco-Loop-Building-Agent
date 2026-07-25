"""
Generate Comparison Report
===========================
Post-hoc script that compares baseline and optimized simulation results
and generates a summary report with visualizations.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.energyplus.parser import EnergyPlusParser
from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("Generating Comparison Report")
    logger.info("=" * 60)

    baseline_dir = Path("data/baseline_results")
    optimized_dir = Path("data/optimized_results")

    # Find latest run directories
    baseline_runs = sorted(baseline_dir.glob("baseline_*")) if baseline_dir.exists() else []
    optimized_runs = sorted(optimized_dir.glob("sim_*")) if optimized_dir.exists() else []

    if not baseline_runs:
        logger.error("No baseline results found. Run `scripts/run_baseline.py` first.")
        sys.exit(1)

    if not optimized_runs:
        logger.error("No optimized results found. Run `scripts/run_loop.py` first.")
        sys.exit(1)

    # Parse latest runs
    baseline_parser = EnergyPlusParser(str(baseline_runs[-1]))
    optimized_parser = EnergyPlusParser(str(optimized_runs[-1]))

    baseline_kpis = baseline_parser.compute_kpis()
    optimized_kpis = optimized_parser.compute_kpis()

    # Compute deltas
    report = {
        "baseline": baseline_kpis,
        "optimized": optimized_kpis,
        "comparison": {},
    }

    b_kwh = baseline_kpis.get("total_kwh", 0)
    o_kwh = optimized_kpis.get("total_kwh", 0)

    if b_kwh and o_kwh:
        report["comparison"]["energy_savings_kwh"] = b_kwh - o_kwh
        report["comparison"]["energy_savings_pct"] = ((b_kwh - o_kwh) / b_kwh) * 100

    b_comfort = baseline_kpis.get("comfort_hours_pct", 0)
    o_comfort = optimized_kpis.get("comfort_hours_pct", 0)
    if b_comfort is not None and o_comfort is not None:
        report["comparison"]["comfort_improvement_pct"] = o_comfort - b_comfort

    # Save report
    report_path = Path("data/comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("  COMPARISON REPORT")
    print("=" * 50)
    print(f"\n  Baseline Energy:     {b_kwh:.1f} kWh")
    print(f"  Optimized Energy:    {o_kwh:.1f} kWh")
    if b_kwh:
        print(f"  Energy Savings:      {report['comparison'].get('energy_savings_pct', 0):.1f}%")
    print(f"\n  Baseline Comfort:    {b_comfort:.0f}%")
    print(f"  Optimized Comfort:   {o_comfort:.0f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
