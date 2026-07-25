"""
Run Anomaly Analysis
======================
Analyzes the baseline simulation output for equipment anomalies,
sensor faults, and operational issues. Generates anomaly_report.json.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.energyplus.parser import EnergyPlusParser
from src.agent.anomaly_detector import AnomalyDetector
from loguru import logger


def find_latest_baseline() -> Path:
    """Find the most recent baseline results directory."""
    base = Path("data/baseline_results")
    subdirs = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []
    if not subdirs:
        raise FileNotFoundError("No baseline results found. Run scripts/run_baseline.py first.")
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def main():
    logger.info("=" * 60)
    logger.info("Running Predictive Anomaly Analysis")
    logger.info("=" * 60)

    # Find latest baseline
    baseline_dir = find_latest_baseline()
    logger.info(f"Analyzing: {baseline_dir}")

    # Parse the simulation output
    parser = EnergyPlusParser(str(baseline_dir))
    df = parser.parse_csv()
    baseline_kpis = parser.compute_kpis()

    # Run anomaly detection
    detector = AnomalyDetector()
    anomalies = detector.analyze_dataframe(df, baseline_kpis)

    # Summary
    counts = detector.get_severity_counts()
    logger.info(f"Anomalies found: {len(anomalies)}")
    logger.info(f"  CRITICAL: {counts['CRITICAL']}")
    logger.info(f"  HIGH:     {counts['HIGH']}")
    logger.info(f"  MEDIUM:   {counts['MEDIUM']}")
    logger.info(f"  LOW:      {counts['LOW']}")

    # Print top anomalies
    for a in anomalies[:10]:
        logger.warning(f"[{a.severity}] {a.category} — {a.zone}: {a.description[:100]}")
        logger.info(f"  → Action: {a.recommended_action[:100]}")

    # Save report
    detector.save_report("data/anomaly_report.json")
    logger.success(f"Report saved to data/anomaly_report.json")


if __name__ == "__main__":
    main()
