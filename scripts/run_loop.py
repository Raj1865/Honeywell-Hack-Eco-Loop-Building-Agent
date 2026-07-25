"""
Run Closed-Loop Optimization
==============================
One-click script to launch the full AI-driven closed-loop
building optimization pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import Orchestrator
from loguru import logger


def main():
    logger.add("data/loop_run_{time}.log", rotation="10 MB", level="DEBUG")
    logger.info("=" * 60)
    logger.info("Eco-Loop Closed-Loop Optimization")
    logger.info("=" * 60)

    orchestrator = Orchestrator(config_path="config/settings.yaml")

    if not orchestrator.setup():
        logger.error("Setup failed. Check:")
        logger.error("  1. Is Ollama running? (ollama serve)")
        logger.error("  2. Is the LLM model pulled? (ollama pull qwen2.5:7b-instruct)")
        logger.error("  3. Is EnergyPlus installed and path correct in config/settings.yaml?")
        logger.error("  4. Does models/baseline.idf exist?")
        sys.exit(1)

    # Run for 96 steps = 24 hours at 15-minute intervals
    # Increase for longer simulations
    orchestrator.run(max_steps=96)

    # Print final stats
    stats = orchestrator.memory.get_stats()
    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
