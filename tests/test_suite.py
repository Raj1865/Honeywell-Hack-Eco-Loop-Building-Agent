"""
Eco-Loop Building Agent — Production Test Suite
=================================================
Comprehensive automated testing for the Honeywell Automation hackathon.
Covers unit, integration, system, and regression tests across all modules.

Usage:
    python tests/test_suite.py
"""

import sys
import os
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_s: float
    detail: str = ""
    error: str = ""


class TestRunner:
    """Lightweight test runner that collects results and writes a report."""

    def __init__(self):
        self.results: list[TestResult] = []
        self._start = time.time()

    def run(self, name: str, category: str, fn):
        t0 = time.time()
        try:
            detail = fn()
            elapsed = time.time() - t0
            self.results.append(TestResult(name, category, True, elapsed, detail or "OK"))
        except Exception as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            self.results.append(TestResult(name, category, False, elapsed, error=f"{exc}\n{tb}"))

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        wall = time.time() - self._start
        return (
            f"Total: {total} | Passed: {passed} | Failed: {failed} | "
            f"Wall time: {wall:.2f}s"
        )

    def write_report(self, path: str):
        """Write a human-readable result.txt report."""
        lines = []
        lines.append("=" * 76)
        lines.append("  Eco-Loop Building Agent — Automated Test Report")
        lines.append("  Honeywell Automation Hackathon 2026")
        lines.append("=" * 76)
        lines.append(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Platform  : {sys.platform} | Python {sys.version.split()[0]}")
        lines.append(f"  {self.summary()}")
        lines.append("=" * 76)
        lines.append("")

        # Group by category
        categories = dict.fromkeys(r.category for r in self.results)
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            cat_passed = sum(1 for r in cat_results if r.passed)
            lines.append("-" * 76)
            lines.append(f"  {cat}  ({cat_passed}/{len(cat_results)} passed)")
            lines.append("-" * 76)
            for r in cat_results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(f"  [{status}] {r.name}  ({r.duration_s:.3f}s)")
                if r.detail and r.detail != "OK":
                    for dl in r.detail.strip().split("\n"):
                        lines.append(f"         {dl}")
                if r.error:
                    for el in r.error.strip().split("\n")[:6]:
                        lines.append(f"         ERROR: {el}")
            lines.append("")

        # Final verdict
        all_passed = all(r.passed for r in self.results)
        lines.append("=" * 76)
        if all_passed:
            lines.append("  VERDICT: ALL TESTS PASSED")
        else:
            lines.append("  VERDICT: SOME TESTS FAILED — review errors above")
        lines.append("=" * 76)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Also print to stdout
        print("\n".join(lines))


# ======================================================================
# 1  UNIT TESTS
# ======================================================================

def unit_tests(runner: TestRunner):

    # ---- 1.1  SimulationConfig dataclass ----------------------------------
    def test_simulation_config():
        from src.energyplus.runner import SimulationConfig
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="weather.epw",
            output_dir="data/out",
        )
        assert cfg.design_day_only is False
        assert cfg.readvars is True
        assert cfg.extra_args == []
        return f"Defaults correct: design_day_only={cfg.design_day_only}, readvars={cfg.readvars}"

    runner.run("SimulationConfig defaults", "1. Unit Tests — EnergyPlus Runner", test_simulation_config)

    # ---- 1.2  SimulationResult dataclass ----------------------------------
    def test_simulation_result():
        from src.energyplus.runner import SimulationResult
        r = SimulationResult(success=True, return_code=0, output_dir="/tmp")
        assert r.success is True
        assert r.csv_path is None
        return "SimulationResult initialises with expected defaults"

    runner.run("SimulationResult defaults", "1. Unit Tests — EnergyPlus Runner", test_simulation_result)

    # ---- 1.3  Runner._find_file ------------------------------------------
    def test_find_file():
        from src.energyplus.runner import EnergyPlusRunner
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "eplusout.csv"
            p.write_text("header\n1,2,3")
            result = EnergyPlusRunner._find_file(Path(td), "eplusout.csv")
            assert result is not None
            missing = EnergyPlusRunner._find_file(Path(td), "nonexistent.csv")
            assert missing is None
        return "Found existing file, returned None for missing file"

    runner.run("Runner._find_file", "1. Unit Tests — EnergyPlus Runner", test_find_file)

    # ---- 1.4  SetpointUpdate dataclass ------------------------------------
    def test_setpoint_update():
        from src.energyplus.actuator import SetpointUpdate
        u = SetpointUpdate(zone="Core_ZN", heating_setpoint_c=21.0, cooling_setpoint_c=24.0)
        assert u.lighting_fraction is None
        assert u.ventilation_rate is None
        return f"zone={u.zone}, heating={u.heating_setpoint_c}, cooling={u.cooling_setpoint_c}"

    runner.run("SetpointUpdate dataclass", "1. Unit Tests — Actuator", test_setpoint_update)

    # ---- 1.5  Safety bounds clamping (heating) ----------------------------
    def test_heating_clamp():
        from src.energyplus.actuator import SetpointActuator, SetpointUpdate
        with tempfile.NamedTemporaryFile(suffix=".idf", delete=False, mode="w") as f:
            f.write("! minimal idf\n")
            tmp = f.name
        try:
            act = SetpointActuator(tmp)
            update = SetpointUpdate(zone="Test", heating_setpoint_c=10.0)  # below min 15
            result = act.apply_setpoint(update)
            assert result["applied"]["heating_setpoint_c"] == 15.0
            assert "heating_setpoint_c" in result["clamped"]
            return f"Requested 10°C, clamped to {result['applied']['heating_setpoint_c']}°C"
        finally:
            os.unlink(tmp)

    runner.run("Safety clamp — heating below min", "1. Unit Tests — Actuator", test_heating_clamp)

    # ---- 1.6  Safety bounds clamping (cooling) ----------------------------
    def test_cooling_clamp():
        from src.energyplus.actuator import SetpointActuator, SetpointUpdate
        with tempfile.NamedTemporaryFile(suffix=".idf", delete=False, mode="w") as f:
            f.write("! minimal idf\n")
            tmp = f.name
        try:
            act = SetpointActuator(tmp)
            update = SetpointUpdate(zone="Test", cooling_setpoint_c=35.0)  # above max 30
            result = act.apply_setpoint(update)
            assert result["applied"]["cooling_setpoint_c"] == 30.0
            return f"Requested 35°C, clamped to {result['applied']['cooling_setpoint_c']}°C"
        finally:
            os.unlink(tmp)

    runner.run("Safety clamp — cooling above max", "1. Unit Tests — Actuator", test_cooling_clamp)

    # ---- 1.7  Safety bounds — valid value passes through ------------------
    def test_valid_setpoint_passthrough():
        from src.energyplus.actuator import SetpointActuator, SetpointUpdate
        with tempfile.NamedTemporaryFile(suffix=".idf", delete=False, mode="w") as f:
            f.write("! minimal idf\n")
            tmp = f.name
        try:
            act = SetpointActuator(tmp)
            update = SetpointUpdate(zone="Test", heating_setpoint_c=21.0, cooling_setpoint_c=24.0)
            result = act.apply_setpoint(update)
            assert result["applied"]["heating_setpoint_c"] == 21.0
            assert result["applied"]["cooling_setpoint_c"] == 24.0
            assert result["clamped"] == {}
            return "21°C / 24°C passed through without clamping"
        finally:
            os.unlink(tmp)

    runner.run("Valid setpoints — no clamping", "1. Unit Tests — Actuator", test_valid_setpoint_passthrough)

    # ---- 1.8  Lighting fraction clamping ----------------------------------
    def test_lighting_clamp():
        from src.energyplus.actuator import SetpointActuator, SetpointUpdate
        with tempfile.NamedTemporaryFile(suffix=".idf", delete=False, mode="w") as f:
            f.write("! minimal idf\n")
            tmp = f.name
        try:
            act = SetpointActuator(tmp)
            update = SetpointUpdate(zone="Test", lighting_fraction=1.5)
            result = act.apply_setpoint(update)
            assert result["applied"]["lighting_fraction"] == 1.0
            return f"Requested 1.5, clamped to {result['applied']['lighting_fraction']}"
        finally:
            os.unlink(tmp)

    runner.run("Safety clamp — lighting above 1.0", "1. Unit Tests — Actuator", test_lighting_clamp)

    # ---- 1.9  ContextMemory sliding window --------------------------------
    def test_context_memory_sliding_window():
        from src.agent.memory import ContextMemory, TimestepRecord
        mem = ContextMemory(max_recent=3, max_actions=5)
        for i in range(7):
            mem.add_timestep(TimestepRecord(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                sensor_data={"temp": 22 + i},
                energy_kwh=float(i),
                pmv=0.1 * i,
            ))
        assert len(mem.recent) == 3, f"Expected 3 recent, got {len(mem.recent)}"
        assert mem._total_steps == 7
        stats = mem.get_stats()
        assert stats["total_steps"] == 7
        assert stats["recent_steps"] == 3
        return f"7 steps added, window holds {len(mem.recent)}, total tracked = {mem._total_steps}"

    runner.run("ContextMemory sliding window", "1. Unit Tests — Memory", test_context_memory_sliding_window)

    # ---- 1.10  ContextMemory clear ----------------------------------------
    def test_context_memory_clear():
        from src.agent.memory import ContextMemory, TimestepRecord
        mem = ContextMemory(max_recent=5)
        mem.add_timestep(TimestepRecord(timestamp="t0", sensor_data={}, energy_kwh=1.0, pmv=0.2))
        mem.add_action({"tool": "update_setpoints"})
        mem.clear()
        stats = mem.get_stats()
        assert stats["total_steps"] == 0
        assert stats["total_actions"] == 0
        return "Memory cleared: total_steps=0, total_actions=0"

    runner.run("ContextMemory clear", "1. Unit Tests — Memory", test_context_memory_clear)

    # ---- 1.11  ContextMemory summary generation ---------------------------
    def test_memory_summary():
        from src.agent.memory import ContextMemory, TimestepRecord
        mem = ContextMemory(max_recent=2)
        for i in range(5):
            mem.add_timestep(TimestepRecord(
                timestamp=f"t{i}", sensor_data={}, energy_kwh=10.0, pmv=0.3
            ))
        msgs = mem.get_context_messages()
        has_summary = any("Historical Summary" in m.get("content", "") for m in msgs)
        assert has_summary, "Expected historical summary in context messages"
        return f"Context messages contain historical summary block ({len(msgs)} messages)"

    runner.run("ContextMemory summary generation", "1. Unit Tests — Memory", test_memory_summary)

    # ---- 1.12  LLMClient extract_tool_calls --------------------------------
    def test_extract_tool_calls():
        from src.agent.llm_client import LLMClient
        client = LLMClient(base_url="http://localhost:11434")
        mock_response = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "update_setpoints", "arguments": {"zone": "Core_ZN", "heating_setpoint_c": 21}}}
                ]
            }
        }
        calls = client.extract_tool_calls(mock_response)
        assert len(calls) == 1
        assert calls[0]["name"] == "update_setpoints"
        assert calls[0]["arguments"]["zone"] == "Core_ZN"
        client.close()
        return f"Extracted 1 tool call: {calls[0]['name']}(zone={calls[0]['arguments']['zone']})"

    runner.run("LLMClient extract_tool_calls", "1. Unit Tests — LLM Client", test_extract_tool_calls)

    # ---- 1.13  LLMClient extract_content ----------------------------------
    def test_extract_content():
        from src.agent.llm_client import LLMClient
        client = LLMClient(base_url="http://localhost:11434")
        mock = {"message": {"content": "Zone temperatures are nominal."}}
        content = client.extract_content(mock)
        assert content == "Zone temperatures are nominal."
        client.close()
        return f"Extracted content: '{content[:40]}...'"

    runner.run("LLMClient extract_content", "1. Unit Tests — LLM Client", test_extract_content)

    # ---- 1.14  LLMClient empty response -----------------------------------
    def test_extract_empty():
        from src.agent.llm_client import LLMClient
        client = LLMClient()
        calls = client.extract_tool_calls({})
        content = client.extract_content({})
        assert calls == []
        assert content == ""
        client.close()
        return "Empty response handled gracefully"

    runner.run("LLMClient handles empty response", "1. Unit Tests — LLM Client", test_extract_empty)

    # ---- 1.15  Prompts — SYSTEM_PROMPT is not empty -----------------------
    def test_system_prompt():
        from src.agent.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
        assert len(SYSTEM_PROMPT) > 200, "System prompt too short"
        assert "PMV" in SYSTEM_PROMPT
        assert "EnergyPlus" in SYSTEM_PROMPT
        assert len(TOOL_DEFINITIONS) == 9
        return f"System prompt length: {len(SYSTEM_PROMPT)} chars, {len(TOOL_DEFINITIONS)} tool definitions"

    runner.run("System prompt and tool definitions", "1. Unit Tests — Prompts", test_system_prompt)

    # ---- 1.16  Prompts — format_timestep_message --------------------------
    def test_format_timestep():
        from src.agent.prompts import format_timestep_message
        msg = format_timestep_message(
            timestamp="2024-01-15 12:00",
            outdoor_temp_c=22.5,
            outdoor_humidity_pct=45,
            solar_w_m2=600,
            zone_readings=[{
                "zone_name": "Core_ZN", "temp_c": 23, "humidity_pct": 40,
                "pmv": 0.1, "comfort_category": "neutral", "ppd": 5.2,
                "heating_sp": 21, "cooling_sp": 24,
            }],
            energy_hour_kwh=12.5,
            energy_today_kwh=150.0,
            demand_kw=8.3,
            tariff_per_kwh=0.12,
            tariff_period="off-peak",
            carbon_gco2_kwh=400,
            occupancy_status="occupied",
            occupant_count=25,
        )
        assert "22.5" in msg
        assert "Core_ZN" in msg
        assert "off-peak" in msg
        return f"Formatted message length: {len(msg)} chars"

    runner.run("format_timestep_message", "1. Unit Tests — Prompts", test_format_timestep)

    # ---- 1.17  Config file loading ----------------------------------------
    def test_config_loading():
        import yaml
        with open("config/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        assert "energyplus" in cfg
        assert "llm" in cfg
        assert "constraints" in cfg
        assert "tariff" in cfg
        assert "dashboard" in cfg
        assert cfg["llm"]["model"] == "qwen2.5:7b-instruct"
        return f"Config sections: {list(cfg.keys())}"

    runner.run("settings.yaml loads correctly", "1. Unit Tests — Configuration", test_config_loading)

    # ---- 1.18  Config constraint values -----------------------------------
    def test_config_constraints():
        import yaml
        with open("config/settings.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        c = cfg["constraints"]
        assert c["heating_setpoint_min_c"] < c["heating_setpoint_max_c"]
        assert c["cooling_setpoint_min_c"] < c["cooling_setpoint_max_c"]
        assert c["cooling_setpoint_min_c"] >= c["heating_setpoint_min_c"]
        return (f"Heating: {c['heating_setpoint_min_c']}-{c['heating_setpoint_max_c']}°C, "
                f"Cooling: {c['cooling_setpoint_min_c']}-{c['cooling_setpoint_max_c']}°C")

    runner.run("Constraint bounds are consistent", "1. Unit Tests — Configuration", test_config_constraints)


# ======================================================================
# 2  INTEGRATION TESTS
# ======================================================================

def integration_tests(runner: TestRunner):

    # ---- 2.1  EnergyPlus executable exists ---------------------------------
    def test_energyplus_exists():
        exe = Path("C:/EnergyPlusV26-1-0/energyplus.exe")
        assert exe.exists(), f"EnergyPlus not found at {exe}"
        return f"Found: {exe} ({exe.stat().st_size / 1e6:.1f} MB)"

    runner.run("EnergyPlus executable exists", "2. Integration Tests — EnergyPlus", test_energyplus_exists)

    # ---- 2.2  Baseline IDF is valid ---------------------------------------
    def test_baseline_idf():
        idf = Path("models/baseline.idf")
        assert idf.exists(), "baseline.idf not found"
        text = idf.read_text(encoding="utf-8")
        assert "Building," in text
        assert "Zone," in text
        assert "People," in text
        assert "Fanger" in text
        assert "CLOTHING_SCH" in text
        assert "AIR_VELO_SCH" in text
        zones = text.count("Zone,")
        return f"IDF size: {len(text)} bytes, Zone count: {zones}, Fanger comfort: present"

    runner.run("Baseline IDF structure is valid", "2. Integration Tests — EnergyPlus", test_baseline_idf)

    # ---- 2.3  Weather file exists -----------------------------------------
    def test_weather_file():
        epw = Path("C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
        assert epw.exists(), f"Weather file not found at {epw}"
        return f"Found: {epw.name} ({epw.stat().st_size / 1e3:.0f} KB)"

    runner.run("Weather file exists", "2. Integration Tests — EnergyPlus", test_weather_file)

    # ---- 2.4  Design-day simulation runs end-to-end -----------------------
    def test_design_day_run():
        from src.energyplus.runner import SimulationConfig, EnergyPlusRunner
        out_dir = "data/_test_dd"
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            output_dir=out_dir,
            energyplus_exe="C:/EnergyPlusV26-1-0/energyplus.exe",
            design_day_only=True,
        )
        run = EnergyPlusRunner(cfg)
        result = run.run(run_label="test")
        assert result.success, f"Simulation failed: {result.error_message}"
        assert result.csv_path is not None, "CSV output not generated"
        lines = Path(result.csv_path).read_text().splitlines()
        assert len(lines) > 10, f"CSV too short: {len(lines)} lines"
        shutil.rmtree(out_dir, ignore_errors=True)
        return f"Success in {result.elapsed_seconds:.1f}s, CSV rows: {len(lines)-1}"

    runner.run("Design-day simulation end-to-end", "2. Integration Tests — EnergyPlus", test_design_day_run)

    # ---- 2.5  CSV parser produces correct DataFrame -----------------------
    def test_csv_parser():
        from src.energyplus.runner import SimulationConfig, EnergyPlusRunner
        from src.energyplus.parser import EnergyPlusParser
        out_dir = "data/_test_parser"
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            output_dir=out_dir,
            energyplus_exe="C:/EnergyPlusV26-1-0/energyplus.exe",
            design_day_only=True,
        )
        run = EnergyPlusRunner(cfg)
        result = run.run(run_label="parse_test")
        parser = EnergyPlusParser(result.output_dir)
        df = parser.parse_csv()
        assert len(df) > 0
        assert any("temperature" in c.lower() for c in df.columns)
        shutil.rmtree(out_dir, ignore_errors=True)
        return f"DataFrame: {len(df)} rows x {len(df.columns)} cols"

    runner.run("CSV parser produces valid DataFrame", "2. Integration Tests — Parser", test_csv_parser)

    # ---- 2.6  KPI computation produces expected keys ----------------------
    def test_kpi_computation():
        from src.energyplus.runner import SimulationConfig, EnergyPlusRunner
        from src.energyplus.parser import EnergyPlusParser
        out_dir = "data/_test_kpi"
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            output_dir=out_dir,
            energyplus_exe="C:/EnergyPlusV26-1-0/energyplus.exe",
            design_day_only=True,
        )
        run = EnergyPlusRunner(cfg)
        result = run.run(run_label="kpi_test")
        parser = EnergyPlusParser(result.output_dir)
        kpis = parser.compute_kpis()
        required = ["total_kwh", "peak_kw", "avg_pmv", "comfort_hours_pct", "avg_zone_temp_c"]
        for key in required:
            assert key in kpis, f"Missing KPI: {key}"
            assert kpis[key] is not None, f"KPI {key} is None"
        assert kpis["total_kwh"] > 0
        assert 0 <= kpis["comfort_hours_pct"] <= 100
        shutil.rmtree(out_dir, ignore_errors=True)
        return (f"total_kwh={kpis['total_kwh']:.1f}, peak_kw={kpis['peak_kw']:.1f}, "
                f"avg_pmv={kpis['avg_pmv']:.3f}, comfort={kpis['comfort_hours_pct']:.1f}%")

    runner.run("KPI computation returns all keys", "2. Integration Tests — Parser", test_kpi_computation)

    # ---- 2.7  Error log parser --------------------------------------------
    def test_error_log_parser():
        from src.energyplus.runner import SimulationConfig, EnergyPlusRunner
        from src.energyplus.parser import EnergyPlusParser
        out_dir = "data/_test_err"
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            output_dir=out_dir,
            energyplus_exe="C:/EnergyPlusV26-1-0/energyplus.exe",
            design_day_only=True,
        )
        run = EnergyPlusRunner(cfg)
        result = run.run(run_label="err_test")
        parser = EnergyPlusParser(result.output_dir)
        err = parser.parse_error_log()
        assert err["exists"] is True
        assert err["fatal_errors"] == 0, f"Fatal errors found: {err['fatal_errors']}"
        shutil.rmtree(out_dir, ignore_errors=True)
        return f"Warnings: {err['warnings']}, Severe: {err['severe_errors']}, Fatal: {err['fatal_errors']}"

    runner.run("Error log parser — no fatal errors", "2. Integration Tests — Parser", test_error_log_parser)

    # ---- 2.8  get_latest_timestep -----------------------------------------
    def test_latest_timestep():
        from src.energyplus.runner import SimulationConfig, EnergyPlusRunner
        from src.energyplus.parser import EnergyPlusParser
        out_dir = "data/_test_latest"
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = SimulationConfig(
            idf_path="models/baseline.idf",
            weather_path="C:/EnergyPlusV26-1-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
            output_dir=out_dir,
            energyplus_exe="C:/EnergyPlusV26-1-0/energyplus.exe",
            design_day_only=True,
        )
        run = EnergyPlusRunner(cfg)
        result = run.run(run_label="latest_test")
        parser = EnergyPlusParser(result.output_dir)
        latest = parser.get_latest_timestep()
        assert "data" in latest
        assert len(latest["data"]) > 5
        shutil.rmtree(out_dir, ignore_errors=True)
        return f"Latest timestep has {len(latest['data'])} variables"

    runner.run("get_latest_timestep returns data", "2. Integration Tests — Parser", test_latest_timestep)

    # ---- 2.9  Ollama health check -----------------------------------------
    def test_ollama_health():
        from src.agent.llm_client import LLMClient
        client = LLMClient()
        healthy = client.health_check()
        client.close()
        assert healthy, "Ollama health check failed — is the server running?"
        return "Ollama is reachable and model is loaded"

    runner.run("Ollama health check", "2. Integration Tests — LLM", test_ollama_health)

    # ---- 2.10  LLM responds to a basic prompt ----------------------------
    def test_llm_basic_chat():
        from src.agent.llm_client import LLMClient
        client = LLMClient(timeout_seconds=60)
        response = client.chat(messages=[
            {"role": "user", "content": "Respond with exactly: READY"}
        ])
        content = client.extract_content(response)
        client.close()
        assert len(content) > 0, "LLM returned empty content"
        return f"LLM responded: '{content[:60]}...'"

    runner.run("LLM responds to basic prompt", "2. Integration Tests — LLM", test_llm_basic_chat)


# ======================================================================
# 3  SYSTEM / REGRESSION TESTS
# ======================================================================

def system_tests(runner: TestRunner):

    # ---- 3.1  Full baseline pipeline (run_baseline.py equivalent) ----------
    def test_full_baseline_pipeline():
        from src.energyplus.runner import run_baseline
        result = run_baseline()
        assert result.success, f"Baseline failed: {result.error_message}"
        from src.energyplus.parser import EnergyPlusParser
        parser = EnergyPlusParser(result.output_dir)
        kpis = parser.compute_kpis()
        assert kpis["total_kwh"] > 10000, f"Energy too low: {kpis['total_kwh']}"
        assert kpis["total_kwh"] < 50000, f"Energy unreasonably high: {kpis['total_kwh']}"
        assert kpis["comfort_hours_pct"] > 0, "Comfort metric is zero"
        return (f"Baseline energy: {kpis['total_kwh']:.0f} kWh, "
                f"Comfort: {kpis['comfort_hours_pct']:.1f}%, "
                f"Avg PMV: {kpis['avg_pmv']:.3f}")

    runner.run("Full baseline pipeline", "3. System Tests", test_full_baseline_pipeline)

    # ---- 3.2  Orchestrator initialisation ---------------------------------
    def test_orchestrator_init():
        from src.agent.orchestrator import Orchestrator
        orch = Orchestrator(config_path="config/settings.yaml")
        assert orch is not None
        assert orch.state is not None
        return f"Orchestrator initialised, state={orch.state}"

    runner.run("Orchestrator initialisation", "3. System Tests", test_orchestrator_init)

    # ---- 3.3  Orchestrator setup ------------------------------------------
    def test_orchestrator_setup():
        from src.agent.orchestrator import Orchestrator
        orch = Orchestrator(config_path="config/settings.yaml")
        ok = orch.setup()
        assert ok, "Orchestrator setup failed"
        return "Orchestrator setup completed (E+ validated, LLM connected)"

    runner.run("Orchestrator setup", "3. System Tests", test_orchestrator_setup)

    # ---- 3.4  Orchestrator tool dispatch (valid tool) ---------------------
    def test_tool_dispatch_valid():
        from src.agent.orchestrator import Orchestrator
        orch = Orchestrator(config_path="config/settings.yaml")
        orch.setup()
        result = orch._dispatch_tool("read_sensors", {})
        assert isinstance(result, dict)
        return f"read_sensors returned: {list(result.keys())}"

    runner.run("Tool dispatch — valid tool", "3. System Tests", test_tool_dispatch_valid)

    # ---- 3.5  Orchestrator tool dispatch (invalid tool) -------------------
    def test_tool_dispatch_invalid():
        from src.agent.orchestrator import Orchestrator
        orch = Orchestrator(config_path="config/settings.yaml")
        orch.setup()
        try:
            orch._dispatch_tool("nonexistent_tool", {})
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Unknown tool" in str(e)
        return "ValueError raised for unknown tool as expected"

    runner.run("Tool dispatch — invalid tool raises error", "3. System Tests", test_tool_dispatch_invalid)

    # ---- 3.6  Action log file is written ----------------------------------
    def test_action_log_exists():
        log_path = Path("data/eco_loop.json")
        if log_path.exists():
            data = json.loads(log_path.read_text())
            assert isinstance(data, list)
            return f"Action log exists with {len(data)} entries"
        return "Action log not yet created (loop has not run in this session)"

    runner.run("Action log file format", "3. System Tests", test_action_log_exists)

    # ---- 3.7  Dashboard module imports ------------------------------------
    def test_dashboard_imports():
        from src.dashboard import app as dash_app
        assert hasattr(dash_app, "run_dashboard") or hasattr(dash_app, "app")
        return "Dashboard module imports without errors"

    runner.run("Dashboard module imports", "3. System Tests", test_dashboard_imports)

    # ---- 3.8  Project file structure completeness -------------------------
    def test_project_structure():
        required = [
            "config/settings.yaml",
            "models/baseline.idf",
            "src/__init__.py",
            "src/energyplus/__init__.py",
            "src/energyplus/runner.py",
            "src/energyplus/parser.py",
            "src/energyplus/actuator.py",
            "src/agent/__init__.py",
            "src/agent/orchestrator.py",
            "src/agent/llm_client.py",
            "src/agent/memory.py",
            "src/agent/prompts.py",
            "src/dashboard/__init__.py",
            "scripts/run_baseline.py",
            "scripts/run_loop.py",
            "requirements.txt",
            "README.md",
        ]
        missing = [f for f in required if not Path(f).exists()]
        assert not missing, f"Missing files: {missing}"
        return f"All {len(required)} required files present"

    runner.run("Project file structure complete", "3. System Tests", test_project_structure)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    runner = TestRunner()
    print("\n>>> Starting Eco-Loop automated test suite...\n")

    unit_tests(runner)
    integration_tests(runner)
    system_tests(runner)

    report_path = "result.txt"
    runner.write_report(report_path)
    print(f"\n>>> Report written to: {report_path}")
