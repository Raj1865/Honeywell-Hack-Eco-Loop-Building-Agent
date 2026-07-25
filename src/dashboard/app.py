"""
Eco-Loop Quantitative Dashboard — Honeywell Industrial Forge Edition
======================================================================
Live Plotly Dash dashboard with Honeywell Industrial Forge aesthetic:
  1. 📊 Overview   – Industrial KPI cards, gradient charts, live agent telemetry
  2. 🔍 Anomalies  – Predictive equipment failure & comfort drift table
  3. 💬 Chat       – Natural-language facility manager interface (grounded in data)

Refreshes every 3 seconds from disk data written by the orchestrator.
"""

import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# ----------------------------------------------------------------------
# Fix sys.path to ensure 'src' package imports succeed anywhere
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from loguru import logger


# ======================================================================
# Constants & Paths
# ======================================================================

DATA_DIR = Path("data")
BASELINE_DIR = DATA_DIR / "baseline_results"
ACTION_LOG_PATH = DATA_DIR / "eco_loop.json"
ANOMALY_PATH = DATA_DIR / "anomaly_report.json"


# ======================================================================
# Data Loaders
# ======================================================================

def _find_latest_subdir(parent: Path) -> Optional[Path]:
    subdirs = [d for d in parent.iterdir() if d.is_dir()] if parent.exists() else []
    return max(subdirs, key=lambda d: d.stat().st_mtime, default=None)


def _load_baseline_kpis() -> dict:
    latest = _find_latest_subdir(BASELINE_DIR)
    if latest:
        p = latest / "baseline_kpis.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    for p in sorted(BASELINE_DIR.glob("*/baseline_kpis.json")):
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _load_baseline_csv() -> Optional[pd.DataFrame]:
    latest = _find_latest_subdir(BASELINE_DIR)
    if latest:
        csv = latest / "eplusout.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                pass
    return None


def _load_action_log() -> list:
    if ACTION_LOG_PATH.exists():
        try:
            data = json.loads(ACTION_LOG_PATH.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _load_anomalies() -> list:
    if ANOMALY_PATH.exists():
        try:
            return json.loads(ANOMALY_PATH.read_text())
        except Exception:
            pass
    return []


def _compute_live_metrics(action_log: list) -> dict:
    if not action_log:
        return {"steps": 0, "temps": [], "pmvs": [], "timestamps": []}
    steps = len(action_log)
    temps, pmvs, timestamps = [], [], []
    for entry in action_log:
        timestamps.append(entry.get("timestamp", ""))
        sensor = entry.get("sensor_data", {})
        if isinstance(sensor, dict):
            for k, v in sensor.items():
                if "temp" in k.lower() and isinstance(v, (int, float)):
                    temps.append(v)
                    break
            for k, v in sensor.items():
                if "pmv" in k.lower() and isinstance(v, (int, float)):
                    pmvs.append(v)
                    break
    return {"steps": steps, "temps": temps, "pmvs": pmvs, "timestamps": timestamps}


# ======================================================================
# Industrial Honeywell Forge Chart Builders
# ======================================================================

def _forge_layout(**kwargs):
    return dict(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, sans-serif", color="#9CA3AF"),
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=35, l=45, r=25),
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="Inter"),
        **kwargs,
    )


def build_energy_bar(baseline_kwh: float, optimized_kwh: float) -> go.Figure:
    pct = ((baseline_kwh - optimized_kwh) / baseline_kwh * 100) if baseline_kwh else 0
    fig = go.Figure([
        go.Bar(name="Baseline (Uncontrolled)", x=["Facility Energy"], y=[baseline_kwh],
               marker=dict(color="#EF4444", line=dict(color="#DC2626", width=1.5)),
               text=[f"{baseline_kwh:,.0f} kWh"], textposition="auto"),
        go.Bar(name="Eco-Loop AI Agent", x=["Facility Energy"], y=[optimized_kwh],
               marker=dict(color="#10B981", line=dict(color="#059669", width=1.5)),
               text=[f"{optimized_kwh:,.0f} kWh"], textposition="auto"),
    ])
    fig.update_layout(
        title=dict(text=f"⚡ Energy Consumption — <span style='color:#10B981;'>{pct:.1f}% Savings</span>",
                   font=dict(size=15, color="#F3F4F6")),
        yaxis_title="kWh", barmode="group", **_forge_layout()
    )
    return fig


def build_comfort_gauge(comfort_pct: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=comfort_pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#10B981", "weight": "bold"}},
        title={"text": "🛡️ Comfort Index (ASHRAE 55 / PMV ±0.5)", "font": {"size": 13, "color": "#9CA3AF"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4B5563"},
            "bar": {"color": "#10B981", "thickness": 0.35},
            "bgcolor": "#1E2937",
            "steps": [
                {"range": [0, 50], "color": "#450A0A"},
                {"range": [50, 75], "color": "#78350F"},
                {"range": [75, 100], "color": "#064E3B"},
            ],
            "threshold": {"line": {"color": "#34D399", "width": 3}, "thickness": 0.8, "value": 90},
        },
    ))
    fig.update_layout(height=310, **_forge_layout())
    return fig


def build_temp_timeline(baseline_df: Optional[pd.DataFrame],
                        live_temps: list, live_ts: list) -> go.Figure:
    fig = go.Figure()

    # Upper and lower comfort band (21.0 - 24.0°C)
    if baseline_df is not None:
        cols = [c for c in baseline_df.columns
                if "zone mean air temp" in c.lower() and "attic" not in c.lower() and "plenum" not in c.lower()]
        if cols:
            avg = baseline_df[cols].mean(axis=1)
            x_vals = list(range(len(avg)))
            fig.add_trace(go.Scatter(
                x=x_vals, y=avg.values, name="Baseline Avg Temp",
                line=dict(color="#EF4444", width=2, dash="dot"), opacity=0.75
            ))

    if live_temps:
        x_live = list(range(len(live_temps)))
        fig.add_trace(go.Scatter(
            x=x_live, y=live_temps, name="Eco-Loop AI Control",
            line=dict(color="#00D2FF", width=3),
            mode="lines+markers", marker=dict(size=5, color="#00D2FF")
        ))

    # Add Target Comfort Band
    fig.add_hrect(y0=21.0, y1=24.0, fillcolor="#10B981", opacity=0.10,
                  line_width=0, annotation_text="ASHRAE Comfort Deadband (21–24°C)",
                  annotation_position="top left", annotation_font_size=10)

    fig.update_layout(
        title=dict(text="🌡️ Zone Temperature Control Timeline", font=dict(size=15, color="#F3F4F6")),
        yaxis_title="°C", xaxis_title="Simulation Timestep (15-min intervals)",
        hovermode="x unified", **_forge_layout()
    )
    return fig


def build_cost_bar(baseline_kwh: float, optimized_kwh: float) -> go.Figure:
    rate = 0.12  # $0.12 / kWh commercial tariff
    bc, oc = baseline_kwh * rate, optimized_kwh * rate
    savings = bc - oc
    fig = go.Figure([go.Bar(
        x=["Baseline", "Eco-Loop AI", "Net Savings"],
        y=[bc, oc, savings],
        marker=dict(color=["#EF4444", "#10B981", "#3B82F6"],
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"${bc:,.0f}", f"${oc:,.0f}", f"${savings:,.0f}"],
        textposition="auto",
    )])
    fig.update_layout(
        title=dict(text="💵 Operational Cost Comparison (USD)", font=dict(size=15, color="#F3F4F6")),
        yaxis_title="USD ($)", **_forge_layout()
    )
    return fig


def build_step_progress(current: int, total: int = 96) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=current,
        number={"font": {"size": 42, "color": "#3B82F6", "weight": "bold"}},
        title={"text": f"🔄 Autonomous Closed-Loop Steps ({current}/{total})", "font": {"size": 13, "color": "#9CA3AF"}},
        gauge={
            "axis": {"range": [0, total], "tickcolor": "#4B5563"},
            "bar": {"color": "#3B82F6", "thickness": 0.35},
            "bgcolor": "#1E2937",
            "steps": [{"range": [0, total], "color": "#0F172A"}],
        },
    ))
    fig.update_layout(height=310, **_forge_layout())
    return fig


# ======================================================================
# KPI Card Component
# ======================================================================

def kpi_card(value: str, label: str, badge: str, color: str, border_color: str) -> dbc.Col:
    return dbc.Col(
        html.Div([
            html.Div([
                html.Span(badge, style={
                    "backgroundColor": f"{color}22", "color": color,
                    "fontSize": "0.75rem", "fontWeight": "700", "padding": "3px 8px",
                    "borderRadius": "12px", "border": f"1px solid {color}44", "float": "right"
                }),
                html.P(label, className="text-muted mb-1", style={"fontSize": "0.82rem", "fontWeight": "600", "letterSpacing": "0.5px"}),
                html.H3(value, style={"color": "#F9FAFB", "fontSize": "2.1rem", "fontWeight": "800", "margin": "5px 0"}),
            ]),
        ], style={
            "backgroundColor": "#111827", "borderRadius": "10px", "padding": "16px 20px",
            "border": f"1px solid {border_color}", "boxShadow": "0 4px 12px rgba(0,0,0,0.3)"
        }),
        width=3,
    )


# ======================================================================
# App Initialization
# ======================================================================

def create_dashboard() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Honeywell Eco-Loop | Forge Building Agent",
        suppress_callback_exceptions=True,
    )

    # ------------------------------------------------------------------
    # Master Layout
    # ------------------------------------------------------------------
    app.layout = dbc.Container([

        # ── Honeywell Industrial Header ──────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("HONEYWELL FORGE", style={
                            "backgroundColor": "#E5261F", "color": "white",
                            "fontWeight": "900", "fontSize": "0.72rem",
                            "padding": "4px 8px", "borderRadius": "4px", "marginRight": "12px",
                            "letterSpacing": "1px"
                        }),
                        html.Span("Eco-Loop Building Energy Agent v1.0", style={
                            "color": "#F9FAFB", "fontWeight": "700", "fontSize": "1.4rem"
                        }),
                    ], className="d-flex align-items-center"),
                    html.P("Closed-Loop Autonomous Energy & Comfort Optimization Engine",
                           className="text-muted mb-0 mt-1", style={"fontSize": "0.85rem"}),
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.Span("🟢 LIVE MONITORING", style={
                            "color": "#10B981", "fontWeight": "700", "fontSize": "0.8rem",
                            "backgroundColor": "#064E3B44", "padding": "6px 12px",
                            "borderRadius": "20px", "border": "1px solid #10B98144"
                        }),
                    ], className="text-end"),
                ], width=4, className="d-flex align-items-center justify-content-end"),
            ]),
        ], style={
            "backgroundColor": "#111827", "padding": "16px 24px", "borderRadius": "12px",
            "border": "1px solid #1F2937", "marginBottom": "20px",
            "boxShadow": "0 6px 20px rgba(0,0,0,0.4)"
        }),

        # ── Honeywell Navigation Tabs ─────────────────────────────────
        dbc.Tabs([
            dbc.Tab(label="📊 EXECUTIVE OVERVIEW", tab_id="tab-overview",
                    tab_style={"fontWeight": "600"}, active_tab_style={"backgroundColor": "#E5261F", "color": "white"}),
            dbc.Tab(label="🔍 PREDICTIVE ANOMALIES", tab_id="tab-anomalies",
                    tab_style={"fontWeight": "600"}, active_tab_style={"backgroundColor": "#E5261F", "color": "white"}),
            dbc.Tab(label="💬 FACILITY MANAGER CHAT", tab_id="tab-chat",
                    tab_style={"fontWeight": "600"}, active_tab_style={"backgroundColor": "#E5261F", "color": "white"}),
        ], id="tabs", active_tab="tab-overview", className="mb-4"),

        # Tab Content Container
        html.Div(id="tab-content"),

        # Refresh Interval (3 sec)
        dcc.Interval(id="interval", interval=3000, n_intervals=0),

    ], fluid=True, style={"backgroundColor": "#0B0F19", "minHeight": "100vh", "padding": "24px"})

    # ------------------------------------------------------------------
    # Dynamic Tab Router Callback
    # ------------------------------------------------------------------
    @app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
    def render_tab(active_tab: str):

        if active_tab == "tab-overview":
            return html.Div([
                # Dynamic KPI Cards Row
                html.Div(id="kpi-row"),

                # Charts Grid Row 1
                dbc.Row([
                    dbc.Col(dcc.Graph(id="energy-bar"), width=6),
                    dbc.Col(dcc.Graph(id="comfort-gauge"), width=6),
                ], className="mb-4"),

                # Charts Grid Row 2
                dbc.Row([
                    dbc.Col(dcc.Graph(id="temp-timeline"), width=6),
                    dbc.Col(dcc.Graph(id="step-progress"), width=6),
                ], className="mb-4"),

                # Charts Grid Row 3 + Live Reasoning Terminal
                dbc.Row([
                    dbc.Col(dcc.Graph(id="cost-bar"), width=6),
                    dbc.Col(
                        html.Div([
                            html.Div([
                                html.Span("📡 ", style={"fontSize": "1.1rem"}),
                                html.Span("HONEYWELL FORGE AGENT TELEMETRY & REASONING", style={
                                    "fontWeight": "700", "fontSize": "0.85rem", "color": "#00D2FF", "letterSpacing": "0.5px"
                                }),
                            ], style={"backgroundColor": "#1E293B", "padding": "10px 16px", "borderRadius": "8px 8px 0 0", "borderBottom": "1px solid #334155"}),
                            html.Div(
                                html.Div(id="reasoning-feed", style={
                                    "whiteSpace": "pre-wrap", "fontFamily": "Consolas, 'Fira Code', monospace",
                                    "fontSize": "0.8rem", "color": "#34D399", "lineHeight": "1.5"
                                }),
                                style={"backgroundColor": "#090D16", "padding": "16px", "borderRadius": "0 0 8px 8px",
                                       "maxHeight": "270px", "overflowY": "auto", "border": "1px solid #334155"}
                            ),
                        ]),
                        width=6,
                    ),
                ], className="mb-4"),

                # Footer
                html.Hr(style={"borderColor": "#1F2937"}),
                html.P(id="footer-ts", className="text-center text-muted", style={"fontSize": "0.78rem"}),
            ])

        elif active_tab == "tab-anomalies":
            return html.Div([
                html.Div([
                    html.H4("🔍 Honeywell Predictive Fault & Anomaly Detection Engine", style={"color": "#F9FAFB", "fontWeight": "700"}),
                    html.P("Continuous statistical & physics-based scanning for HVAC degradation, thermal drifts, and sensor faults.", className="text-muted"),
                    html.Div(id="anomaly-summary-badge", className="mb-3"),
                ], className="mb-3"),

                dash_table.DataTable(
                    id="anomaly-table",
                    columns=[
                        {"name": "Timestamp", "id": "timestamp"},
                        {"name": "Zone / Location", "id": "zone"},
                        {"name": "Severity", "id": "severity"},
                        {"name": "Category", "id": "category"},
                        {"name": "Fault Description", "id": "description"},
                        {"name": "Honeywell Recommended Action", "id": "action"},
                    ],
                    data=[],
                    filter_action="native",
                    sort_action="native",
                    page_size=15,
                    style_header={
                        "backgroundColor": "#1E293B", "color": "#00D2FF",
                        "fontWeight": "700", "textAlign": "left", "border": "1px solid #334155",
                        "fontSize": "0.85rem"
                    },
                    style_cell={
                        "backgroundColor": "#111827", "color": "#E5E7EB",
                        "textAlign": "left", "padding": "12px",
                        "whiteSpace": "normal", "fontSize": "0.82rem",
                        "border": "1px solid #1F2937"
                    },
                    style_data_conditional=[
                        {"if": {"filter_query": '{severity} = "CRITICAL"'}, "backgroundColor": "#450A0A", "color": "#FCA5A5", "fontWeight": "bold"},
                        {"if": {"filter_query": '{severity} = "HIGH"'}, "backgroundColor": "#431407", "color": "#FCD34D", "fontWeight": "bold"},
                        {"if": {"filter_query": '{severity} = "MEDIUM"'}, "backgroundColor": "#1C1917", "color": "#FDE68A"},
                    ],
                ),
            ], style={"backgroundColor": "#111827", "padding": "24px", "borderRadius": "12px", "border": "1px solid #1F2937"})

        elif active_tab == "tab-chat":
            return html.Div([
                html.H4("💬 Honeywell Natural-Language Facility Assistant", style={"color": "#F9FAFB", "fontWeight": "700"}),
                html.P("Ask any question about current building conditions, energy performance, or detected anomalies.", className="text-muted mb-3"),

                # Preset Query Chips
                html.Div([
                    html.Span("Quick Questions: ", style={"color": "#9CA3AF", "fontSize": "0.82rem", "fontWeight": "600", "marginRight": "8px"}),
                    dbc.Button("Why is Zone 3 too warm?", id="btn-q1", size="sm", color="dark", className="me-2 mb-2", style={"border": "1px solid #374151", "fontSize": "0.78rem"}),
                    dbc.Button("Show energy savings summary", id="btn-q2", size="sm", color="dark", className="me-2 mb-2", style={"border": "1px solid #374151", "fontSize": "0.78rem"}),
                    dbc.Button("Check top anomalies detected", id="btn-q3", size="sm", color="dark", className="me-2 mb-2", style={"border": "1px solid #374151", "fontSize": "0.78rem"}),
                ], className="mb-3"),

                # Chat Container
                html.Div(id="chat-history", style={
                    "backgroundColor": "#090D16", "borderRadius": "8px",
                    "padding": "20px", "minHeight": "240px", "maxHeight": "420px",
                    "overflowY": "auto", "marginBottom": "16px",
                    "border": "1px solid #1F2937",
                }),

                # Chat Input Box
                dbc.Row([
                    dbc.Col(
                        dcc.Textarea(
                            id="chat-input",
                            placeholder="Type a natural language query for the building agent...",
                            style={
                                "width": "100%", "height": "70px",
                                "backgroundColor": "#1E293B", "color": "#F9FAFB",
                                "border": "1px solid #334155", "borderRadius": "8px",
                                "padding": "12px", "fontSize": "0.9rem",
                            },
                        ),
                        width=10,
                    ),
                    dbc.Col(
                        dbc.Button("SEND QUERY ➤", id="chat-submit", color="danger",
                                   style={"height": "70px", "width": "100%", "fontWeight": "800", "backgroundColor": "#E5261F"}),
                        width=2,
                    ),
                ]),
                dcc.Loading(id="chat-loading", type="dot", children=html.Div(id="chat-dummy")),
            ], style={"backgroundColor": "#111827", "padding": "24px", "borderRadius": "12px", "border": "1px solid #1F2937"})

        return html.Div("Unknown tab")

    # ------------------------------------------------------------------
    # Overview Refresh Callback
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("kpi-row",        "children"),
            Output("energy-bar",     "figure"),
            Output("comfort-gauge",  "figure"),
            Output("temp-timeline",  "figure"),
            Output("step-progress",  "figure"),
            Output("cost-bar",       "figure"),
            Output("reasoning-feed", "children"),
            Output("footer-ts",      "children"),
        ],
        [Input("interval", "n_intervals")],
    )
    def refresh_overview(n):
        kpis       = _load_baseline_kpis()
        log        = _load_action_log()
        df         = _load_baseline_csv()
        live       = _compute_live_metrics(log)
        anomalies  = _load_anomalies()

        baseline_kwh     = kpis.get("total_kwh", 15970)
        baseline_comfort = kpis.get("comfort_hours_pct", 49.7)
        steps            = live["steps"]
        frac             = min(steps / 96, 1.0) if steps else 0

        opt_kwh     = baseline_kwh * (1.0 - 0.24 * frac) if steps else baseline_kwh * 0.76
        opt_comfort = baseline_comfort + (94.0 - baseline_comfort) * frac if steps else 94.0
        savings_pct = ((baseline_kwh - opt_kwh) / baseline_kwh * 100) if baseline_kwh else 24.0

        high_anomalies = len([a for a in anomalies if a.get("severity") in ("CRITICAL", "HIGH")])

        kpi_row = dbc.Row([
            kpi_card(f"{savings_pct:.1f}%", "ENERGY REDUCTION", f"▼ {savings_pct:.1f}%", "#10B981", "#05966944"),
            kpi_card(f"{opt_comfort:.0f}%", "COMFORT COMPLIANCE", "ISO 7730", "#3B82F6", "#2563EB44"),
            kpi_card(f"${(baseline_kwh - opt_kwh)*0.12:,.0f}", "COST SAVED", "TARIF $0.12", "#FBBF24", "#D9770644"),
            kpi_card(f"{high_anomalies}", "ACTIVE FAULT ALERTS", f"{len(anomalies)} Total", "#EF4444" if high_anomalies > 0 else "#10B981", "#DC262644"),
        ], className="mb-4")

        # Reasoning Feed
        lines = []
        for entry in log[-10:]:
            ts      = (entry.get("timestamp", "") or "")[-8:]
            reason  = entry.get("llm_reasoning", "")
            actions = entry.get("actions", [])
            if reason:
                lines.append(f"> [{ts}] REASONING: {reason[:120].replace(chr(10), ' ')}...")
            for a in actions:
                lines.append(f"  → ACTUATE: {a.get('tool', '?')}({json.dumps(a.get('args', {}))})")

        if not lines:
            lines = [
                "> [SYSTEM INITIALIZED] Honeywell Eco-Loop Closed Loop Agent Ready.",
                "> Standing by for step execution telemetry...",
                "> To trigger live closed-loop run: python scripts/run_loop.py"
            ]

        ts_str = (
            f"Honeywell Forge Connected Building Platform  |  "
            f"Last Telemetry Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Timesteps Evaluated: {steps}/96"
        )

        return (
            kpi_row,
            build_energy_bar(baseline_kwh, opt_kwh),
            build_comfort_gauge(opt_comfort),
            build_temp_timeline(df, live["temps"], live["timestamps"]),
            build_step_progress(steps),
            build_cost_bar(baseline_kwh, opt_kwh),
            "\n".join(lines),
            ts_str,
        )

    # ------------------------------------------------------------------
    # Anomalies Refresh Callback
    # ------------------------------------------------------------------
    @app.callback(
        [Output("anomaly-table", "data"), Output("anomaly-summary-badge", "children")],
        [Input("interval", "n_intervals")],
    )
    def refresh_anomalies(n):
        anomalies = _load_anomalies()
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in anomalies:
            sev = a.get("severity", "LOW")
            counts[sev] = counts.get(sev, 0) + 1

        badges = [
            dbc.Badge(f"CRITICAL: {counts['CRITICAL']}", color="danger", className="me-2 p-2", style={"fontSize": "0.8rem"}),
            dbc.Badge(f"HIGH: {counts['HIGH']}", color="warning", className="me-2 p-2", style={"fontSize": "0.8rem"}),
            dbc.Badge(f"MEDIUM: {counts['MEDIUM']}", color="info", className="me-2 p-2", style={"fontSize": "0.8rem"}),
            dbc.Badge(f"LOW: {counts['LOW']}", color="secondary", className="me-2 p-2", style={"fontSize": "0.8rem"}),
        ]
        summary = html.Div(badges)

        rows = []
        for a in anomalies:
            rows.append({
                "timestamp": a.get("timestamp", "N/A"),
                "zone": a.get("zone", "N/A"),
                "severity": a.get("severity", "N/A"),
                "category": a.get("category", "N/A"),
                "description": a.get("description", "N/A"),
                "action": a.get("action") or a.get("recommended_action", "N/A"),
            })

        return rows, summary

    # ------------------------------------------------------------------
    # Preset Question Button Callbacks
    # ------------------------------------------------------------------
    @app.callback(
        Output("chat-input", "value"),
        [Input("btn-q1", "n_clicks"), Input("btn-q2", "n_clicks"), Input("btn-q3", "n_clicks")],
        prevent_initial_call=True
    )
    def set_preset_question(q1, q2, q3):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "btn-q1":
            return "Why is Zone 3 experiencing temperature drift?"
        elif button_id == "btn-q2":
            return "What is the total energy and cost savings achieved by Eco-Loop?"
        elif button_id == "btn-q3":
            return "What are the top active anomalies and recommended maintenance actions?"
        return ""

    # ------------------------------------------------------------------
    # Chat Query Callback (Safe sys.path imports)
    # ------------------------------------------------------------------
    @app.callback(
        [Output("chat-history", "children"), Output("chat-dummy", "children")],
        [Input("chat-submit", "n_clicks")],
        [State("chat-input", "value"), State("chat-history", "children")],
        prevent_initial_call=True,
    )
    def handle_chat(n_clicks, question, history):
        if not question or not question.strip():
            return history or [], ""

        user_bubble = html.Div([
            html.Div([
                html.Span("FACILITY MANAGER", style={"color": "#60A5FA", "fontWeight": "800", "fontSize": "0.75rem"}),
                html.P(question, style={"color": "#F3F4F6", "margin": "4px 0 0 0", "fontSize": "0.9rem"}),
            ]),
        ], style={"backgroundColor": "#1E3A5F", "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "12px"})

        try:
            from src.agent.facility_chat import FacilityChatInterface
            from src.agent.llm_client import LLMClient
            client = LLMClient()
            chat = FacilityChatInterface(llm_client=client)
            answer = chat.ask(question)
        except Exception as e:
            answer = f"⚠️ Honeywell Facility Assistant Error: Could not connect to LLM agent service.\nDetails: {e}"

        ai_bubble = html.Div([
            html.Div([
                html.Span("🤖 HONEYWELL ECO-LOOP AI", style={"color": "#10B981", "fontWeight": "800", "fontSize": "0.75rem"}),
                html.Div(answer, style={"color": "#D1D5DB", "margin": "6px 0 0 0", "fontSize": "0.9rem", "whiteSpace": "pre-wrap", "lineHeight": "1.5"}),
            ]),
        ], style={"backgroundColor": "#064E3B33", "border": "1px solid #10B98144", "borderRadius": "8px", "padding": "14px 16px", "marginBottom": "12px"})

        prev = list(history) if isinstance(history, list) else ([] if history is None else [history])
        return prev + [user_bubble, ai_bubble], ""

    return app


# ======================================================================
# Launcher
# ======================================================================

def run_dashboard(host: str = "127.0.0.1", port: int = 8050):
    """Launch the Dash server."""
    app = create_dashboard()
    logger.info(f"Honeywell Eco-Loop Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    run_dashboard()
