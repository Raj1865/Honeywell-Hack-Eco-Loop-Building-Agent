"""
Eco-Loop Quantitative Dashboard
================================
Plotly Dash dashboard comparing baseline vs. AI-optimized building performance.
Proves energy savings while maintaining thermal comfort.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from loguru import logger


# ======================================================================
# Chart Components
# ======================================================================

def create_energy_comparison_chart(baseline_kwh: float, optimized_kwh: float) -> go.Figure:
    """Side-by-side bar chart comparing total energy consumption."""
    savings_pct = ((baseline_kwh - optimized_kwh) / baseline_kwh) * 100

    fig = go.Figure(data=[
        go.Bar(
            name="Baseline",
            x=["Total Energy"],
            y=[baseline_kwh],
            marker_color="#EF4444",
            text=[f"{baseline_kwh:.1f} kWh"],
            textposition="auto",
        ),
        go.Bar(
            name="AI-Optimized",
            x=["Total Energy"],
            y=[optimized_kwh],
            marker_color="#10B981",
            text=[f"{optimized_kwh:.1f} kWh"],
            textposition="auto",
        ),
    ])

    fig.update_layout(
        title=f"Energy Consumption — {savings_pct:.1f}% Reduction",
        yaxis_title="Energy (kWh)",
        barmode="group",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_timeseries_chart(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    variable: str = "energy",
    title: str = "Energy Over Time",
) -> go.Figure:
    """Dual-line chart comparing baseline and optimized timeseries."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=baseline_df.index,
            y=baseline_df[variable] if variable in baseline_df.columns else baseline_df.iloc[:, 0],
            name="Baseline",
            line=dict(color="#EF4444", width=2),
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=optimized_df.index,
            y=optimized_df[variable] if variable in optimized_df.columns else optimized_df.iloc[:, 0],
            name="AI-Optimized",
            line=dict(color="#10B981", width=2),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def create_pmv_heatmap(pmv_data: pd.DataFrame) -> go.Figure:
    """Heatmap of PMV values by zone and hour."""
    fig = go.Figure(data=go.Heatmap(
        z=pmv_data.values,
        x=pmv_data.columns,
        y=pmv_data.index,
        colorscale=[
            [0, "#3B82F6"],    # cold (blue)
            [0.25, "#93C5FD"],  # cool
            [0.5, "#10B981"],   # neutral (green)
            [0.75, "#FBBF24"],  # warm
            [1, "#EF4444"],     # hot (red)
        ],
        zmid=0,
        zmin=-2,
        zmax=2,
        colorbar=dict(title="PMV"),
    ))

    fig.update_layout(
        title="Thermal Comfort Heatmap (PMV)",
        xaxis_title="Hour of Day",
        yaxis_title="Zone",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_comfort_gauge(comfort_pct: float) -> go.Figure:
    """Gauge chart showing comfort compliance percentage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=comfort_pct,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Comfort Compliance<br>(PMV within ±0.5)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#10B981"},
            "steps": [
                {"range": [0, 60], "color": "#EF4444"},
                {"range": [60, 80], "color": "#FBBF24"},
                {"range": [80, 100], "color": "#10B981"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=300,
    )
    return fig


def create_cost_comparison(baseline_cost: float, optimized_cost: float) -> go.Figure:
    """Bar chart comparing energy costs."""
    savings = baseline_cost - optimized_cost

    fig = go.Figure(data=[
        go.Bar(
            x=["Baseline", "AI-Optimized", "Savings"],
            y=[baseline_cost, optimized_cost, savings],
            marker_color=["#EF4444", "#10B981", "#3B82F6"],
            text=[f"${baseline_cost:.2f}", f"${optimized_cost:.2f}", f"${savings:.2f}"],
            textposition="auto",
        ),
    ])

    fig.update_layout(
        title="Energy Cost Comparison",
        yaxis_title="Cost (USD)",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ======================================================================
# Dashboard App
# ======================================================================

def create_dashboard(
    baseline_kpis: dict = None,
    optimized_kpis: dict = None,
    action_log: list = None,
) -> dash.Dash:
    """
    Create the Plotly Dash dashboard application.
    
    Args:
        baseline_kpis: KPI dict from baseline simulation.
        optimized_kpis: KPI dict from optimized simulation.
        action_log: List of action log entries.
    """
    # Default demo data if none provided
    if baseline_kpis is None:
        baseline_kpis = {
            "total_kwh": 1250.0,
            "peak_kw": 85.0,
            "avg_pmv": 0.15,
            "comfort_hours_pct": 78.0,
            "avg_zone_temp_c": 23.5,
        }
    if optimized_kpis is None:
        optimized_kpis = {
            "total_kwh": 950.0,
            "peak_kw": 62.0,
            "avg_pmv": 0.08,
            "comfort_hours_pct": 94.0,
            "avg_zone_temp_c": 22.8,
        }

    baseline_kwh = baseline_kpis.get("total_kwh", 1000)
    optimized_kwh = optimized_kpis.get("total_kwh", 800)
    savings_pct = ((baseline_kwh - optimized_kwh) / baseline_kwh) * 100
    comfort_pct = optimized_kpis.get("comfort_hours_pct", 90)

    # Cost calculation (simplified)
    cost_per_kwh = 0.12
    baseline_cost = baseline_kwh * cost_per_kwh
    optimized_cost = optimized_kwh * cost_per_kwh

    # Carbon calculation
    carbon_per_kwh = 0.4  # kg CO2 / kWh
    carbon_saved = (baseline_kwh - optimized_kwh) * carbon_per_kwh

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Eco-Loop Dashboard",
    )

    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🏢 Eco-Loop Building Agent", className="text-center mb-0",
                         style={"color": "#10B981", "fontWeight": "bold"}),
                html.P("Autonomous AI-Driven Building Energy Optimization",
                       className="text-center text-muted"),
            ], width=12),
        ], className="my-4"),

        # KPI Cards Row
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(f"{savings_pct:.1f}%", style={"color": "#10B981", "fontSize": "2.5rem"}),
                    html.P("Energy Reduction", className="text-muted mb-0"),
                ], className="text-center"),
            ], className="bg-dark border-success"), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(f"{baseline_kwh - optimized_kwh:.0f}", style={"color": "#3B82F6", "fontSize": "2.5rem"}),
                    html.P("kWh Saved", className="text-muted mb-0"),
                ], className="text-center"),
            ], className="bg-dark border-primary"), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(f"{comfort_pct:.0f}%", style={"color": "#FBBF24", "fontSize": "2.5rem"}),
                    html.P("Comfort Compliance", className="text-muted mb-0"),
                ], className="text-center"),
            ], className="bg-dark border-warning"), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(f"{carbon_saved:.0f} kg", style={"color": "#8B5CF6", "fontSize": "2.5rem"}),
                    html.P("CO₂ Avoided", className="text-muted mb-0"),
                ], className="text-center"),
            ], className="bg-dark border-info"), width=3),
        ], className="mb-4"),

        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_energy_comparison_chart(baseline_kwh, optimized_kwh)),
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=create_comfort_gauge(comfort_pct)),
            ], width=6),
        ], className="mb-4"),

        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_cost_comparison(baseline_cost, optimized_cost)),
            ], width=6),
            dbc.Col([
                # Summary table
                dbc.Card([
                    dbc.CardHeader("Performance Summary", className="bg-dark"),
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=[
                                {"Metric": "Total Energy (kWh)", "Baseline": f"{baseline_kwh:.1f}", "Optimized": f"{optimized_kwh:.1f}", "Δ": f"{savings_pct:.1f}%"},
                                {"Metric": "Peak Demand (kW)", "Baseline": f"{baseline_kpis.get('peak_kw', 0):.1f}", "Optimized": f"{optimized_kpis.get('peak_kw', 0):.1f}", "Δ": "—"},
                                {"Metric": "Avg PMV", "Baseline": f"{baseline_kpis.get('avg_pmv', 0):.2f}", "Optimized": f"{optimized_kpis.get('avg_pmv', 0):.2f}", "Δ": "—"},
                                {"Metric": "Comfort Compliance", "Baseline": f"{baseline_kpis.get('comfort_hours_pct', 0):.0f}%", "Optimized": f"{comfort_pct:.0f}%", "Δ": "—"},
                                {"Metric": "Cost (USD)", "Baseline": f"${baseline_cost:.2f}", "Optimized": f"${optimized_cost:.2f}", "Δ": f"${baseline_cost - optimized_cost:.2f}"},
                                {"Metric": "CO₂ (kg)", "Baseline": f"{baseline_kwh * carbon_per_kwh:.0f}", "Optimized": f"{optimized_kwh * carbon_per_kwh:.0f}", "Δ": f"{carbon_saved:.0f}"},
                            ],
                            columns=[
                                {"name": "Metric", "id": "Metric"},
                                {"name": "Baseline", "id": "Baseline"},
                                {"name": "AI-Optimized", "id": "Optimized"},
                                {"name": "Change", "id": "Δ"},
                            ],
                            style_header={
                                "backgroundColor": "#1F2937",
                                "color": "#10B981",
                                "fontWeight": "bold",
                            },
                            style_cell={
                                "backgroundColor": "#111827",
                                "color": "white",
                                "textAlign": "center",
                                "padding": "10px",
                            },
                            style_data_conditional=[
                                {"if": {"column_id": "Δ"}, "color": "#10B981", "fontWeight": "bold"},
                            ],
                        ),
                    ]),
                ], className="bg-dark"),
            ], width=6),
        ], className="mb-4"),

        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(style={"borderColor": "#374151"}),
                html.P(
                    "Eco-Loop Building Agent — Honeywell Hackathon 2026",
                    className="text-center text-muted",
                ),
            ]),
        ]),

        # Auto-refresh interval (5s)
        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),

    ], fluid=True, style={"backgroundColor": "#0F172A", "minHeight": "100vh", "padding": "20px"})

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, **kwargs):
    """Launch the dashboard server."""
    app = create_dashboard(**kwargs)
    logger.info(f"Starting dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    run_dashboard()
