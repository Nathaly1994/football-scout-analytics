"""
Aplicación principal Dash — Edición Premium Soccer.
Layout de alto impacto visual con carga lazy, gráficos narrativos
y paleta exclusiva verde cancha + dorado elite.
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from utils.data_loader import DataLoader, SKILL_COLS
from models.valuation_model import PlayerValuationModel
from utils.charts import (
    player_projection_chart,
    player_overall_history,
    player_skills_evolution,
    radar_comparison,
    salary_vs_performance,
    bar_skill_comparison,
    undervalued_scatter,
    undervalued_bar_top20,
    feature_importance_chart,
    gems_by_position,
    COLORS,
    FONT_FAMILY,
    FONT_BODY,
)


# ─────────────────────────────────────────────────────────────────
# Estilos globales inyectados via assets o inline
# ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Reset y base ─────────────────────────────────────────── */
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body, html {{
    background: #050510 !important;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(0,229,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 0%, rgba(201,168,76,0.08) 0%, transparent 50%),
        radial-gradient(circle, rgba(60,60,120,0.15) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 30px 30px !important;
    font-family: 'Inter', Arial, sans-serif;
    color: #e8eaf6;
    min-height: 100vh;
}}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: #0d0d1a; }}
::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, {COLORS['gold']}, {COLORS['electric']});
    border-radius: 4px;
}}

/* ── Header ───────────────────────────────────────────────── */
.app-header {{
    background: linear-gradient(180deg, #070718 0%, #0d0d28 100%) !important;
    border-bottom: 1px solid rgba(201,168,76,0.3) !important;
    box-shadow: 0 4px 40px rgba(0,0,0,0.8), 0 1px 0 rgba(201,168,76,0.15) !important;
    position: relative;
    z-index: 100;
}}
.app-header::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, {COLORS['gold']}, {COLORS['electric']}, {COLORS['gold']}, transparent);
    opacity: 0.6;
}}

/* ── Nav tabs ─────────────────────────────────────────────── */
.nav-tab {{
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #6e6e9e !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    border-radius: 6px 6px 0 0 !important;
    margin-right: 3px !important;
}}
.nav-tab:hover {{
    color: {COLORS['gold']} !important;
    background: rgba(201,168,76,0.08) !important;
    border-color: rgba(201,168,76,0.3) !important;
}}
.nav-tab.active {{
    background: linear-gradient(180deg, rgba(201,168,76,0.18) 0%, rgba(13,13,26,0.9) 100%) !important;
    border-top: 2px solid {COLORS['gold']} !important;
    border-left: 1px solid rgba(201,168,76,0.4) !important;
    border-right: 1px solid rgba(201,168,76,0.4) !important;
    border-bottom: 2px solid #050510 !important;
    color: {COLORS['gold']} !important;
    font-size: 12px !important;
    text-shadow: 0 0 12px rgba(201,168,76,0.5) !important;
}}

/* ── Filter bar ───────────────────────────────────────────── */
.filter-bar {{
    background: linear-gradient(180deg, #0a0a20 0%, #0d0d1a 100%) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    padding: 14px 30px !important;
}}
.filter-label {{
    color: #5a5a8a !important;
    font-size: 9px !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 5px !important;
}}

/* ── Panel card ───────────────────────────────────────────── */
.panel-card {{
    background: linear-gradient(145deg, #12122a 0%, #0e0e22 100%) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    overflow: hidden !important;
    transition: box-shadow 0.3s ease, border-color 0.3s ease !important;
}}
.panel-card:hover {{
    border-color: rgba(201,168,76,0.2) !important;
    box-shadow: 0 12px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(201,168,76,0.08), inset 0 1px 0 rgba(255,255,255,0.07) !important;
}}

/* ── Panel header ─────────────────────────────────────────── */
.panel-header {{
    background: linear-gradient(90deg, rgba(201,168,76,0.1) 0%, rgba(0,229,255,0.03) 100%) !important;
    border-bottom: 1px solid rgba(201,168,76,0.18) !important;
    padding: 12px 18px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: {COLORS['gold']} !important;
    text-transform: uppercase !important;
    display: flex !important;
    align-items: center !important;
}}

/* ── KPI card ─────────────────────────────────────────────── */
.kpi-card {{
    background: linear-gradient(145deg, #14142e 0%, #0e0e24 100%) !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    border-left: 3px solid {COLORS['gold']} !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    border-right: 1px solid rgba(255,255,255,0.03) !important;
    border-bottom: 1px solid rgba(0,0,0,0.5) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}
.kpi-card:hover {{
    transform: translateY(-4px) !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.7) !important;
}}

/* ── Botón ejecutar ───────────────────────────────────────── */
@keyframes shimmer {{
    0%   {{ background-position: -300% center; }}
    100% {{ background-position: 300% center; }}
}}
@keyframes pulse-border {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(201,168,76,0); }}
    50%       {{ box-shadow: 0 0 0 4px rgba(201,168,76,0.15); }}
}}
.btn-run-ai {{
    background: linear-gradient(135deg, #1c3d1c 0%, #0f2610 50%, #1c3d1c 100%) !important;
    color: {COLORS['gold']} !important;
    border: 1px solid rgba(201,168,76,0.5) !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 3px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
    animation: pulse-border 3s infinite !important;
    text-shadow: 0 0 10px rgba(201,168,76,0.4) !important;
}}
.btn-run-ai::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(201,168,76,0.25) 45%,
        rgba(255,255,255,0.15) 50%,
        rgba(201,168,76,0.25) 55%,
        transparent 100%);
    background-size: 300% 100%;
    animation: shimmer 3s infinite linear;
}}
.btn-run-ai:hover {{
    background: linear-gradient(135deg, #224a22 0%, #153215 50%, #224a22 100%) !important;
    border-color: {COLORS['gold']} !important;
    box-shadow: 0 0 30px rgba(201,168,76,0.3), 0 6px 20px rgba(0,0,0,0.5) !important;
    transform: translateY(-2px) !important;
    letter-spacing: 4px !important;
}}

/* ── Player button ────────────────────────────────────────── */
.player-btn {{
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #9e9ec8 !important;
    font-size: 11px !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    border-radius: 6px !important;
}}
.player-btn:hover {{
    background: rgba(201,168,76,0.1) !important;
    border-color: rgba(201,168,76,0.4) !important;
    color: {COLORS['gold']} !important;
    transform: translateX(3px) !important;
}}

/* ── Dropdowns blancos ────────────────────────────────────── */
.Select, .Select-control, .Select--single {{
    background: white !important;
    border-color: {COLORS['border2']} !important;
}}
.Select-control {{ background: white !important; border-color: {COLORS['border2']} !important; }}
.Select-value {{ background: white !important; }}
.Select-value-label {{ color: #111111 !important; }}
.Select-single-value {{ color: #111111 !important; }}
.Select-placeholder {{ color: #555555 !important; }}
.Select-input {{ background: white !important; }}
.Select-input input {{ color: #111111 !important; background: white !important; }}
.Select-menu-outer {{ background: white !important; border-color: {COLORS['border2']} !important; z-index: 9999 !important; }}
.Select-menu {{ background: white !important; }}
.Select-option {{ color: #111111 !important; background: white !important; }}
.Select-option.is-focused {{ background: #e8f0fe !important; color: #111111 !important; }}
.Select-option.is-selected {{ background: #d0e4ff !important; color: #111111 !important; }}
.Select-arrow {{ border-top-color: #444444 !important; }}
.VirtualizedSelectOption {{ color: #111111 !important; background: white !important; }}
.VirtualizedSelectFocusedOption {{ background: #e8f0fe !important; color: #111111 !important; }}

/* ── Inputs ───────────────────────────────────────────────── */
input.form-control {{ background: white !important; border-color: {COLORS['border2']} !important; color: #111111 !important; }}
input.form-control:focus {{ border-color: {COLORS['gold']} !important; box-shadow: 0 0 0 3px rgba(201,168,76,0.25) !important; }}
input[type=number], input[type=text] {{ color: #111111 !important; background: white !important; }}

/* ── Sliders ──────────────────────────────────────────────── */
.rc-slider-tooltip-inner {{ background: #1a1a3e !important; color: {COLORS['gold']} !important;
    border: 1px solid rgba(201,168,76,0.4) !important; font-weight: 700 !important; font-size: 11px !important; border-radius: 6px !important; }}
.rc-slider-mark-text {{ color: #5a5a8a !important; }}

/* ── Gráficas ─────────────────────────────────────────────── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}
.dash-graph {{ background: #050510 !important; border-radius: 8px; }}

/* ── Tablas ───────────────────────────────────────────────── */
table {{ color: #e8eaf6 !important; }}
.table-striped tbody tr:nth-of-type(odd) {{ background: rgba(255,255,255,0.03) !important; }}
.table-hover tbody tr:hover {{ background: rgba(201,168,76,0.1) !important; cursor: pointer; }}
table th {{ color: {COLORS['gold']} !important; font-family: 'Rajdhani', sans-serif;
    font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase;
    border-color: rgba(255,255,255,0.08) !important; background: rgba(0,0,0,0.3) !important; }}
table td {{ border-color: rgba(255,255,255,0.04) !important; font-size: 12px; }}

/* ── Fade-in pestaña ──────────────────────────────────────── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.tab-content > .active {{ animation: fadeInUp 0.35s ease forwards !important; }}
"""


class FootballScoutApp:
    """Plataforma analítica de scouting - edicion premium Soccer."""

    def __init__(self):
        self._loader = DataLoader()
        self._model = PlayerValuationModel()
        self._df: pd.DataFrame | None = None
        self._df_all: pd.DataFrame | None = None

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.CYBORG,
                dbc.icons.BOOTSTRAP,
            ],
            suppress_callback_exceptions=True,
            title="Football Scout Analytics",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        )
        # Inyectar CSS global
        self.app.index_string = self.app.index_string.replace(
            "</head>",
            f"<style>{GLOBAL_CSS}</style></head>",
        )
        self._build_layout()
        self._register_callbacks()

    # ─── Datos lazy-load ───────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._loader.load_latest()
        return self._df

    @property
    def df_all(self) -> pd.DataFrame:
        if self._df_all is None:
            self._df_all = self._loader.load_all()
        return self._df_all

    # ─── Layout principal ──────────────────────────────────────────

    def _build_layout(self):
        seasons = self._loader.available_seasons()
        positions = ["Todas", "GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST", "CF"]
        POS_LABELS = {
            "Todas": "Todas las Posiciones",
            "GK":  "GK — Portero",
            "CB":  "CB — Defensa Central",
            "LB":  "LB — Lateral Izquierdo",
            "RB":  "RB — Lateral Derecho",
            "CDM": "CDM — Mediocentro Defensivo",
            "CM":  "CM — Centrocampista Central",
            "CAM": "CAM — Mediapunta Central",
            "LM":  "LM — Mediocampista Izquierdo",
            "RM":  "RM — Mediocampista Derecho",
            "LW":  "LW — Extremo Izquierdo",
            "RW":  "RW — Extremo Derecho",
            "ST":  "ST — Delantero Centro",
            "CF":  "CF — Media Punta / Falso 9",
        }

        self.app.layout = html.Div([

            # ══ HEADER ══════════════════════════════════════════════
            html.Div([
                html.Div([
                    # Logo / Marca
                    html.Div([
                        html.Span( style={"fontSize": "36px", "marginRight": "12px"}),
                        html.Div([
                            html.H1("FOOTBALL SCOUT ANALYTICS",
                                    style={
                                        "color": COLORS["gold"],
                                        "margin": "0",
                                        "fontWeight": "700",
                                        "fontFamily": FONT_FAMILY,
                                        "fontSize": "24px",
                                        "letterSpacing": "2px",
                                    }),
                            html.P("Plataforma de Inteligencia de Mercado · Motor de Valoración XGBoost · FIFA 15–21",
                                   style={
                                       "color": COLORS["muted"],
                                       "margin": "0",
                                       "fontSize": "11px",
                                       "letterSpacing": "0.5px",
                                   }),
                        ]),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Indicador live
                    html.Div([
                        html.Span("●", style={"color": COLORS["electric"], "marginRight": "6px", "fontSize": "10px"}),
                        html.Span("IA ACTIVA", style={
                            "color": COLORS["electric"], "fontSize": "10px",
                            "fontFamily": FONT_FAMILY, "letterSpacing": "1px",
                        }),
                    ], style={
                        "border": f"1px solid {COLORS['electric']}33",
                        "borderRadius": "20px", "padding": "4px 12px",
                        "background": f"{COLORS['electric']}0A",
                    }),
                ], style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center", "padding": "16px 30px",
                }),
            ], className="app-header", style={"position": "relative"}),

            # ══ FILTROS GLOBALES ═════════════════════════════════════
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(" TEMPORADA", className="filter-label"),
                        dcc.Dropdown(
                            id="global-season",
                            options=[{"label": s, "value": s} for s in seasons],
                            value=seasons[-1], clearable=False,
                            style={"backgroundColor": "white", "color": "#111111"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Div(" POSICIÓN", className="filter-label"),
                        dcc.Dropdown(
                            id="global-position",
                            options=[{"label": POS_LABELS.get(p, p), "value": p} for p in positions],
                            value="Todas", clearable=False,
                            style={"backgroundColor": "white", "color": "#111111"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Div(" LIGA", className="filter-label"),
                        dcc.Dropdown(
                            id="global-league",
                            options=[{"label": "Todas las Ligas", "value": "Todas las Ligas"}],
                            value="Todas las Ligas", clearable=False,
                            style={"backgroundColor": "white", "color": "#111111"},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Div(" RANGO DE EDAD", className="filter-label"),
                        dcc.RangeSlider(
                            id="global-age", min=15, max=45, step=1, value=[17, 35],
                            marks={i: {"label": str(i), "style": {"color": COLORS["muted"], "fontSize": "10px"}}
                                   for i in range(15, 46, 5)},
                            tooltip={"placement": "bottom"},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Div(" VALOR MÁX (€M)", className="filter-label"),
                        dcc.Slider(
                            id="global-max-value", min=0, max=200, step=5, value=200,
                            marks={0: {"label": "0", "style": {"color": COLORS["muted"]}},
                                   100: {"label": "100M", "style": {"color": COLORS["muted"]}},
                                   200: {"label": "200M+", "style": {"color": COLORS["muted"]}}},
                            tooltip={"placement": "bottom"},
                        ),
                    ], width=2),
                ], align="center"),
            ], className="filter-bar"),

            # ══ TABS ═════════════════════════════════════════════════
            html.Div([
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="tab-valuation",
                    children=[
                        dbc.Tab(label=" Valoración & Proyección", tab_id="tab-valuation",
                                tab_class_name="nav-tab"),
                        dbc.Tab(label=" Comparación de Jugadores", tab_id="tab-comparison",
                                tab_class_name="nav-tab"),
                        dbc.Tab(label=" Joyas Ocultas del Mercado", tab_id="tab-undervalued",
                                tab_class_name="nav-tab"),
                        dbc.Tab(label=" Diccionario FIFA", tab_id="tab-dictionary",
                                tab_class_name="nav-tab"),
                    ],
                    style={"borderBottom": f"1px solid {COLORS['border2']}"},
                ),
            ], style={"padding": "0 20px", "background": COLORS["surface"]}),

            # ══ CONTENIDO ════════════════════════════════════════════
            dcc.Loading(
                id="tab-loading",
                type="circle",
                color=COLORS["gold"],
                children=html.Div(id="tab-content", style={"padding": "20px 24px"}),
            ),

            # Stores
            dcc.Store(id="store-undervalued"),
            dcc.Store(id="store-model-trained", data=False),
            dcc.Store(id="store-cmp-id-1", data=None),
            dcc.Store(id="store-cmp-id-2", data=None),
            dcc.Store(id="store-cmp-id-3", data=None),
            dcc.Store(id="store-selected-player-id", data=None),

        ], style={"minHeight": "100vh", "background": COLORS["dark"]})

    # ─── Sub-layouts de pestañas ───────────────────────────────────

    def _tab_valuation(self) -> html.Div:
        return html.Div([
            dbc.Row([
                # Panel lateral de búsqueda
                dbc.Col([
                    html.Div([
                        html.Div(" BUSCAR JUGADOR", className="panel-header"),
                        html.Div([
                            dbc.Input(
                                id="val-search",
                                placeholder="Nombre del jugador...",
                                type="text", debounce=True,
                                style={"marginBottom": "10px"},
                            ),
                            dcc.Loading(
                                type="dot", color=COLORS["gold"],
                                children=html.Div(id="val-search-results"),
                            ),
                            html.Hr(style={"borderColor": COLORS["border2"], "margin": "12px 0"}),
                            html.Div(id="val-player-card"),
                            html.Hr(style={"borderColor": COLORS["border2"], "margin": "12px 0"}),
                            html.Div(" HORIZONTE DE PROYECCIÓN", className="filter-label"),
                            dcc.Slider(
                                id="val-years", min=1, max=5, step=1, value=3,
                                marks={i: {"label": f"{i}a", "style": {"color": COLORS["muted"], "fontSize": "10px"}}
                                       for i in range(1, 6)},
                                tooltip={"placement": "bottom"},
                            ),
                        ], style={"padding": "14px"}),
                    ], className="panel-card"),
                ], width=3),

                # Área de gráficos
                dbc.Col([
                    dcc.Loading(
                        type="circle", color=COLORS["gold"],
                        children=[
                            dbc.Row([
                                dbc.Col(html.Div(dcc.Graph(id="val-projection-chart", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Busca un jugador<br>para ver su proyección de valor","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=12),
                            ]),
                            dbc.Row([
                                dbc.Col(html.Div(dcc.Graph(id="val-overall-chart", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Evolución del Overall<br>y Potencial por temporada","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#4fc3f7"},"bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#4fc3f744", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=6),
                                dbc.Col(html.Div(dcc.Graph(id="val-skills-evolution-chart", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Evolución de habilidades<br>Velocidad · Disparo · Pase","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#00e676"},"bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#00e67644", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=6),
                            ], style={"marginTop": "12px"}),
                            dbc.Row([
                                dbc.Col(html.Div(dcc.Graph(id="val-feature-imp-chart", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Importancia de variables<br>en el modelo XGBoost","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#9e9ec8"},"bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#2a2a4a", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=12),
                            ], style={"marginTop": "12px"}),
                        ],
                    ),
                ], width=9),
            ]),

            # KPI cards
            dbc.Row(id="val-kpi-row", style={"marginTop": "16px"}),
        ])

    def _tab_comparison(self) -> html.Div:
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(" SELECCIONAR JUGADORES", className="panel-header"),
                        html.Div([
                            self._search_slot(1, COLORS["gold"], "Jugador 1"),
                            html.Hr(style={"borderColor": COLORS["border"], "margin": "10px 0"}),
                            self._search_slot(2, COLORS["electric"], "Jugador 2"),
                            html.Hr(style={"borderColor": COLORS["border"], "margin": "10px 0"}),
                            self._search_slot(3, COLORS["sky"], "Jugador 3 (opcional)"),
                            html.Hr(style={"borderColor": COLORS["border"], "margin": "10px 0"}),
                            html.Div(" DIMENSIONES RADAR", className="filter-label"),
dbc.ButtonGroup([
    dbc.Button("Jugador de Campo", id="btn-outfield", n_clicks=0, size="sm",
               color="warning", outline=True,
               style={"fontSize": "11px", "fontFamily": FONT_FAMILY}),
    dbc.Button("Portero", id="btn-gk", n_clicks=0, size="sm",
               color="warning", outline=True,
               style={"fontSize": "11px", "fontFamily": FONT_FAMILY}),
], style={"marginBottom": "8px", "width": "100%"}),
dcc.Dropdown(
    id="cmp-radar-dims",
    options=[{"label": {
                 "pace": "Velocidad", "shooting": "Disparo", "passing": "Pase",
                 "dribbling": "Regate", "defending": "Defensa", "physic": "Físico",
                 "gk_diving": "Portero - Estirada", "gk_handling": "Portero - Control",
                 "gk_kicking": "Portero - Saque", "gk_reflexes": "Portero - Reflejos",
                 "gk_speed": "Portero - Velocidad", "gk_positioning": "Portero - Posicionamiento",
             }.get(c, c.replace("_", " ").title()), "value": c}
             for c in [
                 "pace", "shooting", "passing", "dribbling", "defending", "physic",
                 "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"
             ]],
    value=["pace", "shooting", "passing", "dribbling", "defending", "physic"],
    multi=True, style={"fontSize": "12px", "backgroundColor": "white", "color": "#111111"},

                            ),
                        ], style={"padding": "12px"}),
                    ], className="panel-card"),
                ], width=3),

                dbc.Col([
                    dcc.Loading(type="circle", color=COLORS["gold"], children=[
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="cmp-radar", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Selecciona jugadores<br>para comparar su radar","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#4fc3f7"},"bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#4fc3f744", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=5),
                            dbc.Col(html.Div(dcc.Graph(id="cmp-salary-scatter", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Comparación de<br>Valor de Mercado","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=7),
                        ]),
                        dbc.Row([
                            dbc.Col(html.Div(dcc.Graph(id="cmp-skill-bars", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Barras de habilidades<br>Velocidad · Disparo · Pase","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#00e676"},"bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": "#00e67644", "borderwidth": 1}]}}), style={"background": "#0d0d1a", "borderRadius": "8px"}), width=12),
                        ], style={"marginTop": "12px"}),
                    ]),
                ], width=9),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="cmp-table"), width=12),
            ], style={"marginTop": "16px"}),
        ])

    def _tab_undervalued(self) -> html.Div:
        return html.Div([
            # Panel de control
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(" UMBRAL DE INFRAVALORACIÓN (%)", className="filter-label"),
                        dcc.Slider(
                            id="uv-threshold", min=10, max=60, step=5, value=30,
                            marks={i: {"label": f"{i}%", "style": {"color": COLORS["muted"], "fontSize": "10px"}}
                                   for i in range(10, 65, 10)},
                            tooltip={"placement": "bottom"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Div(" VALOR MÍNIMO DE MERCADO (€)", className="filter-label"),
                        dcc.Slider(
                            id="uv-min-value", min=0, max=5_000_000, step=100_000, value=500_000,
                            marks={0: {"label": "0", "style": {"color": COLORS["muted"]}},
                                   1_000_000: {"label": "1M", "style": {"color": COLORS["muted"]}},
                                   5_000_000: {"label": "5M", "style": {"color": COLORS["muted"]}}},
                            tooltip={"placement": "bottom"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Button(
                            " EJECUTAR ANÁLISIS IA",
                            id="uv-run-btn",
                            n_clicks=0, className="btn-run-ai",
                            style={
                                "background": f"linear-gradient(135deg, {COLORS['pitch_green']}, {COLORS['pitch_light']})",
                                "color": COLORS["gold"],
                                "border": f"1px solid {COLORS['gold']}",
                                "borderRadius": "6px",
                                "padding": "10px 20px",
                                "fontSize": "13px",
                                "fontFamily": FONT_FAMILY,
                                "letterSpacing": "1px",
                                "cursor": "pointer",
                                "width": "100%",
                                "marginTop": "18px",
                                "fontWeight": "700",
                                "transition": "all 0.2s ease",
                            },
                        ),
                    ], width=4),
                ], align="center"),
            ], style={
                "background": COLORS["surface"],
                "border": f"1px solid {COLORS['border2']}",
                "borderRadius": "10px",
                "padding": "16px 20px",
                "marginBottom": "16px",
            }),

            dcc.Loading(
                id="uv-loading",
                type="circle",
                color=COLORS["gold"],
                children=[
                    dbc.Row(id="uv-kpi-row", style={"marginBottom": "16px"}),
                    dbc.Row([
                        dbc.Col(html.Div(dcc.Graph(id="uv-scatter", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px", "height": "100%"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Presiona EJECUTAR ANÁLISIS IA<br>para ver los resultados","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#9e9ec8"},"bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#2a2a4a", "borderwidth": 1}]}}), style={"background":"#0d0d1a","borderRadius":"8px"}), width=7),
                        dbc.Col(html.Div(dcc.Graph(id="uv-feat-imp", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px", "height": "100%"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Presiona EJECUTAR ANÁLISIS IA<br>para ver los resultados","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#9e9ec8"},"bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#2a2a4a", "borderwidth": 1}]}}), style={"background":"#0d0d1a","borderRadius":"8px"}), width=5),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(dcc.Graph(id="uv-top20-bar", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px", "height": "100%"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Presiona EJECUTAR ANÁLISIS IA<br>para ver los resultados","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#9e9ec8"},"bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#2a2a4a", "borderwidth": 1}]}}), style={"background":"#0d0d1a","borderRadius":"8px"}), width=6),
                        dbc.Col(html.Div(dcc.Graph(id="uv-league-efficiency", config={"displayModeBar": False}, style={"background": "#0d0d1a", "borderRadius": "8px", "height": "100%"}, figure={"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": "Presiona EJECUTAR ANÁLISIS IA<br>para ver los resultados","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#9e9ec8"},"bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#2a2a4a", "borderwidth": 1}]}}), style={"background":"#0d0d1a","borderRadius":"8px"}), width=6),
                    ], style={"marginTop": "14px"}),
                    dbc.Row([
                        dbc.Col(html.Div(id="uv-table"), width=12),
                    ], style={"marginTop": "14px"}),
                ],
            ),
        ])

    # ─── Pestaña: Diccionario FIFA ─────────────────────────────────

    def _tab_dictionary(self) -> html.Div:
        """Glosario profesional de términos usados en la aplicación."""

        # ── Términos del modelo y métricas ─────────────────────────────
        TERMINOS = [
            {"icono": "", "termino": "Valor Real",               "subtitulo": "value_eur",
             "descripcion": "Precio de mercado actual del jugador en euros. Es el valor con el que el club lo vendería hoy.",
             "ejemplo": "€85.5M", "color": COLORS["electric"], "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "Valor Predicho",           "subtitulo": "predicted_value",
             "descripcion": "Valor calculado por el modelo XGBoost según las estadísticas del jugador. Si supera al Valor Real, el jugador está infravalorado.",
             "ejemplo": "€110M",  "color": COLORS["gold"],     "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "Δ Modelo",                 "subtitulo": "diferencia porcentual",
             "descripcion": "Diferencia en % entre el Valor Predicho y el Valor Real. Positivo = infravalorado. Negativo = sobrevalorado.",
             "ejemplo": "+28.6%", "color": COLORS["sky"],      "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "Veredicto",               "subtitulo": "clasificación del modelo",
             "descripcion": "Resumen del análisis: Infravalorado (Δ > +10%), Sobrevalorado (Δ < -10%) o Precio Justo (entre -10% y +10%).",
             "ejemplo": "Infravalorado", "color": COLORS["undervalued"], "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "R² del Modelo",            "subtitulo": "coeficiente de determinación",
             "descripcion": "Qué tan bien predice el modelo. Va de 0 a 1. Un R² de 0.95 significa que el modelo explica el 95% de la variación de precios.",
             "ejemplo": "0.947", "color": COLORS["electric"],   "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "MAE del Modelo",           "subtitulo": "error absoluto medio",
             "descripcion": "Error promedio en las predicciones. Si es €2.1M, el modelo se equivoca en promedio €2.1M por jugador.",
             "ejemplo": "€2.1M", "color": COLORS["gold"],       "grupo": "Indicadores del Modelo"},
            {"icono": "", "termino": "Overall (OVR)",            "subtitulo": "overall",
             "descripcion": "Valoración general del jugador del 1 al 99. Representa su rendimiento actual total. 99 es la máxima puntuación posible.",
             "ejemplo": "93 — Messi", "color": COLORS["highlight"], "grupo": "Valoraciones del Jugador"},
            {"icono": "", "termino": "Potencial (POT)",          "subtitulo": "potential",
             "descripcion": "Puntuación máxima que puede alcanzar el jugador si se desarrolla correctamente. Clave para evaluar jugadores jóvenes.",
             "ejemplo": "95", "color": COLORS["gold_bright"],   "grupo": "Valoraciones del Jugador"},
            {"icono": "", "termino": "Reputación Internacional", "subtitulo": "international_reputation",
             "descripcion": "Fama mundial del jugador del 1 al 5. Solo los grandes astros llegan a 5 (Messi, Ronaldo). Influye directamente en el valor de mercado.",
             "ejemplo": "5 — estrella mundial", "color": COLORS["sky"], "grupo": "Valoraciones del Jugador"},
            {"icono": "", "termino": "Velocidad",                "subtitulo": "pace",
             "descripcion": "Promedio de Aceleración y Velocidad Sprint. Qué tan rápido puede moverse el jugador.",
             "ejemplo": "94 — jugador muy rápido", "color": "#fb923c", "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Disparo",                  "subtitulo": "shooting",
             "descripcion": "Capacidad ofensiva para disparar al arco: potencia, precisión, definición y volleas.",
             "ejemplo": "92 — delantero élite", "color": COLORS["overvalued"], "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Pase",                     "subtitulo": "passing",
             "descripcion": "Visión, precisión y calidad en la distribución del balón, tanto corto como largo.",
             "ejemplo": "91 — mediocampista creativo", "color": COLORS["electric"], "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Regate",                   "subtitulo": "dribbling",
             "descripcion": "Habilidad técnica para sortear rivales con el balón. Control y destreza en espacios reducidos.",
             "ejemplo": "96 — mejor del mundo", "color": COLORS["highlight"], "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Defensa",                 "subtitulo": "defending",
             "descripcion": "Capacidad para recuperar el balón, marcar rivales y hacer entradas efectivas.",
             "ejemplo": "87 — defensor sólido", "color": COLORS["sky"], "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Físico",                   "subtitulo": "physic",
             "descripcion": "Fuerza en duelos, resistencia física durante el partido y presencia corporal.",
             "ejemplo": "88 — jugador robusto", "color": "#a78bfa", "grupo": "Atributos de Campo"},
            {"icono": "", "termino": "Estirada (GK)",            "subtitulo": "gk_diving",
             "descripcion": "Capacidad del portero para lanzarse a los lados y detener disparos esquinados. Determinante en mano a mano.",
             "ejemplo": "90 — portero acrobático", "color": "#34d399", "grupo": "Atributos de Portero"},
            {"icono": "", "termino": "Agarre (GK)",              "subtitulo": "gk_handling",
             "descripcion": "Seguridad al atrapar el balón sin dejarlo rebotar. Un buen agarre evita segundas oportunidades de gol.",
             "ejemplo": "88", "color": "#34d399", "grupo": "Atributos de Portero"},
            {"icono": "", "termino": "Reflejos (GK)",            "subtitulo": "gk_reflexes",
             "descripcion": "Velocidad de reacción ante disparos cercanos o inesperados. El atributo más importante para un portero de élite.",
             "ejemplo": "92 — felino bajo los palos", "color": "#34d399", "grupo": "Atributos de Portero"},
            {"icono": "", "termino": "Joya Oculta",              "subtitulo": "undervalued player",
             "descripcion": "Jugador cuyo Valor Predicho supera significativamente su Valor Real. Representa una oportunidad de fichaje muy rentable.",
             "ejemplo": "Predicho €50M, Real €20M  +150%", "color": COLORS["undervalued"], "grupo": "Términos de Análisis"},
            {"icono": "", "termino": "Precio Justo",             "subtitulo": "fair value line",
             "descripcion": "Línea diagonal en el gráfico de dispersión. Jugadores sobre la línea están infravalorados; los de abajo, sobrevalorados.",
             "ejemplo": "Línea punteada en el gráfico", "color": COLORS["muted"], "grupo": "Términos de Análisis"},
            {"icono": "", "termino": "Proyección",               "subtitulo": "value projection",
             "descripcion": "Estimación del valor de mercado del jugador en los próximos 1 a 5 años, basada en su trayectoria histórica y el modelo.",
             "ejemplo": "€85M  €120M en 3 años", "color": COLORS["gold"], "grupo": "Términos de Análisis"},
            {"icono": "", "termino": "% Infravalorado",         "subtitulo": "undervalued_pct",
             "descripcion": "Porcentaje en que el Valor Predicho supera al Valor Real. Cuanto mayor, mayor la oportunidad de negocio.",
             "ejemplo": "+75% = gran oportunidad", "color": COLORS["undervalued"], "grupo": "Términos de Análisis"},
        ]

        # ── Posiciones de jugadores de campo ───────────────────────────
        POSICIONES_CAMPO = [
            {"sigla": "ST",  "nombre": "Delantero Centro",          "en": "Striker",
             "descripcion": "Jugador más avanzado del equipo. Su función principal es marcar goles. Se ubica frente al arco rival.",
             "atributos": ["Disparo", "Definición", "Velocidad", "Físico"],
             "referentes": "R. Lewandowski, Karim Benzema"},
            {"sigla": "CF",  "nombre": "Media Punta / Falso 9",     "en": "Centre Forward",
             "descripcion": "Delantero que se mueve entre líneas para generar juego y también definir. Versátil entre el ataque y la creación.",
             "atributos": ["Regate", "Pase", "Disparo", "Visión"],
             "referentes": "L. Messi, Roberto Firmino"},
            {"sigla": "LW",  "nombre": "Extremo Izquierdo",         "en": "Left Winger",
             "descripcion": "Atacante que opera por el lado izquierdo del campo. Usa su velocidad y regate para desbordar y centrar o disparar.",
             "atributos": ["Velocidad", "Regate", "Centro", "Disparo"],
             "referentes": "Sadio Mané, Raheem Sterling"},
            {"sigla": "RW",  "nombre": "Extremo Derecho",           "en": "Right Winger",
             "descripcion": "Igual que el extremo izquierdo pero por el lado derecho. Suele ser zurdo para entrar hacia el centro a disparar.",
             "atributos": ["Velocidad", "Regate", "Disparo", "Centro"],
             "referentes": "Cristiano Ronaldo, Mohamed Salah"},
            {"sigla": "CAM", "nombre": "Mediapunta Central",        "en": "Central Attacking Mid",
             "descripcion": "Centrocampista ofensivo que conecta el mediocampo con el ataque. Es el creador principal de oportunidades de gol.",
             "atributos": ["Visión", "Pase", "Regate", "Disparo"],
             "referentes": "Kevin De Bruyne, Luka Modric"},
            {"sigla": "CM",  "nombre": "Centrocampista Central",    "en": "Central Midfielder",
             "descripcion": "Motor del equipo. Participa tanto en ataque como en defensa. Distribuye el balón y controla el ritmo del partido.",
             "atributos": ["Pase", "Resistencia", "Visión", "Defensa"],
             "referentes": "Toni Kroos, Thiago Alcántara"},
            {"sigla": "CDM", "nombre": "Mediocentro Defensivo",     "en": "Central Defensive Mid",
             "descripcion": "Centrocampista que protege la defensa. Corta balones, presiona rivales y da inicio al juego desde atrás.",
             "atributos": ["Defensa", "Físico", "Intercepciones", "Pase"],
             "referentes": "N'Golo Kanté, Casemiro"},
            {"sigla": "LM",  "nombre": "Mediocampista Izquierdo",   "en": "Left Midfielder",
             "descripcion": "Centrocampista que trabaja el carril izquierdo. Combina tareas defensivas con participación en el ataque.",
             "atributos": ["Resistencia", "Pase", "Centro", "Velocidad"],
             "referentes": "Jordan Henderson"},
            {"sigla": "RM",  "nombre": "Mediocampista Derecho",     "en": "Right Midfielder",
             "descripcion": "Igual que el mediocampista izquierdo pero por el lado derecho del campo.",
             "atributos": ["Resistencia", "Pase", "Centro", "Velocidad"],
             "referentes": "Serge Gnabry"},
            {"sigla": "LB",  "nombre": "Lateral Izquierdo",         "en": "Left Back",
             "descripcion": "Defensor del carril izquierdo. Debe cubrir su zona defensivamente pero también subir a apoyar el ataque.",
             "atributos": ["Defensa", "Velocidad", "Centro", "Resistencia"],
             "referentes": "Andrew Robertson, Alphonso Davies"},
            {"sigla": "RB",  "nombre": "Lateral Derecho",           "en": "Right Back",
             "descripcion": "Defensor del carril derecho. Mismas funciones que el lateral izquierdo. Hoy en día es muy ofensivo.",
             "atributos": ["Defensa", "Velocidad", "Centro", "Resistencia"],
             "referentes": "Trent Alexander-Arnold, Achraf Hakimi"},
            {"sigla": "LWB", "nombre": "Carrilero Izquierdo",       "en": "Left Wing Back",
             "descripcion": "Lateral muy ofensivo que actúa como extremo en sistemas con tres defensores. Cubre todo el carril izquierdo.",
             "atributos": ["Velocidad", "Resistencia", "Centro", "Defensa"],
             "referentes": "Robin Gosens, Marcos Alonso"},
            {"sigla": "RWB", "nombre": "Carrilero Derecho",         "en": "Right Wing Back",
             "descripcion": "Igual que el carrilero izquierdo pero por el lado derecho. Muy importante en sistemas 3-5-2.",
             "atributos": ["Velocidad", "Resistencia", "Centro", "Defensa"],
             "referentes": "Serge Aurier, Kieran Trippier"},
            {"sigla": "CB",  "nombre": "Defensa Central",           "en": "Centre Back",
             "descripcion": "Pilar defensivo del equipo. Impide que los delanteros rivales lleguen al arco. Fundamental en el juego aéreo.",
             "atributos": ["Defensa", "Físico", "Cabezazo", "Barrida"],
             "referentes": "Virgil van Dijk, Sergio Ramos"},
        ]

        # ── Posiciones de portero ──────────────────────────────────────
        POSICIONES_GK = [
            {"sigla": "GK",  "nombre": "Portero",                   "en": "Goalkeeper",
             "descripcion": "El último guardián del equipo. Su misión es evitar que el balón entre al arco. Requiere un conjunto de habilidades exclusivas muy distintas a las de los jugadores de campo.",
             "atributos_gk": [
                 ("Estirada", "gk_diving",      "Lanzarse a los lados para desviar disparos esquinados."),
                 ("Agarre",   "gk_handling",    "Atrapar el balón con seguridad sin dejarlo rebotar."),
                 ("Despeje",  "gk_kicking",     "Potencia y precisión al despejar con el pie o sacar de portería."),
                 ("Reflejos", "gk_reflexes",    "Reacción fulminante ante disparos cercanos o desviados."),
                 ("Velocidad","gk_speed",       "Rapidez para salir del arco y anticipar al delantero rival."),
                 ("Posicionamiento","gk_positioning","Lectura del juego para estar siempre en el lugar correcto."),
             ],
             "referentes": "Manuel Neuer, Thibaut Courtois, Alisson Becker"},
        ]

        GRUPO_COLORES = {
            "Indicadores del Modelo":   COLORS["gold"],
            "Valoraciones del Jugador": COLORS["electric"],
            "Atributos de Campo":       COLORS["undervalued"],
            "Atributos de Portero":     "#34d399",
            "Términos de Análisis":     "#a78bfa",
        }

        COLOR_POS   = COLORS["sky"]
        COLOR_GK    = "#34d399"

        grupos = {}
        for t in TERMINOS:
            grupos.setdefault(t["grupo"], []).append(t)

        # ── Helper: tarjeta de término ────────────────────────────────
        def _tarjeta(t):
            color = t["color"]
            return html.Div([
                html.Div([
                    html.Span(t["icono"], style={"fontSize": "22px", "marginRight": "10px"}),
                    html.Div([
                        html.Div(t["termino"], style={
                            "color": color, "fontFamily": FONT_FAMILY,
                            "fontSize": "14px", "fontWeight": "700", "letterSpacing": "0.5px",
                        }),
                        html.Code(t["subtitulo"], style={
                            "color": COLORS["muted"], "fontSize": "10px",
                            "background": "transparent", "fontFamily": "monospace",
                        }),
                    ]),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "7px"}),
                html.P(t["descripcion"], style={
                    "color": COLORS["text"], "fontSize": "12px",
                    "lineHeight": "1.6", "margin": "0 0 8px 0",
                }),
                html.Div([
                    html.Span("Ej: ", style={"color": COLORS["muted"], "fontSize": "10px",
                                             "fontFamily": FONT_FAMILY}),
                    html.Span(t["ejemplo"], style={
                        "color": color, "fontSize": "11px", "fontWeight": "600",
                        "background": f"{color}18", "padding": "2px 10px", "borderRadius": "10px",
                    }),
                ]),
            ], style={
                "background": COLORS["surface"],
                "border": f"1px solid {color}33",
                "borderLeft": f"3px solid {color}",
                "borderRadius": "8px",
                "padding": "13px 15px",
                "marginBottom": "10px",
            })

        # ── Helper: tarjeta de posición de campo ──────────────────────
        def _tarjeta_posicion(p):
            return html.Div([
                # Encabezado sigla + nombre
                html.Div([
                    html.Div(p["sigla"], style={
                        "background": f"{COLOR_POS}22",
                        "color": COLOR_POS,
                        "border": f"2px solid {COLOR_POS}",
                        "borderRadius": "6px",
                        "padding": "4px 10px",
                        "fontFamily": FONT_FAMILY,
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "letterSpacing": "1px",
                        "minWidth": "54px",
                        "textAlign": "center",
                        "marginRight": "12px",
                        "flexShrink": "0",
                    }),
                    html.Div([
                        html.Div(p["nombre"], style={
                            "color": COLORS["text"], "fontFamily": FONT_FAMILY,
                            "fontSize": "13px", "fontWeight": "700",
                        }),
                        html.Div(p["en"], style={
                            "color": COLORS["muted"], "fontSize": "10px",
                            "fontStyle": "italic", "letterSpacing": "0.3px",
                        }),
                    ]),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

                # Descripción
                html.P(p["descripcion"], style={
                    "color": COLORS["text"], "fontSize": "12px",
                    "lineHeight": "1.6", "margin": "0 0 8px 0",
                }),

                # Atributos clave
                html.Div([
                    html.Span("Atributos: ", style={"color": COLORS["muted"], "fontSize": "10px",
                                                     "fontFamily": FONT_FAMILY, "marginRight": "4px"}),
                    *[html.Span(a, style={
                        "background": f"{COLOR_POS}18",
                        "color": COLOR_POS,
                        "border": f"1px solid {COLOR_POS}44",
                        "borderRadius": "10px",
                        "padding": "1px 8px",
                        "fontSize": "10px",
                        "marginRight": "4px",
                        "display": "inline-block",
                    }) for a in p["atributos"]],
                ], style={"marginBottom": "6px"}),

                # Referentes
                html.Div([
                    html.Span(" Referentes: ", style={"color": COLORS["muted"], "fontSize": "10px",
                                                        "fontFamily": FONT_FAMILY}),
                    html.Span(p["referentes"], style={"color": COLORS["gold"], "fontSize": "11px",
                                                       "fontWeight": "600"}),
                ]),
            ], style={
                "background": COLORS["surface"],
                "border": f"1px solid {COLOR_POS}22",
                "borderLeft": f"3px solid {COLOR_POS}",
                "borderRadius": "8px",
                "padding": "14px 16px",
                "marginBottom": "10px",
            })

        # ── Helper: tarjeta especial portero ─────────────────────────
        def _tarjeta_portero(p):
            return html.Div([
                # Header
                html.Div([
                    html.Div(p["sigla"], style={
                        "background": f"{COLOR_GK}22",
                        "color": COLOR_GK,
                        "border": f"2px solid {COLOR_GK}",
                        "borderRadius": "6px",
                        "padding": "4px 12px",
                        "fontFamily": FONT_FAMILY,
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "marginRight": "12px",
                        "flexShrink": "0",
                    }),
                    html.Div([
                        html.Div(p["nombre"], style={
                            "color": COLORS["text"], "fontFamily": FONT_FAMILY,
                            "fontSize": "14px", "fontWeight": "700",
                        }),
                        html.Div(p["en"], style={
                            "color": COLORS["muted"], "fontSize": "10px",
                            "fontStyle": "italic",
                        }),
                    ]),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

                # Descripción
                html.P(p["descripcion"], style={
                    "color": COLORS["text"], "fontSize": "12px",
                    "lineHeight": "1.6", "margin": "0 0 12px 0",
                }),

                # Atributos exclusivos en tabla
                html.Div("ATRIBUTOS EXCLUSIVOS DEL PORTERO", style={
                    "color": COLOR_GK, "fontFamily": FONT_FAMILY,
                    "fontSize": "10px", "letterSpacing": "1px",
                    "marginBottom": "8px", "fontWeight": "700",
                }),
                *[html.Div([
                    html.Div([
                        html.Span(nombre, style={
                            "background": f"{COLOR_GK}22",
                            "color": COLOR_GK,
                            "border": f"1px solid {COLOR_GK}44",
                            "borderRadius": "4px",
                            "padding": "2px 10px",
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "fontFamily": FONT_FAMILY,
                            "minWidth": "130px",
                            "display": "inline-block",
                            "marginRight": "10px",
                        }),
                        html.Code(sigla, style={
                            "color": COLORS["muted"], "fontSize": "10px",
                            "fontFamily": "monospace", "marginRight": "10px",
                        }),
                        html.Span(desc, style={
                            "color": COLORS["text"], "fontSize": "11px",
                        }),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "padding": "6px 0",
                    "borderBottom": f"1px solid {COLORS['border']}",
                }) for nombre, sigla, desc in p["atributos_gk"]],

                # Referentes
                html.Div([
                    html.Span(" Porteros de referencia: ", style={
                        "color": COLORS["muted"], "fontSize": "10px",
                        "fontFamily": FONT_FAMILY,
                    }),
                    html.Span(p["referentes"], style={
                        "color": COLORS["gold"], "fontSize": "11px", "fontWeight": "600",
                    }),
                ], style={"marginTop": "12px"}),
            ], style={
                "background": COLORS["surface"],
                "border": f"1px solid {COLOR_GK}33",
                "borderLeft": f"3px solid {COLOR_GK}",
                "borderRadius": "8px",
                "padding": "16px 18px",
                "marginBottom": "10px",
            })

        # ── Construir secciones de grupos de términos ─────────────────
        def _seccion_grupo(grupo, terminos, color_g):
            return html.Div([
                html.Div([
                    html.Span("▍", style={"color": color_g, "marginRight": "8px", "fontSize": "16px"}),
                    html.Span(grupo.upper(), style={
                        "color": color_g, "fontFamily": FONT_FAMILY,
                        "fontSize": "11px", "letterSpacing": "1.5px", "fontWeight": "700",
                    }),
                    html.Span(f" {len(terminos)} términos", style={
                        "color": COLORS["muted"], "fontSize": "10px", "marginLeft": "8px",
                    }),
                ], style={
                    "background": f"{color_g}10", "borderBottom": f"1px solid {color_g}33",
                    "padding": "8px 14px", "marginBottom": "10px",
                    "borderRadius": "6px 6px 0 0",
                }),
                *[_tarjeta(t) for t in terminos],
            ], style={
                "border": f"1px solid {color_g}33", "borderRadius": "8px",
                "padding": "0 10px 10px 10px", "marginBottom": "16px",
            })

        cols_terminos = [[], []]
        for i, (grupo, terminos) in enumerate(grupos.items()):
            color_g = GRUPO_COLORES.get(grupo, COLORS["muted"])
            cols_terminos[0 if i < 3 else 1].append(
                _seccion_grupo(grupo, terminos, color_g)
            )

        # ── Sección de posiciones de campo ────────────────────────────
        # Dividir en 2 columnas de 7 y 7
        mitad = len(POSICIONES_CAMPO) // 2
        col_pos_izq = [_tarjeta_posicion(p) for p in POSICIONES_CAMPO[:mitad]]
        col_pos_der = [_tarjeta_posicion(p) for p in POSICIONES_CAMPO[mitad:]]

        def _header_seccion(icono, titulo, subtitulo, color, total):
            return html.Div([
                html.Div([
                    html.Span(icono, style={"fontSize": "24px", "marginRight": "12px"}),
                    html.Div([
                        html.Div(titulo, style={
                            "color": color, "fontFamily": FONT_FAMILY,
                            "fontSize": "15px", "fontWeight": "700", "letterSpacing": "1px",
                        }),
                        html.Div(subtitulo, style={
                            "color": COLORS["muted"], "fontSize": "11px",
                        }),
                    ]),
                    html.Div(f"{total} posiciones", style={
                        "marginLeft": "auto",
                        "background": f"{color}18",
                        "color": color,
                        "border": f"1px solid {color}44",
                        "borderRadius": "20px",
                        "padding": "3px 12px",
                        "fontSize": "10px",
                        "fontFamily": FONT_FAMILY,
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                "background": f"linear-gradient(135deg, {color}15 0%, transparent 100%)",
                "border": f"1px solid {color}33",
                "borderRadius": "10px",
                "padding": "14px 20px",
                "marginBottom": "16px",
            })

        # ── Layout final ──────────────────────────────────────────────
        return html.Div([

            # ── Banner principal ──────────────────────────────────────
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4([html.Span("", style={"marginRight": "10px"}), "GLOSARIO PROFESIONAL"],
                                style={"color": COLORS["gold"], "fontFamily": FONT_FAMILY,
                                       "letterSpacing": "2px", "margin": "0", "fontSize": "20px"}),
                        html.P("Guía completa para entender los términos, métricas, atributos y posiciones que se muestran en esta plataforma",
                               style={"color": COLORS["muted"], "fontSize": "12px", "margin": "6px 0 0 0"}),
                    ], width=8),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.Div(str(len(TERMINOS)), style={
                                    "color": COLORS["electric"], "fontSize": "26px",
                                    "fontFamily": FONT_FAMILY, "fontWeight": "700", "lineHeight": "1",
                                }),
                                html.Div("métricas", style={"color": COLORS["muted"], "fontSize": "9px"}),
                            ], style={"textAlign": "center"})),
                            dbc.Col(html.Div([
                                html.Div(str(len(POSICIONES_CAMPO)), style={
                                    "color": COLOR_POS, "fontSize": "26px",
                                    "fontFamily": FONT_FAMILY, "fontWeight": "700", "lineHeight": "1",
                                }),
                                html.Div("posiciones", style={"color": COLORS["muted"], "fontSize": "9px"}),
                            ], style={"textAlign": "center"})),
                            dbc.Col(html.Div([
                                html.Div("1", style={
                                    "color": COLOR_GK, "fontSize": "26px",
                                    "fontFamily": FONT_FAMILY, "fontWeight": "700", "lineHeight": "1",
                                }),
                                html.Div("portero", style={"color": COLORS["muted"], "fontSize": "9px"}),
                            ], style={"textAlign": "center"})),
                        ]),
                    ], width=4),
                ], align="center"),
            ], style={
                "background": f"linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface2']} 100%)",
                "border": f"1px solid {COLORS['border2']}", "borderRadius": "10px",
                "padding": "20px 28px", "marginBottom": "24px",
            }),

            # ── Navegación por chips ──────────────────────────────────
            html.Div([
                *[html.Span(grupo, style={
                    "background": f"{GRUPO_COLORES.get(grupo, COLORS['muted'])}22",
                    "color": GRUPO_COLORES.get(grupo, COLORS["muted"]),
                    "border": f"1px solid {GRUPO_COLORES.get(grupo, COLORS['muted'])}55",
                    "borderRadius": "20px", "padding": "4px 14px", "fontSize": "10px",
                    "fontFamily": FONT_FAMILY, "marginRight": "8px", "marginBottom": "6px",
                    "display": "inline-block", "letterSpacing": "0.5px",
                }) for grupo in grupos.keys()],
                html.Span("Posiciones de Campo", style={
                    "background": f"{COLOR_POS}22", "color": COLOR_POS,
                    "border": f"1px solid {COLOR_POS}55",
                    "borderRadius": "20px", "padding": "4px 14px", "fontSize": "10px",
                    "fontFamily": FONT_FAMILY, "marginRight": "8px", "marginBottom": "6px",
                    "display": "inline-block",
                }),
                html.Span("Portero (GK)", style={
                    "background": f"{COLOR_GK}22", "color": COLOR_GK,
                    "border": f"1px solid {COLOR_GK}55",
                    "borderRadius": "20px", "padding": "4px 14px", "fontSize": "10px",
                    "fontFamily": FONT_FAMILY, "marginRight": "8px", "marginBottom": "6px",
                    "display": "inline-block",
                }),
            ], style={"marginBottom": "24px", "lineHeight": "2.4"}),

            # ══ SECCIÓN 1: Términos y métricas ════════════════════════
            html.Div([
                html.Div("▍ TÉRMINOS Y MÉTRICAS DE LA PLATAFORMA", style={
                    "color": COLORS["gold"], "fontFamily": FONT_FAMILY,
                    "fontSize": "13px", "letterSpacing": "2px", "fontWeight": "700",
                    "marginBottom": "16px",
                }),
                dbc.Row([
                    dbc.Col(cols_terminos[0], width=6),
                    dbc.Col(cols_terminos[1], width=6),
                ]),
            ], style={
                "background": COLORS["surface2"],
                "border": f"1px solid {COLORS['border2']}",
                "borderRadius": "12px", "padding": "20px 20px 10px 20px",
                "marginBottom": "24px",
            }),

            # ══ SECCIÓN 2: Posiciones de campo ════════════════════════
            html.Div([
                _header_seccion("", "POSICIONES DE JUGADORES DE CAMPO",
                                "Guía de las posiciones tácticas que verás en el apartado de Comparación y Joyas",
                                COLOR_POS, len(POSICIONES_CAMPO)),
                dbc.Row([
                    dbc.Col(col_pos_izq, width=6),
                    dbc.Col(col_pos_der, width=6),
                ]),
            ], style={
                "background": COLORS["surface2"],
                "border": f"1px solid {COLOR_POS}33",
                "borderRadius": "12px", "padding": "20px 20px 10px 20px",
                "marginBottom": "24px",
            }),

            # ══ SECCIÓN 3: Portero ════════════════════════════════════
            html.Div([
                _header_seccion("", "PORTERO (GOALKEEPER — GK)",
                                "Posición única con atributos exclusivos distintos a todos los jugadores de campo",
                                COLOR_GK, 1),
                *[_tarjeta_portero(p) for p in POSICIONES_GK],
            ], style={
                "background": COLORS["surface2"],
                "border": f"1px solid {COLOR_GK}33",
                "borderRadius": "12px", "padding": "20px 20px 10px 20px",
                "marginBottom": "24px",
            }),

        ])


    # ─── Helper: slot de búsqueda por jugador ──────────────────────

    def _search_slot(self, idx: int, color: str, label: str) -> html.Div:
        return html.Div([
            html.Div([
                html.Span("●", style={"color": color, "marginRight": "6px", "fontSize": "10px"}),
                html.Span(label, style={"color": color, "fontSize": "11px",
                                        "fontFamily": FONT_FAMILY, "letterSpacing": "0.5px"}),
            ], style={"marginBottom": "6px"}),
            dbc.Input(
                id=f"cmp-search-{idx}",
                placeholder=f"Buscar {label.lower()}...",
                type="text", debounce=True,
                style={"marginBottom": "5px", "fontSize": "12px"},
            ),
            dcc.Loading(type="dot", color=color,
                        children=html.Div(id=f"cmp-results-{idx}")),
        ])

    # ─── Callbacks ─────────────────────────────────────────────────

    def _register_callbacks(self):

        # Routing de pestañas
        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "active_tab"),
        )
        def render_tab(tab):
            if tab == "tab-valuation":
                return self._tab_valuation()
            elif tab == "tab-comparison":
                return self._tab_comparison()
            elif tab == "tab-dictionary":
                return self._tab_dictionary()
            else:
                return self._tab_undervalued()

        # Actualizar ligas según temporada
        @self.app.callback(
            Output("global-league", "options"),
            Output("global-league", "value"),
            Input("global-season", "value"),
        )
        def update_leagues(season):
            df = self._loader.load_season(season)
            leagues = ["Todas las Ligas"] + sorted(df["league_name"].dropna().unique().tolist())
            return [{"label": l, "value": l} for l in leagues], "Todas las Ligas"

        # Búsqueda de jugador en pestaña valoración
        @self.app.callback(
            Output("val-search-results", "children"),
            Input("val-search", "value"),
            State("global-season", "value"),
        )
        def search_player_val(query, season):
            if not query or len(query) < 2:
                return html.P("Escribe al menos 2 caracteres",
                              style={"color": COLORS["muted"], "fontSize": "11px"})
            df = self._loader.load_season(season)
            results = df[
                df["short_name"].str.contains(query, case=False, na=False) |
                df["long_name"].str.contains(query, case=False, na=False)
            ].head(8)
            if results.empty:
                return html.P("Sin resultados", style={"color": COLORS["muted"]})

            return html.Div([
                html.Button(
                    f"{row['short_name']}  ·  {row.get('club_name', '')}  ·  {row.get('overall', '')} ",
                    id={"type": "val-player-btn", "index": int(row["sofifa_id"])},
                    className="player-btn",
                    style={"width": "100%", "marginBottom": "4px", "padding": "7px 10px",
                           "cursor": "pointer", "borderRadius": "4px", "display": "block"},
                )
                for _, row in results.iterrows()
            ])
        

        # Gráficos de valoración al seleccionar jugador
        @self.app.callback(
            Output("val-player-card", "children"),
            Output("val-projection-chart", "figure"),
            Output("val-overall-chart", "figure"),
            Output("val-skills-evolution-chart", "figure"),
            Output("val-feature-imp-chart", "figure"),
            Output("val-kpi-row", "children"),
            Output("store-selected-player-id", "data"),
            Input({"type": "val-player-btn", "index": dash.ALL}, "n_clicks"),
            Input("global-season", "value"),
            State("val-years", "value"),
            State("store-selected-player-id", "data"),
            prevent_initial_call=True,
        )
        def update_valuation_tab(n_clicks, season, years, stored_player_id):
            triggered_id = callback_context.triggered_id

            # Si el trigger es la temporada, usar el jugador guardado
            if triggered_id == "global-season":
                player_id = stored_player_id
                if not player_id:
                    return [dash.no_update] * 7
            else:
                # Si el trigger es un botón de jugador
                if not n_clicks or not any(c for c in n_clicks if c):
                    return [dash.no_update] * 7
                if not triggered_id or not isinstance(triggered_id, dict):
                    return [dash.no_update] * 7
                player_id = triggered_id["index"]

            history = self._loader.get_player_history(player_id)
            if history.empty:
                return html.P("Jugador no encontrado"), {}, {}, {}, {}, [], dash.no_update

            latest = history.iloc[-1]
            player_name = latest["short_name"]

            card = self._player_info_card(latest)

            if not self._model._is_trained:
                self._model.train("FIFA 21")
            projection = self._model.project_value(player_id, years=years or 3)

            fig_proj   = player_projection_chart(history, projection, player_name)
            fig_ovr    = player_overall_history(history, player_name)
            fig_skills = player_skills_evolution(history, player_name)
            fig_feat   = feature_importance_chart(self._model.feature_importance())

            current_val   = float(latest.get("value_eur", 0))
            predicted_val = self._model.predict_single(latest)
            ratio = (predicted_val / current_val - 1) * 100 if current_val > 0 else 0
            label = " Infravalorado" if ratio > 10 else (" Sobrevalorado" if ratio < -10 else "Precio Justo")

            kpis = dbc.Row([
                self._kpi_card("Valor Real",      f"€{current_val/1e6:.1f}M",    COLORS["electric"]),
                self._kpi_card("Valor Predicho",  f"€{predicted_val/1e6:.1f}M",  COLORS["gold"]),
                self._kpi_card("Δ Modelo",         f"{ratio:+.1f}%",             COLORS["sky"]),
                self._kpi_card("Veredicto",        label,
                               COLORS["undervalued"] if ratio > 10 else COLORS["overvalued"]),
                self._kpi_card("Potencial",        str(int(latest.get("potential", 0))), COLORS["gold_bright"]),
                self._kpi_card("Overall",          str(int(latest.get("overall", 0))),   COLORS["electric"]),
            ])
            return card, fig_proj, fig_ovr, fig_skills, fig_feat, kpis, player_id

        # Comparación: guardar jugadores en stores
        for _slot in [1, 2, 3]:
            @self.app.callback(
                Output(f"store-cmp-id-{_slot}", "data"),
                Input({"type": f"cmp-select-btn-{_slot}", "index": dash.ALL}, "n_clicks"),
                prevent_initial_call=True,
            )
            def _save_selected(n_clicks, _s=_slot):
                if not n_clicks or not any(c for c in n_clicks if c):
                    return dash.no_update
                tid = callback_context.triggered_id
                if not tid or not isinstance(tid, dict):
                    return dash.no_update
                return tid["index"]

        # Comparación: gráficos
        @self.app.callback(
            Output("cmp-radar", "figure"),
            Output("cmp-skill-bars", "figure"),
            Output("cmp-salary-scatter", "figure"),
            Output("cmp-table", "children"),
            Input("store-cmp-id-1", "data"),
            Input("store-cmp-id-2", "data"),
            Input("store-cmp-id-3", "data"),
            Input("cmp-radar-dims", "value"),
            Input("global-season", "value"),
            prevent_initial_call=True,
        )
        def update_comparison(pid1, pid2, pid3, radar_dims, season):
            df = self._loader.load_season(season)
            ids = [p for p in [pid1, pid2, pid3] if p is not None]

            def _empty_fig(msg="Selecciona jugadores<br>para comparar su perfil", color="#4fc3f7", border="#4fc3f744"):
                return {"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a", "font": {"color": "#e8eaf6"}, "xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": msg, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 13, "color": color}, "align": "center", "bgcolor": "#1a1a2e", "borderpad": 18, "bordercolor": border, "borderwidth": 1}]}}

            if not ids:
                return _empty_fig("Selecciona jugadores<br>para comparar su radar", "#4fc3f7", "#4fc3f744"), _empty_fig("Barras de habilidades<br>Velocidad · Disparo · Pase", "#00e676", "#00e67644"), _empty_fig("Salario vs Rendimiento<br>Selecciona jugadores primero", "#c9a84c", "#c9a84c44"), html.P("Selecciona jugadores para comparar.", style={"color": COLORS["muted"], "padding": "20px"})

            dims = radar_dims or ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
            players_data, rows_info = [], []

            for pid in ids[:3]:
                row = df[df["sofifa_id"] == pid]
                if row.empty:
                    continue
                r = row.iloc[0]
                vals = [float(r.get(d, 0) or 0) for d in dims]
                players_data.append({"name": r["short_name"], "values": vals})
                rows_info.append(r)

            if not players_data:
                return _empty_fig("Jugador no encontrado<br>en esta temporada", "#c9a84c", "#c9a84c44"), _empty_fig("Barras de habilidades<br>Velocidad · Disparo · Pase", "#00e676", "#00e67644"), _empty_fig("Salario vs Rendimiento<br>Selecciona jugadores primero", "#c9a84c", "#c9a84c44"), dash.no_update

            fig_radar = radar_comparison(players_data, dims)

            is_gk_mode = dims and any("gk_" in d for d in dims)
            if is_gk_mode:
                detail_cols = ["gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]
            else:
                detail_cols = ["pace", "shooting", "passing", "dribbling", "defending", "physic", "skill_ball_control", "movement_reactions", "mentality_vision"]
            detail_cols = [c for c in detail_cols if c in df.columns]
            players_detail = [{"name": r["short_name"], **{c: r.get(c, 0) for c in detail_cols}}
                               for r in rows_info]
            fig_bars    = bar_skill_comparison(players_detail, detail_cols)
            fig_scatter = salary_vs_performance(df.head(400))

            cols = ["short_name", "age", "nationality", "club_name", "league_name",
                    "overall", "potential", "value_eur", "wage_eur", "primary_position"]
            tbl_df = pd.DataFrame([{c: r.get(c, "N/A") for c in cols} for r in rows_info])
            tbl_df["value_eur"] = tbl_df["value_eur"].apply(
                lambda v: f"€{v/1e6:.1f}M" if pd.notna(v) and v != "N/A" else "N/A")
            tbl_df["wage_eur"] = tbl_df["wage_eur"].apply(
                lambda v: f"€{v:,.0f}/sem" if pd.notna(v) and v != "N/A" else "N/A")

            table = dbc.Table.from_dataframe(
                tbl_df.rename(columns={
                    "short_name": "Jugador", "age": "Edad", "nationality": "País",
                    "club_name": "Club", "league_name": "Liga", "overall": "OVR",
                    "potential": "POT", "value_eur": "Valor", "wage_eur": "Salario",
                    "primary_position": "Pos",
                }),
                striped=True, bordered=True, hover=True,
                style={"color": COLORS["text"], "fontSize": "12px"},
            )
            return fig_radar, fig_bars, fig_scatter, table

        # Joyas ocultas: análisis con IA
        @self.app.callback(
            Output("uv-scatter", "figure"),
            Output("uv-feat-imp", "figure"),
            Output("uv-top20-bar", "figure"),
            Output("uv-league-efficiency", "figure"),
            Output("uv-kpi-row", "children"),
            Output("uv-table", "children"),
            Output("store-undervalued", "data"),
            Input("uv-run-btn", "n_clicks"),
            State("uv-threshold", "value"),
            State("uv-min-value", "value"),
            State("global-season", "value"),
            State("global-position", "value"),
            State("global-league", "value"),
            State("global-age", "value"),
            prevent_initial_call=True,
        )
        def run_undervalued_analysis(n_clicks, threshold, min_value, season, position, league, age_range):
            if not n_clicks:
                return [dash.no_update] * 7

            metrics = self._model.train(season)
            df = self._loader.load_season(season)
            df = self._apply_global_filters(df, position, league, age_range, 200)

            ratio = 1 - (threshold / 100)
            undervalued = self._model.find_undervalued(df, threshold_ratio=ratio, min_value=min_value)

            if undervalued.empty:
                msg = html.P("No se encontraron jugadores con ese umbral. Reduce el porcentaje.",
                             style={"color": COLORS["muted"], "padding": "20px"})
                return {"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": " Sin jugadores con ese umbral — prueba reduciendo el %","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}, {"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": " Sin jugadores con ese umbral — prueba reduciendo el %","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}, {"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": " Sin jugadores con ese umbral — prueba reduciendo el %","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}, {"data": [], "layout": {"paper_bgcolor": "#0d0d1a", "plot_bgcolor": "#0d0d1a","font": {"color": "#e8eaf6"},"xaxis": {"visible": False}, "yaxis": {"visible": False},"annotations": [{"text": " Sin jugadores con ese umbral — prueba reduciendo el %","xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,"showarrow": False, "font": {"size": 13, "color": "#c9a84c"}, "align": "center","bgcolor": "#1a1a2e", "borderpad": 16, "bordercolor": "#c9a84c44", "borderwidth": 1}]}}, [], msg, None

            kpis = dbc.Row([
                self._kpi_card("R² Modelo",          f"{metrics['r2']:.3f}",                    COLORS["electric"]),
                self._kpi_card("MAE Modelo",         f"€{metrics['mae']/1e6:.1f}M",             COLORS["gold"]),
                self._kpi_card("Joyas Encontradas",  str(len(undervalued)),                      COLORS["undervalued"]),
                self._kpi_card("Ahorro Potencial",   f"€{undervalued['value_gap'].sum()/1e6:.0f}M", COLORS["sky"]),
            ])

            fig_scatter  = undervalued_scatter(undervalued.head(200))
            fig_feat     = feature_importance_chart(self._model.feature_importance())
            fig_top20    = undervalued_bar_top20(undervalued)
            fig_league   = gems_by_position(undervalued)

            display_cols = ["short_name", "age", "nationality", "club_name", "league_name",
                            "overall", "value_eur", "predicted_value", "undervalued_pct", "wage_eur"]
            display_cols = [c for c in display_cols if c in undervalued.columns]
            tbl = undervalued[display_cols].head(20).copy()
            tbl["value_eur"]       = tbl["value_eur"].apply(lambda v: f"€{v/1e6:.1f}M")
            tbl["predicted_value"] = tbl["predicted_value"].apply(lambda v: f"€{v/1e6:.1f}M")
            tbl["undervalued_pct"] = tbl["undervalued_pct"].apply(lambda v: f"+{v:.0f}%")
            if "wage_eur" in tbl.columns:
                tbl["wage_eur"] = tbl["wage_eur"].apply(lambda v: f"€{v:,.0f}/sem")

            rename = {
                "short_name": "Jugador", "age": "Edad", "nationality": "País",
                "club_name": "Club", "league_name": "Liga", "overall": "OVR",
                "value_eur": "Valor Real", "predicted_value": "Valor Predicho",
                "undervalued_pct": "% Infravalorado", "wage_eur": "Salario",
            }
            table_html = dbc.Table.from_dataframe(
                tbl.rename(columns=rename),
                striped=True, bordered=True, hover=True, responsive=True,
                style={"color": COLORS["text"], "fontSize": "12px"},
            )
            return fig_scatter, fig_feat, fig_top20, fig_league, kpis, table_html, undervalued.to_json()

        # Búsqueda en comparación por slot
        for idx in [1, 2, 3]:
            @self.app.callback(
                Output(f"cmp-results-{idx}", "children"),
                Input(f"cmp-search-{idx}", "value"),
                State("global-season", "value"),
                prevent_initial_call=True,
            )
            def search_cmp(query, season, _idx=idx):
                return self._search_results_for_comparison(query, season, _idx)
            @self.app.callback(
                Output("cmp-radar-dims", "value"),
                Input("btn-outfield", "n_clicks"),
                Input("btn-gk", "n_clicks"),
                prevent_initial_call=True,
            )
            def toggle_radar_dims(btn_outfield, btn_gk):
                triggered = callback_context.triggered_id
                if triggered == "btn-gk":
                    return ["gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]
                return ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
            


    # ─── Helpers de UI ─────────────────────────────────────────────

    def _search_results_for_comparison(self, query, season, idx):
        if not query or len(query) < 2:
            return html.P("Escribe para buscar",
                          style={"color": COLORS["muted"], "fontSize": "11px"})
        df = self._loader.load_season(season)
        results = df[df["short_name"].str.contains(query, case=False, na=False)].head(6)
        if results.empty:
            return html.P("Sin resultados", style={"color": COLORS["muted"]})

        palette = [COLORS["gold"], COLORS["electric"], COLORS["sky"]]
        return html.Div([
            html.Button(
                f"{row['short_name']}  ·  {row.get('overall', '')} ",
                id={"type": f"cmp-select-btn-{idx}", "index": int(row["sofifa_id"])},
                className="player-btn",
                style={"width": "100%", "marginBottom": "4px", "padding": "6px 10px",
                       "cursor": "pointer", "borderRadius": "4px", "display": "block",
                       "borderColor": palette[idx - 1] + "44"},
            )
            for _, row in results.iterrows()
        ])

    def _player_info_card(self, row: pd.Series) -> html.Div:
        fields = [
            ("Nombre",   row.get("long_name", row.get("short_name", ""))),
            ("Edad",     str(int(row.get("age", 0)))),
            ("País",     row.get("nationality", "")),
            ("Club",     row.get("club_name", "")),
            ("Liga",     row.get("league_name", "")),
            ("Posición", row.get("primary_position", "")),
            ("Overall",  str(int(row.get("overall", 0)))),
            ("Potencial",str(int(row.get("potential", 0)))),
        ]
        return html.Div([
            html.Div(
                f" {row.get('short_name', '').upper()}",
                style={"color": COLORS["gold"], "fontFamily": FONT_FAMILY,
                       "fontSize": "13px", "marginBottom": "8px", "letterSpacing": "0.5px"},
            ),
            *[
                html.Div([
                    html.Span(k, style={"color": COLORS["muted"], "fontSize": "10px",
                                        "fontFamily": FONT_FAMILY, "letterSpacing": "0.5px",
                                        "textTransform": "uppercase", "minWidth": "70px",
                                        "display": "inline-block"}),
                    html.Span(str(v), style={"color": COLORS["text"], "fontSize": "11px",
                                             "fontWeight": "600"}),
                ], style={"marginBottom": "4px"})
                for k, v in fields
            ],
        ], style={"fontSize": "11px"})

    @staticmethod
    def _kpi_card(title: str, value: str, color: str) -> dbc.Col:
        return dbc.Col(
            html.Div([
                html.Div(title, style={
                    "color": "#5a5a8a",
                    "fontSize": "9px",
                    "fontFamily": FONT_FAMILY,
                    "letterSpacing": "2px",
                    "textTransform": "uppercase",
                    "marginBottom": "8px",
                }),
                html.Div(value, style={
                    "color": color,
                    "fontWeight": "700",
                    "fontFamily": FONT_FAMILY,
                    "fontSize": "22px",
                    "lineHeight": "1",
                    "textShadow": f"0 0 24px {color}66, 0 0 48px {color}22",
                    "letterSpacing": "0.5px",
                }),
                html.Div(style={
                    "height": "2px",
                    "marginTop": "10px",
                    "background": f"linear-gradient(90deg, {color} 0%, {color}44 60%, transparent 100%)",
                    "borderRadius": "2px",
                }),
            ], className="kpi-card", style={
                "borderLeftColor": color,
                "borderLeftWidth": "4px",
            }),
            width=2,
        )

    def _apply_global_filters(self, df, position, league, age_range, max_value_m):
        if position and position != "Todas":
            df = df[df["primary_position"] == position]
        if league and league != "Todas las Ligas":
            df = df[df["league_name"] == league]
        if age_range:
            df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
        if max_value_m and max_value_m < 200:
            df = df[df["value_eur"] <= max_value_m * 1_000_000]
        return df

    # ─── Lanzamiento ───────────────────────────────────────────────

    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        print(f"\n Football Scout Analytics · Edición Premium")
        print(f" Servidor en http://{host}:{port}/\n")
        self.app.run(host=host, port=port, debug=debug)