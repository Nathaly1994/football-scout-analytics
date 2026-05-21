"""
Visualizaciones basadas en principios de Storytelling con Datos
(Cole Nussbaumer Knaflic) — Resaltar lo importante, atenuar el resto.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# Diccionario de traducción de habilidades al español
_SKILL_ES = {
    "pace": "Velocidad", "shooting": "Disparo", "passing": "Pase",
    "dribbling": "Regate", "defending": "Defensa", "physic": "Físico",
    "skill ball control": "Control de Balón", "movement reactions": "Reacción",
    "mentality vision": "Visión", "mentality composure": "Compostura",
    "mentality aggression": "Agresividad", "mentality positioning": "Posicionamiento",
    "power stamina": "Resistencia", "power strength": "Fuerza",
    "power shot power": "Potencia de Tiro", "power long shots": "Tiro Lejano",
    "movement acceleration": "Aceleración", "movement sprint speed": "Velocidad Sprint",
    "movement agility": "Agilidad", "movement balance": "Equilibrio",
    "attacking crossing": "Centro", "attacking finishing": "Finalización",
    "attacking heading accuracy": "Remate de Cabeza", "attacking short passing": "Pase Corto",
    "attacking volleys": "Volea", "skill dribbling": "Regate Técnico",
    "skill curve": "Efecto", "skill long passing": "Pase Largo",
    "defending standing tackle": "Entrada", "defending sliding tackle": "Barrida",
    "overall": "Valoración General", "potential": "Potencial",
    "wage eur": "Salario (€)", "value eur": "Valor (€)",
    "international reputation": "Reputación Int.",
    "age": "Edad", "weak foot": "Pie Débil", "skill moves": "Habilidades",
    "gk diving": "Portero Estirada", "gk handling": "Portero Control",
    "gk kicking": "Portero Saque", "gk reflexes": "Portero Reflejos",
    "gk speed": "Portero Velocidad", "gk positioning": "Portero Posicionamiento",
}

def _es(nombre: str) -> str:
    """Traduce nombre de columna al español."""
    key = nombre.replace("_", " ").lower()
    return _SKILL_ES.get(key, nombre.replace("_", " ").title())


COLORS = {
    "primary":      "#1a73e8",
    "secondary":    "#34a853",
    "accent":       "#fbbc05",
    "danger":       "#ea4335",
    "dark":         "#0d0d1a",
    "surface":      "#1a1a2e",
    "surface2":     "#16213e",
    "border":       "#2a2a4a",
    "border2":      "#3a3a6e",
    "text":         "#e8eaf6",
    "muted":        "#9e9ec8",
    "undervalued":  "#00e5ff",
    "overvalued":   "#ff4081",
    "gold":         "#c9a84c",
    "gold_bright":  "#f0c060",
    "electric":     "#00e5ff",
    "sky":          "#40c4ff",
    "pitch_green":  "#1a4a2e",
    "pitch_light":  "#2d7a4a",
    "highlight":    "#f0c060",
    "gray_line":    "#3a3a5e",
    "gray_text":    "#6e6e9e",
}

FONT_FAMILY = "Rajdhani, Barlow Condensed, Arial, sans-serif"
FONT_BODY    = "Barlow, Inter, Arial, sans-serif"

DARK_LAYOUT = dict(
    paper_bgcolor=COLORS["dark"],
    plot_bgcolor=COLORS["dark"],
    font=dict(color=COLORS["text"], family="Barlow, Inter, Arial, sans-serif", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        showline=True, linecolor=COLORS["border2"],
        tickfont=dict(color=COLORS["gray_text"]),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        showline=False,
        tickfont=dict(color=COLORS["gray_text"]),
    ),
    margin=dict(l=55, r=30, t=60, b=50),
)


def _base_fig(**kwargs) -> go.Figure:
    fig = go.Figure(**kwargs)
    fig.update_layout(**DARK_LAYOUT)
    return fig


def _empty_fig(mensaje: str) -> go.Figure:
    """Figura vacía con mensaje centrado."""
    fig = _base_fig()
    fig.add_annotation(
        text=mensaje,
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS["muted"]),
        bgcolor=COLORS["surface"],
        borderpad=12,
    )
    return fig


# ── Gráfico 1: Proyección ──────────────────────────────────────────
def player_projection_chart(history_df, projection_df, player_name):
    fig = _base_fig()
    if not history_df.empty:
        fig.add_trace(go.Scatter(
            x=history_df["season_year"],
            y=history_df["value_eur"] / 1_000_000,
            mode="lines+markers", name="Valor Histórico",
            line=dict(color=COLORS["gray_line"], width=2),
            marker=dict(size=6, color=COLORS["gray_line"]),
            hovertemplate="<b>%{x}</b><br>Valor: €%{y:.1f}M<extra></extra>",
        ))
    if not projection_df.empty:
        proj_vals = projection_df["projected_value"] / 1_000_000
        upper = proj_vals * 1.15
        lower = proj_vals * 0.85
        fig.add_trace(go.Scatter(
            x=pd.concat([projection_df["year"], projection_df["year"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(240,192,96,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Banda ±15%", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=projection_df["year"], y=proj_vals,
            mode="lines+markers", name="Proyección",
            line=dict(color=COLORS["highlight"], width=3, dash="dash"),
            marker=dict(size=9, color=COLORS["highlight"], symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Proyección: €%{y:.1f}M<extra></extra>",
        ))
        final_val = proj_vals.iloc[-1]
        final_year = projection_df["year"].iloc[-1]
        fig.add_annotation(
            x=final_year, y=final_val,
            text=f"<b>€{final_val:.1f}M</b>",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["highlight"],
            font=dict(color=COLORS["highlight"], size=13),
            bgcolor=COLORS["surface"], bordercolor=COLORS["highlight"], borderwidth=1,
        )
    if not history_df.empty and not projection_df.empty:
        split_year = history_df["season_year"].max()
        fig.add_vline(x=split_year, line_dash="dot",
                      line_color=COLORS["gray_text"], opacity=0.5,
                      annotation_text="Hoy",
                      annotation_font_color=COLORS["gray_text"])
    fig.update_layout(
        title=dict(text=f"Trayectoria y Proyección — {player_name}", font=dict(size=17)),
        xaxis_title="Año", yaxis_title="Valor de Mercado (€M)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["gray_text"], size=11)),
        hovermode="x unified",
    )
    return fig


def player_overall_history(history_df, player_name):
    fig = _base_fig()
    if history_df.empty:
        return fig
    if "potential" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df["season_year"], y=history_df["potential"],
            mode="lines", name="Potencial",
            line=dict(color=COLORS["gray_line"], width=1, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=history_df["season_year"], y=history_df["overall"],
        mode="lines+markers", name="Overall",
        line=dict(color=COLORS["highlight"], width=3),
        marker=dict(size=9, color=COLORS["highlight"]),
    ))
    fig.update_layout(
        title=dict(text=f"Evolución Overall — {player_name}", font=dict(size=16)),
        xaxis_title="Temporada", yaxis_title="Valoración", yaxis_range=[50, 100],
    )
    return fig


# ── Gráfico 2: Comparación ────────────────────────────────────────
def radar_comparison(players: list, categories: list) -> go.Figure:
    """
    El jugador con MAYOR promedio de stats  DORADO brillante y opaco fuerte.
    Los demás  gris y muy transparentes.
    """
    if not players:
        return _empty_fig("Selecciona jugadores para comparar")

    promedios = [sum(p["values"]) / max(len(p["values"]), 1) for p in players]
    mejor_idx = promedios.index(max(promedios))

    fig = go.Figure()
    for i, player in enumerate(players[:3]):
        es_mejor = (i == mejor_idx)
        vals = player["values"] + [player["values"][0]]
        cats = [_es(c) for c in categories] + [_es(categories[0])]

        if es_mejor:
            color = COLORS["highlight"]
            fill_opacity = 0.30
            line_width = 3
            opacity = 1.0
            nombre = player["name"] + " "
        else:
            color = COLORS["gray_line"]
            fill_opacity = 0.05
            line_width = 1
            opacity = 0.4
            nombre = player["name"]

        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself",
            name=nombre,
            line=dict(color=color, width=line_width),
            fillcolor=f"rgba({r},{g},{b},{fill_opacity})",
            opacity=opacity,
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        polar=dict(
            bgcolor=COLORS["dark"],
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.05)",
                            tickfont=dict(color=COLORS["gray_text"], size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                             tickfont=dict(color=COLORS["text"], size=11)),
        ),
        title=dict(text="Perfil Técnico — Mejor jugador en dorado ", font=dict(size=16)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
    )
    return fig


def salary_vs_performance(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df = df[(df["wage_eur"] > 0) & (df["overall"] > 0)].head(500)
    med_wage = df["wage_eur"].median()
    med_overall = df["overall"].median()
    df["quadrant"] = "Normal"
    df.loc[(df["wage_eur"] > med_wage) & (df["overall"] < med_overall), "quadrant"] = "Supersueldo"
    df.loc[(df["wage_eur"] < med_wage) & (df["overall"] > med_overall), "quadrant"] = "Infracoste"
    color_map = {"Normal": COLORS["gray_line"], "Supersueldo": COLORS["overvalued"], "Infracoste": COLORS["highlight"]}
    fig = px.scatter(df, x="wage_eur", y="overall", color="quadrant",
                     color_discrete_map=color_map,
                     hover_data=["short_name", "club_name"],
                     size="value_eur", size_max=25,
                     labels={"wage_eur": "Salario Semanal (€)", "overall": "Valoración General"})
    fig.add_vline(x=med_wage, line_dash="dot", line_color=COLORS["gray_text"], opacity=0.3)
    fig.add_hline(y=med_overall, line_dash="dot", line_color=COLORS["gray_text"], opacity=0.3)
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text="Salario vs Rendimiento — Infracoste en dorado", font=dict(size=16)))
    return fig


def bar_skill_comparison(players_data: list, skill_cols: list) -> go.Figure:
    """
    En CADA habilidad: la barra más alta = DORADO con número encima.
    Las demás barras = GRIS opaco (40%).
    Así se ve de un vistazo quién gana en cada estadística.
    """
    if not players_data:
        return _empty_fig("Selecciona jugadores para comparar")

    fig = _base_fig()

    # Calcular valores de todos los jugadores
    all_vals = []
    for p in players_data[:3]:
        vals = [float(p.get(c, 0) or 0) for c in skill_cols]
        all_vals.append(vals)

    # Para cada skill, quién tiene el mayor valor
    ganadores = []
    for col_idx in range(len(skill_cols)):
        vals_col = [all_vals[j][col_idx] for j in range(len(all_vals))]
        ganadores.append(vals_col.index(max(vals_col)))

    for i, p in enumerate(players_data[:3]):
        vals = all_vals[i]

        # Color de cada barra: dorado si gana, gris si no
        bar_colors = [
            COLORS["highlight"] if ganadores[col_idx] == i else COLORS["gray_line"]
            for col_idx in range(len(skill_cols))
        ]
        # Opacidad: 1.0 si gana, 0.35 si no
        bar_opacity = [
            1.0 if ganadores[col_idx] == i else 0.35
            for col_idx in range(len(skill_cols))
        ]
        # Texto: mostrar valor solo si gana
        textos = [
            f"{vals[col_idx]:.0f}" if ganadores[col_idx] == i else ""
            for col_idx in range(len(skill_cols))
        ]

        fig.add_trace(go.Bar(
            name=p["name"],
            x=skill_cols,
            y=vals,
            marker=dict(
                color=bar_colors,
                opacity=bar_opacity,
            ),
            text=textos,
            textposition="outside",
            textfont=dict(color=COLORS["highlight"], size=12, family="Barlow, Arial"),
        ))

    labels = [_es(c) for c in skill_cols]
    fig.update_xaxes(tickvals=skill_cols, ticktext=labels, tickangle=-30)
    fig.update_layout(
        barmode="group",
        title=dict(text="Habilidades — Ganador de cada stat en dorado", font=dict(size=16)),
        yaxis_range=[0, 115],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"], size=11)),
    )
    return fig


# ── Gráfico 3: Joyas ocultas ──────────────────────────────────────
def undervalued_scatter(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return _empty_fig("No se encontraron jugadores en el rango del umbral seleccionado")

    fig = _base_fig()
    max_val = max(df["value_eur"].max(), df["predicted_value"].max()) / 1e6

    # Línea de precio justo (sin aparecer en leyenda)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        name="Precio Justo",
        showlegend=True,
        line=dict(color=COLORS["gray_text"], dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    # Puntos de jugadores con colorbar separado visualmente
    fig.add_trace(go.Scatter(
        x=df["value_eur"] / 1e6, y=df["predicted_value"] / 1e6,
        mode="markers",
        showlegend=False,
        marker=dict(
            color=df["undervalued_pct"],
            colorscale=[[0.0, COLORS["gray_line"]], [0.5, COLORS["accent"]], [1.0, COLORS["highlight"]]],
            size=10,
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
            colorbar=dict(
                title=dict(text="% Infravalorado", side="right",
                           font=dict(color=COLORS["muted"], size=11)),
                tickfont=dict(color=COLORS["gray_text"], size=10),
                thickness=14,
                len=0.7,
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                bordercolor=COLORS["border2"],
                borderwidth=1,
            ),
            showscale=True,
        ),
        text=df["short_name"],
        customdata=df[["club_name", "league_name", "overall", "undervalued_pct"]],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Club: %{customdata[0]}<br>"
            "Liga: %{customdata[1]}<br>"
            "Overall: %{customdata[2]}<br>"
            "Valor Real: €%{x:.1f}M<br>"
            "Valor Predicho: €%{y:.1f}M<br>"
            "<b>Infravalorado: +%{customdata[3]:.0f}%</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Valor Real vs Valor Predicho", font=dict(size=16)),
        xaxis_title="Valor Real (€M)",
        yaxis_title="Valor Predicho (€M)",
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border2"],
            borderwidth=1,
            font=dict(color=COLORS["text"], size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        margin=dict(l=55, r=90, t=60, b=50),
    )
    return fig


def undervalued_bar_top20(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return _empty_fig("No se encontraron jugadores en el rango del umbral seleccionado")

    top = df.head(20).copy()
    top["label"] = top["short_name"] + " (" + top["league_name"].str[:15] + ")"

    # El #1 en dorado brillante, los demás en gris con opacidad decreciente
    n = len(top)
    colors = [COLORS["highlight"]] + [COLORS["gray_line"]] * (n - 1)
    opacities = [1.0] + [max(0.3, 0.6 - i * 0.02) for i in range(1, n)]

    fig = _base_fig()
    fig.add_trace(go.Bar(
        y=top["label"], x=top["undervalued_pct"],
        orientation="h",
        marker=dict(color=colors, opacity=opacities),
        text=top["undervalued_pct"].apply(lambda v: f"+{v:.0f}%"),
        textposition="outside",
        textfont=dict(
            color=[COLORS["highlight"] if i == 0 else COLORS["gray_text"] for i in range(n)],
            size=11,
        ),
        hovertemplate="<b>%{y}</b><br>Infravalorado: %{x:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 20 Joyas — Mejor oportunidad en dorado", font=dict(size=16)),
        xaxis_title="% Infravalorado vs Modelo",
        yaxis=dict(autorange="reversed"),
        height=600,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    if importance_df is None or importance_df.empty:
        return _empty_fig("Sin datos de importancia")

    # El factor más importante en dorado, los demás en gris
    colors = [COLORS["highlight"]] + [COLORS["gray_line"]] * (len(importance_df) - 1)

    fig = _base_fig()
    fig.add_trace(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"].apply(_es),
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Factores que más Influyen en el Precio", font=dict(size=16)),
        xaxis_title="Importancia XGBoost",
        yaxis=dict(autorange="reversed"),
        height=450,
    )
    return fig


def league_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return _empty_fig("Sin datos de ligas")

    top_leagues = df["league_name"].value_counts().head(10).index
    sub = df[df["league_name"].isin(top_leagues)].copy()
    best_league = sub.groupby("league_name")["wage_efficiency"].median().idxmax()
    color_map = {
        league: COLORS["highlight"] if league == best_league else COLORS["gray_line"]
        for league in top_leagues
    }
    fig = px.box(sub, x="league_name", y="wage_efficiency", color="league_name",
                 color_discrete_map=color_map,
                 labels={"league_name": "Liga", "wage_efficiency": "Overall / (Salario €k)"})
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Eficiencia Salarial por Liga — Mejor en dorado", font=dict(size=16)),
        showlegend=False, xaxis_tickangle=-35,
    )
    return fig


def player_skills_evolution(history_df: pd.DataFrame, player_name: str) -> go.Figure:
    SKILL_KEYS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    fig = _base_fig()
    if history_df.empty:
        fig.update_layout(title=dict(text=f"Evolución de Habilidades — {player_name}", font=dict(size=16)))
        return fig

    available = [s for s in SKILL_KEYS if s in history_df.columns]
    last = history_df.iloc[-1]
    best_skill = max(available, key=lambda s: float(last.get(s, 0) or 0))

    for skill in available:
        es_mejor = skill == best_skill
        fig.add_trace(go.Scatter(
            x=history_df["season_year"], y=history_df[skill],
            mode="lines+markers", name=_es(skill),
            line=dict(color=COLORS["highlight"] if es_mejor else COLORS["gray_line"],
                      width=3 if es_mejor else 1.5),
            marker=dict(size=8 if es_mejor else 4,
                        color=COLORS["highlight"] if es_mejor else COLORS["gray_line"]),
            opacity=1.0 if es_mejor else 0.4,
        ))

    fig.update_layout(
        title=dict(text=f"Evolución de Habilidades — {player_name}", font=dict(size=16)),
        xaxis_title="Temporada", yaxis_title="Valoración (0-100)", yaxis_range=[0, 100],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["gray_text"], size=10)),
        hovermode="x unified",
    )
    return fig
