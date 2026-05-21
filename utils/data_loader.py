"""
Módulo de carga y preprocesamiento de datos FIFA.
Responsable de leer el Excel y preparar el DataFrame unificado.
"""

import os
import pandas as pd
import numpy as np


# Columnas clave para análisis
SKILL_COLS = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_stamina", "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_positioning", "mentality_vision",
    "mentality_composure",
    "defending_standing_tackle", "defending_sliding_tackle",
]

META_COLS = [
    "sofifa_id", "short_name", "long_name", "age", "nationality",
    "club_name", "league_name", "league_rank", "overall", "potential",
    "value_eur", "wage_eur", "player_positions", "preferred_foot",
    "international_reputation", "skill_moves", "weak_foot",
]


class DataLoader:
    """Carga y preprocesa los datos de jugadores FIFA."""

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "FIFA_15-21.xlsx")

    def __init__(self):
        self._raw_cache: dict[str, pd.DataFrame] = {}
        self._unified_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Carga bruta por temporada
    # ------------------------------------------------------------------

    def load_season(self, season: str) -> pd.DataFrame:
        """Devuelve el DataFrame de una temporada específica (ej. 'FIFA 21')."""
        if season not in self._raw_cache:
            df = pd.read_excel(self.DATA_PATH, sheet_name=season, engine='calamine')
            df["season"] = season
            df["season_year"] = int(season.split()[-1])
            self._raw_cache[season] = self._clean(df)
        return self._raw_cache[season]

    def available_seasons(self) -> list[str]:
        """Lista de temporadas disponibles de forma ultra rápida."""
        with pd.ExcelFile(self.DATA_PATH, engine='calamine') as xls:
            return xls.sheet_names
        

    # ------------------------------------------------------------------
    # Dataset unificado multi-temporada
    # ------------------------------------------------------------------

    def load_all(self) -> pd.DataFrame:
        """Carga y unifica todas las temporadas en un único DataFrame."""
        if self._unified_cache is not None:
            return self._unified_cache

        frames = []
        for season in self.available_seasons():
            frames.append(self.load_season(season))

        df = pd.concat(frames, ignore_index=True)
        self._unified_cache = df
        return df

    def load_latest(self) -> pd.DataFrame:
        """Devuelve sólo los datos de la última temporada disponible."""
        seasons = self.available_seasons()
        return self.load_season(seasons[-1])

    # ------------------------------------------------------------------
    # Limpieza y enriquecimiento
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y enriquece el DataFrame de una temporada."""
        df = df.copy()

        # Convertir columnas de posición de rating (formato "67+2") a numérico
        pos_cols = ["ls","st","rs","lw","lf","cf","rf","rw","lam","cam","ram",
                    "lm","lcm","cm","rcm","rm","lwb","ldm","cdm","rdm","rwb",
                    "lb","lcb","cb","rcb","rb"]
        for col in pos_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.split("+").str[0], errors="coerce"
                )

        # Forzar tipos numéricos en skills
        for col in SKILL_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Valores monetarios: rellenar NaN con 0
        for col in ["value_eur", "wage_eur", "release_clause_eur"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Posición primaria
        if "player_positions" in df.columns:
            df["primary_position"] = (
                df["player_positions"]
                .astype(str)
                .str.split(",")
                .str[0]
                .str.strip()
            )

        # Ratio potencial / overall
        if "potential" in df.columns and "overall" in df.columns:
            df["growth_potential"] = df["potential"] - df["overall"]

        # Eficiencia salarial: overall por cada 1000€ de salario
        df["wage_efficiency"] = np.where(
            df["wage_eur"] > 0,
            df["overall"] / (df["wage_eur"] / 1_000),
            np.nan,
        )

        return df

    # ------------------------------------------------------------------
    # Helpers de consulta
    # ------------------------------------------------------------------

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Devuelve el historial de un jugador usando solo temporadas ya cargadas en caché.
        Evita cargar load_all() (68 MB × 7 hojas) que congela la UI.
        """
        frames = []
        for season_df in self._raw_cache.values():
            player_rows = season_df[season_df["sofifa_id"] == player_id]
            if not player_rows.empty:
                frames.append(player_rows)

        if not frames:
            # Ninguna temporada cacheada aún  usar solo la última (ya cargada por el buscador)
            df = self.load_latest()
            return df[df["sofifa_id"] == player_id].reset_index(drop=True)

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values("season_year").reset_index(drop=True)

    def search_players(self, query: str, season: str | None = None) -> pd.DataFrame:
        """Búsqueda de jugadores por nombre (parcial, insensible a mayúsculas)."""
        df = self.load_latest() if season is None else self.load_season(season)
        mask = (
            df["short_name"].str.contains(query, case=False, na=False)
            | df["long_name"].str.contains(query, case=False, na=False)
        )
        return df[mask].reset_index(drop=True)

    def get_feature_matrix(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Devuelve sólo las columnas de skills disponibles (sin NaN en >50% de filas)."""
        if df is None:
            df = self.load_latest()
        available = [c for c in SKILL_COLS if c in df.columns]
        sub = df[available].copy()
        # Eliminar columnas con >50% NaN
        thresh = len(sub) * 0.5
        sub = sub.dropna(axis=1, thresh=int(thresh))
        return sub
