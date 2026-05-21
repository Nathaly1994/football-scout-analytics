"""
Modelo de valoración de jugadores usando XGBoost.
Predice el valor de mercado (value_eur) basándose en estadísticas técnicas.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

from utils.data_loader import DataLoader, SKILL_COLS


class PlayerValuationModel:
    """
    Modelo XGBoost para estimar el valor intrínseco de un jugador.
    Permite detectar jugadores infravalorados (precio de mercado < valor predicho).
    """

    TARGET = "value_eur"

    def __init__(self):
        self._loader = DataLoader()
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []
        self._is_trained = False
        self._metrics: dict = {}

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, season: str | None = None) -> dict:
        """
        Entrena el modelo con los datos de la temporada indicada
        (o la última disponible si no se especifica).
        Devuelve métricas de rendimiento.
        """
        df = self._loader.load_latest() if season is None else self._loader.load_season(season)
        X, y = self._prepare(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators=150,      # era 400 → demasiado lento en callback
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb_model),
        ])

        self._pipeline.fit(X_train, y_train)
        self._is_trained = True

        # Métricas
        y_pred = self._pipeline.predict(X_test)
        self._metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return self._metrics

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Importancia de características
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Devuelve las N características más influyentes en el precio."""
        self._ensure_trained()
        xgb_step = self._pipeline.named_steps["xgb"]
        importances = xgb_step.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Detección de infravalorados
    # ------------------------------------------------------------------

    def find_undervalued(
        self,
        df: pd.DataFrame,
        threshold_ratio: float = 0.70,
        min_value: float = 500_000,
    ) -> pd.DataFrame:
        """
        Devuelve jugadores cuyo valor real de mercado es ≤ threshold_ratio
        del valor predicho por el modelo (ej. valen 30%+ más de lo que cuestan).

        Args:
            threshold_ratio: Máximo ratio valor_real/valor_predicho para considerarse infravalorado.
            min_value: Valor mínimo para filtrar jugadores irrelevantes.
        """
        self._ensure_trained()
        df = df.copy()
        df["predicted_value"] = self.predict(df)
        df["market_ratio"] = np.where(
            df["predicted_value"] > 0,
            df[self.TARGET] / df["predicted_value"],
            np.nan,
        )
        df["value_gap"] = df["predicted_value"] - df[self.TARGET]
        df["undervalued_pct"] = ((df["value_gap"] / df["predicted_value"]) * 100).round(1)

        mask = (
            (df["market_ratio"] <= threshold_ratio)
            & (df[self.TARGET] >= min_value)
            & (df["predicted_value"] > 0)
            & (df["value_gap"] > 0)
        )
        result = df[mask].sort_values("value_gap", ascending=False).reset_index(drop=True)

        return result

    # ------------------------------------------------------------------
    # Proyección de valor a N años
    # ------------------------------------------------------------------

    def project_value(
        self, player_id: int, years: int = 5
    ) -> pd.DataFrame:
        """
        Proyecta el valor de mercado de un jugador a lo largo de N años futuros,
        basándose en la tendencia histórica y el potencial de crecimiento.
        """
        history = self._loader.get_player_history(player_id)
        if history.empty:
            return pd.DataFrame()

        latest = history.iloc[-1]
        base_year = int(latest["season_year"])

        # Calcular tendencia usando las ultimas 2 temporadas (valor mas reciente)
        if len(history) >= 2:
            recent = history.tail(2)
            prev_val = float(recent["value_eur"].iloc[0])
            last_val = float(recent["value_eur"].iloc[-1])
            cagr = (last_val / prev_val) - 1 if prev_val > 0 else 0
            # Acotar CAGR entre -20% y +30%
            cagr = max(-0.20, min(0.30, cagr))
        else:
            cagr = 0.05  # crecimiento por defecto del 5%

        # Aplicar descuento por edad: jugadores > 30 deprecian más rápido
        age = int(latest.get("age", 25))
        age_factor = -0.02 if age > 30 else (0.01 if age < 23 else 0)
        adjusted_cagr = cagr + age_factor

        rows = []
        current_val = float(latest["value_eur"])
        current_overall = float(latest.get("overall", 70))
        potential = float(latest.get("potential", current_overall))

        for i in range(1, years + 1):
            proj_year = base_year + i
            proj_age = age + i
            # Valor proyectado
            proj_val = current_val * ((1 + adjusted_cagr) ** i)
            # Overall crece hasta el potencial en los primeros años, luego cae con la edad
            if proj_age <= 28:
                delta_overall = min((potential - current_overall) * 0.25, 2)
            elif proj_age <= 32:
                delta_overall = 0
            else:
                delta_overall = -1.5
            proj_overall = min(potential, current_overall + delta_overall * i)
            rows.append({
                "year": proj_year,
                "age": proj_age,
                "projected_value": max(0, proj_val),
                "projected_overall": round(proj_overall, 1),
                "growth_rate": adjusted_cagr * 100,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Clonación táctica (similitud euclidiana)
    # ------------------------------------------------------------------

    def find_similar_players(
        self,
        player_id: int,
        df: pd.DataFrame | None = None,
        n: int = 10,
        same_position: bool = True,
    ) -> pd.DataFrame:
        """
        Encuentra los N jugadores con perfil estadístico más similar
        al jugador objetivo usando distancia euclidiana normalizada.
        """
        if df is None:
            df = self._loader.load_latest()

        feat_cols = [c for c in SKILL_COLS if c in df.columns]
        target_row = df[df["sofifa_id"] == player_id]
        if target_row.empty:
            return pd.DataFrame()

        target = target_row[feat_cols].fillna(0).values

        if same_position:
            pos = target_row.iloc[0].get("primary_position", "")
            candidates = df[df["primary_position"] == pos].copy()
        else:
            candidates = df.copy()

        candidates = candidates[candidates["sofifa_id"] != player_id].copy()
        X = candidates[feat_cols].fillna(0).values

        # Normalizar por rango
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        all_data = np.vstack([target, X])
        all_scaled = scaler.fit_transform(all_data)
        target_scaled = all_scaled[0]
        X_scaled = all_scaled[1:]

        distances = np.sqrt(((X_scaled - target_scaled) ** 2).sum(axis=1))
        candidates["similarity_distance"] = distances
        candidates["similarity_score"] = (1 - distances / distances.max()) * 100

        return (
            candidates.nsmallest(n, "similarity_distance")
            [["sofifa_id", "short_name", "age", "nationality", "club_name",
              "league_name", "overall", "potential", "value_eur", "wage_eur",
              "primary_position", "similarity_score"]]
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepara X e y para entrenamiento."""
        avail = [c for c in SKILL_COLS if c in df.columns]
        extra = ["age", "overall", "potential", "international_reputation",
                 "skill_moves", "weak_foot"]
        extra = [c for c in extra if c in df.columns]
        self._feature_cols = avail + extra

        sub = df[self._feature_cols + [self.TARGET]].dropna(subset=[self.TARGET])
        sub = sub[sub[self.TARGET] > 0]
        X = sub[self._feature_cols].fillna(0)
        y = np.log1p(sub[self.TARGET])  # log-transform para reducir skewness
        return X, y

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice el valor de mercado (en €) para un DataFrame de jugadores."""
        self._ensure_trained()
        X = df[self._feature_cols].fillna(0)
        log_pred = self._pipeline.predict(X)
        return np.expm1(log_pred)  # revertir log-transform

    def predict_single(self, player_row: pd.Series) -> float:
        self._ensure_trained()
        X = player_row[self._feature_cols].fillna(0).values.reshape(1, -1)
        log_pred = self._pipeline.predict(X)[0]
        return float(np.expm1(log_pred))

    def _ensure_trained(self):
        if not self._is_trained:
            self.train()