# ⚽ Football Scout Analytics

Plataforma analítica de scouting futbolístico basada en **Python + Dash + XGBoost**.  
Permite a directores deportivos y jefes de scouting tomar decisiones basadas en datos, no en intuición.

---

## 📁 Estructura del Proyecto

```
football_scout/
│
├── main.py                    # Punto de entrada
├── requirements.txt           # Dependencias
├── run.bat                    # Lanzador Windows
├── run.sh                     # Lanzador macOS/Linux
│
├── data/
│   └── FIFA_15-21.xlsx        # Dataset FIFA (15 → 21)
│
├── models/
│   ├── __init__.py
│   └── valuation_model.py     # Modelo XGBoost de valoración
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Carga y limpieza de datos
│   └── charts.py              # Visualizaciones Plotly
│
└── ui/
    ├── __init__.py
    └── app.py                 # App Dash (layout + callbacks)
```

---

## 🚀 Instalación y Ejecución

### Opción A — Script automático (recomendado)

**Windows** (doble clic o desde CMD):
```cmd
run.bat
```

**macOS / Linux**:
```bash
chmod +x run.sh
./run.sh
```

### Opción B — Manual desde Símbolo del Sistema

```cmd
REM 1. Crear entorno virtual
python -m venv venv

REM 2. Activar entorno (Windows)
venv\Scripts\activate

REM 3. Instalar dependencias
pip install -r requirements.txt

REM 4. Lanzar la app
python main.py
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Abrir en el navegador
Una vez arrancado, abre: **http://127.0.0.1:8050/**

---

## 🧪 Requisitos del Sistema

| Requisito | Versión |
|-----------|---------|
| Python    | 3.10+   |
| RAM       | 4 GB mín (8 GB recomendado) |
| SO        | Windows 10+, macOS 12+, Ubuntu 20.04+ |

---

## 🗂️ Pestañas de la Plataforma

### 📈 Pestaña 1 — Valoración & Proyección
- Busca cualquier jugador del dataset FIFA 15–21
- Visualiza su **trayectoria histórica de valor** con banda de confianza
- **Proyección a 1–5 años** basada en CAGR histórico + ajuste por edad/potencial
- KPIs en tiempo real: valor real vs predicho, veredicto (infravalorado / justo / sobrevalorado)
- Gráfico de **importancia de características XGBoost** (qué atributos pesan más en el precio)

### ⚔️ Pestaña 2 — Comparación de Jugadores
- Compara hasta **3 jugadores simultáneamente**
- **Radar chart** de perfil técnico multidimensional
- **Barras agrupadas** de habilidades detalladas
- Scatter de salario vs rendimiento: identifica **jugadores supersueldo** vs **jugadores infracoste**
- Tabla comparativa completa con valor, salario, liga y posición

### 💎 Pestaña 3 — Joyas Ocultas del Mercado
- Entrena el modelo XGBoost **en tiempo real** con la temporada seleccionada
- Muestra el **mapa de infravaloración** (valor real vs valor predicho)
- **Top 20 jugadores más infravalorados** con porcentaje de ganancia potencial
- Eficiencia salarial por liga (box plot)
- Tabla exportable con todas las joyas detectadas

---

## 🎛️ Filtros Globales

Todos los análisis responden en tiempo real a:
- **Temporada**: FIFA 15 → FIFA 21
- **Posición**: GK, CB, CM, ST, etc.
- **Liga**: Filtrar por competición específica
- **Rango de edad**: Deslizador 15–45 años
- **Valor máximo**: Acotar el mercado por precio

---

## 🧠 Modelo XGBoost — Detalles Técnicos

- **Target**: `log(value_eur)` — transformación logarítmica para reducir skewness
- **Features**: 35+ estadísticas técnicas (pace, shooting, vision, stamina, etc.) + edad, overall, potencial
- **Hiperparámetros**: 400 árboles, learning rate 0.05, max_depth 6, regularización L1+L2
- **Validación**: 80/20 train-test split, métricas MAE y R²
- **Similitud**: Distancia euclidiana normalizada (MinMaxScaler) para clonación táctica

---

## ⚠️ Notas

- El primer entrenamiento del modelo puede tardar **30–60 segundos** dependiendo del hardware.
- Los datos corresponden a **FIFA 15–21** (Sofifa). No representan datos reales de contratos.
- Para actualizar con datos nuevos, reemplaza `data/FIFA_15-21.xlsx` manteniendo el mismo formato de columnas.
