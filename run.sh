#!/bin/bash
echo "============================================================"
echo "  Football Scout Analytics - Instalador y Lanzador"
echo "============================================================"
echo ""

# Verificar Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 no encontrado. Instálalo con:"
    echo "  macOS: brew install python3"
    echo "  Ubuntu: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
echo "[OK] Python3 encontrado: $(python3 --version)"
echo ""

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "[1/3] Creando entorno virtual..."
    python3 -m venv venv
    echo "[OK] Entorno virtual creado."
else
    echo "[INFO] Entorno virtual ya existe."
fi

echo ""
echo "[2/3] Activando entorno e instalando dependencias..."
source venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "[ERROR] Fallo al instalar dependencias."
    exit 1
fi

echo "[OK] Dependencias instaladas."
echo ""
echo "[3/3] Iniciando Football Scout Analytics..."
echo ""
echo "============================================================"
echo "  Abre tu navegador en: http://127.0.0.1:8050/"
echo "  Presiona CTRL+C para detener la aplicación"
echo "============================================================"
echo ""

python3 main.py
