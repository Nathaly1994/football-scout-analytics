@echo off
echo ============================================================
echo   Football Scout Analytics - Instalador y Lanzador
echo ============================================================
echo.

REM Verificar Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python no encontrado. Descargalo en https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python encontrado.
echo.

REM Crear entorno virtual si no existe
IF NOT EXIST "venv\" (
    echo [1/3] Creando entorno virtual...
    python -m venv venv
    echo [OK] Entorno virtual creado.
) ELSE (
    echo [INFO] Entorno virtual ya existe.
)

echo.
echo [2/3] Activando entorno virtual e instalando dependencias...
call venv\Scripts\activate.bat

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo al instalar dependencias. Revisa requirements.txt
    pause
    exit /b 1
)

echo [OK] Dependencias instaladas correctamente.
echo.
echo [3/3] Iniciando Football Scout Analytics...
echo.
echo ============================================================
echo   Abre tu navegador en: http://127.0.0.1:8050/
echo   Presiona CTRL+C para detener la aplicacion
echo ============================================================
echo.

python main.py

pause
