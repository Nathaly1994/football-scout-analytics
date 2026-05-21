"""
FootballScout Analytics Platform
Punto de entrada principal de la aplicación.
"""

import sys
import os

# Añadir el directorio raíz al path para importaciones
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import FootballScoutApp


def main():
    """Inicializa y lanza la plataforma analítica."""
    app = FootballScoutApp()
    app.run()


if __name__ == "__main__":
    main()
