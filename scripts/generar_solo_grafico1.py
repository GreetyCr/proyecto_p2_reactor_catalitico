"""
Script para generar SOLO el gráfico 1 (t=0) que funciona.

Proyecto Personal 2 - Fenómenos de Transferencia
"""

import logging
from pathlib import Path

from src.config.parametros import ParametrosMaestros
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.postproceso.visualizacion import (
    configurar_estilo_matplotlib,
    plot_perfil_t0,
    guardar_figura
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Generando Gráfico 1: Perfil en t=0")
    
    configurar_estilo_matplotlib(dpi=150)
    params = ParametrosMaestros()
    params.validar_todo()
    
    solver = CrankNicolsonSolver2D(params, dt=0.001)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    
    output_dir = Path("data/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig1, ax1 = plot_perfil_t0(solver.malla.r, solver.malla.theta, solver.C, params)
    filepath1 = output_dir / "grafico_1_perfil_t0.png"
    guardar_figura(fig1, filepath1, dpi=300)
    
    logger.info(f"✓ Gráfico 1 guardado: {filepath1}")


if __name__ == "__main__":
    main()

