"""
Script rápido para generar los 3 gráficos obligatorios.

Usa dt más grande para acelerar simulación (menor precisión pero más rápido).
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configuración básica del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.parametros import ParametrosMaestros
from src.geometria.mallado import MallaPolar2D
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.postproceso.visualizacion import (
    configurar_estilo_matplotlib,
    plot_perfil_t0,
    plot_perfil_50pct,
    plot_perfil_estado_estacionario,
    guardar_figura,
)

def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN RÁPIDA DE GRÁFICOS (dt más grande)")
    logger.info("=" * 70)

    # Configurar estilo
    configurar_estilo_matplotlib()

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info(f"✓ Parámetros validados")

    # Usar dt más grande para acelerar
    dt = 0.002  # 2ms (en lugar de 0.534ms)
    logger.info(f"✓ dt = {dt} s (optimizado para velocidad)")

    # Inicializar solver
    solver = CrankNicolsonSolver2D(params, dt=dt)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial()
    logger.info(f"✓ Solver inicializado")

    # Directorio de salida
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    # GRÁFICO 1: t=0
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRÁFICO 1: Perfil en t=0")
    logger.info("=" * 70)
    fig1, ax1 = plot_perfil_t0(
        solver.malla.r, solver.malla.theta, solver.C, params
    )
    guardar_figura(fig1, os.path.join(output_dir, "grafico_1_perfil_t0.png"))
    logger.info("✓ Gráfico 1 guardado")
    plt.close(fig1)

    # GRÁFICO 2: 50% (t=15s)
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRÁFICO 2: Perfil al 50% (t=15s)")
    logger.info("=" * 70)
    
    t_50pct = 15.0  # s
    n_pasos_50 = int(t_50pct / dt)
    logger.info(f"Ejecutando {n_pasos_50} pasos hasta t={t_50pct}s...")
    
    for i in range(n_pasos_50):
        solver.paso_temporal()
        if (i + 1) % 500 == 0:
            logger.info(f"  Paso {i+1}: t={solver.t:.2f}s, C∈[{np.min(solver.C):.3e}, {np.max(solver.C):.3e}]")
    
    logger.info(f"✓ t={solver.t:.2f}s alcanzado")
    
    fig2, (ax2_1, ax2_2) = plot_perfil_50pct(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    guardar_figura(fig2, os.path.join(output_dir, "grafico_2_perfil_evolucion.png"))
    logger.info("✓ Gráfico 2 guardado")
    plt.close(fig2)

    # GRÁFICO 3: Estado estacionario (t=30s)
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRÁFICO 3: Estado Estacionario (t=30s)")
    logger.info("=" * 70)
    
    t_final = 30.0  # s
    n_pasos_final = int((t_final - solver.t) / dt)
    logger.info(f"Ejecutando {n_pasos_final} pasos adicionales hasta t={t_final}s...")
    
    for i in range(n_pasos_final):
        solver.paso_temporal()
        if (i + 1) % 500 == 0:
            logger.info(f"  Paso {i+1}: t={solver.t:.2f}s, C∈[{np.min(solver.C):.3e}, {np.max(solver.C):.3e}]")
    
    logger.info(f"✓ t={solver.t:.2f}s alcanzado")
    
    fig3, (ax3_1, ax3_2, ax3_3) = plot_perfil_estado_estacionario(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    guardar_figura(fig3, os.path.join(output_dir, "grafico_3_perfil_ss.png"))
    logger.info("✓ Gráfico 3 guardado")
    plt.close(fig3)

    # Resumen
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info(f"✓ 3 gráficos generados exitosamente")
    logger.info(f"✓ dt = {dt} s")
    logger.info(f"✓ Tiempo final: {solver.t:.2f} s")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_bulk = {params.operacion.C_bulk:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {100 * np.max(solver.C) / params.operacion.C_bulk:.2f}%")
    logger.info("")
    logger.info("📊 Archivos:")
    logger.info(f"  {os.path.join(output_dir, 'grafico_1_perfil_t0.png')}")
    logger.info(f"  {os.path.join(output_dir, 'grafico_2_perfil_evolucion.png')}")
    logger.info(f"  {os.path.join(output_dir, 'grafico_3_perfil_ss.png')}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

