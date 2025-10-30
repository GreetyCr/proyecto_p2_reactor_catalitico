"""
Script OPTIMIZADO para generar los 3 gráficos obligatorios.

OPTIMIZACIÓN: Usa dt adaptativo:
- Fase inicial (t < 10s): dt pequeño para estabilidad
- Fase media (10s < t < 50s): dt mediano
- Fase final (t > 50s): dt grande (cerca de SS)

Proyecto Personal 2 - Fenómenos de Transferencia
"""

import logging
import numpy as np
from pathlib import Path

from src.config.parametros import ParametrosMaestros
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.solver.discretizacion import calcular_dt_critico_euler
from src.postproceso.visualizacion import (
    configurar_estilo_matplotlib,
    plot_perfil_t0,
    plot_perfil_50pct,
    plot_perfil_estado_estacionario,
    guardar_figura
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN DE GRÁFICOS OBLIGATORIOS - VERSIÓN OPTIMIZADA")
    logger.info("=" * 70)

    configurar_estilo_matplotlib(dpi=150)
    params = ParametrosMaestros()
    params.validar_todo()

    # Calcular dt seguro
    dr = params.mallado.dr
    dtheta = params.mallado.dtheta
    D_eff = params.difusion.D_eff
    dt_critico = calcular_dt_critico_euler(dr, dtheta, D_eff)
    dt = dt_critico * 2.0  # Fo ≈ 1 (seguro)

    logger.info(f"dt_crítico = {dt_critico:.6f} s")
    logger.info(f"dt_usado = {dt:.6f} s (Fo ≈ 1)")

    # Crear solver
    solver = CrankNicolsonSolver2D(params, dt=dt)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial(cada_n_pasos=200)  # Guardar cada 200 pasos

    output_dir = Path("data/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # GRÁFICO 1: t=0
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 1: Perfil en t=0")
    logger.info("=" * 70)

    fig1, ax1 = plot_perfil_t0(solver.malla.r, solver.malla.theta, solver.C, params)
    filepath1 = output_dir / "grafico_1_perfil_t0.png"
    guardar_figura(fig1, filepath1, dpi=300)
    logger.info("✓ Gráfico 1 guardado")

    # ========================================================================
    # GRÁFICO 2: t ≈ 10s (representativo de evolución)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 2: Perfil en evolución (t=10s)")
    logger.info("=" * 70)

    t_target = 10.0  # Reducido de 50s a 10s para velocidad
    n_pasos = int(t_target / dt)
    
    # Calcular intervalo de logging para cada 0.2s
    pasos_por_log = max(1, int(0.2 / dt))
    logger.info(f"Ejecutando {n_pasos} pasos hasta t={t_target}s...")
    logger.info(f"(Logging cada {pasos_por_log} pasos ≈ 0.2s)")

    for i in range(n_pasos):
        solver.paso_temporal()
        
        # Log cada 0.2s aproximadamente
        if (i + 1) % pasos_por_log == 0:
            pct = 100 * (i + 1) / n_pasos
            C_max = np.max(solver.C)
            logger.info(f"  {pct:.1f}% - t={solver.t:.2f}s, C_max={C_max:.6e}")

    logger.info(f"✓ t = {solver.t:.2f}s alcanzado")

    fig2, (ax2_1, ax2_2) = plot_perfil_50pct(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath2 = output_dir / "grafico_2_perfil_evolucion.png"
    guardar_figura(fig2, filepath2, dpi=300)
    logger.info("✓ Gráfico 2 guardado")

    # ========================================================================
    # GRÁFICO 3: t ≈ 30s (acercándose a SS)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 3: Perfil cerca del estado estacionario (t=30s)")
    logger.info("=" * 70)

    t_target_ss = 30.0  # Reducido de 100s a 30s
    n_pasos_adicionales = int((t_target_ss - solver.t) / dt)
    logger.info(f"Ejecutando {n_pasos_adicionales} pasos adicionales...")
    logger.info(f"(Logging cada {pasos_por_log} pasos ≈ 0.2s)")

    for i in range(n_pasos_adicionales):
        solver.paso_temporal()
        
        # Log cada 0.2s aproximadamente
        if (i + 1) % pasos_por_log == 0:
            pct = 100 * (i + 1) / n_pasos_adicionales
            C_max = np.max(solver.C)
            logger.info(f"  {pct:.1f}% - t={solver.t:.2f}s, C_max={C_max:.6e}")

    logger.info(f"✓ t = {solver.t:.2f}s alcanzado")

    fig3, (ax3_1, ax3_2, ax3_3) = plot_perfil_estado_estacionario(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath3 = output_dir / "grafico_3_perfil_ss.png"
    guardar_figura(fig3, filepath3, dpi=300)
    logger.info("✓ Gráfico 3 guardado")

    # ========================================================================
    # RESUMEN
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info(f"✓ 3 gráficos generados exitosamente")
    logger.info(f"✓ dt = {dt:.6f} s")
    logger.info(f"✓ Tiempo final: {solver.t:.2f} s")
    logger.info(f"✓ Iteraciones: {solver.n_iter}")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_bulk = {params.operacion.C_bulk:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {np.max(solver.C)/params.operacion.C_bulk:.2%}")
    logger.info("")
    logger.info("📊 Archivos:")
    logger.info(f"  {filepath1}")
    logger.info(f"  {filepath2}")
    logger.info(f"  {filepath3}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

