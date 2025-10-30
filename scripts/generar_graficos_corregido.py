"""
Script CORREGIDO para generar los 3 gráficos obligatorios.

CORRECCIÓN: Usa dt más pequeño para evitar inestabilidad numérica.

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
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN DE GRÁFICOS - VERSIÓN CORREGIDA")
    logger.info("=" * 70)

    configurar_estilo_matplotlib(dpi=150)
    params = ParametrosMaestros()
    params.validar_todo()

    # ========================================================================
    # CALCULAR dt SEGURO
    # ========================================================================
    dr = params.mallado.dr
    dtheta = params.mallado.dtheta
    D_eff = params.difusion.D_eff

    dt_critico = calcular_dt_critico_euler(dr, dtheta, D_eff)
    
    # Usar dt = 2 × dt_crítico para Fo ≈ 1 (seguro para CN)
    dt_seguro = 2 * dt_critico
    
    logger.info(f"\n📊 ANÁLISIS DE ESTABILIDAD:")
    logger.info(f"  dt_crítico (Euler) = {dt_critico:.6f} s")
    logger.info(f"  dt_seguro (CN, Fo≈1) = {dt_seguro:.6f} s")
    logger.info(f"  Usando dt = {dt_seguro:.6f} s")

    # Crear solver
    solver = CrankNicolsonSolver2D(params, dt=dt_seguro)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial()

    output_dir = Path("data/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # GRÁFICO 1: t=0
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 1: PERFIL EN t=0")
    logger.info("=" * 70)

    fig1, ax1 = plot_perfil_t0(solver.malla.r, solver.malla.theta, solver.C, params)
    filepath1 = output_dir / "grafico_1_perfil_t0_corregido.png"
    guardar_figura(fig1, filepath1, dpi=300)
    logger.info(f"✓ Gráfico 1 guardado")

    # ========================================================================
    # GRÁFICO 2: 50% del tiempo (estimado: 25-50s)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 2: PERFIL AL 50% (ejecutando hasta t=25s)")
    logger.info("=" * 70)

    t_target_50pct = 25.0  # s
    n_pasos_50pct = int(t_target_50pct / dt_seguro)
    logger.info(f"Ejecutando {n_pasos_50pct} pasos...")
    logger.info(f"NOTA: Esto puede tardar varios minutos con dt={dt_seguro:.6f}s")

    for i in range(n_pasos_50pct):
        solver.paso_temporal()
        
        # Log cada 1000 pasos
        if (i + 1) % 1000 == 0:
            pct = 100 * (i + 1) / n_pasos_50pct
            logger.info(f"  Progreso: {pct:.1f}% (t = {solver.t:.2f} s)")
            
            # Verificar que no haya NaN
            if np.any(np.isnan(solver.C)):
                logger.error("❌ NaN detectado! Abortando.")
                raise RuntimeError("Inestabilidad numérica detectada")
    
    logger.info(f"✓ Simulación completada: t = {solver.t:.2f} s")

    fig2, (ax2_1, ax2_2) = plot_perfil_50pct(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath2 = output_dir / "grafico_2_perfil_50pct_corregido.png"
    guardar_figura(fig2, filepath2, dpi=300)
    logger.info(f"✓ Gráfico 2 guardado")

    # ========================================================================
    # GRÁFICO 3: Estado Estacionario (ejecutar hasta t=100s)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 3: ESTADO ESTACIONARIO (ejecutando hasta t=100s)")
    logger.info("=" * 70)

    t_target_ss = 100.0  # s
    n_pasos_adicionales = int((t_target_ss - solver.t) / dt_seguro)
    logger.info(f"Ejecutando {n_pasos_adicionales} pasos adicionales...")

    for i in range(n_pasos_adicionales):
        solver.paso_temporal()
        
        if (i + 1) % 1000 == 0:
            pct = 100 * (i + 1) / n_pasos_adicionales
            logger.info(f"  Progreso: {pct:.1f}% (t = {solver.t:.2f} s)")
            
            if np.any(np.isnan(solver.C)):
                logger.error("❌ NaN detectado! Abortando.")
                raise RuntimeError("Inestabilidad numérica detectada")
    
    logger.info(f"✓ Simulación finalizada: t = {solver.t:.2f} s")

    fig3, (ax3_1, ax3_2, ax3_3) = plot_perfil_estado_estacionario(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath3 = output_dir / "grafico_3_perfil_ss_corregido.png"
    guardar_figura(fig3, filepath3, dpi=300)
    logger.info(f"✓ Gráfico 3 guardado")

    # ========================================================================
    # RESUMEN
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info(f"✓ 3 gráficos generados exitosamente")
    logger.info(f"✓ dt usado: {dt_seguro:.6f} s (seguro)")
    logger.info(f"✓ Tiempo final: {solver.t:.2f} s")
    logger.info(f"✓ Iteraciones totales: {solver.n_iter}")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_mean = {np.mean(solver.C):.6e} mol/m³")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

