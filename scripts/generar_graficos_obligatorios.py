"""
Script para generar los 3 gráficos obligatorios de la sección 1.5.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-29

Genera:
1. Perfil de concentración en t=0
2. Perfil al 50% del tiempo para estado estacionario
3. Perfil en estado estacionario

Los gráficos se guardan en data/output/figures/
"""

import logging
import numpy as np
from pathlib import Path

from src.config.parametros import ParametrosMaestros
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.postproceso.visualizacion import (
    configurar_estilo_matplotlib,
    plot_perfil_t0,
    plot_perfil_50pct,
    plot_perfil_estado_estacionario,
    guardar_figura
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """
    Función principal para generar los 3 gráficos obligatorios.
    """
    logger.info("=" * 70)
    logger.info("GENERACIÓN DE GRÁFICOS OBLIGATORIOS - SECCIÓN 1.5")
    logger.info("=" * 70)

    # Configurar estilo de matplotlib
    configurar_estilo_matplotlib(dpi=150)  # 150 para desarrollo
    logger.info("✓ Estilo de matplotlib configurado")

    # Cargar parámetros
    logger.info("\n📋 Cargando parámetros del proyecto...")
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info("✓ Parámetros validados")

    # Crear solver
    logger.info("\n🔧 Inicializando solver Crank-Nicolson...")
    dt = 0.01  # 10 ms
    solver = CrankNicolsonSolver2D(params, dt=dt)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial()
    logger.info(f"✓ Solver inicializado (dt = {dt} s)")

    # Crear directorio de salida
    output_dir = Path("data/output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Directorio de salida: {output_dir}")

    # ========================================================================
    # GRÁFICO 1: PERFIL EN t=0
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 1: PERFIL DE CONCENTRACIÓN EN t=0")
    logger.info("=" * 70)

    fig1, ax1 = plot_perfil_t0(
        solver.malla.r,
        solver.malla.theta,
        solver.C,
        params
    )

    filepath1 = output_dir / "grafico_1_perfil_t0.png"
    guardar_figura(fig1, filepath1, dpi=300)
    logger.info(f"✓ Gráfico 1 guardado: {filepath1}")

    # ========================================================================
    # GRÁFICO 2: PERFIL AL 50% DEL TIEMPO
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 2: PERFIL AL 50% HACIA ESTADO ESTACIONARIO")
    logger.info("=" * 70)

    # Ejecutar hasta aproximadamente 50% del tiempo de estado estacionario
    # Estimación: ~50-100 segundos para acercarse a SS
    # 50% = ~25-50 segundos
    n_pasos_50pct = int(50 / dt)  # 50 s / 0.01 s = 5000 pasos
    logger.info(f"Ejecutando {n_pasos_50pct} pasos ({n_pasos_50pct * dt:.1f} s)...")

    for i in range(n_pasos_50pct):
        solver.paso_temporal()
        if (i + 1) % 500 == 0:
            logger.info(f"  Paso {i+1}/{n_pasos_50pct} (t = {solver.t:.2f} s)")

    logger.info(f"✓ Simulación completada hasta t = {solver.t:.2f} s")

    fig2, (ax2_1, ax2_2) = plot_perfil_50pct(
        solver.malla.r,
        solver.malla.theta,
        solver.C,
        solver.t,
        params
    )

    filepath2 = output_dir / "grafico_2_perfil_50pct.png"
    guardar_figura(fig2, filepath2, dpi=300)
    logger.info(f"✓ Gráfico 2 guardado: {filepath2}")

    # ========================================================================
    # GRÁFICO 3: PERFIL EN ESTADO ESTACIONARIO
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GRÁFICO 3: PERFIL EN ESTADO ESTACIONARIO")
    logger.info("=" * 70)

    # Continuar hasta alcanzar estado estacionario (o máximo 100 s total)
    t_max = 100.0  # s
    n_pasos_ss = int((t_max - solver.t) / dt)
    logger.info(f"Ejecutando {n_pasos_ss} pasos adicionales...")
    logger.info(f"Tiempo objetivo: {t_max} s")

    # Ejecutar con detección de convergencia
    solver.ejecutar(
        n_pasos=n_pasos_ss,
        check_convergencia=True,
        tol_convergencia=1e-5,
        cada_n_pasos=100
    )

    logger.info(f"✓ Simulación finalizada en t = {solver.t:.2f} s")
    logger.info(f"✓ Total de iteraciones: {solver.n_iter}")

    fig3, (ax3_1, ax3_2, ax3_3) = plot_perfil_estado_estacionario(
        solver.malla.r,
        solver.malla.theta,
        solver.C,
        solver.t,
        params
    )

    filepath3 = output_dir / "grafico_3_perfil_estado_estacionario.png"
    guardar_figura(fig3, filepath3, dpi=300)
    logger.info(f"✓ Gráfico 3 guardado: {filepath3}")

    # ========================================================================
    # RESUMEN
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN DE GENERACIÓN DE GRÁFICOS")
    logger.info("=" * 70)
    logger.info(f"✓ 3 gráficos obligatorios generados")
    logger.info(f"✓ Resolución: 300 DPI (alta calidad para reporte)")
    logger.info(f"✓ Directorio: {output_dir}")
    logger.info(f"✓ Tiempo final de simulación: {solver.t:.2f} s")
    logger.info(f"✓ Concentración máxima: {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ Concentración promedio: {np.mean(solver.C):.6e} mol/m³")
    logger.info("")
    logger.info("📊 Archivos generados:")
    logger.info(f"  1. {filepath1.name}")
    logger.info(f"  2. {filepath2.name}")
    logger.info(f"  3. {filepath3.name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

