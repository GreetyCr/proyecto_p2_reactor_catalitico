"""
Generación de gráficos obligatorios con simulación hasta t=1000s.

Solo genera gráficos 2 y 3 (asume que gráfico 1 ya existe).
Usa dt más grande para reducir iteraciones y tiempo de cómputo.
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
    plot_perfil_50pct,
    _dibujar_defecto_anillo,
    guardar_figura,
)
from src.solver.discretizacion import calcular_numero_fourier_dimensional
from mpl_toolkits.mplot3d import Axes3D


def plot_perfil_estado_estacionario_simplificado(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    tiempo: float,
    params,
    figsize=(16, 7)
):
    """
    Gráfico 3 simplificado: solo 2D polar + 3D (sin perfiles radiales).
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Vista 2D polar
    ax1 = fig.add_subplot(121, projection='polar')

    R_grid, THETA_grid = np.meshgrid(r, theta)
    
    # C_bulk puede ser float o CantidadDimensional
    C_bulk_val = params.operacion.C_bulk if isinstance(params.operacion.C_bulk, float) else params.operacion.C_bulk.valor
    
    levels = np.linspace(0, C_bulk_val, 25)
    contour1 = ax1.contourf(THETA_grid, R_grid, C.T, levels=levels, cmap='plasma')

    # Marcar defecto (anillo con arcos y líneas radiales)
    _dibujar_defecto_anillo(ax1, params, color='red', linewidth=2, label='Defecto')

    ax1.set_title(f'Vista 2D Polar\n(Estado Estacionario, t={tiempo:.1f}s)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)

    plt.colorbar(contour1, ax=ax1, pad=0.1, label='C [mol/m³]')

    # Subplot 2: Vista 3D
    ax2 = fig.add_subplot(122, projection='3d')

    # Convertir a coordenadas cartesianas
    X, Y = R_grid * np.cos(THETA_grid), R_grid * np.sin(THETA_grid)

    # Surface plot
    surf2 = ax2.plot_surface(
        X, Y, C.T, cmap='plasma', edgecolor='none', rstride=1, cstride=1, alpha=0.9
    )
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='C [mol/m³]')

    ax2.set_xlabel('X [m]', fontsize=10)
    ax2.set_ylabel('Y [m]', fontsize=10)
    ax2.set_zlabel('Concentración CO [mol/m³]', fontsize=10)
    ax2.set_title(f'Vista 3D Cartesiana\n(Estado Estacionario, t={tiempo:.1f}s)', fontsize=12)
    ax2.view_init(elev=25, azim=45)

    plt.suptitle(
        f'Concentración CO en Estado Estacionario (t={tiempo:.1f}s)', 
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    return fig, (ax1, ax2)


def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN DE GRÁFICOS 2 y 3 - SIMULACIÓN HASTA t=1000s")
    logger.info("=" * 70)

    # Configurar estilo de Matplotlib
    configurar_estilo_matplotlib()

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info("✓ Parámetros validados")

    # Configuración de simulación
    # Usar dt más grande para reducir iteraciones, pero mantener estabilidad
    dt_simulacion = 0.01  # s (5x más grande que 0.002s)
    t_final = 1000.0  # s
    
    # Calcular número de Fourier para verificar estabilidad
    Fo_r = calcular_numero_fourier_dimensional(
        dt_simulacion, params.mallado.dr, params.difusion.D_eff
    )
    logger.info(f"✓ dt = {dt_simulacion} s (Fo_r = {Fo_r.valor:.2f})")
    
    if Fo_r.valor > 10:
        logger.warning(f"⚠️  Fo_r = {Fo_r.valor:.2f} es alto, puede haber oscilaciones")
        logger.warning("   Crank-Nicolson sigue siendo estable, pero la precisión puede degradarse")
    
    # Directorio de salida
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # INICIALIZACIÓN DEL SOLVER
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("INICIALIZANDO SOLVER CRANK-NICOLSON 2D")
    logger.info("=" * 70)

    solver = CrankNicolsonSolver2D(params, dt=dt_simulacion)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial()

    logger.info("✓ Solver inicializado")

    # ======================================================================
    # SIMULACIÓN HASTA t=1000s
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SIMULACIÓN HASTA t=1000s")
    logger.info("=" * 70)

    n_pasos_total = int(t_final / dt_simulacion)
    n_pasos_50pct = n_pasos_total // 2
    
    logger.info(f"Total de pasos: {n_pasos_total}")
    logger.info(f"Paso al 50%: {n_pasos_50pct} (t={n_pasos_50pct * dt_simulacion:.1f}s)")
    logger.info(f"Progreso se registrará cada 500 pasos (~{500 * dt_simulacion:.0f}s)")
    logger.info("")
    logger.info("Iniciando simulación...")
    
    C_50pct = None
    t_50pct = None

    for i in range(n_pasos_total):
        solver.paso_temporal()

        # Logging cada 500 pasos
        if (i + 1) % 500 == 0:
            progreso = 100 * (i + 1) / n_pasos_total
            logger.info(
                f"  Paso {i+1:5d}/{n_pasos_total} ({progreso:5.1f}%): "
                f"t={solver.t:7.1f}s, "
                f"C∈[{np.min(solver.C):.3e}, {np.max(solver.C):.3e}] mol/m³"
            )
        
        # Guardar snapshot al 50%
        if (i + 1) == n_pasos_50pct:
            C_50pct = solver.C.copy()
            t_50pct = solver.t
            logger.info(f"   → Guardado perfil al 50% en t={t_50pct:.1f}s")
    
    logger.info(f"\n✓ Simulación completada hasta t = {solver.t:.1f}s")
    logger.info(f"✓ Iteraciones totales: {solver.n_iter}")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    
    # C_bulk puede ser float o CantidadDimensional
    C_bulk_val = params.operacion.C_bulk if isinstance(params.operacion.C_bulk, float) else params.operacion.C_bulk.valor
    logger.info(f"✓ C_bulk = {C_bulk_val:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {np.max(solver.C)/C_bulk_val*100:.2f}%")

    # ======================================================================
    # GRÁFICO 2: Perfil al 50% del tiempo (t=500s)
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 2: Perfil al 50% (t=500s)")
    logger.info("=" * 70)
    
    if C_50pct is not None:
        fig2, ax2 = plot_perfil_50pct(
            solver.malla.r, solver.malla.theta, C_50pct, t_50pct, params
        )
        output_path_2 = os.path.join(output_dir, "grafico_2_perfil_t500s.png")
        guardar_figura(fig2, output_path_2)
        logger.info(f"✓ Gráfico 2 guardado: {output_path_2}")
    else:
        logger.error("❌ No se pudo generar Gráfico 2: C_50pct no disponible")

    # ======================================================================
    # GRÁFICO 3: Estado Estacionario (t=1000s) - SIMPLIFICADO
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 3: Estado Estacionario (t=1000s)")
    logger.info("=" * 70)
    
    fig3, ax3 = plot_perfil_estado_estacionario_simplificado(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    output_path_3 = os.path.join(output_dir, "grafico_3_perfil_ss_t1000s.png")
    guardar_figura(fig3, output_path_3)
    logger.info(f"✓ Gráfico 3 guardado: {output_path_3}")

    # ======================================================================
    # RESUMEN FINAL
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 70)
    logger.info("✓ 2 gráficos generados exitosamente (Gráfico 2 y 3)")
    logger.info(f"✓ dt = {dt_simulacion} s")
    logger.info(f"✓ Tiempo final: {solver.t:.1f} s")
    logger.info(f"✓ Iteraciones: {solver.n_iter}")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_bulk = {C_bulk_val:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {np.max(solver.C)/C_bulk_val*100:.2f}%")
    logger.info("")
    logger.info("📊 Archivos generados:")
    logger.info(f"  {output_path_2}")
    logger.info(f"  {output_path_3}")
    logger.info("=" * 70)
    logger.info("✅ SIMULACIÓN COMPLETADA")

if __name__ == "__main__":
    main()

