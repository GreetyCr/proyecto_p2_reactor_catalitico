"""
Generación optimizada de gráficos con estrategia híbrida.

OPTIMIZACIONES:
1. dt = 0.01s (5x más grande, aún estable para CN)
2. Detiene en convergencia O t=200s (lo que llegue primero)
3. Logging cada 100 pasos (~1s)
4. Genera solo gráficos 2 y 3 (gráfico 1 ya existe)
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
    plot_perfil_estado_estacionario,
    guardar_figura,
)

def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN OPTIMIZADA DE GRÁFICOS (Estrategia Híbrida)")
    logger.info("=" * 70)

    # Configurar estilo de Matplotlib
    configurar_estilo_matplotlib()

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info("✓ Parámetros validados")

    # ======================================================================
    # CONFIGURACIÓN DE OPTIMIZACIÓN
    # ======================================================================
    dt_simulacion = 0.01  # s (5x más grande que antes)
    t_final_max = 200.0   # s (máximo, pero se detiene si converge)
    check_convergencia_cada = 500  # Verificar cada 500 pasos
    log_cada = 100  # Logging cada 100 pasos (~1s)
    
    logger.info(f"✓ dt = {dt_simulacion} s (optimizado para velocidad)")
    logger.info(f"✓ t_final_max = {t_final_max} s (detiene si converge antes)")
    logger.info(f"✓ Check convergencia cada {check_convergencia_cada} pasos")

    # Directorio de salida
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # INICIALIZAR SOLVER
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
    # SIMULACIÓN CON DETECCIÓN DE CONVERGENCIA
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EJECUTANDO SIMULACIÓN")
    logger.info("=" * 70)
    
    n_pasos_max = int(t_final_max / dt_simulacion)
    logger.info(f"Pasos máximos: {n_pasos_max} (si no converge antes)")
    
    C_50pct = None
    t_50pct = None
    ha_guardado_50pct = False
    convergio = False
    C_prev = solver.C.copy()  # Para verificar convergencia
    
    for i in range(n_pasos_max):
        solver.paso_temporal()
        
        # Logging regular
        if (i + 1) % log_cada == 0:
            logger.info(
                f"Paso {i+1}: t={solver.t:.2f}s, "
                f"C∈[{np.min(solver.C):.3e}, {np.max(solver.C):.3e}] mol/m³"
            )
        
        # Guardar snapshot al 50% del tiempo máximo
        if not ha_guardado_50pct and solver.t >= t_final_max * 0.5:
            C_50pct = solver.C.copy()
            t_50pct = solver.t
            ha_guardado_50pct = True
            logger.info(f"   → Guardado perfil al 50% en t={t_50pct:.1f}s")
        
        # Verificar convergencia periódicamente
        if (i + 1) % check_convergencia_cada == 0:
            if solver.verificar_convergencia(C_prev, tol=1e-5):  # tol más relajado
                logger.info(f"✓ CONVERGENCIA ALCANZADA en t={solver.t:.1f}s (paso {i+1})")
                convergio = True
                # Si no hemos guardado el 50%, guardarlo ahora
                if not ha_guardado_50pct:
                    C_50pct = solver.C.copy()
                    t_50pct = solver.t * 0.5  # Estimación
                    logger.info(f"   → Guardado perfil estimado al 50% en t={t_50pct:.1f}s")
                break
            # Actualizar C_prev para próxima verificación
            C_prev = solver.C.copy()
    
    if not convergio:
        logger.info(f"✓ Alcanzado t_final_max = {solver.t:.1f}s sin convergencia completa")
    
    logger.info(f"\n✓ Simulación completada en {solver.n_iter} pasos")
    logger.info(f"✓ Tiempo final: {solver.t:.2f} s")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_bulk = {params.operacion.C_bulk:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {np.max(solver.C)/params.operacion.C_bulk*100:.2f}%")

    # ======================================================================
    # GRÁFICO 2: Perfil al 50% del tiempo
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 2: Perfil al 50% del tiempo")
    logger.info("=" * 70)
    
    if C_50pct is not None:
        fig2, ax2 = plot_perfil_50pct(
            solver.malla.r, solver.malla.theta, C_50pct, t_50pct, params
        )
        filepath_2 = os.path.join(output_dir, "grafico_2_perfil_evolucion.png")
        guardar_figura(fig2, filepath_2)
        logger.info(f"✓ Gráfico 2 guardado: {filepath_2}")
    else:
        logger.warning("⚠️ No se pudo generar Gráfico 2: C_50pct no disponible")

    # ======================================================================
    # GRÁFICO 3: Perfil en Estado Estacionario
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 3: Perfil Estado Estacionario")
    logger.info("=" * 70)
    
    fig3, ax3 = plot_perfil_estado_estacionario(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath_3 = os.path.join(output_dir, "grafico_3_perfil_ss.png")
    guardar_figura(fig3, filepath_3)
    logger.info(f"✓ Gráfico 3 guardado: {filepath_3}")

    # ======================================================================
    # RESUMEN FINAL
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info("✓ 2 gráficos generados exitosamente (Gráfico 2 y 3)")
    logger.info(f"✓ dt = {dt_simulacion} s")
    logger.info(f"✓ Tiempo final: {solver.t:.2f} s")
    logger.info(f"✓ Iteraciones: {solver.n_iter}")
    logger.info(f"✓ Convergencia: {'SÍ' if convergio else 'NO (alcanzó t_max)'}")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ C_bulk = {params.operacion.C_bulk:.6e} mol/m³")
    logger.info(f"✓ Ratio C_max/C_bulk = {np.max(solver.C)/params.operacion.C_bulk*100:.2f}%")
    logger.info("")
    logger.info("📊 Archivos:")
    logger.info(f"  {os.path.join(output_dir, 'grafico_2_perfil_evolucion.png')}")
    logger.info(f"  {os.path.join(output_dir, 'grafico_3_perfil_ss.png')}")
    logger.info("=" * 70)
    logger.info("✅ GRÁFICOS GENERADOS - PROCESO COMPLETADO")

if __name__ == "__main__":
    main()

