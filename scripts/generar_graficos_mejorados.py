"""
Generación de versiones mejoradas de los gráficos 2 y 3.

Usa escalas de color ajustadas al rango real para resaltar el efecto del defecto.
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
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.postproceso.visualizacion import configurar_estilo_matplotlib
from src.postproceso.visualizacion_mejorada import (
    plot_perfil_50pct_mejorado,
    plot_perfil_ss_mejorado,
    guardar_figura
)

def main():
    logger.info("=" * 70)
    logger.info("GENERACIÓN DE GRÁFICOS MEJORADOS (Escalas Ajustadas)")
    logger.info("=" * 70)

    # Configurar estilo
    configurar_estilo_matplotlib()

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info("✓ Parámetros validados")

    # Directorio de salida
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # SIMULAR HASTA CONVERGENCIA
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SIMULACIÓN HASTA CONVERGENCIA")
    logger.info("=" * 70)
    
    dt_simulacion = 0.01  # s
    
    solver = CrankNicolsonSolver2D(params, dt=dt_simulacion)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    solver.habilitar_historial()
    
    logger.info(f"✓ Solver inicializado (dt={dt_simulacion}s)")
    logger.info("Simulando...")
    
    C_50pct = None
    t_50pct = None
    ha_guardado_50pct = False
    C_prev = solver.C.copy()
    
    for i in range(20000):  # Max 200s
        solver.paso_temporal()
        
        # Logging cada 500 pasos
        if (i + 1) % 500 == 0:
            logger.info(
                f"Paso {i+1}: t={solver.t:.1f}s, "
                f"C∈[{np.min(solver.C):.3e}, {np.max(solver.C):.3e}]"
            )
        
        # Guardar snapshot al 50% del tiempo estimado de convergencia (~70s)
        if not ha_guardado_50pct and solver.t >= 35.0:
            C_50pct = solver.C.copy()
            t_50pct = solver.t
            ha_guardado_50pct = True
            logger.info(f"   → Guardado perfil al 50% en t={t_50pct:.1f}s")
        
        # Verificar convergencia cada 500 pasos
        if (i + 1) % 500 == 0:
            if solver.verificar_convergencia(C_prev, tol=1e-5):
                logger.info(f"✓ CONVERGENCIA en t={solver.t:.1f}s (paso {i+1})")
                break
            C_prev = solver.C.copy()
    
    logger.info(f"✓ Simulación completada en {solver.n_iter} pasos")
    logger.info(f"✓ Tiempo final: {solver.t:.2f}s")
    
    # Calcular rango para resumen
    C_min_final = np.min(solver.C)
    C_max_final = np.max(solver.C)

    # ======================================================================
    # GRÁFICO 2 MEJORADO
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 2 MEJORADO")
    logger.info("=" * 70)
    
    if C_50pct is not None:
        fig2, ax2 = plot_perfil_50pct_mejorado(
            solver.malla.r, solver.malla.theta, C_50pct, t_50pct, params
        )
        filepath_2 = os.path.join(output_dir, "grafico_2_mejorado_escala_ajustada.png")
        guardar_figura(fig2, filepath_2)
        logger.info(f"✓ Gráfico 2 mejorado guardado")
    else:
        logger.warning("⚠️ No se pudo generar Gráfico 2 mejorado")

    # ======================================================================
    # GRÁFICO 3 MEJORADO
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO GRÁFICO 3 MEJORADO")
    logger.info("=" * 70)
    
    fig3, ax3 = plot_perfil_ss_mejorado(
        solver.malla.r, solver.malla.theta, solver.C, solver.t, params
    )
    filepath_3 = os.path.join(output_dir, "grafico_3_mejorado_escala_ajustada.png")
    guardar_figura(fig3, filepath_3)
    logger.info(f"✓ Gráfico 3 mejorado guardado")

    # ======================================================================
    # RESUMEN
    # ======================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info("✓ 2 gráficos mejorados generados con escalas ajustadas")
    logger.info(f"✓ Tiempo de convergencia: {solver.t:.1f}s")
    logger.info(f"✓ C_max = {np.max(solver.C):.6e} mol/m³")
    logger.info(f"✓ Rango real: [{C_min_final*1000:.3f}, {C_max_final*1000:.3f}] mmol/m³")
    logger.info("")
    logger.info("📊 Archivos:")
    logger.info(f"  {filepath_2}")
    logger.info(f"  {filepath_3}")
    logger.info("=" * 70)
    logger.info("✅ GRÁFICOS MEJORADOS GENERADOS")

if __name__ == "__main__":
    main()

