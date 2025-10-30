"""
Análisis cuantitativo del efecto del defecto en la distribución de CO.

Verifica si el defecto tiene un efecto real (aunque sutil) en la concentración.
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

def main():
    logger.info("=" * 70)
    logger.info("ANÁLISIS CUANTITATIVO DEL EFECTO DEL DEFECTO")
    logger.info("=" * 70)

    # Cargar parámetros y simular hasta estado estacionario
    params = ParametrosMaestros()
    params.validar_todo()
    
    solver = CrankNicolsonSolver2D(params, dt=0.01)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    
    logger.info("Simulando hasta estado estacionario...")
    
    C_prev = solver.C.copy()
    for i in range(10000):  # Máximo 100s
        solver.paso_temporal()
        
        if (i + 1) % 500 == 0:
            if solver.verificar_convergencia(C_prev, tol=1e-5):
                logger.info(f"✓ Convergencia en t={solver.t:.1f}s")
                break
            C_prev = solver.C.copy()
    
    # ==================================================================
    # ANÁLISIS 1: Comparar concentraciones promedio
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS 1: Concentraciones Promedio")
    logger.info("=" * 70)
    
    mascara_defecto = solver.malla.identificar_region_defecto()
    mascara_activa = solver.malla.identificar_region_activa()
    
    C_defecto_promedio = np.mean(solver.C[mascara_defecto])
    C_activa_promedio = np.mean(solver.C[mascara_activa])
    C_total_promedio = np.mean(solver.C)
    
    diferencia_absoluta = C_defecto_promedio - C_activa_promedio
    diferencia_relativa = (diferencia_absoluta / C_activa_promedio) * 100
    
    logger.info(f"C_promedio en región ACTIVA:  {C_activa_promedio:.6e} mol/m³")
    logger.info(f"C_promedio en región DEFECTO: {C_defecto_promedio:.6e} mol/m³")
    logger.info(f"C_promedio TOTAL:             {C_total_promedio:.6e} mol/m³")
    logger.info(f"C_bulk (frontera):            {params.operacion.C_bulk:.6e} mol/m³")
    logger.info("")
    logger.info(f"Diferencia absoluta: {diferencia_absoluta:.6e} mol/m³")
    logger.info(f"Diferencia relativa: {diferencia_relativa:.3f}%")
    
    # ==================================================================
    # ANÁLISIS 2: Perfiles radiales comparativos
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS 2: Perfiles Radiales")
    logger.info("=" * 70)
    
    # Encontrar índices de θ en defecto y activo
    idx_theta_defecto = np.argmin(np.abs(solver.malla.theta - np.pi/8))  # 22.5° (centro del defecto)
    idx_theta_activo = np.argmin(np.abs(solver.malla.theta - np.pi/2))   # 90° (lejos del defecto)
    
    C_perfil_defecto = solver.C[:, idx_theta_defecto]
    C_perfil_activo = solver.C[:, idx_theta_activo]
    
    logger.info(f"Perfil en θ=22.5° (DEFECTO):")
    logger.info(f"  C_min = {np.min(C_perfil_defecto):.6e} mol/m³")
    logger.info(f"  C_max = {np.max(C_perfil_defecto):.6e} mol/m³")
    logger.info(f"  C_promedio = {np.mean(C_perfil_defecto):.6e} mol/m³")
    logger.info("")
    logger.info(f"Perfil en θ=90° (ACTIVO):")
    logger.info(f"  C_min = {np.min(C_perfil_activo):.6e} mol/m³")
    logger.info(f"  C_max = {np.max(C_perfil_activo):.6e} mol/m³")
    logger.info(f"  C_promedio = {np.mean(C_perfil_activo):.6e} mol/m³")
    
    # ==================================================================
    # ANÁLISIS 3: Mapa de diferencias
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS 3: Estadísticas Globales")
    logger.info("=" * 70)
    
    C_min_global = np.min(solver.C)
    C_max_global = np.max(solver.C)
    C_std_global = np.std(solver.C)
    
    logger.info(f"C_min global:  {C_min_global:.6e} mol/m³")
    logger.info(f"C_max global:  {C_max_global:.6e} mol/m³")
    logger.info(f"C_std global:  {C_std_global:.6e} mol/m³")
    logger.info(f"Rango (max-min): {C_max_global - C_min_global:.6e} mol/m³")
    logger.info(f"Variación relativa: {(C_std_global/C_total_promedio)*100:.3f}%")
    
    # ==================================================================
    # ANÁLISIS 4: Comparación con escala de colormaps
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS 4: Escala Visual")
    logger.info("=" * 70)
    
    rango_colormap = C_max_global - C_min_global
    rango_cbulk = params.operacion.C_bulk - 0
    
    logger.info(f"Rango de colormap usado (0 a C_bulk): {rango_cbulk:.6e} mol/m³")
    logger.info(f"Rango real de concentraciones:         {rango_colormap:.6e} mol/m³")
    logger.info(f"Fracción del colormap usada: {(rango_colormap/rango_cbulk)*100:.2f}%")
    
    # ==================================================================
    # VISUALIZACIÓN MEJORADA
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO VISUALIZACIÓN MEJORADA")
    logger.info("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Perfil radial comparativo
    ax1 = axes[0, 0]
    ax1.plot(solver.malla.r * 1000, C_perfil_defecto * 1000, 
             'r-', linewidth=2, label='θ=22.5° (Defecto)', marker='o', markersize=3)
    ax1.plot(solver.malla.r * 1000, C_perfil_activo * 1000,
             'b-', linewidth=2, label='θ=90° (Activo)', marker='s', markersize=3)
    
    # Marcar región del defecto radialmente
    r1_mm = params.geometria.r1 * 1000
    r2_mm = params.geometria.r2 * 1000
    ax1.axvspan(r1_mm, r2_mm, alpha=0.2, color='red', label='Zona defecto (radial)')
    
    ax1.set_xlabel('Radio [mm]', fontsize=12)
    ax1.set_ylabel('Concentración CO [mmol/m³]', fontsize=12)
    ax1.set_title('Perfiles Radiales: Defecto vs Activo', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diferencia (Defecto - Activo) vs radio
    ax2 = axes[0, 1]
    diferencia_radial = (C_perfil_defecto - C_perfil_activo) * 1e6  # μmol/m³
    ax2.plot(solver.malla.r * 1000, diferencia_radial,
             'g-', linewidth=2, marker='o', markersize=3)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.axvspan(r1_mm, r2_mm, alpha=0.2, color='red', label='Zona defecto')
    
    ax2.set_xlabel('Radio [mm]', fontsize=12)
    ax2.set_ylabel('ΔC = C_defecto - C_activo [μmol/m³]', fontsize=12)
    ax2.set_title('Diferencia de Concentración (Defecto - Activo)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mapa 2D con escala ajustada
    ax3 = axes[1, 0]
    ax3.axis('equal')
    ax3.axis('off')
    
    # Convertir a cartesianas para plot más claro
    R_grid, THETA_grid = np.meshgrid(solver.malla.r, solver.malla.theta)
    X = R_grid * np.cos(THETA_grid)
    Y = R_grid * np.sin(THETA_grid)
    
    # Usar escala de colores AJUSTADA al rango real
    levels_ajustado = np.linspace(C_min_global, C_max_global, 50)
    contour = ax3.contourf(X * 1000, Y * 1000, solver.C.T, 
                           levels=levels_ajustado, cmap='hot', extend='both')
    
    # Marcar defecto
    theta_defecto = np.linspace(params.geometria.theta1, params.geometria.theta2, 50)
    r1_x = params.geometria.r1 * np.cos(theta_defecto) * 1000
    r1_y = params.geometria.r1 * np.sin(theta_defecto) * 1000
    r2_x = params.geometria.r2 * np.cos(theta_defecto) * 1000
    r2_y = params.geometria.r2 * np.sin(theta_defecto) * 1000
    
    ax3.plot(r1_x, r1_y, 'c--', linewidth=2, label='Defecto')
    ax3.plot(r2_x, r2_y, 'c--', linewidth=2)
    ax3.plot([r1_x[0], r2_x[0]], [r1_y[0], r2_y[0]], 'c--', linewidth=2)
    ax3.plot([r1_x[-1], r2_x[-1]], [r1_y[-1], r2_y[-1]], 'c--', linewidth=2)
    
    cbar3 = plt.colorbar(contour, ax=ax3)
    cbar3.set_label('C [mol/m³]', fontsize=10)
    ax3.set_title('Mapa 2D - Escala Ajustada', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    
    # Plot 4: Histograma de concentraciones
    ax4 = axes[1, 1]
    
    C_defecto_flat = solver.C[mascara_defecto] * 1000
    C_activa_flat = solver.C[mascara_activa] * 1000
    
    ax4.hist(C_activa_flat, bins=30, alpha=0.6, color='blue', label='Activo', density=True)
    ax4.hist(C_defecto_flat, bins=30, alpha=0.6, color='red', label='Defecto', density=True)
    
    ax4.axvline(C_activa_promedio * 1000, color='blue', linestyle='--', 
                linewidth=2, label=f'Promedio Activo: {C_activa_promedio*1000:.3f}')
    ax4.axvline(C_defecto_promedio * 1000, color='red', linestyle='--',
                linewidth=2, label=f'Promedio Defecto: {C_defecto_promedio*1000:.3f}')
    
    ax4.set_xlabel('Concentración CO [mmol/m³]', fontsize=12)
    ax4.set_ylabel('Densidad de Probabilidad', fontsize=12)
    ax4.set_title('Distribución de Concentraciones', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis Detallado del Efecto del Defecto', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "data/output/figures/analisis_efecto_defecto_detallado.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Figura guardada: {output_path}")
    plt.close()
    
    # ==================================================================
    # CONCLUSIÓN
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONCLUSIÓN")
    logger.info("=" * 70)
    
    if diferencia_relativa < 1.0:
        logger.info("⚠️  El efecto del defecto es MÍNIMO (<1% diferencia)")
        logger.info("    Razón: Módulo de Thiele pequeño (φ ≈ 0.124)")
        logger.info("    → Difusión >> Reacción (control difusional externo)")
        logger.info("    → Los gradientes se 'aplanan' por difusión rápida")
    elif diferencia_relativa < 5.0:
        logger.info("✓  El efecto del defecto es SUTIL (~{:.1f}% diferencia)".format(diferencia_relativa))
        logger.info("    Visible en análisis cuantitativo pero no en gráficos estándar")
    else:
        logger.info("✓  El efecto del defecto es SIGNIFICATIVO (>{:.1f}% diferencia)".format(diferencia_relativa))
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ANÁLISIS COMPLETADO")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

