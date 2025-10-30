"""
Análisis del efecto del defecto en la concentración.

Compara perfiles de concentración en diferentes ángulos para verificar
que el defecto (k_app=0) está afectando la distribución de CO.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.parametros import ParametrosMaestros
from src.geometria.mallado import MallaPolar2D
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.postproceso.visualizacion import configurar_estilo_matplotlib

def main():
    logger.info("=" * 70)
    logger.info("ANÁLISIS DEL EFECTO DEL DEFECTO EN LA CONCENTRACIÓN")
    logger.info("=" * 70)

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()

    # Inicializar solver
    dt = 0.002  # s
    solver = CrankNicolsonSolver2D(params, dt=dt)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    # Simular hasta t=15s
    t_final = 15.0
    n_pasos = int(t_final / dt)
    
    logger.info(f"\nSimulando hasta t={t_final}s con dt={dt}s ({n_pasos} pasos)...")
    
    for i in range(n_pasos):
        solver.paso_temporal()
        if (i + 1) % 1000 == 0:
            logger.info(f"  Paso {i+1}/{n_pasos}: t={solver.t:.2f}s")
    
    logger.info(f"✓ Simulación completada en t={solver.t:.2f}s")
    
    # ANÁLISIS DE PERFILES RADIALES
    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS DE PERFILES RADIALES")
    logger.info("=" * 70)
    
    # Seleccionar 3 ángulos:
    # - θ=0° (DENTRO del defecto)
    # - θ=22.5° (BORDE del defecto, θ2=45°/2)
    # - θ=90° (FUERA del defecto, región activa)
    
    theta_grados = [0, 22.5, 90, 180]
    theta_rad = [np.deg2rad(ang) for ang in theta_grados]
    
    # Encontrar índices más cercanos
    indices_theta = [np.argmin(np.abs(solver.malla.theta - t)) for t in theta_rad]
    
    logger.info("\nPerfiles seleccionados:")
    for i, (ang, idx) in enumerate(zip(theta_grados, indices_theta)):
        theta_real = np.rad2deg(solver.malla.theta[idx])
        # Verificar si está en defecto
        en_defecto = (params.geometria.theta1 <= solver.malla.theta[idx] <= params.geometria.theta2)
        estado = "DEFECTO" if en_defecto else "ACTIVA"
        logger.info(f"  θ={ang}° (idx={idx}, real={theta_real:.1f}°) -> Región {estado}")
    
    # Extraer perfiles radiales
    perfiles = []
    for idx in indices_theta:
        perfil = solver.C[:, idx]  # C(r) para θ fijo
        perfiles.append(perfil)
    
    # Estadísticas
    logger.info("\n" + "=" * 70)
    logger.info("ESTADÍSTICAS DE CONCENTRACIÓN")
    logger.info("=" * 70)
    
    for i, (ang, perfil) in enumerate(zip(theta_grados, perfiles)):
        logger.info(f"\nθ={ang}°:")
        logger.info(f"  C_min = {perfil.min():.6e} mol/m³")
        logger.info(f"  C_max = {perfil.max():.6e} mol/m³")
        logger.info(f"  C_mean = {perfil.mean():.6e} mol/m³")
        logger.info(f"  ΔC = {perfil.max() - perfil.min():.6e} mol/m³")
    
    # Comparar concentración promedio en defecto vs activa
    logger.info("\n" + "=" * 70)
    logger.info("COMPARACIÓN DEFECTO vs ACTIVA")
    logger.info("=" * 70)
    
    # Concentración promedio en diferentes regiones radiales
    # r1 = R/3, r2 = 2R/3
    idx_r1 = np.argmin(np.abs(solver.malla.r - params.geometria.r1))
    idx_r2 = np.argmin(np.abs(solver.malla.r - params.geometria.r2))
    
    logger.info(f"\nÍndices radiales del defecto:")
    logger.info(f"  r1 (R/3) = idx {idx_r1}, r={solver.malla.r[idx_r1]*1000:.3f} mm")
    logger.info(f"  r2 (2R/3) = idx {idx_r2}, r={solver.malla.r[idx_r2]*1000:.3f} mm")
    
    # Concentración en el anillo del defecto
    C_anillo_defecto_ang0 = solver.C[idx_r1:idx_r2+1, indices_theta[0]].mean()  # θ=0° (defecto)
    C_anillo_activo_ang90 = solver.C[idx_r1:idx_r2+1, indices_theta[2]].mean()  # θ=90° (activa)
    
    logger.info(f"\nConcentración promedio en anillo r∈[R/3, 2R/3]:")
    logger.info(f"  θ=0° (DEFECTO):  C_mean = {C_anillo_defecto_ang0:.6e} mol/m³")
    logger.info(f"  θ=90° (ACTIVA):  C_mean = {C_anillo_activo_ang90:.6e} mol/m³")
    logger.info(f"  Diferencia:      ΔC = {abs(C_anillo_defecto_ang0 - C_anillo_activo_ang90):.6e} mol/m³")
    logger.info(f"  Ratio:           C_defecto/C_activa = {C_anillo_defecto_ang0/C_anillo_activo_ang90:.4f}")
    
    # VISUALIZACIÓN
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO VISUALIZACIÓN")
    logger.info("=" * 70)
    
    configurar_estilo_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # SUBPLOT 1: Perfiles radiales comparativos
    colores = ['red', 'orange', 'blue', 'green']
    for i, (ang, perfil, color) in enumerate(zip(theta_grados, perfiles, colores)):
        en_defecto = i < 2  # 0° y 22.5° están en defecto
        estilo = '--' if en_defecto else '-'
        label = f'θ={ang}° ({"DEFECTO" if en_defecto else "ACTIVA"})'
        ax1.plot(solver.malla.r * 1000, perfil, color=color, linestyle=estilo, 
                 linewidth=2.5, label=label, marker='o', markersize=3, markevery=5)
    
    # Marcar región del defecto (anillo)
    ax1.axvspan(params.geometria.r1 * 1000, params.geometria.r2 * 1000, 
                alpha=0.2, color='red', label='Anillo defecto (R/3 a 2R/3)')
    
    ax1.set_xlabel('Radio [mm]', fontsize=12)
    ax1.set_ylabel('Concentración CO [mol/m³]', fontsize=12)
    ax1.set_title(f'Perfiles Radiales en Diferentes Ángulos (t={solver.t:.1f}s)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: Concentración en el anillo del defecto vs ángulo
    C_anillo_vs_theta = []
    for j in range(solver.malla.ntheta):
        C_anillo = solver.C[idx_r1:idx_r2+1, j].mean()
        C_anillo_vs_theta.append(C_anillo)
    
    ax2.plot(np.rad2deg(solver.malla.theta), C_anillo_vs_theta, 'b-', linewidth=2)
    
    # Marcar región angular del defecto
    ax2.axvspan(np.rad2deg(params.geometria.theta1), np.rad2deg(params.geometria.theta2), 
                alpha=0.2, color='red', label='Región angular del defecto')
    
    ax2.set_xlabel('Ángulo θ [°]', fontsize=12)
    ax2.set_ylabel('Concentración promedio en anillo\n[R/3, 2R/3] [mol/m³]', fontsize=12)
    ax2.set_title(f'Concentración en Anillo Defectuoso vs Ángulo (t={solver.t:.1f}s)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 360])
    
    plt.suptitle('Análisis del Efecto del Defecto en la Distribución de CO', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "analisis_efecto_defecto.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Figura guardada: {output_path}")
    
    plt.close(fig)
    
    # CONCLUSIÓN
    logger.info("\n" + "=" * 70)
    logger.info("CONCLUSIÓN")
    logger.info("=" * 70)
    
    diferencia_porcentual = abs(C_anillo_defecto_ang0 - C_anillo_activo_ang90) / C_anillo_activo_ang90 * 100
    
    if diferencia_porcentual < 1:
        logger.warning(f"⚠️  DIFERENCIA MUY PEQUEÑA: {diferencia_porcentual:.2f}%")
        logger.warning("El defecto parece NO estar afectando significativamente la concentración")
        logger.warning("Posibles causas:")
        logger.warning("  1. El defecto es muy pequeño (4.17% del área)")
        logger.warning("  2. El tiempo de simulación es insuficiente")
        logger.warning("  3. El solver no está aplicando correctamente k_app")
    elif diferencia_porcentual < 5:
        logger.info(f"✓ Diferencia moderada: {diferencia_porcentual:.2f}%")
        logger.info("El defecto está afectando la concentración, pero el efecto es sutil")
    else:
        logger.info(f"✅ Diferencia significativa: {diferencia_porcentual:.2f}%")
        logger.info("El defecto está afectando claramente la distribución de concentración")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

