"""
Script de diagnóstico para verificar que el defecto se aplica correctamente.

Valida que k_app = 0 en la región del defecto y visualiza el campo k_app.
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
from src.postproceso.visualizacion import _dibujar_defecto_anillo, configurar_estilo_matplotlib

def main():
    logger.info("=" * 70)
    logger.info("DIAGNÓSTICO DEL CAMPO k_app Y REGIÓN DE DEFECTO")
    logger.info("=" * 70)

    # Cargar parámetros
    params = ParametrosMaestros()
    params.validar_todo()

    # Crear malla
    malla = MallaPolar2D(params)
    
    # Obtener campo k_app
    k_app_field = malla.generar_campo_k_app()
    
    # Obtener máscaras de regiones
    mask_defecto = malla.identificar_region_defecto()
    mask_activa = malla.identificar_region_activa()
    
    # Estadísticas
    logger.info("")
    logger.info("=" * 70)
    logger.info("ESTADÍSTICAS DE REGIONES")
    logger.info("=" * 70)
    
    n_total = malla.nr * malla.ntheta
    n_defecto = mask_defecto.sum()
    n_activa = mask_activa.sum()
    
    logger.info(f"Total de nodos: {n_total}")
    logger.info(f"Nodos en defecto: {n_defecto} ({100*n_defecto/n_total:.2f}%)")
    logger.info(f"Nodos en activa: {n_activa} ({100*n_activa/n_total:.2f}%)")
    logger.info(f"Suma (debe ser total): {n_defecto + n_activa}")
    
    # Verificar k_app en cada región
    logger.info("")
    logger.info("=" * 70)
    logger.info("CAMPO k_app")
    logger.info("=" * 70)
    
    k_app_defecto = k_app_field[mask_defecto]
    k_app_activa = k_app_field[mask_activa]
    
    logger.info(f"k_app en región DEFECTO:")
    logger.info(f"  - min: {k_app_defecto.min():.6e} s⁻¹")
    logger.info(f"  - max: {k_app_defecto.max():.6e} s⁻¹")
    logger.info(f"  - mean: {k_app_defecto.mean():.6e} s⁻¹")
    logger.info(f"  - ¿Todos cero?: {np.all(k_app_defecto == 0)}")
    
    logger.info(f"\nk_app en región ACTIVA:")
    logger.info(f"  - min: {k_app_activa.min():.6e} s⁻¹")
    logger.info(f"  - max: {k_app_activa.max():.6e} s⁻¹")
    logger.info(f"  - mean: {k_app_activa.mean():.6e} s⁻¹")
    logger.info(f"  - Valor esperado: {params.cinetica.k_app:.6e} s⁻¹")
    logger.info(f"  - ¿Todos iguales?: {np.all(k_app_activa == params.cinetica.k_app)}")
    
    # Geometría del defecto
    logger.info("")
    logger.info("=" * 70)
    logger.info("GEOMETRÍA DEL DEFECTO (ANILLO)")
    logger.info("=" * 70)
    logger.info(f"Radio interno: r1 = {params.geometria.r1*1000:.3f} mm (R/3)")
    logger.info(f"Radio externo: r2 = {params.geometria.r2*1000:.3f} mm (2R/3)")
    logger.info(f"Ángulo inicial: θ1 = {np.rad2deg(params.geometria.theta1):.1f}°")
    logger.info(f"Ángulo final: θ2 = {np.rad2deg(params.geometria.theta2):.1f}°")
    
    area_total = malla.calcular_area_total()
    area_defecto = malla.calcular_area_defecto()
    fraccion = area_defecto / area_total
    
    logger.info(f"\nÁrea total: {area_total*1e6:.3f} mm²")
    logger.info(f"Área defecto: {area_defecto*1e6:.3f} mm² ({100*fraccion:.2f}%)")
    
    # VISUALIZACIÓN DEL CAMPO k_app
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERANDO VISUALIZACIÓN DEL CAMPO k_app")
    logger.info("=" * 70)
    
    configurar_estilo_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), 
                                     subplot_kw=dict(projection='polar'))
    
    # Crear meshgrids compatibles con contourf polar (indexing='xy')
    # Esto da shape (ntheta, nr) que es compatible con k_app_field.T
    R_grid, THETA_grid = np.meshgrid(malla.r, malla.theta)
    
    # SUBPLOT 1: Campo k_app
    # k_app_field tiene shape (nr, ntheta)
    # meshgrid tiene shape (ntheta, nr)
    # Por lo tanto, usamos k_app_field.T
    im1 = ax1.contourf(THETA_grid, R_grid, k_app_field.T, 
                       levels=20, cmap='RdYlGn')
    _dibujar_defecto_anillo(ax1, params, color='blue', linewidth=2.5, label='Defecto')
    ax1.set_title('Campo k_app [s⁻¹]\n(Verde=Activo, Rojo=Defecto)', fontsize=14)
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='k_app [s⁻¹]', pad=0.1)
    
    # SUBPLOT 2: Máscara de regiones
    mascara_visual = np.zeros_like(k_app_field)
    mascara_visual[mask_activa] = 1  # Activa = 1
    mascara_visual[mask_defecto] = 0  # Defecto = 0
    
    im2 = ax2.contourf(THETA_grid, R_grid, mascara_visual.T, 
                       levels=[0, 0.5, 1], cmap='RdYlGn', alpha=0.7)
    _dibujar_defecto_anillo(ax2, params, color='blue', linewidth=2.5, label='Borde defecto')
    ax2.set_title('Máscara de Regiones\n(Verde=Activo, Rojo=Defecto)', fontsize=14)
    ax2.legend(loc='upper right')
    
    plt.suptitle('Diagnóstico del Campo k_app y Defecto', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar
    output_dir = "data/output/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diagnostico_defecto_k_app.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Figura guardada: {output_path}")
    
    plt.close(fig)
    
    # RESUMEN
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESUMEN DE VALIDACIÓN")
    logger.info("=" * 70)
    
    checks = [
        ("k_app = 0 en defecto", np.all(k_app_defecto == 0)),
        ("k_app > 0 en activa", np.all(k_app_activa > 0)),
        ("k_app activa = valor esperado", np.all(k_app_activa == params.cinetica.k_app)),
        ("Regiones mutuamente excluyentes", np.all((mask_defecto & mask_activa) == False)),
        ("Regiones cubren todo", np.all((mask_defecto | mask_activa) == True)),
    ]
    
    all_pass = True
    for check_name, check_result in checks:
        status = "✅ PASS" if check_result else "❌ FAIL"
        logger.info(f"  {status}: {check_name}")
        if not check_result:
            all_pass = False
    
    logger.info("")
    if all_pass:
        logger.info("🎉 TODAS LAS VERIFICACIONES PASARON")
        logger.info("✅ El defecto se está aplicando correctamente en la simulación")
    else:
        logger.error("⚠️  ALGUNAS VERIFICACIONES FALLARON")
        logger.error("❌ Revisar implementación del campo k_app")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

