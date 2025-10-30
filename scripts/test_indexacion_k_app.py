"""
Test para verificar que la indexación 2D→1D→2D es consistente.

Verifica que k_app_field.ravel() produce el mismo orden que la matriz K.
"""
import logging
import numpy as np
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.parametros import ParametrosMaestros
from src.geometria.mallado import MallaPolar2D
from src.solver.matrices import construir_matriz_reaccion, indexar_2d_a_1d, indexar_1d_a_2d

def main():
    logger.info("=" * 70)
    logger.info("TEST DE CONSISTENCIA DE INDEXACIÓN k_app")
    logger.info("=" * 70)

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    
    # Generar campo k_app
    k_app_field = malla.generar_campo_k_app()
    logger.info(f"\nCampo k_app generado: shape={k_app_field.shape}")
    
    # Construir matriz K
    K = construir_matriz_reaccion(k_app_field)
    logger.info(f"Matriz K construida: shape={K.shape}, diagonal extraída")
    
    # Extraer diagonal de K
    k_app_vec_from_K = K.diagonal()
    logger.info(f"Vector k_app desde K: shape={k_app_vec_from_K.shape}")
    
    # Aplanar k_app_field
    k_app_vec_from_field = k_app_field.ravel()
    logger.info(f"Vector k_app desde field.ravel(): shape={k_app_vec_from_field.shape}")
    
    # VERIFICACIÓN 1: ¿Son idénticos?
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICACIÓN 1: ¿field.ravel() == K.diagonal()?")
    logger.info("=" * 70)
    
    son_identicos = np.allclose(k_app_vec_from_field, k_app_vec_from_K)
    logger.info(f"¿Son idénticos?: {son_identicos}")
    
    if not son_identicos:
        logger.error("❌ PROBLEMA: Los vectores NO son idénticos")
        diff = np.abs(k_app_vec_from_field - k_app_vec_from_K)
        logger.error(f"   Max diferencia: {np.max(diff):.6e}")
    else:
        logger.info("✅ Los vectores son idénticos")
    
    # VERIFICACIÓN 2: ¿El orden de indexación es correcto?
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICACIÓN 2: Orden de indexación 2D → 1D → 2D")
    logger.info("=" * 70)
    
    # Seleccionar algunos nodos específicos EN el defecto y FUERA del defecto
    nodos_test = [
        # (i, j, descripción)
        (20, 0, "R/3, θ=0° (DEFECTO - borde interno)"),
        (30, 6, "~R/2, θ=22.7° (DEFECTO - centro)"),
        (40, 0, "2R/3, θ=0° (DEFECTO - borde externo)"),
        (30, 24, "~R/2, θ=90° (ACTIVA)"),
        (60, 47, "R, θ=180° (ACTIVA - frontera)"),
    ]
    
    logger.info("\nNodos de prueba:")
    for i, j, desc in nodos_test:
        # Valor directo del campo 2D
        k_app_2d = k_app_field[i, j]
        
        # Convertir a índice 1D
        k = indexar_2d_a_1d(i, j, malla.ntheta)
        
        # Valor del vector 1D
        k_app_1d_ravel = k_app_vec_from_field[k]
        k_app_1d_K = k_app_vec_from_K[k]
        
        # Verificar si está en defecto
        r_i = malla.r[i]
        theta_j = malla.theta[j]
        en_defecto_radial = (params.geometria.r1 <= r_i <= params.geometria.r2)
        en_defecto_angular = (params.geometria.theta1 <= theta_j <= params.geometria.theta2)
        en_defecto = en_defecto_radial and en_defecto_angular
        
        # Valor esperado
        k_app_esperado = 0.0 if en_defecto else params.cinetica.k_app
        
        logger.info(f"\n  {desc}")
        logger.info(f"    (i, j) = ({i}, {j}) → k = {k}")
        logger.info(f"    r = {r_i*1000:.3f} mm, θ = {np.rad2deg(theta_j):.1f}°")
        logger.info(f"    ¿En defecto?: {en_defecto}")
        logger.info(f"    k_app[i,j] (2D):     {k_app_2d:.6e}")
        logger.info(f"    k_app_vec[k] (ravel): {k_app_1d_ravel:.6e}")
        logger.info(f"    K.diagonal()[k]:     {k_app_1d_K:.6e}")
        logger.info(f"    Esperado:            {k_app_esperado:.6e}")
        
        # Verificar consistencia
        ok_2d = np.isclose(k_app_2d, k_app_esperado)
        ok_1d = np.isclose(k_app_1d_K, k_app_esperado)
        
        if ok_2d and ok_1d:
            logger.info(f"    ✅ CORRECTO")
        else:
            logger.error(f"    ❌ ERROR: valor no coincide con esperado")
    
    # VERIFICACIÓN 3: Estadísticas globales
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICACIÓN 3: Estadísticas globales")
    logger.info("=" * 70)
    
    n_ceros_field = np.sum(k_app_field == 0)
    n_ceros_K = np.sum(K.diagonal() == 0)
    n_activos_field = np.sum(k_app_field > 0)
    n_activos_K = np.sum(K.diagonal() > 0)
    
    logger.info(f"\nCampo k_app (2D):")
    logger.info(f"  Nodos con k_app=0: {n_ceros_field} ({100*n_ceros_field/k_app_field.size:.2f}%)")
    logger.info(f"  Nodos con k_app>0: {n_activos_field} ({100*n_activos_field/k_app_field.size:.2f}%)")
    
    logger.info(f"\nMatriz K (diagonal):")
    logger.info(f"  Nodos con k_app=0: {n_ceros_K} ({100*n_ceros_K/K.shape[0]:.2f}%)")
    logger.info(f"  Nodos con k_app>0: {n_activos_K} ({100*n_activos_K/K.shape[0]:.2f}%)")
    
    # CONCLUSIÓN
    logger.info("\n" + "=" * 70)
    logger.info("CONCLUSIÓN")
    logger.info("=" * 70)
    
    if son_identicos and n_ceros_field == n_ceros_K:
        logger.info("✅ La indexación es CONSISTENTE")
        logger.info("✅ k_app_field se transfiere correctamente a la matriz K")
        logger.info("\n💡 Si el defecto no se ve en la concentración,")
        logger.info("   entonces el problema es FÍSICO, no de implementación:")
        logger.info("   - El defecto es muy pequeño (4.17%)")
        logger.info("   - La difusión domina sobre la reacción (Da ~ 0.015)")
        logger.info("   - Se necesita más tiempo o defecto más grande")
    else:
        logger.error("❌ HAY UN PROBLEMA DE INDEXACIÓN")
        logger.error("   k_app no se está aplicando en los nodos correctos")
    
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

