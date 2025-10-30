"""
Test rápido de estabilidad del solver.
Ejecuta solo hasta t=1s para verificar que no haya NaN.
"""

import logging
import numpy as np

from src.config.parametros import ParametrosMaestros
from src.solver.crank_nicolson import CrankNicolsonSolver2D
from src.solver.discretizacion import calcular_dt_critico_euler, calcular_numero_fourier

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_con_dt(dt_test, t_max=1.0):
    """Prueba el solver con un dt específico."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST CON dt = {dt_test:.6f} s")
    logger.info(f"{'='*60}")
    
    params = ParametrosMaestros()
    dr = params.mallado.dr
    D_eff = params.difusion.D_eff
    
    Fo = calcular_numero_fourier(dt_test, dr, D_eff)
    logger.info(f"Número de Fourier: Fo = {Fo:.2f}")
    
    solver = CrankNicolsonSolver2D(params, dt=dt_test)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    
    n_pasos = int(t_max / dt_test)
    logger.info(f"Ejecutando {n_pasos} pasos hasta t={t_max}s...")
    
    try:
        for i in range(n_pasos):
            solver.paso_temporal()
            
            if (i + 1) % 100 == 0:
                C_min, C_max = np.min(solver.C), np.max(solver.C)
                logger.info(f"  Paso {i+1}/{n_pasos}: C∈[{C_min:.6e}, {C_max:.6e}]")
                
                if np.any(np.isnan(solver.C)):
                    logger.error(f"❌ NaN detectado en paso {i+1}")
                    return False
                
                if np.abs(C_max) > 1.0:  # C_bulk = 0.0145, así que 1.0 es muy alto
                    logger.error(f"❌ Valores muy grandes detectados en paso {i+1}")
                    return False
        
        logger.info(f"✅ TEST EXITOSO: t={solver.t:.3f}s sin NaN")
        logger.info(f"   C_min={np.min(solver.C):.6e}, C_max={np.max(solver.C):.6e}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        return False


def main():
    logger.info("🔬 TEST DE ESTABILIDAD DEL SOLVER")
    logger.info("=" * 60)
    
    params = ParametrosMaestros()
    dr = params.mallado.dr
    dtheta = params.mallado.dtheta
    D_eff = params.difusion.D_eff
    
    dt_critico = calcular_dt_critico_euler(dr, dtheta, D_eff)
    logger.info(f"\ndt_crítico (Euler) = {dt_critico:.6f} s")
    logger.info(f"Para Crank-Nicolson:")
    logger.info(f"  - Fo < 1: dt < {dt_critico*2:.6f} s (muy seguro)")
    logger.info(f"  - Fo < 5: dt < {dt_critico*10:.6f} s (seguro)")
    
    # Probar diferentes dt
    resultados = []
    
    # Test 1: dt muy pequeño (Fo ≈ 0.5, muy seguro)
    dt1 = dt_critico
    resultado1 = test_con_dt(dt1, t_max=1.0)
    resultados.append(("Fo≈0.5 (muy seguro)", dt1, resultado1))
    
    # Test 2: dt medio (Fo ≈ 1, seguro)
    dt2 = 2 * dt_critico
    resultado2 = test_con_dt(dt2, t_max=1.0)
    resultados.append(("Fo≈1 (seguro)", dt2, resultado2))
    
    # Test 3: dt grande (Fo ≈ 5, límite)
    dt3 = 10 * dt_critico
    resultado3 = test_con_dt(dt3, t_max=1.0)
    resultados.append(("Fo≈5 (límite)", dt3, resultado3))
    
    # Test 4: dt muy grande (Fo ≈ 9.4, el problemático)
    dt4 = 0.0025  # Un poco menor que 0.01
    resultado4 = test_con_dt(dt4, t_max=1.0)
    resultados.append(("Fo≈2.3", dt4, resultado4))
    
    # Resumen
    logger.info(f"\n{'='*60}")
    logger.info("RESUMEN DE TESTS")
    logger.info(f"{'='*60}")
    for nombre, dt, resultado in resultados:
        estado = "✅ ESTABLE" if resultado else "❌ INESTABLE"
        logger.info(f"{nombre:20s} dt={dt:.6f}s  {estado}")
    
    logger.info(f"\n💡 RECOMENDACIÓN:")
    dt_recomendado = 2 * dt_critico
    logger.info(f"   Usar dt ≤ {dt_recomendado:.6f} s para simulaciones")


if __name__ == "__main__":
    main()

