"""
Test del solver SIN condición Robin para aislar el problema.

Si funciona sin Robin → El bug está en aplicar_condicion_robin
Si falla sin Robin → El bug está en otro lado
"""

import logging
import numpy as np
from src.config.parametros import ParametrosMaestros
from src.solver.crank_nicolson import CrankNicolsonSolver2D

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("🔬 TEST DEL SOLVER SIN CONDICIÓN ROBIN")
    logger.info("=" * 60)
    
    params = ParametrosMaestros()
    
    # Crear solver con dt pequeño
    dt = 0.001  # 1 ms
    solver = CrankNicolsonSolver2D(params, dt=dt)
    
    # Construir SOLO hasta antes de Robin
    logger.info("Construyendo sistema SIN condición Robin...")
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.solver.condiciones_frontera import aplicar_condicion_centro
    
    k_app_field = solver.malla.generar_campo_k_app()
    
    # Construir A y B base
    A, B = construir_matrices_crank_nicolson(
        solver.malla, params.difusion.D_eff, k_app_field, dt
    )
    
    # Aplicar SOLO simetría en centro
    A, B = aplicar_condicion_centro(A, B, solver.malla)
    
    # NO aplicar Robin - usar Neumann homogéneo en su lugar (∂C/∂r = 0)
    logger.info("Usando Neumann homogéneo (∂C/∂r=0) en r=R")
    
    solver.A = A
    solver.B = B
    solver.b_robin = np.zeros(solver.malla.nr * solver.malla.ntheta)
    
    # Inicializar
    solver.inicializar_campo(C_inicial=0.0)
    
    # Ejecutar 1000 pasos
    logger.info(f"\nEjecutando 1000 pasos con dt={dt}s...")
    
    try:
        for i in range(1000):
            solver.paso_temporal()
            
            if (i + 1) % 100 == 0:
                C_min, C_max = np.min(solver.C), np.max(solver.C)
                logger.info(f"  Paso {i+1}: t={solver.t:.3f}s, C∈[{C_min:.6e}, {C_max:.6e}]")
                
                if np.any(np.isnan(solver.C)) or np.abs(C_max) > 100:
                    logger.error("❌ PROBLEMA DETECTADO")
                    return False
        
        logger.info(f"\n✅ TEST EXITOSO SIN ROBIN")
        logger.info(f"   Simulación estable hasta t={solver.t:.3f}s")
        logger.info(f"   C_max = {np.max(solver.C):.6e} mol/m³")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    main()

