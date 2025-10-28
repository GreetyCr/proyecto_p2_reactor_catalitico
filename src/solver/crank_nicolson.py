"""
Solver Crank-Nicolson 2D para ecuación de difusión-reacción.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa el solver principal usando el método
Crank-Nicolson para resolver la ecuación de difusión-reacción 2D
en coordenadas polares.

Ecuación:
    ∂C/∂t = D_eff·∇²C - k_app·C

Condiciones de frontera:
    - r=0: Simetría (∂C/∂r = 0)
    - r=R: Robin (-D_eff·∂C/∂r = k_c·(C_bulk - C_s))
    - θ=0 ≡ θ=2π: Periodicidad angular
    - Interfaz: Continuidad de flujo

Referencias
----------
.. [1] Crank, J., & Nicolson, P. (1947)
.. [2] Patankar, S.V. (1980)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Optional, Dict, Any
import logging

from src.geometria.mallado import MallaPolar2D
from src.solver.matrices import construir_matrices_crank_nicolson
from src.solver.condiciones_frontera import (
    aplicar_condicion_centro,
    aplicar_condicion_robin,
    imponer_simetria_centro,
    construir_vector_fuente_robin,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CLASE PRINCIPAL: CRANK-NICOLSON SOLVER 2D
# ============================================================================


class CrankNicolsonSolver2D:
    """
    Solver Crank-Nicolson para ecuación de difusión-reacción 2D.

    Parámetros
    ----------
    params : ParametrosMaestros
        Parámetros del sistema
    dt : float
        Paso temporal [s]
    malla : MallaPolar2D, optional
        Malla preexistente. Si None, se crea una nueva.

    Atributos
    ---------
    malla : MallaPolar2D
        Malla polar 2D
    k_app_field : np.ndarray
        Campo de k_app (con defecto)
    A : sparse.csr_matrix
        Matriz del lado implícito
    B : sparse.csr_matrix
        Matriz del lado explícito
    b_robin : np.ndarray
        Vector de términos fuente de Robin
    C : np.ndarray
        Campo de concentración actual [mol/m³]
    t : float
        Tiempo actual [s]
    n_iter : int
        Número de iteración actual

    Ejemplos
    --------
    >>> params = ParametrosMaestros()
    >>> solver = CrankNicolsonSolver2D(params, dt=0.001)
    >>> solver.construir_sistema()
    >>> solver.inicializar_campo(C_inicial=0.0)
    >>> for _ in range(1000):
    ...     solver.paso_temporal()
    >>> print(f"t = {solver.t:.3f} s, C_max = {np.max(solver.C):.6f}")
    """

    def __init__(
        self,
        params,
        dt: float,
        malla: Optional[MallaPolar2D] = None,
    ):
        """
        Inicializa el solver Crank-Nicolson 2D.

        Parameters
        ----------
        params : ParametrosMaestros
            Parámetros del sistema
        dt : float
            Paso temporal [s]
        malla : MallaPolar2D, optional
            Malla preexistente
        """
        logger.info("=" * 70)
        logger.info("INICIALIZANDO SOLVER CRANK-NICOLSON 2D")
        logger.info("=" * 70)

        # Guardar parámetros
        self.params = params
        self.dt = dt

        # Crear o usar malla
        if malla is None:
            logger.info("Creando malla polar 2D...")
            self.malla = MallaPolar2D(params)
        else:
            self.malla = malla

        logger.info(
            f"Malla: {self.malla.nr} × {self.malla.ntheta} = {self.malla.nr * self.malla.ntheta} nodos"
        )

        # Generar campo k_app
        logger.info("Generando campo k_app...")
        self.k_app_field = self.malla.generar_campo_k_app()

        # Inicializar variables de estado
        self.C = None  # Campo de concentración
        self.t = 0.0  # Tiempo actual
        self.n_iter = 0  # Número de iteración

        # Matrices del sistema (se construyen después)
        self.A = None
        self.B = None
        self.b_robin = None

        logger.info(f"Solver inicializado con dt = {dt:.3e} s")

    def construir_sistema(self):
        """
        Construye matrices A y B del sistema Crank-Nicolson.

        Aplica todas las condiciones de frontera:
        - Centro (r=0): Simetría
        - Frontera (r=R): Robin
        - Angular: Periodicidad (implícita)
        - Interfaz: Continuidad (implícita)
        """
        logger.info("Construyendo sistema Crank-Nicolson...")

        # Construir matrices base A y B
        logger.info("  - Matrices A y B base...")
        A, B = construir_matrices_crank_nicolson(
            self.malla,
            self.params.difusion.D_eff,
            self.k_app_field,
            self.dt,
        )

        # Aplicar condición de centro (simetría)
        logger.info("  - Aplicando condición de simetría en r=0...")
        A, B = aplicar_condicion_centro(A, B, self.malla)

        # Aplicar condición Robin en frontera externa
        logger.info("  - Aplicando condición Robin en r=R...")
        A, B = aplicar_condicion_robin(
            A,
            B,
            self.malla,
            self.params.difusion.D_eff,
            self.params.transferencia.k_c,
            self.params.operacion.C_bulk,
        )

        # Guardar matrices
        self.A = A
        self.B = B

        # Construir vector de términos fuente de Robin
        self.b_robin = construir_vector_fuente_robin(
            self.malla,
            self.params.transferencia.k_c,
            self.params.operacion.C_bulk,
        )

        logger.info(f"Sistema construido: A nnz={A.nnz}, B nnz={B.nnz}")

    def inicializar_campo(self, C_inicial: float = 0.0):
        """
        Inicializa el campo de concentración.

        Parameters
        ----------
        C_inicial : float, optional
            Valor inicial de concentración [mol/m³]. Default: 0.0

        Notes
        -----
        Condición inicial: C(r, θ, t=0) = C_inicial
        """
        logger.info(f"Inicializando campo con C = {C_inicial:.6f} mol/m³")

        # Crear campo uniforme
        self.C = np.full(
            (self.malla.nr, self.malla.ntheta), C_inicial, dtype=np.float64
        )

        # Imponer simetría en centro
        self.C = imponer_simetria_centro(self.C, self.malla)

        # Resetear tiempo e iteración
        self.t = 0.0
        self.n_iter = 0

        logger.info("Campo inicializado")

    def paso_temporal(self):
        """
        Ejecuta un paso temporal del método Crank-Nicolson.

        Resuelve:
            A·C^(n+1) = B·C^n + b

        Actualiza:
            - self.C: Campo de concentración
            - self.t: Tiempo
            - self.n_iter: Número de iteración
        """
        # Convertir campo 2D a vector 1D
        C_vec_n = self.C.ravel()

        # Construir RHS
        rhs = self.B @ C_vec_n + self.b_robin

        # Resolver sistema lineal A·C^(n+1) = rhs
        try:
            C_vec_np1 = spla.spsolve(self.A, rhs)
        except Exception as e:
            logger.error(f"Error al resolver sistema lineal: {str(e)}")
            raise RuntimeError(
                f"Fallo en resolver sistema en iter {self.n_iter}, t={self.t:.3e}s"
            ) from e

        # Convertir vector 1D a campo 2D
        self.C = C_vec_np1.reshape((self.malla.nr, self.malla.ntheta))

        # Imponer simetría en centro (por seguridad)
        self.C = imponer_simetria_centro(self.C, self.malla)

        # Actualizar tiempo e iteración
        self.t += self.dt
        self.n_iter += 1

        # Log cada cierto número de pasos
        if self.n_iter % 100 == 0:
            C_min = np.min(self.C)
            C_max = np.max(self.C)
            logger.info(
                f"Iter {self.n_iter}: t={self.t:.3e}s, "
                f"C∈[{C_min:.3e}, {C_max:.3e}] mol/m³"
            )

    def obtener_info(self) -> Dict[str, Any]:
        """
        Obtiene información del estado actual del solver.

        Returns
        -------
        info : Dict
            Diccionario con información del solver
        """
        if self.C is None:
            C_min, C_max, C_mean = 0.0, 0.0, 0.0
        else:
            C_min = np.min(self.C)
            C_max = np.max(self.C)
            C_mean = np.mean(self.C)

        info = {
            "t": self.t,
            "n_iter": self.n_iter,
            "dt": self.dt,
            "C_min": C_min,
            "C_max": C_max,
            "C_mean": C_mean,
            "N": self.malla.nr * self.malla.ntheta,
            "nr": self.malla.nr,
            "ntheta": self.malla.ntheta,
        }

        return info

    def generar_reporte(self) -> str:
        """
        Genera reporte del estado del solver.

        Returns
        -------
        reporte : str
            Reporte formateado
        """
        info = self.obtener_info()

        reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        SOLVER CRANK-NICOLSON 2D                              ║
╚══════════════════════════════════════════════════════════════╝

Estado Actual:
  - Tiempo:             {info['t']:.6f} s
  - Iteración:          {info['n_iter']}
  - Paso temporal:      {info['dt']:.3e} s

Campo de Concentración:
  - C_min:              {info['C_min']:.6e} mol/m³
  - C_max:              {info['C_max']:.6e} mol/m³
  - C_mean:             {info['C_mean']:.6e} mol/m³

Malla:
  - Total nodos:        {info['N']}
  - nr × nθ:            {info['nr']} × {info['ntheta']}

Parámetros:
  - D_eff:              {self.params.difusion.D_eff:.3e} m²/s
  - k_app (activo):     {self.params.cinetica.k_app:.3e} s⁻¹
  - k_c:                {self.params.transferencia.k_c:.3e} m/s
  - C_bulk:             {self.params.operacion.C_bulk:.6e} mol/m³
        """

        return reporte


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
