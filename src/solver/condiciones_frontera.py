"""
Módulo de Condiciones de Frontera.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa las condiciones de frontera para la ecuación
de difusión-reacción 2D en coordenadas polares.

Condiciones de frontera del problema:
1. r=0 (centro):     ∂C/∂r = 0  (simetría)
2. r=R (frontera):   Condición Robin: -D_eff·∂C/∂r = k_c·(C - C_bulk)
3. θ=0 ≡ θ=2π:       Periodicidad angular (implícita en indexación)
4. Interfaz:         Continuidad de flujo entre activo y defecto

Referencias
----------
.. [1] Patankar, S.V. (1980). "Numerical Heat Transfer and Fluid Flow"
.. [2] Ferziger, J.H. & Peric, M. (2002). "Computational Methods for Fluid Dynamics"
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple
import logging

from src.solver.matrices import indexar_2d_a_1d, indexar_1d_a_2d

logger = logging.getLogger(__name__)


# ============================================================================
# CONDICIÓN EN CENTRO (r=0): SIMETRÍA
# ============================================================================


def obtener_nodos_centro(malla) -> np.ndarray:
    """
    Obtiene índices lineales de todos los nodos en r=0.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    nodos_centro : np.ndarray, shape (ntheta,)
        Array con índices lineales de nodos en centro

    Notes
    -----
    Los nodos en r=0 son: (i=0, j=0), (i=0, j=1), ..., (i=0, j=ntheta-1)

    Examples
    --------
    >>> nodos = obtener_nodos_centro(malla)
    >>> print(f"Nodos en centro: {len(nodos)}")
    """
    nodos = np.array([indexar_2d_a_1d(0, j, malla.ntheta) for j in range(malla.ntheta)])

    return nodos


def verificar_simetria_centro(C_field: np.ndarray, malla, tol: float = 1e-8) -> bool:
    """
    Verifica si C es independiente de θ en r=0 (simetría).

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    tol : float, optional
        Tolerancia para considerar simétrico

    Returns
    -------
    es_simetrico : bool
        True si la variación en θ es < tol

    Notes
    -----
    Por simetría física: ∂C/∂r = 0 en r=0
    Implica: C(r=0, θ) = constante para todo θ

    Examples
    --------
    >>> es_sim = verificar_simetria_centro(C_field, malla)
    >>> print(f"¿Simétrico? {es_sim}")
    """
    # Extraer valores en centro (primera fila radial)
    C_centro = C_field[0, :]

    # Calcular desviación estándar
    std_centro = np.std(C_centro)

    # Calcular variación relativa
    mean_centro = np.mean(C_centro)
    if abs(mean_centro) > 1e-12:
        variacion_relativa = std_centro / abs(mean_centro)
    else:
        variacion_relativa = std_centro

    logger.debug(
        f"Simetría centro: std={std_centro:.3e}, variación={variacion_relativa:.3e}"
    )

    return variacion_relativa < tol


def imponer_simetria_centro(C_field: np.ndarray, malla) -> np.ndarray:
    """
    Impone condición de simetría en r=0.

    Promedia valores en θ para r=0 y asigna el promedio a todos los nodos.

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración

    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    C_simetrico : np.ndarray, shape (nr, ntheta)
        Campo con simetría impuesta en centro

    Notes
    -----
    Operación: C(r=0, θ) = ⟨C(r=0, θ)⟩_θ  (promedio angular)

    Examples
    --------
    >>> C_sym = imponer_simetria_centro(C_field, malla)
    >>> assert np.std(C_sym[0, :]) < 1e-12  # Todos iguales en r=0
    """
    # Copiar campo
    C_simetrico = C_field.copy()

    # Calcular promedio en centro
    promedio_centro = np.mean(C_field[0, :])

    # Asignar a todos los nodos en r=0
    C_simetrico[0, :] = promedio_centro

    return C_simetrico


def aplicar_condicion_centro(
    A: sparse.spmatrix, B: sparse.spmatrix, malla
) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Aplica condición de simetría en r=0 a matrices A y B.

    Modifica las filas correspondientes a nodos en r=0 para imponer
    que C sea independiente de θ.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz del lado implícito
    B : sparse.spmatrix
        Matriz del lado explícito
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    A_bc : sparse.spmatrix
        Matriz A con condición de frontera aplicada
    B_bc : sparse.spmatrix
        Matriz B con condición de frontera aplicada

    Notes
    -----
    Estrategia:
    1. Para cada nodo en r=0, la fila de A se modifica a:
       A[k, k] = 1  (diagonal)
       A[k, otros] = 0

    2. La fila de B se modifica para imponer promedio:
       B[k, k_j] = 1/ntheta  para todos los j en r=0

    Esto impone: C^(n+1)[k] = ⟨C^n[r=0]⟩_θ

    Examples
    --------
    >>> A_bc, B_bc = aplicar_condicion_centro(A, B, malla)
    >>> # Resolver A_bc·C_np1 = B_bc·C_n impone simetría en centro
    """
    logger.info("Aplicando condición de simetría en centro (r=0)...")

    # Convertir a LIL para modificación eficiente
    A_bc = A.tolil()
    B_bc = B.tolil()

    # Obtener nodos en centro
    nodos_centro = obtener_nodos_centro(malla)

    # Para cada nodo en r=0
    for k in nodos_centro:
        # ============================================================
        # MODIFICAR MATRIZ A (lado implícito)
        # ============================================================
        # Imponer: C^(n+1)[k] = promedio
        # Fila k de A: solo diagonal = 1, resto = 0
        A_bc[k, :] = 0.0  # Limpiar fila
        A_bc[k, k] = 1.0  # Diagonal

        # ============================================================
        # MODIFICAR MATRIZ B (lado explícito)
        # ============================================================
        # RHS: promedio de todos los nodos en r=0
        # B[k, todos_en_centro] = 1/ntheta
        B_bc[k, :] = 0.0  # Limpiar fila

        # Asignar 1/ntheta a todos los nodos en r=0
        for k_centro in nodos_centro:
            B_bc[k, k_centro] = 1.0 / malla.ntheta

    # Convertir de vuelta a CSR
    A_bc = A_bc.tocsr()
    B_bc = B_bc.tocsr()

    logger.info(f"Condición centro aplicada a {len(nodos_centro)} nodos")

    return A_bc, B_bc


# ============================================================================
# UTILIDADES Y VALIDACIÓN
# ============================================================================


def generar_reporte_condicion_centro(malla) -> str:
    """
    Genera reporte de la condición de frontera en centro.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> print(generar_reporte_condicion_centro(malla))
    """
    nodos_centro = obtener_nodos_centro(malla)

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        CONDICIÓN DE FRONTERA: CENTRO (r=0)                   ║
╚══════════════════════════════════════════════════════════════╝

Tipo de Condición:
  - Simetría radial: ∂C/∂r = 0 en r=0

Implicación Física:
  - C debe ser independiente de θ en el centro
  - C(r=0, θ) = constante para todo θ

Implementación Numérica:
  - Promediar C sobre todos los θ en r=0
  - C^(n+1)[r=0, θ] = ⟨C^n[r=0, θ]⟩_θ

Nodos Afectados:
  - Total:              {len(nodos_centro)}
  - Índices lineales:   {nodos_centro[0]} ... {nodos_centro[-1]}
  - Índices 2D:         (0, 0) ... (0, {malla.ntheta - 1})

Modificación de Matrices:
  - Filas de A:         Diagonal = 1, resto = 0
  - Filas de B:         1/nθ en todos los nodos de centro
    """

    return reporte


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
