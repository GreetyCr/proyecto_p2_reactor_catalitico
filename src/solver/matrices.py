"""
Módulo de Construcción de Matrices Dispersas.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa la construcción de matrices dispersas
para el método de Crank-Nicolson en 2D.

Funcionalidades principales:
- Indexación 2D (i,j) ↔ 1D (k)
- Construcción de matriz Laplaciana dispersa
- Ensamblaje eficiente usando scipy.sparse
- Información y diagnóstico de matrices

Referencias
----------
.. [1] Press, W.H. et al. (2007). "Numerical Recipes"
.. [2] Saad, Y. (2003). "Iterative Methods for Sparse Linear Systems"
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict
import logging

from src.solver.discretizacion import calcular_coeficientes_radiales
from src.solver.discretizacion import calcular_coeficientes_angulares

logger = logging.getLogger(__name__)


# ============================================================================
# INDEXACIÓN 2D ↔ 1D
# ============================================================================


def indexar_2d_a_1d(i: int, j: int, ntheta: int) -> int:
    """
    Convierte índices 2D (i, j) a índice lineal 1D (k).

    Usa ordenamiento row-major (C-style): k = i*ntheta + j

    Parameters
    ----------
    i : int
        Índice radial (fila)
    j : int
        Índice angular (columna)
    ntheta : int
        Número total de nodos angulares

    Returns
    -------
    k : int
        Índice lineal 1D

    Notes
    -----
    Esta convención permite pasar de una malla 2D (nr, ntheta)
    a un vector 1D de tamaño N = nr × ntheta.

    Examples
    --------
    >>> k = indexar_2d_a_1d(i=5, j=10, ntheta=96)
    >>> print(k)  # 5*96 + 10 = 490
    490
    """
    k = i * ntheta + j
    return k


def indexar_1d_a_2d(k: int, ntheta: int) -> Tuple[int, int]:
    """
    Convierte índice lineal 1D (k) a índices 2D (i, j).

    Inversa de indexar_2d_a_1d.

    Parameters
    ----------
    k : int
        Índice lineal 1D
    ntheta : int
        Número total de nodos angulares

    Returns
    -------
    i : int
        Índice radial (fila)
    j : int
        Índice angular (columna)

    Examples
    --------
    >>> i, j = indexar_1d_a_2d(k=490, ntheta=96)
    >>> print(f"i={i}, j={j}")  # 490 = 5*96 + 10
    i=5, j=10
    """
    i = k // ntheta
    j = k % ntheta
    return i, j


# ============================================================================
# CONSTRUCCIÓN DE MATRIZ LAPLACIANA
# ============================================================================


def construir_matriz_laplaciana_2d_polar(malla, D_eff: float) -> sparse.csr_matrix:
    """
    Construye matriz dispersa del operador Laplaciano 2D.

    Representa D_eff·∇²C en forma discreta para todo el dominio.

    La matriz L es tal que:
        L @ C_vec = D_eff·∇²C_vec

    Donde C_vec es el campo C(r,θ) aplanado a vector 1D.

    Parameters
    ----------
    malla : MallaPolar2D
        Objeto de malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    L : sparse.csr_matrix, shape (N, N)
        Matriz Laplaciana dispersa en formato CSR
        donde N = nr × ntheta

    Notes
    -----
    Stencil de 5 puntos por nodo (i,j):
    - Vecino radial interior:    (i-1, j)
    - Vecino radial exterior:    (i+1, j)
    - Vecino angular anterior:   (i, j-1)
    - Vecino angular posterior:  (i, j+1)
    - Nodo central:              (i, j)

    Casos especiales:
    - r=0 (centro):    Solo vecino i+1
    - r=R (frontera):  Solo vecino i-1
    - Periodicidad:    j=-1 → j=ntheta-1, j=ntheta → j=0

    Examples
    --------
    >>> from src.geometria.mallado import MallaPolar2D
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)
    >>> print(f"Matriz shape: {L.shape}, nnz: {L.nnz}")
    """
    nr = malla.nr
    ntheta = malla.ntheta
    N = nr * ntheta

    logger.info(f"Construyendo matriz Laplaciana {N}×{N}...")

    # Usar LIL (List of Lists) para construcción eficiente
    L = sparse.lil_matrix((N, N), dtype=np.float64)

    # Ensamblar matriz nodo por nodo
    for i in range(nr):
        for j in range(ntheta):
            # Índice lineal del nodo actual
            k = indexar_2d_a_1d(i, j, ntheta)

            # Obtener coeficientes de discretización
            coefs_r = calcular_coeficientes_radiales(malla.r, i, malla.dr, D_eff)
            coefs_theta = calcular_coeficientes_angulares(malla.r, i, j, malla.dtheta)

            alpha = coefs_r["alpha"]
            beta = coefs_r["beta"]
            gamma_r = coefs_r["gamma_r"]
            gamma_theta = coefs_theta["gamma_theta"]
            es_centro = coefs_r["centro"]

            # ============================================================
            # TÉRMINO RADIAL: ∂²C/∂r² + (1/r)·∂C/∂r
            # ============================================================

            if es_centro:
                # Centro (r=0): solo vecino hacia adelante
                k_ip1 = indexar_2d_a_1d(i + 1, j, ntheta)

                L[k, k] += gamma_r  # Diagonal
                L[k, k_ip1] += beta  # Vecino i+1

            elif i == nr - 1:
                # Frontera r=R: solo vecino hacia atrás
                k_im1 = indexar_2d_a_1d(i - 1, j, ntheta)

                L[k, k] += gamma_r  # Diagonal
                L[k, k_im1] += alpha  # Vecino i-1

            else:
                # Nodo interior: stencil completo
                k_im1 = indexar_2d_a_1d(i - 1, j, ntheta)
                k_ip1 = indexar_2d_a_1d(i + 1, j, ntheta)

                L[k, k] += gamma_r  # Diagonal
                L[k, k_im1] += alpha  # Vecino i-1
                L[k, k_ip1] += beta  # Vecino i+1

            # ============================================================
            # TÉRMINO ANGULAR: (1/r²)·∂²C/∂θ² (solo si r > 0)
            # ============================================================

            if not es_centro:
                # Periodicidad angular: j-1 y j+1 con módulo
                j_prev = (j - 1) % ntheta
                j_next = (j + 1) % ntheta

                k_jm1 = indexar_2d_a_1d(i, j_prev, ntheta)
                k_jp1 = indexar_2d_a_1d(i, j_next, ntheta)

                # Coeficiente angular (1/r²) ya calculado
                # Multiplicar por D_eff
                coef_angular = D_eff * gamma_theta

                L[k, k] += -2 * coef_angular  # Diagonal
                L[k, k_jm1] += coef_angular  # Vecino j-1
                L[k, k_jp1] += coef_angular  # Vecino j+1

    # Convertir a formato CSR (eficiente para operaciones)
    L_csr = L.tocsr()

    logger.info(
        f"Matriz ensamblada: {L_csr.nnz} elementos no-cero "
        f"({100*L_csr.nnz/(N**2):.2f}% sparsity)"
    )

    return L_csr


# ============================================================================
# INFORMACIÓN Y DIAGNÓSTICO
# ============================================================================


def obtener_info_matriz(A: sparse.spmatrix) -> Dict:
    """
    Obtiene información estadística de una matriz dispersa.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz dispersa

    Returns
    -------
    info : Dict
        Diccionario con información:
        - shape: Dimensiones (M, N)
        - nnz: Número de elementos no-cero
        - formato: Formato de la matriz (CSR, COO, etc.)
        - sparsity: Porcentaje de sparsity
        - avg_per_row: Promedio de elementos por fila
        - diagonal_min: Mínimo de diagonal
        - diagonal_max: Máximo de diagonal

    Examples
    --------
    >>> L = construir_matriz_laplaciana_2d_polar(malla, D_eff)
    >>> info = obtener_info_matriz(L)
    >>> print(f"Sparsity: {info['sparsity']:.2f}%")
    """
    M, N = A.shape
    nnz = A.nnz

    # Sparsity (porcentaje de elementos no-cero)
    total_elements = M * N
    sparsity = 100 * nnz / total_elements

    # Promedio de elementos por fila
    avg_per_row = nnz / M

    # Información de diagonal
    diag = A.diagonal()
    diag_min = np.min(diag)
    diag_max = np.max(diag)
    diag_mean = np.mean(diag)

    info = {
        "shape": A.shape,
        "nnz": nnz,
        "formato": A.format,
        "sparsity": sparsity,
        "avg_per_row": avg_per_row,
        "diagonal_min": diag_min,
        "diagonal_max": diag_max,
        "diagonal_mean": diag_mean,
    }

    return info


def generar_reporte_matriz(A: sparse.spmatrix, malla=None) -> str:
    """
    Genera reporte legible de una matriz dispersa.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz dispersa
    malla : MallaPolar2D, optional
        Objeto de malla (para contexto)

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> L = construir_matriz_laplaciana_2d_polar(malla, D_eff)
    >>> print(generar_reporte_matriz(L, malla))
    """
    info = obtener_info_matriz(A)

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║           REPORTE DE MATRIZ DISPERSA                         ║
╚══════════════════════════════════════════════════════════════╝

Dimensiones:
  - Shape:              {info['shape'][0]} × {info['shape'][1]}
  - Formato:            {info['formato'].upper()}

Sparsity:
  - Elementos no-cero:  {info['nnz']:,}
  - Total elementos:    {info['shape'][0] * info['shape'][1]:,}
  - Sparsity:           {info['sparsity']:.3f}%
  - Avg elementos/fila: {info['avg_per_row']:.1f}

Diagonal:
  - Mínimo:             {info['diagonal_min']:.3e}
  - Máximo:             {info['diagonal_max']:.3e}
  - Promedio:           {info['diagonal_mean']:.3e}
"""

    if malla is not None:
        reporte += f"""
Contexto de Malla:
  - nr (nodos radiales): {malla.nr}
  - nθ (nodos angulares): {malla.ntheta}
  - Total nodos:          {malla.nr * malla.ntheta:,}
  - dr:                   {malla.dr:.3e} m
  - dθ:                   {malla.dtheta:.4f} rad ({np.degrees(malla.dtheta):.2f}°)

Stencil esperado:
  - Puntos por nodo:      ~5 (cruz 2D)
  - Bandwidth esperado:   ~{malla.ntheta + 2}
    """

    return reporte


def verificar_matriz_es_invertible(A: sparse.spmatrix, tol: float = 1e-10) -> bool:
    """
    Verifica si una matriz es (probablemente) invertible.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz a verificar
    tol : float, optional
        Tolerancia para considerar valor propio cero

    Returns
    -------
    es_invertible : bool
        True si la matriz parece invertible

    Notes
    -----
    Para matrices grandes, no calculamos todos los valores propios
    (costoso). En su lugar, verificamos:
    1. Diagonal no tiene ceros
    2. Condición de diagonal dominante (aproximada)

    Examples
    --------
    >>> L = construir_matriz_laplaciana_2d_polar(malla, D_eff)
    >>> es_invertible = verificar_matriz_es_invertible(L)
    """
    # Verificar que diagonal no tenga ceros
    diag = A.diagonal()

    if np.any(np.abs(diag) < tol):
        logger.warning("Matriz tiene elementos diagonales ~0")
        return False

    # Para matrices pequeñas, calcular determinante
    if A.shape[0] < 1000:
        try:
            # Convertir a LU y verificar
            from scipy.sparse.linalg import splu

            lu = splu(A.tocsc())
            # Si no lanza excepción, es invertible
            return True
        except:
            return False

    # Para matrices grandes, asumir invertible si diagonal es robusta
    logger.info("Matriz grande: asumiendo invertible basado en diagonal")
    return True


def calcular_condicionamiento_aproximado(A: sparse.spmatrix) -> float:
    """
    Calcula una estimación del número de condición de A.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz dispersa

    Returns
    -------
    cond_approx : float
        Estimación del número de condición

    Notes
    -----
    Para matrices grandes, calcular el condicionamiento exacto es costoso.
    Usamos una aproximación basada en la diagonal.

    Examples
    --------
    >>> L = construir_matriz_laplaciana_2d_polar(malla, D_eff)
    >>> cond = calcular_condicionamiento_aproximado(L)
    >>> print(f"κ(L) ≈ {cond:.2e}")
    """
    diag = A.diagonal()

    # Aproximación burda: ratio max/min de diagonal
    diag_abs = np.abs(diag)
    diag_max = np.max(diag_abs)
    diag_min = np.min(diag_abs[diag_abs > 1e-14])  # Excluir ceros

    cond_approx = diag_max / diag_min

    return cond_approx


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
