"""
Módulo de Stencils (Plantillas) de Diferencias Finitas.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa los stencils (plantillas) de diferencias finitas
para calcular el operador Laplaciano en coordenadas polares 2D.

Un stencil define el patrón de nodos vecinos y sus coeficientes
para aproximar derivadas espaciales.

Para la ecuación de difusión-reacción:
    ∂C/∂t = D_eff·∇²C - k_app·C

Donde:
    ∇²C = ∂²C/∂r² + (1/r)·∂C/∂r + (1/r²)·∂²C/∂θ²

Referencias
----------
.. [1] LeVeque, R.J. (2007). "Finite Difference Methods"
.. [2] Moin, P. (2010). "Fundamentals of Engineering Numerical Analysis"
"""

import numpy as np
from typing import Tuple
import logging

from src.solver.discretizacion import calcular_coeficientes_radiales
from src.solver.discretizacion import calcular_coeficientes_angulares

logger = logging.getLogger(__name__)


# ============================================================================
# STENCIL RADIAL (TÉRMINO ∂²C/∂r² + (1/r)·∂C/∂r)
# ============================================================================


def aplicar_stencil_radial(
    C_radial: np.ndarray, i: int, r: np.ndarray, dr: float, D_eff: float
) -> float:
    """
    Aplica stencil de diferencias finitas para término radial.

    Calcula: D_eff·[∂²C/∂r² + (1/r)·∂C/∂r] en el nodo i

    Stencil de 3 puntos:
        α·C[i-1] + γ_r·C[i] + β·C[i+1]

    Parameters
    ----------
    C_radial : np.ndarray, shape (nr,)
        Perfil de concentración radial (fijando θ)
    i : int
        Índice radial del nodo
    r : np.ndarray
        Array de coordenadas radiales [m]
    dr : float
        Paso radial [m]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    resultado : float
        Valor del término radial del Laplaciano en nodo i

    Notes
    -----
    Tratamiento especial en r=0 (singularidad):
    - Usar límite de L'Hôpital
    - Solo considerar vecino hacia adelante

    Examples
    --------
    >>> r = np.linspace(0, 0.002, 61)
    >>> C = np.ones(61)  # Campo constante
    >>> resultado = aplicar_stencil_radial(C, i=30, r=r, dr=r[1]-r[0], D_eff=1.04e-6)
    >>> print(f"{resultado:.3e}")  # Debe ser ~0
    """
    # Obtener coeficientes de discretización
    coefs = calcular_coeficientes_radiales(r, i, dr, D_eff)

    alpha = coefs["alpha"]
    beta = coefs["beta"]
    gamma_r = coefs["gamma_r"]
    es_centro = coefs["centro"]

    # Caso especial: Centro (r=0)
    if es_centro:
        # Solo hay vecino i+1
        resultado = gamma_r * C_radial[i] + beta * C_radial[i + 1]
        return resultado

    # Caso especial: Frontera r=R (último nodo)
    nr = len(C_radial)
    if i == nr - 1:
        # Solo hay vecino i-1 (diferencias hacia atrás)
        # Usar aproximación de primer orden o retornar 0
        # Por ahora, usar solo el vecino interior
        resultado = alpha * C_radial[i - 1] + gamma_r * C_radial[i]
        return resultado

    # Nodo interior
    # Stencil de 3 puntos: i-1, i, i+1
    resultado = alpha * C_radial[i - 1] + gamma_r * C_radial[i] + beta * C_radial[i + 1]

    return resultado


# ============================================================================
# STENCIL ANGULAR (TÉRMINO (1/r²)·∂²C/∂θ²)
# ============================================================================


def aplicar_stencil_angular(
    C_angular: np.ndarray, j: int, r_i: float, dtheta: float
) -> float:
    """
    Aplica stencil de diferencias finitas para término angular.

    Calcula: (1/r²)·∂²C/∂θ² en el nodo j

    Stencil de 3 puntos con periodicidad:
        γ_θ·[C[j-1] - 2C[j] + C[j+1]]

    Parameters
    ----------
    C_angular : np.ndarray, shape (ntheta,)
        Perfil de concentración angular (fijando r)
    j : int
        Índice angular del nodo
    r_i : float
        Coordenada radial del nodo [m]
    dtheta : float
        Paso angular [rad]

    Returns
    -------
    resultado : float
        Valor del término angular del Laplaciano

    Notes
    -----
    Periodicidad angular:
    - θ=0 ≡ θ=2π
    - j=-1 → j=ntheta-1
    - j=ntheta → j=0

    En r=0, el término es cero (simetría).

    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 96)
    >>> C = np.ones(96)  # Constante en θ
    >>> resultado = aplicar_stencil_angular(C, j=48, r_i=0.001, dtheta=theta[1]-theta[0])
    >>> print(f"{resultado:.3e}")  # Debe ser ~0
    """
    # Caso especial: Centro (r=0)
    if r_i < 1e-10:  # Prácticamente cero
        # En centro, C debe ser independiente de θ
        return 0.0

    # Obtener coeficientes
    ntheta = len(C_angular)

    # Implementar periodicidad angular
    j_prev = (j - 1) % ntheta
    j_next = (j + 1) % ntheta

    # Diferencias finitas centradas
    d2C_dtheta2 = (
        C_angular[j_prev] - 2 * C_angular[j] + C_angular[j_next]
    ) / dtheta**2

    # Multiplicar por (1/r²)
    resultado = d2C_dtheta2 / r_i**2

    return resultado


# ============================================================================
# STENCIL COMPLETO (LAPLACIANO 2D)
# ============================================================================


def aplicar_stencil_completo(
    C_field: np.ndarray, i: int, j: int, malla, D_eff: float
) -> float:
    """
    Aplica stencil completo para Laplaciano 2D en coordenadas polares.

    Calcula: D_eff·∇²C = D_eff·[∂²C/∂r² + (1/r)·∂C/∂r + (1/r²)·∂²C/∂θ²]

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración 2D
    i : int
        Índice radial
    j : int
        Índice angular
    malla : MallaPolar2D
        Objeto de malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    laplaciano : float
        Valor del Laplaciano en nodo (i,j)

    Examples
    --------
    >>> from src.geometria.mallado import MallaPolar2D
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> C = np.ones((malla.nr, malla.ntheta))
    >>> resultado = aplicar_stencil_completo(C, 30, 48, malla, params.difusion.D_eff)
    >>> print(f"{resultado:.3e}")  # Campo constante → Laplaciano ≈ 0
    """
    # Término radial: operar sobre la columna j (fijar θ)
    termino_radial = aplicar_stencil_radial(C_field[:, j], i, malla.r, malla.dr, D_eff)

    # Término angular: operar sobre la fila i (fijar r)
    # Este ya NO está multiplicado por D_eff
    termino_angular_sin_D = aplicar_stencil_angular(
        C_field[i, :], j, malla.r[i], malla.dtheta
    )

    # El término angular debe multiplicarse por D_eff
    termino_angular = D_eff * termino_angular_sin_D

    # Laplaciano completo
    laplaciano = termino_radial + termino_angular

    return laplaciano


# ============================================================================
# LAPLACIANO SOBRE CAMPO COMPLETO
# ============================================================================


def calcular_laplaciano_campo_completo(
    C_field: np.ndarray, malla, D_eff: float
) -> np.ndarray:
    """
    Calcula el Laplaciano para todo el campo 2D.

    Aplica el stencil completo en cada nodo (i,j) del dominio.

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración 2D
    malla : MallaPolar2D
        Objeto de malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    laplaciano : np.ndarray, shape (nr, ntheta)
        Campo del Laplaciano D_eff·∇²C

    Notes
    -----
    Esta función aplica el stencil en TODOS los nodos,
    incluyendo fronteras. Las condiciones de frontera
    se aplicarán posteriormente.

    Para eficiencia, se podría vectorizar, pero por claridad
    se usa loop explícito.

    Examples
    --------
    >>> from src.geometria.mallado import MallaPolar2D
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> C = np.ones((malla.nr, malla.ntheta))
    >>> lap = calcular_laplaciano_campo_completo(C, malla, params.difusion.D_eff)
    >>> print(f"max(|∇²C|) = {np.max(np.abs(lap)):.3e}")  # Debe ser ~0
    """
    nr, ntheta = C_field.shape
    laplaciano = np.zeros_like(C_field)

    # Aplicar stencil en cada nodo
    for i in range(nr):
        for j in range(ntheta):
            laplaciano[i, j] = aplicar_stencil_completo(C_field, i, j, malla, D_eff)

    return laplaciano


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================


def verificar_laplaciano_funcion_armonica(
    C_field: np.ndarray, malla, D_eff: float, tol: float = 0.1
) -> bool:
    """
    Verifica si un campo es aproximadamente armónico (∇²C ≈ 0).

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo a verificar
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva
    tol : float, optional
        Tolerancia máxima para |∇²C|

    Returns
    -------
    es_armonico : bool
        True si max(|∇²C|) < tol en el interior

    Notes
    -----
    Una función armónica satisface la ecuación de Laplace: ∇²C = 0

    Examples
    --------
    >>> # Función armónica: C(r,θ) = r·cos(θ)
    >>> C = malla.R_grid * np.cos(malla.THETA_grid)
    >>> es_armonico = verificar_laplaciano_funcion_armonica(C, malla, 1.0)
    >>> print(f"¿Es armónica? {es_armonico}")
    """
    laplaciano = calcular_laplaciano_campo_completo(C_field, malla, D_eff)

    # Excluir fronteras (pueden tener error mayor)
    laplaciano_interior = laplaciano[1:-1, :]

    # Verificar que sea aproximadamente cero
    max_abs_lap = np.max(np.abs(laplaciano_interior))

    logger.debug(f"Verificación función armónica: max(|∇²C|) = {max_abs_lap:.3e}")

    return max_abs_lap < tol


def calcular_norma_laplaciano(
    C_field: np.ndarray, malla, D_eff: float, tipo_norma: str = "max"
) -> float:
    """
    Calcula la norma del Laplaciano de un campo.

    Parameters
    ----------
    C_field : np.ndarray
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva
    tipo_norma : str, optional
        Tipo de norma: 'max', 'l2', 'l1'

    Returns
    -------
    norma : float
        Norma del Laplaciano

    Examples
    --------
    >>> norma_max = calcular_norma_laplaciano(C, malla, D_eff, tipo_norma='max')
    >>> norma_l2 = calcular_norma_laplaciano(C, malla, D_eff, tipo_norma='l2')
    """
    laplaciano = calcular_laplaciano_campo_completo(C_field, malla, D_eff)

    if tipo_norma == "max":
        return np.max(np.abs(laplaciano))
    elif tipo_norma == "l2":
        return np.sqrt(np.mean(laplaciano**2))
    elif tipo_norma == "l1":
        return np.mean(np.abs(laplaciano))
    else:
        raise ValueError(f"Tipo de norma desconocida: {tipo_norma}")


# ============================================================================
# INFORMACIÓN Y DEBUGGING
# ============================================================================


def obtener_info_stencil(malla) -> str:
    """
    Genera reporte de información sobre el stencil usado.

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
    >>> from src.geometria.mallado import MallaPolar2D
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> print(obtener_info_stencil(malla))
    """
    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║               INFORMACIÓN DE STENCIL                         ║
╚══════════════════════════════════════════════════════════════╝

Tipo de Stencil:
  - Radial:  3 puntos (i-1, i, i+1) - Diferencias centradas
  - Angular: 3 puntos (j-1, j, j+1) - Diferencias centradas
  - Completo: 5 puntos (cruz en 2D)

Características:
  - Orden de aproximación: O(dr², dθ²) - Segundo orden
  - Tratamiento especial en r=0: Límite de L'Hôpital
  - Periodicidad angular: θ=0 ≡ θ=2π
  - Conserva simetría del operador

Malla:
  - Nodos radiales (nr):     {malla.nr}
  - Nodos angulares (nθ):    {malla.ntheta}
  - Total de nodos:          {malla.nr * malla.ntheta}
  - Paso radial (dr):        {malla.dr:.3e} m
  - Paso angular (dθ):       {malla.dtheta:.4f} rad ({np.degrees(malla.dtheta):.2f}°)

Stencil por nodo:
  - Nodos vecinos radiales:  2 (excepto r=0: solo 1)
  - Nodos vecinos angulares: 2 (con periodicidad)
  - Total de operaciones:    5 nodos × {malla.nr * malla.ntheta} = {5 * malla.nr * malla.ntheta}
    """

    return reporte


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
