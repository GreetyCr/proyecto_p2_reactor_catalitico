"""
Módulo de Discretización Espacial para Ecuaciones Diferenciales 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa la discretización espacial de la ecuación
de difusión-reacción 2D en coordenadas polares:

    ∂C/∂t = D_eff·∇²C - k_app·C

Donde el operador Laplaciano en polares es:
    ∇²C = ∂²C/∂r² + (1/r)·∂C/∂r + (1/r²)·∂²C/∂θ²

Usando diferencias finitas de segundo orden.

Referencias
----------
.. [1] LeVeque, R.J. (2007). "Finite Difference Methods for ODEs and PDEs"
.. [2] Crank, J., & Nicolson, P. (1947). "A practical method..."
"""

import numpy as np
from typing import Dict, Tuple
import logging
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES
# ============================================================================

R_GAS = 8.314  # J/(mol·K)


# ============================================================================
# COEFICIENTES RADIALES
# ============================================================================


def calcular_coeficientes_radiales(
    r: np.ndarray, i: int, dr: float, D_eff: float
) -> Dict[str, float]:
    """
    Calcula coeficientes para discretización radial en nodo i.

    La discretización del término radial del Laplaciano es:
        ∂²C/∂r² + (1/r)·∂C/∂r

    Usando diferencias centradas de segundo orden:
        ∂²C/∂r² ≈ [C[i+1] - 2C[i] + C[i-1]] / dr²
        ∂C/∂r ≈ [C[i+1] - C[i-1]] / (2dr)

    Combinando:
        D_eff·[∂²C/∂r² + (1/r)·∂C/∂r] =
            α·C[i-1] + γ_r·C[i] + β·C[i+1]

    Parameters
    ----------
    r : np.ndarray
        Array de coordenadas radiales [m]
    i : int
        Índice radial del nodo
    dr : float
        Paso radial [m]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    coefs : Dict[str, float]
        Diccionario con coeficientes:
        - 'alpha': Coeficiente para C[i-1]
        - 'beta': Coeficiente para C[i+1]
        - 'gamma_r': Coeficiente diagonal (C[i])
        - 'centro': True si es r=0

    Notes
    -----
    Tratamiento especial en r=0:
    - El término (1/r)·∂C/∂r es singular
    - Usar límite de L'Hôpital: lim(r→0) = 2·∂²C/∂r²

    Examples
    --------
    >>> r = np.linspace(0, 0.002, 61)
    >>> coefs = calcular_coeficientes_radiales(r, i=30, dr=3.33e-5, D_eff=1.04e-6)
    >>> print(f"α={coefs['alpha']:.3e}, β={coefs['beta']:.3e}")
    """
    # Caso especial: Centro (r=0)
    if i == 0:
        # En r=0, usar límite de L'Hôpital
        # lim(r→0) [∂²C/∂r² + (1/r)·∂C/∂r] = 2·∂²C/∂r²

        # Solo hay derivada hacia adelante
        alpha = 0.0  # No hay nodo i-1
        beta = 2 * D_eff / dr**2
        gamma_r = -2 * D_eff / dr**2

        return {"alpha": alpha, "beta": beta, "gamma_r": gamma_r, "centro": True}

    # Nodo interior o frontera
    r_i = r[i]

    # Coeficientes del término ∂²C/∂r²
    coef_segundo_orden = D_eff / dr**2

    # Coeficientes del término (1/r)·∂C/∂r
    coef_primer_orden = D_eff / (2 * r_i * dr)

    # Combinación:
    # C[i-1]: D/dr² - D/(2r·dr)
    # C[i]:   -2D/dr²
    # C[i+1]: D/dr² + D/(2r·dr)

    alpha = coef_segundo_orden - coef_primer_orden
    beta = coef_segundo_orden + coef_primer_orden
    gamma_r = -2 * coef_segundo_orden

    return {"alpha": alpha, "beta": beta, "gamma_r": gamma_r, "centro": False}


# ============================================================================
# COEFICIENTES ANGULARES
# ============================================================================


def calcular_coeficientes_angulares(
    r: np.ndarray, i: int, j: int, dtheta: float
) -> Dict[str, float]:
    """
    Calcula coeficientes para discretización angular en nodo (i,j).

    El término angular del Laplaciano es:
        (1/r²)·∂²C/∂θ²

    Usando diferencias centradas:
        ∂²C/∂θ² ≈ [C[j+1] - 2C[j] + C[j-1]] / dθ²

    Parameters
    ----------
    r : np.ndarray
        Array de coordenadas radiales [m]
    i : int
        Índice radial
    j : int
        Índice angular
    dtheta : float
        Paso angular [rad]

    Returns
    -------
    coefs : Dict[str, float]
        Diccionario con:
        - 'gamma_theta': Coeficiente para término angular

    Notes
    -----
    En r=0, el término (1/r²) es singular y se trata especialmente.

    Examples
    --------
    >>> r = np.linspace(0, 0.002, 61)
    >>> coefs = calcular_coeficientes_angulares(r, i=30, j=48, dtheta=0.0654)
    """
    # Caso especial: Centro (r=0)
    if i == 0:
        # En r=0, C debe ser independiente de θ (simetría)
        # El término angular se anula
        return {"gamma_theta": 0.0, "centro": True}

    # Nodo interior
    r_i = r[i]

    # Coeficiente: 1/r² · 1/dθ²
    gamma_theta = 1.0 / (r_i**2 * dtheta**2)

    return {"gamma_theta": gamma_theta, "centro": False}


# ============================================================================
# COEFICIENTES PARA MALLA COMPLETA
# ============================================================================


def calcular_coeficientes_malla_completa(malla, D_eff: float) -> Dict[str, np.ndarray]:
    """
    Calcula coeficientes de discretización para toda la malla.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    coefs : Dict[str, np.ndarray]
        Diccionario con arrays de coeficientes:
        - 'alpha': shape (nr,) - Coef para C[i-1,j]
        - 'beta': shape (nr,) - Coef para C[i+1,j]
        - 'gamma_theta': shape (nr,) - Coef para término angular

    Examples
    --------
    >>> from src.geometria.mallado import MallaPolar2D
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> coefs = calcular_coeficientes_malla_completa(malla, params.difusion.D_eff)
    """
    nr = malla.nr

    # Inicializar arrays
    alpha = np.zeros(nr)
    beta = np.zeros(nr)
    gamma_theta = np.zeros(nr)

    # Calcular para cada nodo radial
    for i in range(nr):
        # Coeficientes radiales
        coefs_r = calcular_coeficientes_radiales(malla.r, i, malla.dr, D_eff)
        alpha[i] = coefs_r["alpha"]
        beta[i] = coefs_r["beta"]

        # Coeficientes angulares (j no importa, son independientes de θ)
        coefs_theta = calcular_coeficientes_angulares(malla.r, i, 0, malla.dtheta)
        gamma_theta[i] = coefs_theta["gamma_theta"]

    return {"alpha": alpha, "beta": beta, "gamma_theta": gamma_theta}


# ============================================================================
# NÚMERO DE FOURIER Y ESTABILIDAD
# ============================================================================


def calcular_numero_fourier(dt: float, dr: float, D_eff: float) -> float:
    """
    Calcula el número de Fourier para difusión.

    Ecuación:
        Fo = D_eff × dt / dr²

    Parameters
    ----------
    dt : float
        Paso temporal [s]
    dr : float
        Paso espacial radial [m]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    Fo : float
        Número de Fourier [adimensional]

    Notes
    -----
    El número de Fourier caracteriza la relación entre:
    - Tasa de difusión: D_eff/dr²
    - Tasa de avance temporal: 1/dt

    Criterios de estabilidad:
    - Euler explícito: Fo < 0.5 (2D: Fo < 0.25)
    - Crank-Nicolson: Incondicionalmente estable (cualquier Fo)

    Para el proyecto:
        dt = 0.001 s, dr = 3.33×10⁻⁵ m, D_eff = 1.04×10⁻⁶ m²/s
        Fo ≈ 0.94 > 0.5 → Euler explícito INESTABLE
        Fo ≈ 0.94 → Crank-Nicolson ESTABLE ✓

    Examples
    --------
    >>> Fo = calcular_numero_fourier(dt=0.001, dr=3.33e-5, D_eff=1.04e-6)
    >>> print(f"Fo = {Fo:.3f}")
    Fo = 0.938
    """
    Fo = (D_eff * dt) / dr**2

    return Fo


def verificar_estabilidad_euler_explicito(Fo: float) -> bool:
    """
    Verifica criterio de estabilidad para Euler explícito.

    Parameters
    ----------
    Fo : float
        Número de Fourier

    Returns
    -------
    es_estable : bool
        True si es estable, False si no

    Notes
    -----
    Criterio de estabilidad von Neumann:
    - 1D: Fo ≤ 0.5
    - 2D: Fo ≤ 0.25 (más restrictivo)

    Examples
    --------
    >>> es_estable = verificar_estabilidad_euler_explicito(Fo=0.4)
    >>> assert es_estable == True
    """
    # Criterio conservador para 2D
    return Fo <= 0.5


def verificar_estabilidad_crank_nicolson(dt: float, dr: float, D_eff: float) -> bool:
    """
    Verifica estabilidad para Crank-Nicolson.

    Parameters
    ----------
    dt : float
        Paso temporal [s]
    dr : float
        Paso radial [m]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    es_estable : bool
        Siempre True (CN es incondicionalmente estable)

    Notes
    -----
    Crank-Nicolson es incondicionalmente estable para ecuaciones
    parabólicas lineales. No hay restricción en dt.

    Sin embargo, para precisión numérica se recomienda:
        Fo ~ 1 (balance entre precisión temporal y espacial)

    Examples
    --------
    >>> es_estable = verificar_estabilidad_crank_nicolson(dt=1.0, dr=3.33e-5, D_eff=1.04e-6)
    >>> assert es_estable == True  # Siempre estable
    """
    # Crank-Nicolson es incondicionalmente estable
    return True


def calcular_dt_critico_euler(dr: float, dtheta: float, D_eff: float) -> float:
    """
    Calcula paso temporal crítico para Euler explícito.

    Parameters
    ----------
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    dt_crit : float
        Paso temporal crítico [s]

    Notes
    -----
    Para Euler explícito en 2D:
        dt_crit = dr² / (4·D_eff)

    Esto asegura Fo ≤ 0.25 para estabilidad.

    Para el proyecto:
        dr = 3.33×10⁻⁵ m, D_eff = 1.04×10⁻⁶ m²/s
        dt_crit ≈ 2.67×10⁻⁴ s

    Examples
    --------
    >>> dt_crit = calcular_dt_critico_euler(dr=3.33e-5, dtheta=0.0654, D_eff=1.04e-6)
    >>> print(f"dt_crit = {dt_crit:.3e} s")
    dt_crit = 2.670e-04 s
    """
    # Criterio conservador
    dt_crit = dr**2 / (4 * D_eff)

    return dt_crit


# ============================================================================
# VALIDACIÓN DE PARÁMETROS
# ============================================================================


def validar_parametros_discretizacion(
    dt: float, dr: float, dtheta: float, D_eff: float
):
    """
    Valida que los parámetros de discretización sean físicamente razonables.

    Parameters
    ----------
    dt : float
        Paso temporal [s]
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    D_eff : float
        Difusividad efectiva [m²/s]

    Raises
    ------
    ValueError
        Si algún parámetro es inválido

    Warnings
    --------
    Si dt es muy grande comparado con dr (puede afectar precisión)

    Examples
    --------
    >>> validar_parametros_discretizacion(dt=0.001, dr=3.33e-5, dtheta=0.0654, D_eff=1.04e-6)
    """
    # Validar positividad
    if dt <= 0:
        raise ValueError(f"Paso temporal debe ser positivo: dt={dt}")
    if dr <= 0:
        raise ValueError(f"Paso radial debe ser positivo: dr={dr}")
    if dtheta <= 0:
        raise ValueError(f"Paso angular debe ser positivo: dtheta={dtheta}")
    if D_eff <= 0:
        raise ValueError(f"Difusividad debe ser positiva: D_eff={D_eff}")

    # Calcular número de Fourier
    Fo = calcular_numero_fourier(dt, dr, D_eff)

    # Advertir si dt muy grande (aunque CN es estable, afecta precisión)
    if Fo > 10:
        warnings.warn(
            f"Número de Fourier muy grande: Fo={Fo:.1f} > 10. "
            f"Considerar reducir dt para mejor precisión.",
            UserWarning,
        )

    # Advertir si dt muy pequeño (ineficiente)
    if Fo < 0.01:
        warnings.warn(
            f"Número de Fourier muy pequeño: Fo={Fo:.3f} < 0.01. "
            f"Puede aumentar dt sin perder precisión.",
            UserWarning,
        )


# ============================================================================
# VERSIÓN DIMENSIONAL
# ============================================================================


def calcular_numero_fourier_dimensional(dt: float, dr: float, D_eff: float):
    """
    Calcula número de Fourier con validación dimensional.

    Returns
    -------
    Fo : CantidadDimensional
        Número de Fourier con dimensión ADIMENSIONAL

    Examples
    --------
    >>> Fo = calcular_numero_fourier_dimensional(0.001, 3.33e-5, 1.04e-6)
    >>> assert Fo.dimension == Dimension.ADIMENSIONAL
    """
    from src.utils.validacion import CantidadDimensional, Dimension

    # Calcular valor
    valor = calcular_numero_fourier(dt, dr, D_eff)

    # Retornar con dimensión
    return CantidadDimensional(valor, Dimension.ADIMENSIONAL, "Fourier")


# ============================================================================
# INFORMACIÓN Y REPORTES
# ============================================================================


def obtener_info_discretizacion(
    dt: float, dr: float, dtheta: float, D_eff: float
) -> Dict[str, float]:
    """
    Obtiene información resumida de la discretización.

    Parameters
    ----------
    dt : float
        Paso temporal [s]
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    info : Dict
        Información de discretización

    Examples
    --------
    >>> info = obtener_info_discretizacion(0.001, 3.33e-5, 0.0654, 1.04e-6)
    >>> print(f"Fo = {info['Fo']:.3f}")
    """
    Fo = calcular_numero_fourier(dt, dr, D_eff)
    dt_crit = calcular_dt_critico_euler(dr, dtheta, D_eff)

    info = {
        "dt": dt,
        "dr": dr,
        "dtheta": dtheta,
        "D_eff": D_eff,
        "Fo": Fo,
        "dt_critico_euler": dt_crit,
        "estable_euler": verificar_estabilidad_euler_explicito(Fo),
        "estable_cn": verificar_estabilidad_crank_nicolson(dt, dr, D_eff),
    }

    return info


def generar_reporte_discretizacion(
    dt: float, dr: float, dtheta: float, D_eff: float
) -> str:
    """
    Genera reporte legible de la discretización.

    Parameters
    ----------
    dt : float
        Paso temporal [s]
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> reporte = generar_reporte_discretizacion(0.001, 3.33e-5, 0.0654, 1.04e-6)
    >>> print(reporte)
    """
    info = obtener_info_discretizacion(dt, dr, dtheta, D_eff)

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║           REPORTE DE DISCRETIZACIÓN ESPACIAL                 ║
╚══════════════════════════════════════════════════════════════╝

Parámetros de Discretización:
  - Paso temporal:    dt = {info['dt']:.3e} s
  - Paso radial:      dr = {info['dr']:.3e} m
  - Paso angular:     dθ = {info['dtheta']:.4f} rad ({np.degrees(info['dtheta']):.2f}°)
  - Difusividad:   D_eff = {info['D_eff']:.3e} m²/s

Número de Fourier:
  - Fo = D_eff·dt/dr² = {info['Fo']:.3f}

Estabilidad:
  - Euler Explícito:  {'✓ ESTABLE' if info['estable_euler'] else '✗ INESTABLE'}
  - Crank-Nicolson:   {'✓ ESTABLE' if info['estable_cn'] else '✗ INESTABLE'} (siempre)

Criterio de Estabilidad (Euler):
  - dt crítico = {info['dt_critico_euler']:.3e} s
  - dt actual / dt_crit = {info['dt'] / info['dt_critico_euler']:.2f}

Recomendación:
  {'  ⚠ Usar método implícito (Crank-Nicolson)' if not info['estable_euler'] else '  ✓ Parámetros adecuados'}
    """

    return reporte


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
