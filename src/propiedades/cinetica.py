"""
Módulo de Cinética de Reacción.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa la cinética de reacción catalítica:
- Ecuación de Arrhenius para dependencia con temperatura
- Campo espacial de k_app(r,θ) considerando región de defecto
- Cálculo de tasas de reacción
- Integración con otros módulos

Reacción: CO + ½O₂ → CO₂  (sobre catalizador Pt/Al₂O₃)
Cinética: Primer orden en CO

References
----------
.. [1] Hill (2025); Abello (2002)
.. [2] PARAMETROS_PROYECTO.md - Tabla VI
"""

import numpy as np
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================

R_GAS = 8.314  # J/(mol·K) - Constante universal de los gases


# ============================================================================
# ECUACIÓN DE ARRHENIUS
# ============================================================================


def calcular_k_arrhenius(k0: float, Ea: float, T: float) -> float:
    """
    Calcula la constante cinética usando la ecuación de Arrhenius.

    Ecuación:
        k = k₀ × exp(-Ea/RT)

    Parameters
    ----------
    k0 : float
        Factor pre-exponencial [s⁻¹]
    Ea : float
        Energía de activación [J/mol]
    T : float
        Temperatura [K]

    Returns
    -------
    k : float
        Constante cinética [s⁻¹]

    Raises
    ------
    ValueError
        Si algún parámetro es no positivo o fuera de rango físico

    Notes
    -----
    La ecuación de Arrhenius describe la dependencia de la constante
    cinética con la temperatura:

    - k₀: Frecuencia de colisiones (factor pre-exponencial)
    - Ea: Barrera energética para que ocurra la reacción
    - T: Temperatura absoluta

    A mayor temperatura, más moléculas tienen energía suficiente
    para superar la barrera Ea, aumentando k exponencialmente.

    Para CO sobre Pt/Al₂O₃ a 673K:
        k₀ = 60 s⁻¹, Ea = 26 kJ/mol → k_app ≈ 4.0×10⁻³ s⁻¹

    Examples
    --------
    >>> k = calcular_k_arrhenius(k0=60, Ea=26000, T=673)
    >>> print(f"k = {k:.3e} s⁻¹")
    k = 4.000e-03 s⁻¹

    References
    ----------
    .. [1] Arrhenius, S. (1889). "Über die Reaktionsgeschwindigkeit..."
    .. [2] PARAMETROS_PROYECTO.md - Tabla VI
    """
    # Validación de inputs
    if k0 <= 0:
        raise ValueError(
            f"Factor pre-exponencial debe ser positivo, recibido: {k0} s⁻¹"
        )
    if Ea < 0:
        raise ValueError(
            f"Energía de activación debe ser no negativa, recibida: {Ea} J/mol"
        )
    if T <= 0:
        raise ValueError(f"Temperatura debe ser positiva, recibida: {T} K")

    # Ecuación de Arrhenius
    k = k0 * np.exp(-Ea / (R_GAS * T))

    # Post-validación (sanity check)
    if not 0 < k < 1e6:
        logger.warning(f"Constante cinética fuera de rango típico: k={k:.3e} s⁻¹")

    return k


# ============================================================================
# CAMPO ESPACIAL DE k_app
# ============================================================================


def generar_campo_k_app(malla, params) -> np.ndarray:
    """
    Genera campo espacial de constante cinética aparente k_app(r,θ).

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D del dominio
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    k_app_field : np.ndarray, shape (nr, ntheta)
        Campo de k_app [s⁻¹]: 0 en defecto, k_app en activa

    Notes
    -----
    La distribución espacial de k_app refleja el defecto catalítico:

    k_app(r,θ) = {
        0               si (r,θ) en región de defecto
        k_app_param     si (r,θ) en región activa
    }

    Región de defecto:
    - Radial: r ∈ [R/3, 2R/3]
    - Angular: θ ∈ [0°, 45°]

    Examples
    --------
    >>> from src.config.parametros import ParametrosMaestros
    >>> from src.geometria.mallado import MallaPolar2D
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> k_field = generar_campo_k_app(malla, params)
    >>> assert k_field.shape == (61, 96)
    """
    # Usar método de la malla (ya implementado y testeado)
    k_app_field = malla.generar_campo_k_app()

    logger.debug(
        f"Campo k_app generado: " f"activa={params.cinetica.k_app:.3e} s⁻¹, defecto=0"
    )

    return k_app_field


def generar_campo_k_app_temperatura(malla, params, T: float) -> np.ndarray:
    """
    Genera campo k_app recalculando con temperatura diferente.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D
    params : ParametrosMaestros
        Parámetros del proyecto
    T : float
        Temperatura [K] para recalcular k_app

    Returns
    -------
    k_app_field : np.ndarray, shape (nr, ntheta)
        Campo de k_app recalculado a temperatura T [s⁻¹]

    Notes
    -----
    Recalcula k_app usando Arrhenius a la temperatura especificada:
    k_app(T) = k₀ × exp(-Ea/RT)

    Útil para análisis de sensibilidad o perfiles de temperatura
    no uniformes.

    Examples
    --------
    >>> k_field_700K = generar_campo_k_app_temperatura(malla, params, T=700)
    >>> # k debe ser mayor a 700K que a 673K
    """
    # Recalcular k_app a la temperatura T
    k_app_T = calcular_k_arrhenius(k0=params.cinetica.k0, Ea=params.cinetica.Ea, T=T)

    # Inicializar campo con nuevo valor
    k_app_field = np.full((malla.nr, malla.ntheta), k_app_T)

    # Aplicar ceros en región de defecto
    mascara_defecto = malla.identificar_region_defecto()
    k_app_field[mascara_defecto] = 0.0

    logger.debug(f"Campo k_app a T={T}K: " f"activa={k_app_T:.3e} s⁻¹, defecto=0")

    return k_app_field


def generar_campo_k_app_temperatura_espacial(
    malla, params, T_field: np.ndarray
) -> np.ndarray:
    """
    Genera campo k_app con temperatura espacialmente variable.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D
    params : ParametrosMaestros
        Parámetros del proyecto
    T_field : np.ndarray, shape (nr, ntheta)
        Campo de temperatura [K]

    Returns
    -------
    k_app_field : np.ndarray, shape (nr, ntheta)
        Campo de k_app con dependencia espacial de T [s⁻¹]

    Notes
    -----
    Calcula k_app(r,θ) considerando que la temperatura puede variar
    espacialmente: T = T(r,θ)

    Cada nodo tiene: k_app[i,j] = k₀ × exp(-Ea/(R×T[i,j]))

    Útil para problemas no isotérmicos con gradientes de temperatura.

    Examples
    --------
    >>> T_field = np.full((61, 96), 673.0)  # Uniforme
    >>> k_field = generar_campo_k_app_temperatura_espacial(malla, params, T_field)
    """
    # Validar shape
    if T_field.shape != (malla.nr, malla.ntheta):
        raise ValueError(
            f"T_field.shape {T_field.shape} no coincide con malla "
            f"({malla.nr}, {malla.ntheta})"
        )

    # Inicializar campo
    k_app_field = np.zeros_like(T_field)

    # Calcular k_app en cada nodo usando Arrhenius
    for i in range(malla.nr):
        for j in range(malla.ntheta):
            k_app_field[i, j] = calcular_k_arrhenius(
                k0=params.cinetica.k0, Ea=params.cinetica.Ea, T=T_field[i, j]
            )

    # Aplicar ceros en región de defecto
    mascara_defecto = malla.identificar_region_defecto()
    k_app_field[mascara_defecto] = 0.0

    logger.debug(
        f"Campo k_app con T variable: "
        f"min={np.min(k_app_field):.3e}, "
        f"max={np.max(k_app_field):.3e} s⁻¹"
    )

    return k_app_field


# ============================================================================
# TASA DE REACCIÓN
# ============================================================================


def calcular_tasa_reaccion(
    k_app: Union[float, np.ndarray], C: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calcula tasa de reacción volumétrica (cinética de primer orden).

    Ecuación:
        r = k_app × C

    Parameters
    ----------
    k_app : float or np.ndarray
        Constante cinética aparente [s⁻¹]
    C : float or np.ndarray
        Concentración de CO [mol/m³]

    Returns
    -------
    r : float or np.ndarray
        Tasa de reacción volumétrica [mol/(m³·s)]

    Notes
    -----
    La reacción es de primer orden en CO:
        CO + ½O₂ → CO₂

    La tasa de reacción es:
        r = k_app × C_CO

    Donde:
    - k_app: Constante cinética aparente (incluye [O₂])
    - C_CO: Concentración de monóxido de carbono

    En la región de defecto: k_app = 0 → r = 0

    Examples
    --------
    >>> r = calcular_tasa_reaccion(k_app=4.0e-3, C=0.01)
    >>> print(f"r = {r:.3e} mol/(m³·s)")
    r = 4.000e-05 mol/(m³·s)

    >>> # Para campos completos:
    >>> k_field = malla.generar_campo_k_app()
    >>> C_field = np.ones((61, 96)) * 0.01
    >>> r_field = calcular_tasa_reaccion(k_field, C_field)
    """
    # Multiplicación element-wise (funciona para escalares y arrays)
    r = k_app * C

    return r


# ============================================================================
# INTEGRACIÓN CON PARÁMETROS
# ============================================================================


def calcular_k_desde_parametros(params) -> float:
    """
    Calcula k_app desde ParametrosMaestros.

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    k_app : float
        Constante cinética aparente [s⁻¹]

    Examples
    --------
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> k = calcular_k_desde_parametros(params)
    >>> assert np.isclose(k, params.cinetica.k_app, rtol=0.05)
    """
    k_app = calcular_k_arrhenius(
        k0=params.cinetica.k0, Ea=params.cinetica.Ea, T=params.operacion.T
    )

    return k_app


def generar_campo_desde_parametros(params) -> np.ndarray:
    """
    Genera campo k_app completo desde ParametrosMaestros.

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    k_app_field : np.ndarray, shape (nr, ntheta)
        Campo de k_app [s⁻¹]

    Examples
    --------
    >>> params = ParametrosMaestros()
    >>> k_field = generar_campo_desde_parametros(params)
    >>> assert k_field.shape == (61, 96)
    """
    # Crear malla
    from src.geometria.mallado import MallaPolar2D

    malla = MallaPolar2D(params)

    # Generar campo
    k_app_field = generar_campo_k_app(malla, params)

    return k_app_field


# ============================================================================
# VERSIONES CON VALIDACIÓN DIMENSIONAL
# ============================================================================


def calcular_k_arrhenius_dimensional(k0: float, Ea: float, T: float):
    """
    Calcula k con validación dimensional.

    Returns
    -------
    k : CantidadDimensional
        Constante cinética con dimensión FRECUENCIA [1/T]

    Examples
    --------
    >>> k = calcular_k_arrhenius_dimensional(60, 26000, 673)
    >>> assert k.dimension == Dimension.FRECUENCIA
    """
    from src.utils.validacion import CantidadDimensional, Dimension

    # Calcular valor
    valor = calcular_k_arrhenius(k0, Ea, T)

    # Retornar con dimensión
    return CantidadDimensional(valor, Dimension.FRECUENCIA, "k_app")


def calcular_tasa_reaccion_dimensional(k_app: float, C: float):
    """
    Calcula tasa de reacción con validación dimensional.

    Returns
    -------
    r : CantidadDimensional
        Tasa de reacción con dimensión [N/(L³·T)]

    Examples
    --------
    >>> r = calcular_tasa_reaccion_dimensional(4.0e-3, 0.01)
    >>> assert r.dimension == Dimension.TASA_REACCION_VOLUMETRICA
    """
    from src.utils.validacion import CantidadDimensional, Dimension

    # Calcular valor
    valor = calcular_tasa_reaccion(k_app, C)

    # Retornar con dimensión
    return CantidadDimensional(
        valor, Dimension.TASA_REACCION_VOLUMETRICA, "r_volumetrica"
    )


# ============================================================================
# INFORMACIÓN Y UTILIDADES
# ============================================================================


def obtener_info_cinetica(params) -> Dict[str, float]:
    """
    Obtiene información resumida de la cinética.

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    info : Dict[str, float]
        Diccionario con información de cinética

    Examples
    --------
    >>> info = obtener_info_cinetica(params)
    >>> print(f"k_app = {info['k_app']:.3e} s⁻¹")
    """
    # Calcular k_app
    k_app = calcular_k_desde_parametros(params)

    info = {
        "k0": params.cinetica.k0,
        "Ea": params.cinetica.Ea,
        "T": params.operacion.T,
        "k_app": k_app,
        "k_app_tabla": params.cinetica.k_app,
        "diferencia_pct": abs(k_app - params.cinetica.k_app)
        / params.cinetica.k_app
        * 100,
    }

    return info


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
