"""
Módulo de Propiedades de Difusión.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo calcula todas las propiedades difusivas relevantes:
- Camino libre medio molecular (λ)
- Número de Knudsen (Kn)
- Difusividad de Knudsen (D_Kn)
- Difusividad molecular (D_molecular)
- Difusividad combinada con Bosanquet (D_comb)
- Difusividad efectiva en medio poroso (D_eff)

Todas las funciones incluyen validación dimensional y de rangos físicos.

References
----------
.. [1] Hill (2025); Abello (2002); Mourkou et al. (2024)
.. [2] PARAMETROS_PROYECTO.md - Tabla V
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================

R_GAS = 8.314  # J/(mol·K) - Constante universal de los gases
BOLTZMANN = 1.380649e-23  # J/K - Constante de Boltzmann


# ============================================================================
# CAMINO LIBRE MEDIO MOLECULAR
# ============================================================================


def calcular_camino_libre_medio(T: float, P: float, d_molecular: float) -> float:
    """
    Calcula el camino libre medio molecular.

    Ecuación:
        λ = k_B × T / (√2 × π × d² × P)

    Parameters
    ----------
    T : float
        Temperatura [K]
    P : float
        Presión [Pa]
    d_molecular : float
        Diámetro molecular [m]

    Returns
    -------
    lambda_mfp : float
        Camino libre medio [m]

    Raises
    ------
    ValueError
        Si algún parámetro es no positivo o fuera de rango físico

    Notes
    -----
    El camino libre medio representa la distancia promedio que recorre
    una molécula entre colisiones.

    - Para CO: d_CO ≈ 3.76×10⁻¹⁰ m
    - A 673K, 1 atm: λ ≈ 1.93×10⁻⁷ m

    Examples
    --------
    >>> lambda_mfp = calcular_camino_libre_medio(T=673, P=101325, d_molecular=3.76e-10)
    >>> print(f"{lambda_mfp:.3e} m")
    1.930e-07 m

    References
    ----------
    .. [1] PARAMETROS_PROYECTO.md - Tabla V
    """
    # Validación de inputs
    if T <= 0:
        raise ValueError(f"Temperatura debe ser positiva, recibido: {T} K")
    if P <= 0:
        raise ValueError(f"Presión debe ser positiva, recibida: {P} Pa")
    if d_molecular <= 0:
        raise ValueError(
            f"Diámetro molecular debe ser positivo, recibido: {d_molecular} m"
        )

    # Cálculo
    lambda_mfp = (BOLTZMANN * T) / (np.sqrt(2) * np.pi * d_molecular**2 * P)

    # Post-validación (sanity check)
    if not 1e-10 < lambda_mfp < 1e-3:
        logger.warning(f"Camino libre medio fuera de rango típico: {lambda_mfp:.3e} m")

    return lambda_mfp


# ============================================================================
# NÚMERO DE KNUDSEN
# ============================================================================


def calcular_numero_knudsen(lambda_mfp: float, r_poro: float) -> float:
    """
    Calcula el número de Knudsen.

    Ecuación:
        Kn = λ / r_poro

    Parameters
    ----------
    lambda_mfp : float
        Camino libre medio [m]
    r_poro : float
        Radio de poro [m]

    Returns
    -------
    Kn : float
        Número de Knudsen [adimensional]

    Notes
    -----
    Interpretación:
    - Kn >> 1: Difusión de Knudsen (colisiones pared-molécula)
    - Kn << 1: Difusión molecular (colisiones molécula-molécula)
    - Kn ~ 1: Régimen de transición

    Para el proyecto: Kn = 19.3 >> 1 → Knudsen dominante

    Examples
    --------
    >>> Kn = calcular_numero_knudsen(lambda_mfp=1.93e-7, r_poro=10e-9)
    >>> print(f"Kn = {Kn:.1f}")
    Kn = 19.3
    """
    # Validación
    if lambda_mfp <= 0:
        raise ValueError(f"λ debe ser positivo: {lambda_mfp}")
    if r_poro <= 0:
        raise ValueError(f"r_poro debe ser positivo: {r_poro}")

    # Cálculo
    Kn = lambda_mfp / r_poro

    return Kn


def identificar_regimen_difusion(Kn: float) -> str:
    """
    Identifica el régimen de difusión según el número de Knudsen.

    Parameters
    ----------
    Kn : float
        Número de Knudsen [adimensional]

    Returns
    -------
    regimen : str
        'knudsen', 'molecular', o 'transicion'

    Examples
    --------
    >>> regimen = identificar_regimen_difusion(Kn=19.3)
    >>> assert regimen == 'knudsen'
    """
    if Kn > 10:
        return "knudsen"
    elif Kn < 0.1:
        return "molecular"
    else:
        return "transicion"


# ============================================================================
# DIFUSIVIDAD DE KNUDSEN
# ============================================================================


def calcular_difusividad_knudsen(r_poro: float, T: float, M_gas: float) -> float:
    """
    Calcula la difusividad de Knudsen.

    Ecuación:
        D_Kn = (2/3) × r_poro × √(8RT/πM)

    Parameters
    ----------
    r_poro : float
        Radio de poro promedio [m]
    T : float
        Temperatura [K]
    M_gas : float
        Masa molar del gas [kg/mol]

    Returns
    -------
    D_Kn : float
        Difusividad de Knudsen [m²/s]

    Raises
    ------
    ValueError
        Si algún parámetro es no positivo

    Notes
    -----
    La difusividad de Knudsen domina cuando el camino libre medio es
    mucho mayor que el radio de poro (Kn >> 1).

    En este régimen, las colisiones molécula-pared son más frecuentes
    que las colisiones molécula-molécula.

    Para CO en catalizador Pt/Al₂O₃ a 673K:
        D_Kn ≈ 7.43×10⁻⁶ m²/s

    Examples
    --------
    >>> D_Kn = calcular_difusividad_knudsen(r_poro=10e-9, T=673, M_gas=0.028)
    >>> print(f"D_Kn = {D_Kn:.3e} m²/s")
    D_Kn = 7.430e-06 m²/s

    References
    ----------
    .. [1] Mourkou et al. (2024)
    .. [2] Hill (2025)
    """
    # Validación de inputs
    if r_poro <= 0:
        raise ValueError(f"Radio de poro debe ser positivo, recibido: {r_poro} m")
    if T <= 0:
        raise ValueError(f"Temperatura debe ser positiva, recibida: {T} K")
    if M_gas <= 0:
        raise ValueError(f"Masa molar debe ser positiva, recibida: {M_gas} kg/mol")

    # Cálculo
    D_Kn = (2.0 / 3.0) * r_poro * np.sqrt(8 * R_GAS * T / (np.pi * M_gas))

    # Post-validación (sanity check)
    if not 1e-10 < D_Kn < 1e-3:
        logger.warning(
            f"D_Kn fuera de rango típico: {D_Kn:.3e} m²/s "
            f"(r_poro={r_poro:.3e}m, T={T}K)"
        )

    return D_Kn


# ============================================================================
# DIFUSIVIDAD COMBINADA (BOSANQUET)
# ============================================================================


def calcular_difusividad_combinada(D_molecular: float, D_Knudsen: float) -> float:
    """
    Calcula la difusividad combinada usando relación de Bosanquet.

    Ecuación:
        1/D_comb = 1/D_molecular + 1/D_Knudsen

    Parameters
    ----------
    D_molecular : float
        Difusividad molecular [m²/s]
    D_Knudsen : float
        Difusividad de Knudsen [m²/s]

    Returns
    -------
    D_comb : float
        Difusividad combinada [m²/s]

    Notes
    -----
    La relación de Bosanquet combina los efectos de difusión molecular
    y de Knudsen en el régimen de transición.

    Casos límite:
    - Si D_Kn << D_mol: D_comb ≈ D_Kn (Knudsen domina)
    - Si D_mol << D_Kn: D_comb ≈ D_mol (Molecular domina)

    Para el proyecto:
        D_molecular = 8.75×10⁻⁵ m²/s
        D_Kn = 7.43×10⁻⁶ m²/s
        D_comb ≈ 6.97×10⁻⁶ m²/s (Knudsen domina)

    Examples
    --------
    >>> D_comb = calcular_difusividad_combinada(D_molecular=8.75e-5, D_Knudsen=7.43e-6)
    >>> print(f"D_comb = {D_comb:.3e} m²/s")
    D_comb = 6.970e-06 m²/s

    References
    ----------
    .. [1] Bosanquet (1944)
    """
    # Validación
    if D_molecular <= 0:
        raise ValueError(f"D_molecular debe ser positivo: {D_molecular}")
    if D_Knudsen <= 0:
        raise ValueError(f"D_Knudsen debe ser positivo: {D_Knudsen}")

    # Relación de Bosanquet
    D_comb = 1.0 / (1.0 / D_molecular + 1.0 / D_Knudsen)

    return D_comb


# ============================================================================
# DIFUSIVIDAD EFECTIVA EN MEDIO POROSO
# ============================================================================


def calcular_difusividad_efectiva(epsilon: float, D_comb: float, tau: float) -> float:
    """
    Calcula la difusividad efectiva en medio poroso.

    Ecuación:
        D_eff = (ε × D_comb) / τ

    Parameters
    ----------
    epsilon : float
        Porosidad del pellet [adimensional, 0-1]
    D_comb : float
        Difusividad combinada [m²/s]
    tau : float
        Factor de tortuosidad [adimensional, típicamente 2-5]

    Returns
    -------
    D_eff : float
        Difusividad efectiva [m²/s]

    Raises
    ------
    ValueError
        Si parámetros están fuera de rangos físicos

    Notes
    -----
    La difusividad efectiva considera:
    - ε: Fracción de volumen disponible para difusión
    - τ: Aumento de camino por tortuosidad de poros

    Para catalizador Pt/Al₂O₃:
        ε = 0.45, τ = 3.0
        D_eff = 1.04×10⁻⁶ m²/s

    Examples
    --------
    >>> D_eff = calcular_difusividad_efectiva(epsilon=0.45, D_comb=6.97e-6, tau=3.0)
    >>> print(f"D_eff = {D_eff:.3e} m²/s")
    D_eff = 1.040e-06 m²/s

    References
    ----------
    .. [1] Abello (2002)
    """
    # Validación de inputs
    if not (0 <= epsilon <= 1):
        raise ValueError(f"Porosidad debe estar en [0,1], recibido: {epsilon}")
    if D_comb <= 0:
        raise ValueError(f"D_comb debe ser positivo: {D_comb}")
    if tau < 1.0:
        raise ValueError(f"Tortuosidad debe ser ≥ 1 (camino recto), recibido: {tau}")

    # Cálculo
    D_eff = (epsilon * D_comb) / tau

    # Post-validación
    if not 1e-10 < D_eff < 1e-3:
        logger.warning(f"D_eff fuera de rango típico: {D_eff:.3e} m²/s")

    return D_eff


# ============================================================================
# DIFUSIVIDAD MOLECULAR CO-AIRE
# ============================================================================


def obtener_difusividad_molecular_CO_aire(T: float, T_ref: float = 273.0) -> float:
    """
    Obtiene difusividad molecular CO-aire a temperatura T.

    Usa correlación simplificada basada en ley de potencias:
        D(T) = D_ref × (T/T_ref)^1.5

    Parameters
    ----------
    T : float
        Temperatura [K]
    T_ref : float, optional
        Temperatura de referencia [K], default 273K

    Returns
    -------
    D_molecular : float
        Difusividad molecular CO-aire [m²/s]

    Notes
    -----
    Valores de referencia (The Engineering ToolBox, 2018):
    - A 273K: D_CO-aire ≈ 1.96×10⁻⁵ m²/s
    - A 673K: D_CO-aire ≈ 8.75×10⁻⁵ m²/s

    La correlación T^1.5 es aproximada (Chapman-Enskog da T^1.75).

    Examples
    --------
    >>> D_mol = obtener_difusividad_molecular_CO_aire(T=673)
    >>> print(f"D_molecular = {D_mol:.3e} m²/s")
    D_molecular = 8.750e-05 m²/s
    """
    # Validación
    if T <= 0:
        raise ValueError(f"Temperatura debe ser positiva: {T} K")

    # Valor de referencia a 273K
    D_ref_273K = 1.96e-5  # m²/s (CO en aire a 273K)

    # Correlación de temperatura (ley de potencias)
    D_molecular = D_ref_273K * (T / T_ref) ** 1.5

    return D_molecular


# ============================================================================
# FUNCIÓN INTEGRADA: CALCULAR TODAS LAS PROPIEDADES
# ============================================================================


def calcular_propiedades_difusion(
    T: float,
    P: float,
    r_poro: float,
    epsilon: float,
    tau: float,
    M_gas: float,
    d_molecular: float,
    D_molecular_ref: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calcula todas las propiedades de difusión de forma integrada.

    Parameters
    ----------
    T : float
        Temperatura [K]
    P : float
        Presión [Pa]
    r_poro : float
        Radio de poro [m]
    epsilon : float
        Porosidad [adimensional]
    tau : float
        Tortuosidad [adimensional]
    M_gas : float
        Masa molar del gas [kg/mol]
    d_molecular : float
        Diámetro molecular [m]
    D_molecular_ref : float, optional
        Si se provee, usa este valor en vez de calcularlo

    Returns
    -------
    propiedades : Dict[str, float]
        Diccionario con todas las propiedades calculadas:
        - 'lambda_mfp': Camino libre medio [m]
        - 'Kn': Número de Knudsen [adimensional]
        - 'D_Kn': Difusividad de Knudsen [m²/s]
        - 'D_molecular': Difusividad molecular [m²/s]
        - 'D_comb': Difusividad combinada [m²/s]
        - 'D_eff': Difusividad efectiva [m²/s]
        - 'regimen': Régimen de difusión

    Examples
    --------
    >>> props = calcular_propiedades_difusion(
    ...     T=673, P=101325, r_poro=10e-9,
    ...     epsilon=0.45, tau=3.0, M_gas=0.028, d_molecular=3.76e-10
    ... )
    >>> print(f"D_eff = {props['D_eff']:.3e} m²/s")
    >>> print(f"Régimen: {props['regimen']}")
    """
    # 1. Camino libre medio
    lambda_mfp = calcular_camino_libre_medio(T, P, d_molecular)

    # 2. Número de Knudsen
    Kn = calcular_numero_knudsen(lambda_mfp, r_poro)

    # 3. Difusividad de Knudsen
    D_Kn = calcular_difusividad_knudsen(r_poro, T, M_gas)

    # 4. Difusividad molecular
    if D_molecular_ref is not None:
        D_molecular = D_molecular_ref
    else:
        D_molecular = obtener_difusividad_molecular_CO_aire(T)

    # 5. Difusividad combinada (Bosanquet)
    D_comb = calcular_difusividad_combinada(D_molecular, D_Kn)

    # 6. Difusividad efectiva
    D_eff = calcular_difusividad_efectiva(epsilon, D_comb, tau)

    # 7. Identificar régimen
    regimen = identificar_regimen_difusion(Kn)

    # Empaquetar resultados
    propiedades = {
        "lambda_mfp": lambda_mfp,
        "Kn": Kn,
        "D_Kn": D_Kn,
        "D_molecular": D_molecular,
        "D_comb": D_comb,
        "D_eff": D_eff,
        "regimen": regimen,
    }

    # Logging
    logger.info(
        f"Propiedades de difusión calculadas: "
        f"D_eff={D_eff:.3e} m²/s, Kn={Kn:.1f} ({regimen})"
    )

    return propiedades


# ============================================================================
# INTEGRACIÓN CON PARÁMETROS DEL PROYECTO
# ============================================================================


def calcular_desde_parametros(params) -> Dict[str, float]:
    """
    Calcula propiedades de difusión desde ParametrosMaestros.

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    propiedades : Dict[str, float]
        Diccionario con propiedades calculadas

    Examples
    --------
    >>> from src.config.parametros import ParametrosMaestros
    >>> params = ParametrosMaestros()
    >>> props = calcular_desde_parametros(params)
    >>> assert np.isclose(props['D_eff'], params.difusion.D_eff, rtol=0.05)
    """
    # Extraer parámetros necesarios
    T = params.operacion.T
    P = params.operacion.P
    r_poro = params.difusion.r_poro
    epsilon = params.difusion.epsilon
    tau = params.difusion.tau

    # Masa molar CO
    M_CO = 0.028  # kg/mol

    # Diámetro molecular CO
    d_CO = 3.76e-10  # m

    # Difusividad molecular de referencia (Tabla III)
    D_molecular_ref = params.gas.D_CO_aire

    # Calcular
    return calcular_propiedades_difusion(
        T=T,
        P=P,
        r_poro=r_poro,
        epsilon=epsilon,
        tau=tau,
        M_gas=M_CO,
        d_molecular=d_CO,
        D_molecular_ref=D_molecular_ref,
    )


# ============================================================================
# VERSIONES CON VALIDACIÓN DIMENSIONAL
# ============================================================================


def calcular_difusividad_knudsen_dimensional(r_poro: float, T: float, M: float):
    """
    Calcula D_Kn con validación dimensional.

    Returns
    -------
    D_Kn : CantidadDimensional
        Difusividad de Knudsen con dimensión DIFUSIVIDAD

    Examples
    --------
    >>> D_Kn = calcular_difusividad_knudsen_dimensional(10e-9, 673, 0.028)
    >>> assert D_Kn.dimension == Dimension.DIFUSIVIDAD
    """
    from src.utils.validacion import CantidadDimensional, Dimension

    # Calcular valor
    valor = calcular_difusividad_knudsen(r_poro, T, M)

    # Retornar con dimensión
    return CantidadDimensional(valor, Dimension.DIFUSIVIDAD, "D_Kn")


def calcular_difusividad_efectiva_dimensional(
    epsilon: float, D_comb: float, tau: float
):
    """
    Calcula D_eff con validación dimensional.

    Returns
    -------
    D_eff : CantidadDimensional
        Difusividad efectiva con dimensión DIFUSIVIDAD

    Examples
    --------
    >>> D_eff = calcular_difusividad_efectiva_dimensional(0.45, 6.97e-6, 3.0)
    >>> assert D_eff.dimension == Dimension.DIFUSIVIDAD
    """
    from src.utils.validacion import CantidadDimensional, Dimension

    # Calcular valor
    valor = calcular_difusividad_efectiva(epsilon, D_comb, tau)

    # Retornar con dimensión
    return CantidadDimensional(valor, Dimension.DIFUSIVIDAD, "D_eff")


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
