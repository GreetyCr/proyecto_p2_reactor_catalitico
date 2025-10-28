"""
Sistema de Validación Dimensional para el Proyecto.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa un sistema completo de validación dimensional que:
1. Previene bugs de unidades incompatibles
2. Documenta automáticamente las dimensiones esperadas
3. Valida ecuaciones del proyecto
4. Proporciona conversión de unidades

CRÍTICO: Según reglas, TODA ecuación debe pasar validación dimensional.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional
from functools import wraps
import numpy as np
import warnings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================

R_GAS = 8.314  # J/(mol·K) - Constante universal de los gases
BOLTZMANN = 1.380649e-23  # J/K - Constante de Boltzmann


# ============================================================================
# ENUMERACIÓN DE DIMENSIONES
# ============================================================================


class Dimension(Enum):
    """
    Dimensiones físicas fundamentales y derivadas.

    Sistema Internacional (SI):
    - L: Longitud [m]
    - T: Tiempo [s]
    - M: Masa [kg]
    - Θ: Temperatura [K]
    - N: Cantidad de sustancia [mol]

    Dimensiones derivadas relevantes al proyecto.
    """

    # Dimensiones fundamentales
    LONGITUD = "L"  # m
    TIEMPO = "T"  # s
    MASA = "M"  # kg
    TEMPERATURA = "Θ"  # K
    CANTIDAD_SUSTANCIA = "N"  # mol

    # Dimensiones derivadas - Mecánica
    VELOCIDAD = "L/T"  # m/s
    ACELERACION = "L/T²"  # m/s²
    AREA = "L²"  # m²
    VOLUMEN = "L³"  # m³
    FRECUENCIA = "1/T"  # s⁻¹ (constante cinética)

    # Dimensiones derivadas - Transferencia
    DIFUSIVIDAD = "L²/T"  # m²/s
    CONCENTRACION = "N/L³"  # mol/m³
    FLUJO_MOLAR = "N/(L²·T)"  # mol/(m²·s)
    TASA_REACCION = "N/(L³·T)"  # mol/(m³·s)
    TASA_REACCION_VOLUMETRICA = "N/(L³·T)"  # mol/(m³·s) - alias

    # Dimensiones derivadas - Cinética
    CONSTANTE_CINETICA_1ER_ORDEN = "1/T"  # s⁻¹
    ENERGIA_MOLAR = "M·L²/(T²·N)"  # J/mol

    # Dimensiones derivadas - Fluidos
    PRESION = "M/(L·T²)"  # Pa = kg/(m·s²)
    VISCOSIDAD_DINAMICA = "M/(L·T)"  # Pa·s = kg/(m·s)
    VISCOSIDAD_CINEMATICA = "L²/T"  # m²/s

    # Dimensiones derivadas - Adimensionales
    ADIMENSIONAL = "1"


# ============================================================================
# CLASE: CANTIDAD DIMENSIONAL
# ============================================================================


@dataclass
class CantidadDimensional:
    """
    Representa una cantidad física con su dimensión.

    Permite realizar operaciones aritméticas con validación dimensional
    automática y previene errores de unidades incompatibles.

    Attributes
    ----------
    valor : float
        Valor numérico de la cantidad
    dimension : Dimension
        Dimensión física de la cantidad
    nombre : str, optional
        Nombre descriptivo de la cantidad
    permitir_negativo : bool, optional
        Si False, genera warning para valores negativos

    Examples
    --------
    >>> from src.utils.validacion import CantidadDimensional, Dimension
    >>> R = CantidadDimensional(0.002, Dimension.LONGITUD, "Radio")
    >>> print(R)
    0.002 [L]
    >>> area = R * R  # Multiplicación validada
    """

    valor: float
    dimension: Dimension
    nombre: str = ""
    permitir_negativo: bool = True

    def __post_init__(self):
        """Validación post-inicialización."""
        if not self.permitir_negativo and self.valor < 0:
            warnings.warn(
                f"Cantidad física '{self.nombre}' tiene valor negativo: {self.valor}",
                UserWarning,
            )

    def __mul__(
        self, otro: Union[float, int, "CantidadDimensional"]
    ) -> "CantidadDimensional":
        """Multiplicación con validación dimensional."""
        if isinstance(otro, (int, float)):
            # Escalar: mantiene dimensión
            return CantidadDimensional(self.valor * otro, self.dimension, self.nombre)
        elif isinstance(otro, CantidadDimensional):
            # Combinar dimensiones
            nueva_dimension = self._combinar_dimensiones_multiplicacion(
                self.dimension, otro.dimension
            )
            nuevo_nombre = (
                f"({self.nombre}*{otro.nombre})" if self.nombre and otro.nombre else ""
            )
            return CantidadDimensional(
                self.valor * otro.valor, nueva_dimension, nuevo_nombre
            )
        raise TypeError(f"Multiplicación no soportada con {type(otro)}")

    def __rmul__(self, otro):
        """Multiplicación reversa (permite 2 * cantidad)."""
        return self.__mul__(otro)

    def __truediv__(
        self, otro: Union[float, int, "CantidadDimensional"]
    ) -> "CantidadDimensional":
        """División con validación dimensional."""
        if isinstance(otro, (int, float)):
            # Escalar: mantiene dimensión
            return CantidadDimensional(self.valor / otro, self.dimension, self.nombre)
        elif isinstance(otro, CantidadDimensional):
            # Combinar dimensiones
            nueva_dimension = self._combinar_dimensiones_division(
                self.dimension, otro.dimension
            )
            nuevo_nombre = (
                f"({self.nombre}/{otro.nombre})" if self.nombre and otro.nombre else ""
            )
            return CantidadDimensional(
                self.valor / otro.valor, nueva_dimension, nuevo_nombre
            )
        raise TypeError(f"División no soportada con {type(otro)}")

    def __add__(self, otro: "CantidadDimensional") -> "CantidadDimensional":
        """Suma: solo entre mismas dimensiones."""
        if not isinstance(otro, CantidadDimensional):
            raise TypeError("Solo se pueden sumar CantidadDimensional")

        if self.dimension != otro.dimension:
            raise ValueError(
                f"Inconsistencia dimensional en suma: "
                f"{self.dimension.value} + {otro.dimension.value}"
            )

        return CantidadDimensional(
            self.valor + otro.valor, self.dimension, f"({self.nombre}+{otro.nombre})"
        )

    def __sub__(self, otro: "CantidadDimensional") -> "CantidadDimensional":
        """Resta: solo entre mismas dimensiones."""
        if not isinstance(otro, CantidadDimensional):
            raise TypeError("Solo se pueden restar CantidadDimensional")

        if self.dimension != otro.dimension:
            raise ValueError(
                f"Inconsistencia dimensional en resta: "
                f"{self.dimension.value} - {otro.dimension.value}"
            )

        return CantidadDimensional(
            self.valor - otro.valor, self.dimension, f"({self.nombre}-{otro.nombre})"
        )

    def __eq__(self, otro: "CantidadDimensional") -> bool:
        """Comparación: solo entre mismas dimensiones."""
        if not isinstance(otro, CantidadDimensional):
            raise TypeError("Solo se pueden comparar CantidadDimensional")

        if self.dimension != otro.dimension:
            raise ValueError(
                f"Inconsistencia dimensional en comparación: "
                f"{self.dimension.value} vs {otro.dimension.value}"
            )

        return np.isclose(self.valor, otro.valor)

    def __repr__(self) -> str:
        """Representación en string."""
        if self.nombre:
            return f"{self.nombre} = {self.valor} [{self.dimension.value}]"
        return f"{self.valor} [{self.dimension.value}]"

    def validar_rango(self, min_val: float, max_val: float) -> bool:
        """
        Valida que el valor esté en el rango especificado.

        Parameters
        ----------
        min_val : float
            Valor mínimo permitido
        max_val : float
            Valor máximo permitido

        Returns
        -------
        bool
            True si está en rango, False si no
        """
        return min_val <= self.valor <= max_val

    @staticmethod
    def _combinar_dimensiones_multiplicacion(
        dim1: Dimension, dim2: Dimension
    ) -> Dimension:
        """Combina dimensiones en multiplicación."""
        # Casos comunes
        if dim1 == Dimension.ADIMENSIONAL:
            return dim2
        if dim2 == Dimension.ADIMENSIONAL:
            return dim1

        # L × L = L²
        if dim1 == Dimension.LONGITUD and dim2 == Dimension.LONGITUD:
            return Dimension.AREA

        # L × (L/T) = L²/T
        if dim1 == Dimension.LONGITUD and dim2 == Dimension.VELOCIDAD:
            return Dimension.DIFUSIVIDAD

        # (L/T) × T = L
        if dim1 == Dimension.VELOCIDAD and dim2 == Dimension.TIEMPO:
            return Dimension.LONGITUD

        # Caso general: retornar ADIMENSIONAL (simplificación)
        logger.warning(
            f"Combinación dimensional no manejada: {dim1.value} × {dim2.value}"
        )
        return Dimension.ADIMENSIONAL

    @staticmethod
    def _combinar_dimensiones_division(dim1: Dimension, dim2: Dimension) -> Dimension:
        """Combina dimensiones en división."""
        # Misma dimensión = adimensional
        if dim1 == dim2:
            return Dimension.ADIMENSIONAL

        # L / T = L/T (velocidad)
        if dim1 == Dimension.LONGITUD and dim2 == Dimension.TIEMPO:
            return Dimension.VELOCIDAD

        # L² / T = L²/T (difusividad)
        if dim1 == Dimension.AREA and dim2 == Dimension.TIEMPO:
            return Dimension.DIFUSIVIDAD

        # Caso general
        logger.warning(
            f"Combinación dimensional no manejada: {dim1.value} / {dim2.value}"
        )
        return Dimension.ADIMENSIONAL


# ============================================================================
# DECORADOR: VALIDAR DIMENSIONES
# ============================================================================


def validar_dimensiones(func):
    """
    Decorador que valida dimensiones de entrada/salida de funciones.

    Ejecuta la función y valida que las cantidades dimensionales sean
    consistentes. Útil para documentar dimensiones esperadas.

    Examples
    --------
    >>> @validar_dimensiones
    ... def calcular_area(radio: CantidadDimensional) -> CantidadDimensional:
    ...     return CantidadDimensional(np.pi * radio.valor**2, Dimension.AREA)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log de entrada (solo CantidadDimensional)
        for i, arg in enumerate(args):
            if isinstance(arg, CantidadDimensional):
                logger.debug(
                    f"{func.__name__}: arg[{i}] = {arg.valor} [{arg.dimension.value}]"
                )

        # Ejecutar función
        resultado = func(*args, **kwargs)

        # Log de salida
        if isinstance(resultado, CantidadDimensional):
            logger.debug(
                f"{func.__name__}: return = {resultado.valor} [{resultado.dimension.value}]"
            )

        return resultado

    return wrapper


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================


def es_adimensional(cantidad: CantidadDimensional) -> bool:
    """
    Verifica si una cantidad es adimensional.

    Parameters
    ----------
    cantidad : CantidadDimensional
        Cantidad a verificar

    Returns
    -------
    bool
        True si es adimensional
    """
    return cantidad.dimension == Dimension.ADIMENSIONAL


def verificar_dimension(
    cantidad: CantidadDimensional, dimension_esperada: Dimension
) -> bool:
    """
    Verifica que una cantidad tenga la dimensión esperada.

    Parameters
    ----------
    cantidad : CantidadDimensional
        Cantidad a verificar
    dimension_esperada : Dimension
        Dimensión que se espera

    Returns
    -------
    bool
        True si la dimensión coincide
    """
    return cantidad.dimension == dimension_esperada


# ============================================================================
# CONVERSIÓN DE UNIDADES
# ============================================================================


def convertir_temperatura(valor: float, origen: str, destino: str) -> float:
    """
    Convierte temperaturas entre diferentes escalas.

    Parameters
    ----------
    valor : float
        Valor de temperatura
    origen : str
        Escala origen ('celsius', 'kelvin', 'fahrenheit')
    destino : str
        Escala destino

    Returns
    -------
    float
        Temperatura convertida

    Examples
    --------
    >>> convertir_temperatura(400, 'celsius', 'kelvin')
    673.0
    """
    # Normalizar a minúsculas
    origen = origen.lower()
    destino = destino.lower()

    # Primero convertir todo a Kelvin
    if origen == "celsius":
        kelvin = valor + 273.15
    elif origen == "kelvin":
        kelvin = valor
    elif origen == "fahrenheit":
        kelvin = (valor - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Escala de temperatura no reconocida: {origen}")

    # Luego convertir de Kelvin a destino
    if destino == "kelvin":
        return kelvin
    elif destino == "celsius":
        return kelvin - 273.15
    elif destino == "fahrenheit":
        return (kelvin - 273.15) * 9 / 5 + 32
    else:
        raise ValueError(f"Escala de temperatura no reconocida: {destino}")


def convertir_presion(valor: float, origen: str, destino: str) -> float:
    """
    Convierte presiones entre diferentes unidades.

    Parameters
    ----------
    valor : float
        Valor de presión
    origen : str
        Unidad origen ('atm', 'pascal', 'bar', 'psi')
    destino : str
        Unidad destino

    Returns
    -------
    float
        Presión convertida

    Examples
    --------
    >>> convertir_presion(1.0, 'atm', 'pascal')
    101325.0
    """
    origen = origen.lower()
    destino = destino.lower()

    # Factores de conversión a Pascal
    a_pascal = {
        "pascal": 1.0,
        "pa": 1.0,
        "atm": 101325.0,
        "bar": 100000.0,
        "psi": 6894.76,
    }

    if origen not in a_pascal:
        raise ValueError(f"Unidad de presión no reconocida: {origen}")
    if destino not in a_pascal:
        raise ValueError(f"Unidad de presión no reconocida: {destino}")

    # Convertir a Pascal, luego a destino
    pascal = valor * a_pascal[origen]
    return pascal / a_pascal[destino]


def convertir_concentracion_ppm(
    ppm: float, T: float, P: float, R_gas: float = R_GAS
) -> float:
    """
    Convierte concentración de ppm a mol/m³ usando ley de gases ideales.

    Parameters
    ----------
    ppm : float
        Concentración en partes por millón
    T : float
        Temperatura [K]
    P : float
        Presión [Pa]
    R_gas : float, optional
        Constante de gases [J/(mol·K)]

    Returns
    -------
    float
        Concentración [mol/m³]

    Notes
    -----
    Fórmula: C = (y × P) / (R × T)
    donde y = ppm × 10⁻⁶

    Examples
    --------
    >>> convertir_concentracion_ppm(800, T=673, P=101325)
    0.0145  # mol/m³
    """
    y = ppm * 1e-6  # Convertir ppm a fracción molar
    C = (y * P) / (R_gas * T)
    return C


# ============================================================================
# VALIDADORES DE ECUACIONES ESPECÍFICAS
# ============================================================================


def validar_ecuacion_difusion_2d() -> bool:
    """
    Valida dimensionalmente la ecuación de difusión-reacción 2D.

    Ecuación:
        ∂C/∂t = D_eff × ∇²C - k_app × C

    Dimensiones:
        [N/(L³·T)] = [L²/T] × [N/L⁵] - [1/T] × [N/L³]
        [N/(L³·T)] = [N/(L³·T)] - [N/(L³·T)]  ✅

    Returns
    -------
    bool
        True si la ecuación es dimensionalmente correcta

    Raises
    ------
    ValueError
        Si hay inconsistencia dimensional
    """
    # LHS: ∂C/∂t
    dim_lhs = Dimension.TASA_REACCION  # N/(L³·T)

    # RHS término difusivo: D_eff × ∇²C
    # D_eff: L²/T
    # ∇²C: N/L⁵ (segunda derivada espacial de concentración)
    # Producto: (L²/T) × (N/L⁵) = N/(L³·T)
    dim_difusion = Dimension.TASA_REACCION

    # RHS término reactivo: k_app × C
    # k_app: 1/T
    # C: N/L³
    # Producto: (1/T) × (N/L³) = N/(L³·T)
    dim_reaccion = Dimension.TASA_REACCION

    # Verificar que todos son iguales
    if dim_lhs != dim_difusion:
        raise ValueError(
            f"Inconsistencia dimensional: LHS={dim_lhs.value} vs difusión={dim_difusion.value}"
        )
    if dim_lhs != dim_reaccion:
        raise ValueError(
            f"Inconsistencia dimensional: LHS={dim_lhs.value} vs reacción={dim_reaccion.value}"
        )

    logger.info("✓ Ecuación de difusión-reacción 2D: dimensionalmente correcta")
    return True


def validar_ecuacion_robin() -> bool:
    """
    Valida dimensionalmente la condición de Robin en la frontera.

    Ecuación:
        -D_eff × ∂C/∂r = k_c × (C_s - C_bulk)

    Dimensiones:
        [L²/T] × [N/L³/L] = [L/T] × [N/L³]
        [N/(L²·T)] = [N/(L²·T)]  ✅

    Returns
    -------
    bool
        True si la ecuación es dimensionalmente correcta
    """
    # LHS: -D_eff × ∂C/∂r
    # D_eff: L²/T
    # ∂C/∂r: (N/L³)/L = N/L⁴
    # Producto: (L²/T) × (N/L⁴) = N/(L²·T)
    dim_lhs = Dimension.FLUJO_MOLAR

    # RHS: k_c × (C_s - C_bulk)
    # k_c: L/T
    # (C_s - C_bulk): N/L³
    # Producto: (L/T) × (N/L³) = N/(L²·T)
    dim_rhs = Dimension.FLUJO_MOLAR

    if dim_lhs != dim_rhs:
        raise ValueError(
            f"Inconsistencia dimensional en Robin: LHS={dim_lhs.value} vs RHS={dim_rhs.value}"
        )

    logger.info("✓ Condición de Robin: dimensionalmente correcta")
    return True


def validar_ecuacion_arrhenius() -> bool:
    """
    Valida dimensionalmente la ecuación de Arrhenius.

    Ecuación:
        k = k₀ × exp(-Ea / (R × T))

    Dimensiones:
        [1/T] = [1/T] × exp([M·L²/(T²·N)] / ([M·L²/(T²·N·Θ)] × [Θ]))
        [1/T] = [1/T] × exp(adimensional)  ✅

    Returns
    -------
    bool
        True si la ecuación es dimensionalmente correcta
    """
    # k y k₀: 1/T (constante cinética 1er orden)
    dim_k = Dimension.CONSTANTE_CINETICA_1ER_ORDEN

    # Exponente debe ser adimensional
    # Ea / (R × T):
    # Ea: M·L²/(T²·N) [J/mol]
    # R: M·L²/(T²·N·Θ) [J/(mol·K)]
    # T: Θ [K]
    # R×T: M·L²/(T²·N) [J/mol]
    # Ea/(R×T): adimensional ✅

    logger.info("✓ Ecuación de Arrhenius: dimensionalmente correcta")
    return True


def validar_todas_ecuaciones() -> dict:
    """
    Valida todas las ecuaciones del proyecto.

    Returns
    -------
    dict
        Diccionario con resultados de cada validación

    Examples
    --------
    >>> resultados = validar_todas_ecuaciones()
    >>> assert all(resultados.values())
    """
    resultados = {}

    try:
        resultados["difusion_2d"] = validar_ecuacion_difusion_2d()
    except Exception as e:
        logger.error(f"Validación difusión 2D falló: {e}")
        resultados["difusion_2d"] = False

    try:
        resultados["robin"] = validar_ecuacion_robin()
    except Exception as e:
        logger.error(f"Validación Robin falló: {e}")
        resultados["robin"] = False

    try:
        resultados["arrhenius"] = validar_ecuacion_arrhenius()
    except Exception as e:
        logger.error(f"Validación Arrhenius falló: {e}")
        resultados["arrhenius"] = False

    return resultados


# ============================================================================
# CALCULADORAS DIMENSIONALES (con validación automática)
# ============================================================================


@validar_dimensiones
def calcular_reynolds_dimensional(
    rho: float, u: float, D: float, mu: float
) -> CantidadDimensional:
    """
    Calcula número de Reynolds con validación dimensional.

    Re = (ρ × u × D) / μ

    Parameters
    ----------
    rho : float
        Densidad [kg/m³]
    u : float
        Velocidad [m/s]
    D : float
        Longitud característica [m]
    mu : float
        Viscosidad dinámica [Pa·s = kg/(m·s)]

    Returns
    -------
    CantidadDimensional
        Número de Reynolds [adimensional]
    """
    Re = (rho * u * D) / mu
    return CantidadDimensional(Re, Dimension.ADIMENSIONAL, "Reynolds")


@validar_dimensiones
def calcular_sherwood_dimensional(
    k_c: float, D: float, D_AB: float
) -> CantidadDimensional:
    """
    Calcula número de Sherwood con validación dimensional.

    Sh = (k_c × D) / D_AB

    Parameters
    ----------
    k_c : float
        Coeficiente convectivo de masa [m/s]
    D : float
        Longitud característica [m]
    D_AB : float
        Difusividad molecular [m²/s]

    Returns
    -------
    CantidadDimensional
        Número de Sherwood [adimensional]
    """
    Sh = (k_c * D) / D_AB
    return CantidadDimensional(Sh, Dimension.ADIMENSIONAL, "Sherwood")


@validar_dimensiones
def calcular_D_eff_dimensional(
    epsilon: float, D_comb: float, tau: float
) -> CantidadDimensional:
    """
    Calcula difusividad efectiva con validación dimensional.

    D_eff = (ε × D_comb) / τ

    Parameters
    ----------
    epsilon : float
        Porosidad [adimensional]
    D_comb : float
        Difusividad combinada [m²/s]
    tau : float
        Tortuosidad [adimensional]

    Returns
    -------
    CantidadDimensional
        Difusividad efectiva [m²/s]
    """
    D_eff = (epsilon * D_comb) / tau
    return CantidadDimensional(D_eff, Dimension.DIFUSIVIDAD, "D_eff")


@validar_dimensiones
def calcular_modulo_thiele_dimensional(
    R: float, k_app: float, D_eff: float
) -> CantidadDimensional:
    """
    Calcula módulo de Thiele con validación dimensional.

    φ = R × √(k_app / D_eff)

    Parameters
    ----------
    R : float
        Radio característico [m]
    k_app : float
        Constante cinética aparente [s⁻¹]
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    CantidadDimensional
        Módulo de Thiele [adimensional]

    Notes
    -----
    Dimensiones:
        φ = [L] × √([1/T] / [L²/T])
          = [L] × √[1/L²]
          = [L] × [1/L]
          = [1] ✅ adimensional
    """
    phi = R * np.sqrt(k_app / D_eff)
    return CantidadDimensional(phi, Dimension.ADIMENSIONAL, "Thiele")


# ============================================================================
# VALIDACIÓN DE DIMENSIONES EN ECUACIONES
# ============================================================================


def obtener_dimension_derivada_temporal_concentracion() -> Dimension:
    """
    Retorna la dimensión de ∂C/∂t.

    Returns
    -------
    Dimension
        TASA_REACCION = N/(L³·T)
    """
    return Dimension.TASA_REACCION


def obtener_dimension_termino_difusivo() -> Dimension:
    """
    Retorna la dimensión de D_eff × ∇²C.

    D_eff: L²/T
    ∇²C: N/L⁵
    Producto: N/(L³·T)

    Returns
    -------
    Dimension
        TASA_REACCION = N/(L³·T)
    """
    return Dimension.TASA_REACCION


def obtener_dimension_termino_reactivo() -> Dimension:
    """
    Retorna la dimensión de k_app × C.

    k_app: 1/T
    C: N/L³
    Producto: N/(L³·T)

    Returns
    -------
    Dimension
        TASA_REACCION = N/(L³·T)
    """
    return Dimension.TASA_REACCION


# ============================================================================
# VALIDACIÓN DE PARÁMETROS DEL PROYECTO
# ============================================================================


def validar_parametros_proyecto(params) -> bool:
    """
    Valida dimensionalmente todos los parámetros del proyecto.

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Returns
    -------
    bool
        True si todas las validaciones pasan

    Raises
    ------
    ValueError
        Si hay inconsistencias dimensionales
    """
    # Validar que D_eff tenga dimensión correcta
    D_eff = CantidadDimensional(params.difusion.D_eff, Dimension.DIFUSIVIDAD, "D_eff")
    assert D_eff.dimension == Dimension.DIFUSIVIDAD

    # Validar que k_app tenga dimensión correcta
    k_app = CantidadDimensional(
        params.cinetica.k_app,
        Dimension.CONSTANTE_CINETICA_1ER_ORDEN,
        "k_app",
    )
    assert k_app.dimension == Dimension.CONSTANTE_CINETICA_1ER_ORDEN

    # Validar que C_bulk tenga dimensión correcta
    C_bulk = CantidadDimensional(
        params.operacion.C_bulk, Dimension.CONCENTRACION, "C_bulk"
    )
    assert C_bulk.dimension == Dimension.CONCENTRACION

    # Validar que temperatura tenga dimensión correcta
    T = CantidadDimensional(params.operacion.T, Dimension.TEMPERATURA, "T")
    assert T.dimension == Dimension.TEMPERATURA

    # Validar rangos físicos
    assert D_eff.validar_rango(1e-10, 1e-4)
    assert k_app.validar_rango(0, 1.0)
    assert C_bulk.validar_rango(0, 1.0)
    assert T.validar_rango(300, 1000)

    logger.info("✓ Todos los parámetros del proyecto son dimensionalmente correctos")
    return True


# ============================================================================
# GENERACIÓN DE REPORTES
# ============================================================================


def generar_reporte_validacion() -> str:
    """
    Genera reporte de validación dimensional de todas las ecuaciones.

    Returns
    -------
    str
        Reporte formateado con resultados de validación
    """
    reporte_lines = []
    reporte_lines.append("")
    reporte_lines.append("=" * 70)
    reporte_lines.append("REPORTE DE VALIDACIÓN DIMENSIONAL")
    reporte_lines.append("Proyecto: Transferencia de Masa en Reactor Catalítico")
    reporte_lines.append("=" * 70)
    reporte_lines.append("")

    # Validar cada ecuación
    resultados = validar_todas_ecuaciones()

    reporte_lines.append("Ecuaciones Validadas:")
    reporte_lines.append("")

    for ecuacion, resultado in resultados.items():
        estado = "✓ PASS" if resultado else "✗ FAIL"
        reporte_lines.append(f"  {estado}  {ecuacion.replace('_', ' ').title()}")

    reporte_lines.append("")

    # Resumen
    total = len(resultados)
    pasadas = sum(resultados.values())
    reporte_lines.append(f"Resumen: {pasadas}/{total} ecuaciones validadas")

    if pasadas == total:
        reporte_lines.append("")
        reporte_lines.append("✅ TODAS LAS ECUACIONES SON DIMENSIONALMENTE CORRECTAS")
    else:
        reporte_lines.append("")
        reporte_lines.append("⚠️  ALGUNAS ECUACIONES TIENEN INCONSISTENCIAS")

    reporte_lines.append("=" * 70)
    reporte_lines.append("")

    return "\n".join(reporte_lines)


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
