"""
Tabla Maestra de Parámetros del Proyecto.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo contiene TODOS los parámetros validados del proyecto organizados
en dataclasses inmutables. Los valores provienen de PARAMETROS_PROYECTO.md.

IMPORTANTE:
- NO modificar valores sin validación dimensional completa
- Todas las unidades en SI (metros, segundos, moles, Kelvin, Pascal)
- Los parámetros son inmutables (frozen=True)
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any


# Constantes físicas globales
R_GAS = 8.314  # J/(mol·K) - Constante universal de los gases


# ============================================================================
# TABLA I: GEOMETRÍA Y DOMINIOS
# ============================================================================


@dataclass(frozen=True)
class GeometriaParams:
    """
    Parámetros geométricos del pellet catalítico (Tabla I).

    Attributes
    ----------
    D : float
        Diámetro del pellet [m]
    R : float
        Radio del pellet [m] (R = D/2)
    theta1 : float
        Ángulo inicial del defecto [rad]
    theta2 : float
        Ángulo final del defecto [rad] (45° = π/4)
    L : float
        Longitud axial [m] (>> D, invarianza axial)
    P : float
        Perímetro externo [m] (P = 2πR)

    Notes
    -----
    La región del defecto está definida por:
    - Radial: TODO el pellet (0 ≤ r ≤ R)
    - Angular: θ ∈ [θ1, θ2] = [0°, 45°]
    - Geometría: SECTOR/CUÑA completo desde centro hasta superficie
    - Característica: Sin reacción química (k_app = 0)

    References
    ----------
    .. [1] Enunciado del Proyecto Personal 2, Sección 2.3
    """

    D: float = 0.004  # m
    R: float = 0.002  # m (R = D/2)
    r1: float = 0.002 / 3  # m (R/3 - radio interno del anillo defectuoso)
    r2: float = 2 * 0.002 / 3  # m (2R/3 - radio externo del anillo defectuoso)
    theta1: float = 0.0  # rad (0° - inicio del sector angular defectuoso)
    theta2: float = np.pi / 4  # rad (45° = π/4 - fin del sector angular defectuoso)
    L: float = 0.100  # m (100 mm, >> D para invarianza axial)
    P: float = 2 * np.pi * 0.002  # m (P = 2πR) - expresión exacta

    def validar(self) -> None:
        """
        Valida la consistencia interna de los parámetros geométricos.

        Raises
        ------
        AssertionError
            Si alguna condición de consistencia falla
        """
        # Radio debe ser mitad del diámetro
        assert np.isclose(
            self.R, self.D / 2, rtol=1e-10
        ), f"R debe ser D/2: R={self.R}, D/2={self.D/2}"

        # Defecto radial ordenado (anillo)
        assert (
            0 < self.r1 < self.r2 < self.R
        ), f"Debe cumplirse 0 < r1 < r2 < R: r1={self.r1}, r2={self.r2}, R={self.R}"

        # r1 debe ser R/3
        assert np.isclose(
            self.r1, self.R / 3, rtol=1e-6
        ), f"r1 debe ser R/3: r1={self.r1}, R/3={self.R/3}"

        # r2 debe ser 2R/3
        assert np.isclose(
            self.r2, 2 * self.R / 3, rtol=1e-6
        ), f"r2 debe ser 2R/3: r2={self.r2}, 2R/3={2*self.R/3}"

        # theta1 debe ser 0
        assert np.isclose(
            self.theta1, 0.0, atol=1e-10
        ), f"theta1 debe ser 0: theta1={self.theta1}"

        # theta2 debe ser π/4
        assert np.isclose(
            self.theta2, np.pi / 4, rtol=1e-6
        ), f"theta2 debe ser π/4: theta2={self.theta2}, π/4={np.pi/4}"

        # Perímetro debe ser 2πR
        assert np.isclose(
            self.P, 2 * np.pi * self.R, rtol=1e-6
        ), f"P debe ser 2πR: P={self.P}, 2πR={2*np.pi*self.R}"


# ============================================================================
# TABLA II: CONDICIONES DE OPERACIÓN EXTERNAS
# ============================================================================


@dataclass(frozen=True)
class OperacionParams:
    """
    Condiciones de operación externas (Tabla II).

    Attributes
    ----------
    T : float
        Temperatura del gas [K]
    P : float
        Presión del sistema [Pa]
    y_CO : float
        Concentración de CO en ppm [adimensional × 10⁶]
    C_bulk : float
        Concentración bulk de CO [mol/m³]
    u_s : float
        Velocidad superficial del gas [m/s]
    epsilon_b : float
        Porosidad del lecho empacado [adimensional]
    u_i : float
        Velocidad intersticial [m/s] (u_i = u_s / ε_b)
    C_inicial : float
        Concentración inicial en el pellet [mol/m³]

    Notes
    -----
    C_bulk se calcula con la ley de gases ideales:
        C = (y × P) / (R × T)
    donde R = 8.314 J/(mol·K)

    References
    ----------
    .. [1] Dixon, A.G. (1988)
    """

    T: float = 673.0  # K (400°C)
    P: float = 101325.0  # Pa (1 atm)
    y_CO: float = 800e-6  # ppm convertido a fracción
    C_bulk: float = 0.0145  # mol/m³
    u_s: float = 0.3  # m/s
    epsilon_b: float = 0.4  # adimensional
    u_i: float = 0.75  # m/s (u_s / epsilon_b)
    C_inicial: float = 0.0  # mol/m³ (pellet libre de CO en t=0)

    def validar(self) -> None:
        """Valida consistencia de parámetros de operación."""
        # C_bulk calculado con ley de gases ideales
        C_bulk_esperado = (self.y_CO * self.P) / (R_GAS * self.T)
        assert np.isclose(
            self.C_bulk, C_bulk_esperado, rtol=0.01
        ), f"C_bulk inconsistente: {self.C_bulk} vs esperado {C_bulk_esperado}"

        # u_i = u_s / epsilon_b
        assert np.isclose(
            self.u_i, self.u_s / self.epsilon_b, rtol=1e-6
        ), f"u_i debe ser u_s/epsilon_b: {self.u_i} vs {self.u_s/self.epsilon_b}"

        # Temperatura positiva y razonable
        assert 300 < self.T < 1000, f"Temperatura fuera de rango: {self.T} K"

        # Presión positiva
        assert self.P > 0, f"Presión debe ser positiva: {self.P} Pa"


# ============================================================================
# TABLA III: PROPIEDADES TERMOFÍSICAS DEL GAS
# ============================================================================


@dataclass(frozen=True)
class GasParams:
    """
    Propiedades termofísicas del gas (Tabla III).

    Attributes
    ----------
    rho_gas : float
        Densidad del gas [kg/m³]
    mu : float
        Viscosidad dinámica [Pa·s]
    nu : float
        Viscosidad cinemática [m²/s] (ν = μ/ρ)
    D_CO_aire : float
        Difusividad CO-aire [m²/s]
    Sc : float
        Número de Schmidt [adimensional] (Sc = ν/D)
    M_aire : float
        Masa molar del aire [kg/mol]

    Notes
    -----
    Densidad calculada con ley de gases ideales:
        ρ = PM/(RT)

    Viscosidad obtenida con correlación de Sutherland.

    References
    ----------
    .. [1] White, F.M. (2016)
    .. [2] The Engineering ToolBox (2018)
    """

    rho_gas: float = 0.524  # kg/m³
    mu: float = 3.32e-5  # Pa·s
    nu: float = 6.34e-5  # m²/s
    D_CO_aire: float = 8.75e-5  # m²/s
    Sc: float = 0.724  # adimensional
    M_aire: float = 0.02897  # kg/mol

    def validar(self) -> None:
        """Valida propiedades del gas."""
        # ν = μ / ρ
        nu_esperado = self.mu / self.rho_gas
        assert np.isclose(
            self.nu, nu_esperado, rtol=0.01
        ), f"ν inconsistente: {self.nu} vs {nu_esperado}"

        # Sc = ν / D
        Sc_esperado = self.nu / self.D_CO_aire
        assert np.isclose(
            self.Sc, Sc_esperado, rtol=0.01
        ), f"Sc inconsistente: {self.Sc} vs {Sc_esperado}"


# ============================================================================
# TABLA IV: TRANSFERENCIA DE MASA INTERFASE
# ============================================================================


@dataclass(frozen=True)
class TransferenciaParams:
    """
    Parámetros de transferencia de masa interfase (Tabla IV).

    Attributes
    ----------
    d_p : float
        Diámetro de partícula [m] (aproximación conservadora: d_p = D)
    Re_p : float
        Reynolds de partícula [adimensional]
    Sh : float
        Número de Sherwood [adimensional]
    k_c : float
        Coeficiente convectivo de masa [m/s]

    Notes
    -----
    Correlación de Wakao-Funazkri para lechos empacados:
        Sh = 2 + 1.1 × Re_p^0.6 × Sc^(1/3)

    Válida para:
    - Re_p ∈ [3, 10⁴]
    - Sc ∈ [0.6, 10³]

    References
    ----------
    .. [1] Wakao, N. & Funazkri, T. (1978)
    """

    d_p: float = 0.004  # m (igual a D del pellet)
    Re_p: float = 19.0  # adimensional
    Sh: float = 7.78  # adimensional
    k_c: float = 0.170  # m/s

    def validar(self) -> None:
        """Valida parámetros de transferencia de masa."""
        # Rangos de validez de Wakao-Funazkri
        assert 3 <= self.Re_p <= 1e4, f"Re_p fuera de rango Wakao-Funazkri: {self.Re_p}"

        # Sh debe estar en rango razonable
        assert 2 < self.Sh < 100, f"Sh fuera de rango razonable: {self.Sh}"

        # k_c debe ser positivo
        assert self.k_c > 0, f"k_c debe ser positivo: {self.k_c}"


# ============================================================================
# TABLA V: DIFUSIÓN INTRAPARTICULAR
# ============================================================================


@dataclass(frozen=True)
class DifusionParams:
    """
    Parámetros de difusión intraparticular (Tabla V).

    Attributes
    ----------
    epsilon : float
        Porosidad del pellet [adimensional]
    tau : float
        Tortuosidad [adimensional]
    r_poro : float
        Radio de poro promedio [m]
    lambda_mfp : float
        Camino libre medio [m]
    Kn : float
        Número de Knudsen [adimensional] (Kn = λ/r_poro)
    D_Kn : float
        Difusividad de Knudsen [m²/s]
    D_comb : float
        Difusividad combinada [m²/s] (Bosanquet)
    D_eff : float
        Difusividad efectiva [m²/s] (D_eff = ε×D_comb/τ)

    Notes
    -----
    Régimen difusivo determinado por Knudsen:
    - Kn >> 1: Difusión de Knudsen dominante (este caso: Kn = 19.3)
    - Kn << 1: Difusión molecular
    - Kn ~ 1: Régimen de transición (Bosanquet)

    Fórmulas:
        D_Kn = (2/3) × r_poro × √(8RT/πM)
        1/D_comb = 1/D_molecular + 1/D_Knudsen
        D_eff = (ε/τ) × D_comb

    References
    ----------
    .. [1] Hill (2025); Abello (2002); Mourkou et al. (2024)
    """

    epsilon: float = 0.45  # adimensional
    tau: float = 3.0  # adimensional
    r_poro: float = 10e-9  # m (10 nm)
    lambda_mfp: float = 1.93e-7  # m
    Kn: float = 19.3  # adimensional
    D_Kn: float = 7.43e-6  # m²/s
    D_comb: float = 6.97e-6  # m²/s
    D_eff: float = 1.04e-6  # m²/s

    def validar(self) -> None:
        """Valida parámetros de difusión."""
        # Kn = λ / r_poro
        Kn_esperado = self.lambda_mfp / self.r_poro
        assert np.isclose(
            self.Kn, Kn_esperado, rtol=0.05
        ), f"Kn inconsistente: {self.Kn} vs {Kn_esperado}"

        # D_eff = ε × D_comb / τ
        D_eff_esperado = self.epsilon * self.D_comb / self.tau
        assert np.isclose(
            self.D_eff, D_eff_esperado, rtol=0.01
        ), f"D_eff inconsistente: {self.D_eff} vs {D_eff_esperado}"

        # Porosidad en rango físico
        assert 0.2 < self.epsilon < 0.8, f"Porosidad fuera de rango: {self.epsilon}"

        # Tortuosidad razonable
        assert 1.5 < self.tau < 10, f"Tortuosidad fuera de rango: {self.tau}"

        # D_eff en rango físico
        assert 1e-10 < self.D_eff < 1e-4, f"D_eff fuera de rango físico: {self.D_eff}"


# ============================================================================
# TABLA VI: CINÉTICA APARENTE
# ============================================================================


@dataclass(frozen=True)
class CineticaParams:
    """
    Parámetros cinéticos (Tabla VI).

    Attributes
    ----------
    k0 : float
        Factor pre-exponencial [s⁻¹]
    Ea : float
        Energía de activación [J/mol]
    k_app : float
        Constante cinética aparente [s⁻¹]
    T : float
        Temperatura de operación [K]
    n : int
        Orden de reacción [adimensional]

    Notes
    -----
    Reacción: 2 CO + O₂ → 2 CO₂

    Cinética aparente de 1er orden en CO (exceso de O₂).
    Ley de Arrhenius:
        k_app = k₀ × exp(-Ea / RT)

    En la región del defecto: k_app = 0

    References
    ----------
    .. [1] Enunciado del Proyecto Personal 2
    """

    k0: float = 2.3e5  # s⁻¹
    Ea: float = 1.0e5  # J/mol (100 kJ/mol)
    k_app: float = 4.0e-3  # s⁻¹
    T: float = 673.0  # K (debe coincidir con OperacionParams.T)
    n: int = 1  # Primer orden en CO

    def validar(self) -> None:
        """Valida parámetros cinéticos."""
        # k_app calculado con Arrhenius
        k_app_esperado = self.k0 * np.exp(-self.Ea / (R_GAS * self.T))
        assert np.isclose(
            self.k_app, k_app_esperado, rtol=0.01
        ), f"k_app inconsistente: {self.k_app} vs {k_app_esperado}"

        # k_app debe ser positivo pero no excesivo
        assert 0 < self.k_app < 1.0, f"k_app fuera de rango razonable: {self.k_app}"


# ============================================================================
# TABLA VII: ESPECIFICACIONES DE MALLADO
# ============================================================================


@dataclass(frozen=True)
class MalladoParams:
    """
    Especificaciones de mallado (Tabla VII).

    Attributes
    ----------
    nr : int
        Número de nodos radiales
    ntheta : int
        Número de nodos angulares
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    dt : float
        Paso temporal inicial [s]

    Notes
    -----
    Criterios de mallado:
    - Radial: Capturar gradientes en defecto (r₁=R/3, r₂=2R/3)
    - Angular: Resolver defecto θ∈[0°,45°] con ~12 nodos
    - Temporal: Incondicionalmente estable (Crank-Nicolson)

    Cálculos:
        Δr = R / (nr - 1)
        Δθ = 2π / (ntheta - 1)
    """

    nr: int = 61  # nodos radiales
    ntheta: int = 96  # nodos angulares
    dr: float = 0.002 / (61 - 1)  # m (Δr = R/(nr-1)) - expresión exacta
    dtheta: float = 2 * np.pi / (96 - 1)  # rad (Δθ = 2π/(ntheta-1)) - expresión exacta
    dt: float = 0.001  # s (paso temporal inicial)

    def validar(self) -> None:
        """Valida parámetros de mallado."""
        # nr razonable
        assert 10 < self.nr < 200, f"nr fuera de rango: {self.nr}"

        # ntheta razonable
        assert 10 < self.ntheta < 200, f"ntheta fuera de rango: {self.ntheta}"

        # dt positivo
        assert self.dt > 0, f"dt debe ser positivo: {self.dt}"


# ============================================================================
# TABLA VIII: CONDICIONES DE FRONTERA
# ============================================================================


@dataclass(frozen=True)
class CondicionesFronteraParams:
    """
    Condiciones de frontera (Tabla VIII).

    Attributes
    ----------
    C_inicial : float
        Condición inicial [mol/m³] (C = 0 en todo el dominio en t=0)
    tipo_centro : str
        Tipo de condición en r=0 ('simetria')
    tipo_superficie : str
        Tipo de condición en r=R ('robin')
    tipo_angular : str
        Tipo de condición en θ=0,2π ('periodicidad')

    Notes
    -----
    Condiciones de frontera:
    1. r = 0: Simetría → ∂C/∂r = 0
    2. r = R: Robin → -D_eff·∂C/∂r = k_c·(C_s - C_bulk)
    3. θ = 0, 2π: Periodicidad → C(r,0) = C(r,2π)
    4. Interfaz defecto-activo: Continuidad de C y flujo

    Condición inicial:
        C(r, θ, 0) = 0  ∀ r, θ
    (Pellet completamente libre de CO al inicio)
    """

    C_inicial: float = 0.0  # mol/m³
    tipo_centro: str = "simetria"  # ∂C/∂r = 0 en r=0
    tipo_superficie: str = "robin"  # Convectiva en r=R
    tipo_angular: str = "periodicidad"  # C(r,0) = C(r,2π)

    def validar(self) -> None:
        """Valida condiciones de frontera."""
        assert self.C_inicial == 0.0, "Condición inicial debe ser C=0 según enunciado"
        assert self.tipo_centro == "simetria"
        assert self.tipo_superficie == "robin"
        assert self.tipo_angular == "periodicidad"


# ============================================================================
# TABLA IX: CRITERIOS DE CONVERGENCIA
# ============================================================================


@dataclass(frozen=True)
class ConvergenciaParams:
    """
    Criterios de convergencia (Tabla IX).

    Attributes
    ----------
    tol_convergencia : float
        Tolerancia para error relativo máximo [adimensional]
    pasos_consecutivos : int
        Número de pasos consecutivos que deben cumplir criterio
    tol_balance_masa : float
        Tolerancia de error de balance de masa [adimensional]
    tol_validacion_dimensional : float
        Tolerancia para validación dimensional [adimensional]

    Notes
    -----
    Estado estacionario se alcanza cuando:
    1. max|C^{n+1} - C^n| / max|C^{n+1}| < tol_convergencia
    2. Condición mantenida por 'pasos_consecutivos' pasos
    3. Error de balance de masa < tol_balance_masa
    """

    tol_convergencia: float = 1e-6  # Error relativo máximo
    pasos_consecutivos: int = 3  # Pasos para confirmar convergencia
    tol_balance_masa: float = 0.01  # 1% error máximo
    tol_validacion_dimensional: float = 1e-10  # Exactitud dimensional

    def validar(self) -> None:
        """Valida criterios de convergencia."""
        assert (
            0 < self.tol_convergencia < 1e-3
        ), "tol_convergencia debe ser pequeña pero realista"
        assert self.pasos_consecutivos >= 1
        assert 0 < self.tol_balance_masa < 0.1, "tol_balance_masa debe ser menor al 10%"


# ============================================================================
# CLASE MAESTRA: TODOS LOS PARÁMETROS
# ============================================================================


@dataclass(frozen=True)
class ParametrosMaestros:
    """
    Clase maestra que agrupa todos los parámetros del proyecto.

    Attributes
    ----------
    geometria : GeometriaParams
        Parámetros geométricos (Tabla I)
    operacion : OperacionParams
        Condiciones de operación (Tabla II)
    gas : GasParams
        Propiedades del gas (Tabla III)
    transferencia : TransferenciaParams
        Transferencia de masa interfase (Tabla IV)
    difusion : DifusionParams
        Difusión intraparticular (Tabla V)
    cinetica : CineticaParams
        Cinética aparente (Tabla VI)
    mallado : MalladoParams
        Especificaciones de mallado (Tabla VII)
    condiciones_frontera : CondicionesFronteraParams
        Condiciones de frontera (Tabla VIII)
    convergencia : ConvergenciaParams
        Criterios de convergencia (Tabla IX)

    Examples
    --------
    >>> params = ParametrosMaestros()
    >>> print(f"Radio del pellet: {params.geometria.R} m")
    >>> print(f"Difusividad efectiva: {params.difusion.D_eff} m²/s")
    >>> params.validar_todo()  # Valida consistencia de todos los parámetros
    """

    geometria: GeometriaParams = GeometriaParams()
    operacion: OperacionParams = OperacionParams()
    gas: GasParams = GasParams()
    transferencia: TransferenciaParams = TransferenciaParams()
    difusion: DifusionParams = DifusionParams()
    cinetica: CineticaParams = CineticaParams()
    mallado: MalladoParams = MalladoParams()
    condiciones_frontera: CondicionesFronteraParams = CondicionesFronteraParams()
    convergencia: ConvergenciaParams = ConvergenciaParams()

    def validar_todo(self) -> None:
        """
        Valida la consistencia de todos los parámetros.

        Ejecuta el método validar() de cada tabla de parámetros y verifica
        la consistencia inter-tablas (e.g., módulo de Thiele).

        Raises
        ------
        AssertionError
            Si alguna validación falla
        """
        # Validar cada tabla individualmente
        self.geometria.validar()
        self.operacion.validar()
        self.gas.validar()
        self.transferencia.validar()
        self.difusion.validar()
        self.cinetica.validar()
        self.mallado.validar()
        self.condiciones_frontera.validar()
        self.convergencia.validar()

        # Validaciones inter-tablas
        self._validar_consistencia_inter_tablas()

    def _validar_consistencia_inter_tablas(self) -> None:
        """Valida consistencia entre diferentes tablas de parámetros."""
        # Módulo de Thiele: φ = R √(k_app/D_eff) ≈ 0.124
        phi_calculado = self.geometria.R * np.sqrt(
            self.cinetica.k_app / self.difusion.D_eff
        )
        assert np.isclose(
            phi_calculado, 0.124, rtol=0.05
        ), f"Módulo de Thiele inconsistente: {phi_calculado} vs esperado 0.124"

        # Temperaturas deben coincidir
        assert (
            self.operacion.T == self.cinetica.T
        ), "Temperatura debe ser consistente entre operación y cinética"

        # Δr debe calcularse correctamente
        dr_esperado = self.geometria.R / (self.mallado.nr - 1)
        assert np.isclose(
            self.mallado.dr, dr_esperado, rtol=1e-6
        ), f"Δr inconsistente: {self.mallado.dr} vs {dr_esperado}"

        # Δθ debe calcularse correctamente
        dtheta_esperado = 2 * np.pi / (self.mallado.ntheta - 1)
        assert np.isclose(
            self.mallado.dtheta, dtheta_esperado, rtol=1e-6
        ), f"Δθ inconsistente: {self.mallado.dtheta} vs {dtheta_esperado}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte todos los parámetros a un diccionario plano.

        Returns
        -------
        Dict[str, Any]
            Diccionario con todos los parámetros, incluyendo acceso
            directo a valores clave como D_eff, k_app, etc.

        Examples
        --------
        >>> params = ParametrosMaestros()
        >>> params_dict = params.to_dict()
        >>> print(params_dict['D_eff'])  # 1.04e-6
        """
        params_dict = {
            "geometria": {
                "D": self.geometria.D,
                "R": self.geometria.R,
                "r1": self.geometria.r1,
                "r2": self.geometria.r2,
                "theta1": self.geometria.theta1,
                "theta2": self.geometria.theta2,
                "L": self.geometria.L,
                "P": self.geometria.P,
            },
            "operacion": {
                "T": self.operacion.T,
                "P": self.operacion.P,
                "y_CO": self.operacion.y_CO,
                "C_bulk": self.operacion.C_bulk,
                "u_s": self.operacion.u_s,
                "epsilon_b": self.operacion.epsilon_b,
                "u_i": self.operacion.u_i,
                "C_inicial": self.operacion.C_inicial,
            },
            "gas": {
                "rho_gas": self.gas.rho_gas,
                "mu": self.gas.mu,
                "nu": self.gas.nu,
                "D_CO_aire": self.gas.D_CO_aire,
                "Sc": self.gas.Sc,
                "M_aire": self.gas.M_aire,
            },
            "transferencia": {
                "d_p": self.transferencia.d_p,
                "Re_p": self.transferencia.Re_p,
                "Sh": self.transferencia.Sh,
                "k_c": self.transferencia.k_c,
            },
            "difusion": {
                "epsilon": self.difusion.epsilon,
                "tau": self.difusion.tau,
                "r_poro": self.difusion.r_poro,
                "lambda_mfp": self.difusion.lambda_mfp,
                "Kn": self.difusion.Kn,
                "D_Kn": self.difusion.D_Kn,
                "D_comb": self.difusion.D_comb,
                "D_eff": self.difusion.D_eff,
            },
            "cinetica": {
                "k0": self.cinetica.k0,
                "Ea": self.cinetica.Ea,
                "k_app": self.cinetica.k_app,
                "T": self.cinetica.T,
                "n": self.cinetica.n,
            },
            "mallado": {
                "nr": self.mallado.nr,
                "ntheta": self.mallado.ntheta,
                "dr": self.mallado.dr,
                "dtheta": self.mallado.dtheta,
                "dt": self.mallado.dt,
            },
            "condiciones_frontera": {
                "C_inicial": self.condiciones_frontera.C_inicial,
                "tipo_centro": self.condiciones_frontera.tipo_centro,
                "tipo_superficie": self.condiciones_frontera.tipo_superficie,
                "tipo_angular": self.condiciones_frontera.tipo_angular,
            },
            "convergencia": {
                "tol_convergencia": self.convergencia.tol_convergencia,
                "pasos_consecutivos": self.convergencia.pasos_consecutivos,
                "tol_balance_masa": self.convergencia.tol_balance_masa,
                "tol_validacion_dimensional": self.convergencia.tol_validacion_dimensional,
            },
        }

        # Agregar accesos directos a valores clave
        params_dict["D_eff"] = self.difusion.D_eff
        params_dict["k_app"] = self.cinetica.k_app
        params_dict["C_bulk"] = self.operacion.C_bulk
        params_dict["R"] = self.geometria.R
        params_dict["T"] = self.operacion.T

        return params_dict

    def __str__(self) -> str:
        """Representación en string legible."""
        return (
            "ParametrosMaestros(\n"
            f"  Geometría: D={self.geometria.D}m, R={self.geometria.R}m\n"
            f"  Operación: T={self.operacion.T}K, P={self.operacion.P}Pa\n"
            f"  Difusión: D_eff={self.difusion.D_eff:.3e} m²/s\n"
            f"  Cinética: k_app={self.cinetica.k_app:.3e} s⁻¹\n"
            f"  Mallado: {self.mallado.nr}×{self.mallado.ntheta} nodos\n"
            ")"
        )


# ============================================================================
# INSTANCIA GLOBAL (para acceso conveniente)
# ============================================================================


# Crear instancia global única de parámetros
PARAMETROS = ParametrosMaestros()


# ============================================================================
# FUNCIÓN DE UTILIDAD
# ============================================================================


def obtener_parametros() -> ParametrosMaestros:
    """
    Obtiene la instancia global de parámetros.

    Returns
    -------
    ParametrosMaestros
        Instancia global con todos los parámetros del proyecto

    Examples
    --------
    >>> from src.config.parametros import obtener_parametros
    >>> params = obtener_parametros()
    >>> print(params.difusion.D_eff)
    """
    return PARAMETROS


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
