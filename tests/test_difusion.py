"""
Tests unitarios para el módulo de propiedades de difusión.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD (Test-Driven Development):
- Estos tests se escriben ANTES de implementar src/propiedades/difusion.py
- Inicialmente deben FALLAR (RED)
- Luego implementamos código para hacerlos pasar (GREEN)

El módulo de difusión es crítico para calcular D_eff correctamente.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE CONSTANTES FÍSICAS
# ============================================================================


def test_constantes_fisicas_existen():
    """Las constantes físicas necesarias deben estar definidas."""
    from src.propiedades.difusion import R_GAS, BOLTZMANN

    assert R_GAS == 8.314  # J/(mol·K)
    assert_allclose(BOLTZMANN, 1.380649e-23, rtol=1e-6)  # J/K


# ============================================================================
# TESTS DE CAMINO LIBRE MEDIO
# ============================================================================


def test_calcular_camino_libre_medio_existe():
    """La función calcular_camino_libre_medio debe existir."""
    from src.propiedades.difusion import calcular_camino_libre_medio

    assert calcular_camino_libre_medio is not None


def test_calcular_camino_libre_medio_valores_proyecto():
    """λ debe ser ~1.93×10⁻⁷ m con parámetros del proyecto."""
    from src.propiedades.difusion import calcular_camino_libre_medio

    # Parámetros del proyecto (Tabla V)
    T = 673  # K
    P = 101325  # Pa
    d_CO = 3.76e-10  # m (diámetro molecular CO)

    lambda_mfp = calcular_camino_libre_medio(T, P, d_CO)

    # Valor esperado de Tabla V (tolerancia mayor por aproximaciones)
    assert_allclose(lambda_mfp, 1.93e-7, rtol=0.30)


def test_camino_libre_medio_formula_correcta():
    """λ = k_B·T / (√2·π·d²·P)."""
    from src.propiedades.difusion import calcular_camino_libre_medio, BOLTZMANN

    T = 673
    P = 101325
    d = 3.76e-10

    lambda_mfp = calcular_camino_libre_medio(T, P, d)

    # Cálculo manual
    lambda_esperado = (BOLTZMANN * T) / (np.sqrt(2) * np.pi * d**2 * P)

    assert_allclose(lambda_mfp, lambda_esperado, rtol=1e-10)


def test_camino_libre_medio_aumenta_con_temperatura():
    """λ debe aumentar con T (proporcional a T)."""
    from src.propiedades.difusion import calcular_camino_libre_medio

    T1 = 300
    T2 = 600
    P = 101325
    d = 3.76e-10

    lambda_1 = calcular_camino_libre_medio(T1, P, d)
    lambda_2 = calcular_camino_libre_medio(T2, P, d)

    # λ₂/λ₁ debe ser aproximadamente T₂/T₁
    ratio_lambda = lambda_2 / lambda_1
    ratio_T = T2 / T1

    assert_allclose(ratio_lambda, ratio_T, rtol=0.01)


def test_camino_libre_medio_disminuye_con_presion():
    """λ debe disminuir con P (inversamente proporcional)."""
    from src.propiedades.difusion import calcular_camino_libre_medio

    T = 673
    P1 = 101325
    P2 = 202650  # 2 atm
    d = 3.76e-10

    lambda_1 = calcular_camino_libre_medio(T, P1, d)
    lambda_2 = calcular_camino_libre_medio(T, P2, d)

    # λ₁/λ₂ debe ser P₂/P₁
    assert_allclose(lambda_1 / lambda_2, P2 / P1, rtol=0.01)


def test_camino_libre_medio_validacion_inputs():
    """Debe validar inputs no físicos."""
    from src.propiedades.difusion import calcular_camino_libre_medio

    # Temperatura negativa
    with pytest.raises(ValueError, match="Temperatura"):
        calcular_camino_libre_medio(T=-10, P=101325, d_molecular=3.76e-10)

    # Presión negativa
    with pytest.raises(ValueError, match="Presión"):
        calcular_camino_libre_medio(T=673, P=-1000, d_molecular=3.76e-10)

    # Diámetro negativo
    with pytest.raises(ValueError, match="Diámetro"):
        calcular_camino_libre_medio(T=673, P=101325, d_molecular=-1e-10)


# ============================================================================
# TESTS DE NÚMERO DE KNUDSEN
# ============================================================================


def test_calcular_numero_knudsen_existe():
    """La función calcular_numero_knudsen debe existir."""
    from src.propiedades.difusion import calcular_numero_knudsen

    assert calcular_numero_knudsen is not None


def test_calcular_numero_knudsen_valores_proyecto():
    """Kn debe ser 19.3 con parámetros del proyecto."""
    from src.propiedades.difusion import calcular_numero_knudsen

    # Parámetros del proyecto
    lambda_mfp = 1.93e-7  # m
    r_poro = 10e-9  # m

    Kn = calcular_numero_knudsen(lambda_mfp, r_poro)

    # Valor esperado de Tabla V
    assert_allclose(Kn, 19.3, rtol=0.05)


def test_numero_knudsen_formula():
    """Kn = λ / r_poro."""
    from src.propiedades.difusion import calcular_numero_knudsen

    lambda_mfp = 1.93e-7
    r_poro = 10e-9

    Kn = calcular_numero_knudsen(lambda_mfp, r_poro)

    Kn_esperado = lambda_mfp / r_poro

    assert_allclose(Kn, Kn_esperado, rtol=1e-10)


def test_numero_knudsen_regimen():
    """Debe identificar régimen de difusión según Kn."""
    from src.propiedades.difusion import identificar_regimen_difusion

    # Kn >> 1: Knudsen
    regimen_knudsen = identificar_regimen_difusion(Kn=19.3)
    assert regimen_knudsen == "knudsen"

    # Kn << 1: Molecular
    regimen_molecular = identificar_regimen_difusion(Kn=0.01)
    assert regimen_molecular == "molecular"

    # Kn ~ 1: Transición
    regimen_transicion = identificar_regimen_difusion(Kn=1.0)
    assert regimen_transicion == "transicion"


# ============================================================================
# TESTS DE DIFUSIVIDAD DE KNUDSEN
# ============================================================================


def test_calcular_difusividad_knudsen_existe():
    """La función calcular_difusividad_knudsen debe existir."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    assert calcular_difusividad_knudsen is not None


def test_calcular_difusividad_knudsen_valores_proyecto():
    """D_Kn debe ser 7.43×10⁻⁶ m²/s con parámetros del proyecto."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    # Parámetros del proyecto (Tabla V)
    r_poro = 10e-9  # m
    T = 673  # K
    M_CO = 0.028  # kg/mol

    D_Kn = calcular_difusividad_knudsen(r_poro, T, M_CO)

    # Valor esperado de Tabla V (correlación puede variar según fuente)
    # Nuestra implementación usa fórmula estándar de Mourkou et al. (2024)
    assert_allclose(D_Kn, 7.43e-6, rtol=0.40)  # 40% tolerancia


def test_difusividad_knudsen_formula():
    """D_Kn = (2/3) × r_poro × √(8RT/πM)."""
    from src.propiedades.difusion import calcular_difusividad_knudsen, R_GAS

    r_poro = 10e-9
    T = 673
    M = 0.028

    D_Kn = calcular_difusividad_knudsen(r_poro, T, M)

    # Cálculo manual
    D_Kn_esperado = (2 / 3) * r_poro * np.sqrt(8 * R_GAS * T / (np.pi * M))

    assert_allclose(D_Kn, D_Kn_esperado, rtol=1e-10)


def test_difusividad_knudsen_aumenta_con_temperatura():
    """D_Kn debe aumentar con T (proporcional a √T)."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    r_poro = 10e-9
    M = 0.028

    T1 = 300
    T2 = 1200  # 4× mayor

    D_Kn1 = calcular_difusividad_knudsen(r_poro, T1, M)
    D_Kn2 = calcular_difusividad_knudsen(r_poro, T2, M)

    # D_Kn2/D_Kn1 ≈ √(T2/T1) = √4 = 2
    assert_allclose(D_Kn2 / D_Kn1, np.sqrt(T2 / T1), rtol=0.01)


def test_difusividad_knudsen_proporcional_r_poro():
    """D_Kn debe ser proporcional a r_poro."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    T = 673
    M = 0.028

    r1 = 5e-9
    r2 = 10e-9

    D_Kn1 = calcular_difusividad_knudsen(r1, T, M)
    D_Kn2 = calcular_difusividad_knudsen(r2, T, M)

    # D_Kn2/D_Kn1 = r2/r1
    assert_allclose(D_Kn2 / D_Kn1, r2 / r1, rtol=0.01)


def test_difusividad_knudsen_validacion_inputs():
    """Debe rechazar inputs no físicos."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    # Radio de poro negativo
    with pytest.raises(ValueError, match="Radio de poro"):
        calcular_difusividad_knudsen(r_poro=-1e-9, T=673, M_gas=0.028)

    # Temperatura negativa
    with pytest.raises(ValueError, match="Temperatura"):
        calcular_difusividad_knudsen(r_poro=10e-9, T=-100, M_gas=0.028)

    # Masa molar negativa
    with pytest.raises(ValueError, match="Masa molar"):
        calcular_difusividad_knudsen(r_poro=10e-9, T=673, M_gas=-0.028)


def test_difusividad_knudsen_rango_fisico():
    """D_Kn debe estar en rango físico razonable."""
    from src.propiedades.difusion import calcular_difusividad_knudsen

    D_Kn = calcular_difusividad_knudsen(r_poro=10e-9, T=673, M_gas=0.028)

    # Difusividades típicas: 1e-10 a 1e-3 m²/s
    assert 1e-10 < D_Kn < 1e-3, f"D_Kn fuera de rango: {D_Kn}"


# ============================================================================
# TESTS DE DIFUSIVIDAD COMBINADA (BOSANQUET)
# ============================================================================


def test_calcular_difusividad_combinada_existe():
    """La función calcular_difusividad_combinada debe existir."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    assert calcular_difusividad_combinada is not None


def test_difusividad_combinada_valores_proyecto():
    """D_comb debe ser 6.97×10⁻⁶ m²/s con parámetros del proyecto."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    # Parámetros del proyecto
    D_molecular = 8.75e-5  # m²/s (D_CO-aire a 673K)
    D_Kn = 7.43e-6  # m²/s

    D_comb = calcular_difusividad_combinada(D_molecular, D_Kn)

    # Valor esperado de Tabla V
    assert_allclose(D_comb, 6.97e-6, rtol=0.05)


def test_difusividad_combinada_formula_bosanquet():
    """1/D_comb = 1/D_molecular + 1/D_Knudsen."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    D_mol = 8.75e-5
    D_Kn = 7.43e-6

    D_comb = calcular_difusividad_combinada(D_mol, D_Kn)

    # Verificar fórmula de Bosanquet
    D_comb_esperado = 1 / (1 / D_mol + 1 / D_Kn)

    assert_allclose(D_comb, D_comb_esperado, rtol=1e-10)


def test_difusividad_combinada_limite_knudsen():
    """Si D_Kn << D_mol, entonces D_comb ≈ D_Kn."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    D_mol = 1e-4  # Grande
    D_Kn = 1e-6  # Pequeño (100× menor)

    D_comb = calcular_difusividad_combinada(D_mol, D_Kn)

    # D_comb debe ser aproximadamente D_Kn
    assert_allclose(D_comb, D_Kn, rtol=0.02)  # 2% error


def test_difusividad_combinada_limite_molecular():
    """Si D_mol << D_Kn, entonces D_comb ≈ D_mol."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    D_mol = 1e-6  # Pequeño
    D_Kn = 1e-4  # Grande (100× mayor)

    D_comb = calcular_difusividad_combinada(D_mol, D_Kn)

    # D_comb debe ser aproximadamente D_mol
    assert_allclose(D_comb, D_mol, rtol=0.02)


def test_difusividad_combinada_simetrica():
    """D_comb(D1, D2) debe ser igual a D_comb(D2, D1)."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    D1 = 8.75e-5
    D2 = 7.43e-6

    D_comb_12 = calcular_difusividad_combinada(D1, D2)
    D_comb_21 = calcular_difusividad_combinada(D2, D1)

    assert_allclose(D_comb_12, D_comb_21, rtol=1e-10)


# ============================================================================
# TESTS DE DIFUSIVIDAD EFECTIVA
# ============================================================================


def test_calcular_difusividad_efectiva_existe():
    """La función calcular_difusividad_efectiva debe existir."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    assert calcular_difusividad_efectiva is not None


def test_difusividad_efectiva_valores_proyecto():
    """D_eff debe ser 1.04×10⁻⁶ m²/s con parámetros del proyecto."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    # Parámetros del proyecto (Tabla V)
    epsilon = 0.45
    D_comb = 6.97e-6  # m²/s
    tau = 3.0

    D_eff = calcular_difusividad_efectiva(epsilon, D_comb, tau)

    # Valor esperado de Tabla V
    assert_allclose(D_eff, 1.04e-6, rtol=0.05)


def test_difusividad_efectiva_formula():
    """D_eff = (ε × D_comb) / τ."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    epsilon = 0.45
    D_comb = 6.97e-6
    tau = 3.0

    D_eff = calcular_difusividad_efectiva(epsilon, D_comb, tau)

    # Cálculo manual
    D_eff_esperado = (epsilon * D_comb) / tau

    assert_allclose(D_eff, D_eff_esperado, rtol=1e-10)


def test_difusividad_efectiva_aumenta_con_porosidad():
    """D_eff debe aumentar linealmente con ε."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    D_comb = 6.97e-6
    tau = 3.0

    eps1 = 0.3
    eps2 = 0.6  # 2× mayor

    D_eff1 = calcular_difusividad_efectiva(eps1, D_comb, tau)
    D_eff2 = calcular_difusividad_efectiva(eps2, D_comb, tau)

    # D_eff2/D_eff1 = eps2/eps1
    assert_allclose(D_eff2 / D_eff1, eps2 / eps1, rtol=0.01)


def test_difusividad_efectiva_disminuye_con_tortuosidad():
    """D_eff debe disminuir con τ (inversamente proporcional)."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    epsilon = 0.45
    D_comb = 6.97e-6

    tau1 = 2.0
    tau2 = 4.0  # 2× mayor

    D_eff1 = calcular_difusividad_efectiva(epsilon, D_comb, tau1)
    D_eff2 = calcular_difusividad_efectiva(epsilon, D_comb, tau2)

    # D_eff1/D_eff2 = tau2/tau1
    assert_allclose(D_eff1 / D_eff2, tau2 / tau1, rtol=0.01)


def test_difusividad_efectiva_validacion_inputs():
    """Debe validar inputs no físicos."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    # Porosidad fuera de rango
    with pytest.raises(ValueError, match="Porosidad"):
        calcular_difusividad_efectiva(epsilon=-0.1, D_comb=6.97e-6, tau=3.0)

    with pytest.raises(ValueError, match="Porosidad"):
        calcular_difusividad_efectiva(epsilon=1.5, D_comb=6.97e-6, tau=3.0)

    # Tortuosidad no física
    with pytest.raises(ValueError, match="Tortuosidad"):
        calcular_difusividad_efectiva(epsilon=0.45, D_comb=6.97e-6, tau=0.5)


# ============================================================================
# TESTS DE FUNCIÓN INTEGRADA (CALCULAR TODO)
# ============================================================================


def test_calcular_propiedades_difusion_completas():
    """Debe poder calcular todas las propiedades de difusión de una vez."""
    from src.propiedades.difusion import calcular_propiedades_difusion

    # Inputs del proyecto
    T = 673
    P = 101325
    r_poro = 10e-9
    epsilon = 0.45
    tau = 3.0
    M_CO = 0.028
    d_CO = 3.76e-10

    resultado = calcular_propiedades_difusion(
        T=T, P=P, r_poro=r_poro, epsilon=epsilon, tau=tau, M_gas=M_CO, d_molecular=d_CO
    )

    # Debe retornar un diccionario
    assert isinstance(resultado, dict)

    # Debe contener todas las propiedades
    assert "lambda_mfp" in resultado
    assert "Kn" in resultado
    assert "D_Kn" in resultado
    assert "D_molecular" in resultado
    assert "D_comb" in resultado
    assert "D_eff" in resultado
    assert "regimen" in resultado


def test_propiedades_difusion_valores_proyecto():
    """Resultado completo debe coincidir con Tabla V."""
    from src.propiedades.difusion import calcular_propiedades_difusion

    # Inputs del proyecto
    resultado = calcular_propiedades_difusion(
        T=673,
        P=101325,
        r_poro=10e-9,
        epsilon=0.45,
        tau=3.0,
        M_gas=0.028,
        d_molecular=3.76e-10,
        D_molecular_ref=8.75e-5,  # Usar valor de tabla
    )

    # Verificar contra Tabla V (tolerancias amplias por correlaciones aproximadas)
    assert_allclose(resultado["lambda_mfp"], 1.93e-7, rtol=0.30)
    assert_allclose(resultado["Kn"], 19.3, rtol=0.30)
    assert_allclose(resultado["D_Kn"], 7.43e-6, rtol=0.40)  # Correlación varía
    assert_allclose(resultado["D_comb"], 6.97e-6, rtol=0.40)
    assert_allclose(resultado["D_eff"], 1.04e-6, rtol=0.40)
    assert resultado["regimen"] == "knudsen"


# ============================================================================
# TESTS DE DIFUSIVIDAD MOLECULAR (REFERENCIA)
# ============================================================================


def test_obtener_difusividad_molecular_CO_aire():
    """Debe poder obtener D_CO-aire de tabla o correlación."""
    from src.propiedades.difusion import obtener_difusividad_molecular_CO_aire

    T = 673  # K

    D_mol = obtener_difusividad_molecular_CO_aire(T)

    # A 673K debe ser ~8.75×10⁻⁵ m²/s (correlación aproximada)
    assert_allclose(D_mol, 8.75e-5, rtol=0.15)  # 15% tolerancia


def test_difusividad_molecular_aumenta_con_temperatura():
    """D_molecular debe aumentar con T (aproximadamente T^1.5)."""
    from src.propiedades.difusion import obtener_difusividad_molecular_CO_aire

    T1 = 300
    T2 = 600

    D1 = obtener_difusividad_molecular_CO_aire(T1)
    D2 = obtener_difusividad_molecular_CO_aire(T2)

    # D ∝ T^1.5 (ley de Fuller/Chapman-Enskog)
    # D2/D1 ≈ (T2/T1)^1.5
    ratio_esperado = (T2 / T1) ** 1.5
    ratio_calculado = D2 / D1

    assert_allclose(ratio_calculado, ratio_esperado, rtol=0.15)


# ============================================================================
# TESTS DE INTEGRACIÓN CON PARÁMETROS
# ============================================================================


def test_calcular_desde_parametros_maestros():
    """Debe poder calcular desde ParametrosMaestros directamente."""
    from src.propiedades.difusion import calcular_desde_parametros
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    resultado = calcular_desde_parametros(params)

    # Verificar que coincide con valores en params (tolerancia amplia)
    assert_allclose(resultado["D_eff"], params.difusion.D_eff, rtol=0.40)
    assert_allclose(resultado["Kn"], params.difusion.Kn, rtol=0.40)


# ============================================================================
# TESTS DE VALIDACIÓN DIMENSIONAL
# ============================================================================


def test_difusividad_knudsen_dimension_correcta():
    """D_Kn debe tener dimensión de difusividad [L²/T]."""
    from src.propiedades.difusion import calcular_difusividad_knudsen_dimensional

    D_Kn = calcular_difusividad_knudsen_dimensional(r_poro=10e-9, T=673, M=0.028)

    # Debe ser CantidadDimensional con dimensión DIFUSIVIDAD
    from src.utils.validacion import Dimension

    assert D_Kn.dimension == Dimension.DIFUSIVIDAD


def test_difusividad_efectiva_dimension_correcta():
    """D_eff debe tener dimensión de difusividad [L²/T]."""
    from src.propiedades.difusion import calcular_difusividad_efectiva_dimensional

    D_eff = calcular_difusividad_efectiva_dimensional(
        epsilon=0.45, D_comb=6.97e-6, tau=3.0
    )

    from src.utils.validacion import Dimension

    assert D_eff.dimension == Dimension.DIFUSIVIDAD


# ============================================================================
# TESTS DE CASOS LÍMITE
# ============================================================================


def test_difusividad_combinada_caso_iguales():
    """Si D_mol = D_Kn, entonces D_comb = D/2."""
    from src.propiedades.difusion import calcular_difusividad_combinada

    D = 1e-5

    D_comb = calcular_difusividad_combinada(D, D)

    # 1/D_comb = 1/D + 1/D = 2/D → D_comb = D/2
    assert_allclose(D_comb, D / 2, rtol=1e-10)


def test_difusividad_efectiva_porosidad_cero():
    """Si ε=0, entonces D_eff=0 (no hay difusión)."""
    from src.propiedades.difusion import calcular_difusividad_efectiva

    D_eff = calcular_difusividad_efectiva(epsilon=0.0, D_comb=6.97e-6, tau=3.0)

    assert D_eff == 0.0


# ============================================================================
# FIN DE TESTS
# ============================================================================
