"""
Tests unitarios para el módulo de parámetros del proyecto.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD (Test-Driven Development):
- Estos tests se escriben ANTES de implementar src/config/parametros.py
- Inicialmente deben FALLAR (RED)
- Luego implementamos código para hacerlos pasar (GREEN)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE GEOMETRÍA (TABLA I)
# ============================================================================


def test_geometria_params_existe():
    """La clase GeometriaParams debe existir."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert params is not None


def test_geometria_radio_es_mitad_diametro():
    """El radio debe ser exactamente D/2."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert_allclose(params.R, params.D / 2, rtol=1e-10)


def test_geometria_defecto_radios_ordenados():
    """Los radios del defecto deben cumplir: 0 < r1 < r2 < R."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert 0 < params.r1, "r1 debe ser positivo"
    assert params.r1 < params.r2, "r1 debe ser menor que r2"
    assert params.r2 < params.R, "r2 debe ser menor que R"


def test_geometria_defecto_r1_es_R_sobre_3():
    """r1 debe ser R/3 según enunciado."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert_allclose(params.r1, params.R / 3, rtol=1e-6)


def test_geometria_defecto_r2_es_2R_sobre_3():
    """r2 debe ser 2R/3 según enunciado."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert_allclose(params.r2, 2 * params.R / 3, rtol=1e-6)


def test_geometria_theta2_es_45_grados():
    """theta2 debe ser π/4 rad (45°)."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert_allclose(params.theta2, np.pi / 4, rtol=1e-6)


def test_geometria_validar_metodo_existe():
    """Debe existir método validar() que chequea consistencia."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()
    assert hasattr(params, "validar"), "Debe tener método validar()"
    params.validar()  # No debe lanzar excepción


# ============================================================================
# TESTS DE OPERACIÓN (TABLA II)
# ============================================================================


def test_operacion_params_existe():
    """La clase OperacionParams debe existir."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert params is not None


def test_operacion_temperatura_673K():
    """Temperatura de operación debe ser 673 K."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert_allclose(params.T, 673, rtol=1e-10)


def test_operacion_presion_1atm():
    """Presión debe ser 1 atm = 101325 Pa."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert_allclose(params.P, 101325, rtol=1e-10)


def test_operacion_concentracion_bulk_800ppm():
    """C_bulk debe ser 0.0145 mol/m³ (800 ppm a 673K)."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert_allclose(params.C_bulk, 0.0145, rtol=0.01)  # 1% tolerancia


def test_operacion_velocidad_intersticial():
    """u_i debe ser u_s / epsilon_b = 0.3 / 0.4 = 0.75 m/s."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert_allclose(params.u_i, params.u_s / params.epsilon_b, rtol=1e-6)


# ============================================================================
# TESTS DE PROPIEDADES DEL GAS (TABLA III)
# ============================================================================


def test_gas_params_existe():
    """La clase GasParams debe existir."""
    from src.config.parametros import GasParams

    params = GasParams()
    assert params is not None


def test_gas_numero_schmidt():
    """Sc debe ser ν / D_CO_aire."""
    from src.config.parametros import GasParams

    params = GasParams()
    Sc_calculado = params.nu / params.D_CO_aire
    assert_allclose(params.Sc, Sc_calculado, rtol=0.01)


def test_gas_viscosidad_cinematica():
    """ν debe ser μ / ρ."""
    from src.config.parametros import GasParams

    params = GasParams()
    nu_calculado = params.mu / params.rho_gas
    assert_allclose(params.nu, nu_calculado, rtol=0.01)


# ============================================================================
# TESTS DE TRANSFERENCIA DE MASA (TABLA IV)
# ============================================================================


def test_transferencia_params_existe():
    """La clase TransferenciaParams debe existir."""
    from src.config.parametros import TransferenciaParams

    params = TransferenciaParams()
    assert params is not None


def test_transferencia_reynolds_19():
    """Reynolds de partícula debe ser ~19."""
    from src.config.parametros import TransferenciaParams

    params = TransferenciaParams()
    assert 18 < params.Re_p < 20, f"Re_p debe ser ~19, obtenido: {params.Re_p}"


def test_transferencia_sherwood_wakao_funazkri():
    """Sherwood debe calcularse con Wakao-Funazkri."""
    from src.config.parametros import TransferenciaParams

    params = TransferenciaParams()
    # Sh = 2 + 1.1 * Re_p^0.6 * Sc^(1/3)
    # Debe dar ~7.78
    assert 7.5 < params.Sh < 8.0, f"Sh debe ser ~7.78, obtenido: {params.Sh}"


def test_transferencia_kc_de_sherwood():
    """k_c debe ser Sh × D_CO_aire / d_p."""
    from src.config.parametros import TransferenciaParams, GasParams

    params_trans = TransferenciaParams()
    params_gas = GasParams()

    k_c_calculado = params_trans.Sh * params_gas.D_CO_aire / params_trans.d_p
    assert_allclose(params_trans.k_c, k_c_calculado, rtol=0.01)


# ============================================================================
# TESTS DE DIFUSIÓN INTRAPARTICULAR (TABLA V)
# ============================================================================


def test_difusion_params_existe():
    """La clase DifusionParams debe existir."""
    from src.config.parametros import DifusionParams

    params = DifusionParams()
    assert params is not None


def test_difusion_numero_knudsen_mayor_1():
    """Kn debe ser >> 1 (régimen Knudsen dominante)."""
    from src.config.parametros import DifusionParams

    params = DifusionParams()
    assert params.Kn > 10, f"Kn debe ser >> 1, obtenido: {params.Kn}"


def test_difusion_D_eff_valor_esperado():
    """D_eff debe ser ~1.04×10⁻⁶ m²/s según Tabla V."""
    from src.config.parametros import DifusionParams

    params = DifusionParams()
    assert_allclose(params.D_eff, 1.04e-6, rtol=0.05)  # 5% tolerancia


def test_difusion_D_eff_formula():
    """D_eff debe ser ε × D_comb / τ."""
    from src.config.parametros import DifusionParams

    params = DifusionParams()
    D_eff_calculado = params.epsilon * params.D_comb / params.tau
    assert_allclose(params.D_eff, D_eff_calculado, rtol=0.01)


def test_difusion_modulo_thiele():
    """Módulo de Thiele debe ser ~0.124."""
    from src.config.parametros import DifusionParams, CineticaParams, GeometriaParams

    params_dif = DifusionParams()
    params_cin = CineticaParams()
    params_geo = GeometriaParams()

    # φ = R × √(k_app / D_eff)
    phi_calculado = params_geo.R * np.sqrt(params_cin.k_app / params_dif.D_eff)
    assert_allclose(phi_calculado, 0.124, rtol=0.05)


# ============================================================================
# TESTS DE CINÉTICA (TABLA VI)
# ============================================================================


def test_cinetica_params_existe():
    """La clase CineticaParams debe existir."""
    from src.config.parametros import CineticaParams

    params = CineticaParams()
    assert params is not None


def test_cinetica_k_app_arrhenius():
    """k_app debe calcularse con Arrhenius: k0 × exp(-Ea/RT)."""
    from src.config.parametros import CineticaParams, OperacionParams

    params_cin = CineticaParams()
    params_op = OperacionParams()

    R_gas = 8.314  # J/(mol·K)
    k_app_calculado = params_cin.k0 * np.exp(-params_cin.Ea / (R_gas * params_op.T))

    assert_allclose(params_cin.k_app, k_app_calculado, rtol=0.01)


def test_cinetica_k_app_valor_esperado():
    """k_app debe ser ~4.0×10⁻³ s⁻¹."""
    from src.config.parametros import CineticaParams

    params = CineticaParams()
    assert_allclose(params.k_app, 4.0e-3, rtol=0.05)


# ============================================================================
# TESTS DE MALLADO (TABLA VII)
# ============================================================================


def test_mallado_params_existe():
    """La clase MalladoParams debe existir."""
    from src.config.parametros import MalladoParams

    params = MalladoParams()
    assert params is not None


def test_mallado_nr_61_nodos():
    """Deben ser 61 nodos radiales."""
    from src.config.parametros import MalladoParams

    params = MalladoParams()
    assert params.nr == 61


def test_mallado_ntheta_96_nodos():
    """Deben ser 96 nodos angulares."""
    from src.config.parametros import MalladoParams

    params = MalladoParams()
    assert params.ntheta == 96


def test_mallado_dr_consistente():
    """Δr debe ser R/(nr-1)."""
    from src.config.parametros import MalladoParams, GeometriaParams

    params_malla = MalladoParams()
    params_geo = GeometriaParams()

    dr_esperado = params_geo.R / (params_malla.nr - 1)
    assert_allclose(params_malla.dr, dr_esperado, rtol=1e-6)


def test_mallado_dtheta_consistente():
    """Δθ debe ser 2π/(ntheta-1)."""
    from src.config.parametros import MalladoParams

    params = MalladoParams()
    dtheta_esperado = 2 * np.pi / (params.ntheta - 1)
    assert_allclose(params.dtheta, dtheta_esperado, rtol=1e-6)


# ============================================================================
# TESTS DE CONDICIONES DE FRONTERA (TABLA VIII)
# ============================================================================


def test_condiciones_frontera_params_existe():
    """La clase CondicionesFronteraParams debe existir."""
    from src.config.parametros import CondicionesFronteraParams

    params = CondicionesFronteraParams()
    assert params is not None


def test_condiciones_frontera_condicion_inicial_cero():
    """Condición inicial debe ser C = 0."""
    from src.config.parametros import CondicionesFronteraParams

    params = CondicionesFronteraParams()
    assert params.C_inicial == 0.0


# ============================================================================
# TESTS DE CONVERGENCIA (TABLA IX)
# ============================================================================


def test_convergencia_params_existe():
    """La clase ConvergenciaParams debe existir."""
    from src.config.parametros import ConvergenciaParams

    params = ConvergenciaParams()
    assert params is not None


def test_convergencia_tolerancia_1e_6():
    """Tolerancia de convergencia debe ser 1e-6."""
    from src.config.parametros import ConvergenciaParams

    params = ConvergenciaParams()
    assert params.tol_convergencia == 1e-6


def test_convergencia_balance_masa_tolerancia_1pct():
    """Tolerancia de balance de masa debe ser 1%."""
    from src.config.parametros import ConvergenciaParams

    params = ConvergenciaParams()
    assert params.tol_balance_masa == 0.01


# ============================================================================
# TESTS DE INTEGRACIÓN ENTRE PARÁMETROS
# ============================================================================


def test_parametros_todos_importables():
    """Todas las clases de parámetros deben ser importables."""
    from src.config.parametros import (
        GeometriaParams,
        OperacionParams,
        GasParams,
        TransferenciaParams,
        DifusionParams,
        CineticaParams,
        MalladoParams,
        CondicionesFronteraParams,
        ConvergenciaParams,
    )

    # Instanciar todas
    geo = GeometriaParams()
    op = OperacionParams()
    gas = GasParams()
    trans = TransferenciaParams()
    dif = DifusionParams()
    cin = CineticaParams()
    malla = MalladoParams()
    cf = CondicionesFronteraParams()
    conv = ConvergenciaParams()

    # Todas deben existir
    assert all([geo, op, gas, trans, dif, cin, malla, cf, conv])


def test_parametros_clase_maestra_existe():
    """Debe existir clase ParametrosMaestros que agrupe todos."""
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    assert params is not None

    # Debe tener atributos para cada tabla
    assert hasattr(params, "geometria")
    assert hasattr(params, "operacion")
    assert hasattr(params, "gas")
    assert hasattr(params, "transferencia")
    assert hasattr(params, "difusion")
    assert hasattr(params, "cinetica")
    assert hasattr(params, "mallado")
    assert hasattr(params, "condiciones_frontera")
    assert hasattr(params, "convergencia")


def test_parametros_maestros_validar_todo():
    """ParametrosMaestros debe tener método validar_todo()."""
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    assert hasattr(params, "validar_todo")

    # Ejecutar validación (no debe fallar)
    params.validar_todo()


def test_parametros_maestros_diccionario():
    """ParametrosMaestros debe poder convertirse a diccionario."""
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    assert hasattr(params, "to_dict")

    params_dict = params.to_dict()
    assert isinstance(params_dict, dict)
    assert "geometria" in params_dict
    assert "D_eff" in params_dict  # Debe tener acceso directo a valores clave


# ============================================================================
# TESTS DE VALIDACIÓN DIMENSIONAL
# ============================================================================


def test_parametros_unidades_documentadas():
    """Todas las clases deben tener unidades documentadas en docstrings."""
    from src.config.parametros import GeometriaParams

    # Verificar que el docstring existe y menciona unidades
    assert GeometriaParams.__doc__ is not None
    docstring = GeometriaParams.__doc__.lower()
    assert "tabla" in docstring or "parámetro" in docstring


def test_validacion_dimensional_temperatura_positiva():
    """Temperatura debe ser positiva (validación física)."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()
    assert params.T > 0, "Temperatura debe ser positiva"
    assert params.T > 273, "Temperatura debe ser > 0°C"


def test_validacion_dimensional_difusividad_rango_fisico():
    """Difusividad debe estar en rango físico razonable."""
    from src.config.parametros import DifusionParams

    params = DifusionParams()
    # Difusividades típicas: 1e-10 a 1e-4 m²/s
    assert 1e-10 < params.D_eff < 1e-4, f"D_eff fuera de rango físico: {params.D_eff}"


def test_validacion_dimensional_constante_cinetica_positiva():
    """Constante cinética debe ser positiva."""
    from src.config.parametros import CineticaParams

    params = CineticaParams()
    assert params.k_app > 0, "k_app debe ser positiva"
    assert params.k_app < 1.0, "k_app sospechosamente grande"


# ============================================================================
# TESTS DE CONSISTENCIA INTER-TABLAS
# ============================================================================


def test_consistencia_modulo_thiele_correcto():
    """Módulo de Thiele calculado debe coincidir con Tabla V."""
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    # Calcular φ = R √(k_app/D_eff)
    phi_calculado = params.geometria.R * np.sqrt(
        params.cinetica.k_app / params.difusion.D_eff
    )

    # Verificar contra valor en Tabla V: φ = 0.124
    assert_allclose(phi_calculado, 0.124, rtol=0.05)


def test_consistencia_c_bulk_de_ley_gases_ideales():
    """C_bulk debe calcularse correctamente con ley de gases ideales."""
    from src.config.parametros import OperacionParams

    params = OperacionParams()

    # C = (y × P) / (R × T)
    R_gas = 8.314  # J/(mol·K)
    y_CO = 800e-6  # ppm a fracción

    C_bulk_esperado = (y_CO * params.P) / (R_gas * params.T)

    assert_allclose(params.C_bulk, C_bulk_esperado, rtol=0.01)


# ============================================================================
# TESTS DE VALIDACIÓN DE RANGOS FÍSICOS
# ============================================================================


@pytest.mark.parametrize(
    "clase_nombre,attr,min_val,max_val",
    [
        ("GeometriaParams", "D", 0.001, 0.010),  # Diámetro en rango razonable
        ("GeometriaParams", "R", 0.0005, 0.005),  # Radio en rango razonable
        ("OperacionParams", "T", 300, 1000),  # Temperatura razonable
        ("OperacionParams", "P", 1e4, 1e6),  # Presión razonable
        ("DifusionParams", "epsilon", 0.2, 0.8),  # Porosidad física
        ("DifusionParams", "tau", 1.5, 10),  # Tortuosidad física
        ("MalladoParams", "nr", 10, 200),  # Número de nodos razonable
        ("MalladoParams", "ntheta", 10, 200),  # Número de nodos razonable
    ],
)
def test_parametros_en_rangos_fisicos(clase_nombre, attr, min_val, max_val):
    """Parámetros deben estar en rangos físicamente razonables."""
    from src.config import parametros

    clase = getattr(parametros, clase_nombre)
    params = clase()
    valor = getattr(params, attr)

    assert (
        min_val <= valor <= max_val
    ), f"{clase_nombre}.{attr} = {valor} fuera de rango [{min_val}, {max_val}]"


# ============================================================================
# TESTS DE INMUTABILIDAD (OPCIONAL PERO RECOMENDADO)
# ============================================================================


def test_parametros_son_inmutables_frozen():
    """Parámetros deben ser inmutables (frozen=True en dataclass)."""
    from src.config.parametros import GeometriaParams

    params = GeometriaParams()

    # Intentar modificar debe fallar
    with pytest.raises(AttributeError):
        params.D = 0.005  # No debe permitir modificación


# ============================================================================
# FIN DE TESTS
# ============================================================================
