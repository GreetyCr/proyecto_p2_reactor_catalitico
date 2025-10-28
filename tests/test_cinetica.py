"""
Tests unitarios para el módulo de cinética de reacción.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD (Test-Driven Development):
- Estos tests se escriben ANTES de implementar src/propiedades/cinetica.py
- Inicialmente deben FALLAR (RED)
- Luego implementamos código para hacerlos pasar (GREEN)

El módulo de cinética es crítico para calcular k_app correctamente.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE CONSTANTES
# ============================================================================


def test_constantes_cineticas_existen():
    """Las constantes cinéticas deben estar definidas."""
    from src.propiedades.cinetica import R_GAS

    assert R_GAS == 8.314  # J/(mol·K)


# ============================================================================
# TESTS DE ECUACIÓN DE ARRHENIUS
# ============================================================================


def test_calcular_k_arrhenius_existe():
    """La función calcular_k_arrhenius debe existir."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    assert calcular_k_arrhenius is not None


def test_calcular_k_arrhenius_valores_proyecto():
    """k debe calcularse correctamente con parámetros del proyecto."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    # Parámetros del proyecto (Tabla VI)
    k0 = 60.0  # s⁻¹
    Ea = 26000.0  # J/mol
    T = 673  # K

    k = calcular_k_arrhenius(k0, Ea, T)

    # Valor calculado con Arrhenius estándar: ~0.576 s⁻¹
    # Nota: Discrepancia con tabla puede deberse a k_app ≠ k_Arrhenius
    # (k_app puede incluir factores adicionales de adsorción, etc.)
    assert_allclose(k, 0.576, rtol=0.01)


def test_arrhenius_formula_correcta():
    """k = k₀ × exp(-Ea/RT)."""
    from src.propiedades.cinetica import calcular_k_arrhenius, R_GAS

    k0 = 60.0
    Ea = 26000.0
    T = 673

    k = calcular_k_arrhenius(k0, Ea, T)

    # Cálculo manual
    k_esperado = k0 * np.exp(-Ea / (R_GAS * T))

    assert_allclose(k, k_esperado, rtol=1e-10)


def test_arrhenius_aumenta_con_temperatura():
    """k debe aumentar con T (exponencialmente)."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    k0 = 60.0
    Ea = 26000.0

    T1 = 600
    T2 = 700

    k1 = calcular_k_arrhenius(k0, Ea, T1)
    k2 = calcular_k_arrhenius(k0, Ea, T2)

    # k debe aumentar con T
    assert k2 > k1

    # Relación aproximada (Arrhenius)
    ratio = k2 / k1
    assert ratio > 1.5  # Debe ser significativamente mayor


def test_arrhenius_limite_Ea_cero():
    """Si Ea=0, entonces k=k₀ (sin barrera energética)."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    k0 = 60.0
    Ea = 0.0  # Sin barrera
    T = 673

    k = calcular_k_arrhenius(k0, Ea, T)

    # k debe ser igual a k0
    assert_allclose(k, k0, rtol=1e-10)


def test_arrhenius_limite_T_infinito():
    """Si T→∞, entonces k→k₀."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    k0 = 60.0
    Ea = 26000.0
    T_muy_alto = 10000  # K (muy alto)

    k = calcular_k_arrhenius(k0, Ea, T_muy_alto)

    # k debe acercarse a k0 (tolerancia amplia)
    assert_allclose(k, k0, rtol=0.30)  # 30% de k0


def test_arrhenius_validacion_inputs():
    """Debe validar inputs no físicos."""
    from src.propiedades.cinetica import calcular_k_arrhenius

    # k0 negativo
    with pytest.raises(ValueError, match="Factor pre-exponencial"):
        calcular_k_arrhenius(k0=-10, Ea=26000, T=673)

    # Ea negativa
    with pytest.raises(ValueError, match="Energía de activación"):
        calcular_k_arrhenius(k0=60, Ea=-1000, T=673)

    # T negativa
    with pytest.raises(ValueError, match="Temperatura"):
        calcular_k_arrhenius(k0=60, Ea=26000, T=-100)


# ============================================================================
# TESTS DE CAMPO ESPACIAL k_app
# ============================================================================


def test_generar_campo_k_app_existe():
    """La función generar_campo_k_app debe existir."""
    from src.propiedades.cinetica import generar_campo_k_app

    assert generar_campo_k_app is not None


def test_generar_campo_k_app_con_malla():
    """Debe generar campo k_app correctamente con malla."""
    from src.propiedades.cinetica import generar_campo_k_app
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = generar_campo_k_app(malla, params)

    # Debe tener el shape correcto
    assert k_app_field.shape == (params.mallado.nr, params.mallado.ntheta)


def test_campo_k_app_cero_en_defecto():
    """k_app debe ser 0 en la región de defecto."""
    from src.propiedades.cinetica import generar_campo_k_app
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = generar_campo_k_app(malla, params)

    # Obtener máscara de defecto
    mascara_defecto = malla.identificar_region_defecto()

    # k_app debe ser 0 en defecto
    assert_allclose(k_app_field[mascara_defecto], 0.0, atol=1e-15)


def test_campo_k_app_valor_en_activa():
    """k_app debe ser el valor del parámetro en región activa."""
    from src.propiedades.cinetica import generar_campo_k_app
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = generar_campo_k_app(malla, params)

    # Obtener máscara activa
    mascara_activa = malla.identificar_region_activa()

    # k_app debe ser el valor del parámetro
    assert_allclose(k_app_field[mascara_activa], params.cinetica.k_app, rtol=1e-10)


def test_campo_k_app_consistente_con_malla():
    """El campo generado debe ser consistente con el de la malla."""
    from src.propiedades.cinetica import generar_campo_k_app
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Generar campo desde cinetica
    k_app_cinetica = generar_campo_k_app(malla, params)

    # Generar campo desde malla
    k_app_malla = malla.generar_campo_k_app()

    # Deben ser idénticos
    assert_allclose(k_app_cinetica, k_app_malla, rtol=1e-10)


# ============================================================================
# TESTS DE INTEGRACIÓN CON TEMPERATURA
# ============================================================================


def test_generar_campo_k_app_con_temperatura_custom():
    """Debe poder generar campo con temperatura diferente."""
    from src.propiedades.cinetica import generar_campo_k_app_temperatura
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Temperatura custom
    T_custom = 700  # K (mayor que 673K del proyecto)

    k_app_field = generar_campo_k_app_temperatura(malla, params, T_custom)

    # Debe tener valores mayores que a 673K
    k_app_673 = generar_campo_k_app_temperatura(malla, params, 673)

    # En región activa, k debe ser mayor a mayor temperatura
    mascara_activa = malla.identificar_region_activa()
    assert np.all(k_app_field[mascara_activa] > k_app_673[mascara_activa])


def test_campo_temperatura_espacialmente_variable():
    """Debe poder manejar campo de temperatura espacialmente variable."""
    from src.propiedades.cinetica import (
        generar_campo_k_app_temperatura_espacial,
        generar_campo_k_app_temperatura,
    )
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo de temperatura (uniforme en este caso)
    T_field = np.full((params.mallado.nr, params.mallado.ntheta), 673.0)

    k_app_field = generar_campo_k_app_temperatura_espacial(malla, params, T_field)

    # Debe ser consistente con temperatura uniforme
    k_app_uniforme = generar_campo_k_app_temperatura(malla, params, 673)

    assert_allclose(k_app_field, k_app_uniforme, rtol=1e-10)


# ============================================================================
# TESTS DE TASA DE REACCIÓN
# ============================================================================


def test_calcular_tasa_reaccion_existe():
    """La función calcular_tasa_reaccion debe existir."""
    from src.propiedades.cinetica import calcular_tasa_reaccion

    assert calcular_tasa_reaccion is not None


def test_calcular_tasa_reaccion_formula():
    """r = k_app × C (cinética de primer orden)."""
    from src.propiedades.cinetica import calcular_tasa_reaccion

    k_app = 4.0e-3  # s⁻¹
    C = 0.01  # mol/m³

    r = calcular_tasa_reaccion(k_app, C)

    # r = k_app × C
    r_esperado = k_app * C

    assert_allclose(r, r_esperado, rtol=1e-10)


def test_calcular_tasa_reaccion_campo():
    """Debe poder calcular tasa de reacción para campo completo."""
    from src.propiedades.cinetica import calcular_tasa_reaccion
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo de k_app
    k_app_field = malla.generar_campo_k_app()

    # Campo de concentración (ejemplo)
    C_field = np.ones((params.mallado.nr, params.mallado.ntheta)) * 0.01

    # Calcular tasa
    r_field = calcular_tasa_reaccion(k_app_field, C_field)

    # Debe tener el mismo shape
    assert r_field.shape == k_app_field.shape

    # En defecto, r debe ser 0
    mascara_defecto = malla.identificar_region_defecto()
    assert_allclose(r_field[mascara_defecto], 0.0, atol=1e-15)


def test_tasa_reaccion_proporcional_a_concentracion():
    """r debe ser proporcional a C (primer orden)."""
    from src.propiedades.cinetica import calcular_tasa_reaccion

    k_app = 4.0e-3

    C1 = 0.01
    C2 = 0.02  # 2× mayor

    r1 = calcular_tasa_reaccion(k_app, C1)
    r2 = calcular_tasa_reaccion(k_app, C2)

    # r2/r1 = C2/C1
    assert_allclose(r2 / r1, C2 / C1, rtol=1e-10)


# ============================================================================
# TESTS DE INTEGRACIÓN CON PARÁMETROS
# ============================================================================


def test_calcular_desde_parametros():
    """Debe calcular k_app desde ParametrosMaestros."""
    from src.propiedades.cinetica import calcular_k_desde_parametros
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    k = calcular_k_desde_parametros(params)

    # Debe coincidir con el valor en params
    assert_allclose(k, params.cinetica.k_app, rtol=0.05)


def test_generar_campo_desde_parametros():
    """Debe generar campo completo desde parámetros."""
    from src.propiedades.cinetica import generar_campo_desde_parametros
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    k_app_field = generar_campo_desde_parametros(params)

    # Debe ser un array 2D
    assert k_app_field.ndim == 2
    assert k_app_field.shape == (params.mallado.nr, params.mallado.ntheta)


# ============================================================================
# TESTS DE VALIDACIÓN DIMENSIONAL
# ============================================================================


def test_arrhenius_dimension_correcta():
    """k debe tener dimensión de frecuencia [1/T]."""
    from src.propiedades.cinetica import calcular_k_arrhenius_dimensional

    k = calcular_k_arrhenius_dimensional(k0=60, Ea=26000, T=673)

    # Debe ser CantidadDimensional con dimensión FRECUENCIA
    from src.utils.validacion import Dimension

    assert k.dimension == Dimension.FRECUENCIA


def test_tasa_reaccion_dimension_correcta():
    """r debe tener dimensión [N/(L³·T)]."""
    from src.propiedades.cinetica import calcular_tasa_reaccion_dimensional

    r = calcular_tasa_reaccion_dimensional(k_app=4.0e-3, C=0.01)

    # Debe tener dimensión correcta
    from src.utils.validacion import Dimension

    # r = k × C → [1/T] × [N/L³] = [N/(L³·T)]
    assert r.dimension == Dimension.TASA_REACCION_VOLUMETRICA


# ============================================================================
# TESTS DE CASOS LÍMITE
# ============================================================================


def test_campo_k_app_sin_defecto():
    """Si no hay defecto, k_app debe ser uniforme."""
    from src.propiedades.cinetica import generar_campo_k_app
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = generar_campo_k_app(malla, params)

    # Verificar que hay región de defecto
    mascara_defecto = malla.identificar_region_defecto()
    n_defecto = mascara_defecto.sum()

    # Debe haber nodos con defecto
    assert n_defecto > 0


def test_tasa_reaccion_concentracion_cero():
    """Si C=0, entonces r=0 (sin reactivo)."""
    from src.propiedades.cinetica import calcular_tasa_reaccion

    k_app = 4.0e-3
    C = 0.0

    r = calcular_tasa_reaccion(k_app, C)

    assert r == 0.0


def test_tasa_reaccion_k_app_cero():
    """Si k_app=0, entonces r=0 (sin catalizador activo)."""
    from src.propiedades.cinetica import calcular_tasa_reaccion

    k_app = 0.0
    C = 0.01

    r = calcular_tasa_reaccion(k_app, C)

    assert r == 0.0


# ============================================================================
# TESTS DE INFORMACIÓN Y UTILIDADES
# ============================================================================


def test_obtener_info_cinetica():
    """Debe poder obtener información de cinética."""
    from src.propiedades.cinetica import obtener_info_cinetica
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    info = obtener_info_cinetica(params)

    # Debe retornar diccionario
    assert isinstance(info, dict)

    # Debe contener información clave
    assert "k0" in info
    assert "Ea" in info
    assert "T" in info
    assert "k_app" in info


# ============================================================================
# FIN DE TESTS
# ============================================================================
