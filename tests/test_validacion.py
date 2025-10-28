"""
Tests unitarios para el sistema de validación dimensional.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD (Test-Driven Development):
- Estos tests se escriben ANTES de implementar src/utils/validacion.py
- Inicialmente deben FALLAR (RED)
- Luego implementamos código para hacerlos pasar (GREEN)

El sistema de validación dimensional es CRÍTICO según reglas:
- Toda ecuación debe pasar validación dimensional
- Previene bugs sutiles de unidades
- Documenta automáticamente las dimensiones esperadas
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE LA CLASE DIMENSION (ENUM)
# ============================================================================


def test_dimension_enum_existe():
    """La clase Dimension (enum) debe existir."""
    from src.utils.validacion import Dimension

    assert Dimension is not None


def test_dimension_tiene_dimensiones_fundamentales():
    """Dimension debe tener las dimensiones fundamentales."""
    from src.utils.validacion import Dimension

    # Dimensiones fundamentales SI
    assert hasattr(Dimension, "LONGITUD")
    assert hasattr(Dimension, "TIEMPO")
    assert hasattr(Dimension, "MASA")
    assert hasattr(Dimension, "TEMPERATURA")
    assert hasattr(Dimension, "CANTIDAD_SUSTANCIA")


def test_dimension_tiene_dimensiones_derivadas():
    """Dimension debe tener dimensiones derivadas comunes."""
    from src.utils.validacion import Dimension

    # Dimensiones derivadas relevantes al proyecto
    assert hasattr(Dimension, "VELOCIDAD")
    assert hasattr(Dimension, "DIFUSIVIDAD")
    assert hasattr(Dimension, "CONCENTRACION")
    assert hasattr(Dimension, "ADIMENSIONAL")


def test_dimension_valores_correctos():
    """Los valores de Dimension deben ser strings descriptivos."""
    from src.utils.validacion import Dimension

    assert Dimension.LONGITUD.value == "L"
    assert Dimension.TIEMPO.value == "T"
    assert Dimension.DIFUSIVIDAD.value == "L²/T"
    assert Dimension.CONCENTRACION.value == "N/L³"
    assert Dimension.ADIMENSIONAL.value == "1"


# ============================================================================
# TESTS DE LA CLASE CantidadDimensional
# ============================================================================


def test_cantidad_dimensional_clase_existe():
    """La clase CantidadDimensional debe existir."""
    from src.utils.validacion import CantidadDimensional

    assert CantidadDimensional is not None


def test_cantidad_dimensional_crear_instancia():
    """Debe poder crear una CantidadDimensional."""
    from src.utils.validacion import CantidadDimensional, Dimension

    longitud = CantidadDimensional(
        valor=0.002, dimension=Dimension.LONGITUD, nombre="R"
    )

    assert longitud.valor == 0.002
    assert longitud.dimension == Dimension.LONGITUD
    assert longitud.nombre == "R"


def test_cantidad_dimensional_multiplicacion_por_escalar():
    """Multiplicar por escalar debe mantener dimensión."""
    from src.utils.validacion import CantidadDimensional, Dimension

    R = CantidadDimensional(0.002, Dimension.LONGITUD, "R")
    resultado = R * 2

    assert resultado.valor == 0.004
    assert resultado.dimension == Dimension.LONGITUD


def test_cantidad_dimensional_division_por_escalar():
    """Dividir por escalar debe mantener dimensión."""
    from src.utils.validacion import CantidadDimensional, Dimension

    D = CantidadDimensional(0.004, Dimension.LONGITUD, "D")
    resultado = D / 2

    assert resultado.valor == 0.002
    assert resultado.dimension == Dimension.LONGITUD


def test_cantidad_dimensional_multiplicacion_entre_cantidades():
    """Multiplicar dos cantidades debe combinar dimensiones."""
    from src.utils.validacion import CantidadDimensional, Dimension

    velocidad = CantidadDimensional(0.3, Dimension.VELOCIDAD, "u")
    tiempo = CantidadDimensional(10, Dimension.TIEMPO, "t")

    # u × t = (L/T) × T = L (longitud)
    distancia = velocidad * tiempo

    assert distancia.valor == 3.0
    # La dimensión resultante debe ser LONGITUD
    assert distancia.dimension == Dimension.LONGITUD


def test_cantidad_dimensional_division_entre_cantidades():
    """Dividir dos cantidades debe combinar dimensiones."""
    from src.utils.validacion import CantidadDimensional, Dimension

    longitud = CantidadDimensional(10, Dimension.LONGITUD, "L")
    tiempo = CantidadDimensional(2, Dimension.TIEMPO, "t")

    # L / T = velocidad
    velocidad = longitud / tiempo

    assert velocidad.valor == 5.0
    assert velocidad.dimension == Dimension.VELOCIDAD


def test_cantidad_dimensional_repr():
    """__repr__ debe mostrar valor y unidad."""
    from src.utils.validacion import CantidadDimensional, Dimension

    D_eff = CantidadDimensional(1.04e-6, Dimension.DIFUSIVIDAD, "D_eff")
    repr_str = repr(D_eff)

    assert "1.04e-06" in repr_str or "1.04e-6" in repr_str
    assert "L²/T" in repr_str or "m²/s" in repr_str


def test_cantidad_dimensional_validar_rango():
    """Debe poder validar que un valor esté en un rango."""
    from src.utils.validacion import CantidadDimensional, Dimension

    temperatura = CantidadDimensional(673, Dimension.TEMPERATURA, "T")

    # En rango válido
    assert temperatura.validar_rango(300, 1000) is True

    # Fuera de rango
    assert temperatura.validar_rango(700, 1000) is False


# ============================================================================
# TESTS DEL DECORADOR @validar_dimensiones
# ============================================================================


def test_decorador_validar_dimensiones_existe():
    """El decorador @validar_dimensiones debe existir."""
    from src.utils.validacion import validar_dimensiones

    assert validar_dimensiones is not None
    assert callable(validar_dimensiones)


def test_decorador_validar_dimensiones_funciona():
    """El decorador debe funcionar en una función simple."""
    from src.utils.validacion import validar_dimensiones, CantidadDimensional, Dimension

    @validar_dimensiones
    def calcular_area_circulo(radio: CantidadDimensional) -> CantidadDimensional:
        """Calcula área = π × r²."""
        return CantidadDimensional(np.pi * radio.valor**2, Dimension.AREA, "area")

    R = CantidadDimensional(0.002, Dimension.LONGITUD, "R")
    area = calcular_area_circulo(R)

    assert area.dimension == Dimension.AREA
    assert_allclose(area.valor, np.pi * 0.002**2, rtol=1e-10)


def test_decorador_rechaza_inputs_invalidos():
    """El decorador debe validar inputs incorrectos."""
    from src.utils.validacion import validar_dimensiones, CantidadDimensional, Dimension

    @validar_dimensiones
    def calcular_velocidad(distancia: CantidadDimensional) -> CantidadDimensional:
        # Función que espera LONGITUD
        if distancia.dimension != Dimension.LONGITUD:
            raise ValueError("Debe ser una longitud")
        return CantidadDimensional(distancia.valor / 10, Dimension.VELOCIDAD, "v")

    # Input correcto: debe funcionar
    d = CantidadDimensional(100, Dimension.LONGITUD, "d")
    v = calcular_velocidad(d)
    assert v.dimension == Dimension.VELOCIDAD

    # Input incorrecto: debe fallar
    t = CantidadDimensional(10, Dimension.TIEMPO, "t")
    with pytest.raises(ValueError, match="Debe ser una longitud"):
        calcular_velocidad(t)


# ============================================================================
# TESTS DE VALIDACIÓN DE ECUACIONES ESPECÍFICAS
# ============================================================================


def test_validar_ecuacion_difusion_2d():
    """La ecuación de difusión-reacción 2D debe ser dimensionalmente correcta."""
    from src.utils.validacion import validar_ecuacion_difusion_2d

    # Ejecutar validación
    resultado = validar_ecuacion_difusion_2d()

    # Debe retornar True o no lanzar excepción
    assert resultado is True


def test_validar_ecuacion_robin_frontera():
    """La condición de Robin debe ser dimensionalmente correcta."""
    from src.utils.validacion import validar_ecuacion_robin

    # -D_eff × ∂C/∂r = k_c × (C_s - C_bulk)
    # [m²/s] × [mol/m³/m] = [m/s] × [mol/m³]
    # [mol/m²/s] = [mol/m²/s] ✅

    resultado = validar_ecuacion_robin()
    assert resultado is True


def test_validar_ecuacion_arrhenius():
    """La ecuación de Arrhenius debe ser dimensionalmente correcta."""
    from src.utils.validacion import validar_ecuacion_arrhenius

    # k = k₀ × exp(-Ea / RT)
    # [1/s] = [1/s] × exp([J/mol] / ([J/(mol·K)] × [K]))
    # [1/s] = [1/s] × exp(adimensional) ✅

    resultado = validar_ecuacion_arrhenius()
    assert resultado is True


def test_validar_todas_las_ecuaciones():
    """Debe existir función que valide TODAS las ecuaciones del proyecto."""
    from src.utils.validacion import validar_todas_ecuaciones

    # Ejecutar validación completa
    resultados = validar_todas_ecuaciones()

    # Debe retornar diccionario con resultados
    assert isinstance(resultados, dict)
    assert "difusion_2d" in resultados
    assert "robin" in resultados
    assert "arrhenius" in resultados

    # Todas deben pasar
    assert all(resultados.values()), f"Algunas validaciones fallaron: {resultados}"


# ============================================================================
# TESTS DE VALIDACIÓN DE ECUACIONES CON VALORES REALES
# ============================================================================


def test_validar_reynolds_dimensional():
    """Número de Reynolds debe ser adimensional."""
    from src.utils.validacion import calcular_reynolds_dimensional, Dimension

    # Re = (ρ × u × D) / μ
    resultado = calcular_reynolds_dimensional(
        rho=0.524,  # kg/m³
        u=0.3,  # m/s
        D=0.004,  # m
        mu=3.32e-5,  # Pa·s = kg/(m·s)
    )

    # Resultado debe ser adimensional
    assert resultado.dimension == Dimension.ADIMENSIONAL
    # Valor debe ser ~19
    assert 18 < resultado.valor < 20


def test_validar_sherwood_dimensional():
    """Número de Sherwood debe ser adimensional."""
    from src.utils.validacion import calcular_sherwood_dimensional, Dimension

    # Sh = (k_c × D) / D_AB
    resultado = calcular_sherwood_dimensional(
        k_c=0.170,  # m/s
        D=0.004,  # m
        D_AB=8.75e-5,  # m²/s
    )

    # Resultado debe ser adimensional
    assert resultado.dimension == Dimension.ADIMENSIONAL
    # Valor debe ser ~7.78
    assert 7.5 < resultado.valor < 8.0


def test_validar_difusividad_efectiva_dimensional():
    """D_eff debe tener dimensión de difusividad."""
    from src.utils.validacion import calcular_D_eff_dimensional, Dimension

    # D_eff = (ε × D_comb) / τ
    resultado = calcular_D_eff_dimensional(
        epsilon=0.45,  # adimensional
        D_comb=6.97e-6,  # m²/s
        tau=3.0,  # adimensional
    )

    # Resultado debe ser difusividad
    assert resultado.dimension == Dimension.DIFUSIVIDAD
    # Valor debe ser ~1.04e-6
    assert_allclose(resultado.valor, 1.04e-6, rtol=0.05)


# ============================================================================
# TESTS DE CONVERSIÓN DE UNIDADES
# ============================================================================


def test_convertir_temperatura_celsius_a_kelvin():
    """Debe poder convertir temperaturas."""
    from src.utils.validacion import convertir_temperatura

    # 400°C = 673.15 K (no 673 exacto)
    T_kelvin = convertir_temperatura(400, origen="celsius", destino="kelvin")

    assert_allclose(T_kelvin, 673.15, rtol=1e-6)


def test_convertir_presion_atm_a_pascal():
    """Debe poder convertir presiones."""
    from src.utils.validacion import convertir_presion

    # 1 atm = 101325 Pa
    P_pascal = convertir_presion(1.0, origen="atm", destino="pascal")

    assert_allclose(P_pascal, 101325, rtol=1e-6)


def test_convertir_concentracion_ppm_a_mol_m3():
    """Debe poder convertir concentraciones de ppm a mol/m³."""
    from src.utils.validacion import convertir_concentracion_ppm

    # 800 ppm CO a 673K, 1 atm
    C_mol_m3 = convertir_concentracion_ppm(ppm=800, T=673, P=101325, R_gas=8.314)

    # Debe dar ~0.0145 mol/m³
    assert_allclose(C_mol_m3, 0.0145, rtol=0.01)


# ============================================================================
# TESTS DE DETECCIÓN DE INCONSISTENCIAS
# ============================================================================


def test_detectar_inconsistencia_dimensional_suma():
    """Sumar cantidades con dimensiones diferentes debe fallar."""
    from src.utils.validacion import CantidadDimensional, Dimension

    longitud = CantidadDimensional(10, Dimension.LONGITUD, "L")
    tiempo = CantidadDimensional(5, Dimension.TIEMPO, "t")

    # Intentar sumar L + T debe fallar
    with pytest.raises((ValueError, TypeError), match="dimensional"):
        resultado = longitud + tiempo


def test_detectar_inconsistencia_dimensional_comparacion():
    """Comparar cantidades con dimensiones diferentes debe fallar o advertir."""
    from src.utils.validacion import CantidadDimensional, Dimension

    masa = CantidadDimensional(1.0, Dimension.MASA, "m")
    longitud = CantidadDimensional(1.0, Dimension.LONGITUD, "L")

    # Comparación directa debe fallar o retornar False
    with pytest.raises((ValueError, TypeError)):
        resultado = masa == longitud


# ============================================================================
# TESTS DE UTILIDADES DE VALIDACIÓN
# ============================================================================


def test_es_adimensional():
    """Debe poder verificar si una cantidad es adimensional."""
    from src.utils.validacion import CantidadDimensional, Dimension, es_adimensional

    # Cantidad adimensional
    Re = CantidadDimensional(19.0, Dimension.ADIMENSIONAL, "Re")
    assert es_adimensional(Re) is True

    # Cantidad dimensional
    D_eff = CantidadDimensional(1.04e-6, Dimension.DIFUSIVIDAD, "D_eff")
    assert es_adimensional(D_eff) is False


def test_verificar_dimension_esperada():
    """Debe poder verificar que una cantidad tenga la dimensión esperada."""
    from src.utils.validacion import (
        CantidadDimensional,
        Dimension,
        verificar_dimension,
    )

    D_eff = CantidadDimensional(1.04e-6, Dimension.DIFUSIVIDAD, "D_eff")

    # Dimensión correcta
    assert verificar_dimension(D_eff, Dimension.DIFUSIVIDAD) is True

    # Dimensión incorrecta
    assert verificar_dimension(D_eff, Dimension.LONGITUD) is False


# ============================================================================
# TESTS DE VALIDACIÓN DE ECUACIÓN DE DIFUSIÓN COMPLETA
# ============================================================================


def test_ecuacion_difusion_lhs_dimension_correcta():
    """∂C/∂t debe tener dimensión [N/(L³·T)]."""
    from src.utils.validacion import obtener_dimension_derivada_temporal_concentracion

    dim = obtener_dimension_derivada_temporal_concentracion()

    # Debe ser concentración por tiempo
    assert "N" in dim.value  # mol
    assert "L" in dim.value  # metros
    assert "T" in dim.value  # segundos


def test_ecuacion_difusion_termino_difusivo_dimension():
    """D_eff × ∇²C debe tener dimensión [N/(L³·T)]."""
    from src.utils.validacion import obtener_dimension_termino_difusivo

    dim = obtener_dimension_termino_difusivo()

    # Debe coincidir con ∂C/∂t
    assert "N" in dim.value
    assert "L" in dim.value
    assert "T" in dim.value


def test_ecuacion_difusion_termino_reactivo_dimension():
    """k_app × C debe tener dimensión [N/(L³·T)]."""
    from src.utils.validacion import obtener_dimension_termino_reactivo

    dim = obtener_dimension_termino_reactivo()

    # Debe coincidir con ∂C/∂t
    assert "N" in dim.value
    assert "L" in dim.value
    assert "T" in dim.value


# ============================================================================
# TESTS DE INTEGRACIÓN CON PARÁMETROS REALES
# ============================================================================


def test_validar_parametros_del_proyecto_dimensionalmente():
    """Todos los parámetros del proyecto deben tener dimensiones consistentes."""
    from src.utils.validacion import validar_parametros_proyecto
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    # Validar todas las dimensiones
    resultado = validar_parametros_proyecto(params)

    assert resultado is True, "Algunos parámetros tienen inconsistencias dimensionales"


def test_calcular_modulo_thiele_dimensionalmente():
    """φ = R√(k_app/D_eff) debe ser adimensional."""
    from src.utils.validacion import calcular_modulo_thiele_dimensional, Dimension

    # Usar valores del proyecto
    phi = calcular_modulo_thiele_dimensional(
        R=0.002,  # m
        k_app=4.0e-3,  # s⁻¹
        D_eff=1.04e-6,  # m²/s
    )

    # Debe ser adimensional
    assert phi.dimension == Dimension.ADIMENSIONAL
    # Valor debe ser ~0.124
    assert_allclose(phi.valor, 0.124, rtol=0.05)


# ============================================================================
# TESTS DE REPORTES Y LOGGING
# ============================================================================


def test_generar_reporte_validacion_dimensional():
    """Debe poder generar reporte de validación dimensional."""
    from src.utils.validacion import generar_reporte_validacion

    reporte = generar_reporte_validacion()

    # Debe retornar un string
    assert isinstance(reporte, str)
    # Debe contener información relevante
    reporte_lower = reporte.lower()
    assert "difusion" in reporte_lower or "ecuacion" in reporte_lower
    assert "✓" in reporte or "pass" in reporte_lower


# ============================================================================
# TESTS DE CASOS LÍMITE Y EDGE CASES
# ============================================================================


def test_cantidad_dimensional_valor_cero():
    """Debe poder manejar valores cero."""
    from src.utils.validacion import CantidadDimensional, Dimension

    C_inicial = CantidadDimensional(0.0, Dimension.CONCENTRACION, "C0")

    assert C_inicial.valor == 0.0
    assert C_inicial.dimension == Dimension.CONCENTRACION


def test_cantidad_dimensional_valor_muy_pequeno():
    """Debe poder manejar valores muy pequeños (notación científica)."""
    from src.utils.validacion import CantidadDimensional, Dimension

    D_eff = CantidadDimensional(1.04e-6, Dimension.DIFUSIVIDAD, "D_eff")

    assert D_eff.valor == 1.04e-6
    assert D_eff.dimension == Dimension.DIFUSIVIDAD


def test_cantidad_dimensional_valor_negativo_advertencia():
    """Valores negativos en cantidades físicas deben generar advertencia."""
    from src.utils.validacion import CantidadDimensional, Dimension

    # Algunas cantidades NO pueden ser negativas (case insensitive)
    with pytest.warns(UserWarning, match="negativ"):
        temperatura_negativa = CantidadDimensional(
            -10, Dimension.TEMPERATURA, "T_invalida", permitir_negativo=False
        )


# ============================================================================
# TESTS DE PERFORMANCE (OPCIONAL)
# ============================================================================


def test_validacion_dimensional_no_afecta_performance():
    """La validación dimensional debe ser rápida."""
    import time
    from src.utils.validacion import CantidadDimensional, Dimension

    # Crear 1000 cantidades
    inicio = time.time()
    for i in range(1000):
        c = CantidadDimensional(i * 1e-6, Dimension.DIFUSIVIDAD, f"D_{i}")
    duracion = time.time() - inicio

    # Debe tomar menos de 1 segundo
    assert duracion < 1.0, f"Creación de cantidades muy lenta: {duracion}s"


# ============================================================================
# FIN DE TESTS
# ============================================================================
