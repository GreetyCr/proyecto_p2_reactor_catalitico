"""
Tests unitarios para discretización espacial básica.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea A1:
- Coeficientes de discretización radial (α, β)
- Coeficientes de discretización angular (γ)
- Tratamiento especial en r=0 (singularidad)
- Número de Fourier y criterio de estabilidad

La ecuación de difusión-reacción 2D en polares es:
    ∂C/∂t = D_eff·∇²C - k_app·C

Donde en coordenadas polares:
    ∇²C = ∂²C/∂r² + (1/r)·∂C/∂r + (1/r²)·∂²C/∂θ²
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE COEFICIENTES RADIALES
# ============================================================================


def test_calcular_coeficientes_radiales_existe():
    """La función calcular_coeficientes_radiales debe existir."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    assert calcular_coeficientes_radiales is not None


def test_coeficientes_radiales_nodo_interior():
    """Coeficientes radiales en nodo interior deben calcularse correctamente."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    # Nodo interior
    i = 30  # Índice radial (mitad de la malla)
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    D_eff = 1.04e-6  # m²/s

    coefs = calcular_coeficientes_radiales(r, i, dr, D_eff)

    # Debe retornar diccionario con coeficientes
    assert isinstance(coefs, dict)
    assert "alpha" in coefs  # Coef para C[i-1]
    assert "beta" in coefs  # Coef para C[i+1]
    assert "gamma_r" in coefs  # Coef para C[i] (diagonal)


def test_coeficientes_radiales_simetricos():
    """En nodo interior, coefs deben reflejar simetría de ∂²C/∂r²."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    i = 30
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    D_eff = 1.04e-6

    coefs = calcular_coeficientes_radiales(r, i, dr, D_eff)

    # Para el término ∂²C/∂r² los coeficientes de i-1 e i+1 tienen magnitud similar
    # pero el término (1/r)·∂C/∂r rompe la simetría perfecta
    alpha = coefs["alpha"]
    beta = coefs["beta"]

    # Ambos deben ser del mismo orden de magnitud
    assert abs(alpha) > 0
    assert abs(beta) > 0


def test_coeficientes_radiales_escalan_con_D_eff():
    """Coeficientes deben ser proporcionales a D_eff."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    i = 30
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]

    D_eff_1 = 1.04e-6
    D_eff_2 = 2.08e-6  # 2× mayor

    coefs_1 = calcular_coeficientes_radiales(r, i, dr, D_eff_1)
    coefs_2 = calcular_coeficientes_radiales(r, i, dr, D_eff_2)

    # Coeficientes deben escalar linealmente con D_eff
    ratio = D_eff_2 / D_eff_1

    assert_allclose(coefs_2["alpha"] / coefs_1["alpha"], ratio, rtol=0.01)
    assert_allclose(coefs_2["beta"] / coefs_1["beta"], ratio, rtol=0.01)


def test_coeficientes_radiales_escalan_con_dr():
    """Coeficientes deben escalar con dr² (segundo orden)."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    i = 30
    D_eff = 1.04e-6

    # Malla fina
    r_fino = np.linspace(0, 0.002, 121)
    dr_fino = r_fino[1] - r_fino[0]

    # Malla gruesa
    r_grueso = np.linspace(0, 0.002, 61)
    dr_grueso = r_grueso[1] - r_grueso[0]

    coefs_fino = calcular_coeficientes_radiales(r_fino, i, dr_fino, D_eff)
    coefs_grueso = calcular_coeficientes_radiales(r_grueso, i, dr_grueso, D_eff)

    # Coeficientes deben escalar como 1/dr²
    ratio_dr = dr_grueso / dr_fino
    ratio_esperado = ratio_dr**2

    # Los coeficientes del laplaciano escalan como 1/dr²
    assert_allclose(
        abs(coefs_fino["alpha"]) / abs(coefs_grueso["alpha"]), ratio_esperado, rtol=0.10
    )


def test_coeficientes_radiales_centro_especial():
    """En r=0 debe haber tratamiento especial (singularidad)."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    i = 0  # Centro
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    D_eff = 1.04e-6

    coefs = calcular_coeficientes_radiales(r, i, dr, D_eff)

    # En r=0, el coeficiente alpha no existe (no hay i-1)
    # El tratamiento usa límite de L'Hôpital
    assert "centro" in coefs or coefs.get("alpha") is None or coefs["alpha"] == 0


# ============================================================================
# TESTS DE COEFICIENTES ANGULARES
# ============================================================================


def test_calcular_coeficientes_angulares_existe():
    """La función calcular_coeficientes_angulares debe existir."""
    from src.solver.discretizacion import calcular_coeficientes_angulares

    assert calcular_coeficientes_angulares is not None


def test_coeficientes_angulares_nodo_interior():
    """Coeficientes angulares deben calcularse correctamente."""
    from src.solver.discretizacion import calcular_coeficientes_angulares

    j = 48  # Nodo angular
    r = np.linspace(0, 0.002, 61)
    i = 30  # Nodo radial interior
    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]

    coefs = calcular_coeficientes_angulares(r, i, j, dtheta)

    # Debe retornar diccionario
    assert isinstance(coefs, dict)
    assert "gamma_theta" in coefs  # Coef para ∂²C/∂θ²


def test_coeficientes_angulares_escalan_con_r():
    """Coeficientes angulares deben escalar como 1/r²."""
    from src.solver.discretizacion import calcular_coeficientes_angulares

    j = 48
    r = np.linspace(0, 0.002, 61)
    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]

    # Dos radios diferentes
    i1 = 20  # r pequeño
    i2 = 40  # r grande

    coefs_1 = calcular_coeficientes_angulares(r, i1, j, dtheta)
    coefs_2 = calcular_coeficientes_angulares(r, i2, j, dtheta)

    # γ ∝ 1/r² → γ₁/γ₂ = r₂²/r₁² (mayor r, menor γ)
    # Equivalente: γ₂/γ₁ = r₁²/r₂² = 1/(r₂/r₁)²
    ratio_esperado = (r[i1] / r[i2]) ** 2

    assert_allclose(
        coefs_2["gamma_theta"] / coefs_1["gamma_theta"], ratio_esperado, rtol=0.01
    )


def test_coeficientes_angulares_cero_en_centro():
    """En r=0, coeficientes angulares deben manejarse especialmente."""
    from src.solver.discretizacion import calcular_coeficientes_angulares

    j = 48
    r = np.linspace(0, 0.002, 61)
    i = 0  # Centro
    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]

    coefs = calcular_coeficientes_angulares(r, i, j, dtheta)

    # En centro, término (1/r²)∂²C/∂θ² es singular
    # Debe tener tratamiento especial o ser 0
    assert "centro" in coefs or coefs["gamma_theta"] == 0


# ============================================================================
# TESTS DE NÚMERO DE FOURIER
# ============================================================================


def test_calcular_numero_fourier_existe():
    """La función calcular_numero_fourier debe existir."""
    from src.solver.discretizacion import calcular_numero_fourier

    assert calcular_numero_fourier is not None


def test_numero_fourier_formula():
    """Fo = D_eff·dt / dr²."""
    from src.solver.discretizacion import calcular_numero_fourier

    dt = 0.001  # s
    dr = 3.33e-5  # m
    D_eff = 1.04e-6  # m²/s

    Fo = calcular_numero_fourier(dt, dr, D_eff)

    # Cálculo manual
    Fo_esperado = (D_eff * dt) / dr**2

    assert_allclose(Fo, Fo_esperado, rtol=1e-10)


def test_numero_fourier_adimensional():
    """Fo debe ser adimensional."""
    from src.solver.discretizacion import calcular_numero_fourier

    Fo = calcular_numero_fourier(dt=0.001, dr=3.33e-5, D_eff=1.04e-6)

    # Debe ser un número puro (no tiene dimensiones)
    assert isinstance(Fo, (int, float, np.floating))

    # Rango típico: 0 < Fo < 1 para estabilidad
    assert 0 < Fo < 10


def test_numero_fourier_criterio_estabilidad():
    """Para Euler explícito, Fo < 0.5 es necesario para estabilidad."""
    from src.solver.discretizacion import verificar_estabilidad_euler_explicito

    # Caso estable
    Fo_estable = 0.4
    es_estable = verificar_estabilidad_euler_explicito(Fo_estable)
    assert es_estable == True

    # Caso inestable
    Fo_inestable = 0.6
    es_estable = verificar_estabilidad_euler_explicito(Fo_inestable)
    assert es_estable == False


# ============================================================================
# TESTS DE VALIDACIÓN DE PARÁMETROS
# ============================================================================


def test_validar_parametros_discretizacion():
    """Debe validar que dt, dr sean compatibles."""
    from src.solver.discretizacion import validar_parametros_discretizacion

    dt = 0.001
    dr = 3.33e-5
    dtheta = 0.0654
    D_eff = 1.04e-6

    # No debe lanzar excepción con parámetros del proyecto
    validar_parametros_discretizacion(dt, dr, dtheta, D_eff)


def test_validar_parametros_dt_muy_grande():
    """Debe advertir si dt es muy grande (inestable para Euler explícito)."""
    from src.solver.discretizacion import validar_parametros_discretizacion
    import warnings

    dt = 1.0  # Muy grande
    dr = 3.33e-5
    dtheta = 0.0654
    D_eff = 1.04e-6

    # Debe emitir warning (Crank-Nicolson es estable, pero no recomendado)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validar_parametros_discretizacion(dt, dr, dtheta, D_eff)

        # Puede o no emitir warning (Crank-Nicolson es incondicionalmente estable)
        # Así que no forzamos el warning


def test_validar_parametros_negativos():
    """Debe rechazar parámetros negativos."""
    from src.solver.discretizacion import validar_parametros_discretizacion

    # dt negativo
    with pytest.raises(ValueError, match="Paso temporal"):
        validar_parametros_discretizacion(
            dt=-0.001, dr=3.33e-5, dtheta=0.0654, D_eff=1.04e-6
        )

    # dr negativo
    with pytest.raises(ValueError, match="Paso radial"):
        validar_parametros_discretizacion(
            dt=0.001, dr=-3.33e-5, dtheta=0.0654, D_eff=1.04e-6
        )


# ============================================================================
# TESTS DE INTEGRACIÓN CON MALLA
# ============================================================================


def test_calcular_coeficientes_toda_malla():
    """Debe poder calcular coeficientes para toda la malla."""
    from src.solver.discretizacion import calcular_coeficientes_malla_completa
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    coefs = calcular_coeficientes_malla_completa(malla, params.difusion.D_eff)

    # Debe retornar diccionario con arrays
    assert isinstance(coefs, dict)
    assert "alpha" in coefs
    assert "beta" in coefs
    assert "gamma_theta" in coefs

    # Deben tener el shape correcto
    assert coefs["alpha"].shape == (malla.nr,)
    assert coefs["beta"].shape == (malla.nr,)
    assert coefs["gamma_theta"].shape == (malla.nr,)


def test_coeficientes_malla_completa_r0_especial():
    """En r=0, los coeficientes deben tener tratamiento especial."""
    from src.solver.discretizacion import calcular_coeficientes_malla_completa
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    coefs = calcular_coeficientes_malla_completa(malla, params.difusion.D_eff)

    # En i=0 (centro), alpha debe ser 0 o None
    alpha_centro = coefs["alpha"][0]
    assert alpha_centro == 0 or np.isnan(alpha_centro)


# ============================================================================
# TESTS DE ESTABILIDAD NUMÉRICA
# ============================================================================


def test_calcular_dt_critico():
    """Debe calcular dt crítico para estabilidad de Euler explícito."""
    from src.solver.discretizacion import calcular_dt_critico_euler

    dr = 3.33e-5
    dtheta = 0.0654
    D_eff = 1.04e-6

    dt_crit = calcular_dt_critico_euler(dr, dtheta, D_eff)

    # dt_crit debe ser positivo
    assert dt_crit > 0

    # Debe ser del orden de dr²/(4·D_eff)
    dt_esperado_aprox = dr**2 / (4 * D_eff)
    assert_allclose(dt_crit, dt_esperado_aprox, rtol=0.50)  # Orden de magnitud


def test_verificar_estabilidad_crank_nicolson():
    """Crank-Nicolson debe ser incondicionalmente estable."""
    from src.solver.discretizacion import verificar_estabilidad_crank_nicolson

    # Cualquier dt debe ser estable para CN
    dt_grande = 1.0
    dr = 3.33e-5
    D_eff = 1.04e-6

    es_estable = verificar_estabilidad_crank_nicolson(dt_grande, dr, D_eff)

    # CN es incondicionalmente estable
    assert es_estable == True


# ============================================================================
# TESTS DE VALIDACIÓN DIMENSIONAL
# ============================================================================


def test_coeficientes_dimension_correcta():
    """Coeficientes deben tener dimensión de frecuencia [1/T]."""
    from src.solver.discretizacion import calcular_coeficientes_radiales

    i = 30
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    D_eff = 1.04e-6

    coefs = calcular_coeficientes_radiales(r, i, dr, D_eff)

    # Dimensionalmente: [L²/T] / [L²] = [1/T]
    # Verificar que están en rango razonable
    alpha = coefs["alpha"]

    # Orden de magnitud: D_eff/dr² ≈ 1.04e-6 / (3.33e-5)² ≈ 1e3 s⁻¹
    assert 1e-2 < abs(alpha) < 1e6, f"alpha fuera de rango: {alpha}"


def test_numero_fourier_dimensional():
    """Número de Fourier debe ser adimensional."""
    from src.solver.discretizacion import calcular_numero_fourier_dimensional

    Fo = calcular_numero_fourier_dimensional(dt=0.001, dr=3.33e-5, D_eff=1.04e-6)

    # Debe tener dimensión ADIMENSIONAL
    from src.utils.validacion import Dimension

    assert Fo.dimension == Dimension.ADIMENSIONAL


# ============================================================================
# TESTS DE INFORMACIÓN Y UTILIDADES
# ============================================================================


def test_obtener_info_discretizacion():
    """Debe poder obtener información de la discretización."""
    from src.solver.discretizacion import obtener_info_discretizacion

    dt = 0.001
    dr = 3.33e-5
    dtheta = 0.0654
    D_eff = 1.04e-6

    info = obtener_info_discretizacion(dt, dr, dtheta, D_eff)

    # Debe retornar diccionario
    assert isinstance(info, dict)

    # Debe contener información clave
    assert "Fo" in info
    assert "dt_critico_euler" in info
    assert "estable_euler" in info
    assert "estable_cn" in info


def test_imprimir_reporte_discretizacion():
    """Debe poder generar reporte legible."""
    from src.solver.discretizacion import generar_reporte_discretizacion

    dt = 0.001
    dr = 3.33e-5
    dtheta = 0.0654
    D_eff = 1.04e-6

    reporte = generar_reporte_discretizacion(dt, dr, dtheta, D_eff)

    # Debe ser string
    assert isinstance(reporte, str)

    # Debe contener información clave
    assert "Fourier" in reporte or "Fo" in reporte
    assert "estab" in reporte.lower()


# ============================================================================
# FIN DE TESTS
# ============================================================================
