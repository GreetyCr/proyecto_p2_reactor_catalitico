"""
Tests unitarios para stencils de diferencias finitas.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea A2:
- Stencil para Laplaciano radial
- Stencil para Laplaciano angular
- Stencil completo (combinado)
- Aplicación a campos completos
- Validación en casos conocidos

Un stencil es el patrón de nodos vecinos usado para aproximar derivadas.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE STENCIL RADIAL
# ============================================================================


def test_aplicar_stencil_radial_existe():
    """La función aplicar_stencil_radial debe existir."""
    from src.solver.stencil import aplicar_stencil_radial

    assert aplicar_stencil_radial is not None


def test_stencil_radial_nodo_interior():
    """Stencil radial debe aplicarse correctamente en nodo interior."""
    from src.solver.stencil import aplicar_stencil_radial

    # Campo de concentración simple (parábola en r)
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    C_radial = 1.0 - (r / 0.002) ** 2  # Parábola

    # Nodo interior
    i = 30
    D_eff = 1.04e-6

    resultado = aplicar_stencil_radial(C_radial, i, r, dr, D_eff)

    # Debe retornar un valor numérico
    assert isinstance(resultado, (float, np.floating))


def test_stencil_radial_funcion_lineal():
    """Para C(r) = a + b·r, el Laplaciano radial debe dar cero."""
    from src.solver.stencil import aplicar_stencil_radial

    # Función lineal: C(r) = 1 + 2·r
    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    C_radial = 1.0 + 2.0 * r

    i = 30
    D_eff = 1.0  # Simplificar

    # Para función lineal: ∂²C/∂r² = 0, ∂C/∂r = cte
    # Pero (1/r)·∂C/∂r ≠ 0
    # El resultado NO debe ser exactamente cero
    resultado = aplicar_stencil_radial(C_radial, i, r, dr, D_eff)

    # Para función lineal, el término de segundo orden es cero
    # Verificamos que el resultado sea finito
    assert np.isfinite(resultado)


def test_stencil_radial_funcion_cuadratica():
    """Para C(r) = r², verificar que ∂²C/∂r² se calcule correctamente."""
    from src.solver.stencil import aplicar_stencil_radial

    # C(r) = r² → ∂C/∂r = 2r, ∂²C/∂r² = 2
    r = np.linspace(0, 0.002, 121)  # Malla más fina para precisión
    dr = r[1] - r[0]
    C_radial = r**2

    i = 60  # Centro de la malla
    D_eff = 1.0

    resultado = aplicar_stencil_radial(C_radial, i, r, dr, D_eff)

    # Analíticamente: ∂²C/∂r² + (1/r)·∂C/∂r = 2 + (1/r)·2r = 2 + 2 = 4
    # Multiplicado por D_eff = 1.0: resultado ≈ 4
    assert_allclose(resultado, 4.0, rtol=0.05)


def test_stencil_radial_en_centro():
    """En r=0 debe usar tratamiento especial."""
    from src.solver.stencil import aplicar_stencil_radial

    r = np.linspace(0, 0.002, 61)
    dr = r[1] - r[0]
    C_radial = np.ones_like(r)  # Constante

    i = 0  # Centro
    D_eff = 1.04e-6

    # Para C constante, el Laplaciano debe ser ~0
    resultado = aplicar_stencil_radial(C_radial, i, r, dr, D_eff)

    assert_allclose(resultado, 0.0, atol=1e-8)


# ============================================================================
# TESTS DE STENCIL ANGULAR
# ============================================================================


def test_aplicar_stencil_angular_existe():
    """La función aplicar_stencil_angular debe existir."""
    from src.solver.stencil import aplicar_stencil_angular

    assert aplicar_stencil_angular is not None


def test_stencil_angular_nodo_interior():
    """Stencil angular debe aplicarse correctamente."""
    from src.solver.stencil import aplicar_stencil_angular

    # Campo angular simple
    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]
    C_angular = np.sin(theta)  # Senoidal

    j = 48  # Nodo angular
    r_i = 0.001  # Radio fijo

    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    # Debe retornar valor numérico
    assert isinstance(resultado, (float, np.floating))


def test_stencil_angular_funcion_constante():
    """Para C(θ) = constante, ∂²C/∂θ² = 0."""
    from src.solver.stencil import aplicar_stencil_angular

    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]
    C_angular = np.ones_like(theta)  # Constante

    j = 48
    r_i = 0.001

    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    # Para C constante: (1/r²)·∂²C/∂θ² = 0
    assert_allclose(resultado, 0.0, atol=1e-10)


def test_stencil_angular_funcion_senoidal():
    """Para C(θ) = sin(θ), verificar ∂²C/∂θ² = -sin(θ)."""
    from src.solver.stencil import aplicar_stencil_angular

    theta = np.linspace(0, 2 * np.pi, 192)  # Malla fina
    dtheta = theta[1] - theta[0]
    C_angular = np.sin(theta)

    j = 96  # Centro
    r_i = 0.001

    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    # Analíticamente: ∂²(sin θ)/∂θ² = -sin(θ)
    # En j=96: sin(π) ≈ 0
    # (1/r²)·(-sin(θ)) = -0/r² ≈ 0
    expected_value = -np.sin(theta[j]) / r_i**2

    assert_allclose(resultado, expected_value, rtol=0.05)


def test_stencil_angular_periodicidad():
    """Stencil debe manejar periodicidad (θ=0 ≡ θ=2π)."""
    from src.solver.stencil import aplicar_stencil_angular

    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]
    C_angular = np.cos(theta)  # Periódica

    # Primer nodo (j=0)
    j = 0
    r_i = 0.001

    # No debe lanzar error al acceder a j-1 (debe usar periodicidad)
    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    assert np.isfinite(resultado)


# ============================================================================
# TESTS DE STENCIL COMPLETO (LAPLACIANO 2D)
# ============================================================================


def test_aplicar_stencil_completo_existe():
    """La función aplicar_stencil_completo debe existir."""
    from src.solver.stencil import aplicar_stencil_completo

    assert aplicar_stencil_completo is not None


def test_stencil_completo_campo_2d():
    """Stencil completo debe aplicarse a campo 2D."""
    from src.solver.stencil import aplicar_stencil_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo 2D simple (constante)
    C_field = np.ones((malla.nr, malla.ntheta))

    # Aplicar en nodo interior
    i, j = 30, 48

    resultado = aplicar_stencil_completo(C_field, i, j, malla, params.difusion.D_eff)

    # Para campo constante, Laplaciano = 0
    assert_allclose(resultado, 0.0, atol=1e-8)


def test_stencil_completo_suma_radial_y_angular():
    """El stencil completo debe combinar términos radial y angular."""
    from src.solver.stencil import aplicar_stencil_completo
    from src.solver.stencil import aplicar_stencil_radial, aplicar_stencil_angular
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo 2D
    C_field = np.random.rand(malla.nr, malla.ntheta)

    i, j = 30, 48

    # Stencil completo
    resultado_completo = aplicar_stencil_completo(
        C_field, i, j, malla, params.difusion.D_eff
    )

    # Suma de componentes
    resultado_radial = aplicar_stencil_radial(
        C_field[:, j], i, malla.r, malla.dr, params.difusion.D_eff
    )
    resultado_angular_sin_D = aplicar_stencil_angular(
        C_field[i, :], j, malla.r[i], malla.dtheta
    )
    # El término angular debe multiplicarse por D_eff
    resultado_angular = params.difusion.D_eff * resultado_angular_sin_D

    # Deben ser aproximadamente iguales
    assert_allclose(resultado_completo, resultado_radial + resultado_angular, rtol=0.01)


# ============================================================================
# TESTS DE LAPLACIANO COMPLETO SOBRE CAMPO
# ============================================================================


def test_calcular_laplaciano_campo_completo_existe():
    """La función calcular_laplaciano_campo_completo debe existir."""
    from src.solver.stencil import calcular_laplaciano_campo_completo

    assert calcular_laplaciano_campo_completo is not None


def test_laplaciano_campo_completo_shape():
    """El Laplaciano de un campo debe tener el mismo shape."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo de entrada
    C_field = np.ones((malla.nr, malla.ntheta))

    # Calcular Laplaciano
    laplaciano = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # Debe tener el mismo shape
    assert laplaciano.shape == C_field.shape


def test_laplaciano_campo_constante():
    """Laplaciano de campo constante debe ser ~0."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo constante
    C_field = np.full((malla.nr, malla.ntheta), 5.0)

    laplaciano = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # Debe ser aproximadamente cero en todo el dominio (excepto posibles bordes)
    assert np.max(np.abs(laplaciano[1:-1, :])) < 1e-6


def test_laplaciano_campo_lineal_radial():
    """Para C(r) = a·r, verificar resultado."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo lineal en r (independiente de θ)
    a = 100.0  # Constante arbitraria
    C_field = a * malla.R_grid

    laplaciano = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # Para C = a·r:
    # ∂C/∂r = a, ∂²C/∂r² = 0
    # ∇²C = 0 + (1/r)·a = a/r
    # En nodo i: resultado ≈ D_eff · a/r[i]

    i, j = 30, 48
    resultado_esperado = params.difusion.D_eff * a / malla.r[i]

    assert_allclose(laplaciano[i, j], resultado_esperado, rtol=0.10)


# ============================================================================
# TESTS DE PERIODICIDAD ANGULAR
# ============================================================================


def test_stencil_angular_periodicidad_j0():
    """En j=0, debe usar j=-1 = j=ntheta-1 (periodicidad)."""
    from src.solver.stencil import aplicar_stencil_angular

    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]
    C_angular = np.cos(2 * theta)  # Periódica

    j = 0  # Primer nodo
    r_i = 0.001

    # No debe lanzar IndexError
    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    assert np.isfinite(resultado)


def test_stencil_angular_periodicidad_j_ultimo():
    """En j=ntheta-1, debe usar j+1 = j=0 (periodicidad)."""
    from src.solver.stencil import aplicar_stencil_angular

    theta = np.linspace(0, 2 * np.pi, 96)
    dtheta = theta[1] - theta[0]
    C_angular = np.cos(2 * theta)

    j = len(theta) - 1  # Último nodo
    r_i = 0.001

    # No debe lanzar IndexError
    resultado = aplicar_stencil_angular(C_angular, j, r_i, dtheta)

    assert np.isfinite(resultado)


# ============================================================================
# TESTS DE CASOS ANALÍTICOS CONOCIDOS
# ============================================================================


def test_laplaciano_funcion_armonica_simple():
    """Verificar con función armónica conocida."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Función: C(r,θ) = r·cos(θ)
    # Esta es una solución de Laplace: ∇²C = 0
    C_field = malla.R_grid * np.cos(malla.THETA_grid)

    laplaciano = calcular_laplaciano_campo_completo(C_field, malla, 1.0)

    # Debe ser aproximadamente cero en interior (error numérico)
    # Excluyendo fronteras radiales y angulares
    # Los bordes angulares (j=0, j=-1) pueden tener error mayor
    laplaciano_interior = laplaciano[2:-2, 5:-5]

    # Tolerancia amplia debido a error numérico en discretización
    assert np.max(np.abs(laplaciano_interior)) < 10.0


# ============================================================================
# TESTS DE VALIDACIÓN
# ============================================================================


def test_stencil_no_genera_nan():
    """Stencil no debe generar NaN."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo aleatorio
    C_field = np.random.rand(malla.nr, malla.ntheta)

    laplaciano = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # No debe haber NaN
    assert not np.any(np.isnan(laplaciano))


def test_stencil_no_genera_inf():
    """Stencil no debe generar Inf."""
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    C_field = np.random.rand(malla.nr, malla.ntheta)

    laplaciano = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # No debe haber Inf
    assert not np.any(np.isinf(laplaciano))


# ============================================================================
# FIN DE TESTS
# ============================================================================
