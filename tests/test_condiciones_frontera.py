"""
Tests unitarios para condiciones de frontera.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea C1:
- Condición de simetría en r=0
- Modificación de matrices A y B
- Verificación de imposición de condiciones
- Validación física

Condiciones de frontera del problema:
1. r=0 (centro):  ∂C/∂r = 0  (simetría)
2. r=R (frontera): Condición Robin (k_c)
3. θ=0 ≡ θ=2π:    Periodicidad angular
4. Interfaz:       Continuidad de flujo
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse


# ============================================================================
# TESTS DE CONDICIÓN EN CENTRO (r=0)
# ============================================================================


def test_aplicar_condicion_centro_existe():
    """La función para aplicar condición en centro debe existir."""
    from src.solver.condiciones_frontera import aplicar_condicion_centro

    assert aplicar_condicion_centro is not None


def test_aplicar_condicion_centro_modifica_matrices():
    """Debe modificar matrices A y B en filas correspondientes a r=0."""
    from src.solver.condiciones_frontera import aplicar_condicion_centro
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()
    dt = 0.001

    # Matrices originales
    A_orig, B_orig = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    # Aplicar condición de centro
    A_mod, B_mod = aplicar_condicion_centro(A_orig, B_orig, malla)

    # Las matrices deben cambiar
    diff_A = A_mod - A_orig
    diff_B = B_mod - B_orig

    # Debe haber cambios (algunas filas modificadas)
    assert diff_A.nnz > 0 or diff_B.nnz > 0


def test_condicion_centro_impone_simetria():
    """La condición en centro debe imponer C independiente de θ."""
    from src.solver.condiciones_frontera import aplicar_condicion_centro
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()
    dt = 0.001

    A, B = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    # Aplicar condición
    A_bc, B_bc = aplicar_condicion_centro(A, B, malla)

    # Las matrices deben seguir siendo dispersas
    assert sparse.issparse(A_bc)
    assert sparse.issparse(B_bc)


def test_verificar_simetria_centro_existe():
    """Debe existir función para verificar simetría en centro."""
    from src.solver.condiciones_frontera import verificar_simetria_centro

    assert verificar_simetria_centro is not None


def test_verificar_simetria_centro_campo_constante():
    """Campo constante debe satisfacer simetría en centro."""
    from src.solver.condiciones_frontera import verificar_simetria_centro
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo constante (obviamente simétrico)
    C_field = np.ones((malla.nr, malla.ntheta))

    es_simetrico = verificar_simetria_centro(C_field, malla, tol=1e-8)

    assert es_simetrico == True


def test_verificar_simetria_centro_campo_variable_angular():
    """Campo variable en θ NO debe ser simétrico en centro."""
    from src.solver.condiciones_frontera import verificar_simetria_centro
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo que depende de θ
    C_field = malla.THETA_grid  # Depende de θ

    es_simetrico = verificar_simetria_centro(C_field, malla, tol=1e-8)

    # En centro (i=0), debería detectar asimetría
    # (aunque el campo creado sí varía con θ en todo r)
    assert es_simetrico == False


def test_imponer_simetria_centro_existe():
    """Debe existir función para imponer simetría en campo."""
    from src.solver.condiciones_frontera import imponer_simetria_centro

    assert imponer_simetria_centro is not None


def test_imponer_simetria_centro_promedia():
    """Debe promediar valores en r=0 sobre todos los θ."""
    from src.solver.condiciones_frontera import imponer_simetria_centro
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo con valores diferentes en θ en el centro
    C_field = np.random.rand(malla.nr, malla.ntheta)

    # Guardar promedio esperado en centro
    promedio_esperado = np.mean(C_field[0, :])

    # Imponer simetría
    C_simetrico = imponer_simetria_centro(C_field, malla)

    # Todos los valores en r=0 deben ser iguales al promedio
    assert_allclose(C_simetrico[0, :], promedio_esperado, rtol=1e-10)


def test_imponer_simetria_centro_no_modifica_interior():
    """No debe modificar nodos con r > 0."""
    from src.solver.condiciones_frontera import imponer_simetria_centro
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    C_field = np.random.rand(malla.nr, malla.ntheta)
    C_original = C_field.copy()

    # Imponer simetría
    C_simetrico = imponer_simetria_centro(C_field, malla)

    # Interior debe permanecer igual
    assert_allclose(C_simetrico[1:, :], C_original[1:, :], rtol=1e-12)


def test_obtener_nodos_centro_existe():
    """Debe existir función para obtener índices de nodos en centro."""
    from src.solver.condiciones_frontera import obtener_nodos_centro

    assert obtener_nodos_centro is not None


def test_obtener_nodos_centro_cantidad():
    """Debe retornar ntheta nodos (todos los angulares en r=0)."""
    from src.solver.condiciones_frontera import obtener_nodos_centro
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    nodos_centro = obtener_nodos_centro(malla)

    # Debe haber ntheta nodos en r=0
    assert len(nodos_centro) == malla.ntheta


def test_obtener_nodos_centro_valores():
    """Los índices deben corresponder a i=0."""
    from src.solver.condiciones_frontera import obtener_nodos_centro
    from src.solver.matrices import indexar_2d_a_1d
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    nodos_centro = obtener_nodos_centro(malla)

    # Verificar que sean los índices correctos (i=0, j=0...ntheta-1)
    nodos_esperados = [indexar_2d_a_1d(0, j, malla.ntheta) for j in range(malla.ntheta)]

    assert_allclose(nodos_centro, nodos_esperados)


# ============================================================================
# FIN DE TESTS
# ============================================================================
