"""
Tests unitarios para matrices de Crank-Nicolson.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea B2:
- Construcción de matriz de reacción K
- Construcción de matrices A y B de Crank-Nicolson
- Propiedades y relaciones entre A y B
- Aplicación del término de reacción

El esquema Crank-Nicolson es:
    A·C^(n+1) = B·C^n + b

Donde:
    A = I - (dt/2)·(L - K)  (lado implícito)
    B = I + (dt/2)·(L - K)  (lado explícito)
    
    L: operador Laplaciano
    K: matriz diagonal con k_app (reacción)
    I: matriz identidad
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


# ============================================================================
# TESTS DE MATRIZ DE REACCIÓN
# ============================================================================


def test_construir_matriz_reaccion_existe():
    """La función de construcción de matriz K debe existir."""
    from src.solver.matrices import construir_matriz_reaccion

    assert construir_matriz_reaccion is not None


def test_matriz_reaccion_es_diagonal():
    """La matriz K debe ser diagonal."""
    from src.solver.matrices import construir_matriz_reaccion
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo k_app uniforme
    k_app_field = np.full((malla.nr, malla.ntheta), params.cinetica.k_app)

    K = construir_matriz_reaccion(k_app_field)

    # Debe ser diagonal
    K_dense = K.toarray()
    K_off_diag = K_dense - np.diag(np.diagonal(K_dense))

    assert np.max(np.abs(K_off_diag)) < 1e-14


def test_matriz_reaccion_valores_correctos():
    """Los valores de la diagonal de K deben ser k_app."""
    from src.solver.matrices import construir_matriz_reaccion
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo k_app conocido
    k_app_field = params.cinetica.k_app * np.ones((malla.nr, malla.ntheta))

    K = construir_matriz_reaccion(k_app_field)

    # La diagonal debe ser k_app
    diag_K = K.diagonal()
    assert_allclose(diag_K, params.cinetica.k_app, rtol=1e-10)


def test_matriz_reaccion_campo_variable():
    """K debe manejar k_app variable espacialmente."""
    from src.solver.matrices import construir_matriz_reaccion
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo k_app del proyecto (con defecto)
    k_app_field = malla.generar_campo_k_app()

    K = construir_matriz_reaccion(k_app_field)

    # Verificar que diagonal tiene valores de k_app_field
    diag_K = K.diagonal()
    k_app_vec = k_app_field.ravel()

    assert_allclose(diag_K, k_app_vec, rtol=1e-10)


# ============================================================================
# TESTS DE MATRICES A Y B DE CRANK-NICOLSON
# ============================================================================


def test_construir_matrices_cn_existe():
    """La función de construcción de A y B debe existir."""
    from src.solver.matrices import construir_matrices_crank_nicolson

    assert construir_matrices_crank_nicolson is not None


def test_matrices_cn_retorna_tupla():
    """Debe retornar tupla (A, B)."""
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()
    dt = 0.001

    resultado = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    assert isinstance(resultado, tuple)
    assert len(resultado) == 2


def test_matrices_cn_shapes_correctos():
    """A y B deben tener shape (N, N)."""
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

    N = malla.nr * malla.ntheta

    assert A.shape == (N, N)
    assert B.shape == (N, N)


def test_matrices_cn_son_dispersas():
    """A y B deben ser matrices dispersas."""
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

    assert sparse.issparse(A)
    assert sparse.issparse(B)


def test_matrices_cn_formato_csr():
    """A y B deben estar en formato CSR."""
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

    assert A.format == "csr"
    assert B.format == "csr"


def test_matrices_cn_relacion_suma():
    """A + B debe relacionarse con I (identidad)."""
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

    N = malla.nr * malla.ntheta
    I = sparse.eye(N)

    # A + B = 2I + dt·K (aproximadamente)
    # Verificar que A + B ≈ 2I cuando K es pequeño
    suma = A + B

    # La diagonal de A + B debe estar cerca de 2
    diag_suma = suma.diagonal()

    # Debe estar cerca de 2 (con corrección por dt·k_app)
    assert np.mean(diag_suma) > 1.5
    assert np.mean(diag_suma) < 2.5


def test_matriz_A_es_invertible():
    """La matriz A debe ser invertible."""
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.solver.matrices import verificar_matriz_es_invertible
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()
    dt = 0.001

    A, B = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    # A debe ser invertible
    es_invertible = verificar_matriz_es_invertible(A)

    assert es_invertible == True


def test_matrices_cn_sin_reaccion():
    """Con k_app=0, A y B solo deben tener términos de difusión."""
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Sin reacción
    k_app_field = np.zeros((malla.nr, malla.ntheta))
    dt = 0.001

    A, B = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    N = malla.nr * malla.ntheta

    # Sin reacción: A + B = 2I (exactamente)
    suma = A + B
    I = sparse.eye(N)

    diff = suma - 2 * I

    # La diferencia debe ser pequeña
    norma_diff = sparse.linalg.norm(diff, ord="fro")

    assert norma_diff < 1e-10


# ============================================================================
# TESTS DE APLICACIÓN DE TÉRMINO DE REACCIÓN
# ============================================================================


def test_aplicar_termino_reaccion_existe():
    """Debe existir función para aplicar término de reacción."""
    from src.solver.matrices import aplicar_termino_reaccion

    assert aplicar_termino_reaccion is not None


def test_aplicar_termino_reaccion_reduce_concentracion():
    """El término -k_app·C debe reducir la concentración."""
    from src.solver.matrices import aplicar_termino_reaccion

    # Vector de concentración
    C_vec = np.ones(100)

    # k_app uniforme
    k_app_vec = 0.5 * np.ones(100)  # 1/s

    dt = 0.001  # s

    # Aplicar reacción
    delta_C = aplicar_termino_reaccion(C_vec, k_app_vec, dt)

    # delta_C debe ser negativo (consume)
    assert np.all(delta_C <= 0)

    # Magnitud esperada: -k_app·C·dt
    delta_esperado = -k_app_vec * C_vec * dt

    assert_allclose(delta_C, delta_esperado, rtol=1e-10)


def test_aplicar_termino_reaccion_proporcional_a_C():
    """El término de reacción debe ser proporcional a C."""
    from src.solver.matrices import aplicar_termino_reaccion

    k_app_vec = 0.1 * np.ones(100)
    dt = 0.001

    # Dos concentraciones diferentes
    C1 = np.ones(100)
    C2 = 2.0 * np.ones(100)

    delta1 = aplicar_termino_reaccion(C1, k_app_vec, dt)
    delta2 = aplicar_termino_reaccion(C2, k_app_vec, dt)

    # delta2 debe ser el doble de delta1
    assert_allclose(delta2, 2 * delta1, rtol=1e-10)


# ============================================================================
# TESTS DE CONSISTENCIA Y ESTABILIDAD
# ============================================================================


def test_matrices_cn_estabilidad_numerica():
    """Las matrices no deben tener valores anormalmente grandes."""
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

    # Verificar que no hay valores extremos
    assert not np.any(np.isnan(A.data))
    assert not np.any(np.isinf(A.data))
    assert not np.any(np.isnan(B.data))
    assert not np.any(np.isinf(B.data))

    # Los valores deben ser razonables
    assert np.max(np.abs(A.data)) < 1e10
    assert np.max(np.abs(B.data)) < 1e10


def test_matrices_cn_consistencia_con_operador():
    """Verificar consistencia: (A - I) + (B - I) = 0."""
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

    N = malla.nr * malla.ntheta
    I = sparse.eye(N)

    # (A - I) = -(dt/2)·(L - K)
    # (B - I) = +(dt/2)·(L - K)
    # Por lo tanto: (A - I) + (B - I) = 0

    suma_operadores = (A - I) + (B - I)

    # La suma debe ser aproximadamente cero (matriz de ceros)
    norma = sparse.linalg.norm(suma_operadores, ord="fro")

    # Debe ser muy pequeña
    assert norma < 1e-10


def test_matrices_cn_paso_temporal_diferentes():
    """A y B deben cambiar proporcionalmente con dt."""
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()

    # Dos pasos temporales diferentes
    dt1 = 0.001
    dt2 = 0.002

    A1, B1 = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt1
    )
    A2, B2 = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt2
    )

    # Las diferencias deben escalar con dt
    # (A2 - I) debe ser ~2×(A1 - I)
    N = malla.nr * malla.ntheta
    I = sparse.eye(N)

    diff_A1 = A1 - I
    diff_A2 = A2 - I

    # Ratio de normas debe ser ~2
    norma1 = sparse.linalg.norm(diff_A1, ord="fro")
    norma2 = sparse.linalg.norm(diff_A2, ord="fro")

    ratio = norma2 / norma1

    # Debe estar cerca de dt2/dt1 = 2
    assert_allclose(ratio, 2.0, rtol=0.05)


# ============================================================================
# TESTS DE INFORMACIÓN
# ============================================================================


def test_obtener_info_matrices_cn_existe():
    """Debe existir función para obtener info de A y B."""
    from src.solver.matrices import obtener_info_matrices_cn

    assert obtener_info_matrices_cn is not None


def test_obtener_info_matrices_cn_contenido():
    """La info debe contener datos útiles."""
    from src.solver.matrices import construir_matrices_crank_nicolson
    from src.solver.matrices import obtener_info_matrices_cn
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()
    dt = 0.001

    A, B = construir_matrices_crank_nicolson(
        malla, params.difusion.D_eff, k_app_field, dt
    )

    info = obtener_info_matrices_cn(A, B, dt)

    assert "dt" in info
    assert "A" in info
    assert "B" in info


# ============================================================================
# FIN DE TESTS
# ============================================================================
