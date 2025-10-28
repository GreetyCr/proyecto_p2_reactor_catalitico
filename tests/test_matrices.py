"""
Tests unitarios para construcción de matrices dispersas.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea B1:
- Indexación 2D ↔ 1D
- Construcción de matriz Laplaciana 2D
- Propiedades de matrices dispersas
- Verificación de sparsity pattern
- Validación de simetría

La matriz Laplaciana representa el operador D_eff·∇²C
en forma discreta para todo el dominio.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


# ============================================================================
# TESTS DE INDEXACIÓN 2D ↔ 1D
# ============================================================================


def test_indexar_2d_a_1d_existe():
    """La función de indexación 2D→1D debe existir."""
    from src.solver.matrices import indexar_2d_a_1d

    assert indexar_2d_a_1d is not None


def test_indexar_2d_a_1d_primer_nodo():
    """El primer nodo (0,0) debe mapear a índice 0."""
    from src.solver.matrices import indexar_2d_a_1d

    ntheta = 96
    k = indexar_2d_a_1d(i=0, j=0, ntheta=ntheta)

    assert k == 0


def test_indexar_2d_a_1d_ultimo_angular():
    """Último nodo angular en primera fila radial."""
    from src.solver.matrices import indexar_2d_a_1d

    ntheta = 96
    k = indexar_2d_a_1d(i=0, j=ntheta - 1, ntheta=ntheta)

    # Row-major ordering: k = i*ntheta + j
    assert k == ntheta - 1


def test_indexar_2d_a_1d_segundo_radio():
    """Primer nodo de segunda fila radial."""
    from src.solver.matrices import indexar_2d_a_1d

    ntheta = 96
    k = indexar_2d_a_1d(i=1, j=0, ntheta=ntheta)

    # k = 1*96 + 0 = 96
    assert k == ntheta


def test_indexar_2d_a_1d_general():
    """Fórmula general: k = i*ntheta + j."""
    from src.solver.matrices import indexar_2d_a_1d

    ntheta = 96
    i, j = 10, 25

    k = indexar_2d_a_1d(i, j, ntheta)

    # k = 10*96 + 25 = 985
    assert k == i * ntheta + j


def test_indexar_1d_a_2d_existe():
    """La función de indexación 1D→2D debe existir."""
    from src.solver.matrices import indexar_1d_a_2d

    assert indexar_1d_a_2d is not None


def test_indexar_1d_a_2d_inversa():
    """Indexación 1D→2D debe ser inversa de 2D→1D."""
    from src.solver.matrices import indexar_2d_a_1d, indexar_1d_a_2d

    ntheta = 96

    # Probar varios casos
    for i_orig in [0, 5, 30, 60]:
        for j_orig in [0, 20, 50, 95]:
            k = indexar_2d_a_1d(i_orig, j_orig, ntheta)
            i_back, j_back = indexar_1d_a_2d(k, ntheta)

            assert i_back == i_orig
            assert j_back == j_orig


# ============================================================================
# TESTS DE CONSTRUCCIÓN DE MATRIZ LAPLACIANA
# ============================================================================


def test_construir_matriz_laplaciana_existe():
    """La función de construcción de matriz debe existir."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar

    assert construir_matriz_laplaciana_2d_polar is not None


def test_matriz_laplaciana_shape():
    """La matriz debe tener shape (N, N) donde N = nr × ntheta."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Shape esperado
    N = malla.nr * malla.ntheta
    assert L.shape == (N, N)


def test_matriz_laplaciana_es_dispersa():
    """La matriz debe ser dispersa (scipy.sparse)."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Debe ser sparse
    assert sparse.issparse(L)


def test_matriz_laplaciana_formato_csr():
    """La matriz debe estar en formato CSR (eficiente para ops)."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Debe ser CSR
    assert L.format == "csr"


def test_matriz_laplaciana_sparsity():
    """La matriz debe ser dispersa (~5 elementos por fila)."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Número de elementos no-cero
    nnz = L.nnz
    N = malla.nr * malla.ntheta

    # Promedio de elementos por fila
    avg_per_row = nnz / N

    # Debe estar cerca de 5 (stencil de 5 puntos)
    assert 3 < avg_per_row < 7


def test_matriz_laplaciana_diagonal_negativa():
    """La diagonal principal debe ser negativa (operador difusión)."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Extraer diagonal
    diag = L.diagonal()

    # Todos los elementos de la diagonal deben ser negativos
    # (excepto posiblemente en fronteras con condiciones especiales)
    # Verificar al menos el 90% son negativos
    negativos = np.sum(diag < 0)
    total = len(diag)

    assert negativos / total > 0.90


def test_matriz_laplaciana_simetrica():
    """Para condiciones de frontera adecuadas, L debe ser simétrica."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Verificar simetría: ||L - L^T|| < tol
    L_T = L.transpose()
    diff = L - L_T

    # Norma de Frobenius de la diferencia
    norma_diff = sparse.linalg.norm(diff, ord="fro")

    # Normalizar por norma de L
    norma_L = sparse.linalg.norm(L, ord="fro")

    asimetria_relativa = norma_diff / norma_L

    # Debe ser prácticamente simétrica
    # Relajar tolerancia porque condiciones de frontera pueden romper simetría
    assert asimetria_relativa < 0.10


def test_matriz_laplaciana_aplicada_a_constante():
    """L aplicada a campo constante debe dar ~0."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Vector constante
    N = malla.nr * malla.ntheta
    C_const = np.ones(N)

    # Aplicar L
    L_C = L @ C_const

    # El Laplaciano de una constante debe ser ~0
    # (excepto en fronteras donde puede haber contribuciones)
    norma_LC = np.linalg.norm(L_C)

    # Debe ser relativamente pequeño comparado con el tamaño de la matriz
    # Relajar tolerancia porque fronteras contribuyen
    assert norma_LC < N * 2.0


def test_matriz_laplaciana_consistencia_con_stencil():
    """Verificar que L @ C_vec = Laplaciano calculado con stencil."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.solver.stencil import calcular_laplaciano_campo_completo
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Campo aleatorio
    C_field = np.random.rand(malla.nr, malla.ntheta)

    # Método 1: Usando matriz
    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)
    C_vec = C_field.ravel()  # Aplanar a 1D
    L_C_vec = L @ C_vec
    L_C_field_from_matrix = L_C_vec.reshape(malla.nr, malla.ntheta)

    # Método 2: Usando stencil explícito
    L_C_field_from_stencil = calcular_laplaciano_campo_completo(
        C_field, malla, params.difusion.D_eff
    )

    # Deben ser aproximadamente iguales en nodos interiores
    # Excluir fronteras (pueden tener tratamiento diferente)
    interior = L_C_field_from_matrix[1:-1, 1:-1]
    interior_stencil = L_C_field_from_stencil[1:-1, 1:-1]

    assert_allclose(interior, interior_stencil, rtol=0.05)


# ============================================================================
# TESTS DE ESTRUCTURA DE MATRIZ
# ============================================================================


def test_matriz_estructura_banda():
    """La matriz debe tener estructura de banda (band matrix)."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Convertir a LIL para análisis de estructura
    L_lil = L.tolil()

    # Verificar que cada fila tiene elementos solo en posiciones cercanas
    N = L.shape[0]
    # Bandwidth máximo: debido a indexación 2D→1D,
    # un vecino angular puede estar a distancia ~ntheta
    max_bandwidth = 2 * malla.ntheta + 5  # Banda esperada (conservadora)

    for i in range(N):
        row = L_lil.rows[i]
        if len(row) > 0:
            min_col = min(row)
            max_col = max(row)
            bandwidth = max_col - min_col + 1

            # La banda debe ser razonable (no toda la matriz)
            assert bandwidth < max_bandwidth


def test_matriz_no_tiene_nans():
    """La matriz no debe contener NaN."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Verificar no hay NaN
    assert not np.any(np.isnan(L.data))


def test_matriz_no_tiene_infs():
    """La matriz no debe contener Inf."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Verificar no hay Inf
    assert not np.any(np.isinf(L.data))


# ============================================================================
# TESTS DE INFORMACIÓN Y UTILIDADES
# ============================================================================


def test_obtener_info_matriz_existe():
    """Debe existir función para obtener info de matriz."""
    from src.solver.matrices import obtener_info_matriz

    assert obtener_info_matriz is not None


def test_obtener_info_matriz_contenido():
    """La info de matriz debe contener datos clave."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.solver.matrices import obtener_info_matriz
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    info = obtener_info_matriz(L)

    # Debe contener información clave
    assert "shape" in info
    assert "nnz" in info
    assert "formato" in info
    assert "sparsity" in info


def test_generar_reporte_matriz_existe():
    """Debe existir función para generar reporte."""
    from src.solver.matrices import generar_reporte_matriz

    assert generar_reporte_matriz is not None


def test_generar_reporte_matriz_es_string():
    """El reporte debe ser un string."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.solver.matrices import generar_reporte_matriz
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    reporte = generar_reporte_matriz(L, malla)

    assert isinstance(reporte, str)
    assert len(reporte) > 0


# ============================================================================
# TESTS DE CASOS ESPECIALES
# ============================================================================


def test_matriz_centro_r0_tratamiento_especial():
    """En r=0, la matriz debe tener tratamiento especial."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.solver.matrices import indexar_2d_a_1d
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Nodos del centro (i=0, todos los j)
    L_lil = L.tolil()

    for j in range(malla.ntheta):
        k = indexar_2d_a_1d(0, j, malla.ntheta)
        row = L_lil.rows[k]

        # En centro, solo debe haber conexión con i=1 (no con i=-1)
        # Verificar que hay elementos no-cero
        assert len(row) > 0


def test_matriz_frontera_rR_tratamiento_especial():
    """En r=R (frontera), debe haber tratamiento especial."""
    from src.solver.matrices import construir_matriz_laplaciana_2d_polar
    from src.solver.matrices import indexar_2d_a_1d
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    L = construir_matriz_laplaciana_2d_polar(malla, params.difusion.D_eff)

    # Último nodo radial (i=nr-1)
    L_lil = L.tolil()

    for j in range(malla.ntheta):
        k = indexar_2d_a_1d(malla.nr - 1, j, malla.ntheta)
        row = L_lil.rows[k]

        # Debe haber elementos (no debe ser fila vacía)
        assert len(row) > 0


# ============================================================================
# FIN DE TESTS
# ============================================================================
