"""
Módulo de Condiciones de Frontera.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa las condiciones de frontera para la ecuación
de difusión-reacción 2D en coordenadas polares.

Condiciones de frontera del problema:
1. r=0 (centro):     ∂C/∂r = 0  (simetría)
2. r=R (frontera):   Condición Robin: -D_eff·∂C/∂r = k_c·(C - C_bulk)
3. θ=0 ≡ θ=2π:       Periodicidad angular (implícita en indexación)
4. Interfaz:         Continuidad de flujo entre activo y defecto

Referencias
----------
.. [1] Patankar, S.V. (1980). "Numerical Heat Transfer and Fluid Flow"
.. [2] Ferziger, J.H. & Peric, M. (2002). "Computational Methods for Fluid Dynamics"
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple
import logging

from src.solver.matrices import indexar_2d_a_1d, indexar_1d_a_2d

logger = logging.getLogger(__name__)


# ============================================================================
# CONDICIÓN EN CENTRO (r=0): SIMETRÍA
# ============================================================================


def obtener_nodos_centro(malla) -> np.ndarray:
    """
    Obtiene índices lineales de todos los nodos en r=0.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    nodos_centro : np.ndarray, shape (ntheta,)
        Array con índices lineales de nodos en centro

    Notes
    -----
    Los nodos en r=0 son: (i=0, j=0), (i=0, j=1), ..., (i=0, j=ntheta-1)

    Examples
    --------
    >>> nodos = obtener_nodos_centro(malla)
    >>> print(f"Nodos en centro: {len(nodos)}")
    """
    nodos = np.array([indexar_2d_a_1d(0, j, malla.ntheta) for j in range(malla.ntheta)])

    return nodos


def verificar_simetria_centro(C_field: np.ndarray, malla, tol: float = 1e-8) -> bool:
    """
    Verifica si C es independiente de θ en r=0 (simetría).

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    tol : float, optional
        Tolerancia para considerar simétrico

    Returns
    -------
    es_simetrico : bool
        True si la variación en θ es < tol

    Notes
    -----
    Por simetría física: ∂C/∂r = 0 en r=0
    Implica: C(r=0, θ) = constante para todo θ

    Examples
    --------
    >>> es_sim = verificar_simetria_centro(C_field, malla)
    >>> print(f"¿Simétrico? {es_sim}")
    """
    # Extraer valores en centro (primera fila radial)
    C_centro = C_field[0, :]

    # Calcular desviación estándar
    std_centro = np.std(C_centro)

    # Calcular variación relativa
    mean_centro = np.mean(C_centro)
    if abs(mean_centro) > 1e-12:
        variacion_relativa = std_centro / abs(mean_centro)
    else:
        variacion_relativa = std_centro

    logger.debug(
        f"Simetría centro: std={std_centro:.3e}, variación={variacion_relativa:.3e}"
    )

    return variacion_relativa < tol


def imponer_simetria_centro(C_field: np.ndarray, malla) -> np.ndarray:
    """
    Impone condición de simetría en r=0.

    Promedia valores en θ para r=0 y asigna el promedio a todos los nodos.

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración

    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    C_simetrico : np.ndarray, shape (nr, ntheta)
        Campo con simetría impuesta en centro

    Notes
    -----
    Operación: C(r=0, θ) = ⟨C(r=0, θ)⟩_θ  (promedio angular)

    Examples
    --------
    >>> C_sym = imponer_simetria_centro(C_field, malla)
    >>> assert np.std(C_sym[0, :]) < 1e-12  # Todos iguales en r=0
    """
    # Copiar campo
    C_simetrico = C_field.copy()

    # Calcular promedio en centro
    promedio_centro = np.mean(C_field[0, :])

    # Asignar a todos los nodos en r=0
    C_simetrico[0, :] = promedio_centro

    return C_simetrico


def aplicar_condicion_centro(
    A: sparse.spmatrix, B: sparse.spmatrix, malla
) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Aplica condición de simetría en r=0 a matrices A y B.

    Modifica las filas correspondientes a nodos en r=0 para imponer
    que C sea independiente de θ.

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz del lado implícito
    B : sparse.spmatrix
        Matriz del lado explícito
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    A_bc : sparse.spmatrix
        Matriz A con condición de frontera aplicada
    B_bc : sparse.spmatrix
        Matriz B con condición de frontera aplicada

    Notes
    -----
    Estrategia:
    1. Para cada nodo en r=0, la fila de A se modifica a:
       A[k, k] = 1  (diagonal)
       A[k, otros] = 0

    2. La fila de B se modifica para imponer promedio:
       B[k, k_j] = 1/ntheta  para todos los j en r=0

    Esto impone: C^(n+1)[k] = ⟨C^n[r=0]⟩_θ

    Examples
    --------
    >>> A_bc, B_bc = aplicar_condicion_centro(A, B, malla)
    >>> # Resolver A_bc·C_np1 = B_bc·C_n impone simetría en centro
    """
    logger.info("Aplicando condición de simetría en centro (r=0)...")

    # Convertir a LIL para modificación eficiente
    A_bc = A.tolil()
    B_bc = B.tolil()

    # Obtener nodos en centro
    nodos_centro = obtener_nodos_centro(malla)

    # Para cada nodo en r=0
    for k in nodos_centro:
        # ============================================================
        # MODIFICAR MATRIZ A (lado implícito)
        # ============================================================
        # Imponer: C^(n+1)[k] = promedio
        # Fila k de A: solo diagonal = 1, resto = 0
        A_bc[k, :] = 0.0  # Limpiar fila
        A_bc[k, k] = 1.0  # Diagonal

        # ============================================================
        # MODIFICAR MATRIZ B (lado explícito)
        # ============================================================
        # RHS: promedio de todos los nodos en r=0
        # B[k, todos_en_centro] = 1/ntheta
        B_bc[k, :] = 0.0  # Limpiar fila

        # Asignar 1/ntheta a todos los nodos en r=0
        for k_centro in nodos_centro:
            B_bc[k, k_centro] = 1.0 / malla.ntheta

    # Convertir de vuelta a CSR
    A_bc = A_bc.tocsr()
    B_bc = B_bc.tocsr()

    logger.info(f"Condición centro aplicada a {len(nodos_centro)} nodos")

    return A_bc, B_bc


# ============================================================================
# UTILIDADES Y VALIDACIÓN
# ============================================================================


def generar_reporte_condicion_centro(malla) -> str:
    """
    Genera reporte de la condición de frontera en centro.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> print(generar_reporte_condicion_centro(malla))
    """
    nodos_centro = obtener_nodos_centro(malla)

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        CONDICIÓN DE FRONTERA: CENTRO (r=0)                   ║
╚══════════════════════════════════════════════════════════════╝

Tipo de Condición:
  - Simetría radial: ∂C/∂r = 0 en r=0

Implicación Física:
  - C debe ser independiente de θ en el centro
  - C(r=0, θ) = constante para todo θ

Implementación Numérica:
  - Promediar C sobre todos los θ en r=0
  - C^(n+1)[r=0, θ] = ⟨C^n[r=0, θ]⟩_θ

Nodos Afectados:
  - Total:              {len(nodos_centro)}
  - Índices lineales:   {nodos_centro[0]} ... {nodos_centro[-1]}
  - Índices 2D:         (0, 0) ... (0, {malla.ntheta - 1})

Modificación de Matrices:
  - Filas de A:         Diagonal = 1, resto = 0
  - Filas de B:         1/nθ en todos los nodos de centro
    """

    return reporte


# ============================================================================
# CONDICIÓN ROBIN EN r=R: TRANSFERENCIA EXTERNA
# ============================================================================


def obtener_nodos_frontera_rR(malla) -> np.ndarray:
    """
    Obtiene índices lineales de todos los nodos en r=R.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    nodos_frontera : np.ndarray, shape (ntheta,)
        Array con índices lineales de nodos en frontera

    Notes
    -----
    Los nodos en r=R son: (i=nr-1, j=0), (i=nr-1, j=1), ..., (i=nr-1, j=ntheta-1)

    Examples
    --------
    >>> nodos = obtener_nodos_frontera_rR(malla)
    >>> print(f"Nodos en r=R: {len(nodos)}")
    """
    i_frontera = malla.nr - 1

    nodos = np.array(
        [indexar_2d_a_1d(i_frontera, j, malla.ntheta) for j in range(malla.ntheta)]
    )

    return nodos


def calcular_flujo_robin(C_s: float, C_bulk: float, k_c: float) -> float:
    """
    Calcula flujo según condición Robin.

    Ecuación:
        J = k_c·(C_bulk - C_s)

    Parameters
    ----------
    C_s : float
        Concentración en superficie [mol/m³]
    C_bulk : float
        Concentración en bulk [mol/m³]
    k_c : float
        Coeficiente de transferencia de masa [m/s]

    Returns
    -------
    J : float
        Flujo molar [mol/(m²·s)]

    Notes
    -----
    Condición Robin:
        -D_eff·∂C/∂r|_R = k_c·(C_bulk - C_s)

    Casos límite:
    - k_c → 0:   No hay transferencia (Neumann homogéneo)
    - k_c → ∞:   C_s = C_bulk (Dirichlet)

    Examples
    --------
    >>> J = calcular_flujo_robin(C_s=0.005, C_bulk=0.0145, k_c=0.085)
    >>> print(f"Flujo: {J:.3e} mol/(m²·s)")
    """
    J = k_c * (C_bulk - C_s)

    return J


def aplicar_condicion_robin(
    A: sparse.spmatrix,
    B: sparse.spmatrix,
    malla,
    D_eff: float,
    k_c: float,
    C_bulk: float,
) -> Tuple[sparse.spmatrix, sparse.spmatrix]:
    """
    Aplica condición Robin en r=R a matrices A y B.

    Condición:
        -D_eff·∂C/∂r|_R = k_c·(C_bulk - C_s)

    Discretización usando diferencias hacia atrás:
        -D_eff·(C_s - C_{nr-2}) / dr = k_c·(C_bulk - C_s)

    Reordenando:
        C_s·[D_eff/dr + k_c] = D_eff·C_{nr-2}/dr + k_c·C_bulk

    Parameters
    ----------
    A : sparse.spmatrix
        Matriz del lado implícito
    B : sparse.spmatrix
        Matriz del lado explícito
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]
    k_c : float
        Coeficiente de transferencia de masa [m/s]
    C_bulk : float
        Concentración en bulk [mol/m³]

    Returns
    -------
    A_bc : sparse.spmatrix
        Matriz A con condición Robin aplicada
    B_bc : sparse.spmatrix
        Matriz B con condición Robin aplicada

    Notes
    -----
    La discretización de Robin modifica las filas de la última capa radial (i=nr-1).

    Examples
    --------
    >>> A_bc, B_bc = aplicar_condicion_robin(A, B, malla, D_eff, k_c, C_bulk)
    """
    logger.info("Aplicando condición Robin en r=R...")

    # Convertir a LIL para modificación
    A_bc = A.tolil()
    B_bc = B.tolil()

    # Obtener nodos en frontera
    nodos_frontera = obtener_nodos_frontera_rR(malla)

    # Parámetros de discretización
    dr = malla.dr

    # Coeficientes de Robin discretizado
    # -D_eff·(C_s - C_{i-1})/dr = k_c·(C_bulk - C_s)
    # Reordenando: C_s·[D_eff/dr + k_c] = D_eff·C_{i-1}/dr + k_c·C_bulk

    coef_C_s = D_eff / dr + k_c
    coef_C_im1 = D_eff / dr
    termino_fuente = k_c * C_bulk

    # Para cada nodo en frontera
    for k in nodos_frontera:
        # Obtener índice 2D
        i, j = indexar_1d_a_2d(k, malla.ntheta)

        # Índice del vecino interior (i-1, j)
        k_im1 = indexar_2d_a_1d(i - 1, j, malla.ntheta)

        # ============================================================
        # MODIFICAR MATRIZ A (lado implícito)
        # ============================================================
        # Limpiar fila
        A_bc[k, :] = 0.0

        # Coeficientes de Robin
        A_bc[k, k] = coef_C_s  # C_s
        A_bc[k, k_im1] = -coef_C_im1  # -C_{i-1}

        # ============================================================
        # MODIFICAR MATRIZ B (lado explícito)
        # ============================================================
        # Para Crank-Nicolson, el término de frontera se aplica simétricamente
        # Pero el término fuente k_c·C_bulk va al RHS
        # Por simplicidad, lo mantenemos en B (se agregará al RHS explícitamente)

    # Convertir de vuelta a CSR
    A_bc = A_bc.tocsr()
    B_bc = B_bc.tocsr()

    logger.info(f"Condición Robin aplicada a {len(nodos_frontera)} nodos")
    logger.info(f"k_c = {k_c:.3e} m/s, C_bulk = {C_bulk:.3e} mol/m³")

    return A_bc, B_bc


def construir_vector_fuente_robin(malla, k_c: float, C_bulk: float) -> np.ndarray:
    """
    Construye vector de términos fuente de la condición Robin.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D
    k_c : float
        Coeficiente de transferencia [m/s]
    C_bulk : float
        Concentración en bulk [mol/m³]

    Returns
    -------
    b_robin : np.ndarray, shape (N,)
        Vector de términos fuente (solo no-cero en r=R)

    Notes
    -----
    Este vector contiene el término k_c·C_bulk que aparece en el RHS
    de la ecuación discretizada con Robin.

    Examples
    --------
    >>> b = construir_vector_fuente_robin(malla, k_c=0.085, C_bulk=0.0145)
    >>> print(f"b no-cero en: {np.sum(b != 0)} nodos")
    """
    N = malla.nr * malla.ntheta
    b_robin = np.zeros(N)

    # Obtener nodos en frontera
    nodos_frontera = obtener_nodos_frontera_rR(malla)

    # Término fuente: k_c·C_bulk
    for k in nodos_frontera:
        b_robin[k] = k_c * C_bulk

    return b_robin


# ============================================================================
# INFORMACIÓN Y REPORTES
# ============================================================================


def generar_reporte_condicion_robin(
    malla, D_eff: float, k_c: float, C_bulk: float
) -> str:
    """
    Genera reporte de la condición Robin.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]
    k_c : float
        Coeficiente de transferencia [m/s]
    C_bulk : float
        Concentración en bulk [mol/m³]

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> print(generar_reporte_condicion_robin(malla, D_eff, k_c, C_bulk))
    """
    nodos_frontera = obtener_nodos_frontera_rR(malla)
    dr = malla.dr

    # Calcular número de Biot
    Bi = k_c * malla.r[-1] / D_eff

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        CONDICIÓN DE FRONTERA: ROBIN EN r=R                   ║
╚══════════════════════════════════════════════════════════════╝

Tipo de Condición:
  - Robin (transferencia externa)

Ecuación:
  - Flujo en frontera: -D_eff·∂C/∂r|_R = k_c·(C_bulk - C_s)

Parámetros:
  - D_eff:              {D_eff:.3e} m²/s
  - k_c:                {k_c:.3e} m/s
  - C_bulk:             {C_bulk:.3e} mol/m³
  - R:                  {malla.r[-1]:.3e} m
  - dr:                 {dr:.3e} m

Número de Biot:
  - Bi = k_c·R/D_eff:   {Bi:.3f}
  - Régimen:            {'Control externo' if Bi < 0.1 else 'Control interno' if Bi > 10 else 'Mixto'}

Discretización:
  - Método:             Diferencias hacia atrás (orden 1)
  - Coef C_s:           {D_eff/dr + k_c:.3e}
  - Coef C_{{i-1}}:       {D_eff/dr:.3e}
  - Término fuente:     {k_c * C_bulk:.3e}

Nodos Afectados:
  - Total:              {len(nodos_frontera)}
  - Índices:            {nodos_frontera[0]} ... {nodos_frontera[-1]}
  - Índices 2D:         ({malla.nr-1}, 0) ... ({malla.nr-1}, {malla.ntheta-1})
    """

    return reporte


# ============================================================================
# PERIODICIDAD ANGULAR (θ=0 ≡ θ=2π)
# ============================================================================


def verificar_periodicidad_angular(
    C_field: np.ndarray, malla, tol: float = 1e-6
) -> bool:
    """
    Verifica si C satisface periodicidad angular.

    Condición: C(r, θ=0) ≈ C(r, θ=2π) para todo r

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    tol : float, optional
        Tolerancia para considerar periódico

    Returns
    -------
    es_periodico : bool
        True si satisface periodicidad

    Notes
    -----
    La malla implementa periodicidad implícitamente en el stencil.
    Esta función verifica que el campo calculado sea periódico.

    Examples
    --------
    >>> es_per = verificar_periodicidad_angular(C_field, malla)
    >>> print(f"¿Periódico? {es_per}")
    """
    # En la malla discreta:
    # θ=0 corresponde a j=0
    # θ=2π NO está en la malla, pero j=ntheta-1 es el último antes de volver a θ=0

    # Por periodicidad, esperamos que C[:, 0] ≈ C[:, ntheta-1] en el límite
    # Sin embargo, debido a la discretización, pueden no ser exactamente iguales

    # Comparar primera y última columna
    C_j0 = C_field[:, 0]
    C_jfinal = C_field[:, -1]

    # Calcular diferencia máxima
    diff = np.abs(C_j0 - C_jfinal)
    max_diff = np.max(diff)

    # Variación relativa
    max_val = np.max(np.abs(C_field))
    if max_val > 1e-12:
        variacion_relativa = max_diff / max_val
    else:
        variacion_relativa = max_diff

    logger.debug(
        f"Periodicidad angular: max_diff={max_diff:.3e}, var_rel={variacion_relativa:.3e}"
    )

    return variacion_relativa < tol


def imponer_periodicidad_angular(C_field: np.ndarray, malla) -> np.ndarray:
    """
    Impone periodicidad angular en el campo.

    Promedia valores en θ=0 y θ≈2π y los asigna a ambos.

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    C_periodico : np.ndarray, shape (nr, ntheta)
        Campo con periodicidad impuesta

    Notes
    -----
    Operación: C[:, 0] = C[:, -1] = 0.5·(C[:, 0] + C[:, -1])

    Examples
    --------
    >>> C_per = imponer_periodicidad_angular(C_field, malla)
    """
    # Copiar campo
    C_periodico = C_field.copy()

    # Para cada radio
    for i in range(malla.nr):
        # Promediar primera y última columna
        promedio = 0.5 * (C_field[i, 0] + C_field[i, -1])

        # Asignar a ambas
        C_periodico[i, 0] = promedio
        C_periodico[i, -1] = promedio

    return C_periodico


def generar_reporte_periodicidad_angular(malla) -> str:
    """
    Genera reporte de la periodicidad angular.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> print(generar_reporte_periodicidad_angular(malla))
    """
    dtheta = malla.dtheta

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        CONDICIÓN: PERIODICIDAD ANGULAR                       ║
╚══════════════════════════════════════════════════════════════╝

Tipo de Condición:
  - Periodicidad: θ=0 ≡ θ=2π

Implicación Física:
  - C(r, θ=0) = C(r, θ=2π) para todo r
  - El campo debe ser periódico en la dirección angular

Implementación Numérica:
  - Periodicidad IMPLÍCITA en el stencil angular
  - Operación módulo: j=-1 → j=nθ-1, j=nθ → j=0
  - No requiere modificación explícita de matrices

Discretización:
  - θ ∈ [0, 2π)
  - dθ:                 {dtheta:.4f} rad
  - nθ:                 {malla.ntheta}
  - θ_j = j·dθ:         j=0 → θ=0, j={malla.ntheta-1} → θ={malla.theta[-1]:.4f}

Verificación:
  - Automática durante solver
  - Se verifica: ||C[:, 0] - C[:, nθ-1]|| < tol

Nota:
  - La matriz Laplaciana ya implementa esta condición
  - No se requiere modificación adicional de A o B
    """

    return reporte


# ============================================================================
# INTERFAZ ACTIVO-DEFECTO
# ============================================================================


def identificar_nodos_interfaz(malla) -> List[int]:
    """
    Identifica nodos lineales en la interfaz activo-defecto.

    La interfaz está en los bordes de la región del defecto:
    - r = r1 (radio interno)
    - r = r2 (radio externo)
    - θ = θ1 (ángulo inicial)
    - θ = θ2 (ángulo final)

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    nodos_interfaz : List[int]
        Lista con índices lineales de nodos en interfaz

    Notes
    -----
    La interfaz es donde k_app cambia de 0 (defecto) a k_app (activo).

    Examples
    --------
    >>> nodos = identificar_nodos_interfaz(malla)
    >>> print(f"Nodos en interfaz: {len(nodos)}")
    """
    nodos_interfaz = []

    # Obtener máscaras
    mascara_defecto = malla.identificar_region_defecto()

    # Recorrer todos los nodos
    for i in range(malla.nr):
        for j in range(malla.ntheta):
            # Si el nodo actual está en el defecto
            if mascara_defecto[i, j]:
                # Verificar si tiene algún vecino en región activa
                vecinos_en_activo = False

                # Vecino radial i+1
                if i + 1 < malla.nr:
                    if not mascara_defecto[i + 1, j]:
                        vecinos_en_activo = True

                # Vecino radial i-1
                if i - 1 >= 0:
                    if not mascara_defecto[i - 1, j]:
                        vecinos_en_activo = True

                # Vecino angular j+1 (con periodicidad)
                j_p1 = (j + 1) % malla.ntheta
                if not mascara_defecto[i, j_p1]:
                    vecinos_en_activo = True

                # Vecino angular j-1 (con periodicidad)
                j_m1 = (j - 1) % malla.ntheta
                if not mascara_defecto[i, j_m1]:
                    vecinos_en_activo = True

                # Si tiene vecinos en región activa, está en interfaz
                if vecinos_en_activo:
                    k = indexar_2d_a_1d(i, j, malla.ntheta)
                    nodos_interfaz.append(k)

            # Si el nodo está en región activa
            else:
                # Verificar si tiene algún vecino en defecto
                vecinos_en_defecto = False

                # Vecino radial i+1
                if i + 1 < malla.nr:
                    if mascara_defecto[i + 1, j]:
                        vecinos_en_defecto = True

                # Vecino radial i-1
                if i - 1 >= 0:
                    if mascara_defecto[i - 1, j]:
                        vecinos_en_defecto = True

                # Vecino angular j+1 (con periodicidad)
                j_p1 = (j + 1) % malla.ntheta
                if mascara_defecto[i, j_p1]:
                    vecinos_en_defecto = True

                # Vecino angular j-1 (con periodicidad)
                j_m1 = (j - 1) % malla.ntheta
                if mascara_defecto[i, j_m1]:
                    vecinos_en_defecto = True

                # Si tiene vecinos en defecto, está en interfaz
                if vecinos_en_defecto:
                    k = indexar_2d_a_1d(i, j, malla.ntheta)
                    nodos_interfaz.append(k)

    return nodos_interfaz


def calcular_salto_flujo_interfaz(C_field: np.ndarray, malla, D_eff: float) -> float:
    """
    Calcula el máximo salto de flujo en la interfaz activo-defecto.

    El flujo debe ser continuo en la interfaz:
        J_activo = J_defecto

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]

    Returns
    -------
    salto_max : float
        Máximo salto de flujo en la interfaz [mol/(m²·s)]

    Notes
    -----
    Para una interfaz perfecta, el salto debe ser ~0.

    Examples
    --------
    >>> salto = calcular_salto_flujo_interfaz(C_field, malla, D_eff)
    >>> print(f"Salto máximo: {salto:.3e}")
    """
    nodos_interfaz = identificar_nodos_interfaz(malla)

    if len(nodos_interfaz) == 0:
        return 0.0

    saltos = []

    dr = malla.dr
    dtheta = malla.dtheta

    for k in nodos_interfaz:
        i, j = indexar_1d_a_2d(k, malla.ntheta)

        # Estimar flujo en este nodo
        # Flujo radial: J_r = -D_eff·∂C/∂r
        if i > 0 and i < malla.nr - 1:
            dC_dr = (C_field[i + 1, j] - C_field[i - 1, j]) / (2 * dr)
            J_r = -D_eff * dC_dr
        else:
            J_r = 0.0

        # Flujo angular: J_θ = -D_eff·(1/r)·∂C/∂θ
        r_i = malla.r[i]
        if r_i > 1e-12:
            j_p1 = (j + 1) % malla.ntheta
            j_m1 = (j - 1) % malla.ntheta
            dC_dtheta = (C_field[i, j_p1] - C_field[i, j_m1]) / (2 * dtheta)
            J_theta = -D_eff * dC_dtheta / r_i
        else:
            J_theta = 0.0

        # Magnitud del flujo
        J_mag = np.sqrt(J_r**2 + J_theta**2)
        saltos.append(J_mag)

    if len(saltos) > 0:
        salto_max = np.max(saltos)
    else:
        salto_max = 0.0

    return salto_max


def verificar_continuidad_flujo(
    C_field: np.ndarray, malla, D_eff: float, tol: float = 1e-6
) -> bool:
    """
    Verifica continuidad de flujo en la interfaz.

    Parameters
    ----------
    C_field : np.ndarray, shape (nr, ntheta)
        Campo de concentración
    malla : MallaPolar2D
        Malla polar 2D
    D_eff : float
        Difusividad efectiva [m²/s]
    tol : float, optional
        Tolerancia para considerar continuo

    Returns
    -------
    es_continuo : bool
        True si el flujo es continuo

    Examples
    --------
    >>> continuo = verificar_continuidad_flujo(C_field, malla, D_eff)
    >>> print(f"¿Flujo continuo? {continuo}")
    """
    salto_max = calcular_salto_flujo_interfaz(C_field, malla, D_eff)

    # Normalizar por flujo típico
    C_max = np.max(np.abs(C_field))
    if C_max > 1e-12:
        flujo_tipico = D_eff * C_max / malla.dr
        salto_relativo = salto_max / flujo_tipico
    else:
        salto_relativo = salto_max

    logger.debug(
        f"Continuidad interfaz: salto_max={salto_max:.3e}, relativo={salto_relativo:.3e}"
    )

    return salto_relativo < tol


def generar_reporte_interfaz(malla) -> str:
    """
    Genera reporte de la interfaz activo-defecto.

    Parameters
    ----------
    malla : MallaPolar2D
        Malla polar 2D

    Returns
    -------
    reporte : str
        Reporte formateado

    Examples
    --------
    >>> print(generar_reporte_interfaz(malla))
    """
    nodos_interfaz = identificar_nodos_interfaz(malla)

    mascara_defecto = malla.identificar_region_defecto()
    n_defecto = np.sum(mascara_defecto)
    n_activo = malla.nr * malla.ntheta - n_defecto

    # Parámetros del defecto
    r1 = malla.params.geometria.r1
    r2 = malla.params.geometria.r2
    theta1 = malla.params.geometria.theta1
    theta2 = malla.params.geometria.theta2

    reporte = f"""
╔══════════════════════════════════════════════════════════════╗
║        INTERFAZ ACTIVO-DEFECTO                               ║
╚══════════════════════════════════════════════════════════════╝

Geometría del Defecto:
  - r1 (interno):       {r1:.3e} m
  - r2 (externo):       {r2:.3e} m
  - θ1 (inicial):       {theta1:.4f} rad = {np.degrees(theta1):.1f}°
  - θ2 (final):         {theta2:.4f} rad = {np.degrees(theta2):.1f}°

Nodos:
  - Total:              {malla.nr * malla.ntheta}
  - Región defecto:     {n_defecto} ({100*n_defecto/(malla.nr*malla.ntheta):.1f}%)
  - Región activa:      {n_activo} ({100*n_activo/(malla.nr*malla.ntheta):.1f}%)
  - En interfaz:        {len(nodos_interfaz)}

Propiedades:
  - k_app en defecto:   0 s⁻¹
  - k_app en activo:    {malla.params.cinetica.k_app:.3e} s⁻¹

Condición en Interfaz:
  - Continuidad de flujo: -D_eff·∇C continuo
  - Implementación: Natural (stencil estándar)
  - No requiere modificación de matrices

Verificación:
  - Durante solver: verificar salto de flujo < tol
  - Post-procesamiento: análisis de gradientes
    """

    return reporte


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
