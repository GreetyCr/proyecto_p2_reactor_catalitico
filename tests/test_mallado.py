"""
Tests unitarios para el módulo de geometría y mallado polar 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD (Test-Driven Development):
- Estos tests se escriben ANTES de implementar src/geometria/mallado.py
- Inicialmente deben FALLAR (RED)
- Luego implementamos código para hacerlos pasar (GREEN)

El módulo de mallado es crítico para la discretización 2D en polares.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


# ============================================================================
# TESTS DE LA CLASE MallaPolar2D
# ============================================================================


def test_malla_polar_2d_clase_existe():
    """La clase MallaPolar2D debe existir."""
    from src.geometria.mallado import MallaPolar2D

    assert MallaPolar2D is not None


def test_malla_polar_2d_crear_instancia_con_parametros():
    """Debe poder crear una malla con parámetros del proyecto."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    assert malla is not None


def test_malla_polar_2d_tiene_atributos_basicos():
    """MallaPolar2D debe tener atributos básicos."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Atributos de la malla
    assert hasattr(malla, "r")  # Array radial
    assert hasattr(malla, "theta")  # Array angular
    assert hasattr(malla, "nr")  # Número de nodos radiales
    assert hasattr(malla, "ntheta")  # Número de nodos angulares
    assert hasattr(malla, "dr")  # Paso radial
    assert hasattr(malla, "dtheta")  # Paso angular


def test_malla_radial_correcta():
    """El array radial debe ir de 0 a R con nr nodos."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Verificar rango
    assert malla.r[0] == 0.0, "r debe empezar en 0"
    assert_allclose(malla.r[-1], params.geometria.R, rtol=1e-10)

    # Verificar número de nodos
    assert len(malla.r) == params.mallado.nr


def test_malla_angular_correcta():
    """El array angular debe ir de 0 a 2π con ntheta nodos."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Verificar rango
    assert malla.theta[0] == 0.0, "θ debe empezar en 0"
    assert_allclose(malla.theta[-1], 2 * np.pi, rtol=1e-6)

    # Verificar número de nodos
    assert len(malla.theta) == params.mallado.ntheta


def test_malla_pasos_consistentes():
    """Los pasos dr y dtheta deben ser consistentes."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # dr debe ser R/(nr-1)
    dr_esperado = params.geometria.R / (params.mallado.nr - 1)
    assert_allclose(malla.dr, dr_esperado, rtol=1e-10)

    # dtheta debe ser 2π/(ntheta-1)
    dtheta_esperado = 2 * np.pi / (params.mallado.ntheta - 1)
    assert_allclose(malla.dtheta, dtheta_esperado, rtol=1e-10)


def test_malla_tiene_mesh_grids():
    """Debe tener meshgrids R y THETA."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    assert hasattr(malla, "R_grid")
    assert hasattr(malla, "THETA_grid")

    # Verificar shapes
    assert malla.R_grid.shape == (params.mallado.nr, params.mallado.ntheta)
    assert malla.THETA_grid.shape == (params.mallado.nr, params.mallado.ntheta)


# ============================================================================
# TESTS DE CONVERSIÓN A COORDENADAS CARTESIANAS
# ============================================================================


def test_convertir_a_cartesianas():
    """Debe poder convertir malla polar a cartesiana."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    X, Y = malla.obtener_coordenadas_cartesianas()

    # Verificar shapes
    assert X.shape == (params.mallado.nr, params.mallado.ntheta)
    assert Y.shape == (params.mallado.nr, params.mallado.ntheta)

    # Verificar punto central (r=0)
    assert_allclose(X[0, :], 0, atol=1e-10)
    assert_allclose(Y[0, :], 0, atol=1e-10)

    # Verificar punto en r=R, θ=0 → (R, 0)
    idx_theta_0 = 0
    assert_allclose(X[-1, idx_theta_0], params.geometria.R, rtol=1e-10)
    assert_allclose(Y[-1, idx_theta_0], 0, atol=1e-10)


def test_conversion_cartesiana_consistente():
    """La conversión debe satisfacer X² + Y² = r²."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    X, Y = malla.obtener_coordenadas_cartesianas()

    # Para cada punto: X² + Y² debe ser igual a r²
    r_calculado = np.sqrt(X**2 + Y**2)
    r_esperado = malla.R_grid

    assert_allclose(r_calculado, r_esperado, rtol=1e-10)


# ============================================================================
# TESTS DE IDENTIFICACIÓN DE REGIÓN DE DEFECTO
# ============================================================================


def test_identificar_region_defecto():
    """Debe identificar correctamente la región del defecto."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    mascara_defecto = malla.identificar_region_defecto()

    # Debe retornar un array booleano
    assert mascara_defecto.dtype == bool
    assert mascara_defecto.shape == (params.mallado.nr, params.mallado.ntheta)


def test_region_defecto_rangos_correctos():
    """La región del defecto debe estar en r∈[r1,r2] y θ∈[0,45°]."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    mascara_defecto = malla.identificar_region_defecto()

    # Verificar algunos puntos específicos
    # Centro (r=0) NO debe estar en defecto
    assert mascara_defecto[0, :].sum() == 0, "Centro no debe tener defecto"

    # Verificar que hay nodos en el defecto
    assert mascara_defecto.sum() > 0, "Debe haber al menos un nodo en defecto"

    # Verificar que no todos los nodos son defecto
    assert mascara_defecto.sum() < mascara_defecto.size, "No todos pueden ser defecto"


def test_region_defecto_consistente():
    """Nodos fuera del rango del defecto NO deben marcarse."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    mascara_defecto = malla.identificar_region_defecto()

    # Verificar punto en r < r1: no debe ser defecto
    idx_r_menor_r1 = np.where(malla.r < params.geometria.r1)[0]
    if len(idx_r_menor_r1) > 0:
        assert mascara_defecto[idx_r_menor_r1, :].sum() == 0

    # Verificar punto en r > r2: no debe ser defecto
    idx_r_mayor_r2 = np.where(malla.r > params.geometria.r2)[0]
    if len(idx_r_mayor_r2) > 0:
        # Pero solo en θ > θ2
        idx_theta_mayor = np.where(malla.theta > params.geometria.theta2)[0]
        if len(idx_theta_mayor) > 0:
            assert mascara_defecto[idx_r_mayor_r2[0], idx_theta_mayor[0]] == False


# ============================================================================
# TESTS DE IDENTIFICACIÓN DE REGIÓN ACTIVA
# ============================================================================


def test_identificar_region_activa():
    """Debe identificar correctamente la región activa (k_app > 0)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    mascara_activa = malla.identificar_region_activa()

    # Debe retornar un array booleano
    assert mascara_activa.dtype == bool
    assert mascara_activa.shape == (params.mallado.nr, params.mallado.ntheta)


def test_region_activa_complemento_defecto():
    """La región activa debe ser el complemento de la región de defecto."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    mascara_defecto = malla.identificar_region_defecto()
    mascara_activa = malla.identificar_region_activa()

    # Complemento: activa = ~defecto
    assert_array_equal(mascara_activa, ~mascara_defecto)


# ============================================================================
# TESTS DE CAMPO DE k_app (REACCIÓN)
# ============================================================================


def test_generar_campo_k_app():
    """Debe generar campo de k_app correcto (0 en defecto, valor en activa)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    k_app_field = malla.generar_campo_k_app()

    # Shape correcto
    assert k_app_field.shape == (params.mallado.nr, params.mallado.ntheta)

    # En región de defecto: k_app = 0
    mascara_defecto = malla.identificar_region_defecto()
    assert_allclose(k_app_field[mascara_defecto], 0.0, atol=1e-15)

    # En región activa: k_app = valor del parámetro
    mascara_activa = malla.identificar_region_activa()
    assert_allclose(k_app_field[mascara_activa], params.cinetica.k_app, rtol=1e-10)


# ============================================================================
# TESTS DE ÍNDICES Y ACCESO
# ============================================================================


def test_obtener_indice_desde_coordenadas():
    """Debe poder obtener índice (i,j) desde coordenadas (r,θ)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Buscar punto cercano a (r=R/2, θ=π/4)
    r_objetivo = params.geometria.R / 2
    theta_objetivo = np.pi / 4

    i, j = malla.encontrar_indice_mas_cercano(r_objetivo, theta_objetivo)

    # Verificar que son índices válidos
    assert 0 <= i < params.mallado.nr
    assert 0 <= j < params.mallado.ntheta

    # Verificar que están razonablemente cerca
    r_encontrado = malla.r[i]
    theta_encontrado = malla.theta[j]

    assert abs(r_encontrado - r_objetivo) < 2 * malla.dr
    assert abs(theta_encontrado - theta_objetivo) < 2 * malla.dtheta


def test_obtener_nodo_central():
    """Debe poder obtener el índice del nodo central (r=0)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    i_centro = malla.obtener_indice_centro()

    # Centro debe ser i=0
    assert i_centro == 0
    assert malla.r[i_centro] == 0.0


def test_obtener_nodos_frontera():
    """Debe poder obtener los índices de los nodos en la frontera r=R."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    i_frontera = malla.obtener_indice_frontera()

    # Frontera debe ser i=nr-1
    assert i_frontera == params.mallado.nr - 1
    assert_allclose(malla.r[i_frontera], params.geometria.R, rtol=1e-10)


# ============================================================================
# TESTS DE VISUALIZACIÓN DE LA MALLA
# ============================================================================


def test_visualizar_malla_sin_error():
    """Debe poder visualizar la malla sin errores."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros
    import matplotlib

    matplotlib.use("Agg")  # Backend no interactivo para tests

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Intentar visualizar (no debe lanzar excepción)
    fig, ax = malla.visualizar_malla(mostrar=False)

    assert fig is not None
    assert ax is not None


def test_visualizar_regiones_sin_error():
    """Debe poder visualizar regiones activa/defecto sin errores."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros
    import matplotlib

    matplotlib.use("Agg")

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Intentar visualizar regiones
    fig, ax = malla.visualizar_regiones(mostrar=False)

    assert fig is not None
    assert ax is not None


# ============================================================================
# TESTS DE PROPIEDADES GEOMÉTRICAS
# ============================================================================


def test_calcular_area_total():
    """Debe calcular el área total del pellet (πR²)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    area_total = malla.calcular_area_total()

    # Área esperada: πR²
    area_esperada = np.pi * params.geometria.R**2

    assert_allclose(area_total, area_esperada, rtol=0.01)  # 1% tolerancia


def test_calcular_area_defecto():
    """Debe calcular el área de la región de defecto."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    area_defecto = malla.calcular_area_defecto()

    # Área de defecto (sector anular):
    # A = (θ2 - θ1) × (r2² - r1²) / 2
    r1 = params.geometria.r1
    r2 = params.geometria.r2
    theta1 = params.geometria.theta1
    theta2 = params.geometria.theta2

    area_esperada = (theta2 - theta1) * (r2**2 - r1**2) / 2

    assert_allclose(area_defecto, area_esperada, rtol=0.05)  # 5% tolerancia


def test_fraccion_defecto():
    """Debe calcular la fracción de área defectuosa."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    fraccion = malla.calcular_fraccion_defecto()

    # Debe estar entre 0 y 1
    assert 0 < fraccion < 1

    # Verificar consistencia
    area_defecto = malla.calcular_area_defecto()
    area_total = malla.calcular_area_total()

    assert_allclose(fraccion, area_defecto / area_total, rtol=1e-10)


# ============================================================================
# TESTS DE VOLÚMENES DE CONTROL
# ============================================================================


def test_calcular_volumenes_control():
    """Debe calcular volúmenes de control para cada nodo."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    volumenes = malla.calcular_volumenes_control()

    # Shape correcto
    assert volumenes.shape == (params.mallado.nr, params.mallado.ntheta)

    # Todos positivos (excepto posiblemente r=0)
    assert np.all(volumenes >= 0)

    # Suma de volúmenes debe aproximar volumen total
    volumen_total_discreto = np.sum(volumenes)
    volumen_total_analitico = np.pi * params.geometria.R**2  # Área (2D)

    assert_allclose(volumen_total_discreto, volumen_total_analitico, rtol=0.05)


# ============================================================================
# TESTS DE VALIDACIÓN DE LA MALLA
# ============================================================================


def test_malla_sin_nodos_negativos():
    """La malla no debe tener coordenadas negativas."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # r debe ser no negativo
    assert np.all(malla.r >= 0)

    # θ debe ser no negativo
    assert np.all(malla.theta >= 0)


def test_malla_espaciado_uniforme():
    """El espaciado de la malla debe ser uniforme."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Espaciado radial uniforme
    dr_array = np.diff(malla.r)
    assert_allclose(dr_array, malla.dr, rtol=1e-10)

    # Espaciado angular uniforme
    dtheta_array = np.diff(malla.theta)
    assert_allclose(dtheta_array, malla.dtheta, rtol=1e-10)


def test_malla_periodicidad_angular():
    """θ=0 y θ=2π deben ser equivalentes (periodicidad)."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # El último θ debe ser ~2π
    assert_allclose(malla.theta[-1], 2 * np.pi, rtol=1e-6)

    # θ[0] y θ[-1] difieren por 2π
    assert_allclose(malla.theta[-1] - malla.theta[0], 2 * np.pi, rtol=1e-6)


# ============================================================================
# TESTS DE INFORMACIÓN DE LA MALLA
# ============================================================================


def test_obtener_info_malla():
    """Debe poder obtener información resumida de la malla."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    info = malla.obtener_info()

    # Debe retornar un diccionario
    assert isinstance(info, dict)

    # Debe contener información clave
    assert "nr" in info
    assert "ntheta" in info
    assert "dr" in info
    assert "dtheta" in info
    assert "area_total" in info
    assert "area_defecto" in info
    assert "fraccion_defecto" in info


def test_str_representacion():
    """__str__ debe retornar información legible."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    repr_str = str(malla)

    # Debe contener información clave
    assert "MallaPolar2D" in repr_str
    assert "61" in repr_str or str(params.mallado.nr) in repr_str
    assert "96" in repr_str or str(params.mallado.ntheta) in repr_str


# ============================================================================
# TESTS DE CASOS LÍMITE
# ============================================================================


def test_malla_con_parametros_minimos():
    """Debe funcionar con número mínimo de nodos."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros, MalladoParams

    # Crear parámetros custom con malla mínima
    params = ParametrosMaestros()

    # Intentar crear malla con pocos nodos (debe funcionar pero con warning)
    malla = MallaPolar2D(params)

    assert malla is not None


def test_malla_consistencia_con_parametros():
    """La malla debe ser consistente con los parámetros dados."""
    from src.geometria.mallado import MallaPolar2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    malla = MallaPolar2D(params)

    # Todos los parámetros de malla deben coincidir
    assert malla.nr == params.mallado.nr
    assert malla.ntheta == params.mallado.ntheta
    assert_allclose(malla.dr, params.mallado.dr, rtol=1e-10)
    assert_allclose(malla.dtheta, params.mallado.dtheta, rtol=1e-10)


# ============================================================================
# FIN DE TESTS
# ============================================================================
