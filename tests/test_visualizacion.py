"""
Tests unitarios para el módulo de visualización.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-29

Tests para los 3 gráficos obligatorios de la sección 1.5:
1. Perfil de concentración en t=0
2. Perfil al 50% del tiempo
3. Perfil en estado estacionario
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para tests
import matplotlib.pyplot as plt


# ============================================================================
# TESTS DE MÓDULO BASE
# ============================================================================


def test_modulo_visualizacion_existe():
    """Módulo de visualización debe existir."""
    import src.postproceso.visualizacion as viz
    
    assert viz is not None


def test_configurar_estilo_existe():
    """Función para configurar estilo de matplotlib debe existir."""
    from src.postproceso.visualizacion import configurar_estilo_matplotlib
    
    assert configurar_estilo_matplotlib is not None


def test_configurar_estilo_aplica_cambios():
    """configurar_estilo_matplotlib debe modificar rcParams."""
    from src.postproceso.visualizacion import configurar_estilo_matplotlib
    
    original_dpi = plt.rcParams['figure.dpi']
    configurar_estilo_matplotlib()
    
    # DPI debe ser 150 para desarrollo
    assert plt.rcParams['figure.dpi'] == 150


# ============================================================================
# TESTS DE GRÁFICO 1: PERFIL EN t=0
# ============================================================================


def test_plot_perfil_t0_existe():
    """Función plot_perfil_t0 debe existir."""
    from src.postproceso.visualizacion import plot_perfil_t0
    
    assert plot_perfil_t0 is not None


def test_plot_perfil_t0_devuelve_figura():
    """plot_perfil_t0 debe retornar figura y axes."""
    from src.postproceso.visualizacion import plot_perfil_t0
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.zeros((malla.nr, malla.ntheta))
    
    fig, ax = plot_perfil_t0(malla.r, malla.theta, C, params)
    
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    plt.close(fig)


def test_plot_perfil_t0_marca_defecto():
    """plot_perfil_t0 debe marcar región de defecto."""
    from src.postproceso.visualizacion import plot_perfil_t0
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.zeros((malla.nr, malla.ntheta))
    
    fig, ax = plot_perfil_t0(malla.r, malla.theta, C, params)
    
    # Verificar que hay líneas en el plot (marcadores de defecto)
    assert len(ax.lines) >= 2  # Al menos 2 líneas para marcar defecto
    plt.close(fig)


def test_plot_perfil_t0_tiene_colorbar():
    """plot_perfil_t0 debe incluir colorbar."""
    from src.postproceso.visualizacion import plot_perfil_t0
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.zeros((malla.nr, malla.ntheta))
    
    fig, ax = plot_perfil_t0(malla.r, malla.theta, C, params)
    
    # Verificar que se creó colorbar
    assert len(fig.axes) == 2  # ax principal + colorbar
    plt.close(fig)


# ============================================================================
# TESTS DE GRÁFICO 2: PERFIL AL 50%
# ============================================================================


def test_plot_perfil_50pct_existe():
    """Función plot_perfil_50pct debe existir."""
    from src.postproceso.visualizacion import plot_perfil_50pct
    
    assert plot_perfil_50pct is not None


def test_plot_perfil_50pct_devuelve_figura_y_2_axes():
    """plot_perfil_50pct debe retornar figura y 2 axes (2D + 3D)."""
    from src.postproceso.visualizacion import plot_perfil_50pct
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.ones((malla.nr, malla.ntheta)) * 0.005
    tiempo = 10.0
    
    fig, (ax1, ax2) = plot_perfil_50pct(malla.r, malla.theta, C, tiempo, params)
    
    assert isinstance(fig, plt.Figure)
    assert ax1 is not None
    assert ax2 is not None
    plt.close(fig)


def test_plot_perfil_50pct_tiene_subplot_3d():
    """plot_perfil_50pct debe incluir subplot 3D."""
    from src.postproceso.visualizacion import plot_perfil_50pct
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    from mpl_toolkits.mplot3d import Axes3D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.ones((malla.nr, malla.ntheta)) * 0.005
    tiempo = 10.0
    
    fig, (ax1, ax2) = plot_perfil_50pct(malla.r, malla.theta, C, tiempo, params)
    
    # ax2 debe ser 3D
    assert isinstance(ax2, Axes3D)
    plt.close(fig)


# ============================================================================
# TESTS DE GRÁFICO 3: PERFIL EN ESTADO ESTACIONARIO
# ============================================================================


def test_plot_perfil_estado_estacionario_existe():
    """Función plot_perfil_estado_estacionario debe existir."""
    from src.postproceso.visualizacion import plot_perfil_estado_estacionario
    
    assert plot_perfil_estado_estacionario is not None


def test_plot_perfil_estado_estacionario_devuelve_figura_y_3_axes():
    """plot_perfil_estado_estacionario debe retornar figura y 3 axes."""
    from src.postproceso.visualizacion import plot_perfil_estado_estacionario
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.ones((malla.nr, malla.ntheta)) * params.operacion.C_bulk
    tiempo = 100.0
    
    fig, (ax1, ax2, ax3) = plot_perfil_estado_estacionario(
        malla.r, malla.theta, C, tiempo, params
    )
    
    assert isinstance(fig, plt.Figure)
    assert ax1 is not None  # 2D polar
    assert ax2 is not None  # 3D
    assert ax3 is not None  # Perfiles radiales
    plt.close(fig)


def test_plot_perfil_estado_estacionario_compara_activo_vs_defecto():
    """plot_perfil_estado_estacionario debe mostrar comparación radial."""
    from src.postproceso.visualizacion import plot_perfil_estado_estacionario
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.ones((malla.nr, malla.ntheta)) * params.operacion.C_bulk
    tiempo = 100.0
    
    fig, (ax1, ax2, ax3) = plot_perfil_estado_estacionario(
        malla.r, malla.theta, C, tiempo, params
    )
    
    # ax3 debe tener al menos 2 líneas (activo vs defecto)
    assert len(ax3.lines) >= 2
    plt.close(fig)


# ============================================================================
# TESTS DE GUARDADO DE FIGURAS
# ============================================================================


def test_guardar_figura_existe():
    """Función guardar_figura debe existir."""
    from src.postproceso.visualizacion import guardar_figura
    
    assert guardar_figura is not None


def test_guardar_figura_alta_resolucion():
    """guardar_figura debe usar 300 DPI para reporte."""
    from src.postproceso.visualizacion import guardar_figura, plot_perfil_t0
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    import tempfile
    import os
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    C = np.zeros((malla.nr, malla.ntheta))
    
    fig, ax = plot_perfil_t0(malla.r, malla.theta, C, params)
    
    # Guardar en archivo temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_fig.png")
        guardar_figura(fig, filepath, dpi=300)
        
        # Verificar que se creó el archivo
        assert os.path.exists(filepath)
    
    plt.close(fig)


# ============================================================================
# TESTS DE INTEGRACIÓN CON SOLVER
# ============================================================================


def test_visualizar_desde_solver():
    """Debe poder visualizar resultados directamente desde solver."""
    from src.postproceso.visualizacion import plot_perfil_t0
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros
    
    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.01)
    
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)
    
    # Ejecutar 10 pasos
    for _ in range(10):
        solver.paso_temporal()
    
    # Visualizar campo actual
    fig, ax = plot_perfil_t0(solver.malla.r, solver.malla.theta, solver.C, params)
    
    assert fig is not None
    plt.close(fig)

