"""
Tests unitarios para el solver Crank-Nicolson 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Tests siguiendo TDD para Mini-Tarea D1:
- Clase base CrankNicolsonSolver2D
- Inicialización y setup
- Construcción de sistema completo
- Preparación para loop temporal
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
import logging


# ============================================================================
# TESTS DE INICIALIZACIÓN
# ============================================================================


def test_solver_cn_clase_existe():
    """La clase CrankNicolsonSolver2D debe existir."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D

    assert CrankNicolsonSolver2D is not None


def test_solver_cn_inicializacion():
    """Debe inicializarse con parámetros."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()

    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert solver is not None
    assert solver.params == params
    assert solver.dt == 0.001


def test_solver_cn_tiene_malla():
    """Debe crear o recibir malla."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "malla")
    assert solver.malla is not None


def test_solver_cn_tiene_campo_k_app():
    """Debe tener campo k_app generado."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "k_app_field")
    assert solver.k_app_field is not None


# ============================================================================
# TESTS DE CONSTRUCCIÓN DE SISTEMA
# ============================================================================


def test_solver_cn_construir_sistema_existe():
    """Debe existir método para construir sistema."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "construir_sistema")


def test_solver_cn_construir_sistema_crea_matrices():
    """construir_sistema debe crear matrices A y B."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()

    assert hasattr(solver, "A")
    assert hasattr(solver, "B")
    assert solver.A is not None
    assert solver.B is not None


def test_solver_cn_matrices_son_dispersas():
    """Matrices A y B deben ser dispersas."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()

    assert sparse.issparse(solver.A)
    assert sparse.issparse(solver.B)


def test_solver_cn_matrices_tienen_shape_correcto():
    """Matrices deben tener shape (N, N)."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()

    N = solver.malla.nr * solver.malla.ntheta

    assert solver.A.shape == (N, N)
    assert solver.B.shape == (N, N)


def test_solver_cn_aplica_todas_condiciones_frontera():
    """Debe aplicar todas las condiciones de frontera."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()

    # Debe tener vector de términos fuente
    assert hasattr(solver, "b_robin")
    assert solver.b_robin is not None


# ============================================================================
# TESTS DE CONDICIÓN INICIAL
# ============================================================================


def test_solver_cn_inicializar_campo_existe():
    """Debe existir método para inicializar campo C."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "inicializar_campo")


def test_solver_cn_campo_inicial_shape():
    """Campo inicial debe tener shape (nr, ntheta)."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.inicializar_campo(C_inicial=0.0)

    assert solver.C.shape == (solver.malla.nr, solver.malla.ntheta)


def test_solver_cn_campo_inicial_ceros():
    """Condición inicial C=0 debe dar campo de ceros."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.inicializar_campo(C_inicial=0.0)

    assert_allclose(solver.C, 0.0, atol=1e-12)


def test_solver_cn_campo_inicial_constante():
    """Condición inicial C=const debe dar campo constante."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    C_const = 0.005

    solver.inicializar_campo(C_inicial=C_const)

    assert_allclose(solver.C, C_const, rtol=1e-10)


# ============================================================================
# TESTS DE PASO TEMPORAL
# ============================================================================


def test_solver_cn_paso_temporal_existe():
    """Debe existir método paso_temporal."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "paso_temporal")


def test_solver_cn_paso_temporal_actualiza_campo():
    """paso_temporal debe actualizar el campo C."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    C_antes = solver.C.copy()

    solver.paso_temporal()

    C_despues = solver.C

    # El campo debe cambiar (excepto si es estado estacionario inicial)
    # Con C=0 inicial y C_bulk > 0, debe haber cambio
    assert not np.allclose(C_despues, C_antes)


def test_solver_cn_paso_temporal_incrementa_tiempo():
    """paso_temporal debe incrementar t."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    t_antes = solver.t

    solver.paso_temporal()

    assert_allclose(solver.t, t_antes + solver.dt)


def test_solver_cn_paso_temporal_incrementa_iteracion():
    """paso_temporal debe incrementar n_iter."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    n_antes = solver.n_iter

    solver.paso_temporal()

    assert solver.n_iter == n_antes + 1


# ============================================================================
# TESTS DE OBTENER INFO
# ============================================================================


def test_solver_cn_obtener_info_existe():
    """Debe existir método para obtener información."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "obtener_info")


def test_solver_cn_obtener_info_contenido():
    """obtener_info debe retornar dict con datos clave."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    info = solver.obtener_info()

    assert "t" in info
    assert "n_iter" in info
    assert "dt" in info
    assert "C_min" in info
    assert "C_max" in info


# ============================================================================
# TESTS DE LOOP TEMPORAL Y CONVERGENCIA
# ============================================================================


def test_solver_cn_ejecutar_existe():
    """Debe existir método ejecutar para loop temporal."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "ejecutar")


def test_solver_cn_ejecutar_n_pasos():
    """ejecutar(n_pasos) debe ejecutar exactamente n pasos."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    n_pasos = 10
    solver.ejecutar(n_pasos=n_pasos)

    assert solver.n_iter == n_pasos
    assert_allclose(solver.t, n_pasos * solver.dt)


def test_solver_cn_ejecutar_hasta_tiempo():
    """ejecutar(t_final) debe ejecutar hasta alcanzar t_final."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    t_final = 0.010  # 10 ms
    solver.ejecutar(t_final=t_final)

    # Debe haber alcanzado o superado t_final
    assert solver.t >= t_final - solver.dt / 2


def test_solver_cn_detectar_convergencia_existe():
    """Debe existir método para detectar convergencia."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    assert hasattr(solver, "verificar_convergencia")


def test_solver_cn_campo_estacionario_convergido():
    """verificar_convergencia detecta campos casi idénticos."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.01)

    # Guardar campo anterior
    C_anterior = solver.C.copy()
    
    # Modificar campo actual solo un poquito
    solver.C = solver.C + 1e-6  # Cambio minúsculo

    # Verificar convergencia con campos casi idénticos
    ha_convergido = solver.verificar_convergencia(C_anterior, tol=1e-4)

    # Debería detectar convergencia
    assert ha_convergido == True


def test_solver_cn_campo_transitorio_no_convergido():
    """Campo en transitorio debe detectarse como NO convergido."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.construir_sistema()

    # Inicializar con C=0 (lejos del estado estacionario)
    solver.inicializar_campo(C_inicial=0.0)

    C_anterior = solver.C.copy()

    # Ejecutar 1 paso
    solver.paso_temporal()

    # Verificar convergencia
    ha_convergido = solver.verificar_convergencia(C_anterior, tol=1e-6)

    # NO debería haber convergido
    assert ha_convergido == False


def test_solver_cn_ejecutar_con_convergencia():
    """ejecutar con check_convergencia debe funcionar sin errores."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.01)

    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.01)  # Cercano a C_bulk

    # Ejecutar SOLO 50 pasos con convergencia
    # Este test solo verifica que el mecanismo NO crashee
    solver.ejecutar(
        n_pasos=50,  # MUY bajo para test rápido
        check_convergencia=True,
        tol_convergencia=1e-2  # Tolerancia muy relajada
    )

    # El test solo verifica que funcione sin errores
    assert solver.n_iter <= 50
    assert solver.t > 0


# ============================================================================
# TESTS DE HISTORIAL
# ============================================================================


def test_solver_cn_guardar_historial_existe():
    """Debe poder guardar historial de la simulación."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    # Debe tener atributo para habilitar historial
    assert hasattr(solver, "habilitar_historial")


def test_solver_cn_historial_guarda_campos():
    """Con historial habilitado, debe guardar campos."""
    from src.solver.crank_nicolson import CrankNicolsonSolver2D
    from src.config.parametros import ParametrosMaestros

    params = ParametrosMaestros()
    solver = CrankNicolsonSolver2D(params, dt=0.001)

    solver.habilitar_historial(cada_n_pasos=5)
    solver.construir_sistema()
    solver.inicializar_campo(C_inicial=0.0)

    # Ejecutar 20 pasos (debe guardar en pasos 0, 5, 10, 15, 20)
    solver.ejecutar(n_pasos=20)

    # Debe tener historial
    assert hasattr(solver, "historial_C")
    assert hasattr(solver, "historial_t")
    assert len(solver.historial_C) > 0
    assert len(solver.historial_t) > 0


# ============================================================================
# FIN DE TESTS
# ============================================================================

