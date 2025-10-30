"""
Tests unitarios para el verificador de balance de masa.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-29

Tests siguiendo TDD para Mini-Tarea D3:
- Balance de masa verificador
- Cálculo de masa total
- Cálculo de flujos y consumo
- Verificación de conservación
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# TESTS DE CLASE BASE
# ============================================================================


def test_balance_masa_verificador_existe():
    """Clase BalanceMasaVerificador debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    assert BalanceMasaVerificador is not None


def test_balance_masa_verificador_inicializacion():
    """Debe inicializarse con tolerancia."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador(tolerancia_relativa=0.01)
    
    assert verificador.tolerancia_relativa == 0.01
    assert hasattr(verificador, "historial_masa")
    assert hasattr(verificador, "historial_tiempo")


# ============================================================================
# TESTS DE CÁLCULO DE MASA TOTAL
# ============================================================================


def test_calcular_masa_total_existe():
    """Método calcular_masa_total debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador()
    assert hasattr(verificador, "calcular_masa_total")


def test_calcular_masa_total_campo_cero():
    """Masa total de campo C=0 debe ser cero."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C = np.zeros((malla.nr, malla.ntheta))
    
    masa_total = verificador.calcular_masa_total(C, malla.r, malla.theta)
    
    assert_allclose(masa_total, 0.0, atol=1e-12)


def test_calcular_masa_total_campo_constante():
    """Masa total de campo constante debe ser C·Área."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C_const = 0.01  # mol/m³
    C = np.full((malla.nr, malla.ntheta), C_const)
    
    masa_total = verificador.calcular_masa_total(C, malla.r, malla.theta)
    
    # Masa esperada = C · πR²
    area_pellet = np.pi * params.geometria.R**2
    masa_esperada = C_const * area_pellet
    
    assert_allclose(masa_total, masa_esperada, rtol=0.03)  # Relajado por error numérico


# ============================================================================
# TESTS DE CÁLCULO DE CONSUMO POR REACCIÓN
# ============================================================================


def test_calcular_consumo_reaccion_existe():
    """Método calcular_consumo_reaccion debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador()
    assert hasattr(verificador, "calcular_consumo_reaccion")


def test_calcular_consumo_reaccion_k_app_cero():
    """Consumo con k_app=0 debe ser cero."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C = np.ones((malla.nr, malla.ntheta))
    k_app = np.zeros((malla.nr, malla.ntheta))
    
    consumo = verificador.calcular_consumo_reaccion(C, malla.r, malla.theta, k_app)
    
    assert_allclose(consumo, 0.0, atol=1e-12)


def test_calcular_consumo_reaccion_C_cero():
    """Consumo con C=0 debe ser cero."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C = np.zeros((malla.nr, malla.ntheta))
    k_app_field = malla.generar_campo_k_app()
    
    consumo = verificador.calcular_consumo_reaccion(C, malla.r, malla.theta, k_app_field)
    
    assert_allclose(consumo, 0.0, atol=1e-12)


# ============================================================================
# TESTS DE CÁLCULO DE FLUJO EN FRONTERA
# ============================================================================


def test_calcular_flujo_frontera_existe():
    """Método calcular_flujo_frontera debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador()
    assert hasattr(verificador, "calcular_flujo_frontera")


def test_calcular_flujo_frontera_C_igual_C_bulk():
    """Flujo debe ser cero si C_superficie = C_bulk."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C_bulk = params.operacion.C_bulk
    C = np.full((malla.nr, malla.ntheta), C_bulk)
    k_c = params.transferencia.k_c
    
    flujo = verificador.calcular_flujo_frontera(
        C, malla.r, malla.theta, k_c, C_bulk
    )
    
    assert_allclose(flujo, 0.0, atol=1e-12)


def test_calcular_flujo_frontera_positivo_si_C_menor_C_bulk():
    """Flujo debe ser positivo (entrante) si C_superficie < C_bulk."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C_bulk = params.operacion.C_bulk
    C = np.full((malla.nr, malla.ntheta), C_bulk * 0.5)  # C < C_bulk
    k_c = params.transferencia.k_c
    
    flujo = verificador.calcular_flujo_frontera(
        C, malla.r, malla.theta, k_c, C_bulk
    )
    
    assert flujo > 0  # Flujo entrante


# ============================================================================
# TESTS DE VERIFICACIÓN DE BALANCE
# ============================================================================


def test_verificar_balance_existe():
    """Método verificar_balance debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador()
    assert hasattr(verificador, "verificar_balance")


def test_verificar_balance_devuelve_dict():
    """verificar_balance debe retornar diccionario con métricas."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    k_app_field = malla.generar_campo_k_app()
    verificador = BalanceMasaVerificador(tolerancia_relativa=2.0)  # Tolerancia relajada para test
    
    C_n = np.ones((malla.nr, malla.ntheta)) * 0.001
    C_np1 = np.ones((malla.nr, malla.ntheta)) * 0.0011
    dt = 0.001
    
    resultado = verificador.verificar_balance(
        C_n, C_np1, malla.r, malla.theta, dt,
        k_app_field, params.transferencia.k_c, params.operacion.C_bulk, tiempo=0.001
    )
    
    assert isinstance(resultado, dict)
    assert "masa_total" in resultado
    assert "delta_masa" in resultado
    assert "flujo_entrada" in resultado
    assert "consumo_reaccion" in resultado
    assert "error_relativo" in resultado


def test_verificar_balance_sin_reaccion_ni_flujo():
    """Balance con k_app=0 y C=C_bulk debe ser perfecto."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    verificador = BalanceMasaVerificador()
    
    C_bulk = params.operacion.C_bulk
    C_n = np.full((malla.nr, malla.ntheta), C_bulk)
    C_np1 = C_n.copy()  # Sin cambio
    dt = 0.001
    k_app_field = np.zeros((malla.nr, malla.ntheta))
    
    resultado = verificador.verificar_balance(
        C_n, C_np1, malla.r, malla.theta, dt,
        k_app_field, params.transferencia.k_c, C_bulk, tiempo=0.001
    )
    
    # Error debe ser muy pequeño
    assert resultado["error_relativo"] < 1e-6


# ============================================================================
# TESTS DE HISTORIAL Y REPORTES
# ============================================================================


def test_historial_se_guarda():
    """Historial debe guardarse tras verificar balance."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    k_app_field = malla.generar_campo_k_app()
    verificador = BalanceMasaVerificador(tolerancia_relativa=2.0)  # Tolerancia relajada para test
    
    C_n = np.ones((malla.nr, malla.ntheta)) * 0.001
    C_np1 = C_n * 1.001
    
    assert len(verificador.historial_masa) == 0
    
    verificador.verificar_balance(
        C_n, C_np1, malla.r, malla.theta, 0.001,
        k_app_field, params.transferencia.k_c, params.operacion.C_bulk, tiempo=0.001
    )
    
    assert len(verificador.historial_masa) == 1
    assert len(verificador.historial_tiempo) == 1


def test_generar_reporte_existe():
    """Método generar_reporte debe existir."""
    from src.solver.balance_masa import BalanceMasaVerificador
    
    verificador = BalanceMasaVerificador()
    assert hasattr(verificador, "generar_reporte")


def test_generar_reporte_con_historial():
    """generar_reporte debe devolver string con estadísticas."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    k_app_field = malla.generar_campo_k_app()
    verificador = BalanceMasaVerificador(tolerancia_relativa=2.0)  # Tolerancia relajada para test
    
    # Simular 3 verificaciones
    C = np.ones((malla.nr, malla.ntheta)) * 0.001
    for i in range(3):
        verificador.verificar_balance(
            C, C, malla.r, malla.theta, 0.001,
            k_app_field, params.transferencia.k_c, params.operacion.C_bulk, tiempo=i*0.001
        )
    
    reporte = verificador.generar_reporte()
    
    assert isinstance(reporte, str)
    assert "BALANCE DE MASA" in reporte or "balance" in reporte.lower()
    assert "Masa inicial" in reporte or "masa" in reporte.lower()


# ============================================================================
# TESTS DE TOLERANCIA Y EXCEPCIONES
# ============================================================================


def test_verificar_balance_raise_si_excede_tolerancia():
    """Debe lanzar RuntimeError si error > tolerancia."""
    from src.solver.balance_masa import BalanceMasaVerificador
    from src.config.parametros import ParametrosMaestros
    from src.geometria.mallado import MallaPolar2D
    
    params = ParametrosMaestros()
    malla = MallaPolar2D(params)
    k_app_field = malla.generar_campo_k_app()
    verificador = BalanceMasaVerificador(tolerancia_relativa=1e-10)  # Tolerancia muy estricta
    
    C_n = np.ones((malla.nr, malla.ntheta)) * 0.001
    C_np1 = C_n * 2.0  # Cambio artificial muy grande
    
    # Debería lanzar error por violación de balance
    with pytest.raises(RuntimeError, match="Balance de masa violado"):
        verificador.verificar_balance(
            C_n, C_np1, malla.r, malla.theta, 0.001,
            k_app_field, params.transferencia.k_c, params.operacion.C_bulk, tiempo=0.001
        )

