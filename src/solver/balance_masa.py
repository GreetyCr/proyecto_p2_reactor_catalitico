"""
Verificador de balance de masa para simulación Crank-Nicolson 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-29

Módulo para verificar continuamente que el balance de masa se conserve
durante la simulación, con tolerancia < 1%.

Balance: d(Masa)/dt = Flujo_entrada - Consumo_reacción
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class BalanceMasaVerificador:
    """
    Verifica conservación de masa en cada paso temporal.

    Ecuación de balance:
        d(Masa)/dt = Flujo_entrada(r=R) - Consumo_reacción

    Attributes
    ----------
    tolerancia_relativa : float
        Error relativo máximo aceptable (default: 1%)
    historial_masa : List[float]
        Historial de masa total en cada verificación
    historial_tiempo : List[float]
        Historial de tiempos de verificación
    """

    def __init__(self, tolerancia_relativa: float = 0.01):
        """
        Inicializa verificador de balance de masa.

        Parameters
        ----------
        tolerancia_relativa : float, optional
            Error relativo máximo aceptable (default: 0.01 = 1%)
        """
        self.tolerancia_relativa = tolerancia_relativa
        self.historial_masa: List[float] = []
        self.historial_tiempo: List[float] = []

    def calcular_masa_total(
        self, C: np.ndarray, r: np.ndarray, theta: np.ndarray
    ) -> float:
        """
        Calcula masa total en el dominio.

        Integral: ∫∫ C(r,θ)·r dr dθ

        Parameters
        ----------
        C : np.ndarray, shape (nr, ntheta)
            Campo de concentración [mol/m³]
        r : np.ndarray, shape (nr,)
            Array radial [m]
        theta : np.ndarray, shape (ntheta,)
            Array angular [rad]

        Returns
        -------
        masa_total : float
            Masa total en el dominio [mol]
        """
        nr, ntheta = C.shape
        dr = r[1] - r[0]
        dtheta = theta[1] - theta[0]

        # Integración usando regla del trapecio 2D
        masa_total = 0.0
        for i in range(nr):
            for j in range(ntheta):
                # Elemento de área: r·dr·dθ
                dA = r[i] * dr * dtheta
                masa_total += C[i, j] * dA

        return masa_total

    def calcular_consumo_reaccion(
        self, C: np.ndarray, r: np.ndarray, theta: np.ndarray, k_app: np.ndarray
    ) -> float:
        """
        Calcula tasa total de consumo por reacción en el dominio.

        Integral: ∫∫ k_app·C(r,θ)·r dr dθ

        Parameters
        ----------
        C : np.ndarray, shape (nr, ntheta)
            Campo de concentración [mol/m³]
        r : np.ndarray, shape (nr,)
            Array radial [m]
        theta : np.ndarray, shape (ntheta,)
            Array angular [rad]
        k_app : np.ndarray, shape (nr, ntheta)
            Campo de constante cinética aparente [1/s]

        Returns
        -------
        consumo_total : float
            Tasa de consumo total [mol/s]
        """
        nr, ntheta = C.shape
        dr = r[1] - r[0]
        dtheta = theta[1] - theta[0]

        consumo_total = 0.0
        for i in range(nr):
            for j in range(ntheta):
                dA = r[i] * dr * dtheta
                consumo_total += k_app[i, j] * C[i, j] * dA

        return consumo_total

    def calcular_flujo_frontera(
        self,
        C: np.ndarray,
        r: np.ndarray,
        theta: np.ndarray,
        k_c: float,
        C_bulk: float,
    ) -> float:
        """
        Calcula flujo total entrante en r=R (frontera externa).

        Integral: ∫ k_c·(C_bulk - C_s)·R dθ

        Parameters
        ----------
        C : np.ndarray, shape (nr, ntheta)
            Campo de concentración [mol/m³]
        r : np.ndarray, shape (nr,)
            Array radial [m]
        theta : np.ndarray, shape (ntheta,)
            Array angular [rad]
        k_c : float
            Coeficiente de transferencia de masa [m/s]
        C_bulk : float
            Concentración en el bulk [mol/m³]

        Returns
        -------
        flujo_total : float
            Flujo total entrante [mol/s]
        """
        # C_s es concentración superficial (última fila radial)
        C_s = C[-1, :]

        R = r[-1]
        dtheta = theta[1] - theta[0]

        flujo_total = 0.0
        for j in range(len(theta)):
            flujo_local = k_c * (C_bulk - C_s[j])
            flujo_total += flujo_local * R * dtheta

        return flujo_total

    def verificar_balance(
        self,
        C_n: np.ndarray,
        C_np1: np.ndarray,
        r: np.ndarray,
        theta: np.ndarray,
        dt: float,
        k_app: np.ndarray,
        k_c: float,
        C_bulk: float,
        tiempo: float,
    ) -> Dict[str, float]:
        """
        Verifica balance de masa entre pasos temporales.

        Balance: d(Masa)/dt = Flujo_entrada - Consumo_reaccion

        Parameters
        ----------
        C_n : np.ndarray
            Campo de concentración en paso n
        C_np1 : np.ndarray
            Campo de concentración en paso n+1
        r : np.ndarray
            Array radial
        theta : np.ndarray
            Array angular
        dt : float
            Paso temporal [s]
        k_app : np.ndarray
            Campo de constante cinética
        k_c : float
            Coeficiente de transferencia de masa [m/s]
        C_bulk : float
            Concentración en el bulk [mol/m³]
        tiempo : float
            Tiempo actual de simulación [s]

        Returns
        -------
        resultado : Dict[str, float]
            Diccionario con métricas de balance

        Raises
        ------
        RuntimeError
            Si el error relativo excede la tolerancia
        """
        # Calcular masas
        masa_n = self.calcular_masa_total(C_n, r, theta)
        masa_np1 = self.calcular_masa_total(C_np1, r, theta)

        # Cambio de masa
        delta_masa = masa_np1 - masa_n

        # Flujo entrante (promedio entre n y n+1)
        flujo_entrada = 0.5 * (
            self.calcular_flujo_frontera(C_n, r, theta, k_c, C_bulk)
            + self.calcular_flujo_frontera(C_np1, r, theta, k_c, C_bulk)
        )

        # Consumo por reacción (promedio entre n y n+1)
        reaccion = 0.5 * (
            self.calcular_consumo_reaccion(C_n, r, theta, k_app)
            + self.calcular_consumo_reaccion(C_np1, r, theta, k_app)
        )

        # Balance esperado
        balance_esperado = flujo_entrada * dt - reaccion * dt

        # Error
        error_absoluto = abs(delta_masa - balance_esperado)
        error_relativo = error_absoluto / (abs(balance_esperado) + 1e-12)

        # Guardar historial
        self.historial_masa.append(masa_np1)
        self.historial_tiempo.append(tiempo)

        # Verificar tolerancia
        if error_relativo > self.tolerancia_relativa:
            raise RuntimeError(
                f"Balance de masa violado en t={tiempo:.6f}s:\n"
                f"  Cambio masa: {delta_masa:.6e}\n"
                f"  Balance esperado: {balance_esperado:.6e}\n"
                f"  Error relativo: {error_relativo:.2%} > {self.tolerancia_relativa:.2%}"
            )

        return {
            "masa_total": masa_np1,
            "delta_masa": delta_masa,
            "flujo_entrada": flujo_entrada * dt,
            "consumo_reaccion": reaccion * dt,
            "error_relativo": error_relativo,
        }

    def generar_reporte(self) -> str:
        """
        Genera reporte de conservación de masa.

        Returns
        -------
        reporte : str
            Reporte formateado con estadísticas
        """
        if len(self.historial_masa) == 0:
            return "No hay datos en el historial de balance de masa."

        masa_inicial = self.historial_masa[0]
        masa_final = self.historial_masa[-1]
        cambio_total = (masa_final - masa_inicial) / (masa_inicial + 1e-12)

        reporte = f"""
╔══════════════════════════════════════════════════╗
║     REPORTE DE BALANCE DE MASA                   ║
╚══════════════════════════════════════════════════╝

Tiempo inicial: {self.historial_tiempo[0]:.6f} s
Tiempo final:   {self.historial_tiempo[-1]:.6f} s

Masa inicial:   {masa_inicial:.6e} mol
Masa final:     {masa_final:.6e} mol
Cambio total:   {cambio_total:.4%}

Pasos verificados: {len(self.historial_masa)}
Tolerancia:        {self.tolerancia_relativa:.2%}

{'✓ Balance de masa SATISFACTORIO' if abs(cambio_total) < self.tolerancia_relativa else '⚠ Balance de masa CUESTIONABLE'}
        """
        return reporte

