"""
Módulo de Geometría y Mallado Polar 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este módulo implementa la generación de malla 2D en coordenadas polares (r, θ)
para el pellet catalítico cilíndrico con defecto.

Características:
- Malla uniforme en coordenadas polares
- Identificación de región de defecto vs activa
- Conversión a coordenadas cartesianas
- Cálculo de propiedades geométricas
- Visualización de la malla y regiones
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CLASE PRINCIPAL: MALLA POLAR 2D
# ============================================================================


class MallaPolar2D:
    """
    Representa una malla 2D en coordenadas polares (r, θ).

    Genera discretización espacial del pellet catalítico cilíndrico,
    identificando regiones activa y defectuosa.

    Attributes
    ----------
    r : np.ndarray
        Array de coordenadas radiales [m]
    theta : np.ndarray
        Array de coordenadas angulares [rad]
    nr : int
        Número de nodos radiales
    ntheta : int
        Número de nodos angulares
    dr : float
        Paso radial [m]
    dtheta : float
        Paso angular [rad]
    R_grid : np.ndarray
        Meshgrid de coordenadas radiales shape (nr, ntheta)
    THETA_grid : np.ndarray
        Meshgrid de coordenadas angulares shape (nr, ntheta)

    Parameters
    ----------
    params : ParametrosMaestros
        Parámetros del proyecto

    Examples
    --------
    >>> from src.config.parametros import ParametrosMaestros
    >>> from src.geometria.mallado import MallaPolar2D
    >>> params = ParametrosMaestros()
    >>> malla = MallaPolar2D(params)
    >>> print(f"Malla: {malla.nr}×{malla.ntheta} nodos")
    >>> mascara_defecto = malla.identificar_region_defecto()
    """

    def __init__(self, params):
        """
        Inicializa la malla polar 2D.

        Parameters
        ----------
        params : ParametrosMaestros
            Parámetros del proyecto
        """
        # Guardar referencia a parámetros
        self.params = params

        # Extraer parámetros de geometría
        self.R = params.geometria.R
        self.r1 = params.geometria.r1
        self.r2 = params.geometria.r2
        self.theta1 = params.geometria.theta1
        self.theta2 = params.geometria.theta2

        # Extraer parámetros de mallado
        self.nr = params.mallado.nr
        self.ntheta = params.mallado.ntheta
        self.dr = params.mallado.dr
        self.dtheta = params.mallado.dtheta

        # Generar arrays de coordenadas
        self._generar_arrays()

        # Crear meshgrids
        self._crear_meshgrids()

        logger.info(
            f"MallaPolar2D creada: {self.nr}×{self.ntheta} nodos, "
            f"dr={self.dr:.3e}m, dθ={self.dtheta:.3f}rad"
        )

    def _generar_arrays(self):
        """Genera arrays de coordenadas radiales y angulares."""
        # Array radial: de 0 a R con nr nodos
        self.r = np.linspace(0, self.R, self.nr)

        # Array angular: de 0 a 2π con ntheta nodos
        self.theta = np.linspace(0, 2 * np.pi, self.ntheta)

    def _crear_meshgrids(self):
        """Crea meshgrids 2D de coordenadas."""
        # Meshgrid: R_grid[i,j] = r[i], THETA_grid[i,j] = theta[j]
        self.R_grid, self.THETA_grid = np.meshgrid(self.r, self.theta, indexing="ij")

    # ========================================================================
    # CONVERSIÓN A COORDENADAS CARTESIANAS
    # ========================================================================

    def obtener_coordenadas_cartesianas(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convierte malla polar a coordenadas cartesianas.

        Returns
        -------
        X : np.ndarray, shape (nr, ntheta)
            Coordenadas X en sistema cartesiano [m]
        Y : np.ndarray, shape (nr, ntheta)
            Coordenadas Y en sistema cartesiano [m]

        Notes
        -----
        Conversión:
            X = r × cos(θ)
            Y = r × sin(θ)

        Examples
        --------
        >>> X, Y = malla.obtener_coordenadas_cartesianas()
        >>> assert X[0, 0] == 0  # Centro
        """
        X = self.R_grid * np.cos(self.THETA_grid)
        Y = self.R_grid * np.sin(self.THETA_grid)

        return X, Y

    # ========================================================================
    # IDENTIFICACIÓN DE REGIONES
    # ========================================================================

    def identificar_region_defecto(self) -> np.ndarray:
        """
        Identifica la región del defecto (k_app = 0).

        Returns
        -------
        mascara_defecto : np.ndarray, shape (nr, ntheta), dtype=bool
            True donde hay defecto, False en región activa

        Notes
        -----
        Región de defecto:
        - Radial: r ∈ [r1, r2] = [R/3, 2R/3]
        - Angular: θ ∈ [θ1, θ2] = [0°, 45°]

        Examples
        --------
        >>> mascara = malla.identificar_region_defecto()
        >>> n_nodos_defecto = mascara.sum()
        """
        # Condiciones para estar en defecto
        en_rango_radial = (self.R_grid >= self.r1) & (self.R_grid <= self.r2)
        en_rango_angular = (self.THETA_grid >= self.theta1) & (
            self.THETA_grid <= self.theta2
        )

        # Defecto = ambas condiciones
        mascara_defecto = en_rango_radial & en_rango_angular

        return mascara_defecto

    def identificar_region_activa(self) -> np.ndarray:
        """
        Identifica la región activa (k_app > 0).

        Returns
        -------
        mascara_activa : np.ndarray, shape (nr, ntheta), dtype=bool
            True en región activa, False en defecto

        Notes
        -----
        La región activa es el complemento de la región de defecto.

        Examples
        --------
        >>> mascara = malla.identificar_region_activa()
        >>> assert mascara.sum() + malla.identificar_region_defecto().sum() == nr*ntheta
        """
        return ~self.identificar_region_defecto()

    # ========================================================================
    # CAMPO DE k_app
    # ========================================================================

    def generar_campo_k_app(self) -> np.ndarray:
        """
        Genera campo espacial de constante cinética aparente k_app.

        Returns
        -------
        k_app_field : np.ndarray, shape (nr, ntheta)
            Campo de k_app [s⁻¹]: 0 en defecto, k_app en activa

        Notes
        -----
        k_app(r,θ) = {
            0               si (r,θ) en defecto
            k_app_param     si (r,θ) en región activa
        }

        Examples
        --------
        >>> k_field = malla.generar_campo_k_app()
        >>> assert k_field[mascara_defecto].max() == 0
        """
        # Inicializar campo con valor de parámetro
        k_app_field = np.full((self.nr, self.ntheta), self.params.cinetica.k_app)

        # Poner ceros en región de defecto
        mascara_defecto = self.identificar_region_defecto()
        k_app_field[mascara_defecto] = 0.0

        return k_app_field

    # ========================================================================
    # ÍNDICES Y ACCESO
    # ========================================================================

    def encontrar_indice_mas_cercano(
        self, r_objetivo: float, theta_objetivo: float
    ) -> Tuple[int, int]:
        """
        Encuentra el índice (i,j) más cercano a las coordenadas dadas.

        Parameters
        ----------
        r_objetivo : float
            Coordenada radial objetivo [m]
        theta_objetivo : float
            Coordenada angular objetivo [rad]

        Returns
        -------
        i : int
            Índice radial más cercano
        j : int
            Índice angular más cercano

        Examples
        --------
        >>> i, j = malla.encontrar_indice_mas_cercano(R/2, np.pi/4)
        """
        # Encontrar índice radial más cercano
        i = np.argmin(np.abs(self.r - r_objetivo))

        # Encontrar índice angular más cercano
        j = np.argmin(np.abs(self.theta - theta_objetivo))

        return i, j

    def obtener_indice_centro(self) -> int:
        """
        Obtiene el índice del nodo central (r=0).

        Returns
        -------
        i_centro : int
            Índice radial del centro (siempre 0)

        Examples
        --------
        >>> i_centro = malla.obtener_indice_centro()
        >>> assert malla.r[i_centro] == 0
        """
        return 0

    def obtener_indice_frontera(self) -> int:
        """
        Obtiene el índice de los nodos en la frontera externa (r=R).

        Returns
        -------
        i_frontera : int
            Índice radial de la frontera (siempre nr-1)

        Examples
        --------
        >>> i_frontera = malla.obtener_indice_frontera()
        >>> assert malla.r[i_frontera] == R
        """
        return self.nr - 1

    # ========================================================================
    # PROPIEDADES GEOMÉTRICAS
    # ========================================================================

    def calcular_area_total(self) -> float:
        """
        Calcula el área total del pellet (círculo).

        Returns
        -------
        area : float
            Área total [m²]

        Notes
        -----
        A_total = π × R²

        Examples
        --------
        >>> area = malla.calcular_area_total()
        >>> assert np.isclose(area, np.pi * R**2)
        """
        return np.pi * self.R**2

    def calcular_area_defecto(self) -> float:
        """
        Calcula el área de la región de defecto (sector anular).

        Returns
        -------
        area_defecto : float
            Área de defecto [m²]

        Notes
        -----
        Área de sector anular:
        A_defecto = (θ2 - θ1) × (r2² - r1²) / 2

        Examples
        --------
        >>> area_def = malla.calcular_area_defecto()
        """
        return (self.theta2 - self.theta1) * (self.r2**2 - self.r1**2) / 2

    def calcular_fraccion_defecto(self) -> float:
        """
        Calcula la fracción de área defectuosa.

        Returns
        -------
        fraccion : float
            Fracción de área con defecto [adimensional, 0-1]

        Notes
        -----
        f_defecto = A_defecto / A_total

        Examples
        --------
        >>> frac = malla.calcular_fraccion_defecto()
        >>> assert 0 < frac < 1
        """
        return self.calcular_area_defecto() / self.calcular_area_total()

    # ========================================================================
    # VOLÚMENES DE CONTROL
    # ========================================================================

    def calcular_volumenes_control(self) -> np.ndarray:
        """
        Calcula volúmenes (áreas en 2D) de control para cada nodo.

        Returns
        -------
        volumenes : np.ndarray, shape (nr, ntheta)
            Área de control de cada nodo [m²]

        Notes
        -----
        Para nodo (i,j), el volumen de control es:
        V_{i,j} = r_i × Δr × Δθ

        En el centro (r=0), el volumen es especial.

        Examples
        --------
        >>> vols = malla.calcular_volumenes_control()
        >>> assert np.all(vols >= 0)
        """
        # Volúmenes de control
        volumenes = np.zeros((self.nr, self.ntheta))

        for i in range(self.nr):
            # Volumen = r × dr × dθ (área en 2D)
            volumenes[i, :] = self.r[i] * self.dr * self.dtheta

        return volumenes

    # ========================================================================
    # VISUALIZACIÓN
    # ========================================================================

    def visualizar_malla(
        self, mostrar: bool = True, figsize: Tuple[int, int] = (10, 10)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualiza la malla polar 2D en coordenadas cartesianas.

        Parameters
        ----------
        mostrar : bool, optional
            Si True, llama a plt.show()
        figsize : Tuple[int, int], optional
            Tamaño de la figura

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> fig, ax = malla.visualizar_malla(mostrar=False)
        """
        # Convertir a cartesianas
        X, Y = self.obtener_coordenadas_cartesianas()

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))

        # Plot de la malla
        ax.plot(X, Y, "k-", alpha=0.3, linewidth=0.5)  # Líneas radiales
        ax.plot(X.T, Y.T, "k-", alpha=0.3, linewidth=0.5)  # Líneas angulares

        # Marcar región de defecto
        mascara_defecto = self.identificar_region_defecto()
        X_defecto = X[mascara_defecto]
        Y_defecto = Y[mascara_defecto]
        ax.scatter(X_defecto, Y_defecto, c="red", s=5, alpha=0.5, label="Defecto")

        # Configuración
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Malla Polar 2D: {self.nr}×{self.ntheta} nodos")
        ax.legend()
        ax.grid(True, alpha=0.2)

        if mostrar:
            plt.show()

        return fig, ax

    def visualizar_regiones(
        self, mostrar: bool = True, figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualiza las regiones activa y defectuosa.

        Parameters
        ----------
        mostrar : bool, optional
            Si True, llama a plt.show()
        figsize : Tuple[int, int], optional
            Tamaño de la figura

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> fig, ax = malla.visualizar_regiones(mostrar=False)
        """
        # Convertir a cartesianas
        X, Y = self.obtener_coordenadas_cartesianas()

        # Máscaras
        mascara_defecto = self.identificar_region_defecto()
        mascara_activa = self.identificar_region_activa()

        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Subplot 1: Vista polar
        ax1 = plt.subplot(1, 2, 1, projection="polar")
        contour = ax1.contourf(
            self.THETA_grid.T,
            self.R_grid.T,
            mascara_defecto.T.astype(float),
            levels=[0, 0.5, 1],
            colors=["green", "red"],
            alpha=0.6,
        )
        ax1.set_title("Vista Polar\n(Verde=Activa, Rojo=Defecto)")

        # Subplot 2: Vista cartesiana
        ax2.contourf(
            X,
            Y,
            mascara_defecto.astype(float),
            levels=[0, 0.5, 1],
            colors=["green", "red"],
            alpha=0.6,
        )
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title("Vista Cartesiana\n(Verde=Activa, Rojo=Defecto)")
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()

        if mostrar:
            plt.show()

        return fig, (ax1, ax2)

    # ========================================================================
    # INFORMACIÓN Y REPRESENTACIÓN
    # ========================================================================

    def obtener_info(self) -> Dict:
        """
        Obtiene información resumida de la malla.

        Returns
        -------
        info : Dict
            Diccionario con información de la malla

        Examples
        --------
        >>> info = malla.obtener_info()
        >>> print(info['fraccion_defecto'])
        """
        return {
            "nr": self.nr,
            "ntheta": self.ntheta,
            "dr": self.dr,
            "dtheta": self.dtheta,
            "R": self.R,
            "area_total": self.calcular_area_total(),
            "area_defecto": self.calcular_area_defecto(),
            "fraccion_defecto": self.calcular_fraccion_defecto(),
            "total_nodos": self.nr * self.ntheta,
            "nodos_defecto": self.identificar_region_defecto().sum(),
            "nodos_activa": self.identificar_region_activa().sum(),
        }

    def __str__(self) -> str:
        """Representación en string legible."""
        info = self.obtener_info()
        return (
            f"MallaPolar2D(\n"
            f"  Nodos: {info['nr']}×{info['ntheta']} = {info['total_nodos']}\n"
            f"  Paso: Δr={info['dr']:.3e}m, Δθ={info['dtheta']:.4f}rad\n"
            f"  Radio: R={info['R']:.3f}m\n"
            f"  Área total: {info['area_total']:.6e}m²\n"
            f"  Defecto: {info['fraccion_defecto']:.1%} del área\n"
            f"  Nodos defecto: {info['nodos_defecto']}/{info['total_nodos']}\n"
            f")"
        )


# ============================================================================
# FIN DEL MÓDULO
# ============================================================================
