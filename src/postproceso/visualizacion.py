"""
Módulo de visualización para resultados del Proyecto 2.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-29

Implementa los 3 gráficos obligatorios de la sección 1.5:
1. Perfil de concentración en t=0
2. Perfil al 50% del tiempo para estado estacionario
3. Perfil en estado estacionario

Basado en: .cursor/rules/guia-visual.mdc
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def configurar_estilo_matplotlib(dpi: int = 150):
    """
    Configura estilo global de matplotlib.

    Parameters
    ----------
    dpi : int, optional
        DPI para figuras (150 para desarrollo, 300 para reporte)
    """
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': dpi,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })


def _dibujar_defecto_anillo(ax, params, color='r', linewidth=2, label='Defecto'):
    """
    Dibuja los 4 bordes del defecto (anillo) en un plot polar.
    
    El defecto forma un "rectángulo polar" (sector anular) con:
    - 2 líneas radiales en θ=θ1 y θ=θ2
    - 2 arcos circulares en r=r1 y r=r2
    
    Parameters
    ----------
    ax : plt.Axes (polar projection)
        Axes de matplotlib con proyección polar
    params : ParametrosMaestros
        Parámetros del sistema
    color : str, optional
        Color de las líneas (default: 'r')
    linewidth : float, optional
        Grosor de las líneas (default: 2)
    label : str, optional
        Label para la leyenda (default: 'Defecto')
    """
    r1 = params.geometria.r1  # R/3
    r2 = params.geometria.r2  # 2R/3
    theta1 = params.geometria.theta1  # 0°
    theta2 = params.geometria.theta2  # 45°
    
    # 1. Línea radial en θ=θ1 (r1 → r2)
    ax.plot([theta1, theta1], [r1, r2], 
            color=color, linestyle='--', linewidth=linewidth, label=label)
    
    # 2. Línea radial en θ=θ2 (r1 → r2)
    ax.plot([theta2, theta2], [r1, r2], 
            color=color, linestyle='--', linewidth=linewidth)
    
    # 3. Arco circular en r=r1 (θ1 → θ2)
    theta_arc = np.linspace(theta1, theta2, 50)
    r_arc_inner = np.full_like(theta_arc, r1)
    ax.plot(theta_arc, r_arc_inner, 
            color=color, linestyle='--', linewidth=linewidth)
    
    # 4. Arco circular en r=r2 (θ1 → θ2)
    r_arc_outer = np.full_like(theta_arc, r2)
    ax.plot(theta_arc, r_arc_outer, 
            color=color, linestyle='--', linewidth=linewidth)


def plot_perfil_t0(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    params,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Grafica perfil de concentración en t=0 (condición inicial).

    Gráfico 1 Obligatorio - Sección 1.5.1

    Parameters
    ----------
    r : np.ndarray
        Array radial [m]
    theta : np.ndarray
        Array angular [rad]
    C : np.ndarray, shape (nr, ntheta)
        Campo de concentración [mol/m³]
    params : ParametrosMaestros
        Parámetros del sistema
    figsize : Tuple[int, int], optional
        Tamaño de figura

    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    ax : plt.Axes
        Axes de matplotlib
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)

    # Mesh grid
    R_grid, THETA_grid = np.meshgrid(r, theta)

    # Contour plot
    C_min = C.min()
    C_max = C.max()
    
    # Si C_min == C_max, añadir pequeño rango para evitar error
    if C_max - C_min < 1e-12:
        C_max = C_min + 1e-6
    
    levels = np.linspace(C_min, C_max, 20)
    contour = ax.contourf(THETA_grid, R_grid, C.T, levels=levels, cmap='viridis')

    # Marcar región del defecto (anillo con arcos y líneas radiales)
    _dibujar_defecto_anillo(ax, params, color='r', linewidth=2, label='Defecto')

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, pad=0.1)
    cbar.set_label('Concentración CO [mol/m³]', rotation=270, labelpad=20)

    # Etiquetas
    ax.set_title('Concentración CO en t=0 s (Condición Inicial)', 
                 va='bottom', fontsize=14, pad=20)
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig, ax


def plot_perfil_50pct(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    tiempo: float,
    params,
    figsize: Tuple[int, int] = (16, 7)
) -> Tuple[plt.Figure, Tuple[plt.Axes, Axes3D]]:
    """
    Grafica perfil de concentración al 50% del tiempo hacia estado estacionario.

    Gráfico 2 Obligatorio - Sección 1.5.1
    Incluye vista 2D polar y vista 3D cartesiana.

    Parameters
    ----------
    r : np.ndarray
        Array radial [m]
    theta : np.ndarray
        Array angular [rad]
    C : np.ndarray, shape (nr, ntheta)
        Campo de concentración [mol/m³]
    tiempo : float
        Tiempo actual de simulación [s]
    params : ParametrosMaestros
        Parámetros del sistema
    figsize : Tuple[int, int], optional
        Tamaño de figura

    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    (ax1, ax2) : Tuple[plt.Axes, Axes3D]
        Axes 2D polar y 3D
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Vista 2D polar
    ax1 = fig.add_subplot(121, projection='polar')

    R_grid, THETA_grid = np.meshgrid(r, theta)

    levels = np.linspace(0, C.max(), 25)
    contour = ax1.contourf(THETA_grid, R_grid, C.T, levels=levels, cmap='hot')

    # Marcar defecto (anillo con arcos y líneas radiales)
    _dibujar_defecto_anillo(ax1, params, color='c', linewidth=2, label='Defecto')

    ax1.set_title(f'Vista 2D Polar\nt = {tiempo:.3f} s', fontsize=12)
    ax1.legend()

    # Subplot 2: Vista 3D cartesiana
    ax2 = fig.add_subplot(122, projection='3d')

    # Convertir a coordenadas cartesianas
    X = r * np.cos(theta[:, None])
    Y = r * np.sin(theta[:, None])
    Z = C.T

    surf = ax2.plot_surface(X, Y, Z, cmap='hot', edgecolor='none', alpha=0.9)

    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('C [mol/m³]')
    ax2.set_title(f'Vista 3D\nt = {tiempo:.3f} s', fontsize=12)
    ax2.view_init(elev=30, azim=45)

    # Colorbar común
    fig.colorbar(surf, ax=[ax1, ax2], orientation='horizontal', 
                 pad=0.05, label='Concentración CO [mol/m³]')

    plt.suptitle(f'Concentración CO en t={tiempo:.2f} s (50% hacia Estado Estacionario)',
                 fontsize=14, y=0.98)

    # No usar tight_layout con mezcla de ejes polares y 3D (causa warning)
    # plt.tight_layout()  # Comentado para evitar UserWarning
    return fig, (ax1, ax2)


def plot_perfil_estado_estacionario(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    tiempo: float,
    params,
    figsize: Tuple[int, int] = (18, 6)
) -> Tuple[plt.Figure, Tuple[plt.Axes, Axes3D, plt.Axes]]:
    """
    Grafica perfil de concentración en estado estacionario.

    Gráfico 3 Obligatorio - Sección 1.5.1
    Incluye vista 2D polar, vista 3D, y perfiles radiales comparativos.

    Parameters
    ----------
    r : np.ndarray
        Array radial [m]
    theta : np.ndarray
        Array angular [rad]
    C : np.ndarray, shape (nr, ntheta)
        Campo de concentración [mol/m³]
    tiempo : float
        Tiempo actual de simulación [s]
    params : ParametrosMaestros
        Parámetros del sistema
    figsize : Tuple[int, int], optional
        Tamaño de figura

    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    (ax1, ax2, ax3) : Tuple[plt.Axes, Axes3D, plt.Axes]
        Axes 2D polar, 3D, y perfiles radiales
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Vista 2D polar
    ax1 = fig.add_subplot(131, projection='polar')

    R_grid, THETA_grid = np.meshgrid(r, theta)
    levels = np.linspace(0, params.operacion.C_bulk, 25)
    contour1 = ax1.contourf(THETA_grid, R_grid, C.T, levels=levels, cmap='plasma')

    # Marcar defecto (anillo con arcos y líneas radiales)
    _dibujar_defecto_anillo(ax1, params, color='red', linewidth=2, label='Defecto')

    ax1.set_title('Vista 2D Polar\n(Estado Estacionario)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)

    plt.colorbar(contour1, ax=ax1, pad=0.1, label='C [mol/m³]')

    # Subplot 2: Vista 3D
    ax2 = fig.add_subplot(132, projection='3d')

    X = r * np.cos(theta[:, None])
    Y = r * np.sin(theta[:, None])
    Z = C.T

    surf = ax2.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.8)

    ax2.set_xlabel('x [m]', fontsize=9)
    ax2.set_ylabel('y [m]', fontsize=9)
    ax2.set_zlabel('C [mol/m³]', fontsize=9)
    ax2.set_title('Vista 3D\n(Estado Estacionario)', fontsize=11)
    ax2.view_init(elev=25, azim=45)

    # Subplot 3: Perfiles radiales comparativos
    ax3 = fig.add_subplot(133)

    # Encontrar índices de θ=0° (defecto) y θ=90° (activo)
    idx_0deg = np.argmin(np.abs(theta - 0))
    idx_90deg = np.argmin(np.abs(theta - np.pi/2))

    # Perfiles radiales
    ax3.plot(r*1000, C[:, idx_0deg], 'r-', linewidth=2, 
             label=f'θ=0° (Defecto)', marker='o', markersize=4)
    ax3.plot(r*1000, C[:, idx_90deg], 'b-', linewidth=2, 
             label=f'θ=90° (Activo)', marker='s', markersize=4)

    # Marcar región del defecto en el eje x (todo el radio)
    # Ya no aplica un rango radial específico, pero mantengo leyenda
    # ax3.axvspan(0, R*1000, alpha=0.1, color='red', label='Sector defectuoso')

    ax3.set_xlabel('Radio [mm]', fontsize=10)
    ax3.set_ylabel('Concentración CO [mol/m³]', fontsize=10)
    ax3.set_title('Comparación Perfiles Radiales\n(Defecto vs Activo)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'Concentración CO en Estado Estacionario (t={tiempo:.2f} s)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def guardar_figura(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 300
):
    """
    Guarda figura en alta resolución.

    Parameters
    ----------
    fig : plt.Figure
        Figura a guardar
    filepath : str
        Ruta del archivo de salida
    dpi : int, optional
        Resolución (default: 300 para reporte)
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white'
    )
    
    logger.info(f"Figura guardada: {output_path}")

