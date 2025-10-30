"""
Visualización mejorada con escalas ajustadas para resaltar el efecto del defecto.

Versiones optimizadas de los gráficos 2 y 3 que usan colormaps ajustados
al rango real de concentraciones, no a [0, C_bulk].
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_perfil_50pct_mejorado(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    tiempo: float,
    params,
    figsize: Tuple[int, int] = (16, 7)
) -> Tuple[plt.Figure, Tuple[plt.Axes, Axes3D]]:
    """
    Grafica perfil de concentración al 50% con ESCALA AJUSTADA.

    Versión mejorada que usa el rango real de concentraciones
    en lugar de [0, C_bulk] para resaltar diferencias sutiles.

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
    """
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Vista 2D polar con escala ajustada
    ax1 = fig.add_subplot(121, projection='polar')

    R_grid, THETA_grid = np.meshgrid(r, theta)

    # CLAVE: Usar rango REAL de concentraciones, no [0, C_bulk]
    C_min = C.min()
    C_max = C.max()
    
    # Añadir pequeño margen para visualización
    margen = (C_max - C_min) * 0.05
    levels = np.linspace(C_min - margen, C_max + margen, 30)
    
    contour = ax1.contourf(THETA_grid, R_grid, C.T, levels=levels, 
                           cmap='hot', extend='both')

    # Marcar defecto
    from src.postproceso.visualizacion import _dibujar_defecto_anillo
    _dibujar_defecto_anillo(ax1, params, color='cyan', linewidth=2.5, label='Defecto')

    ax1.set_title(
        f'Vista 2D Polar (Escala Ajustada)\nt = {tiempo:.3f} s', 
        fontsize=12, fontweight='bold'
    )
    ax1.legend()

    # Colorbar para ax1
    cbar1 = plt.colorbar(contour, ax=ax1, pad=0.1)
    cbar1.set_label('C [mol/m³]', rotation=270, labelpad=20, fontsize=11)

    # Subplot 2: Vista 3D cartesiana con escala ajustada
    ax2 = fig.add_subplot(122, projection='3d')

    # Convertir a coordenadas cartesianas
    X = R_grid * np.cos(THETA_grid)
    Y = R_grid * np.sin(THETA_grid)

    # Surface plot con escala ajustada
    surf = ax2.plot_surface(
        X, Y, C.T, cmap='hot', edgecolor='none', 
        rstride=1, cstride=1, vmin=C_min, vmax=C_max
    )
    
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    cbar2.set_label('C [mol/m³]', rotation=270, labelpad=15, fontsize=11)

    ax2.set_xlabel('X [m]', fontsize=10)
    ax2.set_ylabel('Y [m]', fontsize=10)
    ax2.set_zlabel('Concentración CO [mol/m³]', fontsize=10)
    ax2.set_title(
        f'Vista 3D Cartesiana (Escala Ajustada)\nt = {tiempo:.3f} s', 
        fontsize=12, fontweight='bold'
    )
    ax2.view_init(elev=25, azim=45)

    plt.suptitle(
        f'Concentración CO al 50% - ESCALA AJUSTADA (t={tiempo:.2f} s)\n'
        f'Rango: [{C_min*1000:.3f}, {C_max*1000:.3f}] mmol/m³',
        fontsize=14, fontweight='bold', y=0.98
    )

    # No usar tight_layout con ejes mixtos
    return fig, (ax1, ax2)


def plot_perfil_ss_mejorado(
    r: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
    tiempo: float,
    params,
    figsize: Tuple[int, int] = (20, 7)
) -> Tuple[plt.Figure, Tuple[plt.Axes, Axes3D, plt.Axes]]:
    """
    Grafica perfil de concentración en estado estacionario con ESCALA AJUSTADA.

    Versión mejorada que usa tres subplots:
    1. Vista 2D polar con escala ajustada
    2. Vista 3D cartesiana con escala ajustada
    3. Comparación de perfiles radiales (defecto vs activo)

    Parameters
    ----------
    r : np.ndarray
        Array radial [m]
    theta : np.ndarray
        Array angular [rad]
    C : np.ndarray, shape (nr, ntheta)
        Campo de concentración [mol/m³]
    tiempo : float
        Tiempo de estado estacionario [s]
    params : ParametrosMaestros
        Parámetros del sistema
    figsize : Tuple[int, int], optional
        Tamaño de figura
    """
    fig = plt.figure(figsize=figsize)

    # Calcular rango real de concentraciones
    C_min = C.min()
    C_max = C.max()
    margen = (C_max - C_min) * 0.05

    # Subplot 1: Vista 2D polar
    ax1 = fig.add_subplot(131, projection='polar')

    R_grid, THETA_grid = np.meshgrid(r, theta)
    levels = np.linspace(C_min - margen, C_max + margen, 30)
    contour1 = ax1.contourf(THETA_grid, R_grid, C.T, levels=levels, 
                            cmap='plasma', extend='both')

    # Marcar defecto
    from src.postproceso.visualizacion import _dibujar_defecto_anillo
    _dibujar_defecto_anillo(ax1, params, color='red', linewidth=2, label='Defecto')

    ax1.set_title('Vista 2D Polar\n(Escala Ajustada)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    cbar1 = plt.colorbar(contour1, ax=ax1, pad=0.1)
    cbar1.set_label('C [mol/m³]', rotation=270, labelpad=15, fontsize=10)

    # Subplot 2: Vista 3D
    ax2 = fig.add_subplot(132, projection='3d')

    X = R_grid * np.cos(THETA_grid)
    Y = R_grid * np.sin(THETA_grid)

    surf = ax2.plot_surface(
        X, Y, C.T, cmap='plasma', edgecolor='none', 
        rstride=1, cstride=1, vmin=C_min, vmax=C_max
    )
    
    cbar2 = fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    cbar2.set_label('C [mol/m³]', rotation=270, labelpad=15, fontsize=10)

    ax2.set_xlabel('X [m]', fontsize=9)
    ax2.set_ylabel('Y [m]', fontsize=9)
    ax2.set_zlabel('C [mol/m³]', fontsize=9)
    ax2.set_title('Vista 3D Cartesiana\n(Escala Ajustada)', fontsize=11, fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    # Subplot 3: Perfiles radiales comparativos
    ax3 = fig.add_subplot(133)

    # Encontrar índices de θ=22.5° (defecto) y θ=90° (activo)
    idx_theta_defecto = np.argmin(np.abs(theta - np.pi/8))  # 22.5° (centro del defecto)
    idx_theta_activo = np.argmin(np.abs(theta - np.pi/2))   # 90° (lejos del defecto)

    # Perfiles radiales
    ax3.plot(r * 1000, C[:, idx_theta_defecto] * 1000, 
             'r-', linewidth=2.5, label=f'θ=22.5° (Defecto)', 
             marker='o', markersize=4, markevery=5)
    ax3.plot(r * 1000, C[:, idx_theta_activo] * 1000, 
             'b-', linewidth=2.5, label=f'θ=90° (Activo)', 
             marker='s', markersize=4, markevery=5)

    # Marcar región del defecto en el eje x
    r1_mm = params.geometria.r1 * 1000
    r2_mm = params.geometria.r2 * 1000
    ax3.axvspan(r1_mm, r2_mm, alpha=0.25, color='red', label='Zona defecto (radial)')

    ax3.set_xlabel('Radio [mm]', fontsize=11)
    ax3.set_ylabel('Concentración CO [mmol/m³]', fontsize=11)
    ax3.set_title('Comparación Perfiles Radiales\n(Defecto vs Activo)', 
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Calcular y mostrar diferencia promedio
    C_defecto = C[:, idx_theta_defecto]
    C_activo = C[:, idx_theta_activo]
    dif_promedio = np.mean(C_defecto - C_activo) * 1000  # mmol/m³
    ax3.text(
        0.02, 0.98, 
        f'Δ̄C = {dif_promedio:.3f} mmol/m³',
        transform=ax3.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle(
        f'Estado Estacionario (t={tiempo:.1f}s) - ESCALA AJUSTADA\n'
        f'Rango: [{C_min*1000:.3f}, {C_max*1000:.3f}] mmol/m³',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, (ax1, ax2, ax3)


def guardar_figura(fig: plt.Figure, filename: str):
    """
    Guarda una figura de matplotlib en alta resolución.

    Parameters
    ----------
    fig : plt.Figure
        Figura a guardar
    filename : str
        Ruta completa del archivo
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        filepath, 
        dpi=300, 
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    logger.info(f"Figura guardada: {filepath}")
    plt.close(fig)

