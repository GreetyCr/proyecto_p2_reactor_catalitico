#!/usr/bin/env python3
"""
Script de demostración del módulo de mallado polar 2D.

Proyecto Personal 2 - Fenómenos de Transferencia
Autor: Adrián Vargas Tijerino (C18332)
Fecha: 2025-10-28

Este script demuestra las capacidades del módulo de mallado:
1. Crear malla polar 2D
2. Identificar regiones activa/defecto
3. Visualizar la malla y regiones
4. Calcular propiedades geométricas
"""

import sys
from pathlib import Path

# Agregar src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from config.parametros import ParametrosMaestros
from geometria.mallado import MallaPolar2D
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Función principal de demostración."""
    logger.info("=" * 70)
    logger.info("DEMOSTRACIÓN: MALLADO POLAR 2D")
    logger.info("=" * 70)

    # Cargar parámetros
    logger.info("Cargando parámetros del proyecto...")
    params = ParametrosMaestros()
    params.validar_todo()
    logger.info("✓ Parámetros validados")

    # Crear malla
    logger.info("\nCreando malla polar 2D...")
    malla = MallaPolar2D(params)
    logger.info("✓ Malla creada")

    # Mostrar información
    logger.info("\n" + "=" * 70)
    logger.info("INFORMACIÓN DE LA MALLA")
    logger.info("=" * 70)
    print(malla)

    # Mostrar estadísticas
    logger.info("\n" + "=" * 70)
    logger.info("ESTADÍSTICAS GEOMÉTRICAS")
    logger.info("=" * 70)

    info = malla.obtener_info()
    logger.info(f"Total de nodos: {info['total_nodos']}")
    logger.info(f"Nodos en región activa: {info['nodos_activa']}")
    logger.info(f"Nodos en región defecto: {info['nodos_defecto']}")
    logger.info(f"Fracción defectuosa: {info['fraccion_defecto']:.2%}")
    logger.info(f"Área total: {info['area_total']:.6e} m²")
    logger.info(f"Área defecto: {info['area_defecto']:.6e} m²")

    # Generar campo de k_app
    logger.info("\n" + "=" * 70)
    logger.info("CAMPO DE CONSTANTE CINÉTICA")
    logger.info("=" * 70)

    k_app_field = malla.generar_campo_k_app()
    logger.info(f"k_app en región activa: {params.cinetica.k_app:.3e} s⁻¹")
    logger.info(f"k_app en región defecto: 0.0 s⁻¹")
    logger.info(f"Valores únicos en campo: {np.unique(k_app_field)}")

    # Visualización
    logger.info("\n" + "=" * 70)
    logger.info("GENERANDO VISUALIZACIONES")
    logger.info("=" * 70)

    try:
        import matplotlib

        matplotlib.use("TkAgg")  # Backend interactivo

        logger.info("Visualizando malla...")
        fig1, ax1 = malla.visualizar_malla(mostrar=False)
        fig1.savefig("data/output/malla_polar_2d.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Guardado: data/output/malla_polar_2d.png")

        logger.info("Visualizando regiones...")
        fig2, ax2 = malla.visualizar_regiones(mostrar=False)
        fig2.savefig("data/output/regiones_activa_defecto.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Guardado: data/output/regiones_activa_defecto.png")

        logger.info("\n✅ Visualizaciones guardadas exitosamente")
        logger.info("   Ver: data/output/")

    except Exception as e:
        logger.warning(f"No se pudieron crear visualizaciones: {e}")

    # Resumen final
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    logger.info("✓ Malla polar 2D creada correctamente")
    logger.info("✓ Regiones activa/defecto identificadas")
    logger.info("✓ Propiedades geométricas calculadas")
    logger.info("✓ Visualizaciones generadas")
    logger.info("")
    logger.info("Próximo paso: Implementar propiedades físicas (difusión, cinética)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

