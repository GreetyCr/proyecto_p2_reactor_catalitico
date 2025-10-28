# Proyecto Personal 2: Simulación de Transferencia de Masa en Reactor Catalítico
## Fenómenos de Transferencia - UCR

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

---

## 📋 Descripción del Proyecto

Simulación numérica 2D de **difusión-reacción de CO** en un pellet catalítico cilíndrico poroso con defecto interno, usando el método de **Crank-Nicolson** en coordenadas polares.

### Características del Sistema
- **Geometría**: Sección transversal circular (coordenadas polares)
- **Método numérico**: Crank-Nicolson (implícito, segundo orden)
- **Defecto**: Región sin reacción en r∈[R/3, 2R/3], θ∈[0°, 45°]
- **Catalizador**: Pt/Al₂O₃
- **Temperatura**: 673 K (isotérmico)
- **Validación**: Balance de masa < 1%, tests unitarios, validación dimensional

---

## 🎯 Objetivos

1. ✅ Resolver C(r,θ,t) desde condición inicial hasta estado estacionario
2. ✅ Analizar efecto del defecto en perfiles de concentración
3. ✅ Calcular módulo de Thiele y efectividad del pellet
4. ✅ Generar visualizaciones 2D/3D de alta calidad
5. ✅ Validar resultados contra literatura

---

## 📁 Estructura del Proyecto

```
proyecto_p2_reactor_catalitico/
├── src/                        # Código fuente
│   ├── config/                 # Parámetros del sistema
│   ├── geometria/              # Generación de malla polar
│   ├── propiedades/            # Propiedades físicas (difusión, cinética)
│   ├── solver/                 # Solver Crank-Nicolson
│   ├── postproceso/            # Visualización y análisis
│   └── utils/                  # Utilidades (validación dimensional, logging)
├── tests/                      # Tests unitarios y de integración
├── notebooks/                  # Jupyter notebooks de análisis
├── data/                       # Datos de entrada/salida
├── docs/                       # Documentación
├── logs/                       # Logs de desarrollo y decisiones
├── .cursor/rules/              # Reglas de desarrollo para IA
├── requirements.txt            # Dependencias del proyecto
├── ENUNCIADO_RESUMIDO.md       # Enunciado resumido del proyecto
├── PARAMETROS_PROYECTO.md      # Tabla maestra de parámetros
└── main.py                     # Script principal de ejecución
```

---

## 🚀 Instalación y Setup

### 1. Clonar el repositorio
```bash
git clone [URL_DEL_REPO]
cd proyecto_p2_reactor_catalitico
```

### 2. Crear entorno virtual
```bash
python3.9 -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python -c "import numpy; import scipy; import matplotlib; print('✅ OK')"
pytest tests/ -v  # Ejecutar tests (cuando estén disponibles)
```

---

## 🔧 Uso Básico

### Ejecutar simulación completa
```bash
python main.py
```

### Ejecutar con configuración personalizada
```bash
python main.py --config config/custom.yaml
```

### Ejecutar tests
```bash
# Todos los tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov-report=html

# Test específico
pytest tests/test_solver.py::test_balance_masa -v
```

### Formatear código
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

---

## 📊 Parámetros Clave

| Parámetro | Valor | Unidad | Descripción |
|-----------|-------|--------|-------------|
| D_eff | 1.04×10⁻⁶ | m²/s | Difusividad efectiva |
| k_app | 4.0×10⁻³ | s⁻¹ | Constante cinética (activo) |
| k_c | 0.170 | m/s | Coef. transferencia externa |
| C_bulk | 0.0145 | mol/m³ | Concentración bulk (800 ppm CO) |
| T | 673 | K | Temperatura de operación |
| R | 0.002 | m | Radio del pellet |
| φ (Thiele) | 0.124 | - | Módulo de Thiele |

Ver [`PARAMETROS_PROYECTO.md`](PARAMETROS_PROYECTO.md) para tabla completa.

---

## 📈 Resultados Esperados

### Gráficos Obligatorios (Sección 1.5)
1. Perfil de concentración en **t = 0** (condición inicial)
2. Perfil de concentración en **t = t_ss/2** (50% hacia estado estacionario)
3. Perfil de concentración en **t = t_ss** (estado estacionario)

### Análisis
- Efecto del defecto en distribución de concentración
- Tiempo característico al estado estacionario
- Efectividad del pellet (η)
- Validación de balance de masa

---

## 🧪 Testing

Este proyecto sigue **Test-Driven Development (TDD)**:

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests con coverage (objetivo: >70%)
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Tests específicos por módulo
pytest tests/test_geometria.py -v
pytest tests/test_propiedades.py -v
pytest tests/test_solver.py -v
```

### Validaciones Obligatorias
- ✅ Validación dimensional de todas las ecuaciones
- ✅ Balance de masa < 1% error
- ✅ Tests unitarios de cada función
- ✅ Tests de integración del solver completo

---

## 📚 Documentación

- [`ENUNCIADO_RESUMIDO.md`](ENUNCIADO_RESUMIDO.md): Descripción completa del problema
- [`PARAMETROS_PROYECTO.md`](PARAMETROS_PROYECTO.md): Tabla maestra de parámetros
- [`.cursor/rules/`](.cursor/rules/): Reglas de desarrollo y calidad
- [`docs/`](docs/): Documentación técnica (metodología, referencias)
- [`logs/`](logs/): Logs de desarrollo y decisiones técnicas

---

## 🔬 Metodología

### Ecuación Gobernante
```
∂C/∂t = D_eff [∂²C/∂r² + (1/r)∂C/∂r + (1/r²)∂²C/∂θ²] - k_app × C
```

### Método Numérico
- **Crank-Nicolson**: Implícito de segundo orden en tiempo y espacio
- **Estabilidad**: Incondicionalmente estable
- **Malla**: 61 nodos radiales × 96 nodos angulares
- **Paso temporal**: Δt = 0.001 s (ajustable)

### Condiciones de Frontera
- r = 0: Simetría (∂C/∂r = 0)
- r = R: Robin/Convectiva (-D_eff·∂C/∂r = k_c·(C_s - C_bulk))
- θ = 0, 2π: Periodicidad (C(r,0) = C(r,2π))

---

## 🤖 Uso de IA en el Desarrollo

Este proyecto utiliza **Cursor AI** como asistente de desarrollo. Todas las interacciones están documentadas en:
- [`logs/work_sessions/`](logs/work_sessions/): Sesiones de trabajo con IA
- [`logs/decisions/`](logs/decisions/): Decisiones técnicas documentadas
- [`.cursor/rules/`](.cursor/rules/): Reglas de calidad y desarrollo

**Nota**: Según el enunciado, el uso de herramientas de IA debe ser documentado en la metodología.

---

## 🛠️ Stack Tecnológico

### Core Científico
- **Python 3.9+**: Lenguaje base
- **NumPy**: Arrays y cálculos numéricos
- **SciPy**: Resolución de sistemas lineales dispersos
- **Matplotlib**: Visualización 2D/3D
- **Plotly**: Visualización interactiva

### Testing y Calidad
- **pytest**: Framework de testing
- **pytest-cov**: Cobertura de código
- **black**: Formateo automático
- **flake8**: Linting
- **mypy**: Type checking

### Optimización
- **Numba**: JIT compilation para loops críticos
- **scipy.sparse**: Matrices dispersas eficientes

---

## 📖 Referencias

### Papers Clave
- Wakao & Funazkri (1978) - Correlación de Sherwood
- Dixon (1988) - Porosidad en lechos empacados
- Thiele (1939) - Módulo de Thiele
- Weisz & Prater (1954) - Criterio de limitaciones internas
- Mourkou et al. (2024) - Difusión de Knudsen en pellets

### Libros
- Crank, J., & Nicolson, P. (1947) - Método numérico
- LeVeque, R. J. (2007) - Finite Difference Methods

---

## 👥 Autor

**Adrián Vargas Tijerino**  
Carné: C18332  
Curso: Fenómenos de Transferencia  
Universidad de Costa Rica

---

## 📄 Licencia

Este proyecto es parte de un proyecto personal académico para el curso de Fenómenos de Transferencia de la Universidad de Costa Rica.

---

## 🚧 Estado del Proyecto

**Fecha de inicio**: 2025-10-28  
**Estado actual**: 🟡 En desarrollo  

### Progreso
- [x] Setup inicial del proyecto
- [x] Estructura de carpetas
- [x] Documentación de parámetros
- [ ] Implementación del solver
- [ ] Validación y testing
- [ ] Visualización
- [ ] Análisis de resultados
- [ ] Reporte final

---

## 📞 Contacto

Para dudas o comentarios sobre este proyecto, contactar a:
- Email: [tu_email@ucr.ac.cr]
- GitHub: [tu_usuario]

---

**Última actualización**: 2025-10-28

