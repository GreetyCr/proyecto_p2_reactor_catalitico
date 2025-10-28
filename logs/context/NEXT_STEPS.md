# PRÓXIMOS PASOS
**Última actualización**: 2025-10-28 (Setup Inicial)

---

## 🎯 INMEDIATO (Próxima sesión de trabajo)

### 1. Crear Entorno Virtual e Instalar Dependencias 🔴
**Prioridad**: CRÍTICA  
**Tiempo estimado**: 15 minutos  
**Responsable**: Adrián

**Comandos a ejecutar**:
```bash
# 1. Crear entorno virtual
python3.9 -m venv venv

# 2. Activar entorno
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Actualizar pip
pip install --upgrade pip

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Verificar instalación
python -c "import numpy; import scipy; import matplotlib; print('✅ OK')"
pytest --version
black --version
flake8 --version
```

**Criterio de éxito**: Todas las librerías instaladas sin errores

---

### 2. Crear Tabla Maestra de Parámetros 🟡
**Prioridad**: ALTA  
**Tiempo estimado**: 2-3 horas  
**Responsable**: Adrián + Claude

**Tareas**:
- [ ] Crear `src/config/parametros.py`
- [ ] Transcribir **TODOS** los parámetros de `PARAMETROS_PROYECTO.md`
- [ ] Usar `dataclasses` para estructura
- [ ] Implementar validación de consistencia dimensional
- [ ] Agregar docstrings completos con unidades

**Estructura sugerida**:
```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class GeometriaParams:
    """Parámetros geométricos del pellet (Tabla I)."""
    D: float = 0.004  # m - Diámetro
    R: float = 0.002  # m - Radio
    r1: float = 0.000667  # m - Radio interno defecto
    r2: float = 0.001333  # m - Radio externo defecto
    theta1: float = 0.0  # rad
    theta2: float = 0.7854  # rad (45°)
    
    def validar(self):
        """Valida consistencia de parámetros."""
        assert self.R == self.D / 2
        assert self.r1 < self.r2 < self.R

@dataclass
class OperacionParams:
    """Condiciones de operación (Tabla II)."""
    T: float = 673  # K
    P: float = 101325  # Pa
    # ... etc

# TODO: Completar con todas las tablas
```

**Tests a crear** (TDD):
```python
# tests/test_parametros.py
def test_geometria_consistente():
    """Radio debe ser mitad del diámetro."""
    params = GeometriaParams()
    assert params.R == params.D / 2

def test_defecto_dentro_pellet():
    """Defecto debe estar dentro del pellet."""
    params = GeometriaParams()
    assert 0 < params.r1 < params.r2 < params.R
```

**Criterio de éxito**: 
- Todas las 10 tablas transcritas
- Tests pasando
- Validación dimensional implementada

---

## 📅 CORTO PLAZO (Esta semana)

### 3. Sistema de Validación Dimensional 📐
**Prioridad**: ALTA  
**Tiempo estimado**: 3-4 horas  
**Responsable**: Claude

**Tareas**:
- [ ] Crear `src/utils/validacion.py`
- [ ] Implementar clase `CantidadDimensional`
- [ ] Implementar decorador `@validar_dimensiones`
- [ ] Crear tests de validación

**Código de referencia**: Ver `.cursor/rules/03_REGLAS_CALIDAD_DESARROLLO.md` líneas 48-174

**Tests a crear**:
```python
# tests/test_validacion.py
def test_cantidad_dimensional_multiplicacion():
    """Multiplicación de cantidades debe combinar dimensiones."""
    # ...

def test_validar_ecuacion_difusion():
    """Ecuación de difusión debe ser dimensionalmente correcta."""
    # ...
```

**Criterio de éxito**: 
- Sistema de unidades funcional
- Decorador aplicable a funciones
- Tests de validación dimensional pasando

---

### 4. Módulo de Geometría y Mallado 🗺️
**Prioridad**: ALTA  
**Tiempo estimado**: 4-5 horas  
**Responsable**: Adrián + Claude

**Tareas**:
- [ ] Crear `src/geometria/dominio.py`
  - Definir región activa
  - Definir región de defecto
  - Función para verificar si punto (r,θ) está en defecto

- [ ] Crear `src/geometria/mallado.py`
  - Generar arrays radiales: `r = np.linspace(0, R, nr)`
  - Generar arrays angulares: `θ = np.linspace(0, 2π, ntheta)`
  - Crear meshgrid 2D
  - Identificar índices de nodos en defecto

**Estructura sugerida**:
```python
# src/geometria/mallado.py
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class Malla2DPolar:
    """Malla 2D en coordenadas polares."""
    r: np.ndarray  # Coordenadas radiales [m]
    theta: np.ndarray  # Coordenadas angulares [rad]
    nr: int
    ntheta: int
    dr: float
    dtheta: float
    
    def esta_en_defecto(self, i: int, j: int, params) -> bool:
        """Verifica si nodo (i,j) está en región defecto."""
        r_i = self.r[i]
        theta_j = self.theta[j]
        
        return (params.r1 <= r_i <= params.r2 and 
                params.theta1 <= theta_j <= params.theta2)

def generar_malla_polar(nr: int, ntheta: int, R: float) -> Malla2DPolar:
    """Genera malla polar uniforme."""
    r = np.linspace(0, R, nr)
    theta = np.linspace(0, 2*np.pi, ntheta)
    
    dr = r[1] - r[0] if nr > 1 else 0
    dtheta = theta[1] - theta[0] if ntheta > 1 else 0
    
    return Malla2DPolar(r, theta, nr, ntheta, dr, dtheta)
```

**Tests a crear** (TDD):
```python
# tests/test_geometria.py
def test_malla_simetrica():
    """Malla debe ser simétrica en θ."""
    # ...

def test_defecto_identificado_correctamente():
    """Nodos en región de defecto deben ser identificados."""
    # ...

def test_malla_espaciado_uniforme():
    """Espaciado debe ser uniforme."""
    # ...
```

**Criterio de éxito**:
- Malla generada correctamente
- Región de defecto identificada
- Tests pasando

---

## 📆 MEDIO PLAZO (Próxima semana)

### 5. Módulo de Propiedades Físicas 🔬
**Prioridad**: ALTA  
**Tiempo estimado**: 6-8 horas  
**Responsable**: Adrián + Claude

**Submódulos a crear**:

#### 5.1. Difusión (`src/propiedades/difusion.py`)
```python
def calcular_difusividad_knudsen(r_poro, T, M_CO) -> float:
    """D_K = (2/3) × r_poro × √(8RT/πM)"""
    # Implementar con validación dimensional

def calcular_difusividad_combinada(D_molecular, D_knudsen) -> float:
    """1/D_comb = 1/D_molecular + 1/D_knudsen"""
    # Bosanquet

def calcular_difusividad_efectiva(epsilon, D_comb, tau) -> float:
    """D_eff = ε × D_comb / τ"""
    # Validar rangos físicos
```

**Tests (TDD)**:
- `test_difusividad_knudsen_valores_esperados()` 
- `test_difusividad_knudsen_validacion_inputs()`
- Ver ejemplo completo en `.cursor/rules/03_REGLAS_CALIDAD_DESARROLLO.md`

#### 5.2. Cinética (`src/propiedades/cinetica.py`)
```python
def calcular_k_app(k0, Ea, T) -> float:
    """k_app = k₀ × exp(-Ea/RT)"""
    # Arrhenius

def obtener_k_app_por_region(i, j, malla, params) -> float:
    """Retorna k_app según si está en defecto o activo."""
    if malla.esta_en_defecto(i, j, params):
        return 0.0  # Defecto: sin reacción
    else:
        return params.k_app  # Activo
```

#### 5.3. Gas (`src/propiedades/gas.py`)
```python
def calcular_densidad_gas(P, T, M) -> float:
    """ρ = PM/(RT)"""
    
def calcular_viscosidad_sutherland(T, T0, mu0, S) -> float:
    """Viscosidad con ley de Sutherland."""
```

**Criterio de éxito**:
- Todos los cálculos de Tablas III-VI implementados
- Tests unitarios pasando
- Validación dimensional en todas las funciones

---

### 6. Solver Crank-Nicolson (CRÍTICO) 🧮
**Prioridad**: CRÍTICA  
**Tiempo estimado**: 10-15 horas  
**Responsable**: Adrián + Claude

**Esta es la tarea más compleja y crítica del proyecto.**

#### 6.1. Ensamblaje de Matrices (`src/solver/matrices.py`)
```python
def construir_matriz_laplaciano_2d_polar(
    malla: Malla2DPolar,
    D_eff: float,
    dt: float
) -> sparse.csr_matrix:
    """
    Construye operador Laplaciano discreto.
    
    Tratamiento especial en r=0 (singularidad).
    """
    # Usar scipy.sparse.lil_matrix para construcción
    # Convertir a CSR para operaciones
```

**Puntos críticos**:
- ⚠️ Singularidad en r=0: usar L'Hôpital
- ⚠️ Matriz debe ser simétrica (verificar)
- ⚠️ Debe ser invertible (det ≠ 0)

#### 6.2. Condiciones de Frontera (`src/solver/condiciones_frontera.py`)
```python
def aplicar_simetria_centro(A, b):
    """Aplica ∂C/∂r = 0 en r=0."""

def aplicar_robin_superficie(A, b, k_c, C_bulk, D_eff):
    """Aplica -D_eff∂C/∂r = k_c(C_s - C_bulk) en r=R."""

def aplicar_periodicidad_angular(A, b):
    """Aplica C(r,0) = C(r,2π)."""
```

#### 6.3. Solver Principal (`src/solver/crank_nicolson.py`)
```python
class CrankNicolsonSolver2D:
    def __init__(self, malla, params):
        # ...
    
    def ensamblar_sistema(self):
        """Ensambla matrices A y B."""
    
    def paso_temporal(self, C_n, dt):
        """Ejecuta un paso temporal: A·C^(n+1) = B·C^n + b."""
        # Resolver con scipy.sparse.linalg.spsolve
    
    def ejecutar(self, t_final, dt):
        """Ejecuta simulación completa."""
        # Loop temporal con verificación de convergencia
```

**Tests críticos**:
- `test_matriz_no_singular()`
- `test_simetria_centro()`
- `test_balance_masa_conservado()`
- `test_convergencia_estado_estacionario()`

**Criterio de éxito**:
- Solver converge a estado estacionario
- Balance de masa < 1% error
- Tests de integración pasando

---

## 🎓 LARGO PLAZO (Semanas 3-4)

### 7. Visualización 📊
- Implementar 3 gráficos obligatorios
- Animaciones temporales
- Ver `.cursor/rules/05_GUIA_VISUALIZACION_P2.md`

### 8. Post-Procesamiento 📈
- Módulo de Thiele
- Efectividad η
- Criterio Weisz-Prater

### 9. Documentación Final 📚
- Completar docstrings
- Sphinx documentation
- README actualizado
- Notebooks de análisis

---

## ✅ CHECKLIST DE VALIDACIÓN CONTINUA

En cada paso, verificar:
- [ ] Tests escritos ANTES de código (TDD)
- [ ] Validación dimensional pasa
- [ ] Balance de masa < 1% (cuando aplique)
- [ ] Linting sin errores (flake8)
- [ ] Type hints completos
- [ ] Docstrings completos
- [ ] Git commit con mensaje descriptivo
- [ ] Log de trabajo actualizado

---

## 🚨 BLOQUEADORES POTENCIALES

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Singularidad en r=0 | 🔴 Alto | Usar L'Hôpital, tests específicos |
| Matriz singular | 🔴 Alto | Verificar simetría, condicionamiento |
| Balance de masa > 1% | 🟡 Medio | Debugging, ajustar dt |
| Convergencia lenta | 🟢 Bajo | Adaptive timestepping |

---

**Próxima actualización**: Después de instalar dependencias  
**Última edición por**: Claude AI (2025-10-28)

