# ENUNCIADO RESUMIDO - PROYECTO PERSONAL 2
## Simulación de Transferencia de Masa en Reactor Catalítico

---

> **Documento para consulta rápida de Cursor/IA**  
> **Versión completa**: Ver `/mnt/project/Proyecto_Personal_2_*.pdf`  
> **Parámetros**: Ver `PARAMETROS_PROYECTO.md`

---

## 🎯 OBJETIVO DEL PROYECTO

Resolver numéricamente la **transferencia de masa no estacionaria de CO** en la sección transversal de un pellet catalítico cilíndrico poroso con un defecto interno, hasta alcanzar el estado estacionario.

---

## 📋 DESCRIPCIÓN DEL PROBLEMA FÍSICO

### Sistema
- **Pellet catalítico**: Cilindro poroso de Pt/Al₂O₃
- **Diámetro**: 4 mm
- **Gas**: Aire con 800 ppm CO + exceso O₂
- **Temperatura**: 400°C (673 K) - isotérmico
- **Presión**: 1 atm

### Defecto Interno
- **Ubicación radial**: r ∈ [R/3, 2R/3]
- **Ubicación angular**: θ ∈ [0°, 45°]
- **Característica**: NO hay reacción química (k_app = 0)
- **Resto del pellet**: Reacción catalítica activa

### Reacción Química
```
2 CO + O₂ → 2 CO₂
```
- Cinética aparente de **primer orden** en CO (exceso de O₂)
- Catalizador: Pt/Al₂O₃

### Fenómenos de Transporte
1. **Difusión externa**: Del gas al pellet (convección)
2. **Difusión interna**: Dentro del pellet poroso (Knudsen + molecular)
3. **Reacción**: En sitios activos del catalizador (primer orden)

---

## 📐 GEOMETRÍA DEL PROBLEMA

### Coordenadas
- **Sistema**: Polares 2D (r, θ)
- **Dominio**: Sección transversal circular del pellet
- **Justificación**: Longitud axial >> diámetro → invarianza axial

### Regiones
```
┌─────────────────────────────┐
│   Pellet Activo (reacción)  │
│                              │
│     ┌───────┐                │
│     │Defecto│ ← r∈[R/3,2R/3] │
│     │NO rxn │   θ∈[0°,45°]   │
│     └───────┘                │
│                              │
└─────────────────────────────┘
       Radio R = 2 mm
```

---

## 🧮 ECUACIONES GOBERNANTES

### Región Activa (con reacción)
```
∂C/∂t = D_eff [∂²C/∂r² + (1/r)∂C/∂r + (1/r²)∂²C/∂θ²] - k_app × C
```

### Región del Defecto (sin reacción)
```
∂C/∂t = D_eff [∂²C/∂r² + (1/r)∂C/∂r + (1/r²)∂²C/∂θ²]
```

### Donde:
- **C(r,θ,t)**: Concentración de CO [mol/m³]
- **D_eff**: Difusividad efectiva = 1.04×10⁻⁶ m²/s
- **k_app**: Constante cinética = 4.0×10⁻³ s⁻¹ (0 en defecto)

---

## 🔢 CONDICIONES DE FRONTERA

### 1. Centro del pellet (r = 0)
```
∂C/∂r = 0    [Simetría]
```

### 2. Superficie externa (r = R)
```
-D_eff × ∂C/∂r = k_c × (C_s - C_bulk)    [Robin/Convectiva]
```
Donde:
- k_c = 0.170 m/s (coeficiente de transferencia externa)
- C_bulk = 0.0145 mol/m³ (concentración en el gas)

### 3. Frontera angular (θ = 0 y θ = 2π)
```
C(r, 0, t) = C(r, 2π, t)    [Periodicidad]
```

### 4. Interfaces defecto-activo
```
C continua
Flujo continuo: D_eff × ∇C continuo
```

---

## ⏱️ CONDICIÓN INICIAL

```
C(r, θ, 0) = 0    ∀ r, θ
```
El pellet está completamente **libre de CO** en t=0.

---

## 🔧 MÉTODO NUMÉRICO REQUERIDO

### Técnica
**Crank-Nicolson** (implícito de segundo orden)

### Discretización
- **Espacial**: Volúmenes finitos en malla polar 2D
- **Temporal**: Promedio nivel n y n+1
- **Malla**: 
  - Radial: 61 nodos (Δr = 3.33×10⁻⁵ m)
  - Angular: 96 nodos (Δθ = 3.75°)

### Propiedades
- ✅ Incondicionalmente estable
- ✅ Segundo orden en tiempo y espacio
- ✅ Conserva masa

---

## 📊 DELIVERABLES REQUERIDOS (Sección 1.5)

### 1. Gráficos Obligatorios (3 mínimos)
Superficie 3D o mapas de contorno mostrando C(r,θ) en:

1. **t = 0**: Condición inicial
2. **t = t_ss/2**: 50% del tiempo al estado estacionario
3. **t = t_ss**: Estado estacionario

**Requisitos**:
- ✅ Marcar claramente la **región del defecto**
- ✅ Incluir colorbar con unidades
- ✅ Ejes etiquetados (r, θ o coordenadas cartesianas)
- ✅ Título descriptivo con tiempo

### 2. Gráficos Adicionales Sugeridos
- Evolución temporal C(t) en puntos específicos
- Perfiles radiales C(r) para diferentes θ
- Comparación concentración en defecto vs región activa
- Flujo de masa en superficie

### 3. Análisis de Resultados (10 puntos - CRÍTICO)
Debe incluir:
- ✅ Descripción de observaciones
- ✅ Interpretación física de patrones
- ✅ Comparación con literatura (3+ referencias)
- ✅ Discusión del efecto del defecto
- ✅ Tiempo al estado estacionario
- ✅ Efectividad del pellet (η)
- ✅ Verificación de balance de masa

---

## ✅ CRITERIOS DE CONVERGENCIA

### Estado Estacionario Alcanzado Cuando:
```
1. max|C^(n+1) - C^n| / max|C^(n+1)| < 10⁻⁶
2. Condición mantenida por 3 pasos temporales consecutivos
3. Balance de masa: error < 1%
```

---

## 🔍 VALIDACIONES OBLIGATORIAS

### Durante el desarrollo:
1. ✅ **Validación dimensional**: Toda ecuación debe ser dimensionalmente correcta
2. ✅ **Balance de masa**: Acumulación = Entrada - Salida - Consumo (error < 1%)
3. ✅ **Tests unitarios**: Cada función debe tener tests
4. ✅ **Casos límite**: 
   - k_app → 0: Solo difusión
   - D_eff → ∞: Control cinético
   - Defecto: Perfil diferente al resto

### Post-simulación:
1. ✅ Calcular Módulo de Thiele (φ)
2. ✅ Calcular efectividad del pellet (η)
3. ✅ Criterio de Weisz-Prater (N_WP << 1?)
4. ✅ Comparar con literatura

---

## 📚 PARÁMETROS CLAVE (ver PARAMETROS_PROYECTO.md)

| Parámetro | Valor | Unidades |
|-----------|-------|----------|
| D_eff | 1.04×10⁻⁶ | m²/s |
| k_app (activo) | 4.0×10⁻³ | s⁻¹ |
| k_app (defecto) | 0 | s⁻¹ |
| k_c | 0.170 | m/s |
| C_bulk | 0.0145 | mol/m³ |
| T | 673 | K |
| R | 0.002 | m |
| Φ (Thiele) | 0.124 | - |

---

## 🚨 PUNTOS CRÍTICOS

### Manejo del Centro (r=0)
El término 1/r genera singularidad. Usar **regla de L'Hôpital**:
```
lim (r→0) [1/r × ∂/∂r(r × ∂C/∂r)] = 2 × ∂²C/∂r²
```

### Región del Defecto
- NO olvidar poner k_app = 0
- Mantener D_eff igual que región activa
- Verificar continuidad en interfaces

### Condición de Robin
Implementar correctamente en r=R:
```python
# Aproximación de derivada en frontera
flux = -D_eff * (C[i,j] - C[i-1,j]) / dr
convection = k_c * (C[i,j] - C_bulk)
# Igualar: flux = convection
```

---

## 📖 USO DE LLMs (OBLIGATORIO DOCUMENTAR)

Según enunciado:
> "Si en su reporte aplica una herramienta de IA, debe incorporar en su
> metodología el uso que se le dio a dicha herramienta"

**Acción**: Usar sistema de logs en `.cursor/rules/04_SISTEMA_LOGS_TRABAJO.md`

---

## 🎯 MÉTRICAS DE ÉXITO

Tu simulación es exitosa si:
- ✅ Converge al estado estacionario
- ✅ Balance de masa < 1% error
- ✅ Validación dimensional pasa
- ✅ Tests pasan al 100%
- ✅ 3 gráficos generados correctamente
- ✅ Análisis físicamente coherente
- ✅ Comparación con literatura

---

## 📞 DOCUMENTOS RELACIONADOS

- **Parámetros completos**: `PARAMETROS_PROYECTO.md`
- **Reglas de calidad**: `.cursor/rules/03_REGLAS_CALIDAD_DESARROLLO.md`
- **Guía de visualización**: `.cursor/rules/05_GUIA_VISUALIZACION_P2.md`
- **Sistema de logs**: `.cursor/rules/04_SISTEMA_LOGS_TRABAJO.md`
- **Enunciado completo**: `/mnt/project/Proyecto_Personal_2_*.pdf`

---

## 🏆 PUNTUACIÓN

| Sección | Puntos | Elemento Clave |
|---------|--------|----------------|
| 1.5 Gráficos | - | Mínimo 3 gráficos + defecto marcado |
| 1.5.1 Análisis | 10 | Interpretación física + literatura |
| Código | - | Funcional + validado |
| Metodología | - | Documentar uso de IA |

---

**RECUERDA**: Este es un problema de **difusión-reacción heterogéneo** con geometría no trivial. La región del defecto altera significativamente los perfiles de concentración.

**ÉXITO** = Código correcto + Visualización clara + Análisis profundo

---

**ÚLTIMA ACTUALIZACIÓN**: 2025-10-28  
**VERSIÓN**: 1.0 (Resumido para Cursor)  
**AUTOR**: Adrián Vargas Tijerino (C18332)