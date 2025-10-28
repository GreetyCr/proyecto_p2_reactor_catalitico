# PARÁMETROS DEL PROYECTO - TABLA MAESTRA
## Proyecto Personal 2: Transferencia de Masa en Reactor Catalítico

---

> **IMPORTANTE**: Este archivo contiene TODOS los parámetros validados del proyecto.  
> **NO modificar** sin validación dimensional completa.  
> **Fuente**: Sección 2.3 del enunciado oficial.

---

## TABLA I: GEOMETRÍA Y DOMINIOS

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Diámetro del pellet | D | m | 0.004 | Enunciado |
| Radio del pellet | R | m | 0.002 | R = D/2 |
| Radio interno del defecto | r₁ | m | 0.000667 | r₁ = R/3 |
| Radio externo del defecto | r₂ | m | 0.001333 | r₂ = 2R/3 |
| Ángulo inicial del defecto | θ₁ | rad | 0 | Enunciado |
| Ángulo final del defecto | θ₂ | rad | 0.7854 | 45° = π/4 rad |
| Longitud axial | L | m | >> D | Enunciado (invarianza axial) |
| Perímetro externo | P | m | 0.01257 | P = 2πR |

### 📐 Región del Defecto
- **Radial**: r ∈ [0.000667 m, 0.001333 m]
- **Angular**: θ ∈ [0°, 45°] = [0, 0.7854 rad]
- **Característica**: Sin reacción química (k_app = 0)

---

## TABLA II: CONDICIONES DE OPERACIÓN EXTERNAS

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Temperatura del gas | T | K | 673 | Enunciado (400°C) |
| Presión del sistema | P | Pa | 101325 | 1 atm estándar |
| Concentración CO | y_CO | ppm | 800 | Enunciado |
| Concentración bulk CO | C_bulk | mol·m⁻³ | 0.0145 | C = (y×P)/(RT) = (800×10⁻⁶×101325)/(8.314×673) |
| Velocidad superficial | u_s | m·s⁻¹ | 0.3 | Enunciado |
| Porosidad del lecho | ε_b | adim. | 0.4 | Dixon (1988) |
| Velocidad intersticial | u_i | m·s⁻¹ | 0.75 | u_i = u_s/ε_b = 0.3/0.40 |
| Concentración inicial | C₀ | mol·m⁻³ | 0 | Enunciado (pellet libre de CO en t=0) |

**Fuentes**: Dixon, 1988

---

## TABLA III: PROPIEDADES TERMOFÍSICAS DEL GAS

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Densidad del gas | ρ_gas | kg·m⁻³ | 0.524 | ρ = PM/(RT) = (101325×0.02897)/(8.314×673) |
| Viscosidad dinámica | μ | Pa·s | 3.32×10⁻⁵ | Sutherland con S=110.4 K |
| Viscosidad cinemática | ν | m²·s⁻¹ | 6.34×10⁻⁵ | ν = μ/ρ = 3.32×10⁻⁵/0.524 |
| Difusividad CO-aire | D_CO-aire | m²·s⁻¹ | 8.75×10⁻⁵ | Tabla de difusividades en exceso de aire |
| Número de Schmidt | Sc | adim. | 0.724 | Sc = ν/D = 6.34×10⁻⁵/8.75×10⁻⁵ |

**Fuentes**: White, 2016; The Engineering ToolBox, 2018

### 🔬 Notas Termodinámicas
- Masa molar aire: M = 0.02897 kg/mol
- Gas ideal: PV = nRT
- Temperatura de referencia Sutherland: T₀ = 273 K

---

## TABLA IV: TRANSFERENCIA DE MASA INTERFASE

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Diámetro de partícula | d_p | m | 0.004 | d_p = D (aproximación conservadora) |
| Reynolds de partícula | Re_p | adim. | 19.0 | Re_p = ρu_sd_p/μ = (0.524×0.3×0.004)/(3.32×10⁻⁵) |
| Número de Sherwood | Sh | adim. | 7.78 | Wakao-Funazkri: Sh = 2 + 1.1×Re_p^0.6×Sc^(1/3) |
| Coef. convectivo masa | k_c | m·s⁻¹ | 0.170 | k_c = Sh×D_CO-aire/d_p = 7.78×8.75×10⁻⁵/0.004 |
| Rango de validez Re_p | - | - | [3, 10⁴] | Wakao-Funazkri (1978) |
| Rango de validez Sc | - | - | [0.6, 10³] | Wakao-Funazkri (1978) |

**Fuentes**: Wakao & Funazkri, 1978

### 📊 Correlación de Wakao-Funazkri
```
Sh = 2 + 1.1 × Re_p^0.6 × Sc^(1/3)
```
- Válida para lechos empacados aleatorios
- Partículas esféricas/cilíndricas
- NO usar correlaciones para partículas aisladas (Ranz-Marshall)

---

## TABLA V: DIFUSIÓN INTRAPARTICULAR

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Porosidad del pellet | ε | adim. | 0.45 | Pt/Al₂O₃ típico |
| Tortuosidad | τ | adim. | 3.0 | Pt/Al₂O₃ típico |
| Radio de poro | r_poro | m | 10×10⁻⁹ | Pt/Al₂O₃ promedio (10 nm) |
| Camino libre medio | λ | m | 1.93×10⁻⁷ | λ = k_B T/(√2πd²P); d_CO = 3.76×10⁻¹⁰ m |
| Número de Knudsen | Kn | adim. | 19.3 | Kn = λ/r_poro = 1.93×10⁻⁷/(10×10⁻⁹) |
| Difusividad de Knudsen | D_Kn | m²·s⁻¹ | 7.43×10⁻⁶ | D_Kn = (2r_poro/3)√(8RT/(πM_CO)) |
| Difusividad combinada | D_comb | m²·s⁻¹ | 6.97×10⁻⁶ | 1/D_comb = 1/D_molecular + 1/D_Knudsen |
| Difusividad efectiva | D_eff | m²·s⁻¹ | 1.04×10⁻⁶ | D_eff = ε×D_comb/τ = 0.45×6.97×10⁻⁶/3.0 |
| Módulo de Thiele | φ | adim. | 0.124 | φ = R√(k_app/D_eff) |

**Fuentes**: Hill, 2025; Abello, 2002; Mourkou et al., 2024

### 🔬 Régimen Difusivo
- **Kn >> 1**: Difusión de Knudsen (dominante en este caso: Kn = 19.3)
- **Kn << 1**: Difusión molecular
- **Transición**: Relación de Bosanquet

### 📐 Fórmulas Clave
```
D_Kn = (2/3) × r_poro × √(8RT/πM)
1/D_comb = 1/D_molecular + 1/D_Knudsen
D_eff = (ε/τ) × D_comb
```

---

## TABLA VI: CINÉTICA APARENTE

| Parámetro | Símbolo | Unidades | Valor | Expresión/Fuente |
|-----------|---------|----------|-------|------------------|
| Factor pre-exponencial | k₀ | s⁻¹ | 2.3×10⁵ | Enunciado |
| Energía de activación | E_a | J·mol⁻¹ | 1×10⁵ | 100 kJ/mol; Enunciado |
| Constante cinética aparente | k_app | s⁻¹ | 4.0×10⁻³ | k_app = k₀×exp(-E_a/(RT)) |
| Temperatura de operación | T | K | 673 | Constante (isotermia) |
| Orden de reacción | n | adim. | 1 | Primer orden en CO (exceso O₂) |
| Reacción en defecto | - | - | OFF | k_app = 0 en región defectuosa |

### ⚗️ Reacción
```
2 CO + O₂ → 2 CO₂
```
- Cinética aparente de 1er orden en CO (exceso de O₂)
- Ley de Arrhenius: k = k₀ exp(-E_a/RT)

### 🧮 Cálculo de k_app
```python
k_app = 2.3e5 * exp(-100000 / (8.314 * 673))
k_app = 2.3e5 * exp(-17.866)
k_app ≈ 4.0e-3 s⁻¹
```

---

## TABLA VII: ESPECIFICACIONES DE MALLADO

| Parámetro | Símbolo | Unidades | Valor | Justificación |
|-----------|---------|----------|-------|---------------|
| Nodos radiales | N_r | adim. | 61 | Resolución suficiente para R=2mm |
| Nodos angulares | N_θ | adim. | 96 | Cobertura completa 0-2π |
| Paso radial | Δr | m | 3.33×10⁻⁵ | Δr = R/(N_r-1) = 0.002/60 |
| Paso angular | Δθ | rad | 0.0654 | Δθ = 2π/(N_θ-1) ≈ 3.75° |
| Paso temporal inicial | Δt | s | 0.001 | Estabilidad Crank-Nicolson |

### 🎯 Criterios de Mallado
- **Radial**: Capturar gradientes en defecto (r₁=R/3, r₂=2R/3)
- **Angular**: Resolver defecto θ∈[0°,45°] con ~12 nodos
- **Temporal**: Incondicionalmente estable (Crank-Nicolson)

---

## TABLA VIII: CONDICIONES DE FRONTERA

| Tipo | Ubicación | Expresión | Valor/Descripción |
|------|-----------|-----------|-------------------|
| Simetría | r = 0 | ∂C/∂r = 0 | Centro del pellet |
| Robin (convectiva) | r = R | -D_eff×∂C/∂r = k_c×(C_s - C_bulk) | Superficie externa |
| Periodicidad | θ = 0, 2π | C(r,0) = C(r,2π) | Continuidad angular |
| Continuidad | Interfaz defecto-activo | C y flujo continuos | Interfaces internas |

### 🔄 Condición de Robin (r=R)
```
-D_eff × ∂C/∂r|_{r=R} = k_c × (C_s - C_bulk)

Donde:
- C_s: concentración superficial
- C_bulk = 0.0145 mol/m³
- k_c = 0.170 m/s
```

### 🎯 Condición Inicial (t=0)
```
C(r, θ, 0) = 0  ∀ r, θ
```
Pellet completamente libre de CO al inicio.

---

## TABLA IX: CRITERIOS DE CONVERGENCIA

| Criterio | Expresión | Valor Límite | Propósito |
|----------|-----------|--------------|-----------|
| Error relativo máximo | max\|C^{n+1} - C^n\| / max\|C^{n+1}\| | < 10⁻⁶ | Estado estacionario |
| Consistencia temporal | Condición mantenida | 3 pasos consecutivos | Estabilidad |
| Balance de masa | Error acumulado | < 1% | Conservación |
| Validación dimensional | Todas las ecuaciones | Exacta | Consistencia física |

### ⏱️ Definición de Estado Estacionario
El sistema alcanza estado estacionario cuando:
1. Error relativo < 10⁻⁶
2. Condición mantenida por 3 pasos temporales
3. Balance de masa < 1% error

---

## TABLA X: COEFICIENTES NUMÉRICOS (Crank-Nicolson)

### Para nodo típico i=30 (r=R/2) con Δt=0.001s

| Coeficiente | Símbolo | Valor | Expresión |
|-------------|---------|-------|-----------|
| Coef. radial superior | α₃₀ | 4.82×10⁻¹ | (D_eff×Δt)/(2×Δr²) × (r_{i+1/2}/r_i) |
| Coef. radial inferior | β₃₀ | 4.58×10⁻¹ | (D_eff×Δt)/(2×Δr²) × (r_{i-1/2}/r_i) |
| Coef. angular | γ₃₀ | 1.22×10⁻¹ | (D_eff×Δt)/(2×r_i²×Δθ²) |
| Coef. reacción | k×Δt/2 | 2.0×10⁻³ | k_app×Δt/2 = 4.0×10⁻³ × 0.001/2 |

### 🧮 Fórmulas Generales
```python
α_i = (D_eff * dt) / (2 * dr²) * (r_i + dr/2) / r_i
β_i = (D_eff * dt) / (2 * dr²) * (r_i - dr/2) / r_i
γ_i = (D_eff * dt) / (2 * r_i² * dθ²)
```

---

## 📊 NÚMEROS ADIMENSIONALES IMPORTANTES

| Número | Símbolo | Valor | Interpretación |
|--------|---------|-------|----------------|
| Reynolds | Re_p | 19.0 | Flujo transición (laminar-turbulento) |
| Schmidt | Sc | 0.724 | Difusividad momentum > difusividad masa |
| Sherwood | Sh | 7.78 | Transferencia convectiva moderada |
| Thiele | φ | 0.124 | NO hay limitación difusional interna (φ<<1) |
| Knudsen | Kn | 19.3 | Régimen de difusión de Knudsen |
| Weisz-Prater | N_WP | - | A calcular post-simulación |

### 🎯 Interpretaciones Físicas
- **φ = 0.124 << 1**: La reacción es lenta comparada con la difusión → cinética controla
- **Kn = 19.3 >> 1**: Difusión de Knudsen dominante (colisiones pared-molécula)
- **Re_p = 19**: Flujo en régimen de transición

---

## 🔍 VALIDACIONES DIMENSIONALES CRÍTICAS

### Ecuación de Transporte
```
∂C/∂t = D_eff × ∇²C - k_app × C

[mol/m³/s] = [m²/s] × [mol/m³/m²] - [1/s] × [mol/m³]
[mol/m³/s] = [mol/m³/s] - [mol/m³/s]  ✅ CORRECTO
```

### Condición de Robin
```
-D_eff × ∂C/∂r = k_c × (C_s - C_bulk)

[m²/s] × [mol/m³/m] = [m/s] × [mol/m³]
[mol/m²/s] = [mol/m²/s]  ✅ CORRECTO
```

---

## 📌 NOTAS DE IMPLEMENTACIÓN

### ⚠️ CRÍTICO
1. **Región de defecto**: k_app = 0 para r∈[r₁,r₂] y θ∈[0°,45°]
2. **Centro (r=0)**: Usar L'Hôpital → término radial se duplica
3. **Unidades**: SIEMPRE en SI (metros, segundos, moles)
4. **Validación**: Verificar balance de masa en cada paso temporal

### 🎯 Parámetros Variables Durante Simulación
- C(r,θ,t): Campo de concentración (incógnita)
- t: Tiempo (variable independiente)
- Δt: Puede adaptarse si es necesario

### 🔒 Parámetros Constantes (NO modificar)
- Todos los demás parámetros de las tablas I-VI

---

## 📚 REFERENCIAS

- Dixon, A.G. (1988) - Porosidad en lechos empacados
- Wakao & Funazkri (1978) - Correlación de Sherwood
- White, F.M. (2016) - Propiedades termofísicas
- The Engineering ToolBox (2018) - Difusividades
- Hill (2025); Abello (2002); Mourkou et al. (2024) - Catalizadores Pt/Al₂O₃
- Thiele (1939) - Módulo de Thiele
- Weisz & Prater (1954) - Criterio de limitación difusional

---

**ÚLTIMA ACTUALIZACIÓN**: 2025-10-28  
**VERSIÓN**: 1.0  
**AUTOR**: Adrián Vargas Tijerino (C18332)