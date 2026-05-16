# Protocolo de Verificación del Proyecto K-QGMIP

> **Objetivo:** Confirmar que los tres módulos implementados (Módulo A: `kpartir`, Módulo B: `KQNodes`,
> Módulo C: `KGeometricSIA`) están correctamente instalados, importan sin errores y producen
> resultados válidos antes de la entrega y grabación del video tutorial.

---

## 1. Prerrequisitos

### 1.1 Python

Se requiere **Python 3.10 o superior**. Verificar con:

```bash
python3 --version
```

### 1.2 Librerías necesarias

| Librería  | Uso en el proyecto                        | Instalación                         |
|-----------|-------------------------------------------|-------------------------------------|
| `numpy`   | Operaciones matriciales sobre la TPM      | `pip install numpy`                 |
| `scipy`   | Distancia de Wasserstein (EMD) en SIA     | `pip install scipy`                 |

> **Nota importante:** Las librerías `pyemd`, `pyinstrument`, `pyttsx3` y `colorama` son
> dependencias internas de GeoMIP y QNodes, pero **no necesitan instalarse** para ejecutar
> los tests ni el script de verificación. Todos los archivos de prueba incluyen *stubs*
> automáticos que las simulan sin necesidad de conexión a internet ni `pip install`.

Instalar las dos librerías requeridas con un solo comando:

```bash
pip install numpy scipy
```

### 1.3 Repositorio

Clonar el repositorio si aún no está disponible localmente:

```bash
git clone https://github.com/JuManoel/projecto-analisis-20261.git
cd projecto-analisis-20261
```

Todos los comandos del resto de este protocolo se ejecutan desde la carpeta
**`projecto-analisis-20261/`** como directorio de trabajo.

---

## 2. Estructura del Proyecto (referencia rápida)

```
projecto-analisis-20261/
│
├── QNodes/
│   ├── src/
│   │   ├── models/core/system.py            ← Módulo A: System.kpartir()
│   │   ├── strategies/k_qnodes.py           ← Módulo B: KQNodes
│   │   └── .samples/                        ← TPMs de ejemplo (N2A.csv, N3A.csv, …)
│   └── tests/
│       ├── test_kpartir.py                  ← Suite Módulo A — 5 tests
│       ├── test_kqnodes.py                  ← Suite Módulo B — 6 tests
│       └── test_kgeomip.py                  ← Suite Módulo C — 5 tests
│
├── GeoMIP/src/Method2_Dynamic_Programming_Reformulation/
│   └── src/controllers/strategies/
│       └── k_geometric.py                   ← Módulo C: KGeometricSIA
│
├── verificar_todo.py                        ← Script de verificación integral
└── Protocolo_Verificacion_KQGMIP.md         ← Este documento
```

---

## 3. Ejecución de la Suite de 16 Tests

### 3.1 Suite Módulo A — `System.kpartir()` (5 tests)

**¿Qué verifica?**
Que la función `kpartir()` generaliza correctamente a `bipartir()` cuando `k=2`,
produce distribuciones válidas para `k=3`, lanza `ValueError` ante cobertura incompleta,
activa la memoización y marginaliza correctamente grupos singleton.

```bash
python3 QNodes/tests/test_kpartir.py
```

**Salida esperada:**
```
  [PASS] test_kpartir_k2_equivalente_bipartir_n2
  [PASS] test_kpartir_k3_valido_n3
  [PASS] test_kpartir_cobertura_incompleta_lanza_error
  [PASS] test_kpartir_memoizacion_activa
  [PASS] test_kpartir_singleton_marginaliza_completamente

  RESULTADO: 5/5
```

### 3.2 Suite Módulo B — `KQNodes` (6 tests)

**¿Qué verifica?**
Que el algoritmo voraz produce exactamente `k` grupos no vacíos con cobertura total,
respeta el límite `max_size` en cada fase (garantizando que queden suficientes vértices
para las fases restantes), es determinista ante entradas iguales y formatea
correctamente la notación de k-partición.

```bash
python3 QNodes/tests/test_kqnodes.py
```

**Salida esperada:**
```
  [PASS] test_kqnodes_k2_produce_exactamente_2_grupos
  [PASS] test_kqnodes_k3_produce_exactamente_3_grupos
  [PASS] test_kqnodes_max_size_cumplido_por_fase
  [PASS] test_kqnodes_determinismo
  [PASS] test_kqnodes_vertices_a_fj_mj_correcto
  [PASS] test_kqnodes_formato_k_grupos

  RESULTADO: 6/6
```

### 3.3 Suite Módulo C — `KGeometricSIA` (5 tests)

**¿Qué verifica?**
Que la tabla de costos BFS/DP tiene valores acotados en `[0, 1]`, que
`identificar_particiones_optimas()` genera candidatos con exactamente `k` grupos
de cobertura total, que `System.kpartir()` produce distribuciones válidas a partir
de esos candidatos, y que `find_mip()` retorna la pérdida EMD mínima correcta.

```bash
python3 QNodes/tests/test_kgeomip.py
```

**Salida esperada:**
```
  [PASS] test_kgeomip_tabla_costos_valores_validos
  [PASS] test_kgeomip_candidatos_tienen_k_grupos
  [PASS] test_kgeomip_distribucion_kpartir_valida
  [PASS] test_kgeomip_find_mip_retorna_perdida_valida
  [PASS] test_kgeomip_capa1_cobertura_total_futuros

  RESULTADO: 5/5
```

### 3.4 Ejecutar las tres suites de una sola vez

```bash
for f in test_kpartir test_kqnodes test_kgeomip; do
    echo ""; echo "════ $f ════"
    python3 QNodes/tests/$f.py
done
```

**Resultado global esperado: 16/16 tests en verde.**

---

## 4. Script de Verificación Integral

El archivo `verificar_todo.py` (en la raíz del repositorio) ejecuta una demostración
end-to-end de los tres módulos con datos reales y muestra un diagnóstico completo.

```bash
python3 verificar_todo.py
```

**Salida esperada (resumen):**

```
╔══════════════════════════════════════╗
║    VERIFICACIÓN K-QGMIP — INICIO     ║
╚══════════════════════════════════════╝

[1/3] Módulo A — System.kpartir() ...
      Sistema N=3, k=3 → distribución shape=(8,) ✓
      Pérdida EMD kpartir(k=3): 0.XXXX
      [PASS] Módulo A operativo

[2/3] Módulo B — KQNodes ...
      Sistema N=3, k=3 → 3 grupos encontrados ✓
      Pérdida EMD KQNodes(k=3): 0.XXXX
      Partición: G1:{...}  G2:{...}  G3:{...}
      [PASS] Módulo B operativo

[3/3] Módulo C — KGeometricSIA ...
      Sistema N=2, k=2 → candidatos generados ✓
      Pérdida EMD KGeoMIP(k=2): 0.XXXX
      [PASS] Módulo C operativo

╔══════════════════════════════════════╗
║        SISTEMA VALIDADO              ║
║  Los 3 módulos K-QGMIP funcionan     ║
║  correctamente en este entorno.      ║
╚══════════════════════════════════════╝
```

Si algún módulo falla, el script muestra el traceback específico y termina
con `[FALLO]` en lugar de `SISTEMA VALIDADO`.

---

## 5. Interpretación de los Resultados en Consola

### 5.1 ¿Qué es la Pérdida EMD y cuándo es correcta?

La **pérdida EMD** (Earth Mover's Distance) es el número clave de salida.
Mide cuánta información causal se destruye al separar el sistema en `k` grupos
independientes. Es siempre un número `≥ 0`.

| Rango de pérdida EMD        | Significado                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `EMD ≈ 0.0`                 | El sistema es naturalmente separable: la k-partición casi no destruye información. |
| `0.0 < EMD < 0.15`          | Integración causal baja: los grupos interactúan, pero la partición es razonablemente buena. |
| `0.15 ≤ EMD < 0.40`         | Integración moderada: la partición encontrada es la mejor disponible, pero el sistema resiste la división. |
| `EMD ≥ 0.40`                | Alta integración: el sistema es holístico; dividirlo en `k` grupos destruye información importante. |

### 5.2 ¿Cómo saber que se encontró la MIP (Partición de Mínima Información)?

Los algoritmos implementados buscan la partición que **minimiza** la pérdida EMD entre
todos los candidatos evaluados. La salida confirma que es la MIP cuando se cumplen estas
cuatro condiciones simultáneamente:

1. **`len(grupos) == k` exactamente.**
   Si el algoritmo produce más o menos grupos, la búsqueda falló.
   Los tests `test_kqnodes_k3_produce_exactamente_3_grupos` y
   `test_kgeomip_candidatos_tienen_k_grupos` verifican esto.

2. **`union(grupos) == todos_los_vértices`** (cobertura total, sin duplicados).
   Ninguna variable queda sin asignar y ninguna aparece en dos grupos.
   Verificado por `test_kgeomip_capa1_cobertura_total_futuros`.

3. **La pérdida EMD es un número `float` en `[0, M × 0.5]`**, donde `M` es el número
   de variables futuras. Un valor fuera de ese rango indica un error numérico.

4. **La distribución marginal del sistema particionado tiene `shape == (2^N,)`**
   y valores en `[0, 1]` que suman 1. Verificado por
   `test_kgeomip_distribucion_kpartir_valida`.

### 5.3 Diferencia entre los resultados de KQNodes y KGeometricSIA

Es **normal** que ambos algoritmos produzcan pérdidas EMD distintas para el mismo
sistema y el mismo `k`:

- **KQNodes** usa una heurística voraz: elige el grupo óptimo local en cada fase.
  Puede dar una pérdida ligeramente más alta porque no explora todos los candidatos.

- **KGeometricSIA** genera múltiples candidatos (capa de cuantiles + capa de k-medoids)
  y elige el de menor pérdida EMD entre todos. Tiende a encontrar particiones con
  menor pérdida, pero no garantiza el óptimo global.

Ninguno es "más correcto" que el otro: son aproximaciones complementarias al problema
NP-difícil de encontrar la MIP exacta para `k > 2`.

### 5.4 Señales de advertencia (qué NO debe verse)

| Mensaje en consola                               | Causa probable                                     | Acción                                      |
|--------------------------------------------------|----------------------------------------------------|---------------------------------------------|
| `ModuleNotFoundError: src.xxx`                   | Comando ejecutado desde la carpeta incorrecta      | Moverse a `projecto-analisis-20261/`        |
| `ValueError: NCubo X no pertenece a ningún grupo`| Los índices de la partición no cubren todos los cubos | Revisar que `union(grupos) == todos`     |
| `IndexError` en `algorithm()`                    | `k > N` (más grupos que variables)                 | Usar `k ≤ N`                                |
| `[PASS]` en todos los tests pero EMD = 0.0 siempre | TPM constante (sistema trivial)               | Usar una TPM con variación real             |
| Los tres tests pasan pero `verificar_todo.py` falla | Conflicto de `sys.path` fuera de la carpeta raíz | Ejecutar desde `projecto-analisis-20261/` |

---

## 6. Resumen de Comandos (cheat sheet)

```bash
# Desde projecto-analisis-20261/

# Tests individuales
python3 QNodes/tests/test_kpartir.py     # 5 tests — Módulo A
python3 QNodes/tests/test_kqnodes.py     # 6 tests — Módulo B
python3 QNodes/tests/test_kgeomip.py     # 5 tests — Módulo C

# Todo de una vez
for f in test_kpartir test_kqnodes test_kgeomip; do
    echo "════ $f ════"; python3 QNodes/tests/$f.py
done

# Verificación integral con demostración end-to-end
python3 verificar_todo.py
```

---

*Documento generado como parte del proyecto K-QGMIP — Análisis de Algoritmos, Universidad de Caldas, 2026-1.*
