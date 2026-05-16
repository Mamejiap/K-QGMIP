# Auditoría y Guía de Entrega Final — K-QGMIP

> Universidad de Caldas · Análisis de Algoritmos · 2026-1  
> Ejecutada el 15-05-2026. Refleja el estado real del repositorio tras la limpieza.

---

## 1. Esquema de Directorios (árbol ideal de entrega)

```
K-QGMIP_Final/                          ← carpeta raíz para la entrega
│
├── interfaz_kqmip.py                    ★ Interfaz interactiva principal
├── verificar_todo.py                    ★ Verificación integral (16/16 tests)
├── Protocolo_Verificacion_KQGMIP.md     ★ Guía de instalación y tests
├── K-QGMIP_Final_Auditoria.md           ★ Este documento
│
├── QNodes/                              Módulos A y B (TPM-based SIA)
│   ├── src/
│   │   ├── models/
│   │   │   └── core/
│   │   │       ├── system.py            ★ Módulo A — System + kpartir()
│   │   │       ├── solution.py              Contenedor de resultados
│   │   │       └── ncube.py                 NCubo (marginalización)
│   │   ├── strategies/
│   │   │   ├── k_qnodes.py              ★ Módulo B — KQNodes
│   │   │   └── q_nodes.py                   QNodes original (base)
│   │   ├── middlewares/
│   │   │   ├── slogger.py                   Logger estructurado
│   │   │   └── profile.py                   Decorador de perfilado
│   │   ├── constants/
│   │   │   ├── base.py                      EFFECT, ACTUAL, ABECEDARY…
│   │   │   └── models.py                    Etiquetas y constantes
│   │   └── .samples/                        TPMs de ejemplo (N2A–N15B.csv)
│   └── tests/
│       ├── test_kpartir.py              ★ Suite Módulo A  (5 tests — PASS)
│       ├── test_kqnodes.py              ★ Suite Módulo B  (6 tests — PASS)
│       └── test_kgeomip.py              ★ Suite Módulo C  (5 tests — PASS)
│
├── GeoMIP/                              Módulo C (geometric BFS/DP SIA)
│   ├── data/samples/                        TPMs de ejemplo GeoMIP
│   ├── results/                             Hojas de resultados experimentales
│   └── src/Method2_Dynamic_Programming_Reformulation/
│       └── src/
│           ├── controllers/
│           │   ├── manager.py               Gestor del sistema GeoMIP
│           │   └── strategies/
│           │       ├── k_geometric.py       ★ Módulo C — KGeoMIP / KGeometricSIA
│           │       └── geometric.py             GeometricSIA (clase padre)
│           ├── models/core/
│           │   ├── system.py                System.kpartir() (versión GeoMIP)
│           │   └── solution.py              Contenedor de resultados
│           ├── funcs/
│           │   └── base.py                  emd_efecto(), emd_causal()
│           └── middlewares/
│               ├── slogger.py               Logger GeoMIP
│               └── profile.py               Decorador de perfilado GeoMIP
│
└── docs/                                Documentación del proyecto
    ├── Manual_Usuario_KQMIP.docx        ★ Manual de usuario (Word)
    ├── 1_Guía_Proyecto_ADAV1_2_0.pdf       Guía original del proyecto
    ├── 2_GeoMIP.pdf                         Paper GeoMIP de referencia
    └── Proyecto_KGeoMIP.docx                Documento del proyecto
```

> Los archivos marcados con **★** son los creados o modificados en este proyecto.

---

## 2. Inventario de Archivos Esenciales

### Módulo A — `System.kpartir()` (395 líneas)
**Ruta:** `QNodes/src/models/core/system.py`

Extiende la clase `System` del algoritmo QNodes original añadiendo el método
`kpartir(grupos)`. Recibe una lista de k pares `(futuros_j, mecanismo_j)` y
construye el subsistema factorizado donde cada grupo es independiente. Usa
memoización por clave de grupos. Es la pieza de evaluación que KQNodes y
KGeoMIP llaman internamente para calcular la pérdida EMD de cada candidato.

### Módulo B — `KQNodes` (652 líneas)
**Ruta:** `QNodes/src/strategies/k_qnodes.py`

Implementa el algoritmo **Peeling Greedy Secuencial** para k-particiones. En
k−1 fases consecutivas selecciona el grupo Gⱼ que minimiza la EMD acumulada
(función submodular), garantizando que en cada fase queden suficientes vértices
para las fases restantes mediante la restricción `max_size`. La fase final Gₖ
es el residuo de vértices no asignados. Hereda de `SIA` y expone la API pública
`aplicar_estrategia(estado, condicion, alcance, mecanismo, k)`.

### Módulo C — `KGeometricSIA` / `KGeoMIP` (483 líneas)
**Ruta:** `GeoMIP/.../strategies/k_geometric.py`

Extiende `GeometricSIA` para k-particiones usando la geometría del hipercubo
binario. Construye una tabla de costos BFS/DP (`tx(s₀, sⱼ)[idx]`) y genera
candidatos de k grupos mediante dos capas: (1) partición por cuantiles del
costo y (2) asignación k-medoids usando estados de referencia. Evalúa cada
candidato con `System.kpartir() + emd_efecto()` y retorna la MIP. La clase se
llama `KGeometricSIA` en código pero usa la etiqueta `"KGeoMIP"` en resultados
y documentación (`KGEOMIP_LABEL`).

### Suites de Tests (208 / 251 / 321 líneas)
**Ruta:** `QNodes/tests/test_kpartir.py`, `test_kqnodes.py`, `test_kgeomip.py`

Tres archivos independientes, cada uno autocontenido con stubs para las
dependencias no instalables. No requieren pytest: usan un patrón `try/except`
propio. Resultado verificado: **16/16 PASS**.

### `interfaz_kqmip.py` (601 líneas)
**Ruta:** raíz del repositorio

Script interactivo para el usuario final. Solicita la ruta del CSV, muestra
vista previa de la TPM, valida el estado inicial y k, permite elegir KQNodes o
KGeoMIP, muestra el progreso de fases en tiempo real mediante subprocesos
aislados con salida estructurada, presenta la tabla de resultados con
interpretación automática de la EMD y ofrece guardar el resultado en `.txt`.

### `verificar_todo.py` (299 líneas)
**Ruta:** raíz del repositorio

Verifica los tres módulos con datos reales en subprocesos aislados. Imprime
`SISTEMA VALIDADO` si los tres pasan. Útil como smoke-test rápido antes de
una demo o presentación.

### `Protocolo_Verificacion_KQGMIP.md` (289 líneas)
**Ruta:** raíz del repositorio

Guía completa de prerrequisitos, instalación, ejecución de tests e
interpretación de resultados. Incluye cheat-sheet de comandos y tabla de
señales de advertencia.

---

## 3. Archivos Eliminados (purga ejecutada)

Los siguientes elementos fueron eliminados durante la auditoría porque son
artefactos de desarrollo que se regeneran automáticamente en tiempo de
ejecución y no aportan valor a la entrega:

| Tipo | Descripción | Cantidad |
|---|---|---|
| `__pycache__/` | Bytecode cacheado por Python | 15 carpetas |
| `*.pyc` | Archivos de bytecode compilado | 64 archivos |
| `.logs/` | Logs de ejecución runtime | 3 carpetas |
| `review/profiling/` | Reportes HTML de pyinstrument | 11 archivos HTML |
| `.qodo/` | Artefactos del IDE Qodo | 1 carpeta |

### Archivos con permisos de solo lectura (eliminar manualmente)

Dos elementos del repositorio original tienen permisos de solo lectura y no
pudieron eliminarse automáticamente. Requieren acción manual:

```
QNodes/tests/PruebasIniciales.xlsx       ← hoja de dev anterior al proyecto
GeoMIP/.../pyphi.log                     ← log de librería externa pyphi
```

---

## 4. Convenciones de Nomenclatura

### Nombre canónico: `KGeoMIP` vs `KGeometricSIA`

El manual y la documentación usan **`KGeoMIP`** como nombre de marca. En el
código, la clase se llama `KGeometricSIA` por claridad de herencia
(`KGeometricSIA(GeometricSIA)`). Esta dualidad ya está resuelta internamente:

```python
# En k_geometric.py — línea 33
KGEOMIP_LABEL = "KGeoMIP"   # etiqueta que aparece en todos los resultados

# La clase:
class KGeometricSIA(GeometricSIA):
    """KGeoMIP: extensión de GeometricSIA para k-particiones (k ≥ 2)."""
```

Para que los imports externos puedan usar el nombre canónico `KGeoMIP`, se
recomienda añadir al final de `k_geometric.py` el alias:

```python
# Alias de nombre canónico — añadir al final de k_geometric.py
KGeoMIP = KGeometricSIA
```

### Nombre canónico: `KQNodes`

La clase `KQNodes` en `k_qnodes.py` ya cumple exactamente la convención
exigida. No requiere cambios.

### Archivo de entrada: `interfaz_kqmip.py`

El nombre sigue la convención `snake_case` estándar de Python para scripts
ejecutables. En la documentación se referencia como "Interfaz K-QGMIP".

---

## 5. Instrucciones de Limpieza y Reorganización

### 5.1 Limpieza (Bash / macOS / Linux)

```bash
# Ejecutar desde la raíz del repositorio: projecto-analisis-20261/

# ── Eliminar bytecode de Python ───────────────────────────────────────────
find . -not -path './.git/*' -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -not -path './.git/*' -name "*.pyc" -delete 2>/dev/null
find . -not -path './.git/*' -name "*.pyo" -delete 2>/dev/null

# ── Eliminar logs de runtime ──────────────────────────────────────────────
find . -not -path './.git/*' -name ".logs" -type d -exec rm -rf {} + 2>/dev/null
find . -not -path './.git/*' -name "pyphi.log" -delete 2>/dev/null

# ── Eliminar reportes de profiling ───────────────────────────────────────
find . -not -path './.git/*' -path "*/review/profiling" -type d -exec rm -rf {} + 2>/dev/null

# ── Eliminar artefactos de IDE ────────────────────────────────────────────
rm -rf .qodo/

# ── Eliminar hoja de dev anterior ────────────────────────────────────────
rm -f QNodes/tests/PruebasIniciales.xlsx

# ── Añadir .gitignore robusto (evita que vuelvan a aparecer) ─────────────
cat >> .gitignore << 'EOF'

# K-QGMIP — artefactos de desarrollo
__pycache__/
*.py[cod]
.logs/
pyphi.log
review/profiling/
.qodo/
*.txt.resultado_*
resultado_*.txt
EOF

echo "Limpieza completada."
```

### 5.2 Limpieza (Windows PowerShell)

```powershell
# Ejecutar desde la raíz: projecto-analisis-20261\

# Bytecode
Get-ChildItem -Recurse -Filter "__pycache__" -Directory |
    Where-Object { $_.FullName -notmatch '\\.git' } |
    Remove-Item -Recurse -Force

Get-ChildItem -Recurse -Include "*.pyc","*.pyo" |
    Where-Object { $_.FullName -notmatch '\\.git' } |
    Remove-Item -Force

# Logs de runtime
Get-ChildItem -Recurse -Filter ".logs" -Directory |
    Where-Object { $_.FullName -notmatch '\\.git' } |
    Remove-Item -Recurse -Force

Get-ChildItem -Recurse -Filter "pyphi.log" |
    Remove-Item -Force

# Profiling
Get-ChildItem -Recurse -Filter "profiling" -Directory |
    Where-Object { $_.FullName -notmatch '\\.git' } |
    Remove-Item -Recurse -Force

# IDE
Remove-Item -Recurse -Force ".qodo" -ErrorAction SilentlyContinue

# Dev artifact
Remove-Item -Force "QNodes\tests\PruebasIniciales.xlsx" -ErrorAction SilentlyContinue

Write-Host "Limpieza completada."
```

### 5.3 Crear la carpeta de entrega `K-QGMIP_Final`

```bash
# Bash — desde el directorio PADRE del repositorio
cp -r projecto-analisis-20261 K-QGMIP_Final

# Limpieza adicional en la copia (por si se regeneraron caches)
cd K-QGMIP_Final
find . -not -path './.git/*' -name "__pycache__" -type d -exec rm -rf {} +
find . -not -path './.git/*' -name "*.pyc" -delete

echo "Carpeta K-QGMIP_Final lista para entrega."
```

```powershell
# PowerShell — desde el directorio PADRE del repositorio
Copy-Item -Recurse "projecto-analisis-20261" "K-QGMIP_Final"

Set-Location "K-QGMIP_Final"
Get-ChildItem -Recurse -Filter "__pycache__" -Directory |
    Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Include "*.pyc" | Remove-Item -Force

Write-Host "Carpeta K-QGMIP_Final lista para entrega."
```

### 5.4 Alias KGeoMIP (un solo comando)

```bash
# Añadir alias al final de k_geometric.py
echo "" >> GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/k_geometric.py
echo "# Alias canónico — permite: from k_geometric import KGeoMIP" >> GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/k_geometric.py
echo "KGeoMIP = KGeometricSIA" >> GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/k_geometric.py
echo "OK alias KGeoMIP añadido."
```

### 5.5 Verificación final (post-limpieza)

```bash
# Confirmar que no queda basura y que los 16 tests siguen pasando
find . -not -path './.git/*' -name "__pycache__" | head -3 && echo "(debe estar vacío)"
python3 verificar_todo.py
```

---

## Resumen ejecutivo de la auditoría

| Indicador | Valor |
|---|---|
| Módulos implementados | 3 (A: kpartir, B: KQNodes, C: KGeoMIP) |
| Tests totales | **16 / 16 PASS** |
| Archivos creados esta sesión | 9 |
| Archivos eliminados (limpieza) | 64 .pyc + 15 `__pycache__` + 3 `.logs` + 11 HTML |
| Convención de nombre canónico | `KGeoMIP` (alias) + `KQNodes` (directo) |
| Comando de arranque para entrega | `python3 interfaz_kqmip.py` |
| Comando de smoke-test | `python3 verificar_todo.py` |

---

*Auditoría generada automáticamente — K-QGMIP · Universidad de Caldas · 2026-1*
