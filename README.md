# K-QGMIP — Extensión a k-Particiones para GeoMIP y QNodes

> **Universidad de Caldas · Análisis y Diseño de Algoritmos · 2026-1**  
> Proyecto: Partición de Mínima Información (MIP) generalizada a k grupos

---

## ¿Qué es K-QGMIP?

K-QGMIP extiende dos algoritmos de análisis causal (*QNodes* y *GeoMIP*) para encontrar la **Partición de Mínima Información** cuando el sistema se divide en **k grupos independientes** (k ≥ 2), generalizando la bipartición clásica (k = 2).

Dado un sistema de N variables binarias descrito por su Matriz de Probabilidad de Transición (TPM), K-QGMIP responde:

> *"¿Cómo dividir el sistema en k partes causalmente independientes de modo que se destruya la menor cantidad de información posible?"*

La respuesta es la MIP: la partición que minimiza la distancia EMD (Earth Mover's Distance) entre la distribución del sistema original y la del sistema factorizado.

---

## Estructura del repositorio

```
projecto-analisis-20261/
│
├── interfaz_kqmip.py            ← Interfaz interactiva principal  ★
├── verificar_todo.py            ← Verificación integral (16/16 tests)  ★
├── Protocolo_Verificacion_KQGMIP.md  ★
├── K-QGMIP_Final_Auditoria.md  ★
├── README.md
│
├── QNodes/
│   ├── src/
│   │   ├── models/core/
│   │   │   └── system.py        ★ Módulo A — System.kpartir()
│   │   ├── strategies/
│   │   │   ├── k_qnodes.py      ★ Módulo B — KQNodes (Peeling Greedy)
│   │   │   └── q_nodes.py           QNodes original (base)
│   │   └── .samples/                TPMs de ejemplo (N2A–N15B.csv)
│   └── tests/
│       ├── test_kpartir.py      ★ 5/5 PASS
│       ├── test_kqnodes.py      ★ 6/6 PASS
│       └── test_kgeomip.py      ★ 5/5 PASS
│
├── GeoMIP/
│   └── src/Method2_Dynamic_Programming_Reformulation/src/
│       └── controllers/strategies/
│           ├── k_geometric.py   ★ Módulo C — KGeometricSIA / KGeoMIP
│           └── geometric.py         GeometricSIA original (base)
│
└── docs/
    ├── Manual_Usuario_KQMIP.docx
    └── Manual_Interfaz_KQMIP.docx
```

> Los archivos marcados con **★** son los creados o modificados en este proyecto.

---

## Los tres módulos implementados

### Módulo A — `System.kpartir()` (`QNodes/src/models/core/system.py`)

Extiende `System` con `kpartir(grupos)`: recibe k pares `(futuros_j, mecanismo_j)` y calcula la distribución del sistema factorizado con memoización.

```python
grupos = [
    (np.array([0, 1]), np.array([0, 1])),  # G1
    (np.array([2]),    np.array([2])),      # G2
]
dist = sistema.kpartir(grupos).distribucion_marginal()
```

### Módulo B — `KQNodes` (`QNodes/src/strategies/k_qnodes.py`)

**Peeling Greedy Secuencial**: en k−1 fases selecciona el grupo que minimiza la EMD acumulada. API pública:

```python
kqn = KQNodes(tpm)
resultado = kqn.aplicar_estrategia("000", "111", "111", "111", k=3)
```

### Módulo C — `KGeometricSIA` / `KGeoMIP` (`GeoMIP/.../k_geometric.py`)

Extiende `GeometricSIA` para k grupos: tabla BFS/DP + candidatos por cuantiles y k-medoids. Alias: `KGeoMIP = KGeometricSIA`.

---

## Instalación rápida

```bash
# 1. Clonar
git clone https://github.com/JuManoel/projecto-analisis-20261.git
cd projecto-analisis-20261

# 2. Instalar dependencias mínimas
pip install numpy scipy colorama

# 3. Verificar los 3 módulos (debe imprimir SISTEMA VALIDADO)
python3 verificar_todo.py

# 4. Lanzar la interfaz interactiva
python3 interfaz_kqmip.py
```

---

## Guía rápida de uso

```bash
# Siempre ejecutar desde la raíz del repositorio
cd projecto-analisis-20261
python3 interfaz_kqmip.py
```

La interfaz guía paso a paso:
1. Seleccionar CSV del menú numerado (o ruta propia)
2. Confirmar vista previa de la TPM
3. Ingresar estado inicial binario (p.ej. `000`)
4. Elegir algoritmo: `1` KQNodes · `2` KGeometricSIA
5. Ingresar k (2 ≤ k ≤ N)
6. Ver progreso en tiempo real y tabla de resultados

---

## Resultados de validación

| Suite | Tests | Estado |
|---|---|---|
| Módulo A — `System.kpartir()` | 5/5 | ✅ PASS |
| Módulo B — `KQNodes` | 6/6 | ✅ PASS |
| Módulo C — `KGeometricSIA` | 5/5 | ✅ PASS |
| **Total** | **16/16** | **✅ SISTEMA VALIDADO** |

---

## Dependencias

| Librería | Uso |
|---|---|
| `numpy` ≥ 1.21 | Operaciones matriciales sobre la TPM |
| `scipy` ≥ 1.7 | Distancia EMD (Wasserstein) |
| `colorama` ≥ 0.4 | Colores en la interfaz de terminal |

Las librerías `pyemd`, `pyinstrument`, `pyttsx3` son dependencias internas de los módulos base y **no requieren instalación manual** — se simulan automáticamente mediante stubs.

---

## Autor

**Mateo Mejía** · `mateo.mejia32303@ucaldas.edu.co`  
*Universidad de Caldas · Análisis y Diseño de Algoritmos · 2026-1*
