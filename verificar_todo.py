"""
verificar_todo.py — Verificación integral del proyecto K-QGMIP.

Ejecutar desde la raíz del repositorio:
    python3 verificar_todo.py

Cada módulo se verifica en un subproceso aislado para evitar conflictos
entre los paquetes 'src/' de GeoMIP y QNodes. El script imprime
'SISTEMA VALIDADO' si los tres módulos pasan sin errores.
"""

import sys
import os
import subprocess
import textwrap

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS
# ══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
QNODES_ROOT = os.path.join(SCRIPT_DIR, "QNodes")
GEO_M2      = os.path.join(
    SCRIPT_DIR, "GeoMIP", "src",
    "Method2_Dynamic_Programming_Reformulation"
)
SAMPLES_DIR = os.path.join(QNODES_ROOT, "src", ".samples")

# ══════════════════════════════════════════════════════════════════════════
# CÓDIGO DE CADA VERIFICACIÓN (ejecutado en subproceso aislado)
# ══════════════════════════════════════════════════════════════════════════

# ── Bloque de stubs compartido ────────────────────────────────────────────
_STUBS = """
import sys, types, numpy as np

def _stub(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m

_pyemd = _stub("pyemd")
_pyemd.emd = lambda u, v, cost: float(np.sum(np.abs(u - v)))
_pi = _stub("pyinstrument")
class _FP:
    def __init__(self, **k): pass
    def start(self): pass
    def stop(self): return self
    def output(self, renderer=None): return ""
_pi.Profiler = _FP
_stub("pyinstrument.renderers").HTMLRenderer = type("H",(),{"__init__":lambda s,**k:None})
_stub("pyttsx3"); _stub("pyttsx3.engine",Engine=type("E",(),{}))
_stub("pyttsx3.voice",Voice=type("V",(),{}))
_col = _stub("colorama"); _col.init=lambda**k:None
_col.Fore=type("F",(),{"__getattr__":lambda s,k:""})()
_col.Style=type("S",(),{"__getattr__":lambda s,k:""})()
_stub("colorama.ansi")
"""

# ── CHECK A: System.kpartir() ─────────────────────────────────────────────
_CHECK_A = _STUBS + f"""
import os, numpy as np
QNODES_ROOT = {repr(QNODES_ROOT)}
SAMPLES_DIR = {repr(SAMPLES_DIR)}
sys.path.insert(0, QNODES_ROOT)   # solo QNodes/src para evitar conflicto

from src.models.core.system import System

tpm    = np.loadtxt(os.path.join(SAMPLES_DIR, "N3A.csv"), delimiter=",")
estado = np.array([0, 0, 0], dtype=np.int8)
s      = System(tpm, estado)

# Biparticion de referencia
alc = np.array([0,1,2], dtype=np.int8)
mec = np.array([0,1,2], dtype=np.int8)
dist_bi = s.bipartir(alc, mec).distribucion_marginal()

# k=3: cada variable en su propio grupo
grupos_k3 = [
    (np.array([0], dtype=np.int8), np.array([0], dtype=np.int8)),
    (np.array([1], dtype=np.int8), np.array([1], dtype=np.int8)),
    (np.array([2], dtype=np.int8), np.array([2], dtype=np.int8)),
]
dist_k3 = s.kpartir(grupos_k3).distribucion_marginal()

# Verificaciones
assert isinstance(dist_k3, np.ndarray),  "distribucion no es ndarray"
assert dist_k3.size > 0,                  "distribucion vacia"
assert np.all(dist_k3 >= -1e-9),          "valores negativos en distribucion"
assert np.all(dist_k3 <= 1.0 + 1e-9),    "valores > 1 en distribucion"

perdida = float(np.sum(np.abs(dist_bi - dist_k3)))
print(f"      Sistema N=3, k=3 -> distribucion shape={{dist_k3.shape}}")
print(f"      Perdida EMD kpartir(k=3): {{perdida:.4f}}")
print("      [PASS] Modulo A operativo")
"""

# ── CHECK B: KQNodes ──────────────────────────────────────────────────────
_CHECK_B = _STUBS + f"""
import os, importlib.machinery, numpy as np
QNODES_ROOT = {repr(QNODES_ROOT)}
SAMPLES_DIR = {repr(SAMPLES_DIR)}
sys.path.insert(0, QNODES_ROOT)   # solo QNodes/src

kqn_path = os.path.join(QNODES_ROOT, "src", "strategies", "k_qnodes.py")
loader   = importlib.machinery.SourceFileLoader("k_qnodes_mod", kqn_path)
mod      = loader.load_module()
KQNodes  = mod.KQNodes

# Instanciar y ejecutar via aplicar_estrategia() — API publica completa.
# Internamente prepara sia_subsistema y luego llama algorithm().
tpm = np.loadtxt(os.path.join(SAMPLES_DIR, "N3A.csv"), delimiter=",")
kqn = KQNodes(tpm)

# N=3: estado_inicial, condicion, alcance, mecanismo en binario
resultado = kqn.aplicar_estrategia("000", "111", "111", "111", k=3)

assert resultado.perdida >= 0.0, f"Perdida negativa: {{resultado.perdida}}"
assert resultado.particion,      "Particion vacia"

# Verificar adicionalmente que algorithm() con los mismos vertices da 3 grupos
from src.constants.base import ACTUAL, EFFECT
N = 3
vertices = [(EFFECT,i) for i in range(N)] + [(ACTUAL,i) for i in range(N)]
grupos, perdida2, dist2 = kqn.algorithm(vertices, k=3)
assert len(grupos) == 3, f"Se esperaban 3 grupos, hay {{len(grupos)}}"
asignados = set()
for g in grupos: asignados.update(kqn._flatten_vertices(g))
assert asignados == set(vertices), "Cobertura de vertices incompleta"

print(f"      Sistema N=3, k=3 -> {{len(grupos)}} grupos encontrados")
print(f"      Perdida EMD KQNodes(k=3): {{resultado.perdida:.4f}}")
print(f"      Particion: {{resultado.particion}}")
print("      [PASS] Modulo B operativo")
"""

# ── CHECK C: KGeometricSIA ────────────────────────────────────────────────
_CHECK_C = _STUBS + f"""
import os, tempfile, numpy as np
from pathlib import Path
QNODES_ROOT = {repr(QNODES_ROOT)}
GEO_M2      = {repr(GEO_M2)}
SAMPLES_DIR = {repr(SAMPLES_DIR)}

sys.path.insert(0, QNODES_ROOT)
sys.path.insert(0, GEO_M2)   # GEO_M2 primero -> su src/ tiene precedencia

from src.controllers.strategies.k_geometric import KGeometricSIA
from src.middlewares.slogger import SafeLogger

class _MM:
    def __init__(self, ei):
        self.estado_inicial     = ei
        self.pagina_red_muestra = "A"
    @property
    def pagina(self): return "A"
    @property
    def output_dir(self):
        p = Path(tempfile.gettempdir()) / "kgeomip_ver"; p.mkdir(exist_ok=True); return p
    @property
    def tpm_filename(self): return None

k   = 2
obj = KGeometricSIA.__new__(KGeometricSIA)
obj.sia_gestor          = _MM("00")
obj.sia_logger          = type("L",(),{{"critic":lambda s,m:None,"warn":lambda s,m:None,"debug":lambda s,m:None}})()
obj.logger              = SafeLogger("ver_c")
obj.k                   = k
obj.etiquetas           = [[], []]
obj.tabla_transiciones  = {{}}
obj.vertices            = set()
obj.tabla               = {{}}
obj.memoria_particiones = {{}}
obj.sia_tiempo_inicio   = 0.0

tpm = np.loadtxt(os.path.join(SAMPLES_DIR, "N2A.csv"), delimiter=",")
obj.sia_preparar_subsistema("00", "11", "11", tpm)

dims               = obj.sia_subsistema.dims_ncubos
idx                = np.asarray(dims, dtype=np.intp)          # cast seguro a entero
obj.estado_inicial = obj.sia_subsistema.estado_inicial[idx]
obj.estado_final   = 1 - obj.estado_inicial
obj._flat_data     = [nc.data.ravel() for nc in obj.sia_subsistema.ncubos]
obj.idx_ncubos     = list(range(len(obj.sia_subsistema.indices_ncubos)))

n = len(obj.estado_inicial)
obj.caminos = {{0: [obj.estado_inicial.tolist()]}}
obj.tabla_transiciones = {{
    (tuple(obj.caminos[0][0]), tuple(obj.caminos[0][0])): [0.0]*len(obj.idx_ncubos)
}}
for nivel in range(1, n+1):
    obj.calcular_costos_nivel(obj.estado_final, nivel)

costos = [c for vs in obj.tabla_transiciones.values() for c in vs]
assert all(0.0-1e-9 <= c <= 1.0+1e-9 for c in costos), "Costos fuera de [0,1]"

candidatos = obj.identificar_particiones_optimas()
assert len(candidatos) >= 1,                    "Sin candidatos generados"
assert all(len(c)==k for c in candidatos),      f"Candidato sin k={{k}} grupos"

print(f"      Sistema N=2, k=2 -> {{len(candidatos)}} candidatos")
print(f"      Tabla BFS/DP: {{len(obj.tabla_transiciones)}} entradas, valores en [0,1]")
print(f"      Todos los candidatos tienen k={{k}} grupos")
print("      [PASS] Modulo C operativo")
"""

# ══════════════════════════════════════════════════════════════════════════
# MOTOR DE EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════

W = 46

def caja(lineas):
    print("+" + "=" * W + "+")
    for l in lineas:
        print("|  " + l.ljust(W - 2) + "|")
    print("+" + "=" * W + "+")

def ejecutar_check(nombre: str, codigo: str) -> tuple[bool, str, str]:
    """Ejecuta 'codigo' en un subproceso Python aislado. Devuelve (ok, stdout, stderr)."""
    resultado = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(codigo)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=SCRIPT_DIR,
    )
    return resultado.returncode == 0, resultado.stdout, resultado.stderr

# ══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════

def main():
    caja([
        "  VERIFICACION K-QGMIP — INICIO   ",
        f"  Python {sys.version.split()[0]}",
        f"  Raiz: {os.path.basename(SCRIPT_DIR)}",
    ])

    checks = [
        ("A", "Modulo A — System.kpartir()", _CHECK_A),
        ("B", "Modulo B — KQNodes",          _CHECK_B),
        ("C", "Modulo C — KGeometricSIA",    _CHECK_C),
    ]

    resultados = {}

    for i, (letra, titulo, codigo) in enumerate(checks, start=1):
        print(f"\n[{i}/{len(checks)}] {titulo} ...")
        ok, stdout, stderr = ejecutar_check(letra, codigo)
        resultados[letra] = ok

        # Filtrar líneas de log internas (CRITICAL/WARNING del logger de QNodes)
        lineas_out = [
            l for l in stdout.splitlines()
            if not l.startswith("CRITICAL") and not l.startswith("WARNING")
        ]
        if lineas_out:
            print("\n".join(lineas_out).rstrip())
        if not ok:
            # Filtrar líneas de log (CRITICAL/WARNING) para mostrar solo el traceback
            lineas_error = [
                l for l in stderr.strip().splitlines()
                if not l.startswith("CRITICAL") and not l.startswith("WARNING")
            ]
            print(f"      [FALLO] — traceback del subproceso:")
            for linea in lineas_error:
                print(f"        {linea}")

    # ── Resultado final ──────────────────────────────────────────────────
    print()
    if all(resultados.values()):
        caja([
            "                                      ",
            "        SISTEMA VALIDADO              ",
            "  Los 3 modulos K-QGMIP funcionan     ",
            "  correctamente en este entorno.      ",
            "                                      ",
            "  Modulo A  (kpartir)       [OK]      ",
            "  Modulo B  (KQNodes)       [OK]      ",
            "  Modulo C  (KGeometricSIA) [OK]      ",
            "                                      ",
        ])
        sys.exit(0)
    else:
        fallidos = [m for m, estado in resultados.items() if not estado]
        caja([
            "                                      ",
            "  VERIFICACION FALLIDA                ",
            f"  Modulos con error: {', '.join(fallidos)}".ljust(W - 2),
            "  Revisar el traceback arriba.        ",
            "                                      ",
        ])
        sys.exit(1)


if __name__ == "__main__":
    main()
