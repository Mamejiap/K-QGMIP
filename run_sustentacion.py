"""
run_sustentacion.py - Ejecucion rapida para sustentacion K-QGMIP
Corre KQNodes (todas las redes) y KGeomSIA (N<=10) para k=2,3,4,5
sin interaccion de usuario. Imprime tabla lista para transcribir.

USO:
    cd K-QGMIP_Final
    python run_sustentacion.py

    Solo una red especifica:
    python run_sustentacion.py --N 5

    Sin KGeomSIA (mas rapido):
    python run_sustentacion.py --sin-geo
"""

import os, sys, json, subprocess, time, argparse
from pathlib import Path
import numpy as np

# --- Rutas ---
SCRIPT_DIR  = Path(__file__).parent.resolve()
SAMPLES     = SCRIPT_DIR / "QNodes" / "src" / ".samples"
QNODES_ROOT = str(SCRIPT_DIR / "QNodes")
GEO_M2      = str(SCRIPT_DIR / "GeoMIP" / "src" /
                  "Method2_Dynamic_Programming_Reformulation")

# --- Redes para la sustentacion ---
REDES = [
    {"N":  3, "csv": str(SAMPLES / "N3A.csv"),  "id": "N3A",  "estado": "100"},
    {"N":  5, "csv": str(SAMPLES / "N5A.csv"),  "id": "N5A",  "estado": "10000"},
    {"N": 10, "csv": str(SAMPLES / "N10A.csv"), "id": "N10A", "estado": "1000000000"},
    {"N": 15, "csv": str(SAMPLES / "N15B.csv"), "id": "N15B", "estado": "1"+"0"*14},
    {"N": 20, "csv": None,                      "id": "N20-SINT", "estado": "1"+"0"*19},
    {"N": 22, "csv": None,                      "id": "N22-SINT", "estado": "1"+"0"*21},
]

K_VALORES = [2, 3, 4, 5]
N_MAX_GEO  = 10    # KGeomSIA solo para N <= 10
TIMEOUT_QN = 300
TIMEOUT_GE = 90

# --- STUBS (mismo patron que interfaz_kqmip.py) ---
_STUBS = """
import sys, types, numpy as np
def _stub(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m
_pyemd = _stub("pyemd"); _pyemd.emd = lambda u,v,c: float(np.sum(np.abs(u-v)))
_pi = _stub("pyinstrument")
class _FP:
    def __init__(self,**k): pass
    def start(self): pass
    def stop(self): return self
    def output(self, renderer=None): return ""
_pi.Profiler = _FP
_stub("pyinstrument.renderers").HTMLRenderer = type("H",(),{"__init__":lambda s,**k:None})
_stub("pyttsx3"); _stub("pyttsx3.engine",Engine=type("E",(),{})); _stub("pyttsx3.voice",Voice=type("V",(),{}))
_col = _stub("colorama"); _col.init=lambda**k:None
_col.Fore=type("F",(),{"__getattr__":lambda s,k:""})(); _col.Style=type("S",(),{"__getattr__":lambda s,k:""})()
_stub("colorama.ansi")
"""


def _codigo_kqnodes(csv_path, estado, condicion, alcance, mecanismo, k,
                    N_sint=None):
    if csv_path and os.path.exists(csv_path):
        carga = f"tpm = np.loadtxt({csv_path!r}, delimiter=',')"
    else:
        N = N_sint
        carga = f"tpm = np.random.default_rng(42).random((2**{N}, {N}))"
    return _STUBS + f"""
import os, importlib.machinery, time, json
QNODES_ROOT = {QNODES_ROOT!r}
sys.path.insert(0, QNODES_ROOT)
from src.middlewares.slogger import SafeLogger
SafeLogger.critic = lambda self, msg: None
loader = importlib.machinery.SourceFileLoader(
    "k_qnodes_mod",
    os.path.join(QNODES_ROOT, "src", "strategies", "k_qnodes.py")
)
mod = loader.load_module(); KQNodes = mod.KQNodes
try:
    {carga}
    kqn = KQNodes(tpm)
    t0 = time.perf_counter()
    res = kqn.aplicar_estrategia({estado!r}, {condicion!r}, {alcance!r}, {mecanismo!r}, k={k})
    t1 = time.perf_counter()
    print("RESULTADO|" + json.dumps({{
        "perdida": float(res.perdida), "particion": str(res.particion),
        "tiempo": float(t1-t0), "algoritmo": "KQNodes"
    }}), flush=True)
except Exception as exc:
    import traceback; traceback.print_exc(file=sys.stderr)
    print("ERROR|" + str(exc), flush=True); sys.exit(1)
"""


def _codigo_kgeomip(csv_path, estado, condicion, alcance, mecanismo, k):
    return _STUBS + f"""
import os, tempfile, time, json
from pathlib import Path
QNODES_ROOT = {QNODES_ROOT!r}; GEO_M2 = {GEO_M2!r}
sys.path.insert(0, QNODES_ROOT); sys.path.insert(0, GEO_M2)
from src.controllers.strategies.k_geometric import KGeometricSIA
from src.middlewares.slogger import SafeLogger
class _MM:
    def __init__(self, ei):
        self.estado_inicial = ei; self.pagina_red_muestra = "A"
    @property
    def pagina(self): return "A"
    @property
    def output_dir(self):
        p = Path(tempfile.gettempdir()) / "kgeomip_sust"; p.mkdir(exist_ok=True); return p
    @property
    def tpm_filename(self): return None
try:
    tpm = np.loadtxt({csv_path!r}, delimiter=',')
    obj = KGeometricSIA.__new__(KGeometricSIA)
    obj.sia_gestor = _MM({estado!r})
    obj.sia_logger = type("L",(),{{"critic":lambda s,m:None,"warn":lambda s,m:None,"debug":lambda s,m:None}})()
    obj.logger = SafeLogger("kgeomip_sust"); obj.k = {k}
    obj.etiquetas = [[],[]]; obj.tabla_transiciones = {{}}; obj.vertices = set()
    obj.tabla = {{}}; obj.memoria_particiones = {{}}; obj.sia_tiempo_inicio = 0.0
    t0 = time.perf_counter()
    res = obj.aplicar_estrategia({condicion!r}, {alcance!r}, {mecanismo!r}, tpm, k={k})
    t1 = time.perf_counter()
    print("RESULTADO|" + json.dumps({{
        "perdida": float(res.perdida), "particion": str(res.particion),
        "tiempo": float(t1-t0), "algoritmo": "KGeometricSIA"
    }}), flush=True)
except Exception as exc:
    import traceback; traceback.print_exc(file=sys.stderr)
    print("ERROR|" + str(exc), flush=True); sys.exit(1)
"""


def ejecutar(codigo, timeout):
    try:
        p = subprocess.run(
            [sys.executable, "-c", codigo],
            capture_output=True, text=True, encoding="utf-8", timeout=timeout
        )
        for linea in p.stdout.splitlines():
            if linea.startswith("RESULTADO|"):
                return json.loads(linea[len("RESULTADO|"):])
        return None
    except subprocess.TimeoutExpired:
        return {"perdida": None, "particion": "TIMEOUT", "tiempo": None}
    except Exception as e:
        return {"perdida": None, "particion": f"ERROR: {e}", "tiempo": None}


def fmt_particion(s, ancho=60):
    """Acorta la particion para que quepa en pantalla."""
    s = str(s)
    return s[:ancho] + "..." if len(s) > ancho else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",       type=int, default=None,
                        help="Ejecutar solo la red de N nodos (3,5,10,15,20,22)")
    parser.add_argument("--sin-geo", action="store_true",
                        help="Omitir KGeomSIA completamente")
    args = parser.parse_args()

    redes = [r for r in REDES if args.N is None or r["N"] == args.N]
    if not redes:
        print(f"No hay red con N={args.N}. Opciones: 3, 5, 10, 15, 20, 22")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  K-QGMIP  -  Ejecucion de Sustentacion")
    print("=" * 72)

    resumen = []

    for red in redes:
        N       = red["N"]
        csv     = red["csv"]
        red_id  = red["id"]
        estado  = red["estado"]
        cond    = "1" * N
        alc     = "1" * N
        mec     = "1" * N
        sint    = (csv is None or not os.path.exists(csv or ""))

        print()
        print(f"  RED {red_id}  |  N={N}  |  {'TPM sintetica (seed=42)' if sint else 'CSV real'}")
        print("-" * 72)
        print(f"  {'k':<4}  {'Algoritmo':<14}  {'EMD (perdida)':<16}  {'Tiempo (s)':<12}  k-Particion")
        print("-" * 72)

        for k in K_VALORES:
            # -- KQNodes --
            if N > 20 and sint:
                res_qn = {"perdida": None, "particion": "N/A (N>20 sint.)", "tiempo": None}
            else:
                cod = _codigo_kqnodes(csv, estado, cond, alc, mec, k,
                                      N_sint=N if sint else None)
                t0 = time.perf_counter()
                res_qn = ejecutar(cod, TIMEOUT_QN) or \
                         {"perdida": None, "particion": "ERR", "tiempo": None}

            emd_qn  = f"{res_qn['perdida']:.6f}" if res_qn["perdida"] is not None else "---"
            t_qn    = f"{res_qn['tiempo']:.3f}s"  if res_qn["tiempo"]  is not None else "---"
            part_qn = fmt_particion(res_qn["particion"])
            print(f"  k={k}   {'KQNodes':<14}  {emd_qn:<16}  {t_qn:<12}  {part_qn}")

            # -- KGeomSIA --
            if args.sin_geo or sint or N > N_MAX_GEO:
                motivo = "--sin-geo" if args.sin_geo else ("TPM sint." if sint else f"N={N}>10")
                print(f"  k={k}   {'KGeomSIA':<14}  {'N/A':<16}  {'---':<12}  ({motivo})")
                res_geo = {"perdida": None, "particion": f"N/A ({motivo})", "tiempo": None}
            else:
                cod = _codigo_kgeomip(csv, estado, cond, alc, mec, k)
                res_geo = ejecutar(cod, TIMEOUT_GE) or \
                          {"perdida": None, "particion": "ERR", "tiempo": None}
                emd_g  = f"{res_geo['perdida']:.6f}" if res_geo["perdida"] is not None else "---"
                t_g    = f"{res_geo['tiempo']:.3f}s"  if res_geo["tiempo"]  is not None else "---"
                part_g = fmt_particion(res_geo["particion"])
                print(f"  k={k}   {'KGeomSIA':<14}  {emd_g:<16}  {t_g:<12}  {part_g}")

            resumen.append({
                "red": red_id, "N": N, "k": k,
                "qn_emd": res_qn["perdida"], "qn_t": res_qn["tiempo"],
                "qn_part": res_qn["particion"],
                "geo_emd": res_geo["perdida"], "geo_t": res_geo["tiempo"],
                "geo_part": res_geo["particion"],
            })

        print("-" * 72)

    # Tabla resumen compacta para transcribir al Word
    print()
    print("=" * 72)
    print("  RESUMEN PARA TRANSCRIBIR A LAS TABLAS DEL WORD")
    print("=" * 72)
    red_actual = None
    for r in resumen:
        if r["red"] != red_actual:
            red_actual = r["red"]
            print(f"\n  Red: {r['red']}  (N={r['N']})")
            print(f"  {'k':<3}  {'k-Particion KQNodes':<45}  {'EMD':>8}  {'t(s)':>8}")
            print("  " + "-" * 68)
        emd = f"{r['qn_emd']:.6f}" if r['qn_emd'] is not None else "N/A"
        t   = f"{r['qn_t']:.3f}"   if r['qn_t']   is not None else "N/A"
        pt  = fmt_particion(r["qn_part"], 45)
        print(f"  {r['k']:<3}  {pt:<45}  {emd:>8}  {t:>8}")

    print()
    print("  [!] Haz SCREENSHOT ahora: captura esta pantalla completa.")
    print("      Evidencia necesaria: comando ejecutado + tabla de resultados.")
    print()


if __name__ == "__main__":
    main()
