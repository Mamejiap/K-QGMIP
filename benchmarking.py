"""
benchmarking.py — Módulo de benchmarking comparativo para K-QGMIP.

Uso standalone:
    python benchmarking.py

Uso desde otro script:
    from benchmarking import comparar_algoritmos, graficar_comparativa, graficar_escalabilidad
    resultados = comparar_algoritmos("QNodes/src/.samples/N4D.csv", "0000", k=2)
    graficar_comparativa(resultados, output_dir="outputs/comparativas")
    graficar_escalabilidad(output_dir="outputs/comparativas", k=2)

Protocolo de subproceso (heredado de interfaz_kqmip.py):
    RESULTADO|{"perdida": float, "tiempo": float, "algoritmo": str, "particion": str}
    ERROR|<mensaje>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # backend sin pantalla (apto para servidores y CI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# RUTAS DEL PROYECTO
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
QNODES_ROOT = os.path.join(SCRIPT_DIR, "QNodes")
GEO_M2      = os.path.join(
    SCRIPT_DIR, "GeoMIP", "src",
    "Method2_Dynamic_Programming_Reformulation"
)

# Carpeta de muestras (fuente primaria)
_SAMPLES_QNODES = os.path.join(QNODES_ROOT, "src", ".samples")
_SAMPLES_GEO    = os.path.join(SCRIPT_DIR, "GeoMIP", "data", "samples")

# ══════════════════════════════════════════════════════════════════════════════
# STUBS DE DEPENDENCIAS OPCIONALES (mismo que en interfaz_kqmip.py)
# ══════════════════════════════════════════════════════════════════════════════

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


def _codigo_kqnodes(csv_path: str, estado: str,
                    condicion: str, alcance: str, mecanismo: str, k: int) -> str:
    return _STUBS + f"""
import os, importlib.machinery, time, json
QNODES_ROOT = {QNODES_ROOT!r}
sys.path.insert(0, QNODES_ROOT)

from src.middlewares.slogger import SafeLogger
SafeLogger.critic = lambda s, m: None   # silenciar logs en benchmark

import importlib.machinery
loader = importlib.machinery.SourceFileLoader(
    "k_qnodes_mod",
    os.path.join(QNODES_ROOT, "src", "strategies", "k_qnodes.py")
)
mod    = loader.load_module()
KQNodes = mod.KQNodes

try:
    tpm = np.loadtxt({csv_path!r}, delimiter=",")
    kqn = KQNodes(tpm)
    t0  = time.perf_counter()
    res = kqn.aplicar_estrategia({estado!r}, {condicion!r}, {alcance!r}, {mecanismo!r}, k={k})
    t1  = time.perf_counter()
    print("RESULTADO|" + json.dumps({{
        "perdida":   float(res.perdida),
        "particion": str(res.particion),
        "tiempo":    float(t1 - t0),
        "algoritmo": "KQNodes"
    }}), flush=True)
except Exception as exc:
    import traceback
    print("ERROR|" + str(exc), flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""


def _codigo_kgeomip(csv_path: str, estado: str,
                    condicion: str, alcance: str, mecanismo: str, k: int) -> str:
    return _STUBS + f"""
import os, tempfile, time, json
from pathlib import Path
QNODES_ROOT = {QNODES_ROOT!r}
GEO_M2      = {GEO_M2!r}
sys.path.insert(0, QNODES_ROOT)
sys.path.insert(0, GEO_M2)

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
        p = Path(tempfile.gettempdir()) / "kgeomip_bench"; p.mkdir(exist_ok=True); return p
    @property
    def tpm_filename(self): return None

try:
    tpm = np.loadtxt({csv_path!r}, delimiter=",")
    N   = tpm.shape[1]

    obj = KGeometricSIA.__new__(KGeometricSIA)
    obj.sia_gestor          = _MM({estado!r})
    obj.sia_logger          = type("L",(),{{"critic":lambda s,m:None,"warn":lambda s,m:None,"debug":lambda s,m:None}})()
    obj.logger              = SafeLogger("kgeomip_bench")
    obj.k                   = {k}
    obj.etiquetas           = [[], []]
    obj.tabla_transiciones  = {{}}
    obj.vertices            = set()
    obj.tabla               = {{}}
    obj.memoria_particiones = {{}}
    obj.sia_tiempo_inicio   = 0.0

    t0  = time.perf_counter()
    res = obj.aplicar_estrategia({condicion!r}, {alcance!r}, {mecanismo!r}, tpm, k={k})
    t1  = time.perf_counter()

    print("RESULTADO|" + json.dumps({{
        "perdida":   float(res.perdida),
        "particion": str(res.particion),
        "tiempo":    float(t1 - t0),
        "algoritmo": "KGeometricSIA"
    }}), flush=True)
except Exception as exc:
    import traceback
    print("ERROR|" + str(exc), flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""


# ══════════════════════════════════════════════════════════════════════════════
# EJECUTOR SILENCIOSO CON TIMEOUT
# ══════════════════════════════════════════════════════════════════════════════

TIMEOUT_SEG = 120   # máximo 2 min por algoritmo en benchmark


def _ejecutar_silencioso(codigo: str, timeout: int = TIMEOUT_SEG) -> Optional[dict]:
    """
    Lanza el código en un subproceso, captura el RESULTADO sin mostrarlo
    en pantalla. Devuelve el dict resultado o None si hubo error/timeout.
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(codigo)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=SCRIPT_DIR,
        )
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    except Exception:
        return None

    for linea in stdout.splitlines():
        if linea.startswith("RESULTADO|"):
            try:
                return json.loads(linea[10:])
            except json.JSONDecodeError:
                return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA — CAPA DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

def comparar_algoritmos(
    csv_path: str,
    estado: Optional[str] = None,
    k: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta KQNodes y KGeometricSIA sobre el mismo CSV y k.

    Parámetros
    ----------
    csv_path : ruta al archivo CSV (TPM, 2^N filas × N columnas).
    estado   : cadena binaria de N bits. Si None, se usa '0'*N.
    k        : número de particiones (2 ≤ k ≤ N).
    verbose  : imprime progreso en terminal si True.

    Retorna
    -------
    dict con claves:
        "csv", "N", "k", "estado",
        "kqnodes"    → {"perdida", "tiempo", "particion"} o None,
        "kgeomip"    → {"perdida", "tiempo", "particion"} o None,
        "ganador_emd"     → "KQNodes" | "KGeometricSIA" | "Empate" | "Indeterminado",
        "ganador_tiempo"  → idem,
        "timestamp"
    """
    tpm = np.loadtxt(csv_path, delimiter=",")
    N   = tpm.shape[1]
    if estado is None:
        estado = "0" * N
    condicion = alcance = mecanismo = "1" * N

    if verbose:
        print(f"  📊  Benchmarking  CSV={os.path.basename(csv_path)}  N={N}  k={k}")
        print(f"  ▶  Ejecutando KQNodes …", end=" ", flush=True)

    t_wall_q0 = time.perf_counter()
    res_q = _ejecutar_silencioso(
        _codigo_kqnodes(csv_path, estado, condicion, alcance, mecanismo, k)
    )
    t_wall_q1 = time.perf_counter()

    if verbose:
        _reportar_inline(res_q, t_wall_q1 - t_wall_q0)
        print(f"  ▶  Ejecutando KGeometricSIA …", end=" ", flush=True)

    t_wall_g0 = time.perf_counter()
    res_g = _ejecutar_silencioso(
        _codigo_kgeomip(csv_path, estado, condicion, alcance, mecanismo, k)
    )
    t_wall_g1 = time.perf_counter()

    if verbose:
        _reportar_inline(res_g, t_wall_g1 - t_wall_g0)

    # ── Determinar ganadores ─────────────────────────────────────────────────
    gan_emd    = _ganador(res_q, res_g, "perdida", menor_es_mejor=True)
    gan_tiempo = _ganador(res_q, res_g, "tiempo",  menor_es_mejor=True)

    return {
        "csv":           csv_path,
        "N":             N,
        "k":             k,
        "estado":        estado,
        "kqnodes":       res_q,
        "kgeomip":       res_g,
        "ganador_emd":   gan_emd,
        "ganador_tiempo": gan_tiempo,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _reportar_inline(res: Optional[dict], wall: float):
    if res is None:
        print(f"TIMEOUT / ERROR ({wall:.1f}s)")
    else:
        print(f"EMD={res['perdida']:.6f}  t={res['tiempo']:.3f}s  (wall={wall:.3f}s)")


def _ganador(
    res_q: Optional[dict],
    res_g: Optional[dict],
    campo: str,
    menor_es_mejor: bool,
) -> str:
    if res_q is None and res_g is None:
        return "Indeterminado"
    if res_q is None:
        return "KGeometricSIA"
    if res_g is None:
        return "KQNodes"
    vq, vg = res_q[campo], res_g[campo]
    if abs(vq - vg) < 1e-9:
        return "Empate"
    if menor_es_mejor:
        return "KQNodes" if vq < vg else "KGeometricSIA"
    return "KQNodes" if vq > vg else "KGeometricSIA"


# ══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA — GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

# Paleta de colores consistente
_COLOR_Q   = "#2E75B6"   # azul  → KQNodes
_COLOR_G   = "#37A84E"   # verde → KGeometricSIA
_COLOR_FND = "#F2F2F2"   # gris claro para fondo de panel
_ALPHA_BAR = 0.85


def graficar_comparativa(
    resultados: dict,
    output_dir: str = "outputs/comparativas",
    mostrar: bool = False,
) -> str:
    """
    Genera un panel de dos gráficas de barras:
      · Izquierda : Calidad de la solución (EMD — menor es mejor)
      · Derecha   : Eficiencia temporal (segundos — menor es mejor)

    Guarda el PNG en output_dir y devuelve la ruta del archivo.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_nombre = os.path.basename(resultados["csv"])
    N, k       = resultados["N"], resultados["k"]
    res_q      = resultados["kqnodes"]
    res_g      = resultados["kgeomip"]

    # ── Extraer valores (None → NaN para no romper el gráfico) ──────────────
    emd_q  = res_q["perdida"] if res_q else float("nan")
    emd_g  = res_g["perdida"] if res_g else float("nan")
    tiem_q = res_q["tiempo"]  if res_q else float("nan")
    tiem_g = res_g["tiempo"]  if res_g else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("white")

    _panel_barras(
        ax=ax1,
        valores=[emd_q, emd_g],
        etiquetas=["KQNodes", "KGeometricSIA"],
        colores=[_COLOR_Q, _COLOR_G],
        titulo="Calidad de la Solución",
        ylabel="Pérdida EMD (menor = mejor)",
        nota_ganador=resultados["ganador_emd"],
        formato=".6f",
    )

    _panel_barras(
        ax=ax2,
        valores=[tiem_q, tiem_g],
        etiquetas=["KQNodes", "KGeometricSIA"],
        colores=[_COLOR_Q, _COLOR_G],
        titulo="Eficiencia Temporal",
        ylabel="Tiempo de ejecución (segundos)",
        nota_ganador=resultados["ganador_tiempo"],
        formato=".3f",
        unidad="s",
    )

    titulo_fig = (
        f"Comparativa KQNodes vs KGeometricSIA\n"
        f"Archivo: {csv_nombre}  ·  N={N} variables  ·  k={k} particiones"
    )
    fig.suptitle(titulo_fig, fontsize=12, fontweight="bold", y=1.01)

    # Leyenda global
    parche_q = mpatches.Patch(color=_COLOR_Q, label="KQNodes")
    parche_g = mpatches.Patch(color=_COLOR_G, label="KGeometricSIA")
    fig.legend(
        handles=[parche_q, parche_g],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nombre_base = os.path.splitext(csv_nombre)[0]
    ruta_png = os.path.join(
        output_dir, f"comparativa_{nombre_base}_k{k}_{timestamp}.png"
    )
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)
    return ruta_png


def _panel_barras(
    ax, valores, etiquetas, colores, titulo, ylabel,
    nota_ganador="", formato=".4f", unidad=""
):
    """Dibuja un gráfico de barras en el Axes dado."""
    x     = np.arange(len(etiquetas))
    bars  = ax.bar(x, valores, color=colores, alpha=_ALPHA_BAR,
                   width=0.5, edgecolor="white", linewidth=1.2)

    # Etiquetas de valor encima de cada barra
    for bar, val in zip(bars, valores):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(valores) * 0.02,
                f"{val:{formato}}{unidad}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(v for v in valores if not np.isnan(v)) * 0.05 if any(not np.isnan(v) for v in valores) else 0.01,
                "N/D",
                ha="center", va="bottom", fontsize=9, color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_facecolor(_COLOR_FND)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Nota de ganador
    if nota_ganador and nota_ganador not in ("Indeterminado",):
        color_nota = "#1a5c2a" if "Geometric" in nota_ganador else "#1a3c6e"
        if nota_ganador == "Empate":
            color_nota = "#555555"
        ax.set_xlabel(
            f"Ganador: {nota_ganador}",
            fontsize=9, color=color_nota, fontweight="bold",
        )


def graficar_escalabilidad(
    output_dir: str = "outputs/comparativas",
    k: int = 2,
    archivos_n: Optional[dict] = None,
    verbose: bool = True,
    mostrar: bool = False,
) -> str:
    """
    Genera un gráfico de líneas de escalabilidad:
      Eje X → número de variables N
      Eje Y → tiempo de ejecución (segundos)
      Una línea por algoritmo.

    Parámetros
    ----------
    output_dir  : carpeta donde guardar el PNG.
    k           : valor de k fijo para todas las ejecuciones.
    archivos_n  : dict {N: ruta_csv}. Si None, se detectan automáticamente
                  usando los archivos N{n}A.csv disponibles (N=2..6).
    verbose     : mostrar progreso.
    mostrar     : abrir la ventana de matplotlib.

    Retorna
    -------
    Ruta del PNG generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Auto-detectar CSVs por N ─────────────────────────────────────────────
    if archivos_n is None:
        archivos_n = {}
        for n in range(2, 7):           # N=2..6 (N≥7 puede ser muy lento)
            for variante in ("A", "B", "C", "D"):
                nombre = f"N{n}{variante}.csv"
                for carpeta in (_SAMPLES_QNODES, _SAMPLES_GEO):
                    ruta = os.path.join(carpeta, nombre)
                    if os.path.isfile(ruta):
                        # Validar que sea una TPM bien formada
                        try:
                            t = np.loadtxt(ruta, delimiter=",")
                            if t.ndim == 2 and t.shape[0] == 2 ** t.shape[1] and t.shape[1] == n:
                                if n not in archivos_n:
                                    archivos_n[n] = ruta
                        except Exception:
                            pass
                    if n in archivos_n:
                        break

    if not archivos_n:
        raise FileNotFoundError(
            "No se encontraron archivos CSV válidos para el análisis de escalabilidad."
        )

    ns_sorted  = sorted(archivos_n.keys())
    tiempos_q  = []
    tiempos_g  = []
    ns_validos = []

    for n in ns_sorted:
        csv_path = archivos_n[n]
        estado   = "0" * n
        cond     = "1" * n
        k_actual = min(k, n)   # k no puede superar N

        if verbose:
            print(f"  N={n}  CSV={os.path.basename(csv_path)}  k={k_actual} …", end="  ", flush=True)

        res_q = _ejecutar_silencioso(
            _codigo_kqnodes(csv_path, estado, cond, cond, cond, k_actual)
        )
        res_g = _ejecutar_silencioso(
            _codigo_kgeomip(csv_path, estado, cond, cond, cond, k_actual)
        )

        tq = res_q["tiempo"] if res_q else float("nan")
        tg = res_g["tiempo"] if res_g else float("nan")

        if verbose:
            tq_str = f"{tq:.3f}s" if not np.isnan(tq) else "N/D"
            tg_str = f"{tg:.3f}s" if not np.isnan(tg) else "N/D"
            print(f"KQNodes={tq_str}  KGeometricSIA={tg_str}")

        if not (np.isnan(tq) and np.isnan(tg)):
            tiempos_q.append(tq)
            tiempos_g.append(tg)
            ns_validos.append(n)

    # ── Gráfica ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    _trazar_linea(ax, ns_validos, tiempos_q, _COLOR_Q, "KQNodes",       "o")
    _trazar_linea(ax, ns_validos, tiempos_g, _COLOR_G, "KGeometricSIA", "s")

    ax.set_xlabel("Número de variables (N)", fontsize=11)
    ax.set_ylabel("Tiempo de ejecución (segundos)", fontsize=11)
    ax.set_title(
        f"Escalabilidad: KQNodes vs KGeometricSIA  (k={k} particiones)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(ns_validos)
    ax.set_xticklabels([f"N={n}" for n in ns_validos], fontsize=9)
    ax.set_facecolor(_COLOR_FND)
    ax.grid(linestyle="--", alpha=0.5, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)

    # Anotación de complejidad
    ax.annotate(
        "Complejidad KQNodes: O(S(N,k)·N)\n"
        "Complejidad KGeomSIA: O(2^N · k)",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7),
    )

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ruta_png  = os.path.join(output_dir, f"escalabilidad_k{k}_{timestamp}.png")
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)
    return ruta_png


def _trazar_linea(ax, xs, ys, color, label, marker):
    ys_arr = np.array(ys, dtype=float)
    xs_arr = np.array(xs)
    validos = ~np.isnan(ys_arr)
    if validos.sum() < 1:
        return
    ax.plot(
        xs_arr[validos], ys_arr[validos],
        color=color, label=label, marker=marker,
        linewidth=2, markersize=7, zorder=3,
    )
    ax.fill_between(
        xs_arr[validos], 0, ys_arr[validos],
        color=color, alpha=0.10,
    )
    # Etiquetas de valor en cada punto
    for x, y in zip(xs_arr[validos], ys_arr[validos]):
        ax.annotate(
            f"{y:.3f}s",
            (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color=color, fontweight="bold",
        )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE RESUMEN TEXTUAL
# ══════════════════════════════════════════════════════════════════════════════

def resumen_textual(resultados: dict) -> str:
    """
    Genera un string de resumen legible por humanos del resultado de
    comparar_algoritmos(). Apto para imprimirse en la interfaz o
    incluirse en el Manual de Usuario.
    """
    csv_n  = os.path.basename(resultados["csv"])
    N, k   = resultados["N"], resultados["k"]
    res_q  = resultados["kqnodes"]
    res_g  = resultados["kgeomip"]
    g_emd  = resultados["ganador_emd"]
    g_tiem = resultados["ganador_tiempo"]

    lineas = [
        "=" * 62,
        "  RESUMEN COMPARATIVO K-QGMIP",
        "=" * 62,
        f"  Archivo  : {csv_n}",
        f"  N (vars) : {N}   |   k (grupos) : {k}",
        f"  Fecha    : {resultados['timestamp']}",
        "-" * 62,
    ]

    for nombre, res in [("KQNodes", res_q), ("KGeometricSIA", res_g)]:
        if res is None:
            lineas.append(f"  {nombre:<16}: TIMEOUT o ERROR")
        else:
            lineas.append(
                f"  {nombre:<16}: EMD={res['perdida']:.6f}  "
                f"tiempo={res['tiempo']:.4f}s"
            )

    lineas += [
        "-" * 62,
        f"  Mejor calidad (EMD ↓)   : {g_emd}",
        f"  Más rápido (tiempo ↓)   : {g_tiem}",
    ]

    # Conclusión interpretada
    if g_emd == g_tiem and g_emd not in ("Empate", "Indeterminado"):
        lineas.append(
            f"\n  ► {g_emd} es SUPERIOR en ambas métricas para este caso."
        )
    elif g_emd == "Empate" and g_tiem not in ("Indeterminado",):
        lineas.append(
            f"\n  ► Ambos encuentran la MISMA calidad de partición. "
            f"{g_tiem} lo hace más rápido."
        )
    elif g_emd != g_tiem and "Indeterminado" not in (g_emd, g_tiem):
        lineas.append(
            f"\n  ► {g_emd} gana en precisión; {g_tiem} gana en velocidad."
            f"\n    Elige según si priorizas exactitud o rendimiento."
        )
    else:
        lineas.append("\n  ► Resultado parcial (uno o ambos algoritmos no completaron).")

    lineas.append("=" * 62)
    return "\n".join(lineas)


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

def _main():
    """
    Ejecución standalone: benchmarking interactivo por consola.
    Permite elegir CSV, k, y genera todas las gráficas automáticamente.
    """
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)

    AZUL  = Fore.CYAN + Style.BRIGHT
    VERDE = Fore.GREEN + Style.BRIGHT
    RESET = Style.RESET_ALL

    print(AZUL + "\n  ╔══════════════════════════════════════════════╗")
    print(AZUL + "  ║   K-QGMIP — Benchmarking Comparativo        ║")
    print(AZUL + "  ╚══════════════════════════════════════════════╝\n" + RESET)

    # Listar CSVs disponibles
    muestras = []
    vistos   = set()
    for carpeta in (_SAMPLES_QNODES, _SAMPLES_GEO):
        if not os.path.isdir(carpeta):
            continue
        for f in sorted(os.listdir(carpeta)):
            if not f.endswith(".csv") or f in vistos:
                continue
            ruta = os.path.join(carpeta, f)
            try:
                t = np.loadtxt(ruta, delimiter=",")
                if t.ndim == 2 and t.shape[0] == 2 ** t.shape[1]:
                    muestras.append((ruta, f, t.shape[1]))
                    vistos.add(f)
            except Exception:
                pass

    print(f"  {'N':<4} {'Archivo':<14} {'Variables':>8}")
    print("  " + "-" * 30)
    for i, (ruta, nombre, n) in enumerate(muestras, 1):
        print(f"  {AZUL}{i:<4}{RESET} {nombre:<14} {n:>8} vars")

    print()
    resp = input(AZUL + "  → " + RESET + f"Elige CSV [1-{len(muestras)}]: ").strip()
    try:
        idx = int(resp) - 1
        csv_path, csv_nombre, N = muestras[idx]
    except (ValueError, IndexError):
        print("  Opción inválida."); sys.exit(1)

    resp_k = input(AZUL + "  → " + RESET + f"Número de grupos k (2-{N}): ").strip()
    try:
        k = int(resp_k)
        if not (2 <= k <= N):
            raise ValueError
    except ValueError:
        print(f"  k debe estar entre 2 y {N}."); sys.exit(1)

    output_dir = os.path.join(SCRIPT_DIR, "outputs", "comparativas")
    print()

    # ── Comparativa directa ──────────────────────────────────────────────────
    print(AZUL + "  [1/3] Comparativa directa …\n" + RESET)
    resultados = comparar_algoritmos(csv_path, k=k, verbose=True)
    print()
    print(resumen_textual(resultados))

    print()
    print(AZUL + "  [2/3] Generando gráfica de barras …" + RESET)
    ruta1 = graficar_comparativa(resultados, output_dir=output_dir)
    print(VERDE + f"  ✓ Guardada: {ruta1}" + RESET)

    # ── Escalabilidad ────────────────────────────────────────────────────────
    print()
    print(AZUL + "  [3/3] Generando gráfica de escalabilidad …" + RESET)
    ruta2 = graficar_escalabilidad(output_dir=output_dir, k=2, verbose=True)
    print(VERDE + f"  ✓ Guardada: {ruta2}" + RESET)

    print()
    print(VERDE + f"  Gráficas guardadas en: {output_dir}" + RESET)
    print()


if __name__ == "__main__":
    _main()
