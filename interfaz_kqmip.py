"""
interfaz_kqmip.py — Interfaz interactiva del proyecto K-QGMIP.

Ejecutar desde la raíz del repositorio:
    python3 interfaz_kqmip.py

Permite cargar una TPM desde CSV, elegir el algoritmo (KQNodes o
KGeometricSIA), configurar k y el estado inicial, y ver el resultado
de la k-partición de mínima información (MIP) con progreso en tiempo real.
"""

import os
import sys
import json
import time
import subprocess
import textwrap
import math
import numpy as np
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTES DE RUTA
# ══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
QNODES_ROOT = os.path.join(SCRIPT_DIR, "QNodes")
GEO_M2      = os.path.join(
    SCRIPT_DIR, "GeoMIP", "src",
    "Method2_Dynamic_Programming_Reformulation"
)

AZUL   = Fore.CYAN + Style.BRIGHT
VERDE  = Fore.GREEN + Style.BRIGHT
ROJO   = Fore.RED + Style.BRIGHT
AMARILLO = Fore.YELLOW + Style.BRIGHT
GRIS   = Fore.WHITE + Style.DIM
RESET  = Style.RESET_ALL

ANCHO  = 64   # ancho de la interfaz

# ══════════════════════════════════════════════════════════════════════════
# HELPERS DE PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════

def limpiar():
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")

def linea(char="─", color=GRIS):
    print(color + char * ANCHO + RESET)

def banner():
    limpiar()
    print()
    print(AZUL + "╔" + "═" * (ANCHO - 2) + "╗")
    print(AZUL + "║" + " " * (ANCHO - 2) + "║")
    titulo = "K - Q G M I P"
    pad = (ANCHO - 2 - len(titulo)) // 2
    print(AZUL + "║" + " " * pad + titulo + " " * (ANCHO - 2 - pad - len(titulo)) + "║")
    sub = "Partición de Mínima Información  ·  k grupos"
    pad2 = (ANCHO - 2 - len(sub)) // 2
    print(AZUL + "║" + GRIS + " " * pad2 + sub + " " * (ANCHO - 2 - pad2 - len(sub)) + AZUL + "║")
    print(AZUL + "║" + " " * (ANCHO - 2) + "║")
    print(AZUL + "╚" + "═" * (ANCHO - 2) + "╝" + RESET)
    print()

def seccion(titulo_sec):
    print()
    linea("─")
    print(AZUL + f"  {titulo_sec}" + RESET)
    linea("─")

def ok_msg(msg):
    print(VERDE + "  ✓ " + RESET + msg)

def warn_msg(msg):
    print(AMARILLO + "  ⚠  " + RESET + msg)

def err_msg(msg):
    print(ROJO + "  ✗ " + RESET + msg)

def barra_progreso(paso, total, ancho=28):
    """Dibuja [████████░░░░] paso/total"""
    llenos = int(ancho * paso / total) if total > 0 else 0
    vacios = ancho - llenos
    bar = AZUL + "█" * llenos + GRIS + "░" * vacios + RESET
    pct = int(100 * paso / total) if total > 0 else 0
    return f"[{bar}] {pct:3d}%  ({paso}/{total})"

def tabla_resultado(perdida, particion, tiempo, algoritmo, k, csv_name):
    """Imprime la tabla de resultados final con borde ASCII."""
    linea("═", VERDE)
    print(VERDE + Style.BRIGHT + "  RESULTADO — Partición de Mínima Información" + RESET)
    linea("═", VERDE)
    filas = [
        ("Algoritmo",   algoritmo),
        ("CSV",         os.path.basename(csv_name)),
        ("k (grupos)",  str(k)),
        ("Pérdida EMD", f"{perdida:.6f}"),
        ("Tiempo",      f"{tiempo:.3f} s"),
    ]
    for etiq, val in filas:
        etiq_col = AZUL + f"  {etiq:<14}" + RESET
        print(f"{etiq_col}: {val}")
    linea("─", VERDE)
    print(AZUL + "  Partición:" + RESET)
    # Partir la partición en líneas si es muy larga
    for parte in particion.split("  "):
        print(f"    {AMARILLO}{parte.strip()}{RESET}")
    linea("═", VERDE)


# ══════════════════════════════════════════════════════════════════════════
# ENTRADA INTERACTIVA (loops con validación)
# ══════════════════════════════════════════════════════════════════════════

def pedir(prompt, color=AZUL):
    return input(color + "  → " + RESET + prompt + " ").strip()

def _escanear_muestras():
    """Lista todos los CSV validos en las carpetas de muestras de ambos proyectos."""
    directorios = [
        (os.path.join(QNODES_ROOT, "src", ".samples"),         "QNodes"),
        (os.path.join(SCRIPT_DIR, "GeoMIP", "data", "samples"), "GeoMIP"),
    ]
    encontrados = []
    vistos = set()
    for d, origen in directorios:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv") or f in vistos:
                continue
            vistos.add(f)
            ruta = os.path.join(d, f)
            try:
                t = np.loadtxt(ruta, delimiter=",", dtype=np.float64)
                if t.ndim == 2 and t.shape[0] == 2 ** t.shape[1]:
                    encontrados.append((ruta, f, t.shape[1], t.shape[0], origen))
            except Exception:
                pass
    return encontrados


def pedir_csv():
    """Muestra menu numerado de CSVs; permite seleccion por numero o ruta manual."""
    muestras = _escanear_muestras()

    while True:
        seccion("Seleccion de archivo TPM")

        # Cabecera de la tabla
        print(f"  {GRIS}{'N':<4} {'Archivo':<12} {'Variables':>10} {'Estados':>8}  {'Recomendado para'}{RESET}")
        linea(".", GRIS)

        for i, (ruta, nombre, N, filas, origen) in enumerate(muestras, start=1):
            rec = "KQNodes (grande)" if N > 6 else "KQNodes / KGeoMIP"
            num = AZUL + f"  {i:<3}" + RESET
            nom = AMARILLO + f"{nombre:<12}" + RESET
            print(f"{num} {nom} {N:>10} vars  {filas:>6} estados  {GRIS}{rec}{RESET}")

        print()
        total = len(muestras)
        resp = pedir(
            f"Elige un numero [1-{total}] o escribe la ruta completa de tu CSV:"
        )

        # -- Seleccion por numero ------------------------------------------
        if resp.isdigit():
            idx = int(resp) - 1
            if 0 <= idx < total:
                ruta, nombre, N, filas, _ = muestras[idx]
                tpm = np.loadtxt(ruta, delimiter=",", dtype=np.float64)
                ok_msg(f"Seleccionado: {nombre}  ({filas} estados x {N} variables)")
                return ruta, tpm, N
            else:
                err_msg(f"Numero fuera de rango. Elige entre 1 y {total}.")
                continue

        # -- Ruta personalizada --------------------------------------------
        if resp == "":
            err_msg(f"Escribe un numero (1-{total}) o la ruta de tu archivo CSV.")
            continue

        ruta = resp.strip('"').strip("'")
        if not os.path.isfile(ruta):
            err_msg(f"Archivo no encontrado: {ruta}")
            continue
        try:
            tpm = np.loadtxt(ruta, delimiter=",", dtype=np.float64)
            if tpm.ndim != 2:
                raise ValueError("Se esperaba una matriz 2D.")
            filas, cols = tpm.shape
            N = cols
            if filas != 2 ** N:
                raise ValueError(
                    f"La TPM debe tener 2^N filas. "
                    f"Encontrado: {filas} filas, {N} columnas."
                )
            ok_msg(f"TPM cargada: {filas} estados x {N} variables")
            return ruta, tpm, N
        except Exception as exc:
            err_msg(f"Error al leer el CSV: {exc}")


def vista_previa(tpm, N):
    """Muestra las primeras 5 filas y hasta 5 columnas de la TPM."""
    seccion("Vista previa de la TPM cargada")
    max_f = min(5, tpm.shape[0])
    max_c = min(5, N)
    cabecera = "  Estado  " + "".join(f"  Var{i:<2}" for i in range(max_c))
    if N > 5:
        cabecera += "  …"
    print(GRIS + cabecera + RESET)
    linea("·", GRIS)
    for i in range(max_f):
        estado_bin = format(i, f"0{N}b")
        fila = f"  |{estado_bin}| " + "".join(f"  {tpm[i, j]:.3f}" for j in range(max_c))
        if N > 5:
            fila += "  …"
        print(fila)
    if tpm.shape[0] > 5:
        print(GRIS + f"  … ({tpm.shape[0] - 5} filas más)" + RESET)
    print()
    resp = pedir("¿Es esta la TPM correcta? [S/n]:")
    return resp.lower() not in ("n", "no")

def pedir_estado_inicial(N):
    """Pide estado inicial binario de longitud N."""
    while True:
        s = pedir(f"Estado inicial (cadena binaria de {N} bits, ENTER = {'0'*N}):")
        if s == "":
            s = "0" * N
        if len(s) == N and all(c in "01" for c in s):
            return s
        err_msg(f"Debe ser una cadena de exactamente {N} caracteres '0' o '1'.")

def pedir_k(N):
    """Pide k con validación 2 ≤ k ≤ N."""
    while True:
        raw = pedir(f"Número de grupos k (2 ≤ k ≤ {N}):")
        try:
            k = int(raw)
            if 2 <= k <= N:
                return k
            err_msg(f"k debe estar entre 2 y {N} (número de variables).")
        except ValueError:
            err_msg("Introduce un número entero.")

def pedir_algoritmo():
    """Muestra menú y pide elección 1/2."""
    seccion("Selección de algoritmo")
    print(f"  {AZUL}1{RESET}  KQNodes      — heurístico voraz (más rápido, robusto)")
    print(f"  {AZUL}2{RESET}  KGeometricSIA — geométrico BFS/DP (más preciso, N ≤ 6)")
    print()
    while True:
        op = pedir("Elige algoritmo [1/2]:")
        if op in ("1", "2"):
            return int(op)
        err_msg("Opción no válida. Escribe 1 o 2.")


# ══════════════════════════════════════════════════════════════════════════
# CÓDIGO DE SUBPROCESOS (embebido como strings)
# Protocolo de salida:
#   FASE|n|total|label     → progreso de fases (KQNodes)
#   PASO|message           → paso general (KGeometricSIA)
#   RESULTADO|{json}       → resultado final
#   ERROR|message          → error (el subproceso sale con código 1)
# ══════════════════════════════════════════════════════════════════════════

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


def _codigo_kqnodes(csv_path, estado, condicion, alcance, mecanismo, k):
    return _STUBS + f"""
import os, importlib.machinery, time, json
QNODES_ROOT = {QNODES_ROOT!r}
sys.path.insert(0, QNODES_ROOT)

# ── Monkey-patch SafeLogger: emitir líneas estructuradas ──────────────────
from src.middlewares.slogger import SafeLogger
_fase_actual = [0]
_k_val = {k}

def _critic_estruc(self, msg):
    msg = msg.strip()
    if "Fase" in msg:
        # Extraer n y total de "Fase 1/2 | restantes=6"
        try:
            parte = msg.split("Fase")[1].strip()
            nfase, resto = parte.split("/", 1)
            total_str = resto.split("|")[0].strip()
            n = int(nfase.strip())
            total = int(total_str.strip())
            restantes = msg.split("restantes=")[1].strip() if "restantes=" in msg else "?"
            print(f"FASE|{{n}}|{{total}}|restantes={{restantes}}", flush=True)
        except Exception:
            print(f"FASE|?|?|{{msg}}", flush=True)
    elif "Finalizado" in msg or "grupos=" in msg:
        pass   # lo reportamos como RESULTADO al final
SafeLogger.critic = _critic_estruc

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
    t0  = time.time()
    res = kqn.aplicar_estrategia({estado!r}, {condicion!r}, {alcance!r}, {mecanismo!r}, k={k})
    t1  = time.time()
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


def _codigo_kgeomip(csv_path, estado, condicion, alcance, mecanismo, k):
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
        p = Path(tempfile.gettempdir()) / "kgeomip_iface"; p.mkdir(exist_ok=True); return p
    @property
    def tpm_filename(self): return None

try:
    tpm = np.loadtxt({csv_path!r}, delimiter=",")
    N   = tpm.shape[1]

    obj = KGeometricSIA.__new__(KGeometricSIA)
    obj.sia_gestor          = _MM({estado!r})
    obj.sia_logger          = type("L",(),{{"critic":lambda s,m:None,"warn":lambda s,m:None,"debug":lambda s,m:None}})()
    obj.logger              = SafeLogger("kgeomip_iface")
    obj.k                   = {k}
    obj.etiquetas           = [[], []]
    obj.tabla_transiciones  = {{}}
    obj.vertices            = set()
    obj.tabla               = {{}}
    obj.memoria_particiones = {{}}
    obj.sia_tiempo_inicio   = 0.0

    print("PASO|Preparando subsistema y cargando TPM", flush=True)
    obj.aplicar_estrategia.__func__  # verifica existencia
    t0 = time.time()

    # Llamar aplicar_estrategia directamente (usa sia_gestor.estado_inicial)
    print("PASO|Construyendo tabla de costos BFS/DP", flush=True)
    res = obj.aplicar_estrategia({condicion!r}, {alcance!r}, {mecanismo!r}, tpm, k={k})
    t1  = time.time()

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


# ══════════════════════════════════════════════════════════════════════════
# EJECUTOR CON STREAMING EN TIEMPO REAL
# ══════════════════════════════════════════════════════════════════════════

def _ejecutar_con_progreso(codigo, k, algoritmo):
    """
    Lanza el subproceso del algoritmo y muestra el progreso en tiempo real.
    Retorna el dict de resultado o None si hubo error.
    """
    seccion(f"Ejecutando {algoritmo}")
    resultado = None
    errores   = []

    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(codigo)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=SCRIPT_DIR,
    )

    # Leer stdout en tiempo real
    for linea_raw in proc.stdout:
        linea_raw = linea_raw.rstrip()

        if linea_raw.startswith("FASE|"):
            # FASE|n|total|restantes=6
            partes = linea_raw.split("|", 3)
            try:
                n     = int(partes[1])
                total = int(partes[2])
                label = partes[3] if len(partes) > 3 else ""
                barra = barra_progreso(n, total)
                print(f"\r  {AZUL}Fase {n}/{total}{RESET}  {barra}  {GRIS}{label}{RESET}",
                      end="\n", flush=True)
            except (ValueError, IndexError):
                pass

        elif linea_raw.startswith("PASO|"):
            msg = linea_raw[5:]
            print(f"  {AZUL}►{RESET} {msg} …", flush=True)

        elif linea_raw.startswith("CANDIDATO|"):
            partes = linea_raw.split("|")
            try:
                n, total = int(partes[1]), int(partes[2])
                print(f"\r  {barra_progreso(n, total)}  candidato {n}/{total}",
                      end="", flush=True)
            except (ValueError, IndexError):
                pass

        elif linea_raw.startswith("RESULTADO|"):
            raw_json = linea_raw[10:]
            try:
                resultado = json.loads(raw_json)
            except json.JSONDecodeError as e:
                err_msg(f"Error al parsear resultado: {e}")

        elif linea_raw.startswith("ERROR|"):
            err_msg(linea_raw[6:])

    # Leer stderr (solo si hubo error de proceso)
    stderr_out = proc.stderr.read()
    proc.wait()

    if proc.returncode != 0 and resultado is None:
        lineas_err = [
            l for l in stderr_out.splitlines()
            if not l.startswith("CRITICAL") and not l.startswith("WARNING")
        ]
        err_msg("El subproceso terminó con error:")
        for l in lineas_err[-10:]:   # últimas 10 líneas del traceback
            print(ROJO + f"    {l}" + RESET)

    return resultado


# ══════════════════════════════════════════════════════════════════════════
# GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════

def guardar_resultado(resultado, csv_path, k, estado):
    """Guarda el resultado en un archivo .txt junto al CSV."""
    resp = pedir("¿Deseas guardar el resultado en un archivo .txt? [S/n]:")
    if resp.lower() in ("n", "no"):
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_base  = os.path.splitext(os.path.basename(csv_path))[0]
    nombre    = f"resultado_{csv_base}_k{k}_{resultado['algoritmo']}_{timestamp}.txt"
    destino   = os.path.join(os.path.dirname(csv_path), nombre)

    lineas = [
        "=" * 60,
        "  RESULTADO K-QGMIP — Partición de Mínima Información",
        "=" * 60,
        f"  Fecha / hora  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Archivo CSV   : {csv_path}",
        f"  Estado inicial: {estado}",
        f"  Algoritmo     : {resultado['algoritmo']}",
        f"  k (grupos)    : {k}",
        "-" * 60,
        f"  Pérdida EMD   : {resultado['perdida']:.8f}",
        f"  Tiempo total  : {resultado['tiempo']:.3f} s",
        "-" * 60,
        "  Partición encontrada:",
    ]
    for parte in resultado["particion"].split("  "):
        if parte.strip():
            lineas.append(f"    {parte.strip()}")
    lineas += [
        "=" * 60,
        "",
        "  Cómo interpretar la partición:",
        "  · Cada Gj:{Futuros|Mecanismos} es un grupo independiente.",
        "  · Letras mayúsculas = variables futuras (EFFECT).",
        "  · Letras minúsculas = mecanismos presentes (ACTUAL).",
        "  · '-' = el grupo no incluye ese rol.",
        "=" * 60,
    ]

    try:
        with open(destino, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))
        ok_msg(f"Resultado guardado en: {destino}")
    except OSError as exc:
        err_msg(f"No se pudo guardar: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def main():
    banner()

    # ── 1. Cargar CSV ───────────────────────────────────────────────────
    seccion("Cargar matriz de transición (TPM)")
    csv_path, tpm, N = pedir_csv()

    # ── 2. Vista previa ─────────────────────────────────────────────────
    if not vista_previa(tpm, N):
        warn_msg("Carga cancelada. Reinicia el script con el archivo correcto.")
        sys.exit(0)

    # ── 3. Estado inicial ───────────────────────────────────────────────
    seccion("Configuración del sistema")
    estado    = pedir_estado_inicial(N)
    condicion = "1" * N   # usar todas las variables
    alcance   = "1" * N
    mecanismo = "1" * N
    ok_msg(f"Estado inicial : {estado}")
    ok_msg(f"Alcance/mecan. : {'1' * N} (todas las variables)")

    # ── 4. Algoritmo y k ────────────────────────────────────────────────
    algoritmo_id = pedir_algoritmo()
    k = pedir_k(N)
    nombre_alg = "KQNodes" if algoritmo_id == 1 else "KGeometricSIA"
    ok_msg(f"Algoritmo: {nombre_alg}  |  k = {k}")

    # ── 5. Confirmar y ejecutar ─────────────────────────────────────────
    print()
    warn_msg("Presiona ENTER para ejecutar o Ctrl+C para cancelar.")
    try:
        input()
    except KeyboardInterrupt:
        print()
        warn_msg("Cancelado por el usuario.")
        sys.exit(0)

    if algoritmo_id == 1:
        codigo = _codigo_kqnodes(csv_path, estado, condicion, alcance, mecanismo, k)
    else:
        codigo = _codigo_kgeomip(csv_path, estado, condicion, alcance, mecanismo, k)

    t_inicio = time.time()
    resultado = _ejecutar_con_progreso(codigo, k, nombre_alg)
    t_total   = time.time() - t_inicio

    # ── 6. Mostrar tabla de resultados ──────────────────────────────────
    if resultado is None:
        err_msg("No se obtuvo resultado. Revisa el traceback de arriba.")
        sys.exit(1)

    print()
    tabla_resultado(
        perdida    = resultado["perdida"],
        particion  = resultado["particion"],
        tiempo     = resultado["tiempo"],
        algoritmo  = resultado["algoritmo"],
        k          = k,
        csv_name   = csv_path,
    )

    # ── 7. Interpretación rápida ────────────────────────────────────────
    emd = resultado["perdida"]
    print()
    if emd < 0.05:
        ok_msg("EMD < 0.05 → El sistema es casi separable en estos k grupos.")
    elif emd < 0.20:
        warn_msg("0.05 ≤ EMD < 0.20 → Integración causal moderada.")
    else:
        warn_msg("EMD ≥ 0.20 → Alta integración; la partición destruye información importante.")

    # ── 8. Guardar en .txt ──────────────────────────────────────────────
    print()
    guardar_resultado(resultado, csv_path, k, estado)

    # ── 9. Continuar o salir ────────────────────────────────────────────
    print()
    resp = pedir("¿Deseas analizar otro sistema? [s/N]:")
    if resp.lower() in ("s", "si", "sí", "yes", "y"):
        main()
    else:
        print()
        linea("═", AZUL)
        print(AZUL + Style.BRIGHT + "  Gracias por usar K-QGMIP." + RESET)
        linea("═", AZUL)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        warn_msg("Sesion interrumpida. Hasta pronto!")
        print()
