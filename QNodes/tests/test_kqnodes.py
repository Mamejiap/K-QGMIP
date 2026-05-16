"""
test_kqnodes.py — Suite de pruebas para KQNodes (Módulo B).

CASO DE USO VERIFICADO
======================
KQNodes implementa la estrategia "Peeling Greedy Secuencial" para encontrar
la k-partición de menor pérdida EMD de un subsistema IIT. En lugar de
explorar las S(n,k) particiones de Stirling, aplica k-1 fases de un greedy
submodular, encontrando un grupo Gⱼ en cada fase.

CORRECCIÓN DE LA EXTENSIÓN A k-PARTICIONES
===========================================
La extensión es correcta si y solo si:
  1. algorithm(k=2) es equivalente al QNodes original (misma bipartición).
  2. algorithm(k) produce exactamente k grupos con cobertura total de vértices.
  3. max_size en cada fase garantiza que queden suficientes vértices para
     las fases restantes: max_size(fase_j) = |restantes| - (k-1-j).
  4. La función submodular es llamada con kpartir() (no bipartir()).
  5. El criterio greedy minimiza emd_unión - emd_delta en cada paso.

MÉTRICAS DE ERROR ESPERADAS
============================
  - Corrección de grupos:  len(grupos) == k en todo caso (no k-1 por early exit)
  - Cobertura:             union(vertices_j) == todos_los_vertices (sin duplicados)
  - max_size compliance:   |grupo_j| <= max_size_j para toda fase j
  - Formato output:        texto contiene "G1", "G2", ..., "Gk" y separador "|"
"""

import sys, os, types
import numpy as np

# ── Stubs de dependencias no instalables en entorno sin red ────────────────
def _stub(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m

_pkg = _stub("pyinstrument")
class _P:
    def __init__(self,**k): pass
    def start(self): pass
    def stop(self): return self
    def output(self,renderer=None): return ""
_pkg.Profiler = _P
_stub("pyinstrument.renderers").HTMLRenderer = type("H",(),{"__init__":lambda s,**k:None})
_stub("pyttsx3"); _stub("pyttsx3.engine",Engine=type("E",(),{})); _stub("pyttsx3.voice",Voice=type("V",(),{}))
_col = _stub("colorama"); _col.init=lambda**k:None
_col.Fore=type("F",(),{"__getattr__":lambda s,k:""})(); _col.Style=type("S",(),{"__getattr__":lambda s,k:""})()
_stub("colorama.ansi")

# ── Importar KQNodes desde fuente (evitar .pyc cacheado) ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util, importlib.machinery
loader = importlib.machinery.SourceFileLoader(
    "strategies.k_qnodes",
    os.path.join(ROOT, "src", "strategies", "k_qnodes.py")
)
spec = importlib.util.spec_from_loader("strategies.k_qnodes", loader)
mod  = importlib.util.module_from_spec(spec)
sys.modules["strategies.k_qnodes"] = mod
spec.loader.exec_module(mod)
KQNodes = mod.KQNodes

from src.constants.base import ACTUAL, EFFECT


def make_kqn(csv_name: str) -> "KQNodes":
    samples = os.path.join(ROOT, "src", ".samples")
    tpm = np.loadtxt(os.path.join(samples, csv_name), delimiter=",", dtype=np.float64)
    return KQNodes(tpm)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — algorithm(k=2) produce exactamente 2 grupos
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_k2_produce_exactamente_2_grupos():
    """
    Caso de uso: la extensión k=2 debe comportarse como la bipartición
    original — producir exactamente 2 grupos con todos los vértices cubiertos.

    Corrección: si k=2 falla (0 o 1 grupos), la generalización está rota.
    Métrica esperada: len(grupos) == 2, union(grupos) == todos_vertices.
    """
    kqn = make_kqn("N2A.csv")
    kqn.aplicar_estrategia("00", "11", "11", "11", k=2)
    v2 = [(EFFECT,0),(EFFECT,1),(ACTUAL,0),(ACTUAL,1)]
    grupos, perdida, dist = kqn.algorithm(v2, k=2)

    assert len(grupos) == 2, f"Esperado 2 grupos, obtenido {len(grupos)}"
    asignados = set()
    for g in grupos: asignados.update(kqn._flatten_vertices(g))
    assert asignados == set(v2), f"Cobertura incompleta: {asignados} vs {set(v2)}"
    assert perdida >= 0.0, f"Perdida negativa: {perdida}"


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — algorithm(k=3) produce exactamente 3 grupos para N=3
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_k3_produce_exactamente_3_grupos():
    """
    Caso de uso: con N=3 variables y k=3, el algoritmo debe producir
    exactamente 3 grupos (no 2 por early exit prematuro).

    Corrección de max_size: sin la restricción max_size = r - (k-1-fase),
    la fase 1 absorbe demasiados vértices y la fase 2 no puede ejecutarse.
    Este test valida que la restricción funciona correctamente.

    Métrica esperada: len(grupos) == 3, todos los vértices cubiertos.
    """
    kqn3 = make_kqn("N3A.csv")
    kqn3.aplicar_estrategia("000", "111", "111", "111", k=3)
    v3 = [(EFFECT,i) for i in range(3)] + [(ACTUAL,i) for i in range(3)]
    grupos, perdida, dist = kqn3.algorithm(v3, k=3)

    assert len(grupos) == 3, (
        f"Esperado 3 grupos, obtenido {len(grupos)}. "
        f"Verifica la restriccion max_size en _greedy_fase()."
    )
    asignados = set()
    for g in grupos: asignados.update(kqn3._flatten_vertices(g))
    assert asignados == set(v3), f"Cobertura incompleta: {asignados} vs {set(v3)}"


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Cumplimiento de max_size por fase
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_max_size_cumplido_por_fase():
    """
    Caso de uso: la restricción max_size garantiza que en la fase j de
    una k-partición con r vértices restantes, el grupo creado tiene como
    máximo max_size = r - (k-1-j) vértices.

    Corrección matemática: si max_size no se respeta, alguna fase posterior
    encontrará 0 vértices y no podrá formar su grupo, rompiendo la garantía
    de producir exactamente k grupos.

    Métrica esperada: |G_j| <= max_size_j para toda fase j.
    """
    kqn3 = make_kqn("N3A.csv")
    kqn3.aplicar_estrategia("000", "111", "111", "111", k=3)
    v3 = [(EFFECT,i) for i in range(3)] + [(ACTUAL,i) for i in range(3)]
    grupos, _, _ = kqn3.algorithm(v3, k=3)

    # Verificar tamaños: G1 <= 6-(3-1-0)=4, G2 <= 2-(3-1-1)=1
    # (los tamaños exactos dependen de los datos, pero la garantia es k grupos)
    r = len(v3)
    k = 3
    for fase, g in enumerate(grupos[:-1]):  # excluir el ultimo (Gk = resto)
        flat = list(kqn3._flatten_vertices(g))
        max_s = r - (k - 1 - fase)
        assert len(flat) <= max_s, (
            f"Fase {fase}: grupo de {len(flat)} vertices excede max_size={max_s}"
        )
        r -= len(flat)


# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — Determinismo: misma entrada produce mismo resultado
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_determinismo():
    """
    Caso de uso: el algoritmo greedy es determinista — misma TPM, mismo
    estado inicial, mismo k → misma partición y misma pérdida.

    Corrección: el greedy no tiene componentes aleatorios; la función
    submodular con kpartir() siempre evalúa el mismo EMD para la misma
    configuración. Si no es determinista, hay un bug de estado compartido.

    Métrica esperada: perdida_1 == perdida_2 (exactamente, no solo ≈).
    """
    kqn = make_kqn("N2A.csv")
    kqn.aplicar_estrategia("00", "11", "11", "11", k=2)
    v2 = [(EFFECT,0),(EFFECT,1),(ACTUAL,0),(ACTUAL,1)]

    _, perdida_1, _ = kqn.algorithm(v2, k=2)
    _, perdida_2, _ = kqn.algorithm(v2, k=2)

    assert perdida_1 == perdida_2, (
        f"No determinista: {perdida_1} != {perdida_2}"
    )


# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — _vertices_a_fj_mj: separación correcta de tiempo y espacio
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_vertices_a_fj_mj_correcto():
    """
    Caso de uso: _vertices_a_fj_mj() debe separar correctamente los
    vértices de una lista mixta en Fⱼ (EFFECT) y Mⱼ (ACTUAL).

    Corrección de la extensión: kpartir() necesita (Fⱼ, Mⱼ) bien
    separados. Si esta función falla, kpartir() recibe grupos mal formados.

    Métrica esperada: Fⱼ y Mⱼ contienen exactamente los índices correctos.
    """
    kqn = make_kqn("N2A.csv")
    verts = [(EFFECT, 0), (ACTUAL, 1), (EFFECT, 2), (ACTUAL, 0)]
    resultado = kqn._vertices_a_fj_mj(verts)
    assert resultado is not None, "Debe devolver (Fj, Mj), no None"
    fj, mj = resultado
    assert set(fj.tolist()) == {0, 2}, f"Fj incorrecto: {fj}"
    assert set(mj.tolist()) == {0, 1}, f"Mj incorrecto: {mj}"
    # Sin EFFECT -> None
    assert kqn._vertices_a_fj_mj([(ACTUAL, 0)]) is None


# ══════════════════════════════════════════════════════════════════════════
# TEST 6 — Formato de salida legible con k grupos
# ══════════════════════════════════════════════════════════════════════════
def test_kqnodes_formato_k_grupos():
    """
    Caso de uso: _formatear_k_particion() debe producir una cadena con
    k grupos en el formato "G1:{futuras|presentes}  G2:{...}  ...".

    Corrección: el formato es usado por Solution para mostrar resultados
    al usuario. Si el formato está mal, el reporte de la partición falla.

    Métrica esperada: texto contiene "G1", "G2", ..., "Gk" y el separador "|".
    """
    kqn = make_kqn("N2A.csv")
    kqn.aplicar_estrategia("00", "11", "11", "11", k=2)
    grupos = [((EFFECT,0),(ACTUAL,0)), ((EFFECT,1),(ACTUAL,1))]
    txt = kqn._formatear_k_particion(grupos)
    assert "G1" in txt and "G2" in txt, f"Formato no contiene G1/G2: '{txt}'"
    assert "|" in txt, f"Formato no contiene separador '|': '{txt}'"
    assert txt == "G1:{A|a}  G2:{B|b}", f"Formato inesperado: '{txt}'"


if __name__ == "__main__":
    tests = [
        test_kqnodes_k2_produce_exactamente_2_grupos,
        test_kqnodes_k3_produce_exactamente_3_grupos,
        test_kqnodes_max_size_cumplido_por_fase,
        test_kqnodes_determinismo,
        test_kqnodes_vertices_a_fj_mj_correcto,
        test_kqnodes_formato_k_grupos,
    ]
    resultados = []
    for t in tests:
        try:
            t()
            resultados.append((t.__name__, "PASS"))
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            resultados.append((t.__name__, f"FAIL: {e}"))
            print(f"  [FAIL] {t.__name__}: {e}")
    total = len(resultados)
    pasados = sum(1 for _, r in resultados if r == "PASS")
    print(f"\n  RESULTADO: {pasados}/{total}")
