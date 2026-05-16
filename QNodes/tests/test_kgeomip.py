"""
test_kgeomip.py — Suite de pruebas para KGeometricSIA (Módulo C, KGeoMIP).

CASO DE USO VERIFICADO
======================
KGeometricSIA extiende GeometricSIA para encontrar la k-partición de menor
pérdida EMD usando la tabla de costos geométrica del hipercubo binario.
En lugar de evaluar 1 bipartición óptima, identifica candidatos de k grupos
mediante dos capas y los evalúa con System.kpartir() (Módulo A).

CORRECCIÓN DE LA EXTENSIÓN A k-PARTICIONES
===========================================
La extensión geométrica es correcta si:
  1. La tabla de costos BFS/DP tiene valores tx(s0, sj)[idx] en [0, 1].
  2. identificar_particiones_optimas() genera candidatos de k grupos,
     cada uno con: union(futuros_j) == {0,...,M-1} (cobertura total).
  3. _candidato_a_grupos_kpartir() convierte correctamente a (Fj, Mj)
     para que kpartir() produzca distribuciones válidas en [0, 1].
  4. find_mip() completa el ciclo BFS/DP → candidatos → kpartir → EMD → min.

MÉTRICAS DE ERROR ESPERADAS
============================
  - Tabla de costos:    tx(s0, s_final)[idx] en [0, 1] para todo idx
  - Candidatos:         len(candidatos) >= 1, cada uno tiene k grupos no vacios
  - Distribuciones:     shape (M,) con valores en [0, 1]
  - EMD MIP:            perdida en [0, M * 0.5]
  - Cobertura futuros:  union(futuros_j para j en candidato) == {0,...,M-1}
"""

import sys, os, types
import numpy as np

# ── Stubs de dependencias no instalables (GeoMIP + QNodes) ───────────────
def _stub(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m

# pyemd — usado en funcs/base.py para emd_causal (no por emd_efecto)
_pyemd = _stub("pyemd")
_pyemd.emd = lambda u, v, cost: float(np.sum(np.abs(u - v)))

# pyinstrument — usado en middlewares/profile.py
_pkg = _stub("pyinstrument")
class _P:
    def __init__(self,**k): pass
    def start(self): pass
    def stop(self): return self
    def output(self,renderer=None): return ""
_pkg.Profiler = _P
_stub("pyinstrument.renderers").HTMLRenderer = type("H",(),{"__init__":lambda s,**k:None})

# pyttsx3 — usado en models/core/solution.py
_stub("pyttsx3"); _stub("pyttsx3.engine",Engine=type("E",(),{})); _stub("pyttsx3.voice",Voice=type("V",(),{}))

# colorama — usado en models/core/solution.py
_col = _stub("colorama"); _col.init=lambda**k:None
_col.Fore=type("F",(),{"__getattr__":lambda s,k:""})(); _col.Style=type("S",(),{"__getattr__":lambda s,k:""})()
_stub("colorama.ansi")

# ── Paths ─────────────────────────────────────────────────────────────────
TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
QNODES_ROOT = os.path.dirname(TESTS_DIR)
REPO_ROOT   = os.path.dirname(QNODES_ROOT)
GEO_M2      = os.path.join(
    REPO_ROOT, "GeoMIP", "src",
    "Method2_Dynamic_Programming_Reformulation"
)
sys.path.insert(0, QNODES_ROOT) # posición 1 final
sys.path.insert(0, GEO_M2)     # posición 0 final → GeoMIP src/ tiene precedencia

from src.controllers.strategies.k_geometric import KGeometricSIA
from src.funcs.base import emd_efecto


# ── Fixture: mock manager y constructor de KGeometricSIA ─────────────────

class _MockManager:
    """
    Manager mínimo para tests. Solo proporciona estado_inicial y un
    output_dir en /tmp, sin tocar el filesystem del proyecto.
    Compatible con SIA.__init__() de GeoMIP.
    """
    def __init__(self, estado_inicial: str):
        self.estado_inicial = estado_inicial
        self.pagina_red_muestra = "A"

    @property
    def pagina(self):
        return "A"

    @property
    def output_dir(self):
        from pathlib import Path
        import tempfile
        p = Path(tempfile.gettempdir()) / "kgeomip_tests"
        p.mkdir(exist_ok=True)
        return p

    @property
    def tpm_filename(self):
        return None


def make_kgeo(n_vars: int, k: int = 2) -> KGeometricSIA:
    """
    Construye un KGeometricSIA con estado inicial todo-ceros, sin
    llamar a __init__ completo (evita imports pesados del gestor).
    """
    from src.middlewares.slogger import SafeLogger

    obj = KGeometricSIA.__new__(KGeometricSIA)
    obj.sia_gestor          = _MockManager("0" * n_vars)
    obj.sia_logger          = type("L", (), {
        "critic": lambda s, m: None,
        "warn":   lambda s, m: None,
        "debug":  lambda s, m: None,
    })()
    obj.logger              = SafeLogger("kgeomip_test")
    obj.k                   = k
    obj.etiquetas           = [[], []]
    obj.tabla_transiciones  = {}
    obj.vertices            = set()
    obj.tabla               = {}
    obj.memoria_particiones = {}
    obj.sia_tiempo_inicio   = 0.0
    return obj


def preparar_subsistema(kgeo: KGeometricSIA, csv_name: str,
                        condicion: str, alcance: str, mecanismo: str):
    """Carga TPM y prepara subsistema via sia_preparar_subsistema."""
    tpm = np.loadtxt(
        os.path.join(QNODES_ROOT, "src", ".samples", csv_name),
        delimiter=",", dtype=np.float64
    )
    kgeo.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
    dims = kgeo.sia_subsistema.dims_ncubos
    kgeo.estado_inicial = kgeo.sia_subsistema.estado_inicial[dims]
    kgeo.estado_final   = 1 - kgeo.estado_inicial
    kgeo._flat_data     = [ncubo.data.ravel()
                           for ncubo in kgeo.sia_subsistema.ncubos]
    kgeo.idx_ncubos     = list(range(len(kgeo.sia_subsistema.indices_ncubos)))


def construir_tabla(kgeo: KGeometricSIA):
    """Inicializa la tabla de transiciones y ejecuta el BFS/DP completo."""
    n = len(kgeo.estado_inicial)
    kgeo.caminos = {0: [kgeo.estado_inicial.tolist()]}
    kgeo.tabla_transiciones = {
        (tuple(kgeo.caminos[0][0]), tuple(kgeo.caminos[0][0])): [0.0] * len(kgeo.idx_ncubos)
    }
    for nivel in range(1, n + 1):
        kgeo.calcular_costos_nivel(kgeo.estado_final, nivel)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — La tabla de costos tiene valores en [0, 1]
# ══════════════════════════════════════════════════════════════════════════
def test_kgeomip_tabla_costos_valores_validos():
    """
    Caso de uso: tx(s0, s_final)[idx] en [0,1] para toda variable futura.
    El factor 1/2^dH y la acumulacion recursiva deben acotar los costos.

    Corrección: si los costos superan 1, la asignacion de grupos por
    argmin en la Capa 2 puede priorizar variables incorrectamente.

    Métrica esperada: todos los costos finales en [0.0, 1.0].
    """
    kgeo = make_kgeo(n_vars=2)
    preparar_subsistema(kgeo, "N2A.csv", "11", "11", "11")
    construir_tabla(kgeo)

    clave_fin = (tuple(kgeo.caminos[0][0]), tuple(kgeo.estado_final.tolist()))
    costos = kgeo.tabla_transiciones.get(clave_fin)
    assert costos is not None, "Clave del estado final no encontrada en tabla"
    for idx, c in enumerate(costos):
        assert 0.0 - 1e-9 <= c <= 1.0 + 1e-9, (
            f"Costo[{idx}]={c:.4f} fuera de [0,1]"
        )
    print(f"    costos finales: {[round(c,4) for c in costos]}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — identificar_particiones_optimas() genera candidatos con k grupos
# ══════════════════════════════════════════════════════════════════════════
def test_kgeomip_candidatos_tienen_k_grupos():
    """
    Caso de uso: con N=3 y k=3, cada candidato debe tener exactamente
    3 grupos, cada uno con al menos una variable futura.

    Corrección: candidatos con grupo futuro vacío hacen fallar kpartir()
    con ValueError, abortando la evaluación de ese candidato.

    Métrica esperada: todos los candidatos tienen k=3 grupos no vacios.
    """
    kgeo = make_kgeo(n_vars=3, k=3)
    preparar_subsistema(kgeo, "N3A.csv", "111", "111", "111")
    construir_tabla(kgeo)

    candidatos = kgeo.identificar_particiones_optimas()
    assert len(candidatos) >= 1, "No se generaron candidatos"

    for i, cand in enumerate(candidatos):
        assert len(cand) == 3, f"Candidato {i}: {len(cand)} grupos (esperado 3)"
        for j, (_, fut_j) in enumerate(cand):
            assert len(fut_j) > 0, f"Candidato {i}, grupo {j}: futuros vacio"
    print(f"    {len(candidatos)} candidatos generados, todos con k=3 grupos")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — _candidato_a_grupos_kpartir() produce distribuciones validas
# ══════════════════════════════════════════════════════════════════════════
def test_kgeomip_distribucion_kpartir_valida():
    """
    Caso de uso: el puente entre candidatos geométricos (índices locales)
    y kpartir() (índices globales del subsistema) debe ser correcto para
    que la distribución marginal resultante esté en [0, 1].

    Corrección: si los índices se mapean mal, kpartir() puede recibir
    grupos con índices fuera de rango y lanzar KeyError.

    Métrica esperada: dist.shape == (M,) con todos los valores en [0, 1].
    """
    kgeo = make_kgeo(n_vars=2, k=2)
    preparar_subsistema(kgeo, "N2A.csv", "11", "11", "11")

    # Candidato manual: dos grupos, cada uno con todas las presentes
    candidato = [
        [list(range(2)), [0]],  # G1: futura=0, pres={0,1}
        [list(range(2)), [1]],  # G2: futura=1, pres={0,1}
    ]
    grupos_kpartir = kgeo._candidato_a_grupos_kpartir(candidato)
    dist = kgeo.sia_subsistema.kpartir(grupos_kpartir).distribucion_marginal()

    assert dist.shape == (2,), f"Shape incorrecto: {dist.shape}"
    assert np.all(dist >= -1e-6) and np.all(dist <= 1 + 1e-6), (
        f"Distribucion fuera de [0,1]: {dist}"
    )
    print(f"    distribucion kpartir: {dist}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — find_mip() retorna perdida valida y key en memoria_particiones
# ══════════════════════════════════════════════════════════════════════════
def test_kgeomip_find_mip_retorna_perdida_valida():
    """
    Caso de uso: find_mip() ejecuta el ciclo completo:
    BFS/DP → identificar_particiones_optimas → kpartir → emd → min.
    Debe retornar una clave presente en memoria_particiones con EMD >= 0.

    Corrección: si find_mip() lanza RuntimeError ("no generó candidatos"),
    alguno de los pasos previos produjo únicamente grupos inválidos.

    Métrica esperada: perdida_mip en [0.0, M * 0.5].
    """
    kgeo = make_kgeo(n_vars=2, k=2)
    preparar_subsistema(kgeo, "N2A.csv", "11", "11", "11")

    mip = kgeo.find_mip()

    assert mip in kgeo.memoria_particiones, "Clave MIP no en memoria_particiones"
    perdida, dist = kgeo.memoria_particiones[mip]
    assert perdida >= 0.0, f"Perdida negativa: {perdida}"
    assert perdida <= 2 * 0.5 + 1e-6, f"Perdida mayor que maximo teorico: {perdida}"
    assert dist is not None and len(dist) == 2, f"Distribucion invalida: {dist}"
    print(f"    MIP encontrado | perdida={perdida:.4f} | candidatos evaluados={len(kgeo.memoria_particiones)}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — Capa 1: cobertura total de variables futuras
# ══════════════════════════════════════════════════════════════════════════
def test_kgeomip_capa1_cobertura_total_futuros():
    """
    Caso de uso: cada candidato generado por la Capa 1 (cuantiles) debe
    cubrir el conjunto completo {0,...,M-1} de variables futuras.

    Corrección: si alguna variable futura queda sin grupo, kpartir() lanza
    ValueError y ese candidato se descarta. Si TODOS los candidatos fallan,
    find_mip() lanza RuntimeError. La Capa 1 es el fallback garantizado.

    Métrica esperada: union(futuros_j) == {0,1,2} para todo candidato.
    """
    kgeo = make_kgeo(n_vars=3, k=3)
    preparar_subsistema(kgeo, "N3A.csv", "111", "111", "111")
    construir_tabla(kgeo)

    candidatos = kgeo.identificar_particiones_optimas()

    for i, cand in enumerate(candidatos):
        todos_futuros = set()
        for _, fut_j in cand:
            todos_futuros.update(fut_j)
        assert todos_futuros == {0, 1, 2}, (
            f"Candidato {i} no cubre todos los futuros: {todos_futuros}"
        )
    print(f"    {len(candidatos)} candidatos — todos cubren futuros {{0,1,2}}")


# ── Runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_kgeomip_tabla_costos_valores_validos,
        test_kgeomip_candidatos_tienen_k_grupos,
        test_kgeomip_distribucion_kpartir_valida,
        test_kgeomip_find_mip_retorna_perdida_valida,
        test_kgeomip_capa1_cobertura_total_futuros,
    ]
    resultados = []
    for t in tests:
        try:
            t()
            resultados.append((t.__name__, "PASS"))
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            resultados.append((t.__name__, f"FAIL: {e}"))
            import traceback; traceback.print_exc()
            print(f"  [FAIL] {t.__name__}: {e}")
    total   = len(resultados)
    pasados = sum(1 for _, r in resultados if r == "PASS")
    print(f"\n  RESULTADO: {pasados}/{total}")
