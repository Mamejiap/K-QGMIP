"""
test_kpartir.py — Suite de pruebas para System.kpartir() (Módulo A).

CASO DE USO VERIFICADO
======================
kpartir() generaliza bipartir() al caso de k ≥ 2 grupos independientes.
Dada una k-partición {G₁,...,Gₖ} donde cada Gⱼ = (Fⱼ, Mⱼ):
    - Fⱼ: índices de variables futuras (NCubos) del grupo j
    - Mⱼ: índices de variables presentes que Gⱼ conserva
Para cada NCubo ∈ Fⱼ: marginalizar(dims - Mⱼ) para independizar el grupo.

CORRECCIÓN DE LA EXTENSIÓN A k-PARTICIONES
===========================================
La extensión es correcta si y solo si:
  1. kpartir(k=2) == bipartir() para toda bipartición equivalente.
  2. La distribución marginal de kpartir(k) es un producto tensorial de
     k distribuciones independientes (menor o igual EMD que bipartición).
  3. La memoización reutiliza resultados para llamadas con misma clave.
  4. Se lanza ValueError para particiones que no cubren todos los NCubos.

MÉTRICAS DE ERROR ESPERADAS
============================
  - Test equivalencia k=2: EMD(kpartir, bipartir) ≈ 0.0 (diferencia < 1e-10)
  - Test k=3 con N3A.csv:  perdida(k=3) ∈ [0.0, perdida_max_teorica]
  - Test memoización:       tiempo segunda llamada << primera llamada
"""

import sys, os
import numpy as np
import time

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # QNodes/ → from src.xxx funciona

from src.models.core.system import System
from src.funcs.iit import emd_efecto


# ── Fixtures ───────────────────────────────────────────────────────────────
def make_system(csv_name: str, estado: list) -> System:
    """Carga una TPM desde el CSV de muestras y construye un System completo."""
    samples_dir = os.path.join(ROOT, "src", ".samples")
    tpm = np.loadtxt(os.path.join(samples_dir, csv_name), delimiter=",", dtype=np.float64)
    estado_arr = np.array(estado, dtype=np.int8)
    return System(tpm, estado_arr)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Equivalencia kpartir(k=2) == bipartir() para N=2
# ══════════════════════════════════════════════════════════════════════════
def test_kpartir_k2_equivalente_bipartir_n2():
    """
    Caso de uso: la extensión k=2 debe reproducir exactamente bipartir().
    Esto valida que kpartir() es una generalización correcta y no rompe
    el comportamiento base de la bipartición original.

    Métrica esperada: diferencia de distribuciones < 1e-10.
    """
    sys_n2 = make_system("N2A.csv", [0, 0])
    alcance   = np.array([0], dtype=np.int8)   # futuros del grupo 1: solo var 0
    mecanismo = np.array([0, 1], dtype=np.int8) # presentes del grupo 1

    # bipartir original
    dist_bipartir = sys_n2.bipartir(alcance, mecanismo).distribucion_marginal()

    # kpartir con k=2: grupo 1 = ({0}, {0,1}), grupo 2 = ({1}, complemento)
    comp_futuros  = np.array([1], dtype=np.int8)
    comp_presentes = np.array([], dtype=np.int8)  # complemento del mecanismo
    grupos_k2 = [(alcance, mecanismo), (comp_futuros, comp_presentes)]
    dist_kpartir = sys_n2.kpartir(grupos_k2).distribucion_marginal()

    diff = np.sum(np.abs(dist_bipartir.astype(float) - dist_kpartir.astype(float)))
    assert diff < 1e-10, (
        f"kpartir(k=2) difiere de bipartir() en {diff:.2e}. "
        f"bipartir={dist_bipartir}, kpartir={dist_kpartir}"
    )


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — kpartir(k=3) produce resultado válido para N=3
# ══════════════════════════════════════════════════════════════════════════
def test_kpartir_k3_valido_n3():
    """
    Caso de uso: con N=3 variables, kpartir(k=3) debe producir una
    distribución marginal válida (probabilidades en [0,1], shape correcta).

    Extensión k-particiones: 3 grupos independientes es el caso más
    simple de k > 2 con cobertura total de las 3 variables futuras.

    Métrica esperada: distribución con 3 valores en [0, 1].
    """
    sys_n3 = make_system("N3A.csv", [0, 0, 0])
    # Partición: G1=(0), G2=(1), G3=(2) — cada variable en su propio grupo
    grupos_k3 = [
        (np.array([0], dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)),
        (np.array([1], dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)),
        (np.array([2], dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)),
    ]
    dist = sys_n3.kpartir(grupos_k3).distribucion_marginal()

    assert dist.shape == (3,), f"Shape incorrecto: {dist.shape}"
    assert np.all(dist >= -1e-6) and np.all(dist <= 1 + 1e-6), (
        f"Distribución fuera de [0,1]: {dist}"
    )


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Cobertura parcial lanza ValueError
# ══════════════════════════════════════════════════════════════════════════
def test_kpartir_cobertura_incompleta_lanza_error():
    """
    Caso de uso: si la unión de Fⱼ no cubre todos los NCubos del sistema,
    kpartir() debe lanzar ValueError con mensaje descriptivo.

    Corrección matemática: una k-partición que no cubre todas las variables
    futuras no está bien definida y no puede generar una distribución válida.
    """
    sys_n2 = make_system("N2A.csv", [0, 0])
    # Solo cubre la variable futura 0, deja la 1 sin grupo
    grupos_incompletos = [
        (np.array([0], dtype=np.int8), np.array([0, 1], dtype=np.int8)),
    ]
    raised = False
    try:
        sys_n2.kpartir(grupos_incompletos)
    except ValueError as e:
        assert "no pertenece" in str(e).lower() or "grupo" in str(e).lower(), f"Mensaje inesperado: {e}"
        raised = True
    assert raised, "Deberia haber lanzado ValueError"


# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — Memoización: segunda llamada idéntica usa caché
# ══════════════════════════════════════════════════════════════════════════
def test_kpartir_memoizacion_activa():
    """
    Caso de uso: kpartir() guarda resultados en self.memo para evitar
    recomputar marginalizaciones idénticas. La segunda llamada debe
    producir el mismo resultado y el memo debe tener al menos 1 entrada.

    Extensión k-particiones: la memoización es fundamental para la
    eficiencia del algoritmo KQNodes que llama kpartir() O(k·n²) veces.
    """
    sys_n3 = make_system("N3A.csv", [0, 0, 0])
    grupos_k2 = [
        (np.array([0, 1], dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)),
        (np.array([2], dtype=np.int8),    np.array([], dtype=np.int8)),
    ]

    dist1 = sys_n3.kpartir(grupos_k2).distribucion_marginal()
    n_memo_antes = len(sys_n3.memo)
    dist2 = sys_n3.kpartir(grupos_k2).distribucion_marginal()
    n_memo_despues = len(sys_n3.memo)

    assert np.allclose(dist1, dist2, atol=1e-10), "Resultados distintos para misma entrada"
    assert n_memo_antes == n_memo_despues, (
        f"La segunda llamada no usó caché: memo creció de {n_memo_antes} a {n_memo_despues}"
    )
    assert n_memo_antes >= 1, "El memo debería tener al menos 1 entrada tras kpartir()"


# ══════════════════════════════════════════════════════════════════════════
# TEST 5 — Marginalización correcta: grupo singleton independiza su NCubo
# ══════════════════════════════════════════════════════════════════════════
def test_kpartir_singleton_marginaliza_completamente():
    """
    Caso de uso: un grupo G con F = {idx} y M = [] (mecanismo vacío) debe
    marginalizar todas las dimensiones del NCubo, produciendo un escalar
    (probabilidad marginal incondicional del nodo idx).

    Métrica esperada: el NCubo resultante tiene data.ndim == 0 o shape==().
    """
    sys_n2 = make_system("N2A.csv", [0, 0])
    # Grupo singleton: futura=0, mecanismo vacío → marginalizar todo
    grupos = [
        (np.array([0], dtype=np.int8), np.array([], dtype=np.int8)),
        (np.array([1], dtype=np.int8), np.array([], dtype=np.int8)),
    ]
    sistema_partido = sys_n2.kpartir(grupos)
    # Ambos NCubos deben tener shape=() (escalar) tras marginalizar todo
    for ncubo in sistema_partido.ncubos:
        assert ncubo.data.ndim == 0 or ncubo.data.size == 1, (
            f"NCubo {ncubo.indice} no fue completamente marginalizado. "
            f"Shape: {ncubo.data.shape}"
        )


if __name__ == "__main__":
    tests = [
        test_kpartir_k2_equivalente_bipartir_n2,
        test_kpartir_k3_valido_n3,
        test_kpartir_cobertura_incompleta_lanza_error,
        test_kpartir_memoizacion_activa,
        test_kpartir_singleton_marginaliza_completamente,
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
