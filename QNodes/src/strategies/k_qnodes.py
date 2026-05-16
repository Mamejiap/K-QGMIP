"""
k_qnodes.py — Extensión de QNodes para k-particiones.

Estrategia: Peeling Greedy Secuencial.
Universidad de Caldas — Proyecto K-QGMIP — 2026-1

Arquitectura modular:
    algorithm()               ← Loop externo:  k−1 fases de peeling
    └─ _greedy_fase()         ← Loop interno:  greedy submodular por fase
       └─ _funcion_submodular_k()  ← Evaluación con kpartir()
          └─ _construir_grupos_kpartir()  ← Convierte vértices → (Fj, Mj)
"""

import time
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import QNODES_ANALYSIS_TAG
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion


# ─── Constantes del módulo ─────────────────────────────────────────────────
KQNODES_LABEL = "KQNodes"
KQNODES_TAG   = "kqnodes_strategy"


class KQNodes(SIA):
    """
    KQNodes: extensión del algoritmo QNodes para k-particiones (k ≥ 2).

    Implementa la estrategia de 'Peeling Greedy Secuencial':
    en lugar de buscar una bipartición óptima (1 grupo + complemento),
    ejecuta k−1 fases consecutivas. En cada fase aplica un loop greedy
    submodular sobre los vértices aún no asignados para encontrar el
    grupo Gⱼ que minimiza la pérdida de información.

    La evaluación en cada fase usa `System.kpartir()` en lugar de
    `bipartir()`, incluyendo siempre:
        [grupos ya fijados] + [candidato actual] + [cola restante]

    Esto garantiza que la función objetivo sea siempre una k-partición
    completa, y que el caché (memo) de `kpartir()` se reutilice entre
    fases sin recalcular.

    Complejidad: O(k · n³), donde n = número de vértices del subsistema.

    Args:
    ----
        tpm (np.ndarray):
            Matriz de Probabilidad de Transición del sistema.

    Attributes:
    ----------
        k (int):
            Número de grupos objetivo de la k-partición.
        m (int):
            Número de variables futuras (NCubos) del subsistema.
        n (int):
            Número de variables presentes (dims) del subsistema.
        vertices (set[tuple]):
            Conjunto de todos los vértices (tiempo, índice) del subsistema.
        memoria_delta_k (dict):
            Caché de EMDs individuales por vértice/grupo evaluado.
        memoria_grupos_k (dict):
            Caché de EMDs por grupo candidato encontrado en cada fase.
    """

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.k: int = 2
        self.m: int = INT_ZERO
        self.n: int = INT_ZERO
        self.vertices: set[tuple]
        self.indices_alcance: NDArray[np.int8]
        self.indices_mecanismo: NDArray[np.int8]

        self.memoria_delta_k:  dict[tuple, tuple[float, NDArray]] = {}
        self.memoria_grupos_k: dict[tuple, tuple[float, NDArray]] = {}

        self.logger = SafeLogger(KQNODES_TAG)

    # ══════════════════════════════════════════════════════════════════════
    #  PUNTO DE ENTRADA PÚBLICO
    # ══════════════════════════════════════════════════════════════════════

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solution:
        """
        Prepara el subsistema y lanza el algoritmo KQNodes para k grupos.

        Replica la interfaz de `QNodes.aplicar_estrategia()` añadiendo el
        parámetro `k`. El resto del pipeline (condicionamiento, substracción,
        distribuciones marginales) es idéntico al algoritmo original.

        Args:
        ----
            estado_inicial (str): Cadena binaria del estado inicial del sistema.
            condicion (str):      Bits de condicionamiento (0 = condicionar).
            alcance (str):        Bits del alcance futuro (0 = marginalizar).
            mecanismo (str):      Bits del mecanismo presente (0 = marginalizar).
            k (int):              Número de grupos deseados (k ≥ 2).

        Returns:
        -------
            Solution: Objeto con la k-partición de mínima pérdida encontrada,
                su distribución marginal, la pérdida EMD y el tiempo de ejecución.

        Raises:
        ------
            ValueError: Si k < 2 o si k supera el número de variables futuras.
        """
        if k < 2:
            raise ValueError(f"k debe ser ≥ 2. Recibido: k={k}")

        self.k = k
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        if k > self.sia_subsistema.indices_ncubos.size:
            raise ValueError(
                f"k={k} supera el número de variables futuras del subsistema "
                f"({self.sia_subsistema.indices_ncubos.size}). "
                f"Reduzca k o amplíe el alcance."
            )

        # Construir los vértices: (tiempo, índice)
        # EFFECT (1) → variables futuras / NCubos
        # ACTUAL (0) → variables presentes / dims del mecanismo
        futuro = tuple(
            (EFFECT, int(idx)) for idx in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, int(idx)) for idx in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance   = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.vertices = set(presente + futuro)

        vertices = list(presente + futuro)
        grupos, perdida, dist = self.algorithm(vertices, k)

        fmt_mip = self._formatear_k_particion(grupos)

        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  MÓDULO A — LOOP EXTERNO: k−1 FASES DE PEELING
    # ══════════════════════════════════════════════════════════════════════

    @profile(context={TYPE_TAG: KQNODES_TAG})
    def algorithm(
        self,
        vertices: list[tuple[int, int]],
        k: int,
    ) -> tuple[list[tuple], float, NDArray]:
        """
        Peeling Greedy Secuencial: k−1 fases para formar k grupos.

        En cada fase llama a `_greedy_fase()` sobre los vértices aún no
        asignados. El grupo resultante (Gⱼ) se fija y sus vértices se
        eliminan de la lista restante. Tras k−1 fases, el grupo final
        (Gₖ) es el conjunto de vértices que quedan sin asignar.

        La pérdida de la k-partición completa se evalúa al final con
        una única llamada a `kpartir()` usando todos los grupos fijados.

        Args:
        ----
            vertices (list[tuple[int, int]]):
                Lista de todos los vértices (tiempo, índice) del subsistema.
            k (int):
                Número de grupos objetivo.

        Returns:
        -------
            Tupla (grupos_fijos, perdida_mip, dist_mip) donde:
            - grupos_fijos: lista de k grupos de vértices (cada grupo es una tupla).
            - perdida_mip:  EMD de la k-partición óptima encontrada.
            - dist_mip:     distribución marginal de esa k-partición.
        """
        grupos_fijos: list[tuple] = []
        restantes: list[tuple[int, int]] = list(vertices)

        self.logger.critic(
            f"[KQNodes] Iniciando | k={k} | vértices={len(vertices)}"
        )

        # ── LOOP EXTERNO: una fase por cada grupo a pelar (k−1 en total) ──
        for fase in range(k - 1):

            self.logger.critic(
                f"  Fase {fase + 1}/{k - 1} | restantes={len(restantes)}"
            )

            # Caso borde: si solo queda 1 vértice, no hay nada que separar
            if len(restantes) <= 1:
                grupos_fijos.append(tuple(restantes))
                restantes = []
                break

            # Máximo de vértices que puede tomar este grupo para garantizar
            # que queden suficientes para las (k-1-fase) fases restantes y
            # el grupo final Gₖ:  max_size = r - (k - 1 - fase)
            fases_restantes = (k - 1) - fase          # fases que aún quedan (incluyendo Gk)
            max_size = max(1, len(restantes) - fases_restantes)

            # ── LOOP INTERNO: greedy sobre los vértices restantes ─────────
            mejor_grupo = self._greedy_fase(restantes, grupos_fijos, max_size)

            grupos_fijos.append(mejor_grupo)

            # Eliminar los vértices ya asignados de la lista restante
            asignados = set(self._flatten_vertices(mejor_grupo))
            restantes = [v for v in restantes if v not in asignados]

        # Grupo final Gₖ = todo lo que quedó sin asignar
        if restantes:
            grupos_fijos.append(tuple(restantes))

        # Evaluar la k-partición completa con todos los grupos fijados
        perdida, dist = self._evaluar_k_particion(grupos_fijos)

        self.logger.critic(
            f"[KQNodes] Finalizado | φ={perdida:.6f} | grupos={len(grupos_fijos)}"
        )
        return grupos_fijos, perdida, dist

    # ══════════════════════════════════════════════════════════════════════
    #  MÓDULO B — LOOP INTERNO: GREEDY POR FASE
    # ══════════════════════════════════════════════════════════════════════

    def _greedy_fase(
        self,
        restantes: list[tuple[int, int]],
        grupos_fijos: list[tuple],
        max_size: int = INT_ZERO,
    ) -> tuple:
        """
        Aplica el criterio greedy submodular para encontrar el mejor grupo
        Gⱼ a pelar del conjunto de vértices restantes en la fase actual.

        Replica la estructura de ciclos omega/delta del QNodes original:
        - omega crece agregando un vértice en cada iteración.
        - En cada iteración se elige el δ* que minimiza (emd_unión − emd_δ).
        - Al finalizar, omega es el grupo Gⱼ para esta fase.

        A diferencia de QNodes, la función de evaluación es
        `_funcion_submodular_k()` que usa `kpartir()` y conoce los
        grupos ya fijados de fases anteriores.

        Args:
        ----
            restantes (list[tuple[int, int]]):
                Vértices aún no asignados a ningún grupo fijo.
            grupos_fijos (list[tuple]):
                Grupos determinados en fases anteriores (contexto compartido).
            max_size (int):
                Máximo de vértices que puede contener este grupo. Garantiza
                que queden suficientes vértices para las fases siguientes y
                el grupo final Gₖ. Si es 0 o negativo, no se aplica límite.

        Returns:
        -------
            tuple: El grupo Gⱼ encontrado como tupla de vértices.
        """
        # Priorizar vértices EFFECT como semilla para garantizar que el
        # grupo candidato siempre tenga al menos un NCubo (índice futuro).
        restantes_ord = sorted(
            restantes,
            key=lambda v: (v[ACTUAL] != EFFECT, v),  # EFFECT=1 primero
        )

        omega: list = [restantes_ord[INT_ZERO]]
        delta: list = list(restantes_ord[1:])

        emd_candidato: float  = INFTY_POS
        dist_candidato: Optional[NDArray] = None

        # Límite de iteraciones: min(len(delta)-1, max_size-1)
        # Permite que omega crezca hasta max_size vértices como máximo.
        # Si max_size <= 0 no aplica restricción (k=2, un solo grupo).
        limite = len(delta) - 1
        if max_size > 0:
            limite = min(limite, max_size - 1)  # -1 porque omega ya tiene 1 semilla

        # ── Ciclos greedy: agregar 1 vértice a omega por iteración ─────────
        for _ in range(limite):

            emd_local: float = INFTY_POS
            indice_mip: int  = INT_ZERO

            # ── Iteración: evaluar cada δ candidato ───────────────────────
            for k_idx in range(len(delta)):

                emd_union, emd_delta, dist_delta = self._funcion_submodular_k(
                    delta=delta[k_idx],
                    omega=omega,
                    grupos_fijos=grupos_fijos,
                    restantes=restantes,
                )
                emd_iter = emd_union - emd_delta

                if emd_iter < emd_local:
                    # Salida temprana: pérdida cero = partición perfecta
                    if emd_delta == INT_ZERO:
                        clave = self._vertices_a_clave([delta[k_idx]])
                        self.memoria_grupos_k[clave] = (INT_ZERO, dist_delta)
                        return (delta[k_idx],)

                    emd_local      = emd_iter
                    indice_mip     = k_idx
                    emd_candidato  = emd_delta
                    dist_candidato = dist_delta

            # Agregar el mejor δ a omega y eliminarlo de delta
            omega.append(delta[indice_mip])
            delta.pop(indice_mip)

        # Guardar el grupo resultante en memoria
        clave_grupo = self._vertices_a_clave(omega)
        self.memoria_grupos_k[clave_grupo] = (emd_candidato, dist_candidato)

        return tuple(omega)

    # ══════════════════════════════════════════════════════════════════════
    #  MÓDULO C — FUNCIÓN SUBMODULAR CON kpartir()
    # ══════════════════════════════════════════════════════════════════════

    def _funcion_submodular_k(
        self,
        delta:        tuple[int, int],
        omega:        list,
        grupos_fijos: list[tuple],
        restantes:    list[tuple[int, int]],
    ) -> tuple[float, float, NDArray]:
        """
        Evalúa el criterio submodular para δ respecto a Ω usando kpartir().

        Calcula y compara dos pérdidas EMD:
        1. emd_δ:     pérdida de la k-partición donde δ está solo.
        2. emd_unión: pérdida de la k-partición donde δ se une a Ω.

        La diferencia (emd_unión − emd_δ) es el "costo de agregar δ a Ω".
        El δ con menor costo se elige en el ciclo greedy.

        En ambas evaluaciones la k-partición incluye siempre:
            [grupos_fijos]  +  [candidato]  +  [cola = restantes − candidato]

        Esto garantiza que kpartir() reciba siempre una partición completa
        del subsistema, y que los resultados sean comparables entre sí.

        Args:
        ----
            delta (tuple[int, int]):
                Vértice candidato a evaluar: (tiempo, índice).
            omega (list):
                Vértices ya seleccionados en el ciclo greedy actual.
            grupos_fijos (list[tuple]):
                Grupos determinados en fases anteriores.
            restantes (list[tuple[int, int]]):
                Todos los vértices aún no asignados a ningún grupo fijo.

        Returns:
        -------
            Tupla (emd_union, emd_delta, dist_delta).
        """
        # ── Paso 1: evaluación individual de δ (con memoización) ──────────
        clave_delta = self._vertices_a_clave([delta])

        if clave_delta not in self.memoria_delta_k:
            grupos_delta = self._construir_grupos_kpartir(
                grupos_fijos=grupos_fijos,
                candidato=[delta],
                restantes=restantes,
            )
            part_delta  = self.sia_subsistema.kpartir(grupos_delta)
            dist_delta  = part_delta.distribucion_marginal()
            emd_delta   = emd_efecto(dist_delta, self.sia_dists_marginales)
            self.memoria_delta_k[clave_delta] = (emd_delta, dist_delta)
        else:
            emd_delta, dist_delta = self.memoria_delta_k[clave_delta]

        # ── Paso 2: evaluación de δ ∪ Ω ───────────────────────────────────
        candidato_union = [delta] + list(omega)

        grupos_union = self._construir_grupos_kpartir(
            grupos_fijos=grupos_fijos,
            candidato=candidato_union,
            restantes=restantes,
        )
        part_union  = self.sia_subsistema.kpartir(grupos_union)
        dist_union  = part_union.distribucion_marginal()
        emd_union   = emd_efecto(dist_union, self.sia_dists_marginales)

        return emd_union, emd_delta, dist_delta

    # ══════════════════════════════════════════════════════════════════════
    #  MÓDULO D — CONSTRUCCIÓN DE GRUPOS PARA kpartir()
    # ══════════════════════════════════════════════════════════════════════

    def _construir_grupos_kpartir(
        self,
        grupos_fijos: list[tuple],
        candidato:    list[tuple[int, int]],
        restantes:    list[tuple[int, int]],
    ) -> list[tuple[NDArray[np.int8], NDArray[np.int8]]]:
        """
        Convierte grupos de vértices (tiempo, índice) a pares (F_j, M_j)
        que acepta `System.kpartir()`.

        La partición resultante es siempre completa: todos los índices
        futuros (NCubos) y todas las dims presentes están cubiertos
        exactamente una vez, garantizando que kpartir() sea válido.

        Estructura de la k-partición construida:
            ┌──────────────────┐
            │  grupos_fijos    │  ← Gⱼ de fases anteriores
            ├──────────────────┤
            │  candidato       │  ← Grupo actual en evaluación
            ├──────────────────┤
            │  cola            │  ← restantes − fijos − candidato
            └──────────────────┘

        Args:
        ----
            grupos_fijos (list[tuple]):
                Grupos de vértices ya fijados en fases anteriores.
            candidato (list[tuple[int, int]]):
                Vértices del grupo candidato actual.
            restantes (list[tuple[int, int]]):
                Todos los vértices aún no asignados a ningún grupo fijo.

        Returns:
        -------
            Lista de pares (F_j, M_j) con dtype np.int8 para kpartir().
        """
        # Calcular la cola: lo que no está en fijos ni en el candidato
        verts_fijos     = {v for g in grupos_fijos for v in self._flatten_vertices(g)}
        verts_candidato = set(self._flatten_vertices(candidato))
        cola = [v for v in restantes if v not in verts_fijos and v not in verts_candidato]

        grupos_kpartir: list[tuple[NDArray[np.int8], NDArray[np.int8]]] = []

        # 1) Grupos ya fijados de fases anteriores
        for grupo in grupos_fijos:
            par = self._vertices_a_fj_mj(self._flatten_vertices(grupo))
            if par is not None:
                grupos_kpartir.append(par)

        # 2) Grupo candidato actual (δ solo, o δ ∪ Ω)
        par_cand = self._vertices_a_fj_mj(list(verts_candidato))
        if par_cand is not None:
            grupos_kpartir.append(par_cand)

        # 3) Cola: vértices restantes no asignados (grupo temporal Gₖ)
        if cola:
            par_cola = self._vertices_a_fj_mj(cola)
            if par_cola is not None:
                grupos_kpartir.append(par_cola)

        return grupos_kpartir

    # ══════════════════════════════════════════════════════════════════════
    #  EVALUACIÓN FINAL DE LA k-PARTICIÓN COMPLETA
    # ══════════════════════════════════════════════════════════════════════

    def _evaluar_k_particion(
        self,
        grupos_fijos: list[tuple],
    ) -> tuple[float, NDArray]:
        """
        Calcula la pérdida EMD de la k-partición completa usando kpartir().

        Llamada una sola vez al final de `algorithm()`, cuando todos los
        grupos están determinados. Reutiliza el caché de `System.memo`.

        Args:
        ----
            grupos_fijos (list[tuple]):
                Lista de k grupos de vértices (todos los grupos finales).

        Returns:
        -------
            Tupla (perdida, distribucion_marginal) de la k-partición.
        """
        grupos_kpartir = []
        for g in grupos_fijos:
            par = self._vertices_a_fj_mj(self._flatten_vertices(g))
            if par is not None:
                grupos_kpartir.append(par)

        sistema_partido = self.sia_subsistema.kpartir(grupos_kpartir)
        dist    = sistema_partido.distribucion_marginal()
        perdida = emd_efecto(dist, self.sia_dists_marginales)
        return perdida, dist

    # ══════════════════════════════════════════════════════════════════════
    #  UTILIDADES INTERNAS
    # ══════════════════════════════════════════════════════════════════════

    def _vertices_a_fj_mj(
        self,
        verts: list[tuple[int, int]],
    ) -> Optional[tuple[NDArray[np.int8], NDArray[np.int8]]]:
        """
        Separa una lista de vértices (tiempo, índice) en:
            F_j → índices de NCubos futuros  (tiempo == EFFECT)
            M_j → dims del mecanismo presente (tiempo == ACTUAL)

        Retorna None si F_j estaría vacío: un grupo sin NCubos no puede
        ser un grupo válido en kpartir() (no tendría variables futuras que
        marginalizar).

        Args:
        ----
            verts (list[tuple[int, int]]): Lista de vértices (tiempo, índice).

        Returns:
        -------
            Par (F_j, M_j) con dtype np.int8, o None si F_j es vacío.
        """
        futuros   = sorted(idx for t, idx in verts if t == EFFECT)
        presentes = sorted(idx for t, idx in verts if t == ACTUAL)

        if not futuros:
            return None  # grupo sin NCubos: ignorar en kpartir()

        return (
            np.array(futuros,   dtype=np.int8),
            np.array(presentes, dtype=np.int8),
        )

    def _flatten_vertices(
        self,
        grupo: tuple | list,
    ) -> list[tuple[int, int]]:
        """
        Aplana un grupo de vértices que puede estar anidado en múltiples
        niveles (listas de listas, tuplas de tuplas, etc.).

        Un vértice hoja es siempre una tupla (tiempo, índice) donde ambos
        elementos son enteros.

        Args:
        ----
            grupo (tuple | list): Grupo de vértices, posiblemente anidado.

        Returns:
        -------
            list[tuple[int, int]]: Lista plana de vértices hoja.
        """
        result: list[tuple[int, int]] = []
        for item in grupo:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], (int, np.integer))
                and isinstance(item[1], (int, np.integer))
            ):
                result.append((int(item[0]), int(item[1])))
            elif isinstance(item, (list, tuple)):
                result.extend(self._flatten_vertices(item))
        return result

    def _vertices_a_clave(
        self,
        verts: list,
    ) -> tuple:
        """
        Genera una clave hashable, canónica y única para un grupo de vértices.
        Usada como llave en los diccionarios de memoización.

        Args:
        ----
            verts (list): Grupo de vértices (posiblemente anidado).

        Returns:
        -------
            tuple: Clave canónica ordenada.
        """
        return tuple(sorted(self._flatten_vertices(verts)))


    def _formatear_k_particion(
        self,
        grupos: list[tuple],
    ) -> str:
        """
        Genera una representación legible de la k-partición encontrada.

        Para cada grupo Gj muestra las variables futuras (EFFECT) en
        mayúsculas y las presentes (ACTUAL) en minúsculas, separadas
        por '|', en el formato: G1:{A,B|a,b}  G2:{C|c}  ...

        Usa la misma convencion ABECEDARY del sistema:
        indice 0 -> A/a, indice 1 -> B/b, etc.

        Args:
        ----
            grupos (list[tuple]):
                Lista de k grupos, cada uno como tupla de vertices
                (tiempo, indice).

        Returns:
        -------
            str: Cadena formateada con la k-particion.
        """
        partes = []
        for j, grupo in enumerate(grupos, start=1):
            flat   = list(self._flatten_vertices(grupo))
            future = sorted(v[1] for v in flat if v[0] == EFFECT)
            actual = sorted(v[1] for v in flat if v[0] == ACTUAL)

            future_str = ",".join(ABECEDARY[i].upper() for i in future) if future else "-"
            actual_str = ",".join(ABECEDARY[i].lower() for i in actual) if actual else "-"

            partes.append(f"G{j}:{{{future_str}|{actual_str}}}")

        return "  ".join(partes)
