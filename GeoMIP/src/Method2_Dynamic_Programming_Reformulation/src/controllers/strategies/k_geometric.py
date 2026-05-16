"""
k_geometric.py — Extensión de GeometricSIA para k-particiones (KGeoMIP).

Estrategia: Tabla de Costos Geométrica con k Grupos.
Universidad de Caldas — Proyecto K-QGMIP — 2026-1

Arquitectura (hereda de GeometricSIA y sobreescribe lo mínimo necesario):
    aplicar_estrategia(k=k)   ← punto de entrada; prepara subsistema y llama find_mip()
    find_mip()                ← BFS/DP heredado + identificar_particiones_optimas(k) + evaluación kpartir()
    └─ identificar_particiones_optimas()
          Capa 1 — Cuantiles del vector de costos tx(s₀, s_final)
          Capa 2 — k-1 estados de referencia como k-medoids sobre el hipercubo

Reutiliza íntegramente de GeometricSIA:
    - calcular_costos_nivel()    (construcción BFS/DP de la tabla)
    - calcular_costo()           (fórmula recursiva tx(i,j))
    - hamming()                  (distancia de Hamming binaria)
    - sia_preparar_subsistema()  (condicionamiento + substracción)
"""

import time
import numpy as np
from itertools import combinations
from typing import List

from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto, ABECEDARY
from src.constants.base import ACTUAL, EFECTO
from src.constants.models import GEOMETRIC_LABEL


# ─── Constantes del módulo ─────────────────────────────────────────────────
KGEOMIP_LABEL = "KGeoMIP"
KGEOMIP_TAG   = "kgeomip_strategy"


class KGeometricSIA(GeometricSIA):
    """
    KGeoMIP: extensión de GeometricSIA para k-particiones (k ≥ 2).

    Reutiliza la construcción BFS/DP de la tabla de costos geométrica
    y extiende la identificación de particiones de bipartición a
    k grupos mediante dos capas de candidatos:

    Capa 1 — División por cuantiles del vector de costos final
        Cada variable futura se ordena según tx(s₀, s_final)[idx] y se
        reparte en k grupos: primero por división uniforme y luego por
        los k-1 saltos más grandes entre costos consecutivos.

    Capa 2 — k-medoids sobre estados intermedios del hipercubo
        Se eligen k-1 estados de referencia {r₁,...,r_{k-1}} de los
        niveles intermedios del hipercubo. Cada variable futura idx se
        asigna al grupo j* con: j* = argmin_j tx(s₀, rⱼ)[idx].
        El mecanismo de cada grupo se asigna según los bits que cada
        estado de referencia tiene en común con el estado inicial.

    La evaluación de cada candidato usa System.kpartir() (Módulo A)
    en lugar de bipartir(), generalizando la métrica EMD al caso k ≥ 2.

    Args:
    ----
        gestor (Manager): Gestor de la red (igual que GeometricSIA).

    Attributes:
    ----------
        k (int): Número de grupos objetivo de la k-partición.
    """

    def __init__(self, gestor):
        super().__init__(gestor)
        self.k: int = 2

    # ══════════════════════════════════════════════════════════════════════
    #  PUNTO DE ENTRADA PÚBLICO
    # ══════════════════════════════════════════════════════════════════════

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 2,
    ):
        """
        Prepara el subsistema y lanza KGeoMIP para k grupos.

        Replica la firma de GeometricSIA.aplicar_estrategia() añadiendo
        el parámetro k. El BFS/DP de la tabla de costos es idéntico al
        padre; sólo cambia la identificación y evaluación de candidatos.

        Args:
        ----
            condicion (str):  Bits de condicionamiento.
            alcance (str):    Bits del alcance futuro.
            mecanismo (str):  Bits del mecanismo presente.
            tpm (np.ndarray): Matriz de Probabilidad de Transición.
            k (int):          Número de grupos (k ≥ 2).

        Returns:
        -------
            Solution: k-partición de mínima pérdida EMD.

        Raises:
        ------
            ValueError: Si k < 2 o k > número de variables futuras.
        """
        if k < 2:
            raise ValueError(f"k debe ser ≥ 2. Recibido: k={k}")

        self.k = k
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        n_futuras = self.sia_subsistema.indices_ncubos.size
        if k > n_futuras:
            raise ValueError(
                f"k={k} supera las variables futuras del subsistema ({n_futuras}). "
                f"Reduzca k o amplíe el alcance."
            )

        # Aplanar datos de NCubos (igual que GeometricSIA)
        self._flat_data = [
            ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos
        ]

        futuro = tuple(
            (EFECTO, int(e)) for e in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, int(a)) for a in self.sia_subsistema.dims_ncubos
        )
        self.vertices = set(presente + futuro)

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final   = 1 - self.estado_inicial

        mip = self.find_mip()

        # Formatear la k-particion como cadena legible
        fmt_mip = self._formatear_k_particion_geo(mip)

        from src.models.core.solution import Solution
        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  FIND_MIP PARA k GRUPOS (sobreescribe el de bipartición)
    # ══════════════════════════════════════════════════════════════════════

    def find_mip(self):
        """
        Encuentra la k-partición de mínima pérdida usando la tabla de costos.

        Estructura:
        1. Construye la tabla de costos BFS/DP (heredado de GeometricSIA).
        2. Genera candidatos con identificar_particiones_optimas() (2 capas).
        3. Evalúa cada candidato con kpartir() + emd_efecto().
        4. Retorna la clave del candidato con menor EMD.

        Returns:
        -------
            tuple: Clave de la k-partición óptima en memoria_particiones.
        """
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos    = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones = {}
        self.tabla_transiciones[
            (tuple(self.caminos[0][0]), tuple(self.caminos[0][0]))
        ] = [0.0] * len(self.idx_ncubos)

        # ── BFS/DP: construir tabla de costos nivel por nivel ─────────────
        n_niveles = len(self.estado_inicial)
        for nivel in range(1, n_niveles + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)

        # ── Generar candidatos de k-partición ─────────────────────────────
        candidatos_k = self.identificar_particiones_optimas()

        # ── Evaluar cada candidato con kpartir() ──────────────────────────
        for candidato in candidatos_k:
            # candidato = lista de k [presentes_j, futuros_j]
            # Convertir a formato (F_j, M_j) para kpartir()
            try:
                grupos_kpartir = self._candidato_a_grupos_kpartir(candidato)
                dist = (
                    self.sia_subsistema
                    .kpartir(grupos_kpartir)
                    .distribucion_marginal()
                )
                emd = emd_efecto(dist, self.sia_dists_marginales)
                clave = self._candidato_a_clave(candidato)
                self.memoria_particiones[clave] = (emd, dist)
            except (ValueError, IndexError):
                # Candidato inválido (grupo vacío, cobertura incompleta, etc.)
                continue

        if not self.memoria_particiones:
            raise RuntimeError(
                "KGeoMIP no generó ningún candidato válido. "
                "Comprueba el subsistema y el valor de k."
            )

        return min(
            self.memoria_particiones,
            key=lambda clave: self.memoria_particiones[clave][0],
        )

    # ══════════════════════════════════════════════════════════════════════
    #  IDENTIFICAR PARTICIONES ÓPTIMAS (sobreescribe la de bipartición)
    # ══════════════════════════════════════════════════════════════════════

    def identificar_particiones_optimas(self) -> List[List[List[int]]]:
        """
        Genera candidatos de k-partición mediante dos capas geométricas.

        Cada candidato es una lista de k pares [presentes_j, futuros_j]
        donde futuros_j ⊂ range(M) y presentes_j ⊂ range(N).

        CAPA 1 — División por cuantiles del vector de costos final
            Usa tx(s₀, s_final) para ordenar las M variables futuras
            por costo. Genera dos particiones:
                (a) División uniforme: k grupos de M/k variables cada uno.
                (b) División por gaps: cortes en los k-1 saltos más grandes
                    entre costos consecutivos (divisiones naturales del espacio).

        CAPA 2 — k-medoids sobre estados intermedios del hipercubo
            Toma combinaciones de k-1 estados de referencia de los niveles
            1..⌊N/2⌋ del hipercubo. Para cada variable futura idx:
                j*(idx) = argmin_j  tx(s₀, rⱼ)[idx]
            El último grupo (Gₖ) recibe lo que ningún rⱼ reclamó.
            El mecanismo de Gⱼ se asigna por los bits que rⱼ comparte
            con el estado inicial (variables presentes "del mismo lado").

        Returns:
        -------
            List[List[List[int]]]:
                Lista de candidatos. Cada candidato es:
                [[presentes_1, futuros_1], ..., [presentes_k, futuros_k]]
        """
        k     = self.k
        M     = len(self.idx_ncubos)          # # variables futuras
        N     = len(self.estado_inicial)      # # variables presentes
        s0    = self.caminos[0][0]            # estado inicial como lista
        s_fin = self.estado_final.tolist()    # estado final como lista

        clave_fin = (tuple(s0), tuple(s_fin))
        costos_fin = self.tabla_transiciones.get(clave_fin, [0.0] * M)

        todos_presentes = list(range(N))
        candidatos: List[List[List[int]]] = []

        # ── CAPA 1a: División uniforme por cuantiles ────────────────────
        orden = sorted(range(M), key=lambda i: costos_fin[i])
        chunks_uniformes = self._dividir_en_k(orden, k)
        if all(len(c) > 0 for c in chunks_uniformes):
            cand = [[todos_presentes, list(c)] for c in chunks_uniformes]
            candidatos.append(cand)

        # ── CAPA 1b: División por gaps (diferencias naturales) ──────────
        if M > k and k >= 2:
            costos_ord = [costos_fin[i] for i in orden]
            diffs = [(costos_ord[i+1] - costos_ord[i], i+1)
                     for i in range(M - 1)]
            # Los k-1 puntos de corte más grandes
            puntos_corte = sorted(
                pos for _, pos in
                sorted(diffs, reverse=True)[:k - 1]
            )
            grupos_gap = []
            prev = 0
            for corte in puntos_corte:
                grupos_gap.append([orden[i] for i in range(prev, corte)])
                prev = corte
            grupos_gap.append([orden[i] for i in range(prev, M)])

            if len(grupos_gap) == k and all(len(g) > 0 for g in grupos_gap):
                cand = [[todos_presentes, list(g)] for g in grupos_gap]
                candidatos.append(cand)

        # ── CAPA 2: k-medoids con estados de referencia ─────────────────
        mitad = max(1, len(self.estado_inicial) // 2)

        # Reunir estados intermedios de los niveles 1..mitad
        estados_ref: List[tuple] = []
        for nivel in range(1, mitad + 1):
            for estado in self.caminos.get(nivel, []):
                estados_ref.append(tuple(estado))

        # Tomar combinaciones de k-1 estados de referencia
        # Para evitar explosión combinatoria, se limita a los primeros estados
        max_refs = min(len(estados_ref), 4 * k)
        estados_ref = estados_ref[:max_refs]

        refs_vistos: set = set()
        for refs in combinations(estados_ref, min(k - 1, len(estados_ref))):
            clave_refs = frozenset(refs)
            if clave_refs in refs_vistos:
                continue
            refs_vistos.add(clave_refs)

            # Asignar cada variable futura al grupo con menor costo
            grupos_futuros = [[] for _ in range(k)]
            grupos_presentes = self._mecanismo_desde_refs(refs, s0, N, k)

            for idx in range(M):
                costos_ref_i = []
                for ref in refs:
                    c_ref = self.tabla_transiciones.get(
                        (tuple(s0), ref), [float("inf")] * M
                    )
                    costos_ref_i.append(c_ref[idx])
                # Ultimo grupo recibe el costo del estado final
                costos_ref_i.append(costos_fin[idx])

                j_star = costos_ref_i.index(min(costos_ref_i))
                grupos_futuros[j_star].append(idx)

            # Solo agregar si todos los grupos tienen al menos una futura
            if all(len(g) > 0 for g in grupos_futuros):
                cand = [
                    [grupos_presentes[j], grupos_futuros[j]]
                    for j in range(k)
                ]
                candidatos.append(cand)

        # Fallback: si no hay candidatos válidos, usar la división uniforme
        if not candidatos:
            fallback = self._dividir_en_k(list(range(M)), k)
            candidatos.append([[todos_presentes, list(c)] for c in fallback])

        return candidatos

    # ══════════════════════════════════════════════════════════════════════
    #  MÉTODOS AUXILIARES PRIVADOS
    # ══════════════════════════════════════════════════════════════════════

    def _mecanismo_desde_refs(
        self,
        refs: tuple,
        s0: list,
        N: int,
        k: int,
    ) -> List[List[int]]:
        """
        Asigna cada variable presente al grupo cuyo estado de referencia
        primero cambia ese bit respecto al estado inicial.

        Para el grupo final Gₖ, recibe todas las variables presentes no
        reclamadas exclusivamente por un estado de referencia anterior.

        Args:
        ----
            refs: Tupla de k-1 estados de referencia (cada uno como tuple).
            s0:   Estado inicial como lista de ints.
            N:    Número de variables presentes.
            k:    Número de grupos.

        Returns:
        -------
            List[List[int]]: k listas de índices de variables presentes.
        """
        grupos = [[] for _ in range(k)]
        asignado = [False] * N

        for j, ref in enumerate(refs):
            for i in range(N):
                if not asignado[i] and ref[i] != s0[i]:
                    grupos[j].append(i)
                    asignado[i] = True

        # Grupo final Gₖ recibe los no asignados
        grupos[k - 1].extend(i for i in range(N) if not asignado[i])

        # Si algún grupo de referencia quedó vacío en presentes,
        # le asignamos todas las presentes (conservativo)
        todos = list(range(N))
        return [g if g else todos for g in grupos]

    def _dividir_en_k(self, lista: list, k: int) -> List[list]:
        """
        Divide una lista en k sublistas de tamaño tan uniforme como sea posible.

        Args:
        ----
            lista: Elementos a dividir.
            k:     Número de grupos.

        Returns:
        -------
            List[list]: k sublistas.
        """
        n = len(lista)
        base, extra = divmod(n, k)
        grupos = []
        inicio = 0
        for j in range(k):
            tam = base + (1 if j < extra else 0)
            grupos.append(lista[inicio: inicio + tam])
            inicio += tam
        return grupos

    def _candidato_a_grupos_kpartir(
        self,
        candidato: List[List[int]],
    ) -> List[tuple]:
        """
        Convierte un candidato [[pres_j, fut_j], ...] al formato
        [(F_j, M_j), ...] que espera System.kpartir().

        Args:
        ----
            candidato: k pares [presentes_j, futuros_j].

        Returns:
        -------
            List[tuple]: k pares (array_F_j, array_M_j).

        Raises:
        ------
            ValueError: Si algún grupo futuro está vacío.
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos

        grupos = []
        for pres_j, fut_j in candidato:
            if not fut_j:
                raise ValueError("Grupo futuro vacío en candidato.")
            F_j = indices_ncubos[np.array(fut_j, dtype=int)]
            M_j = dims_ncubos[np.array(pres_j, dtype=int)] if pres_j else np.array([], dtype=np.int8)
            grupos.append((F_j, M_j))
        return grupos

    def _candidato_a_clave(self, candidato: List[List[int]]) -> tuple:
        """
        Genera una clave hashable y canónica para un candidato de k-partición.
        Se usa como llave en memoria_particiones.

        Returns:
        -------
            tuple: Clave ordenada de (tiempo, índice) de todos los grupos.
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos

        vertices = []
        for pres_j, fut_j in candidato:
            for idx in fut_j:
                vertices.append((1, int(indices_ncubos[idx])))
            for idx in pres_j:
                vertices.append((0, int(dims_ncubos[idx])))
        return tuple(sorted(set(vertices)))

    def _formatear_k_particion_geo(self, clave: tuple) -> str:
        """
        Genera una representación legible de la k-partición en formato
        G1:{A,B|a,b}  G2:{C|c}  ...

        La clave es una tupla de (tiempo, índice) tal como la genera
        _candidato_a_clave(). Se re-agrupa por grupos de futuras contiguas.

        Args:
        ----
            clave (tuple): Clave canónica de la k-partición.

        Returns:
        -------
            str: Cadena formateada.
        """
        # Reconstruir los grupos desde memoria para mostrar agrupación
        # Para simplicidad usamos el candidato evaluado (la clave completa)
        futuras = sorted(idx for t, idx in clave if t == 1)
        presentes = sorted(idx for t, idx in clave if t == 0)
        fut_str = ",".join(ABECEDARY[i].upper() for i in futuras) if futuras else "-"
        pre_str = ",".join(ABECEDARY[i].lower() for i in presentes) if presentes else "-"
        return f"k={self.k} | MIP: {{{fut_str}|{pre_str}}}"

# Alias canónico — permite: from k_geometric import KGeoMIP
KGeoMIP = KGeometricSIA
