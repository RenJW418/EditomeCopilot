"""
Knowledge-Graph-Guided Adaptive Query Expansion  (KG-AQE)
==========================================================

Innovation: Standard RAG query expansion (HyDE) is purely LLM-based and
domain-agnostic.  KG-AQE leverages the structured knowledge graph to
*graph-attend* over related entities and inject domain-specific expansion
terms, dramatically improving recall for complex biomedical queries.

Mathematical Formulation
------------------------
Given query *q* and knowledge graph *G = (V, E)*:

1. **Entity Linking**  — Extract entity set E_q ⊂ V from q via fuzzy match.

2. **Neighbourhood Expansion** — For each entity *e ∈ E_q*, retrieve its
   k-hop neighbourhood N_k(e) from G, weighted by inverse path length:

       w(e, n) = 1 / (1 + hop_distance(e, n))

3. **Graph Attention Scoring** — For each neighbour *n ∈ N_k(e)*,
   compute an attention score based on:
     - Edge type relevance to query (relation_weight)
     - Node degree centrality (hub importance)
     - Type alignment (prioritise same-type or therapeutically-linked types)

       A(q, n) = softmax(  w_edge × rel_score(edge_type)
                          + w_deg  × log(1 + degree(n))
                          + w_type × type_bonus(n) )

4. **Query Reformulation** — Select top-M expansion entities and inject
   into the search query:

       q_expanded = q ⊕ " " ⊕ join(top_M_entities)

   Additionally produce *expansion sub-queries* for KG-discovered
   dimensions the user didn't explicitly mention.

5. **Adaptive Depth** — The expansion depth (k hops, M terms) adapts to
   query specificity:
     - Specific query (many matched entities) → shallow (1 hop, few terms)
     - Vague query (few/no matched entities) → deep (2 hops, more terms)

Cite as Algorithm 2 in the paper.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Relation-Type Relevance Weights (hand-curated for gene editing domain)
# ─────────────────────────────────────────────────────────────────────────────
_RELATION_WEIGHTS: Dict[str, float] = {
    "applied_to":        1.0,   # Technology → Disease
    "targets_gene":      0.95,
    "associated_gene":   0.90,
    "uses_technology":   0.85,
    "targets_disease":   0.85,
    "delivered_by":      0.80,
    "treated_via":       0.75,
    "applied_in":        0.70,
    "expressed_in":      0.65,
    "has_product":       0.60,
    "uses_delivery":     0.55,
    "approves_product":  0.50,
    "causal_gene":       0.90,
    "subtype_of":        0.45,
    "is_variant_of":     0.40,
}

# Type alignment bonus: when querying about a disease, gene neighbours are
# more valuable than delivery neighbours, etc.
_TYPE_AFFINITY: Dict[str, Dict[str, float]] = {
    "Technology": {"Disease": 1.0, "Gene": 0.9, "Delivery": 0.7, "CellType": 0.6, "ClinicalProduct": 0.8},
    "Disease":    {"Technology": 1.0, "Gene": 1.0, "ClinicalProduct": 0.9, "Delivery": 0.5, "CellType": 0.6},
    "Gene":       {"Technology": 0.9, "Disease": 1.0, "CellType": 0.8, "Delivery": 0.4, "ClinicalProduct": 0.5},
    "Delivery":   {"Technology": 1.0, "Disease": 0.5, "CellType": 0.6, "ClinicalProduct": 0.7, "Gene": 0.4},
    "ClinicalProduct": {"Technology": 0.9, "Disease": 1.0, "Gene": 0.8, "Delivery": 0.7, "CellType": 0.5},
    "CellType":   {"Technology": 0.7, "Disease": 0.6, "Gene": 0.9, "Delivery": 0.5, "ClinicalProduct": 0.4},
}


# ─────────────────────────────────────────────────────────────────────────────
# Entity Linker
# ─────────────────────────────────────────────────────────────────────────────
class _EntityLinker:
    """
    Fast fuzzy entity linker that matches query tokens to KG node IDs.
    Uses normalised substring matching + alias table.
    """

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph
        self._index: Dict[str, str] = {}  # lowered alias → canonical node ID
        for nid, data in graph.nodes(data=True):
            nid_str = str(nid)
            self._index[nid_str.lower()] = nid_str
            # Also index partial tokens (e.g. "CRISPR-Cas9" → "crispr", "cas9")
            for token in re.split(r"[\s\-/()]+", nid_str):
                if len(token) >= 3:
                    self._index[token.lower()] = nid_str

    def link(self, query: str) -> List[Tuple[str, str]]:
        """
        Return matched (node_id, node_type) pairs from the query.
        Sorts by match length descending so longer names take priority.
        """
        q_lower = query.lower()
        matched: Dict[str, str] = {}
        # Sort by key length descending (prefer longer matches)
        for alias in sorted(self._index, key=len, reverse=True):
            if alias in q_lower and self._index[alias] not in matched:
                nid = self._index[alias]
                ntype = self._graph.nodes[nid].get("type", "Unknown")
                matched[nid] = ntype
        return list(matched.items())


# ─────────────────────────────────────────────────────────────────────────────
# KG-AQE Engine
# ─────────────────────────────────────────────────────────────────────────────
class KGQueryExpander:
    """
    Knowledge-Graph-Guided Adaptive Query Expansion.

    Parameters
    ----------
    graph : nx.DiGraph
        The Gene Editing Knowledge Graph (loaded from kg.json).
    max_hops : int
        Maximum neighbourhood depth (default 2).
    max_expansion_terms : int
        Maximum number of expansion entities to inject (default 8).
    w_edge : float
        Weight for edge-type relevance in attention (default 0.40).
    w_deg : float
        Weight for degree centrality (default 0.30).
    w_type : float
        Weight for type affinity (default 0.30).
    enabled : bool
        Master switch (for ablation).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        max_hops: int = 2,
        max_expansion_terms: int = 8,
        w_edge: float = 0.40,
        w_deg: float = 0.30,
        w_type: float = 0.30,
        enabled: bool = True,
    ):
        self.graph = graph
        self.max_hops = max_hops
        self.max_expansion_terms = max_expansion_terms
        self.w_edge = w_edge
        self.w_deg = w_deg
        self.w_type = w_type
        self.enabled = enabled

        self._linker = _EntityLinker(graph)

        # Precompute degree centrality (normalised)
        self._degree_cent: Dict[str, float] = {}
        if len(graph) > 0:
            self._degree_cent = nx.degree_centrality(graph)

    # ─────────────────────────────────────────────────
    # Core expansion logic
    # ─────────────────────────────────────────────────
    def expand(self, query: str) -> Dict[str, Any]:
        """
        Expand a query using KG neighbourhood attention.

        Returns
        -------
        dict with keys:
            - query_original : str
            - query_expanded : str
            - linked_entities : list of (nid, type)
            - expansion_terms : list of dict {entity, type, attention, hop}
            - sub_queries : list of str (optional KG-discovered dimensions)
            - expansion_depth : int (adaptive hops used)
            - expansion_stats : dict
        """
        if not self.enabled or len(self.graph) == 0:
            return {
                "query_original": query,
                "query_expanded": query,
                "linked_entities": [],
                "expansion_terms": [],
                "sub_queries": [],
                "expansion_depth": 0,
                "expansion_stats": {},
            }

        # Step 1: Entity Linking
        linked = self._linker.link(query)
        entity_types = {ntype for _, ntype in linked}

        # Step 2: Adaptive depth — more entities → shallower expansion
        if len(linked) >= 3:
            actual_hops = 1
            actual_max_terms = max(3, self.max_expansion_terms // 2)
        elif len(linked) == 0:
            actual_hops = self.max_hops
            actual_max_terms = self.max_expansion_terms
        else:
            actual_hops = self.max_hops
            actual_max_terms = self.max_expansion_terms

        # Step 3: Collect k-hop neighbours with attention scoring
        candidates: Dict[str, Dict] = {}  # nid → {score, type, hop, edge_type}
        seen_entities = {nid for nid, _ in linked}

        for seed_nid, seed_type in linked:
            self._bfs_attend(
                seed_nid, seed_type, actual_hops, seen_entities, entity_types, candidates
            )

        # Step 4: Rank expansion candidates by attention score
        ranked = sorted(candidates.values(), key=lambda x: x["attention"], reverse=True)
        top_expansions = ranked[:actual_max_terms]

        # Step 5: Build expanded query string
        expansion_entities = [t["entity"] for t in top_expansions]
        query_expanded = query
        if expansion_entities:
            query_expanded = query + " " + " ".join(expansion_entities)

        # Step 6: Generate KG-discovered sub-queries
        sub_queries = self._generate_sub_queries(query, linked, top_expansions)

        return {
            "query_original": query,
            "query_expanded": query_expanded,
            "linked_entities": linked,
            "expansion_terms": top_expansions,
            "sub_queries": sub_queries,
            "expansion_depth": actual_hops,
            "expansion_stats": {
                "entities_linked": len(linked),
                "candidates_found": len(candidates),
                "terms_selected": len(top_expansions),
                "depth_used": actual_hops,
            },
        }

    def _bfs_attend(
        self,
        seed: str,
        seed_type: str,
        max_hops: int,
        seen: Set[str],
        query_types: set,
        candidates: Dict[str, Dict],
    ):
        """BFS from seed node with attention-based scoring at each hop."""
        frontier = [(seed, 0)]
        visited = {seed}

        while frontier:
            node, hop = frontier.pop(0)
            if hop >= max_hops:
                continue

            # Explore both in-edges and out-edges (undirected walk)
            for _, nbr, edata in self.graph.out_edges(node, data=True):
                self._process_neighbour(
                    nbr, edata, hop + 1, seed_type, query_types,
                    seen, visited, candidates, frontier
                )
            for pred, _, edata in self.graph.in_edges(node, data=True):
                self._process_neighbour(
                    pred, edata, hop + 1, seed_type, query_types,
                    seen, visited, candidates, frontier
                )

    def _process_neighbour(
        self, nbr, edata, hop, seed_type, query_types,
        seen, visited, candidates, frontier
    ):
        """Score a single neighbour and optionally add to frontier."""
        nbr_str = str(nbr)
        if nbr_str in visited:
            return
        visited.add(nbr_str)

        nbr_type = self.graph.nodes.get(nbr, {}).get("type", "Unknown")
        edge_rel = edata.get("relation", "unknown")

        # Compute attention components
        # a) Edge relevance
        rel_score = _RELATION_WEIGHTS.get(edge_rel, 0.3)

        # b) Degree centrality (log-scaled)
        deg = self._degree_cent.get(nbr_str, 0.0)
        deg_score = math.log1p(deg * len(self.graph)) / math.log1p(len(self.graph))

        # c) Type affinity
        type_bonus = 0.5  # default
        if seed_type in _TYPE_AFFINITY:
            type_bonus = _TYPE_AFFINITY[seed_type].get(nbr_type, 0.3)

        # Hop penalty: discount by 1/(1+hop)
        hop_penalty = 1.0 / (1.0 + hop)

        # Combined attention
        attention = (
            self.w_edge * rel_score
            + self.w_deg * deg_score
            + self.w_type * type_bonus
        ) * hop_penalty

        if nbr_str not in seen:
            if nbr_str not in candidates or candidates[nbr_str]["attention"] < attention:
                candidates[nbr_str] = {
                    "entity": nbr_str,
                    "type": nbr_type,
                    "attention": round(attention, 4),
                    "hop": hop,
                    "edge_type": edge_rel,
                }

        frontier.append((nbr_str, hop))

    def _generate_sub_queries(
        self,
        query: str,
        linked: List[Tuple[str, str]],
        expansions: List[Dict],
    ) -> List[str]:
        """
        Generate sub-queries for KG-discovered dimensions.

        If the query asks about a technology but KG reveals it has clinical
        products, safety concerns, or delivery requirements, create sub-queries
        for those dimensions.
        """
        sub_queries: List[str] = []
        q_lower = query.lower()

        # Identify query entity types
        linked_types = {t for _, t in linked}
        expansion_types = {e["type"] for e in expansions}

        # If query has Technology but expansion found Disease links → add disease sub-query
        if "Technology" in linked_types and "Disease" in expansion_types:
            diseases = [e["entity"] for e in expansions if e["type"] == "Disease"][:2]
            if diseases:
                sub_queries.append(
                    f"{query} clinical applications {' '.join(diseases)}"
                )

        # If query has Disease but expansion found Gene/Technology → add strategy sub-query
        if "Disease" in linked_types and "Technology" in expansion_types:
            techs = [e["entity"] for e in expansions if e["type"] == "Technology"][:2]
            if techs:
                sub_queries.append(
                    f"{query} gene editing strategy {' '.join(techs)}"
                )

        # If clinical products found → add clinical evidence sub-query
        if "ClinicalProduct" in expansion_types:
            products = [e["entity"] for e in expansions if e["type"] == "ClinicalProduct"][:2]
            if products:
                sub_queries.append(
                    f"{query} clinical trial results {' '.join(products)}"
                )

        # Safety dimension — always important in gene editing
        if not any(k in q_lower for k in ["safety", "risk", "off-target", "安全", "风险"]):
            if "Technology" in linked_types or "ClinicalProduct" in linked_types:
                sub_queries.append(f"{query} safety off-target risk")

        return sub_queries[:3]  # Limit to 3 sub-queries

    # ── Report generation (for paper / debugging) ──────────────
    @staticmethod
    def format_expansion_report(expansion_result: Dict) -> str:
        """Generate a human-readable expansion report."""
        r = expansion_result
        lines = ["### 🔗 KG-AQE Query Expansion Report"]
        lines.append(f"**Linked Entities**: {len(r['linked_entities'])}")
        for nid, ntype in r["linked_entities"]:
            lines.append(f"  - {nid} ({ntype})")
        lines.append(f"**Expansion Depth**: {r['expansion_depth']} hops")
        lines.append(f"**Expansion Terms**: {len(r['expansion_terms'])}")
        for t in r["expansion_terms"][:8]:
            lines.append(
                f"  - {t['entity']} ({t['type']}) "
                f"attn={t['attention']:.3f} hop={t['hop']}"
            )
        if r["sub_queries"]:
            lines.append(f"**KG-Discovered Sub-queries**:")
            for sq in r["sub_queries"]:
                lines.append(f"  ◦ {sq}")
        return "\n".join(lines)
