"""
Triple Graph — Three-Layer Knowledge Graph Construction
=======================================================
Inspired by: MedGraphRAG (Wu et al. 2024)
  "MedGraphRAG: Towards Safe Medical Large Language Model via
   Graph RAG" — Triple Graph = Document Graph + Entity-Relation Graph
   + Controlled Vocabulary Graph.

Adapted for Gene Editing:
  Layer 1 (Bottom): Literature Entity Graph
    — Entities extracted from 86K papers (genes, diseases, tools, etc.)
    — Edges = co-occurrence / relation extraction
  Layer 2 (Middle): Authoritative Knowledge Graph
    — Curated from GEAlmanac, reviews, guidelines
    — Edges = technology-disease, tool-target, mechanism relationships
  Layer 3 (Top): Controlled Vocabulary / Ontology
    — Gene Ontology (GO), MeSH, OMIM, HGNC gene symbols
    — Cross-layer links from entity mentions → ontology terms
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None


# ─────────────────────────────────────────────────────────────────────────────
# Controlled Vocabulary / Ontology Stubs
# ─────────────────────────────────────────────────────────────────────────────
GENE_EDITING_ONTOLOGY = {
    # Technology hierarchy
    "Gene Editing": {
        "CRISPR Systems": ["SpCas9", "SaCas9", "Cas12a", "Cas13", "CasRx", "dCas9"],
        "Base Editing": ["CBE", "ABE", "BE4max", "ABE8e", "ABE7.10"],
        "Prime Editing": ["PE2", "PE3", "PEmax", "PE5max", "twinPE"],
        "Other Nucleases": ["TALEN", "ZFN", "Meganuclease"],
        "Epigenome Editing": ["CRISPRi", "CRISPRa", "dCas9-KRAB", "dCas9-VPR"],
        "RNA Editing": ["ADAR1", "ADAR2", "REPAIR", "RESCUE"],
    },
    # Delivery hierarchy
    "Delivery": {
        "Viral": ["AAV", "AAV2", "AAV5", "AAV8", "AAV9", "Lentivirus", "Adenovirus"],
        "Non-Viral": ["LNP", "Electroporation", "Microinjection", "RNP"],
    },
    # Disease categories
    "Disease Category": {
        "Hematological": ["Sickle Cell Disease", "Beta-Thalassemia", "Hemophilia A", "Hemophilia B"],
        "Metabolic": ["Familial Hypercholesterolemia", "GSD Ia", "PKU", "Tyrosinemia"],
        "Neurological": ["Huntington", "ALS", "SMA", "DMD"],
        "Ocular": ["LCA10", "Retinitis Pigmentosa", "Stargardt"],
        "Immunological": ["SCID", "Wiskott-Aldrich", "HIV"],
        "Oncological": ["CAR-T (B-ALL)", "CAR-T (DLBCL)", "Solid Tumors"],
    },
    # Measurement / Outcome
    "Outcome Type": {
        "Editing Efficiency": ["On-target indel%", "HDR%", "Base conversion%", "Prime edit%"],
        "Safety Metrics": ["Off-target frequency", "Chromosomal abnormality", "Bystander editing"],
        "Clinical Endpoints": ["Transfusion independence", "Hb level", "LDL reduction", "Visual acuity"],
    },
}

# MeSH-like controlled vocabulary for gene editing
MESH_GE_TERMS: Dict[str, List[str]] = {
    "D000077215": ["CRISPR-Cas Systems"],
    "D064113": ["CRISPR-Associated Proteins"],
    "D016678": ["Genome"],
    "D015316": ["Genetic Therapy"],
    "D005796": ["Genes"],
    "D009154": ["Mutation"],
    "D018389": ["Codon, Nonsense"],
    "D016350": ["RNA Editing"],
    "D053818": ["Guide RNA"],
}


class TripleGraph:
    """Three-layer knowledge graph for gene-editing domain."""

    def __init__(self, persist_path: Optional[str] = None):
        if nx is None:
            raise ImportError("networkx is required: pip install networkx")

        # Three layers, each as a NetworkX DiGraph
        self.L1_literature = nx.DiGraph()     # Entity graph from papers
        self.L2_authority = nx.DiGraph()       # Curated authoritative knowledge
        self.L3_ontology = nx.DiGraph()        # Controlled vocabulary

        # Cross-layer edges (entity → ontology mapping)
        self.cross_links: List[Tuple[str, str, str]] = []  # (L1/L2 node, L3 node, relation)

        self.persist_path = persist_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_base", "triple_graph.json"
        )

        self._build_ontology_layer()
        self._load_persisted()

    # ─── Layer 3: Build Ontology ──────────────────────────────────────────

    def _build_ontology_layer(self):
        """Populate L3 with controlled vocabulary."""
        for category, subcats in GENE_EDITING_ONTOLOGY.items():
            self.L3_ontology.add_node(category, node_type="category", layer=3)
            for subcat, terms in subcats.items():
                self.L3_ontology.add_node(subcat, node_type="subcategory", layer=3, parent=category)
                self.L3_ontology.add_edge(category, subcat, rel_type="has_subcategory")
                for term in terms:
                    self.L3_ontology.add_node(term, node_type="term", layer=3, parent=subcat)
                    self.L3_ontology.add_edge(subcat, term, rel_type="has_term")

    # ─── Layer 2: Build Authority from GEAlmanac ──────────────────────────

    def build_authority_from_almanac(self, almanac) -> int:
        """
        Populate L2 from GEAlmanac data.

        Parameters
        ----------
        almanac : GEAlmanac instance

        Returns
        -------
        edge_count : int
        """
        edge_count = 0

        # Technology → Disease edges from tech_disease matrix
        for entry in almanac.tech_disease:
            tech = entry["technology"]
            disease = entry["disease"]
            gene = entry["target_gene"]
            self.L2_authority.add_node(tech, node_type="technology", layer=2)
            self.L2_authority.add_node(disease, node_type="disease", layer=2)
            self.L2_authority.add_node(gene, node_type="gene", layer=2)

            self.L2_authority.add_edge(tech, disease, rel_type="treats",
                                       evidence_level=entry.get("evidence_level"),
                                       efficiency=entry.get("best_efficiency"))
            self.L2_authority.add_edge(tech, gene, rel_type="targets",
                                       strategy=entry.get("mutation_strategy"))
            self.L2_authority.add_edge(gene, disease, rel_type="involved_in")
            edge_count += 3

            # Cross-link to ontology
            self._add_cross_link(tech, "CRISPR Systems" if "CRISPR" in tech else "Base Editing" if "Base" in tech else "Prime Editing" if "Prime" in tech else "Gene Editing", "is_instance_of")

        # Clinical trials
        for trial in almanac.clinical_trials:
            trial_id = trial["nct_id"]
            self.L2_authority.add_node(trial_id, node_type="clinical_trial", layer=2, **{k: v for k, v in trial.items() if k != "nct_id"})
            tech = trial.get("technology", "")
            disease = trial.get("disease", "")
            if tech:
                self.L2_authority.add_node(tech, node_type="technology", layer=2)
                self.L2_authority.add_edge(trial_id, tech, rel_type="uses_technology")
                edge_count += 1
            if disease:
                self.L2_authority.add_node(disease, node_type="disease", layer=2)
                self.L2_authority.add_edge(trial_id, disease, rel_type="targets_disease")
                edge_count += 1

        return edge_count

    # ─── Layer 1: Ingest Literature Entities ──────────────────────────────

    def ingest_literature_entities(
        self,
        entities: List[Dict[str, Any]],
        source_id: str = "",
    ) -> int:
        """
        Ingest NER-extracted entities from literature.

        Parameters
        ----------
        entities : list of dict
            [{name, type, context, pmid}, ...]
        source_id : str
            Paper identifier (PMID or DOI).

        Returns
        -------
        edge_count : int
        """
        edge_count = 0
        # Group entities by source
        co_occurring: List[Dict] = []

        for ent in entities:
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            etype = ent.get("type", "entity")
            self.L1_literature.add_node(
                name,
                node_type=etype,
                layer=1,
                sources=list(set(
                    self.L1_literature.nodes.get(name, {}).get("sources", []) + [source_id]
                )) if source_id else [],
            )
            co_occurring.append(ent)

            # Cross-link to ontology (fuzzy match)
            self._auto_cross_link(name, etype)

        # Add co-occurrence edges within same paper
        for i, e1 in enumerate(co_occurring):
            for e2 in co_occurring[i + 1:]:
                n1, n2 = e1["name"].strip(), e2["name"].strip()
                if n1 != n2:
                    if self.L1_literature.has_edge(n1, n2):
                        w = self.L1_literature[n1][n2].get("weight", 1)
                        self.L1_literature[n1][n2]["weight"] = w + 1
                    else:
                        self.L1_literature.add_edge(
                            n1, n2,
                            rel_type="co_occurs",
                            weight=1,
                            source=source_id,
                        )
                        edge_count += 1

        return edge_count

    def _auto_cross_link(self, entity_name: str, entity_type: str):
        """Try to link L1 entity to L3 ontology term."""
        en_lower = entity_name.lower()
        for node in self.L3_ontology.nodes:
            if en_lower == node.lower() or en_lower in node.lower():
                self._add_cross_link(entity_name, node, "maps_to")
                return

    def _add_cross_link(self, source: str, target: str, relation: str):
        link = (source, target, relation)
        if link not in self.cross_links:
            self.cross_links.append(link)

    # ─── Queries ──────────────────────────────────────────────────────────

    def get_entity_neighborhood(
        self, entity: str, radius: int = 2, layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get N-hop neighborhood of an entity across layers.

        Returns dict with nodes, edges, cross_links.
        """
        result = {"nodes": [], "edges": [], "cross_links": []}
        graphs = {1: self.L1_literature, 2: self.L2_authority, 3: self.L3_ontology}

        if layer:
            graphs = {layer: graphs[layer]}

        for lyr, G in graphs.items():
            if entity not in G.nodes:
                continue
            try:
                subgraph_nodes = nx.single_source_shortest_path_length(G, entity, cutoff=radius)
            except nx.NetworkXError:
                subgraph_nodes = {entity: 0}

            for node, dist in subgraph_nodes.items():
                result["nodes"].append({
                    "name": node,
                    "layer": lyr,
                    "distance": dist,
                    **{k: v for k, v in G.nodes[node].items() if k not in ("layer",)},
                })
            for u, v, data in G.edges(data=True):
                if u in subgraph_nodes and v in subgraph_nodes:
                    result["edges"].append({"source": u, "target": v, "layer": lyr, **data})

        # Cross-links involving any of the found nodes
        node_names = {n["name"] for n in result["nodes"]}
        for src, tgt, rel in self.cross_links:
            if src in node_names or tgt in node_names:
                result["cross_links"].append({"source": src, "target": tgt, "relation": rel})

        return result

    def find_path(self, source: str, target: str) -> List[str]:
        """Find shortest path across L2 authority graph."""
        G = self.L2_authority
        if source not in G or target not in G:
            return []
        try:
            return nx.shortest_path(G.undirected_view() if hasattr(G, 'undirected_view') else G.to_undirected(), source, target)
        except nx.NetworkXNoPath:
            return []

    def technology_disease_triples(self, technology: str) -> List[Dict]:
        """Get all (technology, relation, disease) triples."""
        results = []
        G = self.L2_authority
        if technology not in G:
            return results
        for _, target, data in G.out_edges(technology, data=True):
            if G.nodes.get(target, {}).get("node_type") == "disease":
                results.append({
                    "technology": technology,
                    "relation": data.get("rel_type", "related"),
                    "disease": target,
                    "evidence_level": data.get("evidence_level", ""),
                    "efficiency": data.get("efficiency", ""),
                })
        return results

    def format_context_for_llm(self, entity: str, radius: int = 2) -> str:
        """Format neighborhood as structured text for LLM injection."""
        neighborhood = self.get_entity_neighborhood(entity, radius)
        if not neighborhood["nodes"]:
            return ""

        lines = [f"### Triple Graph Context: {entity}"]

        by_layer = defaultdict(list)
        for n in neighborhood["nodes"]:
            by_layer[n.get("layer", 0)].append(n)

        layer_names = {1: "Literature Entities", 2: "Authoritative Knowledge", 3: "Ontology/Vocabulary"}
        for lyr in sorted(by_layer.keys()):
            lines.append(f"\n**Layer {lyr} ({layer_names.get(lyr, 'Unknown')}):**")
            for n in by_layer[lyr][:15]:
                ntype = n.get("node_type", "")
                lines.append(f"- {n['name']} ({ntype}, distance={n['distance']})")

        if neighborhood["edges"]:
            lines.append(f"\n**Relationships ({len(neighborhood['edges'])} total, showing top 10):**")
            for e in neighborhood["edges"][:10]:
                rel = e.get("rel_type", "related")
                lines.append(f"- {e['source']} --[{rel}]--> {e['target']}")

        if neighborhood["cross_links"]:
            lines.append(f"\n**Cross-Layer Links:**")
            for cl in neighborhood["cross_links"][:5]:
                lines.append(f"- {cl['source']} ={cl['relation']}=> {cl['target']}")

        return "\n".join(lines)

    # ─── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        return {
            "L1_literature_nodes": self.L1_literature.number_of_nodes(),
            "L1_literature_edges": self.L1_literature.number_of_edges(),
            "L2_authority_nodes": self.L2_authority.number_of_nodes(),
            "L2_authority_edges": self.L2_authority.number_of_edges(),
            "L3_ontology_nodes": self.L3_ontology.number_of_nodes(),
            "L3_ontology_edges": self.L3_ontology.number_of_edges(),
            "cross_links": len(self.cross_links),
        }

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        path = path or self.persist_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "L1": nx.node_link_data(self.L1_literature),
            "L2": nx.node_link_data(self.L2_authority),
            "cross_links": self.cross_links,
            # L3 is rebuilt from GENE_EDITING_ONTOLOGY each time
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[TripleGraph] Saved to {path}")

    def _load_persisted(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "L1" in data:
                self.L1_literature = nx.node_link_graph(data["L1"], directed=True)
            if "L2" in data:
                self.L2_authority = nx.node_link_graph(data["L2"], directed=True)
            self.cross_links = [tuple(cl) for cl in data.get("cross_links", [])]
            print(f"[TripleGraph] Loaded from {self.persist_path}: {self.stats()}")
        except Exception as e:
            print(f"[TripleGraph] Load error: {e}")
