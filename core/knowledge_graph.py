import networkx as nx
import json
import os
import re
from typing import Dict, List, Optional, Any

class GeneEditingKnowledgeGraph:
    def __init__(self, persistence_path="data/knowledge_base/kg.json"):
        self.graph = nx.DiGraph()
        self.persistence_path = persistence_path
        self._initialize_base_ontology()
        self.load_graph()

    def save_graph(self):
        """Persist the graph to disk."""
        data = nx.node_link_data(self.graph)
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Knowledge Graph saved to {self.persistence_path}")

    def load_graph(self):
        """Load the graph from disk if it exists."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                print(f"Loaded Knowledge Graph with {len(self.graph.nodes)} nodes from {self.persistence_path}")
            except Exception as e:
                print(f"Error loading Knowledge Graph: {e}. Starting with base ontology.")
        else:
            print("No existing Knowledge Graph found. Starting fresh.")

    def _initialize_base_ontology(self):
        """
        Comprehensive base ontology for the Gene Editing Knowledge Base.
        Covers: editing tools, Cas variants, mutations, delivery, diseases,
                genes, clinical evidence, institutions, safety profiles.
        """
        # 鈹€鈹€ Editing Technologies 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        tools = {
            "CRISPR KO": {"category": "DNA Double-strand break", "mechanism": "NHEJ/HDR"},
            "Base Editing CBE": {"category": "Precision Editing", "mechanism": "Cytosine deamination"},
            "Base Editing ABE": {"category": "Precision Editing", "mechanism": "Adenosine deamination"},
            "Prime Editing": {"category": "Precision Editing", "mechanism": "Reverse transcription"},
            "RNA Editing ADAR": {"category": "RNA Editing", "mechanism": "Adenosine deamination (RNA)"},
            "Epigenome Editing": {"category": "Epigenetic Editing", "mechanism": "dCas9-effector"},
            "CRISPR Activation": {"category": "CRISPRa", "mechanism": "dCas9-VPR/VP64"},
            "CRISPR Interference": {"category": "CRISPRi", "mechanism": "dCas9-KRAB"},
            "Base Editing": {"category": "Precision Editing", "mechanism": "Deaminase"},
        }
        for name, attrs in tools.items():
            self.add_node(name, "Editing_Tool", **attrs)

        # 鈹€鈹€ Cas Variants (expanded) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        cas_variants = {
            "SpCas9": {"pam": "NGG", "size_aa": 1368, "origin": "S. pyogenes"},
            "SaCas9": {"pam": "NNGRRT", "size_aa": 1053, "origin": "S. aureus"},
            "Cas12a": {"pam": "TTTN", "size_aa": 1228, "origin": "A. acidithermus"},
            "Cas12b": {"pam": "ATTN", "origin": "B. hisashii"},
            "Cas13a": {"target": "RNA", "origin": "L. shahii"},
            "Cas13b": {"target": "RNA", "origin": "P. buccae"},
            "Cas13d": {"target": "RNA", "size_aa": 967, "origin": "RfxCas13d (CasRx)"},
            "Cas9-HF1": {"pam": "NGG", "note": "High-fidelity variant"},
            "eSpCas9": {"pam": "NGG", "note": "Enhanced specificity"},
            "SniperCas9": {"pam": "NGG", "note": "High fidelity, Korea"},
            "xCas9": {"pam": "NG", "note": "Expanded PAM"},
            "SpRY": {"pam": "NRN", "note": "Near-PAMless"},
            "CasPhi": {"size_aa": 700, "note": "Compact, bacteriophage"},
            "SauriCas9": {"pam": "NNGG"},
        }
        for name, attrs in cas_variants.items():
            self.add_node(name, "Cas_Type", **attrs)

        # 鈹€鈹€ Mutation Types 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        mutations = {
            "SNV": {"description": "Single nucleotide variant"},
            "point mutation": {"description": "Alias for SNV"},
            "insertion": {"description": "Sequence insertion"},
            "deletion": {"description": "Sequence deletion"},
            "indel": {"description": "Insertion or deletion"},
            "frameshift": {"description": "Frame-disrupting indel"},
            "missense": {"description": "Amino acid change"},
            "nonsense": {"description": "Premature stop codon"},
            "splice site": {"description": "Splice junction alteration"},
            "duplication": {"description": "Segment duplication"},
            "inversion": {"description": "Segment inversion"},
            "translocation": {"description": "Chromosomal translocation"},
        }
        for name, attrs in mutations.items():
            self.add_node(name, "Mutation_Type", **attrs)

        # 鈹€鈹€ Delivery Systems 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        delivery_systems = {
            "AAV": {"tropism": "Multiple serotypes", "payload_kb": 4.7, "immunogenicity": "Moderate"},
            "LNP": {"route": "IV/local", "payload_unlimited": True, "immunogenicity": "Low"},
            "LNP-GalNAc": {"tropism": "Hepatocyte-targeted", "immunogenicity": "Low"},
            "electroporation": {"route": "Ex vivo", "payload_unlimited": True},
            "mRNA": {"transient": True, "immunogenicity": "Low"},
            "RNP": {"transient": True, "immunogenicity": "Minimal"},
            "viral vector lentivirus": {"integrating": True, "immunogenicity": "Moderate"},
            "adenovirus": {"transient": True, "immunogenicity": "High"},
            "SEND": {"note": "Selective Endogenous eNdosome Delivery (MIT)"},
            "lipofection": {"route": "In vitro"},
            "nanoparticle": {},
        }
        for name, attrs in delivery_systems.items():
            self.add_node(name, "Delivery_System", **attrs)

        # 鈹€鈹€ Disease Areas 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        diseases = {
            "Sickle Cell Disease": {"gene": "HBB", "mutation": "E6V", "mondo": "MONDO:0011382"},
            "Beta-Thalassemia": {"gene": "HBB", "mondo": "MONDO:0013517"},
            "ATTR Amyloidosis": {"gene": "TTR", "mondo": "MONDO:0016462"},
            "Hypercholesterolaemia": {"gene": "PCSK9/LDLR", "mondo": "MONDO:0011711"},
            "Leber Congenital Amaurosis": {"gene": "CEP290/RPE65", "mondo": "MONDO:0016061"},
            "Duchenne Muscular Dystrophy": {"gene": "DMD", "mondo": "MONDO:0010679"},
            "Cystic Fibrosis": {"gene": "CFTR", "mondo": "MONDO:0009061"},
            "Acute Hepatic Porphyria": {"gene": "HMBS/HAO1", "mondo": "MONDO:0015252"},
            "NSCLC": {"gene": "EGFR/KRAS/ALK", "mondo": "MONDO:0005233"},
            "AML": {"gene": "FLT3/NPM1/IDH2", "mondo": "MONDO:0018874"},
            "HIV": {"gene": "CCR5", "mondo": "MONDO:0005109"},
            "Haemophilia A": {"gene": "F8", "mondo": "MONDO:0010602"},
            "Haemophilia B": {"gene": "F9", "mondo": "MONDO:0010595"},
            "Primary Hyperoxaluria Type 1": {"gene": "AGXT"},
        }
        for name, attrs in diseases.items():
            self.add_node(name, "Disease", **attrs)

        # 鈹€鈹€ Key Therapeutic Genes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        genes = {
            "HBB": {"function": "Haemoglobin beta chain", "locus": "11p15.4"},
            "HBG1": {"function": "Foetal haemoglobin gamma", "locus": "11p15.4"},
            "BCL11A": {"function": "HbF repressor", "locus": "2p16.1"},
            "PCSK9": {"function": "LDL receptor degradation", "locus": "1p32.3"},
            "TTR": {"function": "Transthyretin", "locus": "18q12.1"},
            "DMD": {"function": "Dystrophin", "locus": "Xp21.2"},
            "CFTR": {"function": "Chloride channel", "locus": "7q31.2"},
            "CEP290": {"function": "Centrosomal protein", "locus": "12q21.33"},
            "VEGFA": {"function": "Angiogenesis", "locus": "6p21.1"},
            "EGFR": {"function": "EGF receptor kinase", "locus": "7p11.2"},
            "TP53": {"function": "Tumour suppressor", "locus": "17p13.1"},
            "BRCA1": {"function": "DNA repair", "locus": "17q21.31"},
            "BRCA2": {"function": "DNA repair", "locus": "13q12.3"},
            "CCR5": {"function": "HIV co-receptor", "locus": "3p21.31"},
            "HAO1": {"function": "Glycolate oxidase", "locus": "20p12.1"},
            "F8": {"function": "Clotting factor VIII", "locus": "Xq28"},
        }
        for name, attrs in genes.items():
            self.add_node(name, "Gene", **attrs)

        # 鈹€鈹€ Evidence Levels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        evidence_levels = {
            "Level 1": {"description": "Clinical RCT or Phase II+", "weight": 1.0},
            "Level 2": {"description": "Animal in vivo study", "weight": 0.75},
            "Level 3": {"description": "In vitro (cell line) study", "weight": 0.50},
            "Level 4": {"description": "Computational / predictive", "weight": 0.25},
        }
        for name, attrs in evidence_levels.items():
            self.add_node(name, "Evidence_Level", **attrs)

        # 鈹€鈹€ Cell Types 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        cell_types = [
            "HEK293", "HEK293T", "HSC", "T-cell", "NK-cell", "iPSC",
            "primary hepatocyte", "CD34+", "retinal cell", "neurons",
            "cardiomyocyte", "fibroblast", "K562",
        ]
        for ct in cell_types:
            self.add_node(ct, "Cell_Type")

        # 鈹€鈹€ Technology 鈫?Disease canonical relationships 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.add_edge("Base Editing ABE", "Sickle Cell Disease", "addresses_disease",
                      note="ABE corrects HBB E6V (GAG鈫扜TG A>G edit)")
        self.add_edge("Base Editing CBE", "Beta-Thalassemia", "addresses_disease",
                      note="CBE recreates gamma-globin promoter mutations")
        self.add_edge("Prime Editing", "Sickle Cell Disease", "addresses_disease")
        self.add_edge("CRISPR KO", "Beta-Thalassemia", "addresses_disease",
                      note="BCL11A enhancer disruption to reactivate HbF")
        self.add_edge("RNA Editing ADAR", "ATTR Amyloidosis", "investigates_disease")
        self.add_edge("CRISPR KO", "Hypercholesterolaemia", "addresses_disease",
                      note="PCSK9 knockout")

        # 鈹€鈹€ Gene 鈫?Disease 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        gene_disease = [
            ("HBB", "Sickle Cell Disease"), ("HBB", "Beta-Thalassemia"),
            ("TTR", "ATTR Amyloidosis"), ("PCSK9", "Hypercholesterolaemia"),
            ("DMD", "Duchenne Muscular Dystrophy"), ("CFTR", "Cystic Fibrosis"),
            ("CEP290", "Leber Congenital Amaurosis"), ("F8", "Haemophilia A"),
            ("BCL11A", "Beta-Thalassemia"), ("BCL11A", "Sickle Cell Disease"),
        ]
        for gene, disease in gene_disease:
            self.add_edge(gene, disease, "associated_with_disease")

        # 鈹€鈹€ Clinical Milestones 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        milestones = {
            "CRISPR_SCD_2023_FDA": {
                "type": "Clinical_Milestone",
                "event": "FDA approval of Casgevy (CTX001) for SCD",
                "year": 2023,
                "technology": "CRISPR KO",
                "target_gene": "BCL11A",
            },
            "ABE_VERVE101_2023": {
                "type": "Clinical_Milestone",
                "event": "VERVE-101 Phase Ib: LDL reduction 55%",
                "year": 2023,
                "technology": "Base Editing ABE",
                "target_gene": "PCSK9",
            },
        }
        for mid, attrs in milestones.items():
            node_type = attrs.pop("type", "Clinical_Milestone")
            self.add_node(mid, node_type, **attrs)
            if "technology" in attrs:
                self.add_edge(attrs["technology"], mid, "has_clinical_milestone")

    def add_node(self, name, node_type, **kwargs):
        if not name or name == "Unknown":
            return
        self.graph.add_node(name, type=node_type, **kwargs)

    def add_edge(self, source, target, relation, **kwargs):
        if not source or source == "Unknown" or not target or target == "Unknown":
            return
        self.graph.add_edge(source, target, relation=relation, **kwargs)

    def ingest_structured_data(self, data: Dict[str, Any], study_id: str):
        """Ingest structured LLM-extracted data into the KG."""
        self.add_node(study_id, "Study")

        tech = data.get("technology")
        cas = data.get("cas_type")
        mut = data.get("mutation_type")
        cell = data.get("cell_type")
        species = data.get("species")
        delivery = data.get("delivery_system")
        evidence = data.get("evidence_level")
        efficiency = data.get("efficiency", "Unknown")
        off_target = data.get("off_target_risk", "Unknown")
        gene = data.get("target_gene")
        disease = data.get("disease")

        for val, ntype in [(tech, "Editing_Tool"), (cas, "Cas_Type"), (mut, "Mutation_Type"),
                           (cell, "Cell_Type"), (species, "Species"),
                           (delivery, "Delivery_System"), (evidence, "Evidence_Level"),
                           (gene, "Gene"), (disease, "Disease")]:
            if val and val != "Unknown":
                self.add_node(val, ntype)

        rels = [
            (tech, cas, "uses_cas"), (tech, mut, "can_fix"),
            (study_id, tech, "evaluates_technology"),
            (study_id, delivery, "uses_delivery"),
            (study_id, cell, "conducted_in_cell"),
            (study_id, species, "conducted_in_species"),
            (study_id, evidence, "has_evidence_level"),
        ]
        for src, tgt, rel in rels:
            self.add_edge(src, tgt, rel)

        if gene:
            self.add_edge(tech, gene, "targets_gene")
        if disease:
            self.add_edge(tech, disease, "addresses_disease")

        if tech and tech != "Unknown":
            self.add_edge(tech, study_id, "demonstrates_efficiency",
                          efficiency=efficiency, off_target_risk=off_target,
                          cell_type=cell, mutation_type=mut)

    def ingest_from_ner(self, entities: List[Dict[str, Any]], study_id: str):
        """
        Ingest entities extracted by a NER model (scispaCy / PubTator).

        Parameters
        ----------
        entities : list of {"text": str, "label": str}
            LABEL meanings:
                GENE, DISEASE, CHEMICAL, ORGANISM, CELL_TYPE, MUTATION
        """
        self.add_node(study_id, "Study")
        label_map = {
            "GENE": "Gene", "DISEASE": "Disease", "CHEMICAL": "Drug",
            "ORGANISM": "Species", "CELL_TYPE": "Cell_Type", "MUTATION": "Mutation_Type",
        }
        for ent in entities:
            text = ent.get("text", "").strip()
            label = ent.get("label", "")
            ntype = label_map.get(label, label)
            if text and text != "Unknown":
                self.add_node(text, ntype)
                self.add_edge(study_id, text, f"mentions_{label.lower()}")

    def query_technology_capabilities(self, technology: str) -> Optional[Dict]:
        if technology not in self.graph:
            return None
        successors = list(self.graph.successors(technology))
        predecessors = list(self.graph.predecessors(technology))
        can_fix = [n for n in successors
                   if self.graph[technology][n].get("relation") == "can_fix"]
        studies = [n for n in predecessors
                   if self.graph.nodes[n].get("type") == "Study"]
        diseases = [n for n in successors
                    if self.graph.nodes[n].get("type") == "Disease"]
        genes = [n for n in successors
                 if self.graph.nodes[n].get("type") == "Gene"]
        milestones = [n for n in successors
                      if self.graph.nodes[n].get("type") == "Clinical_Milestone"]
        efficiencies = []
        for pred in predecessors:
            edge_data = self.graph.get_edge_data(technology, pred, default={})
            if edge_data.get("relation") == "demonstrates_efficiency":
                efficiencies.append({
                    "study": pred,
                    "efficiency": edge_data.get("efficiency", "Unknown"),
                    "off_target": edge_data.get("off_target_risk", "Unknown"),
                })
        return {
            "technology": technology,
            "can_fix": can_fix,
            "studies": studies,
            "diseases": diseases,
            "genes": genes,
            "clinical_milestones": milestones,
            "efficiency_records": efficiencies[:10],
        }

    def query_disease_landscape(self, disease: str) -> Optional[Dict]:
        """Return all technologies and genes associated with a disease."""
        if disease not in self.graph:
            return None
        preds = list(self.graph.predecessors(disease))
        techs = [n for n in preds
                 if self.graph.nodes[n].get("type") == "Editing_Tool"]
        genes = [n for n in preds
                 if self.graph.nodes[n].get("type") == "Gene"]
        return {"disease": disease, "associated_technologies": techs, "associated_genes": genes}

    def find_path_explanation(self, source: str, target: str, max_depth: int = 4) -> List[List[str]]:
        """Explain how two entities are related via graph paths."""
        if source not in self.graph or target not in self.graph:
            return []
        try:
            return list(nx.all_simple_paths(self.graph, source=source, target=target, cutoff=max_depth))
        except Exception:
            return []

    def get_node_summary(self, node_name: str) -> str:
        """Generate a human-readable summary of a node and its connections."""
        if node_name not in self.graph:
            return f"'{node_name}' not found in Knowledge Graph."
        attrs = dict(self.graph.nodes[node_name])
        ntype = attrs.pop("type", "Unknown")
        neighbours_out = [(n, self.graph[node_name][n].get("relation", "-->"))
                          for n in list(self.graph.successors(node_name))[:5]]
        neighbours_in  = [(n, self.graph[n][node_name].get("relation", "<--"))
                          for n in list(self.graph.predecessors(node_name))[:5]]
        lines = [f"**{node_name}** (Type: {ntype})"]
        if attrs:
            lines.append(f"Attributes: {attrs}")
        if neighbours_out:
            lines.append(f"Outgoing: {neighbours_out}")
        if neighbours_in:
            lines.append(f"Incoming: {neighbours_in}")
        return "\n".join(lines)

    def get_all_nodes_by_type(self, node_type: str) -> List[str]:
        """Retrieve all nodes of a specific type."""
        return [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == node_type]

    def export_to_json(self, filepath: str):
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    kg = GeneEditingKnowledgeGraph()
    print(f"KG initialised: {len(kg.graph.nodes)} nodes, {len(kg.graph.edges)} edges.")
    caps = kg.query_technology_capabilities("Base Editing ABE")
    print(json.dumps(caps, indent=2, default=str))
