"""
Build a rich Knowledge Graph from literature DB + GEAlmanac structured data.
=============================================================================
No NER model needed — uses keyword extraction, regex patterns, and the
structured almanac data to produce a dense KG with 6 node types and
multiple edge types.

Output: data/knowledge_base/kg.json (NetworkX node-link JSON)

Usage:
    python scripts/build_kg_from_almanac.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

KB_PATH = ROOT / "data" / "knowledge_base" / "literature_db_GEA_v2026_Q1.json"
KG_SAVE = ROOT / "data" / "knowledge_base" / "kg.json"

# ── Technology patterns ──────────────────────────────────────────────────
TECH_PATTERNS = [
    (r"\bCRISPR[\-/\s]*Cas9\b", "CRISPR-Cas9"),
    (r"\bSpCas9\b", "SpCas9"),
    (r"\bSaCas9\b", "SaCas9"),
    (r"\bCas12[a-f]?\b|\bCpf1\b", "Cas12a"),
    (r"\bCas13[a-d]?\b|\bCasRx\b", "Cas13"),
    (r"\b[Bb]ase\s*edit(?:ing|or)?\b|\bABE\b|\bCBE\b|\bBE[1-4]\b", "Base Editing"),
    (r"\bABE[0-9]?[a-z]?\b|\badenine\s*base\s*edit", "ABE"),
    (r"\bCBE[0-9]?\b|\bcytosine\s*base\s*edit|\bBE[34]\b", "CBE"),
    (r"\b[Pp]rime\s*edit(?:ing|or)?\b|\bPE[1-5]\b|\bpegRNA\b", "Prime Editing"),
    (r"\bCRISPR[\-\s]*?[Ii](?:nterference)?\b|\bCRISPRi\b|\bdCas9[\-\s]*KRAB\b", "CRISPRi"),
    (r"\bCRISPR[\-\s]*?[Aa](?:ctivation)?\b|\bCRISPRa\b", "CRISPRa"),
    (r"\bRNA\s*edit(?:ing)?\b|\bADAR[12]?\b|\bREPAIR\b|\bRESCUE\b", "RNA Editing"),
    (r"\bTALEN\b", "TALEN"),
    (r"\bZFN\b|\bzinc\s*finger\s*nuclease\b", "ZFN"),
    (r"\bCRISPR[\-\s]*off\b|\bepigenome\s*edit\b|\bepigenetic\s*edit\b", "Epigenome Editing"),
    (r"\bgene\s*drive\b", "Gene Drive"),
    (r"\btransposon\b|\bpiggy\s*Bac\b|\bSleeping\s*Beauty\b", "Transposon"),
    (r"\bretron\b", "Retron"),
]

# ── Disease patterns ─────────────────────────────────────────────────────
DISEASE_PATTERNS = [
    (r"\bsickle\s*cell\b|\bSCD\b|\bHbS\b", "Sickle Cell Disease"),
    (r"\bthalassemia\b|\bTDT\b", "Beta-Thalassemia"),
    (r"\bDuchenne\b|\bDMD\b", "Duchenne Muscular Dystrophy"),
    (r"\bcystic\s*fibrosis\b|\bCFTR\b", "Cystic Fibrosis"),
    (r"\bangioedema\b|\bHAE\b", "Hereditary Angioedema"),
    (r"\bTTR\s*amyloidosis\b|\bhATTR\b|\btransthyretin\b", "TTR Amyloidosis"),
    (r"\bhypercholesterolemia\b|\bHeFH\b|\bFH\b|\bPCSK9\b", "Hypercholesterolemia"),
    (r"\bretinal\s*dystrophy\b|\bLCA\b|\bLeber\b|\bretinitis\s*pigmentosa\b", "Retinal Dystrophy"),
    (r"\bhemophilia\b", "Hemophilia"),
    (r"\bHIV\b|\bAIDS\b", "HIV/AIDS"),
    (r"\bcancer\b|\btumor\b|\boncolog\b|\bcarcinoma\b|\bleukemia\b|\blymphoma\b|\bmelanoma\b", "Cancer"),
    (r"\bHuntington\b|\bHTT\b", "Huntington Disease"),
    (r"\balpha.?1.?antitrypsin\b|\bAAT\b|\bA1AT\b", "Alpha-1 Antitrypsin Deficiency"),
    (r"\bSMA\b|\bspinal\s*muscular\s*atrophy\b", "Spinal Muscular Atrophy"),
    (r"\btyrosinemia\b|\bHT1\b", "Tyrosinemia"),
    (r"\bWilson\b", "Wilson Disease"),
    (r"\bPKU\b|\bphenylketonuria\b", "Phenylketonuria"),
]

# ── Delivery patterns ────────────────────────────────────────────────────
DELIVERY_PATTERNS = [
    (r"\bAAV[0-9]?\b|\badeno.?associated\b", "AAV"),
    (r"\bLNP\b|\blipid\s*nanoparticle\b", "LNP"),
    (r"\belectroporation\b|\belectroporat\b", "Electroporation"),
    (r"\bRNP\b|\bribonucleoprotein\b", "RNP"),
    (r"\blentivir\b|\bLV\b", "Lentivirus"),
    (r"\bVLP\b|\bvirus.like\s*particle\b", "VLP"),
    (r"\bexosome\b", "Exosome"),
    (r"\bmRNA\b", "mRNA delivery"),
    (r"\bplasmid\b", "Plasmid"),
]

# ── Gene patterns ────────────────────────────────────────────────────────
GENE_PATTERNS = [
    (r"\bHBB\b", "HBB"), (r"\bBCL11A\b", "BCL11A"), (r"\bPCSK9\b", "PCSK9"),
    (r"\bTTR\b", "TTR"), (r"\bCFTR\b", "CFTR"), (r"\bDMD\b|\bdystrophin\b", "DMD"),
    (r"\bHTT\b|\bhuntingtin\b", "HTT"), (r"\bVEGF\b", "VEGF"),
    (r"\bTP53\b|\bp53\b", "TP53"), (r"\bKRAS\b", "KRAS"),
    (r"\bPD.?1\b|\bPDCD1\b", "PDCD1"), (r"\bPD.?L1\b|\bCD274\b", "CD274"),
    (r"\bCD19\b", "CD19"), (r"\bTRAC\b", "TRAC"), (r"\bB2M\b", "B2M"),
    (r"\bFAH\b", "FAH"), (r"\bSERPINA1\b", "SERPINA1"),
    (r"\bKLKB1\b|\bprekallikrein\b", "KLKB1"), (r"\bSMN1\b|\bSMN2\b", "SMN1"),
    (r"\bCEP290\b", "CEP290"), (r"\bANGPTL3\b", "ANGPTL3"),
    (r"\bCISH\b", "CISH"), (r"\bJAK[12]\b", "JAK1/2"),
    (r"\bFOXP3\b", "FOXP3"), (r"\bIL2RG\b", "IL2RG"),
]

# ── Cell/Tissue patterns ─────────────────────────────────────────────────
CELL_PATTERNS = [
    (r"\bHSCs?\b|\bhematopoietic\s*stem\b|\bCD34\+?\b", "HSC"),
    (r"\bT\s*cell\b|\bCAR[\-\s]*T\b", "T cell"),
    (r"\bNK\s*cell\b|\bCAR[\-\s]*NK\b", "NK cell"),
    (r"\biPSC\b|\binduced\s*pluripotent\b", "iPSC"),
    (r"\bhepato\b|\bliver\b|\bhepatic\b", "Liver"),
    (r"\bretina\b|\bRPE\b|\bphotoreceptor\b", "Retina"),
    (r"\blung\b|\bairway\b|\bpulmonary\b", "Lung"),
    (r"\bbrain\b|\bneuron\b|\bCNS\b|\bneuronal\b", "Brain/CNS"),
    (r"\bmuscle\b|\bskeletal\s*muscle\b|\bmyocyte\b", "Muscle"),
]

# ── Clinical product patterns ────────────────────────────────────────────
PRODUCT_PATTERNS = [
    (r"\bCasgevy\b|\bCTX001\b|\bexa.?cel\b", "Casgevy"),
    (r"\bNTLA[\-\s]*2001\b", "NTLA-2001"),
    (r"\bNTLA[\-\s]*3001\b", "NTLA-3001"),
    (r"\bVERVE[\-\s]*101\b", "VERVE-101"),
    (r"\bEDIT[\-\s]*101\b", "EDIT-101"),
    (r"\bEBT[\-\s]*101\b", "EBT-101"),
    (r"\bZolgensma\b|\bonasemnogene\b", "Zolgensma"),
    (r"\bLuxturna\b|\bvoretigene\b", "Luxturna"),
]


def extract_entities(text: str, patterns: list) -> set:
    """Extract matching entity names from text using regex patterns."""
    found = set()
    if not text:
        return found
    for regex, name in patterns:
        if re.search(regex, text, re.IGNORECASE):
            found.add(name)
    return found


def build_kg():
    print("Loading literature DB...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  {len(articles)} articles loaded.")

    # ── Node accumulators ─────────────────────────────────────────────
    nodes = {}       # id → {type, ...attributes}
    edges = []       # list of {source, target, relation, weight, ...}
    cooccurrence = defaultdict(Counter)  # (type1, type2) → Counter of (e1,e2) pairs

    def add_node(nid: str, ntype: str, **kwargs):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": ntype, **kwargs}

    # ── Seed from GEAlmanac ──────────────────────────────────────────
    try:
        from core.ge_almanac import GEAlmanac
        alm = GEAlmanac()

        # 1) Seed clinical trials → ClinicalProduct + Technology + Disease links
        for trial in alm.clinical_trials:
            pid = trial.get("name", trial.get("nct_id", "unknown"))
            add_node(pid, "ClinicalProduct",
                     nct_id=trial.get("nct_id"),
                     sponsor=trial.get("sponsor"),
                     indication=trial.get("disease"),
                     phase=trial.get("phase"),
                     modality=trial.get("technology"))
            # Link product → technology
            tech = trial.get("technology")
            if tech:
                add_node(tech, "Technology")
                edges.append({"source": pid, "target": tech,
                              "relation": "uses_technology", "weight": 1.0})
            # Link product → disease
            disease = trial.get("disease")
            if disease:
                add_node(disease, "Disease")
                edges.append({"source": pid, "target": disease,
                              "relation": "targets_disease", "weight": 1.0})
            # Link product → target gene
            gene = trial.get("target_gene")
            if gene:
                add_node(gene, "Gene")
                edges.append({"source": pid, "target": gene,
                              "relation": "targets_gene", "weight": 1.0})

        # 2) Seed regulatory approvals as milestone nodes
        for appr in alm.approvals:
            event_id = f"APPR_{appr.get('date', '?')}_{appr.get('drug', '?')}"
            add_node(event_id, "Milestone",
                     year=appr.get("date", "")[:4],
                     event=f"{appr.get('drug')} approved by {appr.get('agency')}",
                     category="regulatory_approval")
            drug = appr.get("drug", "")
            if drug:
                add_node(drug, "ClinicalProduct")
                edges.append({"source": event_id, "target": drug,
                              "relation": "approves_product", "weight": 1.0})

        # 3) Seed technology registry → Technology nodes (Cas variants etc.)
        for tech_entry in alm.tech_registry:
            tname = tech_entry["name"]
            add_node(tname, "Technology",
                     category=tech_entry.get("category"),
                     mechanism=tech_entry.get("mechanism"),
                     pam=tech_entry.get("pam"),
                     size_aa=tech_entry.get("size_aa"))

        # 4) Seed tech-disease matrix → edges
        for td in alm.tech_disease:
            tech = td.get("technology", "")
            disease = td.get("disease", "")
            gene = td.get("target_gene", "")
            if tech:
                add_node(tech, "Technology")
            if disease:
                add_node(disease, "Disease")
            if gene:
                add_node(gene, "Gene")
            if tech and disease:
                edges.append({"source": tech, "target": disease,
                              "relation": "applied_to", "weight": 1.0,
                              "evidence_level": td.get("evidence_level")})
            if tech and gene:
                edges.append({"source": tech, "target": gene,
                              "relation": "targets_gene", "weight": 1.0})
            if disease and gene:
                edges.append({"source": disease, "target": gene,
                              "relation": "associated_gene", "weight": 1.0})

        # 5) Seed safety profiles → attach to Technology nodes
        for sp in alm.safety:
            tech = sp.get("technology", "")
            if tech:
                add_node(tech, "Technology")
                # Store safety metadata inside node
                nodes[tech]["off_target"] = sp.get("off_target_genomic")
                nodes[tech]["clinical_safety"] = sp.get("clinical_safety")

        print(f"  [GEAlmanac] Seeded {len(nodes)} nodes, {len(edges)} edges")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [GEAlmanac] Skipped: {e}")

    # ── Scan all articles for co-occurrences ────────────────────────
    print("Scanning articles for entity co-occurrences...")
    all_pattern_groups = [
        ("Technology", TECH_PATTERNS),
        ("Disease", DISEASE_PATTERNS),
        ("Delivery", DELIVERY_PATTERNS),
        ("Gene", GENE_PATTERNS),
        ("CellType", CELL_PATTERNS),
        ("ClinicalProduct", PRODUCT_PATTERNS),
    ]

    article_count = 0
    for art in articles:
        text = (art.get("title", "") or "") + " " + (art.get("abstract", "") or "")
        if len(text) < 20:
            continue
        article_count += 1

        # Extract all entities from this article
        found_by_type = {}
        for gtype, patterns in all_pattern_groups:
            ents = extract_entities(text, patterns)
            for e in ents:
                add_node(e, gtype)
            if ents:
                found_by_type[gtype] = ents

        # Build co-occurrence edges between different entity types
        type_pairs = [
            ("Technology", "Disease", "applied_to"),
            ("Technology", "Gene", "targets_gene"),
            ("Technology", "Delivery", "delivered_by"),
            ("Technology", "CellType", "applied_in"),
            ("Disease", "Gene", "associated_gene"),
            ("Disease", "Delivery", "treated_via"),
            ("Disease", "ClinicalProduct", "has_product"),
            ("ClinicalProduct", "Gene", "targets_gene"),
            ("ClinicalProduct", "Delivery", "uses_delivery"),
            ("Gene", "CellType", "expressed_in"),
        ]
        for t1, t2, rel in type_pairs:
            if t1 in found_by_type and t2 in found_by_type:
                for e1 in found_by_type[t1]:
                    for e2 in found_by_type[t2]:
                        cooccurrence[(t1, t2, rel)][(e1, e2)] += 1

    print(f"  Scanned {article_count:,} articles with abstracts.")

    # ── Convert co-occurrences to weighted edges ──────────────────────
    MIN_COOCCUR = 2  # Minimum co-occurrence count to create an edge
    edge_set = set()  # dedup
    for (t1, t2, rel), pair_counts in cooccurrence.items():
        for (e1, e2), count in pair_counts.items():
            if count >= MIN_COOCCUR:
                key = (e1, e2, rel)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({
                        "source": e1, "target": e2,
                        "relation": rel, "weight": count,
                    })

    # ── Add some semantic edges for core technologies ─────────────────
    tech_hierarchy = {
        "CRISPR-Cas9": ["SpCas9", "SaCas9"],
        "Cas12a": [],
        "Base Editing": ["ABE", "CBE"],
        "Prime Editing": [],
        "RNA Editing": [],
        "CRISPRi": [],
        "CRISPRa": [],
    }
    for parent, children in tech_hierarchy.items():
        add_node(parent, "Technology")
        for child in children:
            add_node(child, "Technology")
            edges.append({"source": child, "target": parent,
                          "relation": "subtype_of", "weight": 1.0})

    # Disease → Gene canonical associations
    canonical_assoc = [
        ("Sickle Cell Disease", "HBB"), ("Sickle Cell Disease", "BCL11A"),
        ("Beta-Thalassemia", "HBB"), ("Beta-Thalassemia", "BCL11A"),
        ("Cystic Fibrosis", "CFTR"), ("Duchenne Muscular Dystrophy", "DMD"),
        ("Huntington Disease", "HTT"), ("TTR Amyloidosis", "TTR"),
        ("Hypercholesterolemia", "PCSK9"), ("Hereditary Angioedema", "KLKB1"),
        ("Retinal Dystrophy", "CEP290"), ("Alpha-1 Antitrypsin Deficiency", "SERPINA1"),
        ("Spinal Muscular Atrophy", "SMN1"),
    ]
    for disease, gene in canonical_assoc:
        add_node(disease, "Disease")
        add_node(gene, "Gene")
        key = (disease, gene, "causal_gene")
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"source": disease, "target": gene,
                          "relation": "causal_gene", "weight": 10.0})

    # ── Build final graph JSON ───────────────────────────────────────
    # Dedup edges
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "name": "GeneEditingKG",
            "version": "v2026_Q1",
            "articles_scanned": article_count,
            "min_cooccurrence": MIN_COOCCUR,
        },
        "nodes": list(nodes.values()),
        "links": unique_edges,
    }

    # ── Stats ────────────────────────────────────────────────────────
    type_counts = Counter(n["type"] for n in graph["nodes"])
    rel_counts = Counter(e["relation"] for e in graph["links"])

    print(f"\n{'='*60}")
    print(f"Knowledge Graph Built")
    print(f"{'='*60}")
    print(f"Nodes: {len(graph['nodes'])}")
    for t, c in type_counts.most_common():
        print(f"  {t:25s} {c}")
    print(f"Edges: {len(graph['links'])}")
    for r, c in rel_counts.most_common():
        print(f"  {r:25s} {c}")
    print(f"{'='*60}")

    # ── Save ─────────────────────────────────────────────────────────
    with open(KG_SAVE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {KG_SAVE} ({KG_SAVE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_kg()
