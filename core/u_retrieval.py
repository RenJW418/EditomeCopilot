"""
U-Retrieval — Hierarchical Tag-Based Retrieval
===============================================
Inspired by: MedGraphRAG (Wu et al. 2024)
  "U-Retrieval" = Top-Down precise retrieval + Bottom-Up enrichment.
  Uses tag-based hierarchical clustering to organise documents.

Adapted for Gene Editing:
  Tag Taxonomy:
    Level 0 (Domain): Gene Editing, Delivery, Clinical, Computational
    Level 1 (Technology): CRISPR-Cas9, Base Editing, Prime Editing, ...
    Level 2 (Application): Knockout, Correction, Activation, ...
    Level 3 (Target): BCL11A, PCSK9, TTR, DMD, ...

Flow:
  Top-Down: Query → tag assignment → filter by matching tags → rank
  Bottom-Up: Retrieved docs → expand via related tags → re-rank for diversity
"""

from __future__ import annotations

import re
import os
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Tag Taxonomy
# ─────────────────────────────────────────────────────────────────────────────
TAG_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # Level 0 → Level 1 → Level 2 → Level 3 (keywords for matching)
    "Gene Editing": {
        "CRISPR-Cas9": {
            "Knockout": ["NHEJ", "indel", "knockout", "KO", "disruption", "loss-of-function"],
            "HDR Correction": ["HDR", "homology-directed", "template", "correction", "knock-in"],
            "Exon Skipping": ["exon skip", "splice", "reading frame", "dystrophin"],
            "Multiplexed": ["multiplex", "dual-guide", "array", "library", "screen"],
        },
        "Base Editing": {
            "CBE (C→T)": ["cytosine", "CBE", "BE4", "APOBEC", "C-to-T", "C→T"],
            "ABE (A→G)": ["adenine", "ABE", "ABE8e", "TadA", "A-to-G", "A→G"],
            "Dual Base": ["dual deaminase", "ACBE", "synchronous"],
        },
        "Prime Editing": {
            "Point Mutation": ["pegRNA", "prime edit", "PE2", "PE3", "PEmax", "PE5"],
            "Small Insertion": ["insertion", "prime edit insert"],
            "Small Deletion": ["deletion", "prime edit del"],
            "Twin PE": ["twinPE", "twin prime", "large deletion", "inversion"],
        },
        "RNA Editing": {
            "ADAR-based": ["ADAR", "A-to-I", "inosine", "REPAIR", "RESCUE"],
            "Cas13 Knockdown": ["Cas13", "CasRx", "RNA knockdown", "HEPN"],
        },
        "Epigenome": {
            "CRISPRi": ["CRISPRi", "dCas9-KRAB", "repression", "silencing"],
            "CRISPRa": ["CRISPRa", "VP64", "VPR", "p65", "activation"],
        },
    },
    "Delivery": {
        "Viral": {
            "AAV": ["AAV", "adeno-associated", "AAV2", "AAV5", "AAV8", "AAV9"],
            "Lentivirus": ["lentivirus", "lentiviral", "LV"],
        },
        "Non-Viral": {
            "LNP": ["lipid nanoparticle", "LNP", "ionizable lipid", "MC3"],
            "RNP": ["ribonucleoprotein", "RNP", "Cas9 protein"],
            "Electroporation": ["electroporation", "nucleofection"],
            "VLP": ["virus-like particle", "VLP", "enVLP"],
        },
    },
    "Clinical": {
        "Clinical Trials": {
            "Phase I": ["phase I", "phase 1", "first-in-human"],
            "Phase II": ["phase II", "phase 2"],
            "Phase III": ["phase III", "phase 3", "pivotal"],
            "Approved": ["FDA approved", "EMA approved", "market approval"],
        },
        "Safety": {
            "Off-Target": ["off-target", "GUIDE-seq", "CIRCLE-seq", "Digenome"],
            "Immunogenicity": ["immune response", "anti-Cas9", "immunogenicity"],
            "Genotoxicity": ["chromothripsis", "translocation", "large deletion"],
        },
    },
    "Disease": {
        "Hematological": {
            "Sickle Cell": ["sickle cell", "SCD", "HbS", "HBB", "hemoglobin S"],
            "Thalassemia": ["thalassemia", "HbF", "BCL11A", "fetal hemoglobin"],
            "Hemophilia": ["hemophilia", "Factor VIII", "Factor IX", "F8", "F9"],
        },
        "Metabolic": {
            "Hypercholesterolemia": ["PCSK9", "LDL", "familial hypercholesterolemia", "FH"],
            "Amyloidosis": ["TTR", "transthyretin", "ATTR", "amyloidosis"],
        },
        "Neurological": {
            "DMD": ["Duchenne", "DMD", "dystrophin", "muscular dystrophy"],
            "Huntington": ["huntingtin", "HTT", "CAG repeat", "Huntington"],
        },
        "Ocular": {
            "LCA": ["LCA10", "CEP290", "Leber congenital amaurosis"],
        },
        "Oncology": {
            "CAR-T": ["CAR-T", "chimeric antigen receptor", "CD19", "BCMA"],
            "Immune Checkpoint": ["PD-1", "TRAC", "B2M", "allogeneic"],
        },
    },
}


class URetrieval:
    """Hierarchical tag-based retrieval with top-down and bottom-up passes."""

    def __init__(self, llm_client=None, tag_index_path: Optional[str] = None):
        self.llm = llm_client
        self.taxonomy = TAG_TAXONOMY
        self._flat_tags: Dict[str, List[str]] = {}
        self._flatten_taxonomy()

        self.tag_index_path = tag_index_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "knowledge_base", "tag_index.json"
        )
        # doc_id → set of tags
        self.doc_tags: Dict[str, Set[str]] = defaultdict(set)
        # tag → set of doc_ids
        self.tag_docs: Dict[str, Set[str]] = defaultdict(set)
        self._load_index()

    def _flatten_taxonomy(self):
        """Build flat tag → keywords mapping."""
        def _recurse(d, path=""):
            for key, val in d.items():
                full_path = f"{path}/{key}" if path else key
                if isinstance(val, dict):
                    _recurse(val, full_path)
                elif isinstance(val, list):
                    # val is the keyword list for this leaf tag
                    self._flat_tags[full_path] = [kw.lower() for kw in val]
        _recurse(self.taxonomy)

    # ─── Document Tagging ─────────────────────────────────────────────────

    def tag_document(self, doc_id: str, text: str) -> Set[str]:
        """
        Assign tags to a document based on keyword matching.

        Parameters
        ----------
        doc_id : str
            Unique identifier (PMID, DOI, etc.)
        text : str
            Abstract / full text of the document.

        Returns
        -------
        set of tag paths
        """
        text_lower = text.lower()
        matched_tags = set()

        for tag_path, keywords in self._flat_tags.items():
            for kw in keywords:
                if kw in text_lower:
                    matched_tags.add(tag_path)
                    break  # One match suffices per tag

        # Store
        self.doc_tags[doc_id] = matched_tags
        for tag in matched_tags:
            self.tag_docs[tag].add(doc_id)

        return matched_tags

    def tag_documents_batch(
        self, documents: List[Dict[str, Any]], id_field: str = "pmid", text_field: str = "abstract"
    ) -> int:
        """Tag a batch of documents. Returns count of tagged docs."""
        count = 0
        for doc in documents:
            doc_id = str(doc.get(id_field, doc.get("doi", "")))
            text = doc.get(text_field, "") or ""
            title = doc.get("title", "") or ""
            if doc_id and (text or title):
                self.tag_document(doc_id, f"{title} {text}")
                count += 1
        return count

    # ─── Query Tag Assignment ─────────────────────────────────────────────

    def assign_query_tags(self, query: str) -> List[str]:
        """
        Assign tags to a query. Returns sorted list of matching tag paths.
        """
        query_lower = query.lower()
        matched = []
        for tag_path, keywords in self._flat_tags.items():
            for kw in keywords:
                if kw in query_lower:
                    matched.append(tag_path)
                    break
        return sorted(matched)

    def assign_query_tags_llm(self, query: str) -> List[str]:
        """
        Use LLM for more sophisticated query tag assignment.
        Falls back to keyword-based if LLM unavailable.
        """
        if not self.llm or not hasattr(self.llm, "generate"):
            return self.assign_query_tags(query)

        tag_list = "\n".join(f"- {t}" for t in sorted(self._flat_tags.keys()))
        prompt = (
            "给以下基因编辑领域的用户查询分配标签。\n"
            f"查询: {query}\n\n"
            f"可用标签列表:\n{tag_list}\n\n"
            "请输出一个JSON数组, 包含最相关的标签路径 (1-5个):\n"
            '例: ["Gene Editing/CRISPR-Cas9/Knockout", "Disease/Hematological/Sickle Cell"]'
        )
        result = self.llm.generate(prompt, timeout=20, max_tokens=512)

        try:
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                tags = json.loads(json_match.group())
                # Validate against taxonomy
                valid_tags = [t for t in tags if t in self._flat_tags]
                if valid_tags:
                    return valid_tags
        except (json.JSONDecodeError, AttributeError):
            pass

        return self.assign_query_tags(query)

    # ─── Top-Down Retrieval ───────────────────────────────────────────────

    def top_down_retrieve(
        self,
        query_tags: List[str],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Top-down: Find documents matching the most query tags.

        Returns
        -------
        list of (doc_id, tag_overlap_score) sorted descending.
        """
        if not query_tags:
            return []

        doc_scores: Dict[str, float] = Counter()
        n_tags = len(query_tags)

        for tag in query_tags:
            # Also match parent tags (any tag that is a prefix)
            for indexed_tag, doc_ids in self.tag_docs.items():
                if indexed_tag == tag or indexed_tag.startswith(tag) or tag.startswith(indexed_tag):
                    for doc_id in doc_ids:
                        doc_scores[doc_id] += 1.0 / n_tags

        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ─── Bottom-Up Enrichment ─────────────────────────────────────────────

    def bottom_up_enrich(
        self,
        seed_doc_ids: List[str],
        extra_k: int = 20,
    ) -> List[str]:
        """
        Bottom-up: From retrieved docs, expand via sibling tags to find
        related documents that might have been missed.

        Returns additional doc_ids not already in seed.
        """
        seed_set = set(seed_doc_ids)
        # Collect all tags from seed documents
        seed_tags: Set[str] = set()
        for doc_id in seed_doc_ids:
            seed_tags.update(self.doc_tags.get(doc_id, set()))

        # Find sibling tags (same parent)
        sibling_tags: Set[str] = set()
        for tag in seed_tags:
            parent = "/".join(tag.split("/")[:-1])
            if parent:
                for other_tag in self._flat_tags:
                    if other_tag.startswith(parent) and other_tag != tag:
                        sibling_tags.add(other_tag)

        # Get documents from sibling tags
        expansion_scores: Dict[str, float] = Counter()
        for tag in sibling_tags:
            for doc_id in self.tag_docs.get(tag, set()):
                if doc_id not in seed_set:
                    expansion_scores[doc_id] += 1

        ranked = sorted(expansion_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:extra_k]]

    # ─── Full U-Retrieval Pipeline ────────────────────────────────────────

    def u_retrieve(
        self,
        query: str,
        top_k: int = 30,
        enrich_k: int = 10,
        use_llm_tags: bool = False,
    ) -> Dict[str, Any]:
        """
        Full U-retrieval: assign tags → top-down → bottom-up.

        Returns
        -------
        dict with:
            query_tags: list of assigned tags
            top_down_docs: list of (doc_id, score)
            enriched_docs: list of doc_id
            all_doc_ids: combined list (top-down + enriched)
        """
        # Step 1: Assign query tags
        if use_llm_tags:
            query_tags = self.assign_query_tags_llm(query)
        else:
            query_tags = self.assign_query_tags(query)

        # Step 2: Top-down
        top_down = self.top_down_retrieve(query_tags, top_k=top_k)
        top_down_ids = [doc_id for doc_id, _ in top_down]

        # Step 3: Bottom-up enrichment
        enriched = self.bottom_up_enrich(top_down_ids, extra_k=enrich_k)

        # Combined
        all_ids = top_down_ids + enriched

        return {
            "query_tags": query_tags,
            "top_down_docs": top_down,
            "enriched_docs": enriched,
            "all_doc_ids": all_ids,
        }

    # ─── Tag Statistics ───────────────────────────────────────────────────

    def tag_stats(self) -> Dict[str, int]:
        """Return tag distribution statistics."""
        stats = {"total_docs_tagged": len(self.doc_tags), "total_tags_used": len(self.tag_docs)}
        stats["tag_distribution"] = {
            tag: len(docs) for tag, docs in sorted(
                self.tag_docs.items(), key=lambda x: len(x[1]), reverse=True
            )[:20]
        }
        return stats

    # ─── Persistence ──────────────────────────────────────────────────────

    def save_index(self, path: Optional[str] = None):
        path = path or self.tag_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "doc_tags": {k: list(v) for k, v in self.doc_tags.items()},
            "tag_docs": {k: list(v) for k, v in self.tag_docs.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[URetrieval] Saved tag index ({len(self.doc_tags)} docs) → {path}")

    def _load_index(self):
        if not os.path.exists(self.tag_index_path):
            return
        try:
            with open(self.tag_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.doc_tags = {k: set(v) for k, v in data.get("doc_tags", {}).items()}
            self.tag_docs = {k: set(v) for k, v in data.get("tag_docs", {}).items()}
            print(f"[URetrieval] Loaded tag index: {len(self.doc_tags)} docs, {len(self.tag_docs)} tags")
        except Exception as e:
            print(f"[URetrieval] Load error: {e}")

    # ─── Format for LLM Context ──────────────────────────────────────────

    def format_retrieval_context(self, u_result: Dict[str, Any]) -> str:
        """Format U-Retrieval result as text for LLM context injection."""
        lines = ["**U-Retrieval Context:**"]
        lines.append(f"Query tags: {', '.join(u_result.get('query_tags', []))}")
        lines.append(f"Top-down documents: {len(u_result.get('top_down_docs', []))}")
        lines.append(f"Bottom-up enrichment: {len(u_result.get('enriched_docs', []))}")
        return "\n".join(lines)
