"""
Sequence-Aware Context Module
==============================
Accept DNA sequence input → identify gene → resolve variants →
trigger VariantResolver → fetch gene-specific literature.

Pipeline:
1. Parse DNA/RNA sequence from user input
2. Identify gene via known sequence motifs or gene name co-occurrence
3. Detect variants (if reference sequence known)
4. Invoke VariantResolver for editing strategy recommendations
5. Return gene-specific context for RAG augmentation
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Known Gene Sequences — Curated hotspot regions for gene editing
# (Short sequences around the editing target site)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_SEQUENCES: Dict[str, Dict[str, Any]] = {
    # HBB — Sickle cell mutation site (exon 1, codon 6 region)
    "HBB_E6V": {
        "gene": "HBB",
        "description": "HBB exon 1 codon 6 region (SCD E6V mutation site)",
        "chromosome": "chr11:5227002",
        "ref_sequence": "ACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG",
        #                      ^^^^^^^^ codon 6 (GAG→GTG for E6V)
        "mut_sequence": "ACCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG",
        "variant": "HBB:c.20A>T",
        "mutation_position": 14,  # 0-indexed position of A→T in the snippet
        "annotations": {
            "codon_6": {"start": 12, "end": 15, "ref": "GAG", "mut": "GTG"},
        },
    },
    # PCSK9 — Exon 1 splice donor region
    "PCSK9_splice": {
        "gene": "PCSK9",
        "description": "PCSK9 exon 1 splice donor region (VERVE-101 target)",
        "chromosome": "chr1:55505221",
        "ref_sequence": "ATGGGCACCGTCAGCTCCAGGCGGTCCTGGTACTCGTCCCTGCTGATCTTC",
        "variant": "PCSK9:splice_disruption",
        "mutation_position": None,
        "annotations": {
            "splice_donor": {"start": 0, "end": 20, "note": "ABE target region"},
        },
    },
    # TTR — Exon 2 target region (NTLA-2001)
    "TTR_exon2": {
        "gene": "TTR",
        "description": "TTR exon 2 region (NTLA-2001 Cas9 target site)",
        "chromosome": "chr18:31592900",
        "ref_sequence": "TCAAATTTCAGAAGGCTCAGCAGATGGTCTTCTTGGCTGTGAATAATCCCAAAATGTGG",
        "variant": "TTR:KO",
        "mutation_position": None,
        "annotations": {
            "cas9_target": {"start": 10, "end": 30, "note": "SpCas9 sgRNA region"},
        },
    },
    # CEP290 — IVS26 intronic region
    "CEP290_IVS26": {
        "gene": "CEP290",
        "description": "CEP290 intron 26 (IVS26) — LCA10 mutation creates cryptic splice site",
        "chromosome": "chr12:88071099",
        "ref_sequence": "CTTTTATCCAAATAATCTGTGATCCTCAGTGTATTCACACCCTTAAAATGAT",
        "mut_sequence": "CTTTTATCCAAATAATCTGTGATCCTCAGTGTATACACACCCTTAAAATGAT",
        "variant": "CEP290:IVS26",
        "mutation_position": 30,
        "annotations": {
            "cryptic_splice": {"start": 25, "end": 40, "note": "Aberrant splice site"},
        },
    },
    # DMD — Exon 51 boundaries (exon skipping target)
    "DMD_exon51": {
        "gene": "DMD",
        "description": "DMD exon 51 — target for exon skipping therapy",
        "chromosome": "chrX:31496382",
        "ref_sequence": "GTCTACAACAAAGCTCAGGTCGAAATTGACACTTTGTCTGAAGAAAGAATCAATGATGA",
        "variant": "DMD:exon51_skip",
        "mutation_position": None,
        "annotations": {
            "exon_51": {"start": 0, "end": 58, "note": "Full exon 51 fragment"},
        },
    },
    # CFTR — F508del region
    "CFTR_F508del": {
        "gene": "CFTR",
        "description": "CFTR exon 11 — F508del mutation site (most common CF mutation)",
        "chromosome": "chr7:117559590",
        "ref_sequence": "AATATCATCTTTGGTGTTTCCTATGATGAATATAGATACAGAAGCGTCATC",
        #                    ^^^^^^ codon 508 (CTT = Phe; deleted in F508del)
        "mut_sequence": "AATATCATCGGTGTTTCCTATGATGAATATAGATACAGAAGCGTCATC",
        "variant": "CFTR:F508del",
        "mutation_position": 9,
        "annotations": {
            "codon_508": {"start": 9, "end": 12, "ref": "TTT", "note": "Deleted in F508del"},
        },
    },
    # CCR5 — HIV resistance target
    "CCR5_delta32": {
        "gene": "CCR5",
        "description": "CCR5 coding region — delta32 deletion confirs HIV resistance",
        "chromosome": "chr3:46414947",
        "ref_sequence": "AGTCAGTATCAATTCTGGAAGAATTTCCAGACATTAAAGATAGTCATCTTGGGGCTGGTC",
        "variant": "CCR5:delta32",
        "mutation_position": None,
        "annotations": {
            "delta32": {"start": 10, "end": 42, "note": "32bp deletion region"},
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Gene name → keyword mapping for identification
# ─────────────────────────────────────────────────────────────────────────────
GENE_KEYWORDS: Dict[str, List[str]] = {
    "HBB": ["hemoglobin", "beta-globin", "sickle", "thalassemia", "血红蛋白", "镰刀", "地贫"],
    "PCSK9": ["pcsk9", "cholesterol", "ldl", "hypercholesterolemia", "胆固醇", "高脂"],
    "TTR": ["ttr", "transthyretin", "amyloid", "amyloidosis", "淀粉样", "转甲状腺素"],
    "DMD": ["dmd", "dystrophin", "duchenne", "muscular dystrophy", "肌营养不良", "杜氏"],
    "CEP290": ["cep290", "lca", "leber", "retina", "视网膜"],
    "CFTR": ["cftr", "cystic fibrosis", "cf", "chloride", "囊性纤维化"],
    "CCR5": ["ccr5", "hiv", "aids", "delta32"],
    "BCL11A": ["bcl11a", "fetal hemoglobin", "hbf", "胎儿血红蛋白"],
    "SERPINA1": ["serpina1", "alpha-1", "antitrypsin", "aatd", "z allele"],
    "HTT": ["htt", "huntingtin", "huntington", "亨廷顿", "cag repeat"],
    "F8": ["f8", "factor viii", "hemophilia a", "血友病a"],
    "F9": ["f9", "factor ix", "hemophilia b", "血友病b"],
}


class SequenceContext:
    """Parse DNA sequences and provide gene-editing-aware context."""

    def __init__(self, variant_resolver=None):
        self.variant_resolver = variant_resolver
        self.known_sequences = KNOWN_SEQUENCES
        self.gene_keywords = GENE_KEYWORDS

    # ─── Sequence Detection ───────────────────────────────────────────────

    def extract_sequence(self, query: str) -> Optional[str]:
        """Extract DNA/RNA sequence from text input."""
        # Pattern: 15+ consecutive ACGTU characters (allowing spaces/newlines)
        clean = re.sub(r'[\s\-]', '', query)
        dna_match = re.search(r'[ACGTUacgtu]{15,}', clean)
        if dna_match:
            return dna_match.group(0).upper().replace('U', 'T')
        return None

    def identify_gene(self, query: str, sequence: str = None) -> Optional[str]:
        """
        Identify gene from query text and/or sequence matching.

        Priority: (1) gene name in query, (2) sequence match, (3) keyword match
        """
        q_lower = query.lower()

        # 1. Direct gene name match
        gene_pattern = r'\b(HBB|PCSK9|TTR|DMD|CEP290|CFTR|CCR5|BCL11A|SERPINA1|HTT|F8|F9|SMN1|SMN2|TRAC|B2M)\b'
        gene_match = re.search(gene_pattern, query, re.I)
        if gene_match:
            return gene_match.group(1).upper()

        # 2. Sequence matching
        if sequence:
            best_match = None
            best_score = 0
            for seq_id, info in self.known_sequences.items():
                ref = info.get("ref_sequence", "")
                if not ref:
                    continue

                # Simple subsequence matching (longest common substring approach)
                score = self._sequence_similarity(sequence, ref)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = info["gene"]

            if best_match:
                return best_match

        # 3. Keyword matching
        for gene, keywords in self.gene_keywords.items():
            if any(kw in q_lower for kw in keywords):
                return gene

        return None

    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Simple sliding-window similarity score."""
        if len(seq1) < 15 or len(seq2) < 15:
            return 0.0

        shorter, longer = (seq1, seq2) if len(seq1) <= len(seq2) else (seq2, seq1)
        window = min(len(shorter), 30)
        best_match = 0

        for i in range(len(longer) - window + 1):
            segment = longer[i:i + window]
            matches = sum(1 for a, b in zip(shorter[:window], segment) if a == b)
            best_match = max(best_match, matches / window)

        return best_match

    # ─── Variant Detection ────────────────────────────────────────────────

    def detect_variant(self, sequence: str, gene: str) -> Optional[Dict]:
        """
        Compare input sequence against known reference to detect variant.
        """
        for seq_id, info in self.known_sequences.items():
            if info["gene"] != gene:
                continue

            ref = info.get("ref_sequence", "")
            mut = info.get("mut_sequence", ref)

            if not ref:
                continue

            # Check which sequence the input matches
            ref_sim = self._sequence_similarity(sequence, ref)
            mut_sim = self._sequence_similarity(sequence, mut)

            if mut_sim > ref_sim and mut_sim > 0.7:
                return {
                    "gene": gene,
                    "variant": info.get("variant"),
                    "match_type": "mutant",
                    "similarity": mut_sim,
                    "description": info.get("description"),
                    "annotations": info.get("annotations", {}),
                }
            elif ref_sim > 0.7:
                return {
                    "gene": gene,
                    "variant": info.get("variant"),
                    "match_type": "reference",
                    "similarity": ref_sim,
                    "description": info.get("description"),
                    "annotations": info.get("annotations", {}),
                }

        return None

    # ─── Full Pipeline ────────────────────────────────────────────────────

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Full sequence-aware analysis pipeline.

        Returns:
            Dict with gene, variant info, and optional editing strategy
        """
        result = {
            "has_sequence": False,
            "sequence": None,
            "gene": None,
            "variant_info": None,
            "editing_strategy": None,
            "gene_context": None,
        }

        # 1. Extract sequence
        sequence = self.extract_sequence(query)
        if sequence:
            result["has_sequence"] = True
            result["sequence"] = sequence[:100]  # Truncate for display
            result["sequence_length"] = len(sequence)

        # 2. Identify gene
        gene = self.identify_gene(query, sequence)
        if gene:
            result["gene"] = gene

            # 3. Gene context (known editing targets)
            for seq_id, info in self.known_sequences.items():
                if info["gene"] == gene:
                    result["gene_context"] = {
                        "description": info.get("description"),
                        "chromosome": info.get("chromosome"),
                        "variant": info.get("variant"),
                        "annotations": info.get("annotations"),
                    }
                    break

        # 4. Detect variant from sequence
        if sequence and gene:
            variant = self.detect_variant(sequence, gene)
            if variant:
                result["variant_info"] = variant

        # 5. Invoke VariantResolver if available
        if self.variant_resolver and gene:
            variant_query = query if not sequence else f"{gene} editing strategy"
            resolution = self.variant_resolver.resolve(variant_query)
            if resolution.get("curated_entry") or resolution.get("rule_strategies"):
                result["editing_strategy"] = resolution

        return result

    def format_context(self, analysis: Dict) -> str:
        """Format sequence analysis as structured text for LLM context injection."""
        if not analysis.get("gene") and not analysis.get("has_sequence"):
            return ""

        lines = ["### 🧬 Sequence-Aware Context"]

        if analysis.get("has_sequence"):
            lines.append(f"**Detected DNA sequence:** {analysis['sequence'][:50]}... "
                         f"({analysis.get('sequence_length', '?')} bp)")

        if analysis.get("gene"):
            lines.append(f"**Identified gene:** {analysis['gene']}")

        gc = analysis.get("gene_context")
        if gc:
            lines.append(f"**Description:** {gc.get('description', 'N/A')}")
            if gc.get("chromosome"):
                lines.append(f"**Genomic location:** {gc['chromosome']}")
            if gc.get("variant"):
                lines.append(f"**Key variant:** {gc['variant']}")

        vi = analysis.get("variant_info")
        if vi:
            lines.append(f"\n**Variant detection:** {vi.get('match_type', 'N/A')} sequence "
                         f"(similarity: {vi.get('similarity', 0):.0%})")
            if vi.get("variant"):
                lines.append(f"**Matched variant:** {vi['variant']}")

        # If editing strategy was resolved, VariantResolver.format_context handles that
        # Just note it here
        if analysis.get("editing_strategy"):
            lines.append(f"\n*(Editing strategy resolved — see Variant Resolver output below)*")

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def query_has_sequence(query: str) -> bool:
        """Heuristic: does this query contain a DNA/RNA sequence?"""
        clean = re.sub(r'[\s\-]', '', query)
        # 15+ consecutive nucleotides
        return bool(re.search(r'[ACGTUacgtu]{15,}', clean))

    @staticmethod
    def query_needs_gene_context(query: str) -> bool:
        """Heuristic: does this query need gene-specific editing context?"""
        q = query.lower()
        triggers = [
            r'[ACGTUacgtu]{15,}',  # DNA sequence
            r'序列|sequence|base pair|碱基',
            r'exon|外显子|intron|内含子',
            r'位点|locus|位置|position',
            r'靶点|target site|grna|sgrna',
            r'pam|protospacer',
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
