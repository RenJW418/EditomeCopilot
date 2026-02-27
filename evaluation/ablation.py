"""
Ablation Experiment Framework
==============================
Systematically disable each module in EditomeCopilot and measure impact
on GEBench scores. Produces a delta table suitable for paper Table 3.

Module Groups for Ablation
--------------------------
  A0  Full system (all modules enabled)        — control
  A1  -GEAlmanac                                — domain knowledge impact
  A2  -TripleGraph                              — knowledge graph reasoning
  A3  -SafetyFirewall                           — safety compliance impact
  A4  -EvidenceChain                            — citation grounding impact
  A5  -VariantResolver                          — variant resolution impact
  A6  -FailureCaseDB                            — failure knowledge impact
  A7  -CrossSpeciesTranslator                   — translational knowledge
  A8  -DeliveryDecisionTree                     — delivery recommendation
  A9  -URetrieval (hierarchical tag retrieval)  — retrieval strategy
  A10 -HyDE                                     — query expansion
  A11 -Reranker                                 — cross-encoder reranking
  A12 -RAPTOR                                   — summary index
  A13 -SemanticChunker                          — chunking strategy
  A14 -PatentLandscape                          — patent info
  A15 -SequenceContext                          — sequence analysis
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable


# ---------------------------------------------------------------------------
# Ablation Configuration
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "A0_full": {
        "description": "Full system (all modules)",
        "disabled_modules": [],
    },
    "A1_no_almanac": {
        "description": "Without GEAlmanac domain knowledge",
        "disabled_modules": ["almanac"],
    },
    "A2_no_triple_graph": {
        "description": "Without TripleGraph knowledge graph",
        "disabled_modules": ["triple_graph"],
    },
    "A3_no_safety": {
        "description": "Without SafetyFirewall post-check",
        "disabled_modules": ["safety_firewall"],
    },
    "A4_no_evidence_chain": {
        "description": "Without EvidenceChain citation grounding",
        "disabled_modules": ["evidence_chain"],
    },
    "A5_no_variant_resolver": {
        "description": "Without VariantResolver",
        "disabled_modules": ["variant_resolver"],
    },
    "A6_no_failure_db": {
        "description": "Without FailureCaseDB",
        "disabled_modules": ["failure_case_db"],
    },
    "A7_no_cross_species": {
        "description": "Without CrossSpeciesTranslator",
        "disabled_modules": ["cross_species_translator"],
    },
    "A8_no_delivery_tree": {
        "description": "Without DeliveryDecisionTree",
        "disabled_modules": ["delivery_decision_tree"],
    },
    "A9_no_u_retrieval": {
        "description": "Without URetrieval hierarchical tags",
        "disabled_modules": ["u_retrieval"],
    },
    "A10_no_hyde": {
        "description": "Without HyDE query expansion",
        "disabled_modules": ["hyde"],
    },
    "A11_no_reranker": {
        "description": "Without cross-encoder reranking",
        "disabled_modules": ["reranker"],
    },
    "A12_no_raptor": {
        "description": "Without RAPTOR summary tree",
        "disabled_modules": ["raptor"],
    },
    "A13_no_semantic_chunker": {
        "description": "Without SemanticChunker",
        "disabled_modules": ["semantic_chunker"],
    },
    "A14_no_patent": {
        "description": "Without PatentLandscape",
        "disabled_modules": ["patent_landscape"],
    },
    "A15_no_sequence": {
        "description": "Without SequenceContext",
        "disabled_modules": ["sequence_context"],
    },
    # Grouped ablations
    "AG1_no_domain_modules": {
        "description": "Without ALL domain-specific modules (v2+v3)",
        "disabled_modules": [
            "almanac", "triple_graph", "evidence_chain", "safety_firewall",
            "u_retrieval", "variant_resolver", "failure_case_db",
            "cross_species_translator", "delivery_decision_tree",
            "patent_landscape", "sequence_context",
        ],
    },
    "AG2_no_retrieval_enhancements": {
        "description": "Without retrieval enhancements (HyDE+Reranker+RAPTOR+URetrieval)",
        "disabled_modules": ["hyde", "reranker", "raptor", "u_retrieval"],
    },
    "AG3_no_safety_modules": {
        "description": "Without safety modules (Firewall+FailureCaseDB)",
        "disabled_modules": ["safety_firewall", "failure_case_db"],
    },
}


# ---------------------------------------------------------------------------
# Module-level toggling helpers
# ---------------------------------------------------------------------------

_MODULE_ATTR_MAP = {
    "almanac": "almanac",
    "triple_graph": "triple_graph",
    "safety_firewall": "safety_firewall",
    "evidence_chain": "evidence_chain",
    "u_retrieval": "u_retrieval",
    "semantic_chunker": "semantic_chunker",
    "variant_resolver": "variant_resolver",
    "failure_case_db": "failure_case_db",
    "cross_species_translator": "cross_species_translator",
    "delivery_decision_tree": "delivery_decision_tree",
    "patent_landscape": "patent_landscape",
    "sequence_context": "sequence_context",
    "hyde": "hyde",
    "reranker": "reranker",
    "raptor": "_raptor",
}


class AblationRunner:
    """
    Runs ablation experiments by temporarily disabling modules in AgenticRAG.

    Usage
    -----
    runner = AblationRunner(rag_system, evaluator)
    results = runner.run_all(gebench_path, configs=["A0_full", "A1_no_almanac", ...])
    delta_table = runner.compute_deltas(results)
    """

    def __init__(
        self,
        rag_system: Any,
        evaluator: Any,
        gebench_path: str = None,
    ):
        """
        Parameters
        ----------
        rag_system  : AgenticRAG instance
        evaluator   : GEBenchEvaluator instance
        gebench_path : path to gebench.jsonl
        """
        self.rag = rag_system
        self.evaluator = evaluator
        self.gebench_path = gebench_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "eval", "gebench.jsonl"
        )

    def _disable_modules(self, module_names: List[str]) -> Dict[str, Any]:
        """
        Temporarily disable modules by storing originals and setting to None.
        Returns backup dict for restoration.
        """
        backup = {}
        for mod_name in module_names:
            attr = _MODULE_ATTR_MAP.get(mod_name)
            if attr and hasattr(self.rag, attr):
                backup[attr] = getattr(self.rag, attr)
                setattr(self.rag, attr, None)

                # Also check for 'enabled' attribute
                if hasattr(backup[attr], "enabled"):
                    backup[f"{attr}_enabled"] = backup[attr].enabled

        return backup

    def _restore_modules(self, backup: Dict[str, Any]):
        """Restore previously disabled modules."""
        for attr, original in backup.items():
            if attr.endswith("_enabled"):
                continue
            setattr(self.rag, attr, original)

    def _system_fn(self, question: str) -> Dict[str, Any]:
        """Wrap AgenticRAG.process_query as a baseline-compatible callable."""
        try:
            result = self.rag.process_query(question)
            # Extract answer and contexts from the result structure
            if isinstance(result, dict):
                sections = result.get("sections", [])
                answer_parts = []
                contexts = []
                for section in sections:
                    if isinstance(section, dict):
                        content = section.get("content", "")
                        if content:
                            answer_parts.append(content)
                        if section.get("type") == "literature":
                            contexts.append(content)
                answer = "\n\n".join(answer_parts) if answer_parts else str(result)
                return {"answer": answer, "contexts": contexts}
            return {"answer": str(result), "contexts": []}
        except Exception as e:
            return {"answer": f"[Error] {e}", "contexts": []}

    def run_single_config(
        self,
        config_name: str,
        max_questions: int = 0,
        question_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run GEBench evaluation with a specific ablation configuration.

        Returns the evaluation report dict.
        """
        config = ABLATION_CONFIGS.get(config_name)
        if not config:
            raise ValueError(f"Unknown ablation config: {config_name}")

        print(f"\n{'='*60}")
        print(f"[Ablation] Running: {config_name} — {config['description']}")
        print(f"  Disabled: {config['disabled_modules'] or 'None (full system)'}")
        print(f"{'='*60}")

        # Disable modules
        backup = self._disable_modules(config["disabled_modules"])

        try:
            report = self.evaluator.evaluate_benchmark(
                system_fn=self._system_fn,
                gebench_path=self.gebench_path,
                max_questions=max_questions,
                question_types=question_types,
            )
            report["config_name"] = config_name
            report["config_description"] = config["description"]
            report["disabled_modules"] = config["disabled_modules"]
        finally:
            # Always restore
            self._restore_modules(backup)

        return report

    def run_all(
        self,
        configs: Optional[List[str]] = None,
        max_questions: int = 0,
        save_path: str = None,
    ) -> Dict[str, Dict]:
        """
        Run all ablation configurations.

        Parameters
        ----------
        configs : list of config names (default: all individual ablations)
        max_questions : limit per config (0 = all)
        save_path : path to save full results

        Returns dict: config_name -> report
        """
        if configs is None:
            configs = [f"A{i}" for i in range(16) if f"A{i}_" in
                       "".join(ABLATION_CONFIGS.keys())]
            # More robust: get all A0-A15
            configs = [k for k in ABLATION_CONFIGS.keys() if k.startswith("A") and not k.startswith("AG")]

        all_results = {}
        for config_name in configs:
            try:
                report = self.run_single_config(config_name, max_questions=max_questions)
                all_results[config_name] = report
            except Exception as e:
                print(f"[Ablation] {config_name} FAILED: {e}")
                all_results[config_name] = {"error": str(e)}

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n[Ablation] Full results saved to {save_path}")

        return all_results

    @staticmethod
    def compute_deltas(
        results: Dict[str, Dict],
        baseline_key: str = "A0_full",
    ) -> Dict[str, Dict]:
        """
        Compute performance deltas relative to the full system (A0).

        Returns dict: config_name -> {metric: delta, ...}
        """
        baseline = results.get(baseline_key, {}).get("overall", {})
        if not baseline:
            print(f"[Ablation] Baseline {baseline_key} not found in results.")
            return {}

        deltas = {}
        metrics = [
            "composite_score", "factual_accuracy", "faithfulness",
            "citation_f1", "completeness", "safety_compliance",
        ]

        for config_name, report in results.items():
            if config_name == baseline_key:
                continue
            if "error" in report:
                continue
            overall = report.get("overall", {})
            delta = {}
            for m in metrics:
                base_val = baseline.get(m, 0)
                curr_val = overall.get(m, 0)
                delta[m] = round(curr_val - base_val, 4)
            delta["description"] = report.get("config_description", "")
            delta["disabled"] = report.get("disabled_modules", [])
            deltas[config_name] = delta

        return deltas

    @staticmethod
    def format_delta_table(deltas: Dict[str, Dict]) -> str:
        """Format deltas as a Markdown table for paper inclusion."""
        lines = [
            "## Ablation Study Results (Δ from Full System)\n",
            "| Config | Description | Composite | Accuracy | Faithfulness | Citation | Completeness | Safety |",
            "|--------|------------|-----------|----------|-------------|----------|-------------|--------|",
        ]

        for config_name, d in sorted(deltas.items()):
            desc = d.get("description", "")[:35]
            lines.append(
                f"| {config_name} | {desc} | "
                f"{d.get('composite_score', 0):+.3f} | "
                f"{d.get('factual_accuracy', 0):+.3f} | "
                f"{d.get('faithfulness', 0):+.3f} | "
                f"{d.get('citation_f1', 0):+.3f} | "
                f"{d.get('completeness', 0):+.3f} | "
                f"{d.get('safety_compliance', 0):+.3f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def find_critical_modules(deltas: Dict[str, Dict], threshold: float = -0.05) -> List[str]:
        """Identify modules whose removal causes > threshold drop in composite score."""
        critical = []
        for config_name, d in deltas.items():
            if d.get("composite_score", 0) < threshold:
                critical.append((config_name, d["composite_score"], d.get("description", "")))

        critical.sort(key=lambda x: x[1])  # Most negative first
        return critical
