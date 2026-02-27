"""
End-to-End Experiment Runner for GEBench
=========================================
Orchestrates full evaluation pipeline:

  1. Load GEBench benchmark
  2. Run EditomeCopilot (full system)
  3. Run 4 baselines
  4. Run ablation variants
  5. Aggregate all metrics
  6. Generate publication-ready tables & figures

Output:  results/<timestamp>/
  ├── full_system.json
  ├── baselines/
  │   ├── LLM-Only.json
  │   ├── Naive-RAG.json
  │   ├── PubMed-RAG.json
  │   └── LangChain-RAG.json
  ├── ablation/
  │   ├── A0_full.json ... A15_xxx.json
  ├── comparison_table.md
  ├── ablation_table.md
  └── summary_report.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from evaluation.metrics import GEBenchEvaluator
from evaluation.baselines import (
    BaselineSystem,
    create_all_baselines,
)
from evaluation.ablation import AblationRunner, ABLATION_CONFIGS


class ExperimentRunner:
    """
    End-to-end experiment orchestrator for publication-grade evaluation.

    Usage
    -----
    runner = ExperimentRunner()
    runner.run_full_evaluation(
        max_questions=50,    # 0 for all
        run_baselines=True,
        run_ablation=True,
        ablation_configs=["A0_full", "A1_no_almanac", "AG1_no_domain_modules"],
    )
    """

    def __init__(
        self,
        output_dir: str = "results",
        gebench_path: str = None,
    ):
        self.output_dir = output_dir
        self.gebench_path = gebench_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "eval", "gebench.jsonl"
        )
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(output_dir, self._timestamp)

        # Lazy-loaded components
        self._rag = None
        self._llm = None
        self._evaluator = None

    @property
    def llm(self):
        if self._llm is None:
            from core.llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    @property
    def rag(self):
        if self._rag is None:
            from core.agentic_rag import AgenticRAG
            print("[Runner] Initialising AgenticRAG (this may take a moment)...")
            self._rag = AgenticRAG()
        return self._rag

    @property
    def evaluator(self):
        if self._evaluator is None:
            safety_fw = getattr(self.rag, "safety_firewall", None)
            self._evaluator = GEBenchEvaluator(
                llm_client=self.llm,
                safety_firewall=safety_fw,
            )
        return self._evaluator

    def _ensure_dir(self, *subdirs: str) -> str:
        path = os.path.join(self._run_dir, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path

    def _system_fn(self, question: str) -> Dict[str, Any]:
        """Wrap AgenticRAG for evaluation."""
        try:
            result = self.rag.process_query(question)
            if isinstance(result, dict):
                sections = result.get("sections", [])
                answer_parts = []
                contexts = []
                for sec in sections:
                    if isinstance(sec, dict):
                        content = sec.get("content", "")
                        if content:
                            answer_parts.append(content)
                        if sec.get("type") == "literature":
                            contexts.append(content)
                return {
                    "answer": "\n\n".join(answer_parts) if answer_parts else str(result),
                    "contexts": contexts,
                }
            return {"answer": str(result), "contexts": []}
        except Exception as e:
            return {"answer": f"[Error] {e}", "contexts": []}

    # ------------------------------------------------------------------
    # Phase 1: Full System Evaluation
    # ------------------------------------------------------------------

    def run_full_system(
        self, max_questions: int = 0
    ) -> Dict[str, Any]:
        """Run EditomeCopilot full system on GEBench."""
        print("\n" + "=" * 70)
        print("  Phase 1: Full System Evaluation")
        print("=" * 70)

        save_path = os.path.join(self._ensure_dir(), "full_system.json")
        report = self.evaluator.evaluate_benchmark(
            system_fn=self._system_fn,
            gebench_path=self.gebench_path,
            max_questions=max_questions,
            save_path=save_path,
        )
        report["system_name"] = "EditomeCopilot"
        return report

    # ------------------------------------------------------------------
    # Phase 2: Baseline Comparison
    # ------------------------------------------------------------------

    def run_baselines(
        self,
        max_questions: int = 0,
        baseline_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Run all baseline systems on GEBench."""
        print("\n" + "=" * 70)
        print("  Phase 2: Baseline Comparison")
        print("=" * 70)

        baselines = create_all_baselines(
            llm_client=self.llm,
            data_pipeline=getattr(self.rag, "data_pipeline", None),
        )

        if baseline_names:
            baselines = {k: v for k, v in baselines.items() if k in baseline_names}

        results = {}
        baseline_dir = self._ensure_dir("baselines")

        for name, baseline in baselines.items():
            print(f"\n--- Running baseline: {name} ---")
            save_path = os.path.join(baseline_dir, f"{name.replace(' ', '_')}.json")

            report = self.evaluator.evaluate_benchmark(
                system_fn=baseline.answer,
                gebench_path=self.gebench_path,
                max_questions=max_questions,
                save_path=save_path,
            )
            report["system_name"] = name
            results[name] = report

        return results

    # ------------------------------------------------------------------
    # Phase 3: Ablation Study
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        max_questions: int = 0,
        configs: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Run ablation experiments."""
        print("\n" + "=" * 70)
        print("  Phase 3: Ablation Study")
        print("=" * 70)

        ablation_runner = AblationRunner(
            rag_system=self.rag,
            evaluator=self.evaluator,
            gebench_path=self.gebench_path,
        )

        save_path = os.path.join(self._ensure_dir("ablation"), "all_ablation.json")
        results = ablation_runner.run_all(
            configs=configs,
            max_questions=max_questions,
            save_path=save_path,
        )

        return results

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_comparison_table(
        self,
        full_report: Dict,
        baseline_reports: Dict[str, Dict],
    ) -> str:
        """Generate main comparison table (Table 2 in paper)."""
        lines = [
            "## System Comparison on GEBench\n",
            "| System | Composite | Accuracy | Faithfulness | Citation F1 | Completeness | Safety | Latency (s) |",
            "|--------|-----------|----------|-------------|-------------|-------------|--------|-------------|",
        ]

        # Full system first
        o = full_report.get("overall", {})
        lines.append(
            f"| **EditomeCopilot** | **{o.get('composite_score', 0):.3f}** | "
            f"**{o.get('factual_accuracy', 0):.3f}** | **{o.get('faithfulness', 0):.3f}** | "
            f"**{o.get('citation_f1', 0):.3f}** | **{o.get('completeness', 0):.3f}** | "
            f"**{o.get('safety_compliance', 0):.3f}** | {o.get('latency_seconds', 0):.1f} |"
        )

        # Baselines
        for name, report in sorted(baseline_reports.items()):
            o = report.get("overall", {})
            lines.append(
                f"| {name} | {o.get('composite_score', 0):.3f} | "
                f"{o.get('factual_accuracy', 0):.3f} | {o.get('faithfulness', 0):.3f} | "
                f"{o.get('citation_f1', 0):.3f} | {o.get('completeness', 0):.3f} | "
                f"{o.get('safety_compliance', 0):.3f} | {o.get('latency_seconds', 0):.1f} |"
            )

        return "\n".join(lines)

    def generate_summary_report(
        self,
        full_report: Dict,
        baseline_reports: Dict[str, Dict],
        ablation_results: Dict[str, Dict] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        summary = {
            "timestamp": self._timestamp,
            "benchmark": {
                "path": self.gebench_path,
                "total_questions": full_report.get("n", 0),
            },
            "full_system": full_report.get("overall", {}),
            "baselines": {
                name: report.get("overall", {})
                for name, report in baseline_reports.items()
            },
        }

        # Compute improvements over baselines
        full_composite = full_report.get("overall", {}).get("composite_score", 0)
        improvements = {}
        for name, report in baseline_reports.items():
            baseline_composite = report.get("overall", {}).get("composite_score", 0)
            if baseline_composite > 0:
                pct = (full_composite - baseline_composite) / baseline_composite * 100
                improvements[name] = round(pct, 1)
        summary["improvements_pct"] = improvements

        if ablation_results:
            deltas = AblationRunner.compute_deltas(ablation_results)
            summary["ablation_deltas"] = deltas
            critical = AblationRunner.find_critical_modules(deltas)
            summary["critical_modules"] = [
                {"config": c[0], "delta": c[1], "desc": c[2]} for c in critical
            ]

        return summary

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def run_full_evaluation(
        self,
        max_questions: int = 0,
        run_baselines: bool = True,
        run_ablation: bool = True,
        ablation_configs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Parameters
        ----------
        max_questions : int, limit questions per experiment (0=all)
        run_baselines : whether to run baseline comparisons
        run_ablation : whether to run ablation study
        ablation_configs : specific ablation configs (default: all A0-A15)

        Returns full results dict
        """
        print(f"\n{'#' * 70}")
        print(f"  GEBench Full Evaluation — {self._timestamp}")
        print(f"  Output: {self._run_dir}")
        print(f"{'#' * 70}\n")

        start_time = time.time()

        # Phase 1: Full system
        full_report = self.run_full_system(max_questions)

        # Phase 2: Baselines
        baseline_reports = {}
        if run_baselines:
            baseline_reports = self.run_baselines(max_questions)

        # Phase 3: Ablation
        ablation_results = {}
        if run_ablation:
            ablation_results = self.run_ablation(max_questions, ablation_configs)

        # Generate tables
        comp_table = self.generate_comparison_table(full_report, baseline_reports)
        comp_path = os.path.join(self._ensure_dir(), "comparison_table.md")
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(comp_table)

        if ablation_results:
            deltas = AblationRunner.compute_deltas(ablation_results)
            abl_table = AblationRunner.format_delta_table(deltas)
            abl_path = os.path.join(self._ensure_dir(), "ablation_table.md")
            with open(abl_path, "w", encoding="utf-8") as f:
                f.write(abl_table)

        # Summary
        summary = self.generate_summary_report(full_report, baseline_reports, ablation_results)
        summary["total_time_seconds"] = round(time.time() - start_time, 1)

        summary_path = os.path.join(self._ensure_dir(), "summary_report.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Print summary
        print(f"\n{'#' * 70}")
        print("  EVALUATION COMPLETE")
        print(f"{'#' * 70}")
        print(f"\n  Full system composite:  {full_report.get('overall', {}).get('composite_score', 0):.4f}")
        for name, report in baseline_reports.items():
            print(f"  {name:20s}:  {report.get('overall', {}).get('composite_score', 0):.4f}")
        if summary.get("improvements_pct"):
            print(f"\n  Improvements over baselines:")
            for name, pct in summary["improvements_pct"].items():
                print(f"    vs {name}: +{pct:.1f}%")

        print(f"\n  Total time: {summary['total_time_seconds']:.0f}s")
        print(f"  Results saved to: {self._run_dir}")

        return {
            "full_system": full_report,
            "baselines": baseline_reports,
            "ablation": ablation_results,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Quick-run function
# ---------------------------------------------------------------------------

def quick_eval(max_questions: int = 10, run_ablation: bool = False):
    """Quick evaluation with limited questions for development & debugging."""
    runner = ExperimentRunner()
    return runner.run_full_evaluation(
        max_questions=max_questions,
        run_baselines=True,
        run_ablation=run_ablation,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEBench Experiment Runner")
    parser.add_argument("-n", "--max-questions", type=int, default=0,
                        help="Max questions per experiment (0=all)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline comparison")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip ablation study")
    parser.add_argument("--ablation-configs", nargs="+", default=None,
                        help="Specific ablation configs to run")
    parser.add_argument("-o", "--output", default="results",
                        help="Output directory")
    parser.add_argument("--gebench", default=None,
                        help="Path to gebench.jsonl")

    args = parser.parse_args()

    runner = ExperimentRunner(
        output_dir=args.output,
        gebench_path=args.gebench,
    )

    results = runner.run_full_evaluation(
        max_questions=args.max_questions,
        run_baselines=not args.no_baselines,
        run_ablation=not args.no_ablation,
        ablation_configs=args.ablation_configs,
    )
